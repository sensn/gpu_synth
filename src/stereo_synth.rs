#![allow(warnings)]
use cubecl::prelude::*;

#[cube]
fn apply_moog_ladder<F: Float + CubeElement>(bin_freq: F, cutoff: F, resonance: F) -> F {
    let f = bin_freq / F::max(F::new(1.0), cutoff);
    let pole_step = F::new(1.0) / (F::new(1.0) + f * f);
    let mut magnitude = pole_step * pole_step * pole_step * pole_step;

    let distance = F::abs(bin_freq - cutoff);
    let bandwidth = cutoff * F::new(0.08);
    if distance < bandwidth {
        magnitude += resonance * (F::new(1.0) - (distance / bandwidth));
    }
    magnitude
}

#[cube]
fn apply_oberheim_sem<F: Float + CubeElement>(bin_freq: F, cutoff: F, resonance: F, mode_select: u32) -> F {
    let f = bin_freq / F::max(F::new(1.0), cutoff);
    let f2 = f * f;
    let damping = F::new(1.0) / F::max(F::new(0.1), F::new(1.0) - resonance);
    let denominator = (F::new(1.0) - f2) * (F::new(1.0) - f2) + (f2 / (damping * damping));
    let sqrt_denom = F::sqrt(denominator);

    let mut response = F::new(0.0);
    if mode_select == 0 { response = F::new(1.0) / sqrt_denom; }
    else if mode_select == 1 { response = f2 / sqrt_denom; }
    else if mode_select == 2 { response = f / sqrt_denom; }
    else { response = F::abs(F::new(1.0) - f2) / sqrt_denom; }
    response
}

// --- FUSIONIERTES FRAKTALES JACOBI-ANGER SÄGEZAHN-FM KERNEL ---
#[cube(launch)]
pub fn cubek_true_stereo_synth_reverb<F: Float + CubeElement>(
    output_stereo_audio: &mut Array<F>,
    frequency: F, old_frequency: F,
    dyn_cutoff: F, old_cutoff: F,
    sample_rate: F,
    lfo_accumulated_phase: F,
    op_ratios: &Array<F>,
    op_levels: &Array<F>,
    algo_select: u32,
    moog_resonance: F,
    oberheim_resonance: F,
    oberheim_mode: u32,
    lfo_frequency: F, 
    lfo_depth: F,
    room_size_seconds: F,
    high_freq_damping: F,
    wet_dry_mix: F,
    stereo_width: F,
    attack_time: F,
    decay_time: F,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let mut final_sample_l = F::new(0.0);
        let mut final_sample_r = F::new(0.0);
        let pi = F::new(std::f32::consts::PI);
        let samples_per_block = F::cast_from(fft_size);
        
        // Master ADSR
        let block_progress = F::cast_from(n) / samples_per_block;
        let sustain_lvl = F::new(0.6);
        let mut master_amp = F::new(1.0);

        if attack_time > F::new(0.05) {
            master_amp = block_progress / attack_time;
            if master_amp > F::new(1.0) { master_amp = F::new(1.0); }
        } 
        if decay_time > F::new(0.05) {
            let decay_factor = block_progress * decay_time * (F::new(1.0) - sustain_lvl);
            let mut current_decay_amp = F::new(1.0) - decay_factor;
            if current_decay_amp < sustain_lvl { current_decay_amp = sustain_lvl; }
            if current_decay_amp < master_amp { master_amp = current_decay_amp; }
        }

        // LFO
        let sample_phase_delta = (F::new(2.0) * pi * lfo_frequency * block_progress * samples_per_block) / sample_rate;
        let lfo_mod = F::sin(lfo_accumulated_phase + sample_phase_delta);

        let base_freq = old_frequency + (block_progress * (frequency - old_frequency));
        let base_cutoff = old_cutoff + (block_progress * (dyn_cutoff - old_cutoff));
        
        let mut modulated_cutoff = base_cutoff + (lfo_mod * lfo_depth);
        if modulated_cutoff < F::new(50.0) { modulated_cutoff = F::new(50.0); }

        let num_bins = fft_size / 2 + 1;

        // Register-Extraktion
        let r1 = op_ratios[0]; let l1 = op_levels[0];
        let r2 = op_ratios[1]; let l2 = op_levels[1];
        let r3 = op_ratios[2]; let l3 = op_levels[2];
        let r4 = op_ratios[3]; let l4 = op_levels[3];
        let r5 = op_ratios[4]; let l5 = op_levels[4];
        let r6 = op_ratios[5]; let l6 = op_levels[5];

        for k in 0..num_bins {
            let k_f = F::cast_from(k);
            let bin_freq = (k_f * sample_rate) / samples_per_block;

            let mut real_spec = F::new(0.0);
            let mut imag_spec = F::new(0.0);

            if k > 0 && bin_freq < sample_rate / F::new(2.0) {
                
                // Iteration über die harmonischen Oberschwingungen des Sägezahns
                for h in 1..24 {
                    let h_f = F::cast_from(h);
                    let saw_harmonic_amp = F::new(1.0) / h_f;

                    // Iteration über die Jacobi-Anger Seitenbänder
                    for sideband in 1..16 {
                        let s_f = F::cast_from(sideband);

                        let mut carrier_freq = base_freq * h_f;
                        let mut mod_freq = base_freq;
                        let mut modulation_force = F::new(0.0);
                        let mut carrier_weight = F::new(0.0);

                        // Operator 6 Selbst-Feedback Loop injizieren
                        let op6_fb = l6 * l6 * F::max(F::new(0.1), r6) * F::sin(s_f * F::new(0.5));
                        let effective_r6 = r6 + op6_fb;

                        if algo_select == 0 {
                            let c1 = base_freq * r1 * h_f;
                            let c2 = base_freq * r2 * h_f;
                            
                            let op3_mod = l3 * F::max(F::new(0.1), r3);
                            let op4_mod = l4 * F::max(F::new(0.1), r4);
                            
                            carrier_freq = (c1 * l1 + c2 * l2) / F::max(F::new(0.05), l1 + l2);
                            mod_freq = base_freq * (r3 * l3 + r4 * l4 + r5 * l5 + effective_r6 * l6) / F::max(F::new(0.1), l3 + l4 + l5 + l6);
                            modulation_force = op3_mod + op4_mod;
                            
                            carrier_weight = (l1 + l2) * F::new(0.25);
                        } else {
                            carrier_freq = base_freq * r1 * h_f;
                            mod_freq = base_freq * r2;

                            let raw_stack = l2 * r2 + l3 * r3 + l4 * r4 + l5 * r5 + l6 * effective_r6;
                            modulation_force = F::log1p(raw_stack) * F::new(1.2);
                            
                            carrier_weight = l1 * F::new(0.4);
                        }

                        let target_freq_up = carrier_freq + (s_f * mod_freq);
                        let target_freq_down = F::max(F::new(1.0), carrier_freq - (s_f * mod_freq));

                        let dist_up = F::abs(bin_freq - target_freq_up);
                        let dist_down = F::abs(bin_freq - target_freq_down);
                        
                        let bin_width = sample_rate / samples_per_block;

                        if dist_up < bin_width * F::new(0.5) || dist_down < bin_width * F::new(0.5) {
                            let mut slot_amplitude = carrier_weight * saw_harmonic_amp;

                            if modulation_force > F::new(0.01) {
                                let fm_damping = F::exp(-(s_f * s_f) / (F::new(2.0) * F::max(F::new(0.1), modulation_force * modulation_force)));
                                let energy_compensation = F::new(1.0) / F::sqrt(F::new(1.0) + modulation_force);
                                
                                slot_amplitude = (slot_amplitude + modulation_force * F::new(0.08)) * fm_damping * energy_compensation;
                            } else {
                                if sideband > 1 { slot_amplitude = F::new(0.0); }
                            }

                            if k % 2 == 0 { real_spec += slot_amplitude; } else { imag_spec += slot_amplitude; }
                        }
                    }
                }
            }

            // Parallel-Filterbank
            let moog_gain: F = apply_moog_ladder::<F>(bin_freq, modulated_cutoff, moog_resonance);
            let oberheim_gain: F = apply_oberheim_sem::<F>(bin_freq, modulated_cutoff, oberheim_resonance, oberheim_mode); 

            let combined_filter_gain = (moog_gain + oberheim_gain) * F::new(0.5);
            let filtered_real = real_spec * combined_filter_gain;
            let filtered_imag = imag_spec * combined_filter_gain;

            // Prozeduraler Stereo-Hall
            let freq_factor = F::new(1.0) + (bin_freq * high_freq_damping * F::new(0.0001));
            let effective_decay = room_size_seconds / freq_factor;
            let amplitude = F::exp(-k_f / F::max(F::new(1.0), effective_decay * F::new(10.0)));

            let rand_l_real = (F::sin(k_f * F::new(12.9898)) - F::floor(F::sin(k_f * F::new(12.9898)))) * F::new(2.0) - F::new(1.0);
            let rand_l_imag = (F::cos(k_f * F::new(78.233)) - F::floor(F::cos(k_f * F::new(78.233)))) * F::new(2.0) - F::new(1.0);
            let rand_r_real = (F::sin(k_f * F::new(45.164)) - F::floor(F::sin(k_f * F::new(45.164)))) * F::new(2.0) - F::new(1.0);
            let rand_r_imag = (F::cos(k_f * F::new(92.741)) - F::floor(F::cos(k_f * F::new(92.741)))) * F::new(2.0) - F::new(1.0);

            let mid_real = (rand_l_real + rand_r_real) * F::new(0.25) * amplitude;
            let mid_imag = (rand_l_imag + rand_r_imag) * F::new(0.25) * amplitude;
            let diff_real = (rand_l_real - rand_r_real) * F::new(0.5) * amplitude;
            let diff_imag = (rand_l_imag - rand_r_imag) * F::new(0.5) * amplitude;

            // FIX: Deklaration der rechten Impulsantwort wiederhergestellt
            let ir_l_real = mid_real + stereo_width * diff_real;
            let ir_l_imag = mid_imag + stereo_width * diff_imag;
            let ir_r_real = mid_real - stereo_width * diff_real;
            let ir_r_imag = mid_imag - stereo_width * diff_imag;

            let wet_l_real = filtered_real * ir_l_real - filtered_imag * ir_l_imag;
            let wet_l_imag = filtered_real * ir_l_imag + filtered_imag * ir_l_real;
            let wet_r_real = filtered_real * ir_r_real - filtered_imag * ir_r_imag;
            let wet_r_imag = filtered_real * ir_r_imag + filtered_imag * ir_r_real;

let res_l_real = (F::new(1.0) - wet_dry_mix) * filtered_real + wet_dry_mix * wet_l_real;let res_l_imag = (F::new(1.0) - wet_dry_mix) * filtered_imag + wet_dry_mix * wet_l_imag;let res_r_real = (F::new(1.0) - wet_dry_mix) * filtered_real + wet_dry_mix * wet_r_real;let res_r_imag = (F::new(1.0) - wet_dry_mix) * filtered_imag + wet_dry_mix * wet_r_imag;// Kontinuierliche Phasenführung für den nahtlosen Übergang der IDFT
let angle = (F::new(2.0) * pi * k_f * F::cast_from(n)) / samples_per_block;let cos_a = F::cos(angle);let sin_a = F::sin(angle);final_sample_l += res_l_real * cos_a + res_l_imag * sin_a;final_sample_r += res_r_real * cos_a + res_r_imag * sin_a;}
let scale = F::new(2.0) / samples_per_block;let idx_l: usize = (n * 2) as usize;let idx_r: usize = (n * 2 + 1) as usize;output_stereo_audio[idx_l] = final_sample_l * scale * master_amp;output_stereo_audio[idx_r] = final_sample_r * scale * master_amp;}}