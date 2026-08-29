#![allow(warnings)]
use cubecl::prelude::*;

// Hilfsfunktion: Moog-Ladder Filter-Emulation (24dB/Okt mit Resonanz-Feedback)
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

// Hilfsfunktion: Oberheim SEM 2-Pol Multi-Mode Filter (u32-Gating für WASM)
#[cube]
fn apply_oberheim_sem<F: Float + CubeElement>(bin_freq: F, cutoff: F, resonance: F, mode_select: u32) -> F {
    let f = bin_freq / F::max(F::new(1.0), cutoff);
    let f2 = f * f;
    let damping = F::new(1.0) / F::max(F::new(0.1), F::new(1.0) - resonance);
    let denominator = (F::new(1.0) - f2) * (F::new(1.0) - f2) + (f2 / (damping * damping));
    let sqrt_denom = F::sqrt(denominator);

    let mut response = F::new(0.0);
    if mode_select == 0 {
        response = F::new(1.0) / sqrt_denom;
    } else if mode_select == 1 {
        response = f2 / sqrt_denom;
    } else if mode_select == 2 {
        response = f / sqrt_denom;
    } else {
        response = F::abs(F::new(1.0) - f2) / sqrt_denom;
    }
    response
}

// --- DER PARALLEL VEKTORISIERTE 6-OP DX7 HYBRID KERNEL ---
#[cube(launch)]
pub fn cubek_true_stereo_synth_reverb<F: Float + CubeElement>(
    output_stereo_audio: &mut Array<F>,
    frequency: F, old_frequency: F,
    dyn_cutoff: F, old_cutoff: F,
    sample_rate: F,
    lfo_accumulated_phase: F,
    fm_ratio: F,  // Master FM-Weite
    fm_index: F,  // Master FM-Intensität
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

        // LFO via kontinuierlicher Phasenführung
        let sample_phase_delta = (F::new(2.0) * pi * lfo_frequency * block_progress * samples_per_block) / sample_rate;
        let lfo_mod = F::sin(lfo_accumulated_phase + sample_phase_delta);

        // Slider-Glättung
        let base_freq = old_frequency + (block_progress * (frequency - old_frequency));
        let base_cutoff = old_cutoff + (block_progress * (dyn_cutoff - old_cutoff));
        
        let mut modulated_cutoff = base_cutoff + (lfo_mod * lfo_depth);
        if modulated_cutoff < F::new(50.0) { modulated_cutoff = F::new(50.0); }

        let num_bins = fft_size / 2 + 1;

        // =========================================================================
        // 🎛️ ARCHITEKTUR: MATRIX DER 6 OPERATOREN (UNROLLING IM GPU REGISTER)
        // =========================================================================
        
        // 1. FREQUENZ-RATIOS (Die Multiplikatoren für jeden Operator)
        let op1_ratio = F::new(1.00); // Grundton Carrier
        let op2_ratio = F::new(1.00); // Zweiter Carrier (leicht detuned via Hall-Matrix)
        let op3_ratio = F::new(3.50); // Schneidender Oberton-Modulator
        let op4_ratio = F::new(2.00); // Sub-Modulator
        let op5_ratio = F::new(7.00); // Metallischer Klick-Modulator (Glocken-Charakter)
        let op6_ratio = F::new(0.50); // Tiefer Sub-Bass Operator

        // 2. UNABHÄNGIGE INTERNE ADSR-HÜLLKURVEN FÜR DIE MODULATOREN
        // OP5 simuliert den typischen perkussiven DX7-"Attack-Strike"
        let mut op5_env = F::new(1.0) - (block_progress * F::new(2.0)); 
        if op5_env < F::new(0.0) { op5_env = F::new(0.0); } // Schneller Klick-Drop

        // OP3 (Hauptmodulator) schwillt dynamisch über den Decay-Slider an/ab
        let mut op3_env = master_amp; 

        // 3. EFFEKTIVE AMPILLITUDEN-LEVELS (Skaliert über den globalen fm_index)
        let op1_level = F::new(1.0);
        let op2_level = F::new(0.7);
        let op3_level = fm_index * F::new(1.2) * op3_env;
        let op4_level = fm_index * F::new(0.8);
        let op5_level = fm_index * F::new(2.5) * op5_env; // Starker Initial-Punch
        let op6_level = fm_index * F::new(0.4);

        for k in 0..num_bins {
            let k_f = F::cast_from(k);
            let bin_freq = (k_f * sample_rate) / samples_per_block;

            let mut real_spec = F::new(0.0);
            let mut imag_spec = F::new(0.0);

            if k > 0 && bin_freq < sample_rate / F::new(2.0) {
                
                // SCHLEIFE ÜBER DAS BANDBEGRENZTE SÄGEZAHN-RÜCKGRAT
                for h in 1..33 {
                    let h_f = F::cast_from(h);
                    let carrier_harmonic_freq = base_freq * h_f;

                    if carrier_harmonic_freq < sample_rate / F::new(2.0) {
                        let saw_base_amp = F::new(1.0) / h_f;

                        // =========================================================================
                        // 🧬 DX7 ALGORITHMUS-MATHEMATIK (FUSIONIERTE DIGITALE KASKADE)
                        // Kette: [OP6 -> OP5] -> OP4 -> OP3 -> [OP1 + OP2 Carrier]
                        // =========================================================================
                        
                        // Ebene A: Sub-Bass und Metall-Punch modulieren die Mitte
                        let mod_phase_layer_a = (base_freq * op6_ratio * op6_level) + (base_freq * op5_ratio * op5_level);
                        
                        // Ebene B: Moduliert den Haupt-Modulator (OP4 -> OP3)
                        let mod_phase_layer_b = (base_freq * op4_ratio * op4_level) + mod_phase_layer_a;
                        
                        // Ebene C: Der finale komplexe Frequenz-Hub, der auf das Sägezahn-Rückgrat trifft
                        let final_dx7_mod_freq = base_freq * fm_ratio * (op1_ratio + op2_ratio + (op3_ratio * op3_level) + mod_phase_layer_b);

                        let distance_to_harmonic = F::abs(bin_freq - carrier_harmonic_freq);
                        let harmonic_step = distance_to_harmonic / F::max(F::new(1.0), final_dx7_mod_freq);
                        let fract = harmonic_step - F::floor(harmonic_step);

                        if fract < F::new(0.18) || fract > F::new(0.82) {
                            let order = F::floor(harmonic_step);
                            
                            // Exponentielle Dämpfung basierend auf der kaskadierten Gesamt-FM-Energie
                            let total_index = op3_level + op4_level + op5_level;
                            let width_factor = F::max(F::new(0.001), total_index);
                            let fm_sideband_damping = F::exp(-order / width_factor);

                            let compensation = F::new(1.0) / F::sqrt(F::new(1.0) + total_index * F::new(0.4));
                            let final_amplitude = saw_base_amp * fm_sideband_damping * compensation;

                            if k % 2 == 0 { real_spec += final_amplitude; } else { imag_spec += final_amplitude; }
                        }
                    }
                }
            }

            // Parallel-Filterbank (Moog + Oberheim)
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

let ir_l_real = mid_real + stereo_width * diff_real;let ir_l_imag = mid_imag + stereo_width * diff_imag;let ir_r_real = mid_real - stereo_width * diff_real;let ir_r_imag = mid_imag - stereo_width * diff_imag;let wet_l_real = filtered_real * ir_l_real - filtered_imag * ir_l_imag;let wet_l_imag = filtered_real * ir_l_imag + filtered_imag * ir_l_real;let wet_r_real = filtered_real * ir_r_real - filtered_imag * ir_r_imag;let wet_r_imag = filtered_real * ir_r_imag + filtered_imag * ir_r_real;let res_l_real = (F::new(1.0) - wet_dry_mix) * filtered_real + wet_dry_mix * wet_l_real;let res_l_imag = (F::new(1.0) - wet_dry_mix) * filtered_imag + wet_dry_mix * wet_l_imag;let res_r_real = (F::new(1.0) - wet_dry_mix) * filtered_real + wet_dry_mix * wet_r_real;let res_r_imag = (F::new(1.0) - wet_dry_mix) * filtered_imag + wet_dry_mix * wet_r_imag;let angle = (F::new(2.0) * pi * k_f * F::cast_from(n)) / samples_per_block;let cos_a = F::cos(angle);let sin_a = F::sin(angle);final_sample_l += res_l_real * cos_a + res_l_imag * sin_a;final_sample_r += res_r_real * cos_a + res_r_imag * sin_a;}let scale = F::new(2.0) / samples_per_block;let idx_l: usize = (n * 2) as usize;let idx_r: usize = (n * 2 + 1) as usize;// Finaler Ausgang skaliert mit der globalen ADSR
output_stereo_audio[idx_l] = final_sample_l * scale * master_amp;output_stereo_audio[idx_r] = final_sample_r * scale * master_amp;}}