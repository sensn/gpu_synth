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
fn apply_oberheim_sem<F: Float + CubeElement>(
    bin_freq: F,
    cutoff: F,
    resonance: F,
    mode_select: u32,
) -> F {
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

// --- HOCH-PERFORMANTER FRAKTALER SÄGEZAHN-FM KERNEL ---
#[cube(launch)]
pub fn cubek_true_stereo_synth_reverb<F: Float + CubeElement>(
    output_stereo_audio: &mut Array<F>,
    frequency: F,
    old_frequency: F,
    dyn_cutoff: F,
    old_cutoff: F,
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
    sustain_level: F,
    release_time: F,
    // Note-Event-Zustand (vom Host verwaltet, absolute Sample-Zeitachse)
    // gate_is_on: 1 = Gate offen (A/D/S), 0 = Gate zu (Release)
    gate_is_on: u32,
    note_on_sample: u32,
    note_off_sample: u32,
    // NEU: Globaler Block-Zähler für nahtlose IDFT-Phase
    global_block_index: u32,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let mut final_sample_l = F::new(0.0);
        let mut final_sample_r = F::new(0.0);
        let pi = F::new(std::f32::consts::PI);
        let samples_per_block = F::cast_from(fft_size);

        // --- ABSOLUTE-TIME ADSR: Hüllkurve auf der globalen Sample-Zeitachse ---
        // Alle Zeiten in Sekunden, blockübergreifend kontinuierlich (kein Block-Eiern mehr).
        let global_sample = F::cast_from(global_block_index) * samples_per_block + F::cast_from(n);
        let t_on = F::cast_from(note_on_sample);
        let t_off = F::cast_from(note_off_sample);

        // Zeit seit Note-On in Sekunden (immer >= 0)
        let mut t_since_on = (global_sample - t_on) / sample_rate;
        if t_since_on < F::new(0.0) {
            t_since_on = F::new(0.0);
        }
        // Zeit seit Note-Off in Sekunden (nur relevant wenn Gate zu)
        let mut t_since_off = (global_sample - t_off) / sample_rate;
        if t_since_off < F::new(0.0) {
            t_since_off = F::new(0.0);
        }

        let mut master_amp = F::new(0.0);

        if gate_is_on == 1 {
            // --- ATTACK: linearer Anstieg von 0 auf 1 ---
            if t_since_on < attack_time {
                master_amp = t_since_on / F::max(F::new(0.001), attack_time);
            } else if t_since_on < attack_time + decay_time {
                // --- DECAY: exponentieller Abfall von 1 auf Sustain ---
                let t_decay = t_since_on - attack_time;
                let d = F::max(F::new(0.001), decay_time);
                master_amp = sustain_level + (F::new(1.0) - sustain_level) * F::exp(-t_decay / d);
            } else {
                // --- SUSTAIN ---
                master_amp = sustain_level;
            }
        } else {
            // --- RELEASE: exponentieller Abfall vom Pegel bei Note-Off ---
            // Rekonstruiert den Pegel, den die Hüllkurve zum Note-Off-Zeitpunkt hatte,
            // damit der Release nahtlos anschließt (kein Sprung).
            let amp_at_off = if t_off > t_on {
                let t_at_off = (t_off - t_on) / sample_rate;
                if t_at_off < attack_time {
                    t_at_off / F::max(F::new(0.001), attack_time)
                } else if t_at_off < attack_time + decay_time {
                    let t_decay = t_at_off - attack_time;
                    let d = F::max(F::new(0.001), decay_time);
                    sustain_level + (F::new(1.0) - sustain_level) * F::exp(-t_decay / d)
                } else {
                    sustain_level
                }
            } else {
                F::new(0.0)
            };
            let r = F::max(F::new(0.001), release_time);
            master_amp = amp_at_off * F::exp(-t_since_off / r);
        }

        let block_progress = F::cast_from(n) / samples_per_block;

        let sample_phase_delta =
            (F::new(2.0) * pi * lfo_frequency * block_progress * samples_per_block) / sample_rate;
        let lfo_mod = F::sin(lfo_accumulated_phase + sample_phase_delta);

        let base_freq = old_frequency + (block_progress * (frequency - old_frequency));
        let base_cutoff = old_cutoff + (block_progress * (dyn_cutoff - old_cutoff));

        let mut modulated_cutoff = base_cutoff + (lfo_mod * lfo_depth);
        if modulated_cutoff < F::new(50.0) {
            modulated_cutoff = F::new(50.0);
        }

           // ... (Dein ADSR- und Operator-Setup bleibt identisch) ...

                // ... (Dein ADSR- und LFO-Setup bleibt absolut unberührt) ...

        let num_bins = fft_size / 2 + 1;
        let samples_per_block_f = samples_per_block;
        
        let block_start_sample = F::cast_from(global_block_index) * samples_per_block_f;
        let local_sample_n = F::cast_from(n);
        let block_progress = local_sample_n / samples_per_block_f;

        // Kontinuierliche Frequenz- und Cutoff-Interpolation pro Sample
        let current_base_freq = old_frequency + (block_progress * (frequency - old_frequency));
        let current_modulated_cutoff = old_cutoff + (block_progress * (dyn_cutoff - old_cutoff));

        // --- CUBECL FIX: Explizites Auslesen der Array-Indizes als Skalare vom Typ F ---
        let r1 = op_ratios[0]; let l1 = op_levels[0];
        let r2 = op_ratios[1]; let l2 = op_levels[1];
        let r3 = op_ratios[2]; let l3 = op_levels[2];
        let r4 = op_ratios[3]; let l4 = op_levels[3];
        let r5 = op_ratios[4]; let l5 = op_levels[4];
        let r6 = op_ratios[5]; let l6 = op_levels[5];

        // 1. DX7 CORE FREQUENZ-BERECHNUNG (Jetzt mathematisch korrekt mit Skalaren)
        let f1 = current_base_freq * r1;
        let f2 = current_base_freq * r2;
        let f3 = current_base_freq * r3;
        let f4 = current_base_freq * r4;
        let f5 = current_base_freq * r5;
        let f6 = current_base_freq * r6;

        let mut carrier_freq = current_base_freq;
        let mut mod_freq = current_base_freq;
        let mut modulation_force = F::new(0.0);
        let mut base_amp = F::new(0.0);

        let op6_feedback_noise_avg = l6 * l6 * F::max(F::new(0.1), r6) * F::new(0.5);
        let f6_effective = f6 + op6_feedback_noise_avg;

        if algo_select == 0 {
            let op3_mod = l3 * F::max(F::new(0.1), r3);
            let op4_mod = l4 * F::max(F::new(0.1), r4);
            
            // Rechnet nun stabil mit dem echten Typen F
            carrier_freq = (f1 * l1 + f2 * l2) / F::max(F::new(0.05), l1 + l2);
            mod_freq = (f3 * l3 + f4 * l4 + f5 * l5 + f6_effective * l6) / F::max(F::new(0.1), l3 + l4 + l5 + l6);
            modulation_force = op3_mod + op4_mod;
            base_amp = (l1 + l2) * F::new(0.3);
        } else {
            carrier_freq = f1;
            mod_freq = f2;
            let raw_stack = l2 * r2 + l3 * r3 + l4 * r4 + l5 * r5 + l6 * (f6_effective / F::max(F::new(1.0), current_base_freq));
            modulation_force = F::log1p(raw_stack) * F::new(1.5);
            base_amp = l1 * F::new(0.5);
        }

        // 2. DIREKTE GENERIERUNG DER SEITENBÄNDER
        let max_sidebands = 16; 

        for sideband in 0..max_sidebands {
            let order = F::cast_from(sideband);
            
            // Berechne Amplitudenabfall der Ordnung
            let mut sideband_amp = base_amp / (F::new(1.0) + order * order);
            if modulation_force > F::new(0.01) {
                sideband_amp = sideband_amp * (F::new(1.0) + modulation_force * F::new(0.2));
            } else if sideband > 0 {
                sideband_amp = F::new(0.0);
            }

            if sideband_amp > F::new(0.0001) {
                for sign_idx in 0..2 {
                    if !(sideband == 0 && sign_idx == 1) { 
                        
                        let sign = if sign_idx == 0 { F::new(-1.0) } else { F::new(1.0) };
                        let real_freq = carrier_freq + (sign * order * mod_freq);

                        if real_freq > F::new(10.0) && real_freq < sample_rate / F::new(2.0) {
                            
                            let filter_gain = if real_freq <= current_modulated_cutoff { F::new(1.0) } else { F::new(0.0) };
                            
                            if filter_gain > F::new(0.0) {
                                let filtered_synth_real = sideband_amp * filter_gain;

                                // REVERB SEEDING
                                let rand_l_real = (F::sin(order * F::new(12.9898)) - F::floor(F::sin(order * F::new(12.9898)))) * F::new(2.0) - F::new(1.0);
                                let rand_l_imag = (F::cos(order * F::new(78.2330)) - F::floor(F::cos(order * F::new(78.2330)))) * F::new(2.0) - F::new(1.0);
                                let rand_r_real = (F::sin(order * F::new(45.1640)) - F::floor(F::sin(order * F::new(45.1640)))) * F::new(2.0) - F::new(1.0);
                                let rand_r_imag = (F::cos(order * F::new(92.7410)) - F::floor(F::cos(order * F::new(92.7410)))) * F::new(2.0) - F::new(1.0);

                                let freq_factor = F::new(1.0) + (real_freq * high_freq_damping * F::new(0.0001));
                                let effective_decay = room_size_seconds / freq_factor;
                                let amplitude_decay = F::exp(-order / F::max(F::new(1.0), effective_decay * F::new(2.0))) * F::new(0.15);

                                let ir_l_real = rand_l_real * amplitude_decay; let ir_l_imag = rand_l_imag * amplitude_decay;
                                let ir_r_real = rand_r_real * amplitude_decay; let ir_r_imag = rand_r_imag * amplitude_decay;

                                let mid_real = (ir_l_real + ir_r_real) * F::new(0.5); let mid_imag = (ir_l_imag + ir_r_imag) * F::new(0.5);
                                let final_l_real = mid_real + stereo_width * (ir_l_real - mid_real); let final_l_imag = mid_imag + stereo_width * (ir_l_imag - mid_imag);
                                let final_r_real = mid_real + stereo_width * (ir_r_real - mid_real); let final_r_imag = mid_imag + stereo_width * (ir_r_imag - mid_imag);

                                let wet_l_real = filtered_synth_real * final_l_real; let wet_l_imag = filtered_synth_real * final_l_imag;
                                let wet_r_real = filtered_synth_real * final_r_real; let wet_r_imag = filtered_synth_real * final_r_imag;

                                let res_l_real = (F::new(1.0) - wet_dry_mix) * filtered_synth_real + wet_dry_mix * wet_l_real;
                                let res_l_imag = wet_dry_mix * wet_l_imag;
                                let res_r_real = (F::new(1.0) - wet_dry_mix) * filtered_synth_real + wet_dry_mix * wet_r_real;
                                let res_r_imag = wet_dry_mix * wet_r_imag;

                                // ABSOLUTE, BLOCKÜBERGREIFEND LÜCKENLOSE PHASEN-ROTATION
                                let exact_k_f = (real_freq * samples_per_block_f) / sample_rate;
                                let base_exact_k_f = (old_frequency * samples_per_block_f) / sample_rate;
                                
                                let phase_history = (F::new(2.0) * pi * base_exact_k_f * block_start_sample) / samples_per_block_f;
                                let phase_local = (F::new(2.0) * pi * exact_k_f * local_sample_n) / samples_per_block_f;
                                
                                let angle = phase_history + phase_local;
                                let cos_a = F::cos(angle); let sin_a = F::sin(angle);

                                final_sample_l += res_l_real * cos_a - res_l_imag * sin_a;
                                final_sample_r += res_r_real * cos_a - res_r_imag * sin_a;
                            }
                        }
                    }
                }
            }
        }

        // 3. ABSOLUT SICHERES INTEGRAL-GAIN-STAGING & LIMITER
        let scale = F::new(0.05); 
        let idx_l: usize = (n * 2) as usize;
        let idx_r: usize = (n * 2 + 1) as usize;
        
        let out_l = final_sample_l * scale * master_amp;
        let out_r = final_sample_r * scale * master_amp;

        output_stereo_audio[idx_l] = F::max(-F::new(1.0), F::min(F::new(1.0), out_l));
        output_stereo_audio[idx_r] = F::max(-F::new(1.0), F::min(F::new(1.0), out_r));
    }
}
