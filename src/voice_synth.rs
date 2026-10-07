#![allow(warnings)]
use cubecl::prelude::*;

// ============================================================
// STUFE 1 (Refactoring_plan.md): TROCKENER FM-SYNTH-KERNEL
// ============================================================
// Pro Stimme ein Launch. Enthält alles, was stimmen-spezifisch ist:
//   - Absolute-Zeit-ADSR (Gate/Note-On/Note-Off auf globaler Sample-Achse)
//   - Portamento (old_frequency → frequency, per-Sample geglättet)
//   - FM-Seitenband-Generierung (Algo 0/1, 6 Operatoren)
//   - Phasenrekonstruktion (historischer Anker + lokale IDFT-Phase)
//   - Brickwall-Filter (modulated_cutoff, LFO-moduliert)
// Ausgabe: TROCKENES MONO-Signal (L=R=dry) in den planaren Sammel-Buffer
// voices_dry — die eigene Stimme adressiert sich über voice_index selbst:
//   voices_dry[voice_index * fft_size * 2 + n * 2]     = L
//   voices_dry[voice_index * fft_size * 2 + n * 2 + 1] = R
// Filter/Reverb/Width/Wet-Dry macht der Mixer (mixer_kernel.rs) EINMAL
// für alle Stimmen.

#[cube(launch)]
pub fn cubek_voice_synth<F: Float + CubeElement>(
    voices_dry: &mut Array<F>,
    voice_index: u32,
    frequency: F,
    old_frequency: F,
    dyn_cutoff: F,
    old_cutoff: F,
    sample_rate: F,
    lfo_accumulated_phase: F,
    op_ratios: &Array<F>,
    op_levels: &Array<F>,
    algo_select: u32,
    lfo_frequency: F,
    lfo_depth: F,
    attack_time: F,
    decay_time: F,
    sustain_level: F,
    release_time: F,
    // Note-Event-Zustand (vom Host verwaltet, absolute Sample-Zeitachse)
    gate_is_on: u32,
    note_on_sample: u32,
    note_off_sample: u32,
    // Globaler Block-Zähler für nahtlose IDFT-Phase
    global_block_index: u32,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let mut final_sample = F::new(0.0);
        let pi = F::new(std::f32::consts::PI);
        let samples_per_block = F::cast_from(fft_size);

        // --- ABSOLUTE-TIME ADSR: Hüllkurve auf der globalen Sample-Zeitachse ---
        let global_sample = F::cast_from(global_block_index) * samples_per_block + F::cast_from(n);
        let t_on = F::cast_from(note_on_sample);
        let t_off = F::cast_from(note_off_sample);

        let mut t_since_on = (global_sample - t_on) / sample_rate;
        if t_since_on < F::new(0.0) {
            t_since_on = F::new(0.0);
        }
        let mut t_since_off = (global_sample - t_off) / sample_rate;
        if t_since_off < F::new(0.0) {
            t_since_off = F::new(0.0);
        }

        let mut master_amp = F::new(0.0);

        if gate_is_on == 1 {
            if t_since_on < attack_time {
                master_amp = t_since_on / F::max(F::new(0.001), attack_time);
            } else if t_since_on < attack_time + decay_time {
                let t_decay = t_since_on - attack_time;
                let d = F::max(F::new(0.001), decay_time);
                master_amp = sustain_level + (F::new(1.0) - sustain_level) * F::exp(-t_decay / d);
            } else {
                master_amp = sustain_level;
            }
        } else {
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

        // --- LFO: Phase-akkumulierende Sinus-Modulation, blockübergreifend stetig ---
        let sample_phase_delta =
            (F::new(2.0) * pi * lfo_frequency * block_progress * samples_per_block) / sample_rate;
        let lfo_mod = F::sin(lfo_accumulated_phase + sample_phase_delta);

        // --- PER-SAMPLE GLÄTTUNG: Frequenz + Cutoff (Portamento) ---
        let current_base_freq = old_frequency + (block_progress * (frequency - old_frequency));
        let base_cutoff = old_cutoff + (block_progress * (dyn_cutoff - old_cutoff));

        let mut modulated_cutoff = base_cutoff + (lfo_mod * lfo_depth);
        if modulated_cutoff < F::new(50.0) {
            modulated_cutoff = F::new(50.0);
        }

        let samples_per_block_f = samples_per_block;
        let block_start_sample = F::cast_from(global_block_index) * samples_per_block_f;
        let local_sample_n = F::cast_from(n);

        let r1 = op_ratios[0]; let l1 = op_levels[0];
        let r2 = op_ratios[1]; let l2 = op_levels[1];
        let r3 = op_ratios[2]; let l3 = op_levels[2];
        let r4 = op_ratios[3]; let l4 = op_levels[3];
        let r5 = op_ratios[4]; let l5 = op_levels[4];
        let r6 = op_ratios[5]; let l6 = op_levels[5];

        // --- SCHRITT A: AKTUELLER BLOCK (MOMENTANFREQUENZEN) ---
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

        // --- SCHRITT B: VORHERIGER BLOCK (HISTORISCHE ANKER-FREQUENZEN) ---
        let old_f1 = old_frequency * r1;
        let old_f2 = old_frequency * r2;
        let old_f3 = old_frequency * r3;
        let old_f4 = old_frequency * r4;
        let old_f5 = old_frequency * r5;
        let old_f6 = old_frequency * r6;

        let mut old_carrier_freq = old_frequency;
        let mut old_mod_freq = old_frequency;
        let old_f6_effective = old_f6 + (l6 * l6 * F::max(F::new(0.1), r6) * F::new(0.5));

        if algo_select == 0 {
            old_carrier_freq = (old_f1 * l1 + old_f2 * l2) / F::max(F::new(0.05), l1 + l2);
            old_mod_freq = (old_f3 * l3 + old_f4 * l4 + old_f5 * l5 + old_f6_effective * l6) / F::max(F::new(0.1), l3 + l4 + l5 + l6);
        } else {
            old_carrier_freq = old_f1;
            old_mod_freq = old_f2;
        }

        // 2. DIREKTE GENERIERUNG DER SEITENBÄNDER
        let max_sidebands = 16;

        for sideband in 0..max_sidebands {
            let order = F::cast_from(sideband);

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

                            let filter_gain = if real_freq <= modulated_cutoff { F::new(1.0) } else { F::new(0.0) };

                            if filter_gain > F::new(0.0) {
                                let filtered_synth_real = sideband_amp * filter_gain;

                                // --- KORREKTUR: SEITENBAND-SPEZIFISCHE PHASEN-STETIGKEIT ---
                                let old_real_freq = old_carrier_freq + (sign * order * old_mod_freq);

                                let exact_k_f = (real_freq * samples_per_block_f) / sample_rate;
                                let base_exact_k_f = (old_real_freq * samples_per_block_f) / sample_rate;

                                let phase_history = (F::new(2.0) * pi * base_exact_k_f * block_start_sample) / samples_per_block_f;
                                let phase_local = (F::new(2.0) * pi * exact_k_f * local_sample_n) / samples_per_block_f;

                                let angle = phase_history + phase_local;
                                let cos_a = F::cos(angle);

                                final_sample += filtered_synth_real * cos_a;
                            }
                        }
                    }
                }
            }
        }

        // 3. GAIN-STAGING (kein Limiter — der Mixer clippt am Ende)
        let scale = F::new(0.05);
        let out = final_sample * scale * master_amp;

        // TROCKENES MONO: L=R=dry. Stereo-Width macht der Mixer.
        // Adressierung: Stimme v beginnt bei v * fft_size * 2 (planar).
        let base: usize = (voice_index * fft_size * 2) as usize;
        let idx_l: usize = base + (n * 2) as usize;
        let idx_r: usize = base + (n * 2 + 1) as usize;
        voices_dry[idx_l] = out;
        voices_dry[idx_r] = out;
    }
}
