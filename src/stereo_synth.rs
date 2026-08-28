#![allow(warnings)]
use cubecl::prelude::*;

#[cube(launch)]
pub fn cubek_true_stereo_synth_reverb<F: Float + CubeElement>(
    output_stereo_audio: &mut Array<F>,
    frequency: F,
    dyn_cutoff: F,
    old_frequency: F,
    old_cutoff: F,
    sample_rate: F,
    room_size_seconds: F,
    high_freq_damping: F,
    wet_dry_mix: F,
    stereo_width: F,
    // NEU: Globale Zeit-Parameter für ADSR & LFO
    global_block_index: u32,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let mut final_sample_l = F::new(0.0);
        let mut final_sample_r = F::new(0.0);
        
        let pi = F::new(std::f32::consts::PI);
        let num_bins = fft_size / 2 + 1;

        // --- CUBEK_STD: ZEIT-PROJEKTION PRO SAMPLE ---
        // Berechne die absolute Zeit dieses spezifischen Samples im Gesamt-Stream
        let samples_per_block = F::cast_from(fft_size);
        let total_samples = (F::cast_from(global_block_index) * samples_per_block) + F::cast_from(n);
        let current_time = total_samples / sample_rate;

        // --- ADSR HÜLLKURVEN MATHEMATIK (CUBEK_STD STIL) ---
        let attack_time = F::new(0.1);  // 100ms Einschwingzeit
        let decay_time = F::new(0.3);   // 300ms Abschwellzeit
        let sustain_lvl = F::new(0.6);  // 60% Haltepegel
        
        let mut adsr_amp = F::new(0.0);
        
        if current_time < attack_time {
            adsr_amp = current_time / attack_time;
        } else if current_time < (attack_time + decay_time) {
            let decay_progress = (current_time - attack_time) / decay_time;
            adsr_amp = F::new(1.0) - (decay_progress * (F::new(1.0) - sustain_lvl));
        } else {
            adsr_amp = sustain_lvl;
        }

        // --- LFO MODULATION FÜR DEN CUTOFF ---
        let lfo_freq = F::new(5.0); // 5 Hz Modulationsgeschwindigkeit
        let lfo_depth = F::new(400.0); // Modulations-Breite in Hertz
        // Sinus-Schwingung basierend auf der absoluten Zeit
        let lfo_mod = F::sin(F::new(2.0) * pi * lfo_freq * current_time);

        // --- SLOPE INTERPOLATION (SLIDER-GLÄTTUNG) ---
        let t = F::cast_from(n) / samples_per_block;
        let base_freq = old_frequency + (t * (frequency - old_frequency));
        let base_cutoff = old_cutoff + (t * (dyn_cutoff - old_cutoff));

        // Kombiniere manuellen Cutoff mit der LFO-Modulation
        let mut modulated_cutoff = base_cutoff + (lfo_mod * lfo_depth);
        // Schutzgating via cubek_std Limitierung: Cutoff darf niemals unter 50Hz fallen
        if modulated_cutoff < F::new(50.0) { modulated_cutoff = F::new(50.0); }

        for k in 0..num_bins {
            let bin_freq = (F::cast_from(k) * sample_rate) / samples_per_block;

            let mut synth_real = F::new(0.0);
            let mut synth_imag = F::new(0.0);

            if k > 0 && bin_freq < sample_rate / F::new(2.0) {
                let harmonic_number = bin_freq / base_freq;
                let fract = harmonic_number - F::floor(harmonic_number);
                
                if fract < F::new(0.15) || fract > F::new(0.85) {
                    let amp = F::new(1.0) / F::max(F::new(1.0), F::floor(harmonic_number));
                    if k % 2 == 0 { synth_real = amp; } else { synth_imag = amp; }
                }
            }

            // Anwendung des modulierten Cutoffs
            let filter_gain = if bin_freq <= modulated_cutoff { F::new(1.0) } else { F::new(0.0) };
            let filtered_synth_real = synth_real * filter_gain;
            let filtered_synth_imag = synth_imag * filter_gain;

            let freq_factor = F::new(1.0) + (bin_freq * high_freq_damping * F::new(0.0001));
            let effective_decay = room_size_seconds / freq_factor;
            let amplitude = F::exp(-F::cast_from(k) / F::max(F::new(1.0), effective_decay * F::new(10.0)));

            let k_f = F::cast_from(k);
            let rand_l_real = (F::sin(k_f * F::new(12.9898)) - F::floor(F::sin(k_f * F::new(12.9898)))) * F::new(2.0) - F::new(1.0);
            let rand_l_imag = (F::cos(k_f * F::new(78.233)) - F::floor(F::cos(k_f * F::new(78.233)))) * F::new(2.0) - F::new(1.0);
            let rand_r_real = (F::sin(k_f * F::new(45.164)) - F::floor(F::sin(k_f * F::new(45.164)))) * F::new(2.0) - F::new(1.0);
            let rand_r_imag = (F::cos(k_f * F::new(92.741)) - F::floor(F::cos(k_f * F::new(92.741)))) * F::new(2.0) - F::new(1.0);

            let ir_l_real = rand_l_real * amplitude;
            let ir_l_imag = rand_l_imag * amplitude;
            let ir_r_real = rand_r_real * amplitude;
            let ir_r_imag = rand_r_imag * amplitude;

            let mid_real = (ir_l_real + ir_r_real) * F::new(0.5);
            let mid_imag = (ir_l_imag + ir_r_imag) * F::new(0.5);

            let final_l_real = mid_real + stereo_width * (ir_l_real - mid_real);
            let final_l_imag = mid_imag + stereo_width * (ir_l_imag - mid_imag);
            let final_r_real = mid_real + stereo_width * (ir_r_real - mid_real);
            let final_r_imag = mid_imag + stereo_width * (ir_r_imag - mid_imag);

            let wet_l_real = filtered_synth_real * final_l_real - filtered_synth_imag * final_l_imag;
            let wet_l_imag = filtered_synth_real * final_l_imag + filtered_synth_imag * final_l_real;
            let wet_r_real = filtered_synth_real * final_r_real - filtered_synth_imag * final_r_imag;
            let wet_r_imag = filtered_synth_real * final_r_imag + filtered_synth_imag * final_r_real;

            let res_l_real = (F::new(1.0) - wet_dry_mix) * filtered_synth_real + wet_dry_mix * wet_l_real;
            let res_l_imag = (F::new(1.0) - wet_dry_mix) * filtered_synth_imag + wet_dry_mix * wet_l_imag;
            let res_r_real = (F::new(1.0) - wet_dry_mix) * filtered_synth_real + wet_dry_mix * wet_r_real;
            let res_r_imag = (F::new(1.0) - wet_dry_mix) * filtered_synth_imag + wet_dry_mix * wet_r_imag;

            let angle = (F::new(2.0) * pi * F::cast_from(k) * F::cast_from(n)) / samples_per_block;
            let cos_a = F::cos(angle);
            let sin_a = F::sin(angle);

            final_sample_l += res_l_real * cos_a + res_l_imag * sin_a;
            final_sample_r += res_r_real * cos_a + res_r_imag * sin_a;
        }

        let scale = F::new(2.0) / samples_per_block;
        
        let idx_l: usize = (n * 2) as usize;
        let idx_r: usize = (n * 2 + 1) as usize;
        
        // Multipliziere das Endergebnis mit der ADSR-Lautstärkehüllkurve
        output_stereo_audio[idx_l] = final_sample_l * scale * adsr_amp;
        output_stereo_audio[idx_r] = final_sample_r * scale * adsr_amp;
    }
}
