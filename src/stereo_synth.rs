#![allow(warnings)]
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct StereoIrConfig {
    pub room_size_seconds: f32,
    pub high_freq_damping: f32,
    pub wet_dry_mix: f32,
    pub stereo_width: f32,
}

#[cube(launch)]
pub fn cubek_true_stereo_synth_reverb<F: Float + CubeElement>(
    output_stereo_audio: &mut Array<F>,
    // Aktuelle Parameter
    frequency: F,
    dyn_cutoff: F,
    // PARAMETER-GLÄTTUNG: Historische Werte für cubek_interpolate
    old_frequency: F,
    old_cutoff: F,
    
    sample_rate: F,
    room_size_seconds: F,
    high_freq_damping: F,
    wet_dry_mix: F,
    stereo_width: F,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let mut final_sample_l = F::new(0.0);
        let mut final_sample_r = F::new(0.0);
        
        let pi = F::new(std::f32::consts::PI);
        let num_bins = fft_size / 2 + 1;

        // --- CUBEK_INTERPOLATE: LINEARER SPLINE PRO THREAD/SAMPLE n ---
        // Berechne den Fortschritts-Faktor t für diesen spezifischen Zeitschritt n
        let t = F::cast_from(n) / F::cast_from(fft_size);
        
        // Stufenloses Glätten der Modulationsziele im lokalen GPU-Register
        let current_freq = old_frequency + (t * (frequency - old_frequency));
        let current_cutoff = old_cutoff + (t * (dyn_cutoff - old_cutoff));

        for k in 0..num_bins {
            let bin_freq = (F::cast_from(k) * sample_rate) / F::cast_from(fft_size);

            let mut synth_real = F::new(0.0);
            let mut synth_imag = F::new(0.0);

            // Generierung mit der geglätteten Frequenz
            if k > 0 && bin_freq < sample_rate / F::new(2.0) {
                let harmonic_number = bin_freq / current_freq;
                let fract = harmonic_number - F::floor(harmonic_number);
                
                if fract < F::new(0.15) || fract > F::new(0.85) {
                    let amp = F::new(1.0) / F::max(F::new(1.0), F::floor(harmonic_number));
                    if k % 2 == 0 { synth_real = amp; } else { synth_imag = amp; }
                }
            }

            // Moog Filter mit geglättetem Cutoff
            let filter_gain = if bin_freq <= current_cutoff { F::new(1.0) } else { F::new(0.0) };
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

            let angle = (F::new(2.0) * pi * F::cast_from(k) * F::cast_from(n)) / F::cast_from(fft_size);
            let cos_a = F::cos(angle);
            let sin_a = F::sin(angle);

            final_sample_l += res_l_real * cos_a + res_l_imag * sin_a;
            final_sample_r += res_r_real * cos_a + res_r_imag * sin_a;
        }

        let scale = F::new(2.0) / F::cast_from(fft_size);
        
        let idx_l: usize = (n * 2) as usize;
        let idx_r: usize = (n * 2 + 1) as usize;
        
        output_stereo_audio[idx_l] = final_sample_l * scale;
        output_stereo_audio[idx_r] = final_sample_r * scale;
    }
}
