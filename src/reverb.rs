use cubecl::prelude::*;
/// Konfiguration für den prozeduralen virtuellen Raum
#[derive(Clone, Copy, Debug)]pub struct ProceduralIrConfig {
pub room_size_seconds: f32, // Länge des Nachhalls (z.B. 2.5 Sekunden)
pub high_freq_damping: f32, // Wie schnell hohe Frequenzen absorbiert werden
pub wet_dry_mix: f32, // 0.0 (trocken) bis 1.0 (vollständig verhallt)
}
/// Ein deterministischer Pseudo-Zufallszahlengenerator (PRNG) direkt für das GPU-Register./// Erlaubt es jedem Thread, exakt dasselbe deterministische weiße Rauschen für die IR zu reproduzieren.
#[cube]fn lcg_rand<F: Float>(seed: u32) -> F {
let next_seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
let lcg_max = 2147483647.0;
let val = (next_seed & 0x7fffffff) as f32 / lcg_max;
F::cast_from(val)
}
/// Generiert den Frequenz-Bin-Wert einer physikalischen Impulsantwort im Register./// Nutzt die echten DFT-Symmetrien für die Phasen-Zufälligkeit.
#[cube]fn get_procedural_ir_bin<F: Float>(bin_freq: F, k: u32, config: ProceduralIrConfig) -> (F, F) {
let decay_time = F::new(config.room_size_seconds);

// Frequenzabhängige Absorption: Höhere Frequenzen klingen im Raum schneller ab
let freq_factor = F::new(1.0) + (bin_freq * F::new(config.high_freq_damping) * F::new(0.0001));
let effective_decay = decay_time / freq_factor;

// Wir simulieren das exponentielle Abklingen der Raumenergie
// Für die DFT mitteln wir die Energie über die Frequenz-Bins
let amplitude = F::exp(-F::cast_from(k) / F::max(F::new(1.0), effective_decay * F::new(10.0)));

// Um ein natürliches, dichtes Diffusfeld zu erzeugen, benötigt jedes Frequenzband eine zufällige Phase
let phase_seed_real = k * 73856093;
let phase_seed_imag = k * 19349663;

let rand_real = lcg_rand::<F>(phase_seed_real) * F::new(2.0) - F::new(1.0);
let rand_imag = lcg_rand::<F>(phase_seed_imag) * F::new(2.0) - F::new(1.0);

// Rückgabe des komplexen Frequenzwerts der virtuellen Impulsantwort
(rand_real * amplitude, rand_imag * amplitude)
}
// --- DER VOLLSTÄNDIG FUSIONIERTE SYNTHESIZER + CONVOLUTION KERNEL ---

#[cube(launch)]pub fn cubek_hybrid_synth_with_convolution<F: Float>(
output_audio: &mut Array,
frequency: F,
sample_rate: F,
dyn_cutoff: F, // Vorberechneter Hüllkurven-Cutoff vom Host
ir_config: ProceduralIrConfig,
#[comptime] fft_size: u32,
) {
let n = ABSOLUTE_POS_X;

if n < fft_size {
    let mut final_sample = F::new(0.0);
    let pi = F::new(std::f32::consts::PI);
    let num_bins = fft_size / 2 + 1;

    // Frequenzbereichs-Pipeline
    for k in 0..num_bins {
        let bin_freq = (F::cast_from(k) * sample_rate) / F::cast_from(fft_size);

        // 1. GENERIERUNG & SUBTRAKTIVES FILTER (Wie zuvor)
        let mut synth_real = F::new(0.0);
        let mut synth_imag = F::new(0.0);

        if k > 0 && bin_freq < sample_rate / F::new(2.0) {
            let harmonic_number = bin_freq / frequency;
            let fract = harmonic_number - F::floor(harmonic_number);
            if fract < F::new(0.04) || fract > F::new(0.96) {
                let amp = F::new(1.0) / F::max(F::new(1.0), harmonic_number);
                if k % 2 == 0 { synth_real = amp; } else { synth_imag = amp; }
            }
        }

        // Exakter Brickwall-Filterübergang
        let filter_gain = if bin_freq <= dyn_cutoff { F::new(1.0) } else { F::new(0.0) };
        let filtered_synth_real = synth_real * filter_gain;
        let filtered_synth_imag = synth_imag * filter_gain;

        // 2. NOVEL APPROACH: COMPLEX CONVOLUTION IM FREQUENZBEREICH
        // Wir holen uns das komplexe Spektrum unserer on-the-fly generierten Impulsantwort
        let (ir_real, ir_imag) = get_procedural_ir_bin(bin_freq, k, ir_config);

        // Komplexe Multiplikation: (A + iB) * (C + iD) = (AC - BD) + i(AD + BC)
        let wet_real = filtered_synth_real * ir_real - filtered_synth_imag * ir_imag;
        let wet_imag = filtered_synth_real * ir_imag + filtered_synth_imag * ir_real;

        // Wet/Dry Mix auf Spektralebene anwenden
        let mix = F::new(ir_config.wet_dry_mix);
        let final_real = (F::new(1.0) - mix) * filtered_synth_real + mix * wet_real;
        let final_imag = (F::new(1.0) - mix) * filtered_synth_imag + mix * wet_imag;

        // 3. INVERSE DFT (Direkte Rücktransformation in den Zeitbereich)
        let angle = (F::new(2.0) * pi * F::cast_from(k) * F::cast_from(n)) / F::cast_from(fft_size);
        final_sample += final_real * F::cos(angle) - final_imag * F::sin(angle);
    }

    // Normalisierung des finalen Audiosignals
    output_audio[n] = final_sample / F::cast_from(fft_size);
}
}

