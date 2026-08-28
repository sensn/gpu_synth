use cubecl::prelude::*;
/// Konfiguration für die zeitabhängigen ADSR-Hüllkurven (in Sekunden)
#[derive(Clone, Copy, Debug)]pub struct AdsrConfig {
pub attack: f32,
pub decay: f32,
pub sustain: f32, // 0.0 bis 1.0
pub release: f32,
}
/// Auswahl des Oberheim-Filtermodus
#[derive(Clone, Copy, Debug)]pub enum OberheimMode {
LowPass,
HighPass,
BandPass,
Notch,
}
/// Globale Konfigurationsstruktur für den Hybridsynthesizer
#[derive(Clone, Copy, Debug)]pub struct HybridSynthConfig {
pub sample_rate: f32,
pub fm_mod_index: f32, // Stärke der FM-Modulation
pub fm_ratio: f32, // Frequenzverhältnis Carrier/Modulator
pub moog_cutoff: f32, // Basis-Cutoff für das Moog-Filter
pub moog_resonance: f32, // Resonanz (0.0 bis 4.0+)
pub oberheim_cutoff: f32, // Basis-Cutoff für das Oberheim-Filter
pub oberheim_resonance: f32, // Resonanz Dämpfungsfaktor
pub oberheim_mode: OberheimMode,
}
// --- SUB-ROUTINEN FÜR DIE SPEKTRALEN FILTER (KERNEL INLINING) ---
/// Mathematisches Modell eines Moog-Ladder-Filters im Frequenzbereich (Phasenlinearisiert)/// Inspiriert durch die 4-Pol-Kaskade (24dB/Oktave) mit Resonanz-Feedback
#[cube]fn apply_moog_ladder_filter<F: Float>(bin_freq: F, cutoff: F, resonance: F) -> F {
// Frequenz normalisieren bezogen auf den Cutoff-Punkt
let f = bin_freq / F::max(F::new(1.0), cutoff);
// Die klassische Moog-Übertragungsfunktion im Frequenzbereich (4 identische Pol-Stufen)
// H(f) = 1 / (1 + j*f)^4. Wir approximieren die Dämpfung (Magnitude):
let pole_step = F::new(1.0) / (F::new(1.0) + f * f);
let mut magnitude = pole_step * pole_step * pole_step * pole_step;

// Resonanz-Peak nahe der Grenzfrequenz hinzufügen
let distance = F::abs(bin_freq - cutoff);
if distance < cutoff * F::new(0.08) {
    magnitude += resonance * (F::new(1.0) - (distance / (cutoff * F::new(0.08))));
}

magnitude
}
/// Mathematisches Modell des 2-Pol Oberheim SEM Filters (12dB/Oktave Multi-Mode)
#[cube]fn apply_oberheim_multimode<F: Float>(bin_freq: F, cutoff: F, resonance: F, mode: OberheimMode) -> F {
let f = bin_freq / F::max(F::new(1.0), cutoff);
let f2 = f * f;
// Dämpfungs-Nenner basierend auf der Resonanz (Q-Faktor)
let dämpfung = F::new(1.0) / F::max(F::new(0.1), F::new(1.0) - resonance);
let denominator = (F::new(1.0) - f2) * (F::new(1.0) - f2) + (f2 / (dämpfung * dämpfung));

let mut response = F::new(0.0);

// Filter-Modi basierend auf den Ausgängen der analogen Statik-Struktur
match mode {
    OberheimMode::LowPass => {
        response = F::new(1.0) / F::sqrt(denominator);
    }
    OberheimMode::HighPass => {
        response = f2 / F::sqrt(denominator);
    }
    OberheimMode::BandPass => {
        response = f / F::sqrt(denominator);
    }
    OberheimMode::Notch => {
        response = F::abs(F::new(1.0) - f2) / F::sqrt(denominator);
    }
}

response
}
/// Hilfsfunktion zur schnellen Berechnung des ADSR-Amplitudenwerts pro Sample-Index
#[cube]fn calculate_adsr_envelope<F: Float>(t: F, adsr: AdsrConfig) -> F {
let attack = F::new(adsr.attack);
let decay = F::new(adsr.decay);
let sustain = F::new(adsr.sustain);
if t < attack {
    t / attack
} else if t < (attack + decay) {
    let decay_passed = t - attack;
    F::new(1.0) - (decay_passed / decay) * (F::new(1.0) - sustain)
} else {
    sustain
}
}
// --- DER HAUPT-SYNTHESIZER KERNEL ---

#[cube(launch)]pub fn cubek_hybrid_exact_synth_kernel<F: Float>(
output_audio: &mut Array, // Zielbuffer im Zeitbereich
frequency: F, // Grundfrequenz der MIDI-Note
filter_env: AdsrConfig, // Hüllkurve für die Filter-Frequenzen
amp_env: AdsrConfig, // Hüllkurve für die Lautstärke
config: HybridSynthConfig,
#[comptime] fft_size: u32,
) {
let n = ABSOLUTE_POS_X;

if n < fft_size {
    let sample_rate = F::new(config.sample_rate);
    let t = F::cast_from(n) / sample_rate;

    // 1. ADSR HÜLLKURVEN IM REGISTER BERECHNEN
    let f_env = calculate_adsr_envelope(t, filter_env);
    let a_env = calculate_adsr_envelope(t, amp_env);

    // Dynamische Cutoffs via Hüllkurve modulieren
    let dyn_moog_cutoff = F::new(config.moog_cutoff) * (F::new(1.0) + f_env * F::new(3.0));
    let dyn_oberheim_cutoff = F::new(config.oberheim_cutoff) * (F::new(1.0) + f_env * F::new(3.0));

    let mut final_sample = F::new(0.0);
    let pi = F::new(std::f32::consts::PI);
    let num_bins = fft_size / 2 + 1;

    // 2. SPEKTRALE GENERIERUNG & FILTERBANK (PIPELINE-FUSION)
    for k in 0..num_bins {
        let bin_freq = (F::cast_from(k) * sample_rate) / F::cast_from(fft_size);

        let mut real_spec = F::new(0.0);
        let mut imag_spec = F::new(0.0);

        // HYBRIDE FM-GENERIERUNG IM FREQUENZBEREICH
        if k > 0 && bin_freq < sample_rate / F::new(2.0) {
            // Mathematische Repräsentation der harmonischen Spektren einer FM-Synthese
            // Trägerfrequenz (Carrier) moduliert durch Modulator
            let carrier_freq = frequency;
            let mod_freq = frequency * F::new(config.fm_ratio);
            
            // Wir erzeugen Energie an den Seitenbändern (Fc +/- n*Fm)
            // Bessel-Funktionen-Approximation im Frequenzbereich für schnelles Berechnen:
            let distance_to_carrier = F::abs(bin_freq - carrier_freq);
            let harmonic_step = distance_to_carrier / mod_freq;
            
            let fract = harmonic_step - F::floor(harmonic_step);
            if fract < F::new(0.04) || fract > F::new(0.96) {
                // Die Amplitude sinkt mit dem Abstand, skaliert durch den Modulationsindex
                let order = F::floor(harmonic_step);
                let sideband_amplitude = F::new(config.fm_mod_index) / (F::new(1.0) + order * order);
                
                // Alternierende Phasenkomponenten für lebendigen FM-Charakter
                if k % 2 == 0 {
                    real_spec = sideband_amplitude;
                } else {
                    imag_spec = sideband_amplitude;
                }
            }
        }

        // FILTERBANK: Beide Filter werden exakt parallel ausgewertet
        let moog_gain = apply_moog_ladder_filter(bin_freq, dyn_moog_cutoff, F::new(config.moog_resonance));
        let oberheim_gain = apply_oberheim_multimode(bin_freq, dyn_oberheim_cutoff, F::new(config.oberheim_resonance), config.oberheim_mode);

        // Parallel-Routing der Filterbank (50/50 Mischung von Moog und Oberheim)
        let combined_gain = (moog_gain + oberheim_gain) * F::new(0.5);

        // Filterung auf das Spektrum anwenden
        let filtered_real = real_spec * combined_gain;
        let filtered_imag = imag_spec * combined_gain;

        // 3. INVERSE DFT DIREKT IM GEICHEN SCHRITT
        let angle = (F::new(2.0) * pi * F::cast_from(k) * F::cast_from(n)) / F::cast_from(fft_size);
        final_sample += filtered_real * F::cos(angle) - filtered_imag * F::sin(angle);
    }

    // Amplituden-Hüllkurve (Gain-VCA) am Ausgang anwenden und normalisieren
    output_audio[n] = (final_sample / F::cast_from(fft_size)) * a_env;
}
}