#![allow(warnings)]
use cubecl::prelude::*;

// ============================================================
// STUFE 2 (Refactoring_plan.md): MIXER-KERNEL
// ============================================================
// EIN Launch pro Block, NACH den Voice-Launches. Läuft EINMAL für
// alle Stimmen (statt Reverb/Filter/Width pro Stimme):
//   1. Summe über alle Stimmen (Equal-Power-Skalierung 1/sqrt(N))
//   2. Reverb als WAHRE FALTUNG mit Block-Historie: Die FIR summiert
//      über den aktuellen Block UND den vorherigen (dry_prev). Der
//      Nachhall fließt damit kontinuierlich über Blockgrenzen —
//      kein Zurücksetzen am Blockanfang (das war der Bug: das
//      block-lokale FIR brach bei jedem Block auf 1 Tap zusammen,
//      hörbar als abgehackter Klang mit ~94-Hz-Wiederholung).
//   3. Stereo-Width (M/S) in der IR, Wet/Dry-Mix, Hard-Clip [-1, 1].
//
// Buffer-Layout der Voice-Ausgaben: planar hintereinander,
// voices_dry[v * fft_size * 2 + n * 2] = L, [+1] = R (L=R=trocken).

// ------------------------------------------------------------
// STUFE 1.5: STIMMEN-SUMME — EINMAL pro Block.
// Schreibt die Equal-Power-Mischung aller Stimmen in einen eigenen
// Buffer, damit das Reverb die SUMME faltet und nicht versehentlich
// nur die Stimme 0 (Aliasing-Falle).
// Layout: dry[n * 2] = L, [+1] = R.
// ------------------------------------------------------------
#[cube(launch)]
pub fn cubek_voice_sum<F: Float + CubeElement>(
    voices_dry: &Array<F>,
    dry_mixed: &mut Array<F>,
    num_voices: u32,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let nv = F::cast_from(num_voices);
        let voice_scale = F::new(1.0) / F::sqrt(F::max(F::new(1.0), nv));

        let mut dry_l = F::new(0.0);
        let mut dry_r = F::new(0.0);

        for v in 0..num_voices {
            let base = (v * fft_size * 2) as usize;
            let idx_l = base + (n * 2) as usize;
            let idx_r = idx_l + 1;
            dry_l += voices_dry[idx_l];
            dry_r += voices_dry[idx_r];
        }

        let idx_l: usize = (n * 2) as usize;
        let idx_r: usize = (n * 2 + 1) as usize;
        dry_mixed[idx_l] = dry_l * voice_scale;
        dry_mixed[idx_r] = dry_r * voice_scale;
    }
}

// ------------------------------------------------------------
// STUFE 2a: IR-PRECOMPUTE — EINMAL pro Block (fft_size Taps).
// Berechnet die 4 IR-Komponenten (L/R × real/imag) pro Tap m mit
// allen Transzendentalfunktionen EINMAL; die Faltung im Mixer ist
// danach reines Multiplizieren/Addieren.
//   - PRNG über die SAMPLE-Position m (statt Seitenband-Ordnung)
//   - Abfall exp(-m / (decay_s * sample_rate)) — decay_s in
//     Sekunden, physikalisch korrekt über die Blocklänge hinweg
//   - Gain 0.0375 = 0.15/sqrt(16): Energie-normalisiert, damit
//     512 Taps dieselbe Wet-Lautstärke ergeben wie die alten
//     32 Seitenband-Taps (kein Clipping-Explodieren)
// Layout: ir[m * 4 + 0] = L real, [+1] = L imag,
//         [+2] = R real, [+3] = R imag.
// ------------------------------------------------------------
#[cube(launch)]
pub fn cubek_ir_precompute<F: Float + CubeElement>(
    ir_coeffs: &mut Array<F>,
    sample_rate: F,
    lfo_accumulated_phase: F,
    lfo_frequency: F,
    lfo_depth: F,
    dyn_cutoff: F,
    old_cutoff: F,
    room_size_seconds: F,
    high_freq_damping: F,
    stereo_width: F,
    #[comptime] fft_size: u32,
) {
    let m = ABSOLUTE_POS_X;

    if m < fft_size {
        let pi = F::new(std::f32::consts::PI);
        let samples_per_block = F::cast_from(fft_size);
        let m_f = F::cast_from(m);

        // Deterministische PRNG-Taps (identische Formeln wie im Legacy-Kernel,
        // aber über die SAMPLE-Position m parametrisiert).
        let rand_l_real = (F::sin(m_f * F::new(12.9898)) - F::floor(F::sin(m_f * F::new(12.9898)))) * F::new(2.0) - F::new(1.0);
        let rand_l_imag = (F::cos(m_f * F::new(78.2330)) - F::floor(F::cos(m_f * F::new(78.2330)))) * F::new(2.0) - F::new(1.0);
        let rand_r_real = (F::sin(m_f * F::new(45.1640)) - F::floor(F::sin(m_f * F::new(45.1640)))) * F::new(2.0) - F::new(1.0);
        let rand_r_imag = (F::cos(m_f * F::new(92.7410)) - F::floor(F::cos(m_f * F::new(92.7410)))) * F::new(2.0) - F::new(1.0);

        // LFO-modulierter Cutoff (Proxy für den Spektralschwerpunkt) —
        // Höhen-Dämpfung verkürzt die effektive Decay-Zeit.
        let block_progress = m_f / samples_per_block;
        let sample_phase_delta =
            (F::new(2.0) * pi * lfo_frequency * block_progress * samples_per_block) / sample_rate;
        let lfo_mod = F::sin(lfo_accumulated_phase + sample_phase_delta);
        let base_cutoff = old_cutoff + (block_progress * (dyn_cutoff - old_cutoff));
        let mut modulated_cutoff = base_cutoff + (lfo_mod * lfo_depth);
        if modulated_cutoff < F::new(50.0) {
            modulated_cutoff = F::new(50.0);
        }

        let freq_factor = F::new(1.0) + (modulated_cutoff * high_freq_damping * F::new(0.0001));
        let effective_decay_s = room_size_seconds / freq_factor;

        // Physikalisch korrekter Abfall über die Sample-Position
        // (Decay in Sekunden → Samples). Über einen 10-ms-Block ist
        // der Abfall für Raumgrößen 0.5–5 s flach — wie im Original.
        let amplitude_decay =
            F::exp(-m_f / F::max(F::new(1.0), effective_decay_s * sample_rate)) * F::new(0.0375);

        let ir_l_real = rand_l_real * amplitude_decay;
        let ir_l_imag = rand_l_imag * amplitude_decay;
        let ir_r_real = rand_r_real * amplitude_decay;
        let ir_r_imag = rand_r_imag * amplitude_decay;

        // STEREO-WIDTH (M/S) auf den IR-Taps — einmalig pro Block.
        let mid_real = (ir_l_real + ir_r_real) * F::new(0.5);
        let mid_imag = (ir_l_imag + ir_r_imag) * F::new(0.5);
        let final_l_real = mid_real + stereo_width * (ir_l_real - mid_real);
        let final_l_imag = mid_imag + stereo_width * (ir_l_imag - mid_imag);
        let final_r_real = mid_real + stereo_width * (ir_r_real - mid_real);
        let final_r_imag = mid_imag + stereo_width * (ir_r_imag - mid_imag);

        let base: usize = (m * 4) as usize;
        ir_coeffs[base] = final_l_real;
        ir_coeffs[base + 1] = final_l_imag;
        ir_coeffs[base + 2] = final_r_real;
        ir_coeffs[base + 3] = final_r_imag;
    }
}

// ------------------------------------------------------------
// STUFE 2b: MIXER — EINMAL pro Block, NACH den Voice-Launches.
// WAHRE FALTUNG MIT BLOCK-HISTORIE (das ist der Kontinuitäts-Fix):
//   wet[n] = Σ_{m=0..n}   ir[m] * dry_cur[n-m]      (aktueller Block)
//          + Σ_{m=n+1..L} ir[m] * dry_prev[L+n-m]    (Vorheriger Block)
// Am Blockanfang (n=0) fließt die volle IR über den HISTORIE-Buffer —
// der Nachhall bricht NICHT zusammen. dry_prev ist ein persistenter
// GPU-Buffer (Host tauscht ihn nach jedem Block gegen dry_cur aus).
// ------------------------------------------------------------
#[cube(launch)]
pub fn cubek_mixer_kernel<F: Float + CubeElement>(
    dry_cur: &Array<F>,
    dry_prev: &Array<F>,
    ir_coeffs: &Array<F>,
    output_stereo_audio: &mut Array<F>,
    wet_dry_mix: F,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let idx: usize = (n * 2) as usize;

        // --- 1. TROCKENE SUMME (von Stufe 1.5 vorberechnet) ---
        let dry_l = dry_cur[idx];
        let dry_r = dry_cur[idx + 1];

        // --- 2. FALTUNG: aktueller Block + Historie (kontinuierlich) ---
        let mut wet_l = F::new(0.0);
        let mut wet_r = F::new(0.0);

        // 2a) Taps, die in den aktuellen Block reichen (m <= n)
        for m in 0..=n {
            let ir_base = (m * 4) as usize;
            let ir_l = ir_coeffs[ir_base] - ir_coeffs[ir_base + 1];
            let ir_r = ir_coeffs[ir_base + 2] - ir_coeffs[ir_base + 3];

            let src = ((n - m) * 2) as usize;
            wet_l += dry_cur[src] * ir_l;
            wet_r += dry_cur[src + 1] * ir_r;
        }

        // 2b) Taps, die in den VORHERIGEN Block reichen (m > n) —
        //     Index L+n-m liegt immer in [n+1, L-1], also im Buffer.
        for m in n + 1..fft_size {
            let ir_base = (m * 4) as usize;
            let ir_l = ir_coeffs[ir_base] - ir_coeffs[ir_base + 1];
            let ir_r = ir_coeffs[ir_base + 2] - ir_coeffs[ir_base + 3];

            let src = ((fft_size + n - m) * 2) as usize;
            wet_l += dry_prev[src] * ir_l;
            wet_r += dry_prev[src + 1] * ir_r;
        }

        // --- 3. WET/DRY-MIX ---
        let res_l = (F::new(1.0) - wet_dry_mix) * dry_l + wet_dry_mix * wet_l;
        let res_r = (F::new(1.0) - wet_dry_mix) * dry_r + wet_dry_mix * wet_r;

        // --- 4. HARD-CLIP [-1, 1] ---
        output_stereo_audio[idx] = F::max(-F::new(1.0), F::min(F::new(1.0), res_l));
        output_stereo_audio[idx + 1] = F::max(-F::new(1.0), F::min(F::new(1.0), res_r));
    }
}
