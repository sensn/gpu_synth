#![allow(warnings)]
use cubecl::prelude::*;

// ============================================================
// STUFE 2 (Refactoring_plan.md): MIXER-KERNEL
// ============================================================
// EIN Launch pro Block, NACH den Voice-Launches. Läuft EINMAL für
// alle Stimmen (statt Reverb/Filter/Width pro Stimme):
//   1. Summe über alle Stimmen (Equal-Power-Skalierung 1/sqrt(N))
//   2. Reverb als Block-FIR: IR-Koeffizient über SAMPLE-Position
//      parametrisiert (statt über die Seitenband-Ordnung) —
//      deterministisch, pro Sample ein Koeffizient. Faltung des
//      trockenen Blocks mit der IR (Länge = fft_size).
//   3. Stereo-Width (M/S), Wet/Dry-Mix, Hard-Clip [-1, 1].
//
// Buffer-Layout der Voice-Ausgaben: planar hintereinander,
// voices_dry[v * fft_size * 2 + n * 2] = L, [+1] = R (L=R=trocken).

// ------------------------------------------------------------
// STUFE 2a: IR-PRECOMPUTE — EINMAL pro Block (fft_size Taps).
// Berechnet die 4 IR-Komponenten (L/R × real/imag) pro Tap m,
// inkl. Amplituten-Abfall und Stereo-Width (M/S). Danach ist
// die Faltung im Mixer reine Multiplizieren+Addieren (keine
// Transzendentalfunktionen mehr in der O(N²)-Schleife).
// Layout: ir_coeffs[m * 4 + 0] = L real, [+1] = L imag,
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
        // aber über die SAMPLE-Position m parametrisiert statt über die
        // Seitenband-Ordnung — Refactoring_plan.md, Schritt 2).
        let rand_l_real = (F::sin(m_f * F::new(12.9898)) - F::floor(F::sin(m_f * F::new(12.9898)))) * F::new(2.0) - F::new(1.0);
        let rand_l_imag = (F::cos(m_f * F::new(78.2330)) - F::floor(F::cos(m_f * F::new(78.2330)))) * F::new(2.0) - F::new(1.0);
        let rand_r_real = (F::sin(m_f * F::new(45.1640)) - F::floor(F::sin(m_f * F::new(45.1640)))) * F::new(2.0) - F::new(1.0);
        let rand_r_imag = (F::cos(m_f * F::new(92.7410)) - F::floor(F::cos(m_f * F::new(92.7410)))) * F::new(2.0) - F::new(1.0);

        // Frequenz-abhängige Dämpfung: hohe Frequenzen klingen schneller ab.
        // (Im Zeitbereich gibt es kein real_freq pro Sample — wir nutzen den
        //  LFO-modulierten Cutoff als Proxy für den Spektralschwerpunkt.)
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
        let effective_decay = room_size_seconds / freq_factor;
        let amplitude_decay = F::exp(-m_f / F::max(F::new(1.0), effective_decay * F::new(2.0))) * F::new(0.15);

        let ir_l_real = rand_l_real * amplitude_decay;
        let ir_l_imag = rand_l_imag * amplitude_decay;
        let ir_r_real = rand_r_real * amplitude_decay;
        let ir_r_imag = rand_r_imag * amplitude_decay;

        // STEREO-WIDTH (M/S) auf den IR-Taps — einmalig, nicht pro Faltungs-Schritt.
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
// STUFE 1.5: STIMMEN-SUMME — EINMAL pro Block.
// Schreibt die Equal-Power-Mischung aller Stimmen in einen eigenen
// Buffer. Nötig, damit die Faltung im Mixer die SUMME faltet und
// nicht versehentlich nur die Stimme 0 (Aliasing-Falle).
// Layout: dry_mixed[n * 2] = L, [+1] = R.
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
// STUFE 2b: MIXER — EINMAL pro Block, NACH den Voice-Launches.
// Summiert alle Stimmen (Equal-Power), faltet den Summen-Block
// mit der vorberechneten IR (Block-FIR, kausal innerhalb des
// Blocks), mischt Wet/Dry und clippt hart.
// ------------------------------------------------------------

#[cube(launch)]
pub fn cubek_mixer_kernel<F: Float + CubeElement>(
    dry_mixed: &Array<F>,
    ir_coeffs: &Array<F>,
    output_stereo_audio: &mut Array<F>,
    wet_dry_mix: F,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        // --- 1. TROCKENE SUMME (von Stufe 1.5 vorberechnet) ---
        let idx: usize = (n * 2) as usize;
        let dry_l = dry_mixed[idx];
        let dry_r = dry_mixed[idx + 1];

        // --- 2. REVERB: BLOCK-FIR mit VORBERECHNETER IR (Stufe 2a) ---
        // wet[n] = Σ_m dry[n - m] * ir[m], kausal innerhalb des Blocks.
        // Keine Transzendentalfunktionen hier — nur Multiplizieren/Addieren.
        let mut wet_l = F::new(0.0);
        let mut wet_r = F::new(0.0);

        for m in 0..=n {
            let ir_base = (m * 4) as usize;
            let ir_l_real = ir_coeffs[ir_base];
            let ir_l_imag = ir_coeffs[ir_base + 1];
            let ir_r_real = ir_coeffs[ir_base + 2];
            let ir_r_imag = ir_coeffs[ir_base + 3];

            // Faltungs-Tap: wet[n] += dry[n - m] * ir[m]
            let src = ((n - m) * 2) as usize;
            let dry_m_l = dry_mixed[src];
            let dry_m_r = dry_mixed[src + 1];

            wet_l += dry_m_l * ir_l_real - dry_m_l * ir_l_imag;
            wet_r += dry_m_r * ir_r_real - dry_m_r * ir_r_imag;
        }

        // --- 4. WET/DRY-MIX ---
        let res_l = (F::new(1.0) - wet_dry_mix) * dry_l + wet_dry_mix * wet_l;
        let res_r = (F::new(1.0) - wet_dry_mix) * dry_r + wet_dry_mix * wet_r;

        // --- 5. HARD-CLIP [-1, 1] ---
        let idx_l: usize = (n * 2) as usize;
        let idx_r: usize = (n * 2 + 1) as usize;

        output_stereo_audio[idx_l] = F::max(-F::new(1.0), F::min(F::new(1.0), res_l));
        output_stereo_audio[idx_r] = F::max(-F::new(1.0), F::min(F::new(1.0), res_r));
    }
}
