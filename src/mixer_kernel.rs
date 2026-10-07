#![allow(warnings)]
use cubecl::prelude::*;

// ============================================================
// STUFE 2: MIXER-KERNEL — PURE EQUAL-POWER-SUMME
// ============================================================
// EIN Launch pro Block, NACH den Voice-Launches. Macht GENAU das,
// was der alte funktionierende CPU-Mix nach dem Readback gemacht
// hat (nur jetzt auf der GPU):
//   mixed[n] = (Σ_v voices_out[v][n]) / sqrt(N)
//
// KEIN Reverb, KEIN Filter, KEIN Width, KEIN Clip hier — das ganze
// DSP (inkl. stateless Reverb, Width, Wet/Dry, Limiter) steckt
// wieder in der Stimme selbst (voice_synth.rs, 1:1 vom bewährten
// Legacy-Kernel). Damit ist der Signalpfad bit-identisch zum
// funktionierenden Zustand vor dem Refactoring:
//   Kernel(Stimme) → Equal-Power-Summe → Readback → Worklet.
//
// Warum das wichtig ist: Jedes zusätzliche Block-übergreifende
// Glied (FIR-Historie, Comb-Feedback) hat an Blockgrenzen
// geknackst, weil seine Koeffizienten sich pro Block ändern
// (LFO-Phase, Cutoff-Anker). Der Legacy-Weg ist stateless pro
// Block und kann prinzipbedingt nicht an Blockgrenzen klicken.

#[cube(launch)]
pub fn cubek_mixer_kernel<F: Float + CubeElement>(
    voices_out: &Array<F>,
    mixed_out: &mut Array<F>,
    num_voices: u32,
    #[comptime] fft_size: u32,
) {
    let n = ABSOLUTE_POS_X;

    if n < fft_size {
        let nv = F::cast_from(num_voices);
        let voice_scale = F::new(1.0) / F::sqrt(F::max(F::new(1.0), nv));

        let mut sum_l = F::new(0.0);
        let mut sum_r = F::new(0.0);

        for v in 0..num_voices {
            let base = (v * fft_size * 2) as usize;
            let idx_l = base + (n * 2) as usize;
            let idx_r = idx_l + 1;
            sum_l += voices_out[idx_l];
            sum_r += voices_out[idx_r];
        }

        let idx_l: usize = (n * 2) as usize;
        let idx_r: usize = (n * 2 + 1) as usize;
        mixed_out[idx_l] = sum_l * voice_scale;
        mixed_out[idx_r] = sum_r * voice_scale;
    }
}
