# Technical Report — CubeCL WebGPU 6-OP FM/Saw Synthesizer

**Project:** `synth` v0.1.0 · **Branch:** `feature/FM-SAW-JAC_-light` · **Date:** 2026-09-27

---

## 1. Executive Summary

This project is a **fully GPU-resident polyphonic-capable subtractive/FM hybrid synthesizer** that runs
entirely in the browser via WebAssembly + WebGPU. All digital signal processing — oscillator spectral
modeling, 6-operator FM routing, Moog-ladder and Oberheim SEM filter emulation, LFO modulation, ADSR
envelope, and procedural stereo convolution reverb — is executed inside a **single fused CubeCL compute
kernel** written in Rust and compiled to WGSL. The CPU (JavaScript) only handles parameter smoothing,
scheduling, and buffer handoff to the Web Audio API.

The defining architectural decision: **the entire synthesis pipeline operates in the frequency domain**.
Sound is synthesized directly as a magnitude spectrum (saw harmonics + FM sideband slots), filtered by
multiplying with analytic analog filter transfer functions, reverberated by complex multiplication with a
procedurally generated impulse-response spectrum, and only then transformed back to the time domain via a
per-sample inverse DFT. There is no per-sample time-domain oscillator loop at all.

---

## 2. Architecture Overview

```
┌────────────────────────── Browser ──────────────────────────┐
│  index.html (UI: 8 tabs, 6-op grid, VU meters)             │
│      │  parameter smoothing (lerp 0.25)                     │
│      ▼                                                      │
│  pkg/synth.js (wasm-bindgen glue)                          │
│      │  WebAudioEngine::render_block_async()               │
│      ▼                                                      │
│  src/lib.rs  — WASM boundary, GPU buffer mgmt, state Cells  │
│      │  ArrayArg handles, CubeCount/CubeDim launch          │
│      ▼                                                      │
│  src/stereo_synth.rs — THE fused kernel (WGSL via CubeCL)   │
│      │  spectral synth → filters → stereo reverb → IDFT     │
│      ▼                                                      │
│  WebGPU device → read_async → Float32Array → AudioBuffer    │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Component Responsibilities

| Layer | File | Role |
|---|---|---|
| UI + scheduler | `index.html` | Tab-based control surface, look-ahead scheduler (25 ms tick, 100 ms horizon), per-block VU metering (dB-scaled), parameter ramping |
| WASM bridge | `src/lib.rs` | `WebAudioEngine` struct; owns `ComputeClient<WgpuRuntime>`, block state (`last_frequency`, `last_cutoff`, `lfo_phase`, `block_count`), GPU buffer creation, kernel launch, async readback |
| DSP kernel | `src/stereo_synth.rs` | `cubek_true_stereo_synth_reverb` — the entire audio engine as one `#[cube(launch)]` function |
| Legacy/experimental | `src/synth.rs`, `src/reverb.rs`, `src/stereo_synth (Copy N).rs` | Earlier iterations: time-domain ADSR/Moog/Oberheim implementations, mono procedural IR, FM-saw experiments. Not compiled into the active path |

---

## 3. The DSP Engine (`src/stereo_synth.rs`)

### 3.1 Execution Model

- **FFT size:** 2048 samples per block (~46.4 ms @ 44.1 kHz, ~42.7 ms @ 48 kHz; the engine
  currently runs `fft_size = 512` for lower latency — see §4)
- **Grid:** `(fft_size + 255) / 256` workgroups × 256 threads (CubeDim 256×1×1); one thread per output sample `n`
- **Per-thread work:** each thread computes one stereo frame by integrating `num_bins = 1025` frequency bins of the whole pipeline (an O(N²/2) direct IDFT — see §7 for discussion)

### 3.2 Envelope & Modulation (per sample `n`)

The master amplitude is a **note-event-driven, absolute-time ADSR** evaluated per sample on the global
sample timeline (`global_sample = global_block_index·2048 + n`), fully continuous across block
boundaries:

```
global_sample = global_block_index·fft_size + n
t_since_on    = (global_sample − note_on_sample) / sample_rate
t_since_off   = (global_sample − note_off_sample) / sample_rate

Gate open (note held):
  t < attack:            amp = t / attack                       (linear attack)
  attack ≤ t < A+D:      amp = sustain + (1−sustain)·e^(−(t−A)/D)  (exponential decay)
  t ≥ A+D:               amp = sustain                          (sustain plateau)

Gate closed (note released):
  amp = amp_at_note_off · e^(−t_since_off / release)             (exponential release)
  where amp_at_note_off is reconstructed from the A/D/S curve at the note-off instant,
  so the release joins the envelope seamlessly (no level jump).

base_freq:    old_freq + progress·(freq − old_freq)      ← block-boundary continuity
base_cutoff:  old_cut + progress·(cutoff − old_cutoff)
lfo_mod:      sin(lfo_phase + 2π·lfo_freq·n/sample_rate) ← phase accumulated on host across blocks
modulated_cutoff = clamp(base_cutoff + lfo_mod·lfo_depth, 50 Hz, ∞)
```

`sample_rate` is a runtime kernel argument negotiated from the `AudioContext` (§4) — no
hard-coded 44100 anywhere in the time base.

Cross-block state (`old_frequency`, `old_cutoff`, `lfo_phase`, `global_block_index`, plus the note-event
fields `gate_on`, `note_on_sample`, `note_off_sample`) is threaded through the kernel as runtime
arguments — **per voice** in the polyphonic engine (§4.1) — eliminating the per-block discontinuity
("Block-Eiern") of earlier versions: frequency/cutoff ramps, LFO phase, and the ADSR envelope all
continue seamlessly across block boundaries. The gate flag is passed as `u32` (1/0) since cubecl
runtime arguments do not include `bool`.

### 3.3 Spectral Synthesis: Sawtooth + FM Sidebands

For each bin `k` (frequency `bin_freq = k·sample_rate/fft_size`, e.g. ≈ 21.5 Hz width @ 44.1 kHz/2048):

1. **Nearest-harmonic windowing** (performance-critical optimization): instead of iterating all 32
   sawtooth harmonics, the kernel computes `h_target = floor(bin_freq / base_freq)` and scans only
   `h ∈ [max(1, h_target−1), min(32, h_target+2))` — 3–4 candidates.

2. **6-operator FM model** — two algorithms (`algo_select`):
   - **Algorithm 0 (parallel carriers):** OP1/OP2 form a level-weighted carrier blend; OP3–OP6 form a
     level-weighted modulator stack. `modulation_force = l3·max(0.1,r3) + l4·max(0.1,r4)`.
   - **Algorithm 1 (classic stack):** OP1 is the carrier; OP2–OP6 stack into the modulator with
     `modulation_force = 1.2·log1p(Σ lᵢ·rᵢ)` — a logarithmic compression that keeps extreme
     modulation indices stable.
   - **OP6 feedback:** `effective_r6 = r6 + l6²·max(0.1,r6)·sin(s·0.5)` — self-modulation emulating
     DX7 operator feedback.

3. **Sideband placement:** for sideband order `s ∈ 1..8`, energy is placed at
   `carrier ± s·mod_freq` when `|bin_freq − target| < 0.5·bin_width`. Slot amplitude:
   - FM active (`modulation_force > 0.01`): Gaussian damping
     `exp(−s² / (2·force²))` × energy compensation `1/√(1+force)` — this is the
     "light FM/saw-Jacobian" shaping of the current branch.
   - FM idle: sidebands above order 1 are zeroed (pure saw).

4. **Real/imaginary dithering:** even bins → real part, odd bins → imaginary part (a cheap
   pseudo-phase distribution that avoids coherent summation artifacts of earlier versions).

### 3.4 Analog Filter Bank (analytic transfer functions, applied per bin)

- **Moog ladder** (`apply_moog_ladder`): 4-pole magnitude model
  `|H| = (1/(1+f²))⁴` with a triangular resonance bump of width `0.08·cutoff` around the corner.
- **Oberheim SEM** (`apply_oberheim_sem`): 2nd-order section
  `|H| = 1/√((1−f²)² + f²/damping²)` with four modes: LP (0), HP (1), BP (2), notch (3),
  `damping = 1/max(0.1, 1−resonance)`.
- Both are summed and averaged: `gain = (moog + oberheim)/2` — a parallel filter topology.

### 3.5 Procedural Stereo Convolution Reverb

Per bin, a decorrelated stereo IR spectrum is synthesized in-register:

- **Frequency-dependent decay:** `amplitude = exp(−k / max(1, (room_size/freq_factor)·10))`, where
  `freq_factor = 1 + bin_freq·hf_damping·10⁻⁴` — high frequencies decay faster (air absorption).
- **Deterministic PRNG:** four trigonometric hash functions (`sin/cos` of `k·{12.9898, 78.233, 45.164,
  92.741}`, fractional-part extracted) produce L/R real/imag phase noise — no memory, fully
  reproducible, thread-safe.
- **M/S stereo matrix:** mid = (L+R)/4, diff = (L−R)/2; per-channel IR =
  `mid ± stereo_width·diff` — the width control is a true M/S balance, not a pan.
- **Complex convolution:** wet spectrum = dry spectrum × IR spectrum (complex multiply), then
  wet/dry crossfade at spectral level.

### 3.6 Inverse DFT with Global Phase Continuity

```
global_sample_index = global_block_index·2048 + n
angle = 2π·k·global_sample_index / 2048
sample += real·cos(angle) + imag·sin(angle)     (note: +sin — conjugate/synthesis convention)
```

The `global_block_index` (a host-side `Cell<u32>`, wrapping) makes the IDFT basis **absolute** rather
than block-relative — the phase of every bin is continuous across blocks, which was the fix for the
cyclic per-block phase reset audible as pumping/egging. Output is scaled by `2/N` and written
interleaved L/R (`out[2n]`, `out[2n+1]`).

---

## 4. Host Layer (`src/lib.rs`)

- **`WebAudioEngine`**: `#[wasm_bindgen]` struct holding `Option<ComputeClient<WgpuRuntime>>` plus
  `Cell` state fields (cutoff/LFO/block counters) and a `RefCell<[Voice; 8]>` voice table
  (per-voice gate/note-on/note-off/pitch state, §4.1). Interior mutability via `Cell`/`RefCell` is
  required because `render_block_async(&self)` takes `&self` while mutating counters and voices.
- **`init_engine_async`**: `future_to_promise` wrapping `cubecl_wgpu::init_setup_async::<WebGpu>` →
  `init_device` → `ComputeClient::load`. Async is mandatory: WebGPU device acquisition is
  promise-based in the browser.
- **`render_block_async`**: allocates one 4096-float output handle **per active voice** plus two
  shared 6-float operator arrays via `bytemuck::cast_slice` → `cubecl::bytes::Bytes` →
  `client.create`, launches the kernel **once per voice** with 29 arguments (27 runtime +
  `#[comptime] fft_size`) — per-voice note state, shared global parameters — then mixes the
  read-back buffers with equal-power scaling and resolves to a single `Float32Array` after one
  collective `client.read_async`. With no active voices a single silent dummy launch keeps the
  WGSL pipeline warm (gate closed, `t_off == t_on` ⇒ exact zeros).
- **Parameter contract (18 JS args → 29 kernel args):** frequency, cutoff, room_size, wet_mix,
  attack, decay, sustain, release, ratios[6], levels[6], algo_select, moog_res, obe_res, obe_mode,
  lfo_freq, lfo_depth, stereo_width, hf_damping — plus host-injected old_freq, old_cutoff,
  sample_rate, lfo_phase, gate flag, note_on_sample, note_off_sample, block_index, fft_size.

### 4.1 Note-Event API (polyphonic voice table)

The engine maintains a fixed-size voice table (`MAX_VOICES = 8`). Each voice carries its own
gate/note state (`gate_on`, `note_on_sample`, `note_off_sample`, `frequency`, `old_frequency`,
`is_drone`); every active voice is rendered by **exactly one kernel launch per block** with the
same compiled kernel (§3), and the per-voice stereo buffers are mixed on the CPU with
equal-power scaling (`1/√N`) to prevent clipping on chords.

- **`note_on(frequency)`** — voice allocation: (1) a still-held voice with the same pitch is
  retriggered (ADSR restart); (2) a free slot is used; (3) the oldest releasing voice is
  recycled; (4) otherwise the oldest held voice is stolen. `note_on_sample = block_count·fft_size`.
- **`note_off(frequency)`** — closes the gate of the held voice matching that frequency and
  records `note_off_sample`; its release tail continues until `5× release` has elapsed, then the
  slot is freed.
- **`set_drone(on, frequency)`** — a dedicated drone voice that note events never touch (no
  retrigger, no stealing, no note-off); it follows the frequency slider (legacy behavior), while
  keyboard voices keep their own pitches.
- **`is_gate_on()` / `active_voice_count()` / `get_lead_frequency()`** — UI feedback: any gate
  open, occupied voices, and the pitch of the voice the frequency slider controls.

Because the envelope is evaluated against absolute sample positions, a note held across many blocks
produces a single continuous A→D→S curve, and a release started in one block decays smoothly through
all subsequent blocks — per voice. Retriggering restarts the attack from zero at the next block
boundary. The frontend maps computer keys (A W S E D F T G Z H U J K O L) and an on-screen piano to
these calls; the "Drone-Modus" checkbox spawns the drone voice so the legacy continuous-sound
behavior is preserved while chords play polyphonically alongside it.

---

## 5. Frontend (`index.html`)

- **Scheduling (AudioWorklet):** a `cubecl-synth-processor` `AudioWorkletProcessor`
  (`synth_worklet.js`) runs on the real-time audio thread with a FIFO of finished stereo
  blocks. Flow control is **ack-driven**: whenever the FIFO falls below its target
  (~3 kernel blocks ≈ 140 ms), the worklet posts an `ack` and the main thread renders one
  GPU block per request, sequentially (never in parallel — the WASM engine state and GPU
  pipeline are single-threaded). Blocks are transferred zero-copy (`postMessage` transfer
  list). This replaces the former `AudioBufferSourceNode` chain scheduling: no node
  garbage, sample-accurate timing, and gain changes apply instantly inside the audio
  thread instead of at block granularity.
- **Sample rate (no resampling):** the kernel renders **natively at the `AudioContext` sample
  rate** — negotiated via `set_sample_rate(audioContext.sampleRate)` before the first
  (warm-up) render, so WGSL compilation, the ADSR time base, LFO phase, and bin frequencies
  all work on the real rate (e.g. 48 kHz) from the start. The worklet's linear streaming
  resampler remains as a safety net with `ratio = 1.0` (passthrough) in case the rates ever
  diverge.
- **Underrun protection:** if the FIFO runs dry (GPU readback jitter, GC pause), the
  worklet outputs silence and fades back in over 5 ms when data returns — no hard clicks;
  an underrun counter is reported for debugging.
- **Smoothing:** all parameters (including per-op ratios/levels) are lerped toward targets at 0.25
  per block (~46 ms time constant), preventing spectral zipper noise; operator arrays are copied into
  fresh `Float32Array`s per block to avoid data races at the WASM boundary. The smoothed frequency
  drives the **lead voice** only (drone voice, or newest held note without drone); other voices hold
  their pitch, so chords stay in tune while the lead can glide.
- **Metering:** per-block peak → dB → −45..0 dB window → CSS-width VU bars with gradient.
- **UI:** 8 tabs (Master/VCA with A/D/S/R sliders, 6 operator tabs, Filter & Reverb), on-screen
  piano + computer-keyboard note input, drone-mode toggle, DX7-inspired defaults
  (ratios 1.0/1.0/3.5/2.0/7.0/0.5, levels 1.0/0.7/1.0/0.8/1.5/0.4).

---

## 6. Build & Dependency Stack

| Dependency | Version | Purpose |
|---|---|---|
| `cubecl` / `cubecl-wgpu` / `cubecl-runtime` / `cubecl-common` | 0.10.0 | GPU compute abstraction; Rust→WGSL kernel compilation, WebGPU backend |
| `cubek` | 0.2.0 | HPC kernel library (fft, convolution, reduce, random, matmul, attention, …) |
| `wasm-bindgen` / `js-sys` / `wasm-bindgen-futures` | 0.2.95 / 0.3.72 / 0.4.45 | JS↔Rust bridge, promise interop |
| `bytemuck` | 1.16 | Zero-copy `f32` ↔ byte casting for GPU upload |
| `pollster` | 0.3 | Blocking executor for native (non-wasm) dev runs |
| `getrandom` (wasm32) | 0.4.3 (`wasm_js`) | Entropy source on wasm |

- Crate type: `cdylib` + `rlib`; edition 2021.
- Build: `wasm-pack build --target web` → `pkg/` (committed: `synth_bg.wasm`, `synth.js`, `.d.ts`).
- Git history shows the evolution: ADSR/LFO sliders → stereo refactor → FM generation → 6-op GUI →
  hybrid saw/FM → heavy/light FM-saw-Jacobian variants (current branch = light).

---

## 7. Known Limitations & Technical Debt

1. **O(N²) IDFT:** each of 2048 threads integrates 1025 bins directly — ~2.1M complex MACs per block
   per channel-pair. A radix-2 FFT would reduce this to O(N log N), but would require cross-thread
   communication (shared memory / multiple kernel launches), which the current single-kernel,
   zero-shared-state design deliberately avoids.
2. **Spectral resolution vs. pitch:** bin width is 21.5 Hz; low fundamentals get quantized harmonic
   placement (mitigated by the ±0.5-bin slot window and the 3-harmonic scan window).
3. **Voice stealing is a hard cut:** when all 8 slots are held, the oldest voice is stolen without a
   fade-out, which can click. A short pre-render ramp would fix this (out of scope).
4. **Filter resonance models are magnitude-only:** no phase response, so self-oscillation and true
   ladder nonlinearity are not modeled.
5. **Repo hygiene:** `src/` contains 8+ "Copy N" variant files and legacy modules (`synth.rs`,
   `reverb.rs`) that are dead code on this branch; `#![allow(warnings)]` suppresses diagnostics.
6. **GPU rendering on the main thread:** the WASM/CubeCL engine renders on the main thread
   (WebGPU + wasm-bindgen), so a blocked main thread can starve the worklet FIFO (mitigated by
   the ~3-block FIFO and underrun fade-out; a dedicated worker + its own GPU device would
   decouple it fully).

> **Resolved in this revision:** the ADSR is no longer block-local, the engine is no longer
> monophonic, and the sample rate is no longer hard-coded. The envelope is a note-event-driven,
> absolute-time ADSR (§3.2) driven by a polyphonic voice table (§4.1): per-voice
> gate/note-on/note-off state, one kernel launch per voice per block, CPU equal-power mixing,
> voice stealing with release-tail recycling, and a dedicated drone voice. Scheduling runs on an
> AudioWorklet with ack-driven flow control (§5), and the kernel renders natively at the
> negotiated `AudioContext` sample rate — no resampling. Sustain/release controls and the
> piano/keyboard UI carry over; the legacy drone behavior is preserved via the "Drone-Modus"
> toggle.

---

## 8. Possible Future Directions

- Radix-2 FFT-based spectral engine (cubek `fft` feature is already a dependency) with cross-block
  overlap-add for perfect phase reconstruction.
- Move the WASM/WebGPU engine into a dedicated `Worker` with its own GPU device so the main
  thread can never starve the worklet FIFO (the ack protocol already isolates the audio path).
- Audio-rate parameter automation via `AudioParam` on the worklet node.
- Batched multi-voice kernel: fold the per-voice launches into a single launch with a voice-index
  dimension (the current design deliberately keeps one launch per voice, §4.1).
- Per-voice filter/reverb parameters and per-voice operator tables (currently shared globally).
- Time-domain Moog ladder (the implementation already exists in `synth.rs` legacy code) as a
  post-IDFT pass for authentic resonance behavior.
- Preset system: the 6×(ratio, level) + algorithm topology maps naturally onto DX7 SysEx-style
  patch storage.

---

*Report generated from source analysis of branch `feature/FM-SAW-JAC_-light` (HEAD: bf711b4).*
