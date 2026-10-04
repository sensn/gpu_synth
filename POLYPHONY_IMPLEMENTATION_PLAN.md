# Polyphony Implementation Plan — One Kernel Launch Per Voice

**Goal:** Extend the monophonic note-event layer (TECHNICAL_REPORT.md §3.2/§4.1) with a
per-voice state table so that multiple simultaneous notes sound together. Each active voice
is rendered by **exactly one kernel launch** of the existing
`cubek_true_stereo_synth_reverb` kernel; the host (Rust/WASM) mixes the per-voice stereo
buffers into one output block that is handed to WebAudio.

**Design decision (per task description):** *No new GPU kernel and no kernel signature
change.* The kernel is already fully voice-parametrized — `frequency`, `old_frequency`,
`gate_is_on`, `note_on_sample`, `note_off_sample`, `global_block_index` are plain scalar
arguments. Polyphony is achieved purely on the host side by launching the same kernel once
per voice with that voice's state, then summing the results. This keeps WGSL compilation
cached (one kernel variant), keeps GPU work deterministic, and makes voice count a
runtime property instead of a compile-time one.

---

## 1. Current State (verified against code)

### 1.1 `src/lib.rs` — `WebAudioEngine` (monophonic)

| Field | Type | Role |
|---|---|---|
| `client` | `Option<Arc<WgpuClient>>` | CubeCL compute client |
| `fft_size` | `u32` | Block size (2048) |
| `last_frequency`, `last_cutoff` | `Cell<f32>` | Per-block smoothing anchors |
| `lfo_phase` | `Cell<f32>` | Global LFO phase accumulator |
| `block_count` | `Cell<u32>` | Global block counter (absolute IDFT phase + ADSR time base) |
| `gate_on` | `Cell<bool>` | **Single** gate flag |
| `note_on_sample` | `Cell<u32>` | **Single** note-on timestamp (absolute samples) |
| `note_off_sample` | `Cell<u32>` | **Single** note-off timestamp |

`render_block_async` flow (mono):
1. Allocate one output buffer (`fft_size * 2` f32) on GPU.
2. Upload `ratios`/`levels` arrays once.
3. **One** `launch()` with the mono note state.
4. `read_async` → `Float32Array` → JS.

`note_on(freq)` sets `gate_on = true`, `note_on_sample = block_count * fft_size`,
`last_frequency = freq`. `note_off()` sets `gate_on = false`, `note_off_sample = …`.
`is_gate_on()` exposes the gate for the UI.

### 1.2 `src/stereo_synth.rs` — kernel (already voice-ready)

`cubek_true_stereo_synth_reverb` computes, per output sample `n`:
- Absolute-time ADSR from `global_block_index`, `note_on_sample`, `note_off_sample`,
  `gate_is_on` (§3.2 of the report): attack/decay/sustain while gated, release with
  seamless level reconstruction at note-off.
- Spectral saw-FM synthesis for `frequency` (interpolated from `old_frequency`).
- Moog ladder + Oberheim SEM filterbank, procedural stereo reverb, absolute-phase IDFT.

Everything except the note state is shared/global (LFO phase, block index, filters,
reverb IR). **The kernel needs zero changes.**

### 1.3 `index.html` — frontend (mono assumptions)

- `triggerNoteOn(semi, keyEl)` → `audioEngine.note_on(freq)`; also writes the freq slider
  so the smoothing follows.
- `triggerNoteOff(keyEl)` → `audioEngine.note_off()` (no frequency argument — mono).
- Drone mode checkbox keeps the gate open; toggling calls `note_on(freq)`/`note_off()`.
- `heldKeys` Set already tracks every held computer key — ready for polyphony.
- `render_block_async(...)` is called with 18 args; the returned `Float32Array` is
  de-interleaved into a WebAudio buffer.

---

## 2. Architecture

```
JS (index.html)                      Rust/WASM (lib.rs)                     GPU (unchanged kernel)
────────────────                     ───────────────────────                 ──────────────────────
note_on(freq)  ────────────────►  VoiceAllocator: find/steal voice v
                                   v.gate = true
                                   v.note_on_sample = block_count*fft
                                   v.freq = freq
note_off(freq) ────────────────►  VoiceAllocator: voice with freq →
                                   v.gate = false
                                   v.note_off_sample = block_count*fft
                                   (voice stays active until release ends)

render_block_async(...) ──────►  for each voice v in active set:
                                     launch(kernel, v.state, shared params)   ──►  buffer_v (stereo)
                                   mix: out[i] = Σ_v buffer_v[i] * voice_scale
                                   read out → Float32Array
                               ◄──────────────────────────────────────────────
```

### 2.1 Voice table (Rust)

```rust
const MAX_VOICES: usize = 8;

#[wasm_bindgen]
#[derive(Clone, Copy)]
struct Voice {
    active: bool,          // allocated (sounding or releasing)
    gate_on: bool,         // key still held
    frequency: f32,        // current pitch of this voice
    old_frequency: f32,    // per-voice smoothing anchor (block start)
    note_on_sample: u32,   // absolute sample index of last note-on
    note_off_sample: u32,  // absolute sample index of last note-off
    released: bool,        // release tail finished → free slot
}
```

- **Voice allocation:** first free slot; if none free, steal the oldest voice
  (smallest `note_on_sample`) — classic synth behavior, guarantees a new note always
  sounds.
- **Note-off matching:** `note_off(freq)` finds the voice whose `frequency` matches
  (within a small epsilon, since JS `Math.pow` rounding is deterministic it will match
  exactly, but epsilon guards float noise). If no match (e.g. key released after steal),
  no-op.
- **Release tail:** a voice stays in the active set while releasing; it is freed when
  `t_since_off > release_time * TAIL_FACTOR` (e.g. 5× release time, or when
  `master_amp` contribution is negligible). The kernel's exponential release never
  reaches exactly zero, so the host applies a hard cutoff after the tail window.
- **Mono compatibility:** with `MAX_VOICES = 1` the engine behaves exactly as before.

### 2.2 Per-voice kernel launch + mixing (Rust)

In `render_block_async`:

1. Allocate **one output buffer per voice** (or one buffer of
   `MAX_VOICES * fft_size * 2` floats — simpler: one `handle_out` per voice, created
   from the same `Bytes` zero-vector; CubeCL buffers are cheap handles).
2. Upload `ratios`/`levels` **once** (shared across voices).
3. For each active voice `v`: `launch()` with `v.frequency`, `v.old_frequency`,
   `v.gate_on`, `v.note_on_sample`, `v.note_off_sample`, and the shared global params
   (`global_block_index`, LFO phase, filter/reverb params, ADSR times).
4. Read back all voice buffers in **one** `read_async(vec![...])` call.
5. CPU-mix: `out[i] = Σ_v voice_buf_v[i] * (1/√num_active_voices)` — equal-power
   scaling prevents clipping when many voices stack. (Alternative considered: GPU mix
   kernel — rejected for now; CPU mix of ≤ 8 × 4096 floats is trivial and keeps the
   "one kernel launch per voice" invariant clean.)
6. Update per-voice `old_frequency = frequency`, advance `block_count`, free finished
   voices.

**Why one launch per voice (not a batched kernel):** the task explicitly requires it;
it also means zero WGSL recompiles, per-voice state stays in cheap host memory, and
voice count can change every block without touching comptime constants.

### 2.3 Frontend changes (`index.html`)

- `triggerNoteOn`: keep syncing the freq slider (visual), but call
  `note_on(freq)` — unchanged signature, now polyphonic.
- `triggerNoteOff`: call `note_off(freq)` **with the frequency** so the engine releases
  the right voice. `triggerNoteOff` needs the semitone (or freq) — pass it from the
  key element's `dataset.semitone` (mouse) and from `KEY_TO_SEMITONE` (keyboard).
- Drone mode: unchanged semantics (gate held open on its own voice at slider freq).
  When drone is on, keyboard notes still allocate additional voices.
- Gate status UI: show active voice count, e.g. `Voices: 3 | Gate: ON`.
- `heldKeys` already prevents double note-on from key repeat.

### 2.4 API surface (unchanged signatures, new semantics)

| Export | Before | After |
|---|---|---|
| `note_on(freq: f32)` | retrigger single voice | allocate/steal voice, set its state |
| `note_off()` | close single gate | **`note_off(freq: f32)`** — release matching voice |
| `is_gate_on() -> bool` | mono gate | true if **any** voice gated (UI) |
| `render_block_async(...)` | 1 launch, 1 buffer | N launches (N = active voices), mixed output |
| `active_voice_count() -> u32` | — | **new**: for the UI |

`note_off` gains a required `freq` argument — a breaking JS change, handled in the same
commit by updating `index.html`.

---

## 3. Implementation Steps

### Step 1 — `src/lib.rs`: voice table + allocation
- Add `Voice` struct + `voices: [Voice; MAX_VOICES]` (or `RefCell<Vec<Voice>>`) to
  `WebAudioEngine`.
- Rewrite `note_on`:
  - Find voice with matching frequency that is still gated → retrigger it (legato-ish
    refresh of `note_on_sample`).
  - Else find free slot (`!active`).
  - Else steal oldest (`min note_on_sample`).
  - Set `active = true, gate_on = true, frequency = freq, old_frequency = freq,
    note_on_sample = block_count * fft_size`.
- Rewrite `note_off(freq)`: matching gated voice → `gate_on = false,
  note_off_sample = block_count * fft_size`. No match → no-op.
- Add `active_voice_count()`.
- Keep `is_gate_on()` = any voice gated (drone indicator).

### Step 2 — `src/lib.rs`: per-voice launch loop + mixing
- In `render_block_async`, iterate active voices; for each, create an output handle and
  `launch` with per-voice scalars + shared params.
- Collect all handles, single `read_async`, CPU-mix with equal-power scaling, return
  one `Float32Array` (same shape as before: `fft_size * 2` interleaved stereo).
- After read: per-voice `old_frequency = frequency`; free voices whose release tail
  ended (`t_since_off > release * 5`); advance `block_count` once per block (not per
  voice!).
- **Edge case — zero active voices:** still return a silent block (zeros) so the JS
  scheduler never stalls. (Warm-up call in `startEngine` happens before any note_on,
  so this path must work.)
- **Edge case — drone mode:** drone is just a voice that never gets `note_off`; no
  special casing needed in Rust.

### Step 3 — `index.html`: polyphonic note events
- `triggerNoteOn(semi, keyEl)`: unchanged call, add `keyEl.dataset.freq` for off-matching.
- `triggerNoteOff(keyEl)` → `triggerNoteOff(semi, keyEl)`: compute freq, call
  `note_off(freq)`.
- Keyboard handlers pass semitone on keyup.
- Gate status line shows `Voices: N`.
- Drone toggle: on enable, `note_on(sliderFreq)`; on disable, `note_off(sliderFreq)`
  (was parameterless).

### Step 4 — Verification
- `cargo check` (native, fast syntax/type gate) — note `#[cube]` code needs the
  `cubecl` macro expansion; `cargo check` handles it.
- `wasm-pack build --target web --release` (or the project's usual build command) for
  the actual WASM artifact.
- Manual test matrix (browser):
  1. Drone on → sound; keyboard chords (3–5 keys) → each voice audible, no clipping
     (equal-power mix), no scheduler stalls.
  2. Release keys → per-voice release tails overlap correctly.
  3. > 8 simultaneous notes → oldest voice stolen, new note sounds.
  4. Drone off + no notes → silence, scheduler keeps running (zero block).
  5. Mono regression: single key press/release behaves like before.

### Step 5 — Docs
- Update `TECHNICAL_REPORT.md`: §4.1 note-event layer → describe voice table, per-voice
  launch, mixing, voice stealing; remove/replace the "monophonic only" limitation in
  §5/§7/§8 (limitations/future work) — polyphony is now implemented.

---

## 4. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| 8 launches/block instead of 1 → GPU queue pressure | Launches share one compiled kernel; buffers are small (16 KB each); WebGPU handles dozens of small dispatches fine. If needed later: batch voices into one launch via a voice-index dimension (explicitly out of scope). |
| `read_async` of N buffers per block | Single `read_async` call with all handles — one map read, not N round trips. |
| Voice stealing clicks | Stolen voice's release is cut hard. Acceptable v1; future: quick fade via a pre-render ramp (out of scope). |
| `note_off` freq mismatch (float) | Exact match expected (same `Math.pow` path both directions); epsilon compare as guard. |
| Scheduler starvation on zero voices | Always render a zero block (host-side zeros, no launch needed). |
| CPU mix cost | ≤ 8 voices × 4096 floats × 4 B = 128 KB copy+add per block (~11 ms of audio) — negligible. |

## 5. Out of Scope (explicitly)

- Batched multi-voice kernel (single launch for all voices) — the task mandates one
  launch per voice.
- Per-voice filter/reverb parameter sets (all voices share the panel params).
- Voice-specific operator ratios/levels.
- MIDI input.

---

## 6. As-Built Addendum (post-implementation notes)

The feature is implemented and verified (`cargo check` + `wasm-pack build --release`
both pass; `pkg/synth.d.ts` exposes the full polyphonic API). Deviations and
refinements discovered during implementation:

1. **Dedicated drone voice (`is_drone` flag + `set_drone(on, freq)` API).** The plan
   treated drone as "just a voice that never gets note_off". That breaks in practice:
   the drone checkbox defaults to ON at 110 Hz — identical to the "A" key. Pressing A
   would retrigger the drone voice, and releasing it would kill the drone. The drone is
   now its own voice that note events never retrigger, steal, or release. The
   `triggerNoteOff` drone check was removed from the frontend accordingly (keyboard
   voices must always release).
2. **Lead-voice pitch routing.** The smoothed frequency slider drives exactly one
   voice per block: the drone if present (classic behavior), otherwise the newest
   held keyboard voice (mono portamento). All other voices hold their pitch — chords
   stay in tune. `get_lead_frequency()` mirrors this ranking so the frontend can sync
   the slider on note-off/drone-off without yanking the remaining lead to a stale
   pitch.
3. **Idle blocks keep the pipeline warm.** Instead of returning host-side zeros, one
   silent dummy launch runs (gate closed, `t_off == t_on` ⇒ `amp_at_off = 0` ⇒ exact
   zeros). This preserves the original engine's first-block WGSL compilation behavior
   and guarantees the scheduler never stalls.
4. **Voice allocation order:** retrigger same pitch → free slot → oldest *releasing*
   voice → oldest held voice (stealing). Release-tail recycling before stealing avoids
   cutting audible notes when a tail is expendable.
5. **`CubeCount`/`CubeDim` are not `Copy`** — each per-voice launch clones them.
6. **Report updated:** §3.2/§4/§4.1/§5/§7/§8 of TECHNICAL_REPORT.md now describe the
   polyphonic engine; the monophonic limitation entries are resolved.
