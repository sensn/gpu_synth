# Refactoring Plan — Modularisierung: Voice-Kernel + Mixer-Kernel

**Ziel:** Die Effekt-Verarbeitung (Filter + Reverb + Stereo-Width + Wet/Dry) aus dem
per-Stimme gestarteten Kernel herauslösen und in **einen gemeinsamen Mixer-Kernel**
für alle Stimmen gleichzeitig verlagern. Die Stimmen-Generierung (FM-Synthese + ADSR)
bleibt pro Stimme — aber deutlich schlanker.

**Motivation (aus Technical_Report.md §5/§6):**
- Reverb-IR-Berechnung (4× sin/cos+floor PRNG + exp pro Seitenband·Sample) ist
  **stimmen-invariant** — aktuell wird sie pro Stimme wiederholt: **8× redundant**.
- Filter (Brickwall/Resonanz) ist ebenfalls global (nur cutoff-abhängig), nicht
  stimmen-spezifisch.
- ~80% der Transzendental-Operationen entfallen auf Reverb+Filter → Potenzial:
  **~8-fache Reduktion** dieses Anteils bei 8 Stimmen.

---

## 1. Architektur: Zwei-Stufen-Pipeline

```
STUFE 1: VOICE-KERNEL (pro Stimme, 1 Launch)          STUFE 2: MIXER-KERNEL (1 Launch für ALLE)
─────────────────────────────────────────             ─────────────────────────────────────────
Eingabe:  Voice-Note-State (freq, gate,               Eingabe:  voice_dry[v][n] (N×fft_size,
          note_on/off, ADSR-Params),                            interleaved L/R oder planar)
          op_ratios/levels, algo_select               Effekte:  1. Summe aller Stimmen
Ausgabe:  voice_dry — UNGEFILTETE,                      2. Filter (Moog/Oberheim/Brickwall
          TROCKENE Mono/Stereo-Synthese                        auf dem SUMMENSIGNAL,
          (FM-Seitenbänder × ADSR)                             LFO-moduliert)
                                                     Effekte:  3. Reverb (IR einmal berechnen!)
                                                               4. Stereo-Width, Wet/Dry-Mix
                                                     Ausgabe:  final_stereo (fft_size×2)
```

**Warum das schneller ist:**
- Reverb-IR + Filter werden **einmal pro Sample** statt **einmal pro Sample×Stimme**
  berechnet → 8× weniger Transzendental-Op im dominanten Kostenpfad.
- Voice-Kernel wird massiv schlanker: nur noch FM-Seitenband-Loop + ADSR + Phase.
- Der Mixer-Kernel ist **ein** Launch — die „ein Kernel-Launch pro Stimme"-Regel gilt
  weiterhin für die Generierung; das Mischen/Effekte ist per Definition eine
  Sammel-Operation (das war vorher der CPU-Mix, nur jetzt auf der GPU).

**Was bewusst NICHT geändert wird:**
- Kein FFT/IDFT-Umbau, keine Änderung am FM-Algorithmus, keine Änderung an ADSR,
  LFO, Phasenrekonstruktion oder Voice-Verwaltung (lib.rs-Logik bleibt).
- Das CPU-Mixing entfällt (wird Teil des Mixer-Kernels) — der Readback schrumpft auf
  EINEN finalen Stereo-Buffer statt N Voice-Buffer.

---

## 2. Neue Kernel-Signaturen

### 2.1 `src/voice_synth.rs` — Stimmen-Generierung (aus stereo_synth.rs extrahiert)

```rust
#[cube(launch)]
pub fn cubek_voice_synth<F: Float + CubeElement>(
    voice_dry: &mut Array<F>,      // fft_size×2 (L,R getrennt oder Mono+Width später)
    frequency: F, old_frequency: F,
    op_ratios: &Array<F>, op_levels: &Array<F>,
    algo_select: u32,
    attack: F, decay: F, sustain: F, release: F,
    gate_is_on: u32, note_on_sample: u32, note_off_sample: u32,
    global_block_index: u32,
    #[comptime] fft_size: u32,
) { /* FM-Seitenbänder + ADSR + Phasenrekonstruktion — OHNE Filter/Reverb */ }
```

- Enthält: ADSR (absolute Zeit), FM-Struktur (algo 0/1), Seitenband-Loop mit
  Phasenrekonstruktion, Gain-Staging (0.05).
- **Entfällt:** LFO, Cutoff, Filter, Reverb-IR, Stereo-Width, Wet/Dry — alles Stufe 2.
- Ausgabe: trockenes, ungefiltertes Signal pro Stimme (L=R identisch oder Mono —
  Stereo entsteht erst im Reverb/Width).

### 2.2 `src/mixer_kernel.rs` — Effekte + Mischen (NEU)

```rust
#[cube(launch)]
pub fn cubek_voice_mixer<F: Float + CubeElement>(
    voices_dry: &Array<F>,        // N × fft_size×2 — alle Stimmen hintereinander
    output_stereo: &mut Array<F>, // fft_size×2 final
    num_voices: u32,
    // Filter (auf dem Summensignal)
    dyn_cutoff: F, old_cutoff: F, lfo_accumulated_phase: F,
    lfo_frequency: F, lfo_depth: F,
    moog_resonance: F, oberheim_resonance: F, oberheim_mode: u32,
    // Reverb (einmal für alle Stimmen!)
    room_size_seconds: F, high_freq_damping: F, wet_dry_mix: F, stereo_width: F,
    sample_rate: F,
    global_block_index: u32,
    #[comptime] fft_size: u32, #[comptime] max_voices: u32,
) { /* 1. Summieren, 2. Filtern, 3. Reverb, 4. Width/Mix */ }
```

**Grid-Design des Mixer-Kernels:** 1 Thread pro Output-Sample `n` (wie bisher).
Jeder Thread summiert `num_voices` Trocken-Samples (linearer Speicherzugriff,
koaleszierbar), wendet Filter + Reverb **einmal** an.

### 2.3 lib.rs — Orchestrierung

```rust
// Stufe 1: N Launches (Voice-Generierung)
for v in active { cubek_voice_synth::launch(...); }
// Stufe 2: 1 Launch (Mischen + Effekte)
cubek_voice_mixer::launch(...);
// Readback: NUR final_stereo (ein Buffer!)
```

- `last_cutoff`, `lfo_phase` wandern in die Mixer-Argumente (unverändert logisch).
- CPU-Mix entfällt → JS erhält direkt den finalen Stereo-Block.
- Idle-Fall: Voice-Kernel mit Dummy-Stimme (wie gehabt) oder Mixer mit
  `num_voices=0` + Bypass — Empfehlung: Dummy-Stimme beibehalten (Pipeline warm,
  identisches Verhalten).

---

## 3. Filter-Design im Mixer-Kernel (Lösung für toten Resonanz-Code)

Das aktuelle Brickwall (`gain = freq ≤ cutoff`) ersetzt den nie aufgerufenen
Moog/Oberheim-Code. Im Mixer-Kernel wird das Filtern **im Zeitbereich pro Sample**
angewendet — aber Achtung: Moog/Oberheim sind Spektral-Formeln (Magnitude über
`bin_freq`). Zwei Optionen:

**Option A (empfohlen, minimal-invasiv):** Filter bleibt spektral, aber im
Mixer-Kernel: Der Mixer summiert die Stimmen, führt eine Mini-IDFT-freie
„Spektral-Filterung" durch, indem die (bereits zeitreihen-) trockenen Stimmen
**nicht** gefiltert werden — stattdessen filtert der Voice-Kernel weiterhin
spektral pro Seitenband (wie heute), NUR der Reverb wandert in den Mixer.
→ Reverb ist der 80%-Kostenblock; Filter bleibt billig (Vergleich pro Seitenband).

**Option B (langfristig, sauberer):** Voice-Kernel gibt Spektrum statt Zeitserie,
Mixer macht IDFT + Filter + Reverb. Größerer Umbau, berührt Phasenrekonstruktion.

**Empfehlung: Option A als Schritt 1** (maximaler Gewinn, minimales Risiko),
Option B als späterer Schritt. Details unten.

---

## 4. Umsetzungsschritte

### Schritt 1: Voice-Kernel extrahieren (src/voice_synth.rs)
1. `stereo_synth.rs` kopieren → `voice_synth.rs`; Filter/Reverb/Width/Mix-Code
   entfernen (Seitenband-Loop endet bei `filtered_synth_real`).
   - Seitenband-Loop: `res = sideband_amp` (statt Reverb-Zweig), Akkumulation
     `final_sample_l += res·cos(angle)` etc.
   - Stereo: L=R=trockenes Signal (Width kommt im Mixer).
2. lib.rs: Launch-Schleife auf `cubek_voice_synth` umstellen; Ausgabe-Buffer
   `voices_dry` (N×fft_size×2, planar pro Stimme hintereinander).
3. **Zwischentest**: Sound muss identisch klingen (nur „trockener", da Reverb fehlt).

### Schritt 2: Mixer-Kernel (src/mixer_kernel.rs)
1. Neuer Kernel: Summe über Stimmen → Reverb-IR (PRNG+exp, einmal!) → Stereo-Width
   → Wet/Dry → Hard-Clip.
2. **Wichtig — Reverb-IR-Phase:** Die IR-Koeffizienten hängen von `order` (Seitenband)
   ab, nicht von `n`. Im Mixer (Zeitbereich) gibt es kein `order` mehr! Lösung:
   Die IR-Formel parametrisiert über die **Sample-Position im Block** statt über
   `order` — d.h. `rand(order)` → `rand(n)`-Variante: `amplitude_decay =
   exp(−n/decay)`-artige Block-IR (pro Sample ein Koeffizient, wie ein
   FIR-Block-Filter). Das ist ein Design-Shift: aus „IR pro Seitenband" wird
   „IR pro Sample" — klanglich ein anderer (aber einfacherer, deterministischer)
   Reverb. Alternativ (klangtreuer): Reverb bleibt im Voice-Kernel, nur Filter
   wandert — aber das spart nur die Filter-Kosten (~20%).
   → **Entscheidung nötig:** Klangtreue vs. Rechenersparnis. Empfehlung: IR pro
   Sample (Block-FIR), da die Seitenband-IR ohnehin ein Hack war und der
   Klangcharakter durch `room_size`/`damping`/`width` erhalten bleibt.
3. lib.rs: Mixer-Launch nach den Voice-Launches; Readback nur final_stereo.
4. **Test**: Drone + Akkorde; Reverb-Charakter vergleichen (room_size-Slider).

### Schritt 3: Filter in den Mixer (optional, nach Schritt 2 stabil)
1. Zeitbereich-Filter: 1-Pol-TPT-SVFilter (cheap, self-oscillating bei res>1) oder
   Biquad (RBJ) mit LFO-moduliertem Cutoff — pro Sample, auf dem Summensignal.
   - Moog/Oberheim-Spektralformeln werden durch echte Zeitbereich-Filter ersetzt
     (löst die tote Resonanz-Slider ein!).
2. **Test**: Cutoff-Sweep, Resonanz-Slider, LFO auf Cutoff (jetzt hörbar, da Fix).

### Schritt 4: Aufräumen
1. `stereo_synth.rs` löschen (nach Migration) oder als Referenz behalten.
2. Legacy-Dateien („Copy N", `synth.rs`, `reverb.rs`) aus src/ entfernen (git rm).
3. `#![allow(warnings)]` entfernen, Warnungen fixen.
4. TECHNICAL_REPORT.md + Technical_Report.md aktualisieren.

---

## 5. Risiken & Abwägungen

| Risiko | Bewertung/Minderung |
|---|---|
| Klangänderung beim Reverb-Umbau (Seitenband-IR → Sample-IR) | Reverb war ein Hack; room_size/damping/width bleiben wirksam. A/B-Test mit Drone. |
| Phasenrekonstruktion im Voice-Kernel unberührt | Kein Risiko — Code wandert unverändert. |
| Mixer-Kernel: Summation über N Stimmen pro Thread | N×fft_size×2 Reads pro Thread — linear, koaleszierbar, N≤8: trivial. |
| Buffer-Layout: N Stimmen hintereinander (planar) | `ArrayArg::from_raw_parts(handle, N·fft_size·2)` — ein Handle, ein Launch. |
| „1 Launch pro Stimme"-Anforderung | Weiterhin erfüllt für die Generierung; der Mixer ist die Sammel-Instanz (war vorher CPU). |
| Idle-Fall | Dummy-Stimme wie gehabt — Mixer läuft mit num_voices=1 (Dummy) weiter. |
| LFO-Phase/Cutoff-Anchor | Wandern unverändert in die Mixer-Argumente — Logik identisch. |

## 6. Erwarteter Gewinn (8 Stimmen, fft=512)

| Metrik | Vorher | Nachher |
|---|---|---|
| Transzendental-Op/Block | ~655k (Reverb 8×) | ~82k (Reverb 1×) + Mix-Adds |
| Kernel-Launches | 8 | 8 + 1 (Mixer) |
| GPU-Readback | 8 × 4 KB | 1 × 4 KB |
| CPU-Mix-Aufwand | 8×1024 Adds | 0 (GPU) |
| Latenz | unverändert | unverändert (gleiche Pipeline-Tiefe) |

## 7. Offene Entscheidungen

1. **Reverb-IR-Parametrisierung** (Schritt 2): Sample-basierte Block-IR (empfohlen)
   vs. klangtreues Beibehalten der Seitenband-IR im Voice-Kernel.
2. **Filter-Stufe** (Schritt 3): TPT-SVF (empfohlen, self-osc) vs. Biquad vs.
   Spektral-Filter im Voice-Kernel belassen.
3. **Stereo im Voice-Kernel**: Mono-Ausgabe (L=R) + Width im Mixer (empfohlen,
   spart die Hälfte der Voice-Buffer-Schreibarbeit) vs. echtes Stereo pro Stimme.
