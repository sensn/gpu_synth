# Technical Report — CubeCL WebGPU Synthesizer

**Stand:** Nach Polyphonie-, AudioWorklet- und Samplerate-Revision; Kernel-Analyse
inkl. LFO-Fix. `fft_size = 512`, `MAX_VOICES = 8`, native Context-Samplerate.

---

## 1. Systemübersicht

```
┌────────────────────────── Main-Thread (JS/WASM) ──────────────────────────┐
│  index.html (UI, Slider, Klaviatur)                                        │
│      │ note_on/note_off/set_drone · set_sample_rate · Slider-Targets       │
│      ▼                                                                     │
│  lib.rs: WebAudioEngine ── Voice-Tabelle [8], Lead-Logik, LFO-Phase,        │
│      │                     Block-Zähler, Cutoff-Anchor                    │
│      │ render_block_async(): 1 KERNEL-LAUNCH PRO AKTIVER STIMME           │
│      ▼                                                                     │
│  GPU: cubek_true_stereo_synth_reverb (stereo_synth.rs) × N Voices         │
│      │                                                                     │
│      ▼ read_async (alle Voice-Buffer in einem Sammel-Read)                │
│  CPU-Mix: out[i] = Σ_v voice_v[i] · 1/√N  (Equal-Power, interleaved L/R)  │
│      │ postMessage(transfer) → FIFO                                        │
└──────┼──────────────────────────────────────────────────────────────────────┘
       ▼
┌────────────── Audio-Thread (AudioWorklet: synth_worklet.js) ──────────────┐
│  FIFO (Ziel: 3 Blöcke) → Streaming-Resampler (ratio=1, Sicherheitsnetz)   │
│  → Gain im Audio-Thread → outputs[0] → destination                        │
│  ack-Protokoll: FIFO < Ziel → "ack {count}" → Main-Thread rendert nach     │
└────────────────────────────────────────────────────────────────────────────┘
```

## 2. Host-Schicht (lib.rs)

### 2.1 Engine-Zustand

| Feld | Typ | Rolle |
|---|---|---|
| `client` | `Option<ComputeClient<WgpuRuntime>>` | CubeCL-Client (WebGPU) |
| `fft_size` | `u32` | **512** (User-Setting, Latenz ~11.6 ms/Block @ 44.1 kHz) |
| `sample_rate` | `Cell<f32>` | vom AudioContext verhandelt (`set_sample_rate`), kein Resampling |
| `last_cutoff` | `Cell<f32>` | Cutoff-Anchor für Block-Glättung |
| `lfo_phase` | `Cell<f32>` | LFO-Phasen-Akkumulator (Host, blockübergreifend) |
| `block_count` | `Cell<u32>` | Globaler Block-Zähler (absolute Zeitachse) |
| `voices` | `RefCell<[Voice; 8]>` | Voice-Tabelle: `active, gate_on, note_frequency, frequency, old_frequency, note_on_sample, note_off_sample, is_drone` |

### 2.2 Voice-Verwaltung (Polyphonie)

- **`note_on(freq)`**: Retrigger gleicher Ton → freier Slot → älteste Release-Stimme
  (Recycling) → Voice-Stealing (ältestes Note-On, Drone ausgenommen).
- **`note_off(freq)`**: schließt das Gate der passenden gehaltenen Stimme; Release-Schwanz
  läuft bis `5 × release`, dann Slot-Freigabe (Host prüft nach jedem Launch).
- **`set_drone(on, freq)`**: eigene, unantastbare Drone-Stimme (kein Retrigger/Stealing/
  Note-Off durch Tastatur-Events).
- **Lead-Stimme**: Rang 1 Drone (folgt dem Frequenz-Slider), Rang 2 neueste gehaltene
  Tastatur-Stimme (Mono-Portamento). Alle anderen Stimmen halten ihre Tonhöhe.
- **Idle-Fall**: keine aktive Stimme → 1 stiller Dummy-Launch (Gate zu, `t_off == t_on`
  ⇒ `amp_at_off = 0` ⇒ exakte Nullen; hält WGSL-Pipeline warm).

### 2.3 render_block_async — Ablauf

1. `ratios`/`levels` (je 6 f32) einmalig uploaden (allen Stimmen gemeinsam).
2. Snapshot aktiver Stimmen; Lead-Bestimmung.
3. **Pro Stimme**: Output-Handle (4096→1024 f32 bei fft=512) allokieren, Kernel mit
   stimmen-spezifischen Skalaren (Frequenz, Note-Events) + gemeinsamen Parametern
   (Cutoff, LFO, Filter, Reverb, ADSR) starten.
4. Ein Sammel-`read_async` über alle Voice-Handles.
5. CPU-Mix (Equal-Power `1/√N`), Ausgabe als interleaved `Float32Array`.
6. Host-Update: `old_frequency`-Anker fortschreiben, Release-Schwänze freigeben,
   `lfo_phase += 2π·f·T_block`, `block_count += 1`.

## 3. Kernel: cubek_true_stereo_synth_reverb (stereo_synth.rs)

**Grid:** `(fft_size+255)/256` Blöcke à 256 Threads; jeder Thread = 1 Sample `n`.
**Paradigma:** Direkte spektrale Seitenband-Synthese — kein FFT/IDFT-Lauf, stattdessen
wird jedes FM-Seitenband als analytisches Sinus-Paar direkt in den Zeitbereich
geschrieben („Pseudo-IDFT" mit exakter Phasenrekonstruktion).

### 3.1 Signalfluss im Kernel (pro Sample n, pro Stimme)

```
ADSR (absolute Zeitachse) ──► master_amp
                                   │
LFO: sin(lfo_phase_host + 2π·f·n/sr) ──► modulated_cutoff ──┐
Cutoff-Glättung: old_cut → cut (per-Sample-Lerp) ────────────┤
                                                            ▼
FM-Struktur (algo_select):                                  │
  op1..op6: f_i = base_freq · ratio_i                       │
  Algo 0: carrier = Σ(l·f)/Σl (op1+2), mod = op3..6-Mix     │
  Algo 1: carrier = f1, mod = f2, FM-Index aus op2..6-Stack │
                                                            ▼
Seitenband-Schleife (order 0..15):                          │
  amp = base_amp/(1+order²) · (1+0.2·mod_force)             │
  real_freq = carrier ± order·mod_freq                     │
  ──► BRICKWALL-FILTER: gain = (real_freq <= modulated_cutoff)  ◄─ LFO WIRKT HIER
  ──► Reverb-IR (pro Seitenband!): 4× (sin/cos+floor) PRNG   │
       amplitude_decay = exp(−order/max(1, decay·2))·0.15   │
       L/R = mid + width·(ch − mid)  (Stereo-Width)          │
  ──► Wet/Dry-Mix                                            │
  ──► Phasenrekonstruktion: angle = 2π·(k_hist·t_hist +     │
       k_local·n)/N — nahtlos über Blockgrenzen             │
                                                            ▼
out = Σ Seitenbänder · 0.05 · master_amp, Hard-Clip [-1,1]
```

### 3.2 ADSR — Absolute-Zeit-Hüllkurve (§ Fix aus Vorrevision)

```
global_sample = block_index·fft_size + n
t_since_on  = max(0, (global_sample − note_on_sample)/sr)
t_since_off = max(0, (global_sample − note_off_sample)/sr)

Gate offen:  t<A: amp = t/A · linear
             A≤t<A+D: amp = S + (1−S)·e^(−(t−A)/D)
             t≥A+D:  amp = S
Gate zu:     amp = amp_at_off · e^(−t_since_off/R)
             amp_at_off aus A/D/S-Kurve rekonstruiert (nahtlos)
```

### 3.3 LFO — Funktionsweise (nach Fix)

Host akkumuliert die Phase blockübergreifend: `lfo_phase += 2π·lfo_freq·fft_size/sr`
(pro Block, mod 2π). Im Kernel: `lfo_mod = sin(lfo_phase + 2π·lfo_freq·n/sr)` —
die lokale Phase läuft innerhalb des Blocks weiter, der Host-Anteil sorgt für
Stetigkeit über Blockgrenzen. `modulated_cutoff = clamp(cutoff_lerp + lfo_mod·depth, 50, ∞)`.

**Bug (behoben in dieser Revision):** Der Kernel berechnete `modulated_cutoff` korrekt,
nutzte für das Seitenband-Filter aber `current_modulated_cutoff` — eine zweite Variable
ohne LFO-Term (Relikt einer Merge-Konfliktauflösung). Der LFO-Slider hatte keinerlei
hörbare Wirkung. Fix: Filter nutzt jetzt `modulated_cutoff`; die toten Variablen
(`base_freq`, `num_bins`, doppelter `block_progress`) sind entfernt.

### 3.4 Reverb-Modell (pro Seitenband, pro Sample!)

Für jedes Seitenband `order` werden 4 Pseudo-Zufallszahlen aus deterministischen
Transzendentalen erzeugt (`sin/cos` + `floor`), mit `exp(−order/decay)` gewichtet und
über `stereo_width` zu L/R-IR-Koeffizienten gemischt. Wet/Dry per Skalar-Mix.
**Kosten: 16 Seitenbänder × 2 Signs × (4 sin/cos + 1 exp) ≈ 160 Transzendental-Op
pro Sample pro Stimme** — identisch in jeder Stimme wiederholt (Hauptmotiv des
Refactorings, siehe Refactoring_plan.md).

### 3.5 Phasenrekonstruktion über Blockgrenzen

Jedes Seitenband berechnet seine exakte historische Frequenz des Vorblocks
(`old_carrier ± order·old_mod`) und setzt die Phase als
`angle = 2π·(k_hist·block_start + k_local·n)/N` zusammen — Frequenz-Glides klingen
ohne Block-Knacks („kein Block-Eiern").

## 4. Frontend & Audio-Pfad (index.html, synth_worklet.js)

- **Startup-Reihenfolge**: `init_engine_async` → AudioContext (native Rate) →
  `set_sample_rate` → Warm-Up-Render (WGSL-Kompilierung) → Worklet-Modul →
  `prime` → ack-getriebene Render-Pumpe.
- **Ack-Protokoll**: Worklet fordert Blöcke nur bei FIFO-Unterlauf des Ziels an;
  Main-Thread rendert sequenziell (`pendingRenders`-Queue, `renderInFlight`-Guard).
- **Gain** wirkt im Audio-Thread (sofort, ohne Block-Verzögerung); Metering gedrosselt
  (alle 8 Quanten ≈ 23 ms).
- **Underrun-Schutz**: Stille + 5 ms Fade-in; Underrun-Zähler im ack.
- **Samplerate**: Kernel rendert nativ auf Context-Rate; Worklet-Resampler ist
  ratio=1-Passthrough-Sicherheitsnetz.

## 5. Bekannte Einschränkungen

1. **Filter ist Brickwall**: `gain = (freq ≤ cutoff)` — die Moog/Oberheim-Resonanz-
   Funktionen existieren im Kernel, werden aber nie aufgerufen (toter Code). Keine
   Flanken, keine Resonanz-Spitze, kein Selbstoszillieren.
2. **Reverb pro Stimme redundant**: IR-Berechnung (PRNG + exp) ist für alle Stimmen
   identisch, wird aber pro Stimme pro Sample ausgeführt — 8-facher Aufwand.
3. **CPU-Mix**: Summierung der Voice-Buffer auf der CPU (WASM) statt auf der GPU.
4. **GPU auf Main-Thread**: blockierter Main-Thread kann FIFO aushungern (FIFO +
   Underrun-Fade mildern).
5. **Repo-Hygiene**: „Copy N"-Varianten und Legacy-Module (`synth.rs`, `reverb.rs`)
   sind toter Code; `#![allow(warnings)]` unterdrückt Diagnosen.

## 6. Performance-Profil (Schätzungen, fft=512, 8 Stimmen)

| Pfad | Kosten pro Block |
|---|---|
| Kernel-Launches | 8 (einer pro Stimme) |
| Transzendental-Op | ~160/Sample/Stimme → ~655k/Block gesamt |
| Reverb-IR-Anteil davon | ~80% (PRNG + exp, pro Stimme identisch!) |
| CPU-Mix | 8 × 1024 f32 Adds + Skalierung |
| Readback | 1 Sammel-`read_async` (8 × 4 KB) |

**Schlussfolgerung:** Der Reverb-/Filter-Block dominiert die Rechenzeit und ist
stimmen-invariant — der ideale Kandidat für das geplante Mixer-Kernel-Refactoring
(→ Refactoring_plan.md): Stimmen-Generierung und gemeinsame Effekte trennen, um den
8-fachen Reverb-Aufwand auf 1-fach zu reduzieren.
