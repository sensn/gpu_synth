// ============================================================================
// synth_worklet.js — AudioWorklet-Processor für den CubeCL-Synth
// ============================================================================
// Referenz-Architektur: spessasynth WorkletSynthesizerCore
// (doc/worklet_synthesizer_core.ts): Der Processor läuft im Echtzeit-Audio-Thread
// und hält fertige Audio-Blöcke bereit ("process() füllt nur outputs"). Bei uns
// erzeugt die GPU (WASM/CubeCL) die Blöcke auf dem Main-Thread — der Worklet
// streamt sie mit Flow-Control (ack) in den Audio-Graph.
//
// Datenfluss:
//   Main-Thread: render_block_async() → Float32Array (interleaved L/R)
//                → de-interleave → postMessage({type:"block"}, [transfer])
//   Worklet:     FIFO aus {l, r} Paaren → process() füllt outputs[0]
//                → postMessage({type:"ack", count}) wenn FIFO unter Ziel
//
// FLOW-CONTROL (Fix gegen den Ack-Sturm):
//   Der Worklet zählt offene Aufträge (inFlight = angefordert, aber noch
//   nicht geliefert). requestBlocks() fordert NUR
//       TARGET_QUEUE - fifo - inFlight
//   an. Vorher wurde jede Render-Quantum (2,7 ms) erneut TARGET - fifo
//   angefordert, während die GPU noch renderte → Ack-Sturm → die Pumpe auf
//   dem Main-Thread flutete den FIFO, die Latenz explodierte und das
//   Timing raste. Kommt ein Render nie an (GPU-Fehler), sendet der Host
//   {type:"fail", count} — der Worklet korrigiert inFlight, damit die
//   Flow-Control nicht für immer stehen bleibt.
//
// ANTI-CLICK-MASSNAHMEN:
//   - Gain-Glättung: One-Pole ~5 ms pro Sample statt hartem Sprung
//     (Zipper-Noise bei jedem Slider-Move vorher hörbar als Click).
//   - Soft-Clip (C1-stetig) NACH dem Gain: der Kernel limitiert nur die
//     Einzelstimme auf ±1, die Mixer-Summe × Gain (bis 15x) konnte die
//     DAC-Grenze hart clippen → Crackle auf Transienten.
//   - 5 ms Einblendrampe nach Start/Underrun (kein harter Knacks).
//   - FIFO-Ziel 6 Blöcke (~64 ms @ fft_size=512/48 kHz): Puffer gegen
//     Main-Thread-Stalls (GC, GPU-Readback-Jitter). 3 Blöcke (~32 ms)
//     waren zu dünn — jeder Stall > 32 ms endete in einem Underrun-Click.
//   - SAMPLERATE: Der Kernel rendert nativ auf der Context-Rate — kein
//     Resampling. Der lineare Streaming-Resampler bleibt als Sicherheitsnetz
//     mit ratio = 1.0 aktiv.
// ============================================================================

const WORKLET_NAME = "cubecl-synth-processor";

// FIFO-Ziel: ~6 Kernel-Blöcke. Bei fft_size=512 ≈ 10.7 ms pro Block @ 48 kHz
// → ~64 ms Puffer. 1 Block wird konsumiert + Reserve für GPU-Readback-Jitter
// und GC-Pausen auf dem Main-Thread.
const TARGET_QUEUE = 6;

// Metering alle 8 Render-Quanten (8 × 128 Samples ≈ 21 ms @ 48 kHz) —
// entspricht der alten 25 ms UI-Kadenz, statt ~340 Nachrichten/s.
const METER_EVERY = 8;

// Soft-Clip-Schwelle: darunter komplett linear (transparent), darüber
// C1-stetige Sättigung gegen ±1 — kein hartes Digital-Clipping mehr.
const CLIP_T = 0.9;

class SynthWorkletProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = (options && options.processorOptions) || {};
        // Kernel-Rate = Context-Rate (kein Resampling); Fallback nur falls
        // processorOptions fehlen sollten.
        this.sampleRateIn = opts.sampleRateIn || sampleRate;
        // FIFO: {l: Float32Array, r: Float32Array}
        this.queue = [];
        this.currentBlock = null;
        // Offene Aufträge: bereits per ack angefordert, aber noch nicht
        // geliefert. Das ist der Kern des Ack-Sturm-Fixes.
        this.inFlight = 0;
        // Resampler-Position im Quellblock (in Quell-Samples, fraktional).
        // Der Fraktionalanteil wird über Blockgrenzen mitgenommen (Carry),
        // damit das Resampling sample-kontinuierlich bleibt.
        this.pos = 0;
        // Einblendrampe nach Underrun/Start (5 ms)
        this.fade = 0.0;
        this.fadeStep = 1.0 / (0.005 * sampleRate);
        this.wasUnderrun = false;
        this.underruns = 0;
        // Gain: Ziel (vom Host) und geglätteter Ist-Wert (wirkt per Sample).
        this.gain = 1.0;
        this.gainSmooth = 1.0;
        // One-Pole-Glättung ~5 ms: selbst ein 15x-Sprung wird in ~25 ms
        // abgefahren — hörbar glatt, aber frei von Zipper-Clicks.
        this.gainCoef = 1.0 / (0.005 * sampleRate);
        // Metering
        this.peakL = 0;
        this.peakR = 0;
        this.quantumCount = 0;
        this.alive = true;

        this.port.onmessage = (e) => {
            const m = e.data;
            if (m.type === "block") {
                this.queue.push({ l: m.l, r: m.r });
                if (this.inFlight > 0) this.inFlight--;
            } else if (m.type === "gain") {
                this.gain = m.value;
            } else if (m.type === "fail") {
                // Render scheiterte: dieser Block kommt nie. In-Flight
                // korrigieren, sonst wartet die Flow-Control dauerhaft.
                this.inFlight = Math.max(0, this.inFlight - (m.count || 1));
            } else if (m.type === "prime") {
                // Deterministischer Start: erste Blöcke anfordern
                this.requestBlocks();
            } else if (m.type === "stop") {
                this.alive = false;
            }
        };
    }

    requestBlocks() {
        // NUR die wirklich fehlenden Blöcke anfordern: Ziel minus FIFO minus
        // offener Aufträge. Solange die GPU in Verzug ist, wird hier nichts
        // Neues angefordert — genau das verhindert den Ack-Sturm. Der
        // Main-Thread rendert weiterhin sequenziell (nie parallel — der
        // WASM-Engine-Zustand ist nicht nebenläufigkeitssicher).
        const missing = TARGET_QUEUE - this.queue.length - this.inFlight;
        if (missing > 0) {
            this.inFlight += missing;
            this.port.postMessage({ type: "ack", count: missing, underruns: this.underruns });
        }
    }

    // C1-stetiger Soft-Clip: linear bis CLIP_T, danach asymptotisch gegen ±1.
    // Steigung an der Schwelle = 1 → die Kurve selbst kann nicht klicken.
    softClip(x) {
        const ax = x < 0 ? -x : x;
        if (ax <= CLIP_T) return x;
        const s = x < 0 ? -1 : 1;
        const over = ax - CLIP_T;
        const head = 1.0 - CLIP_T;
        return s * (CLIP_T + head * (over / (over + head)));
    }

    process(inputs, outputs) {
        if (!this.alive) return false;
        const out = outputs[0];
        const L = out[0];
        const R = out[1] || out[0];
        const n = L.length; // Render-Quantum: 128 Samples
        const ratio = this.sampleRateIn / sampleRate; // Quell-Steps pro Ziel-Sample

        for (let i = 0; i < n; i++) {
            // Nächsten Block aktivieren, wenn keiner mehr läuft
            if (this.currentBlock === null) {
                const next = this.queue.shift();
                if (next !== undefined) {
                    this.currentBlock = next;
                    if (this.wasUnderrun) this.pos = 0; // Carry nur nach Gap verwerfen
                }
            }
            // FIFO leer → Underrun: Stille ausgeben, Zähler nur 1x pro Gap
            if (this.currentBlock === null) {
                if (!this.wasUnderrun) {
                    this.underruns++;
                    this.wasUnderrun = true;
                    this.fade = 0; // Einblendrampe für die Wiederkehr
                }
                L[i] = 0;
                R[i] = 0;
                continue;
            }
            this.wasUnderrun = false;

            const blk = this.currentBlock;
            const len = blk.l.length;
            const p = this.pos;
            const i0 = p | 0;
            const frac = p - i0;
            const i1 = i0 + 1 < len ? i0 + 1 : i0;
            let sL = blk.l[i0] + (blk.l[i1] - blk.l[i0]) * frac;
            let sR = blk.r[i0] + (blk.r[i1] - blk.r[i0]) * frac;

            // Gain pro Sample glätten (One-Pole) — kein Sprung, kein Click.
            this.gainSmooth += (this.gain - this.gainSmooth) * this.gainCoef;
            sL = this.softClip(sL * this.gainSmooth);
            sR = this.softClip(sR * this.gainSmooth);

            // 5 ms Einblendrampe (nach Start und nach jedem Underrun)
            if (this.fade < 1.0) {
                sL *= this.fade;
                sR *= this.fade;
                this.fade += this.fadeStep;
            }

            L[i] = sL;
            R[i] = sR;

            const aL = sL < 0 ? -sL : sL, aR = sR < 0 ? -sR : sR;
            if (aL > this.peakL) this.peakL = aL;
            if (aR > this.peakR) this.peakR = aR;

            // Position fortschreiben; Block-Ende → Fraktional-Carry behalten
            this.pos += ratio;
            if (this.pos >= len) {
                this.pos -= len;
                this.currentBlock = null;
            }
        }

        // Metering (gedrosselt) + Nachfrage für verbrauchte Blöcke
        this.quantumCount++;
        if (this.quantumCount % METER_EVERY === 0) {
            this.port.postMessage({ type: "meter", peakL: this.peakL, peakR: this.peakR });
            this.peakL = 0;
            this.peakR = 0;
        }
        this.requestBlocks();
        return true;
    }
}

registerProcessor(WORKLET_NAME, SynthWorkletProcessor);
