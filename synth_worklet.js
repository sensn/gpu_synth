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
// Eigenschaften:
//   - Kein AudioBufferSourceNode-Chain-Scheduling mehr: kein Node-Garbage,
//     sample-genaues Timing, Gain-Änderungen wirken sofort (im Audio-Thread).
//   - SAMPLERATE: Der Kernel rendert nativ auf der Context-Rate — kein
//     Resampling. Der lineare Streaming-Resampler bleibt als Sicherheitsnetz
//     mit ratio = 1.0 aktiv (falls sampleRateIn je von der Context-Rate
//     abweichen sollte, bleibt der Code korrekt).
//   - Underrun-Schutz: bei leerem FIFO Stille + 5 ms Einblendrampe beim
//     Wiederkommen (kein harter Knacks), Underrun-Zähler fürs Debugging.
//   - Ack-Protokoll (bedarfsgesteuerte GPU-Pumpe): der Worklet fordert genau
//     dann neue Blöcke an, wenn sein FIFO unter das Ziel fällt — die GPU
//     rendert nur bei Bedarf, der FIFO läuft nie über.
//   - Start: der Main-Thread sendet "prime" (deterministisch, statt auf den
//     Konstruktor-Ack zu vertrauen), der Worklet fordert darauf die ersten
//     Blöcke an.
// ============================================================================

const WORKLET_NAME = "cubecl-synth-processor";

// FIFO-Ziel: ~3 Kernel-Blöcke. Bei fft_size=512 ≈ 11 ms pro Block @ 44.1 kHz
// (≈ 10.7 ms @ 48 kHz) → ~35 ms Latenz; bei fft_size=2048 ≈ 46 ms → ~140 ms.
// 1 Block wird konsumiert + Reserve für GPU-Readback-Jitter/GC-Pausen.
const TARGET_QUEUE = 3;

// Metering alle 8 Render-Quanten (8 × 128 Samples ≈ 23 ms @ 44.1 kHz) —
// entspricht der alten 25 ms UI-Kadenz, statt ~340 Nachrichten/s.
const METER_EVERY = 8;

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
        // Resampler-Position im Quellblock (in Quell-Samples, fraktional).
        // Der Fraktionalanteil wird über Blockgrenzen mitgenommen (Carry),
        // damit das Resampling sample-kontinuierlich bleibt.
        this.pos = 0;
        // Einblendrampe nach Underrun/Start (5 ms)
        this.fade = 0.0;
        this.fadeStep = 1.0 / (0.005 * sampleRate);
        this.wasUnderrun = false;
        this.underruns = 0;
        this.gain = 1.0;
        // Metering
        this.peakL = 0;
        this.peakR = 0;
        this.quantumCount = 0;

        this.port.onmessage = (e) => {
            const m = e.data;
            if (m.type === "block") {
                this.queue.push({ l: m.l, r: m.r });
            } else if (m.type === "gain") {
                this.gain = m.value;
            } else if (m.type === "prime") {
                // Deterministischer Start: erste Blöcke anfordern
                this.requestBlocks();
            } else if (m.type === "stop") {
                this.alive = false;
            }
        };
    }

    requestBlocks() {
        // Fordert genau die fehlenden Blöcke an. Der Main-Thread rendert pro
        // Anfrage einen Block (sequenziell, nie parallel — der WASM-Engine-
        // Zustand ist nicht nebenläufigkeitssicher).
        const missing = TARGET_QUEUE - this.queue.length;
        if (missing > 0) {
            this.port.postMessage({ type: "ack", count: missing, underruns: this.underruns });
        }
    }

    process(inputs, outputs) {
        if (this.alive === false) return false;
        const out = outputs[0];
        const L = out[0];
        const R = out[1] || out[0];
        const n = L.length; // Render-Quantum: 128 Samples
        const ratio = this.sampleRateIn / sampleRate; // Quell-Steps pro Ziel-Sample

        for (let i = 0; i < n; i++) {
            // Nächsten Block anfordern, wenn keiner aktiv ist
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
            let sL = (blk.l[i0] + (blk.l[i1] - blk.l[i0]) * frac) * this.gain;
            let sR = (blk.r[i0] + (blk.r[i1] - blk.r[i0]) * frac) * this.gain;

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
