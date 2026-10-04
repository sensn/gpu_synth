#![allow(warnings)]
use cubecl::prelude::*;
use cubecl_wgpu::{RuntimeOptions, WebGpu, WgpuDevice, WgpuRuntime};
use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

pub mod stereo_synth;
use stereo_synth::cubek_true_stereo_synth_reverb;

// ============================================================
// POLYPHONIE: Voice-Tabelle — ein Kernel-Launch pro Stimme
// ============================================================
const MAX_VOICES: usize = 8;
/// Release-Schwanz: Stimme gilt als beendet nach 5x Release-Zeit
/// (der exponentielle Release erreicht nie exakt 0).
const RELEASE_TAIL_FACTOR: f32 = 5.0;

/// Eine Stimme des Voice-Tables. Jede aktive Stimme wird pro Block
/// mit GENAU EINEM Kernel-Launch des (unveränderten) Kernels gerendert.
#[derive(Clone, Copy)]
struct Voice {
    /// Slot belegt (klingt oder im Release-Schwanz)
    active: bool,
    /// Gate offen = Taste gehalten (A/D/S-Phase), zu = Release-Phase
    gate_on: bool,
    /// Tonhöhe beim Note-On (Schlüssel für das Note-Off-Matching)
    note_frequency: f32,
    /// Aktuelle Render-Höhe (die Lead-Stimme folgt dem Frequenz-Slider)
    frequency: f32,
    /// Glättungs-Anker: Render-Höhe des letzten Blocks (Portamento)
    old_frequency: f32,
    /// Absoluter Sample-Index des letzten Note-On
    note_on_sample: u32,
    /// Absoluter Sample-Index des letzten Note-Off
    note_off_sample: u32,
    /// Drone-Stimme: wird von Note-Events nie getroffen oder gestohlen
    is_drone: bool,
}

impl Voice {
    fn new() -> Self {
        Self {
            active: false,
            gate_on: false,
            note_frequency: 0.0,
            frequency: 0.0,
            old_frequency: 0.0,
            note_on_sample: 0,
            note_off_sample: 0,
            is_drone: false,
        }
    }

    fn spawn(frequency: f32, now: u32, is_drone: bool) -> Self {
        Self {
            active: true,
            gate_on: true,
            note_frequency: frequency,
            frequency,
            old_frequency: frequency,
            note_on_sample: now,
            note_off_sample: now,
            is_drone,
        }
    }

    /// Stille Dummy-Stimme für Idle-Blöcke: hält die WGSL-Pipeline warm
    /// (Kompilierung beim ersten Block, wie beim ursprünglichen Mono-Engine),
    /// produziert aber exakte Nullen (gate zu, t_off == t_on → amp_at_off = 0).
    fn silent_dummy() -> Self {
        Self {
            active: true,
            gate_on: false,
            note_frequency: 0.0,
            frequency: 110.0,
            old_frequency: 110.0,
            note_on_sample: 0,
            note_off_sample: 0,
            is_drone: false,
        }
    }
}

#[wasm_bindgen]
pub struct WebAudioEngine {
    client: Option<ComputeClient<WgpuRuntime>>,
    fft_size: u32,
    last_cutoff: Cell<f32>,
    lfo_phase: Cell<f32>,
    block_count: Cell<u32>,
    // POLYPHONIE: Voice-Tabelle statt einzelner Mono-Note-Events
    voices: RefCell<[Voice; MAX_VOICES]>,
}

#[wasm_bindgen]
impl WebAudioEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            client: None,
            fft_size: 512, //2048
            last_cutoff: Cell::new(800.0),
            lfo_phase: Cell::new(0.0),
            block_count: Cell::new(0),
            voices: RefCell::new([Voice::new(); MAX_VOICES]),
        }
    }

    /// POLYPHONIE Note-On mit Voice-Allokation:
    /// 1. Gehaltene Stimme mit gleicher Frequenz → Retrigger (ADSR-Neustart).
    /// 2. Freier Slot → neue Stimme.
    /// 3. Älteste Release-Stimme → Slot-Recycling (klingt ohnehin aus).
    /// 4. Älteste gehaltene Stimme → Voice Stealing.
    pub fn note_on(&self, frequency: f32) {
        let mut voices = self.voices.borrow_mut();
        // Note-On-Zeitpunkt: Start des nächsten Blocks (der aktuelle Block ist
        // bereits mit altem Zustand unterwegs).
        let now = self.block_count.get().wrapping_mul(self.fft_size);

        // 1) Retrigger derselben noch gehaltenen Note (Drone ausgenommen)
        for v in voices.iter_mut() {
            if v.active && v.gate_on && !v.is_drone && (v.note_frequency - frequency).abs() < 0.01 {
                v.note_on_sample = now;
                v.note_off_sample = now;
                return;
            }
        }

        // 2) Freien Slot belegen
        for v in voices.iter_mut() {
            if !v.active {
                *v = Voice::spawn(frequency, now, false);
                return;
            }
        }

        // 3) Älteste Release-Stimme recyceln
        let mut oldest_release: Option<usize> = None;
        for (i, v) in voices.iter().enumerate() {
            if v.active && !v.gate_on {
                let better = match oldest_release {
                    None => true,
                    Some(j) => v.note_off_sample < voices[j].note_off_sample,
                };
                if better {
                    oldest_release = Some(i);
                }
            }
        }
        if let Some(i) = oldest_release {
            voices[i] = Voice::spawn(frequency, now, false);
            return;
        }

        // 4) Voice Stealing: älteste Note-On gewinnt den Slot (Drone bleibt)
        let mut oldest: Option<usize> = None;
        for (i, v) in voices.iter().enumerate() {
            if v.is_drone {
                continue;
            }
            if oldest.is_none() || v.note_on_sample < voices[oldest.unwrap()].note_on_sample {
                oldest = Some(i);
            }
        }
        if let Some(i) = oldest {
            voices[i] = Voice::spawn(frequency, now, false);
        }
    }

    /// POLYPHONIE Note-Off: schließt das Gate der gehaltenen Stimme mit dieser
    /// Frequenz; ihre Release-Phase startet ab dem nächsten Block.
    pub fn note_off(&self, frequency: f32) {
        let mut voices = self.voices.borrow_mut();
        let now = self.block_count.get().wrapping_mul(self.fft_size);

        for v in voices.iter_mut() {
            if v.active && v.gate_on && !v.is_drone && (v.note_frequency - frequency).abs() < 0.01 {
                v.gate_on = false;
                v.note_off_sample = now;
                return;
            }
        }
        // Kein Match (z.B. Stimme wurde bereits gestohlen): No-Op.
    }

    /// POLYPHONIE Drone-Modus: eigene, dauerhaft gehaltene Stimme auf der
    /// Slider-Frequenz. Note-Events treffen sie nie (kein Retrigger, kein
    /// Stealing, kein Note-Off) — Tastatur-Akkorde klingen zusätzlich dazu.
    pub fn set_drone(&self, on: bool, frequency: f32) {
        let mut voices = self.voices.borrow_mut();
        let now = self.block_count.get().wrapping_mul(self.fft_size);

        if on {
            // Bereits aktiv → nichts tun (Lead-Logik folgt dem Slider)
            if voices.iter().any(|v| v.is_drone) {
                return;
            }
            // Freien Slot suchen; wenn voll: älteste Nicht-Drone-Stimme stehlen
            let mut slot: Option<usize> = None;
            for (i, v) in voices.iter().enumerate() {
                if !v.active {
                    slot = Some(i);
                    break;
                }
            }
            if slot.is_none() {
                let mut oldest: Option<usize> = None;
                for (i, v) in voices.iter().enumerate() {
                    if v.is_drone {
                        continue;
                    }
                    if oldest.is_none() || v.note_on_sample < voices[oldest.unwrap()].note_on_sample {
                        oldest = Some(i);
                    }
                }
                slot = oldest;
            }
            if let Some(i) = slot {
                voices[i] = Voice::spawn(frequency, now, true);
            }
        } else {
            // Drone-Stimme sauber ins Release schicken
            for v in voices.iter_mut() {
                if v.is_drone {
                    v.is_drone = false;
                    v.gate_on = false;
                    v.note_off_sample = now;
                }
            }
        }
    }

    /// Gate-Status: wahr, wenn mindestens eine Stimme gehalten wird (UI/Drone).
    pub fn is_gate_on(&self) -> bool {
        self.voices.borrow().iter().any(|v| v.active && v.gate_on)
    }

    /// Anzahl belegter Stimmen (klingend oder im Release) — für die UI.
    pub fn active_voice_count(&self) -> u32 {
        self.voices.borrow().iter().filter(|v| v.active).count() as u32
    }

    /// Render-Höhe der Lead-Stimme — dieselbe Rangfolge wie im Renderer:
    /// Rang 1 Drone-Stimme, Rang 2 die zuletzt angeschlagene gehaltene
    /// Tastatur-Stimme. 0.0 wenn nichts gehalten wird. Das Frontend
    /// synchronisiert den Frequenz-Slider damit, damit die Lead-Stimme
    /// nicht auf eine veraltete Tonhöhe gezogen wird.
    pub fn get_lead_frequency(&self) -> f32 {
        let voices = self.voices.borrow();
        if let Some(v) = voices.iter().find(|v| v.active && v.is_drone) {
            return v.frequency;
        }
        let mut lead: Option<&Voice> = None;
        for v in voices.iter() {
            if v.active && v.gate_on && !v.is_drone {
                if lead.is_none() || v.note_on_sample > lead.unwrap().note_on_sample {
                    lead = Some(v);
                }
            }
        }
        lead.map(|v| v.frequency).unwrap_or(0.0)
    }

    pub fn init_engine_async(mut self) -> js_sys::Promise {
        future_to_promise(async move {
            let device = Default::default();
            let setup =
                cubecl_wgpu::init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;
            let wgpu_device = cubecl_wgpu::init_device(setup, RuntimeOptions::default());
            self.client = Some(ComputeClient::load(&wgpu_device));
            Ok(JsValue::from(self))
        })
    }

    /// Rendert einen Block: EIN KERNEL-LAUNCH PRO AKTIVER STIMME mit
    /// stimmen-spezifischen Note-Events, danach CPU-Mix (Equal-Power) der
    /// Stimmen-Puffer zu einem gemeinsamen Stereo-Block. Bei keiner aktiven
    /// Stimme läuft genau ein stiller Dummy-Launch (Pipeline bleibt warm).
    pub fn render_block_async(
        &self,
        frequency: f32,
        cutoff: f32,
        room_size: f32,
        wet_mix: f32,
        attack: f32,
        decay: f32,
        sustain: f32,
        release: f32,
        js_ratios: js_sys::Float32Array,
        js_levels: js_sys::Float32Array,
        algo_select: u32,
        moog_res: f32,
        obe_res: f32,
        obe_mode: u32,
        lfo_freq: f32,
        lfo_depth: f32,
        stereo_width: f32,
        high_freq_damping: f32,
    ) -> js_sys::Promise {
        let client = self
            .client
            .as_ref()
            .expect("Engine nicht initialisiert.")
            .clone();

        let output_len = (self.fft_size * 2) as usize;

        let ratios_vec: Vec<f32> = js_ratios.to_vec();
        let levels_vec: Vec<f32> = js_levels.to_vec();

        let bytes_ratios =
            cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&ratios_vec).to_vec());
        let bytes_levels =
            cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&levels_vec).to_vec());
        let handle_ratios = client.create(bytes_ratios);
        let handle_levels = client.create(bytes_levels);

        let grid_dim = CubeCount::Static((self.fft_size + 255) / 256, 1, 1);
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

        let old_cut = self.last_cutoff.get();
        let current_lfo_phase = self.lfo_phase.get();
        let current_block_index = self.block_count.get();

        // Snapshot der aktiven Stimmen. Ohne aktive Stimme: eine stille
        // Dummy-Stimme, damit der Kernel kompiliert bleibt (Warm-Up) und
        // der Scheduler niemals ins Leeren läuft.
        let active: Vec<Voice> = {
            let voices = self.voices.borrow();
            let snapshot: Vec<Voice> = voices.iter().filter(|v| v.active).copied().collect();
            if snapshot.is_empty() {
                vec![Voice::silent_dummy()]
            } else {
                snapshot
            }
        };

        // LEAD-STIMME: die der Frequenz-Slider steuert (Portamento).
        // Rang 1: die Drone-Stimme (klassisches Verhalten — Drone folgt dem
        // Slider). Rang 2: ohne Drone die zuletzt angeschlagene, gehaltene
        // Tastatur-Stimme (Mono-Portamento). Alle anderen halten ihre Höhe.
        let lead_idx = active
            .iter()
            .position(|v| v.is_drone)
            .or_else(|| {
                let newest_on = active
                    .iter()
                    .filter(|v| v.gate_on && !v.is_drone)
                    .map(|v| v.note_on_sample)
                    .max()
                    .unwrap_or(u32::MAX);
                active
                    .iter()
                    .position(|v| v.gate_on && !v.is_drone && v.note_on_sample == newest_on)
            })
            .unwrap_or(usize::MAX);

        // EIN KERNEL-LAUNCH PRO STIMME — derselbe kompilierte Kernel,
        // stimmen-spezifische Skalar-Argumente, gemeinsame globale Parameter.
        let mut handles_out = Vec::with_capacity(active.len());
        for (i, v) in active.iter().enumerate() {
            let target_freq = if i == lead_idx {
                frequency
            } else {
                v.frequency
            };

            let initial_data = vec![0.0f32; output_len];
            let raw_bytes =
                cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&initial_data).to_vec());
            let handle_out = client.create(raw_bytes);

            let arg_audio = unsafe { ArrayArg::from_raw_parts(handle_out.clone(), output_len) };
            let arg_ratios = unsafe { ArrayArg::from_raw_parts(handle_ratios.clone(), 6) };
            let arg_levels = unsafe { ArrayArg::from_raw_parts(handle_levels.clone(), 6) };

            cubek_true_stereo_synth_reverb::launch::<f32, WgpuRuntime>(
                &client,
                grid_dim.clone(),
                cube_dim.clone(),
                arg_audio,
                target_freq,
                v.old_frequency,
                cutoff,
                old_cut,
                44100.0,
                current_lfo_phase,
                arg_ratios,
                arg_levels,
                algo_select,
                moog_res,
                obe_res,
                obe_mode,
                lfo_freq,
                lfo_depth,
                room_size,
                high_freq_damping,
                wet_mix,
                stereo_width,
                attack,
                decay,
                sustain,
                release,
                if v.gate_on { 1u32 } else { 0u32 },
                v.note_on_sample,
                v.note_off_sample,
                current_block_index, // Globaler Block-Zähler für die absolute IDFT-Phase
                self.fft_size,       // #[comptime] fft_size
            );
            handles_out.push(handle_out);
        }

        let block_duration = (self.fft_size as f32) / 44100.0;
        let next_lfo_phase =
            current_lfo_phase + (2.0 * std::f32::consts::PI * lfo_freq * block_duration);
        self.lfo_phase
            .set(next_lfo_phase % (2.0 * std::f32::consts::PI));

        self.last_cutoff.set(cutoff);
        self.block_count.set(current_block_index.wrapping_add(1));

        // Host-Update nach dem Launch: Glättungs-Anker fortschreiben und
        // abgelaufene Release-Schwänze freigeben.
        {
            let mut voices = self.voices.borrow_mut();
            let block_start_sample = current_block_index.wrapping_mul(self.fft_size);
            let mut slot = 0;
            for v in voices.iter_mut() {
                if v.active {
                    if slot == lead_idx {
                        // Lead-Stimme: Slider-Glättung übernehmen (Portamento)
                        v.old_frequency = frequency;
                        v.frequency = frequency;
                    } else {
                        v.old_frequency = v.frequency;
                    }
                    // Release-Schwanz beendet → Slot freigeben
                    if !v.gate_on {
                        let t_since_off = block_start_sample
                            .wrapping_sub(v.note_off_sample) as f32
                            / 44100.0;
                        if t_since_off > release * RELEASE_TAIL_FACTOR {
                            *v = Voice::new();
                        }
                    }
                    slot += 1;
                }
            }
        }

        // CPU-MIX mit Equal-Power-Skalierung: gleiche Lautheit unabhängig von
        // der Stimmenzahl, verhindert Clipping bei Akkorden.
        let num_voices = active.len();
        let voice_scale = 1.0 / (num_voices as f32).sqrt();

        future_to_promise(async move {
            let result_bytes_res = client.read_async(handles_out).await;
            let result_bytes_vec = result_bytes_res.expect("WebGPU Lesevorgang fehlgeschlagen");
            let mut mixed = vec![0.0f32; output_len];
            for bytes in &result_bytes_vec {
                let samples: &[f32] = bytemuck::cast_slice(bytes.as_ref());
                for (acc, s) in mixed.iter_mut().zip(samples.iter()) {
                    *acc += s * voice_scale;
                }
            }
            let js_array = js_sys::Float32Array::from(mixed.as_slice());
            Ok(JsValue::from(js_array))
        })
    }
}
