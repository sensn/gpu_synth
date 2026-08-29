#![allow(warnings)]
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuRuntime, RuntimeOptions, WebGpu, WgpuDevice};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use std::cell::Cell;

pub mod stereo_synth;
use stereo_synth::cubek_true_stereo_synth_reverb;

#[wasm_bindgen]
pub struct WebAudioEngine {
    client: Option<ComputeClient<WgpuRuntime>>, 
    fft_size: u32,
    last_frequency: Cell<f32>,
    last_cutoff: Cell<f32>,
    lfo_phase: Cell<f32>,
}

#[wasm_bindgen]
impl WebAudioEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { 
            client: None, 
            fft_size: 2048,
            last_frequency: Cell::new(110.0),
            last_cutoff: Cell::new(800.0),
            lfo_phase: Cell::new(0.0),
        }
    }

    pub fn init_engine_async(mut self) -> js_sys::Promise {
        future_to_promise(async move {
            let device = Default::default();
            let setup = cubecl_wgpu::init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;
            let wgpu_device = cubecl_wgpu::init_device(setup, RuntimeOptions::default());
            self.client = Some(ComputeClient::load(&wgpu_device));
            Ok(JsValue::from(self))
        })
    }

    pub fn render_block_async(
        &self, 
        frequency: f32, cutoff: f32, room_size: f32, wet_mix: f32, attack: f32, decay: f32,
        js_ratios: js_sys::Float32Array,
        js_levels: js_sys::Float32Array,
        algo_select: u32,
        moog_res: f32, obe_res: f32, obe_mode: u32,
        lfo_freq: f32, lfo_depth: f32, stereo_width: f32, high_freq_damping: f32
    ) -> js_sys::Promise {
        let client = self.client.as_ref().expect("Engine nicht initialisiert.").clone();
            
        let output_len = (self.fft_size * 2) as usize;
        let initial_data = vec![0.0f32; output_len];
        let byte_vec = bytemuck::cast_slice(&initial_data).to_vec();
        let raw_bytes = cubecl::bytes::Bytes::from_bytes_vec(byte_vec);
        let handle_out = client.create(raw_bytes);

        // Konvertiere die JavaScript-Typen-Arrays in native Rust-Slices
        let ratios_vec: Vec<f32> = js_ratios.to_vec();
        let levels_vec: Vec<f32> = js_levels.to_vec();

        // Reserviere dedizierten VRAM auf der GPU für die DX7 Operator-Eigenschaften
        let bytes_ratios = cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&ratios_vec).to_vec());
        let bytes_levels = cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&levels_vec).to_vec());
        let handle_ratios = client.create(bytes_ratios);
        let handle_levels = client.create(bytes_levels);

        let grid_dim = CubeCount::Static((self.fft_size + 255) / 256, 1, 1);
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        
        let arg_audio = unsafe { ArrayArg::from_raw_parts(handle_out.clone(), output_len) };
        let arg_ratios = unsafe { ArrayArg::from_raw_parts(handle_ratios.clone(), 6) };
        let arg_levels = unsafe { ArrayArg::from_raw_parts(handle_levels.clone(), 6) };

        let old_freq = self.last_frequency.get();
        let old_cut = self.last_cutoff.get();
        let current_lfo_phase = self.lfo_phase.get();

        // KORREKTUR: Reicht jetzt lückenlos alle 25 Argumente in der bit-perfekten Reihenfolge an die GPU weiter!
        cubek_true_stereo_synth_reverb::launch::<f32, WgpuRuntime>(
            &client, grid_dim, cube_dim, arg_audio,
            frequency, old_freq, 
            cutoff, old_cut,
            44100.0, current_lfo_phase,
            arg_ratios, arg_levels, algo_select, // ◄ FIX: Allokierte VRAM Arrays eingebunden
            moog_res, obe_res, obe_mode,
            lfo_freq, lfo_depth,
            room_size, high_freq_damping, wet_mix, stereo_width, 
            attack, decay,
            self.fft_size,
        );

        // Kontinuierliche Phasenfortführung für den knackfreien LFO-Gleitschutz
        let block_duration = (self.fft_size as f32) / 44100.0;
        let next_lfo_phase = current_lfo_phase + (2.0 * std::f32::consts::PI * lfo_freq * block_duration);
        self.lfo_phase.set(next_lfo_phase % (2.0 * std::f32::consts::PI));

        self.last_frequency.set(frequency);
        self.last_cutoff.set(cutoff);

        future_to_promise(async move {
            let result_bytes_res = client.read_async(vec![handle_out]).await;
            let result_bytes_vec = result_bytes_res.expect("WebGPU Lesevorgang fehlgeschlagen");
            if let Some(first_bytes) = result_bytes_vec.first() {
                let js_array = js_sys::Float32Array::from(bytemuck::cast_slice(first_bytes.as_ref()) as &[f32]);
                Ok(JsValue::from(js_array))
            } else {
                Err(JsValue::from_str("Fehler beim Extrahieren des GPU-Streams"))
            }
        })
    }
}
