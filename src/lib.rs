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
    block_counter: Cell<u32>,
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
            block_counter: Cell::new(0),
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

    // FIX: Die Signatur nimmt nun attack und decay als f32 entgegen!
    pub fn render_block_async(
        &self, 
        frequency: f32, 
        cutoff: f32, 
        room_size: f32, 
        wet_mix: f32, 
        attack: f32, 
        decay: f32
    ) -> js_sys::Promise {
        let client = self.client.as_ref()
            .expect("Engine nicht initialisiert. Rufe zuerst init_engine_async auf.")
            .clone();
            
        let output_len = (self.fft_size * 2) as usize;
        let initial_data = vec![0.0f32; output_len];
        
        let byte_vec = bytemuck::cast_slice(&initial_data).to_vec();
        let raw_bytes = cubecl::bytes::Bytes::from_bytes_vec(byte_vec);
        let handle_out = client.create(raw_bytes);

        let grid_dim = CubeCount::Static((self.fft_size + 255) / 256, 1, 1);
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        let array_arg = unsafe { ArrayArg::from_raw_parts(handle_out.clone(), output_len) };

        let old_freq = self.last_frequency.get();
        let old_cut = self.last_cutoff.get();
        let current_block = self.block_counter.get();

        // Übergabe aller Parameter an den GPU-Kernel
        cubek_true_stereo_synth_reverb::launch::<f32, WgpuRuntime>(
            &client,
            grid_dim,
            cube_dim,
            array_arg,
            frequency,
            cutoff,
            old_freq,  
            old_cut,   
            44100.0,
            room_size,
            1.2,
            wet_mix,
            0.85,
            current_block, 
            attack, // Reicht den Attack-Wert an die GPU weiter
            decay,  // Reiche den Decay-Wert an die GPU weiter
            self.fft_size,
        );

        self.last_frequency.set(frequency);
        self.last_cutoff.set(cutoff);
        self.block_counter.set(current_block + 1);

        future_to_promise(async move {
            let result_bytes_res = client.read_async(vec![handle_out]).await;
            let result_bytes_vec = result_bytes_res.expect("WebGPU asynchroner Lesevorgang fehlgeschlagen");
            
            if let Some(first_bytes) = result_bytes_vec.first() {
                let raw_slice: &[u8] = first_bytes.as_ref();
                let audio_samples: &[f32] = bytemuck::cast_slice(raw_slice);
                
                let js_array = js_sys::Float32Array::from(audio_samples);
                Ok(JsValue::from(js_array))
            } else {
                Err(JsValue::from_str("Fehler beim Extrahieren des GPU-Byte-Streams"))
            }
        })
    }
}
