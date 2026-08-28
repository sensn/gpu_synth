#![allow(warnings)]
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuRuntime, RuntimeOptions, WebGpu, WgpuDevice};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

pub mod stereo_synth;
use stereo_synth::{cubek_true_stereo_synth_reverb, StereoIrConfig};

#[wasm_bindgen]
pub struct WebAudioEngine {
    client: Option<ComputeClient<WgpuRuntime>>, 
    fft_size: u32,
}

#[wasm_bindgen]
impl WebAudioEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { client: None, fft_size: 2048 }
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

    pub fn render_block_async(&self, frequency: f32, cutoff: f32, room_size: f32, wet_mix: f32) -> js_sys::Promise {
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

        cubek_true_stereo_synth_reverb::launch::<f32, WgpuRuntime>(
            &client,
            grid_dim,
            cube_dim,
            array_arg,
            frequency,
            44100.0,
            cutoff,
            room_size,
            1.2,
            wet_mix,
            0.85,
            self.fft_size,
        );

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
