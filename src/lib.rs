#![allow(warnings)]
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuRuntime, RuntimeOptions, WebGpu, WgpuDevice};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// Zustandsspeicher aus der Standardbibliothek für die Hüllkurven und Block-Indexierung
use std::cell::Cell;

pub mod stereo_synth;
use stereo_synth::{cubek_true_stereo_synth_reverb};

#[wasm_bindgen]
pub struct WebAudioEngine {
    client: Option<ComputeClient<WgpuRuntime>>, 
    fft_size: u32,
    // Hüllkurven-Zustandsspeicher für die Parameter-Interpolation
    last_frequency: Cell<f32>,
    last_cutoff: Cell<f32>,
    // REPARATUR-ERWEITERUNG: Verfolgt den globalen Fortlauf der generierten Blöcke für LFO & ADSR
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
            block_counter: Cell::new(0), // Startet beim allerersten Audio-Block
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

        // Lese die historischen Parameterwerte aus den Cells aus
        let old_freq = self.last_frequency.get();
        let old_cut = self.last_cutoff.get();
        
        // REPARATUR-ERWEITERUNG: Hole den aktuellen Zeit-Index des Streams
        let current_block = self.block_counter.get();

        // Starte den Kernel mit der erweiterten ADSR/LFO Signatur
        cubek_true_stereo_synth_reverb::launch::<f32, WgpuRuntime>(
            &client,
            grid_dim,
            cube_dim,
            array_arg,
            frequency,
            cutoff,
            old_freq,  // Reiche den alten Frequenzwert an die GPU weiter
            old_cut,   // Reiche den alten Cutoffwert an die GPU weiter
            44100.0,
            room_size,
            1.2,
            wet_mix,
            0.85,
            current_block, // REPARATUR-ERWEITERUNG: Der u32-Block-Zähler fließt direkt in das GPU-Register
            self.fft_size,
        );

        // Aktualisiere den Zustandsspeicher für den nächsten Block
        self.last_frequency.set(frequency);
        self.last_cutoff.set(cutoff);
        
        // REPARATUR-ERWEITERUNG: Erhöhe den Zähler für die nächste Puffer-Berechnung
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
