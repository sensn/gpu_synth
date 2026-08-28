use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;
use wasm_bindgen::prelude::*;

// Importiere den True-Stereo-Kernel aus den vorherigen Schritten
// (Stelle sicher, dass src/stereo_synth.rs in deinem Projekt existiert)
pub mod stereo_synth;
use stereo_synth::{cubek_true_stereo_synth_reverb, StereoIrConfig};

#[wasm_bindgen]
pub struct WebAudioEngine {
    // Nutzen der expliziten WgpuRuntime aus cubecl-wgpu 0.11
    client: ComputeClient<WgpuRuntime>, 
    fft_size: u32,
}

#[wasm_bindgen]
impl WebAudioEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Initialisierung des WebGPU-Devices passend zu cubecl-wgpu 0.11
        let device = Default::default();
        let client = WgpuRuntime::init_device(&device);

        Self { client, fft_size: 2048 }
    }

    /// Berechnet die Synthesizer-Stimme inklusive Faltungshall direkt via WebGPU
    pub fn render_block(&self, frequency: f32, cutoff: f32, room_size: f32, wet_mix: f32) -> Vec<f32> {
        let output_len = (self.fft_size * 2) as usize; // Stereo L + R Interleaved
        
        // Speicherbereich im WebGPU VRAM reservieren
        let handle_out = self.client.create(bytemuck::cast_slice(&vec![0.0f32; output_len]));

        let ir_config = StereoIrConfig {
            room_size_seconds: room_size,
            high_freq_damping: 1.2,
            wet_dry_mix: wet_mix,
            stereo_width: 0.85, // Angenehme, breite Stereobühne
        };

        // Thread-Grid berechnen
        let grid_dim = CubeCount::Static((self.fft_size + 255) / 256, 1, 1);
        let cube_dim = CubeDim::new(256, 1, 1);

        // Kernel abschicken
        cubek_true_stereo_synth_reverb::launch::<f32, WgpuRuntime>(
            &self.client,
            grid_dim,
            cube_dim,
            ArrayArg::new(&handle_out, output_len),
            frequency,
            44100.0,
            cutoff,
            ir_config,
            self.fft_size,
        );

        // Daten zurück in den WASM-Speicher spiegeln
        let result_bytes = self.client.read(handle_out.binding());
        let audio_samples: &[f32] = bytemuck::cast_slice(&result_bytes);
        
        audio_samples.to_vec()
    }
}
