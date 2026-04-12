pub mod octree;
pub mod perf;
pub mod player;
pub mod render;
pub mod rng;
pub mod sim;
pub mod terrain;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// WASM entry point. Sets up logging, inserts a canvas into the DOM, and
/// starts the winit event loop. The actual renderer init requires async
/// wiring (tracked as xb7.6) — this scaffolding compiles and runs but
/// does not render yet.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("console_log init");

    log::info!("hash-thing: WASM/WebGPU mode");

    use winit::event_loop::EventLoop;
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = WasmApp { window: None };
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}

#[cfg(target_arch = "wasm32")]
struct WasmApp {
    window: Option<std::sync::Arc<winit::window::Window>>,
}

#[cfg(target_arch = "wasm32")]
impl winit::application::ApplicationHandler for WasmApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        use winit::window::WindowAttributes;

        let attrs = WindowAttributes::default()
            .with_title("hash-thing | WebGPU")
            .with_inner_size(winit::dpi::LogicalSize::new(960, 540));

        let window = std::sync::Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create WASM window"),
        );

        // Insert canvas into the DOM
        {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("winit should have a canvas on WASM");
            let web_window = web_sys::window().expect("no global window object");
            let document = web_window.document().expect("window has no document");
            let container = document
                .get_element_by_id("hash-thing-canvas")
                .unwrap_or_else(|| {
                    document
                        .body()
                        .expect("document has no body element")
                        .into()
                });
            container
                .append_child(&canvas)
                .expect("failed to append canvas to DOM");
        }

        self.window = Some(window);

        // TODO(hash-thing-xb7.6): wire async renderer init via
        // wasm_bindgen_futures::spawn_local. pollster::block_on panics
        // on WASM — need to restructure the init path.
        log::info!("WASM window created — renderer init requires async wiring (xb7.6)");
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
