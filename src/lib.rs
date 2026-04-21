// Workspace crate re-exports — external API unchanged.
pub use ht_octree as octree;
pub use ht_octree::rng;
pub use ht_render as render;

pub mod acquire_harness;
pub mod perf;
pub mod player;
pub mod sim;
pub mod terrain;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use {
    std::{cell::RefCell, rc::Rc, sync::Arc},
    winit::{application::ApplicationHandler, event::WindowEvent},
};

#[cfg(target_arch = "wasm32")]
const WASM_VOLUME_SIZE: u32 = 64;

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

    let mut app = WasmApp {
        window: None,
        gpu: Rc::new(RefCell::new(WasmGpuState::default())),
    };
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}

#[cfg(target_arch = "wasm32")]
struct WasmApp {
    window: Option<Arc<winit::window::Window>>,
    gpu: Rc<RefCell<WasmGpuState>>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WasmGpuState {
    init_started: bool,
    renderer: Option<render::Renderer>,
}

#[cfg(target_arch = "wasm32")]
impl ApplicationHandler for WasmApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        use winit::window::WindowAttributes;

        let attrs = WindowAttributes::default()
            .with_title("hash-thing | WebGPU")
            .with_inner_size(winit::dpi::LogicalSize::new(960, 540));

        let window = Arc::new(
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

        if self.gpu.borrow().init_started {
            return;
        }

        self.gpu.borrow_mut().init_started = true;
        let gpu = Rc::clone(&self.gpu);
        let window = Arc::clone(self.window.as_ref().expect("window just created"));
        wasm_bindgen_futures::spawn_local(async move {
            let renderer = render::Renderer::new(window.clone(), WASM_VOLUME_SIZE).await;
            gpu.borrow_mut().renderer = Some(renderer);
            log::info!("WASM renderer initialized");
            window.request_redraw();
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.gpu.borrow_mut().renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                let outcome = {
                    let mut gpu = self.gpu.borrow_mut();
                    gpu.renderer.as_mut().map(render::Renderer::render)
                };
                if matches!(outcome, Some(render::FrameOutcome::Occluded)) {
                    return;
                }
                if outcome.is_some() {
                    let window = self.window.as_ref().expect("window exists during redraw");
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
