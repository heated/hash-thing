use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera_pos: [f32; 4],
    camera_dir: [f32; 4],
    camera_up: [f32; 4],
    camera_right: [f32; 4],
    /// x: volume_size, y: aspect_ratio, z: fov_tan, w: time
    params: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode {
    Flat3D,
    Svdag,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Flat 3D texture path
    flat_pipeline: wgpu::RenderPipeline,
    flat_bind_group: wgpu::BindGroup,
    volume_texture: wgpu::Texture,

    // SVDAG path
    svdag_pipeline: wgpu::RenderPipeline,
    svdag_bind_group_layout: wgpu::BindGroupLayout,
    svdag_bind_group: Option<wgpu::BindGroup>,
    svdag_buffer: Option<wgpu::Buffer>,
    svdag_buffer_cap: u64, // current allocation in bytes
    /// Number of u32 slots in the `Svdag::nodes` vector that are already live
    /// on the GPU. `upload_svdag` uses this to write only the newly-appended
    /// tail (plus the root-offset header at slot 0, which changes every frame).
    /// Reset to 0 when the GPU buffer is resized or freshly created.
    svdag_uploaded_len: usize,

    // Shared
    uniform_buffer: wgpu::Buffer,
    volume_size: u32,
    pub mode: RenderMode,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_dist: f32,
    pub camera_target: [f32; 3],
}

impl Renderer {
    pub async fn new(window: Arc<Window>, volume_size: u32) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: Default::default(),
            display: Default::default(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hash-thing device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume"),
            size: wgpu::Extent3d {
                width: volume_size,
                height: volume_size,
                depth_or_array_layers: volume_size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let volume_view = volume_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let uniforms = Uniforms {
            camera_pos: [0.0; 4],
            camera_dir: [0.0; 4],
            camera_up: [0.0, 1.0, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            params: [volume_size as f32, 1.0, 1.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Flat 3D texture pipeline ===

        let flat_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flat_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let flat_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat_bg"),
            layout: &flat_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
            ],
        });

        let flat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat raycast shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raycast.wgsl").into()),
        });

        let flat_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flat_pl"),
            bind_group_layouts: &[Some(&flat_bind_group_layout)],
            immediate_size: 0,
        });

        let flat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flat_rp"),
            layout: Some(&flat_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &flat_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &flat_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // === SVDAG pipeline ===

        let svdag_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("svdag_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let svdag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svdag raycast shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("svdag_raycast.wgsl").into()),
        });

        let svdag_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("svdag_pl"),
                bind_group_layouts: &[Some(&svdag_bind_group_layout)],
                immediate_size: 0,
            });

        let svdag_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("svdag_rp"),
            layout: Some(&svdag_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &svdag_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &svdag_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            flat_pipeline,
            flat_bind_group,
            volume_texture,
            svdag_pipeline,
            svdag_bind_group_layout,
            svdag_bind_group: None,
            svdag_buffer: None,
            svdag_buffer_cap: 0,
            svdag_uploaded_len: 0,
            uniform_buffer,
            volume_size,
            mode: RenderMode::Svdag,
            camera_yaw: std::f32::consts::FRAC_PI_4,
            camera_pitch: 0.4,
            camera_dist: 2.0,
            camera_target: [0.5, 0.5, 0.5],
        }
    }

    /// Upload an `Svdag` to the GPU, writing only the slots that weren't
    /// already uploaded in a previous call.
    ///
    /// Two things make this incremental:
    /// 1. `Svdag` keeps its flat node array across frames (persistent content
    ///    cache keyed by 9-u32 slot bytes), so the tail `nodes[old_len..new_len]`
    ///    contains exactly the slots that are genuinely new this frame.
    /// 2. Slot 0 holds the root-offset header, which the shader reads once per
    ///    ray to start traversal. The root's slot changes almost every
    ///    simulation step, so we always re-upload slot 0 (4 bytes) regardless
    ///    of the tail length.
    ///
    /// When the buffer cap grows or the first call happens, we fall back to a
    /// full re-upload.
    pub fn upload_svdag(&mut self, dag: &super::Svdag) {
        let bytes: &[u8] = bytemuck::cast_slice(&dag.nodes);
        let needed = bytes.len() as u64;

        // Grow the GPU buffer if we can't fit `bytes`. This is a one-way reset
        // of the upload watermark — the fresh buffer has nothing on it, so the
        // next write below must cover every slot.
        let mut require_full = false;
        if self.svdag_buffer.is_none() || needed > self.svdag_buffer_cap {
            let mut cap = self.svdag_buffer_cap.max(65536);
            while cap < needed {
                cap *= 2;
            }
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("svdag_buffer"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.svdag_buffer = Some(buffer);
            self.svdag_buffer_cap = cap;
            self.svdag_uploaded_len = 0;
            require_full = true;

            // Rebuild bind group
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("svdag_bg"),
                layout: &self.svdag_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.svdag_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            });
            self.svdag_bind_group = Some(bg);
        }

        let Some(buf) = &self.svdag_buffer else {
            return;
        };

        if require_full || self.svdag_uploaded_len > dag.nodes.len() {
            // Fresh buffer or `dag.nodes` shrank (e.g. builder was recreated).
            // Full re-upload.
            self.queue.write_buffer(buf, 0, bytes);
            self.svdag_uploaded_len = dag.nodes.len();
            return;
        }

        // Slot 0 is the root-offset header and changes almost every frame.
        // Always re-upload it (4 bytes, effectively free).
        self.queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&dag.nodes[0..1]));

        // Tail: append-only growth since the last upload.
        if self.svdag_uploaded_len < dag.nodes.len() {
            let tail = &dag.nodes[self.svdag_uploaded_len..];
            self.queue.write_buffer(
                buf,
                (self.svdag_uploaded_len as u64) * 4,
                bytemuck::cast_slice(tail),
            );
            self.svdag_uploaded_len = dag.nodes.len();
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn upload_volume(&self, data: &[u8]) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.volume_size),
                rows_per_image: Some(self.volume_size),
            },
            wgpu::Extent3d {
                width: self.volume_size,
                height: self.volume_size,
                depth_or_array_layers: self.volume_size,
            },
        );
    }

    pub fn render(&mut self) -> bool {
        use wgpu::CurrentSurfaceTexture;

        let surface_texture = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(tex) | CurrentSurfaceTexture::Suboptimal(tex) => tex,
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return false,
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return false;
            }
            _ => return false,
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Camera
        let (sin_yaw, cos_yaw) = self.camera_yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.camera_pitch.sin_cos();

        let cam_dir = [-cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw];
        let cam_pos = [
            self.camera_target[0] - cam_dir[0] * self.camera_dist,
            self.camera_target[1] - cam_dir[1] * self.camera_dist,
            self.camera_target[2] - cam_dir[2] * self.camera_dist,
        ];

        let right = [cos_yaw, 0.0, -sin_yaw];
        let up = [
            right[1] * cam_dir[2] - right[2] * cam_dir[1],
            right[2] * cam_dir[0] - right[0] * cam_dir[2],
            right[0] * cam_dir[1] - right[1] * cam_dir[0],
        ];

        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov_tan = (std::f32::consts::FRAC_PI_4 / 2.0).tan();

        let uniforms = Uniforms {
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            camera_dir: [cam_dir[0], cam_dir[1], cam_dir[2], 0.0],
            camera_up: [up[0], up[1], up[2], 0.0],
            camera_right: [right[0], right[1], right[2], 0.0],
            params: [self.volume_size as f32, aspect, fov_tan, 0.0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            match self.mode {
                RenderMode::Flat3D => {
                    render_pass.set_pipeline(&self.flat_pipeline);
                    render_pass.set_bind_group(0, &self.flat_bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
                RenderMode::Svdag => {
                    if let Some(bg) = &self.svdag_bind_group {
                        render_pass.set_pipeline(&self.svdag_pipeline);
                        render_pass.set_bind_group(0, bg, &[]);
                        render_pass.draw(0..6, 0..1);
                    } else {
                        // No SVDAG uploaded yet — fall back to flat path.
                        render_pass.set_pipeline(&self.flat_pipeline);
                        render_pass.set_bind_group(0, &self.flat_bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        true
    }
}
