use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

/// wgpu requires `bytes_per_row` on multi-row texture copies to be a multiple
/// of `COPY_BYTES_PER_ROW_ALIGNMENT` (256). For a 64³ R8Uint volume the natural
/// row length is 64 bytes, which is not aligned — spec-violating even if it
/// happens to work on most backends today. When the format widens to R16Uint
/// (hash-thing-1v0.1) the same row becomes 128 bytes, still not aligned. This
/// helper returns the padded row stride in bytes; callers use it as the upload
/// `bytes_per_row` and pad rows in a staging buffer if the padded value
/// differs from the unpadded one.
fn padded_bytes_per_row(width: u32, bytes_per_texel: u32) -> u32 {
    let unpadded = width * bytes_per_texel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    unpadded.div_ceil(align) * align
}

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

    // Shared
    uniform_buffer: wgpu::Buffer,
    volume_size: u32,
    /// Bytes-per-texel for the volume format. R8Uint = 1, R16Uint = 2. Stored
    /// alongside `volume_size` so `upload_volume` can compute row padding
    /// without re-deriving it from the texture format enum.
    volume_bytes_per_texel: u32,
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
            uniform_buffer,
            volume_size,
            // R8Uint: 1 byte per texel. Update alongside the `format:` line
            // above if the texture format ever widens.
            volume_bytes_per_texel: 1,
            mode: RenderMode::Svdag,
            camera_yaw: std::f32::consts::FRAC_PI_4,
            camera_pitch: 0.4,
            camera_dist: 2.0,
            camera_target: [0.5, 0.5, 0.5],
        }
    }

    /// Upload (or re-upload) a serialized SVDAG to the GPU.
    pub fn upload_svdag(&mut self, dag: &super::Svdag) {
        let bytes: &[u8] = bytemuck::cast_slice(&dag.nodes);
        let needed = bytes.len() as u64;

        // Allocate a larger buffer if necessary (grow-only)
        if self.svdag_buffer.is_none() || needed > self.svdag_buffer_cap {
            // Grow to next power of 2 with a floor of 64KB
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

        if let Some(buf) = &self.svdag_buffer {
            self.queue.write_buffer(buf, 0, bytes);
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
        let w = self.volume_size;
        let bpt = self.volume_bytes_per_texel;
        let unpadded = w * bpt;
        let padded = padded_bytes_per_row(w, bpt);

        // Fast path: row length already aligned, pass the caller's slice
        // straight through. Slow path: copy row-by-row into a padded staging
        // buffer. Per-upload allocation is fine — uploads are infrequent
        // (once per generation tick) and ~64KiB for 64³ R8Uint.
        let staging: Vec<u8>;
        let (rows_data, row_stride) = if padded == unpadded {
            (data, unpadded)
        } else {
            let total_rows = (w * w) as usize; // height × depth
            let mut buf = vec![0u8; total_rows * padded as usize];
            let unpadded_usize = unpadded as usize;
            let padded_usize = padded as usize;
            for row in 0..total_rows {
                let src = &data[row * unpadded_usize..][..unpadded_usize];
                let dst = &mut buf[row * padded_usize..][..unpadded_usize];
                dst.copy_from_slice(src);
            }
            staging = buf;
            (staging.as_slice(), padded)
        };

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rows_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_stride),
                rows_per_image: Some(w),
            },
            wgpu::Extent3d {
                width: w,
                height: w,
                depth_or_array_layers: w,
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

#[cfg(test)]
mod tests {
    use super::padded_bytes_per_row;

    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    #[test]
    fn padded_row_rounds_up_to_alignment() {
        // 64 R8Uint texels = 64 bytes; rounds up to the 256-byte alignment.
        assert_eq!(padded_bytes_per_row(64, 1), ALIGN);
    }

    #[test]
    fn padded_row_respects_bytes_per_texel() {
        // 64 R16Uint texels = 128 bytes; still under ALIGN, still rounds up.
        assert_eq!(padded_bytes_per_row(64, 2), ALIGN);
    }

    #[test]
    fn padded_row_passes_through_already_aligned() {
        // 256 R8Uint texels = 256 bytes exactly: one alignment unit, no padding.
        assert_eq!(padded_bytes_per_row(ALIGN, 1), ALIGN);
        // 128 R16Uint texels = 256 bytes: same story.
        assert_eq!(padded_bytes_per_row(ALIGN / 2, 2), ALIGN);
    }

    #[test]
    fn padded_row_rounds_up_beyond_single_unit() {
        // 300 R8Uint texels = 300 bytes; rounds up to 512 (next alignment unit).
        assert_eq!(padded_bytes_per_row(300, 1), 2 * ALIGN);
        // 300 R16Uint texels = 600 bytes; rounds up to 768 (3 alignment units).
        assert_eq!(padded_bytes_per_row(300, 2), 3 * ALIGN);
    }
}
