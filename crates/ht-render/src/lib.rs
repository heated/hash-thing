pub(crate) mod font;
mod renderer;
mod svdag;

pub use renderer::{
    FrameOutcome, OffSurfacePixels, RenderScaleOverride, Renderer, RendererGpuTimingSample,
};
pub use svdag::{cpu_trace, Svdag};

#[cfg(test)]
mod wgsl_drift_guard {
    //! Pins the Rust `Cell::METADATA_BITS` constant to the hardcoded
    //! `>> 6u` shifts in both raycast shaders. If this test fails, update
    //! `METADATA_BITS` in src/octree/node.rs AND both shader files in the
    //! same change. See comments next to `material_color` in each shader.
    use ht_octree::Cell;
    use ht_octree::NodeStore;
    use wgpu::util::DeviceExt;

    const SVDAG_RAYCAST_WGSL: &str = include_str!("svdag_raycast.wgsl");
    const PARTICLE_WGSL: &str = include_str!("particle.wgsl");

    #[test]
    fn wgsl_metadata_shift_matches_rust() {
        let expected = format!("packed >> {}u", Cell::METADATA_BITS);
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected),
            "svdag_raycast.wgsl must contain `{expected}` — \
             Cell::METADATA_BITS drifted from the hardcoded shift. \
             Update the shader."
        );
    }

    #[test]
    fn wgsl_svdag_entry_face_normal_cascade_matches_rust() {
        let expected_lines = [
            "} else if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {",
            "normal = vec3<f32>(-sign(rd.x), 0.0, 0.0);",
            "} else if tmin_v.y >= tmin_v.z {",
            "normal = vec3<f32>(0.0, -sign(rd.y), 0.0);",
            "normal = vec3<f32>(0.0, 0.0, -sign(rd.z));",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the entry-face normal cascade drifted from the CPU \
                 oracle in svdag.rs::cpu_trace::raycast_with_budget. \
                 Both sides must use `>=` ties on `argmax(tmin_v)` with \
                 `-sign(rd.*)` on the entry axis (hash-thing-rv4). \
                 Update whichever side is wrong so they stay byte-equivalent."
            );
        }
    }

    #[test]
    fn wgsl_svdag_inside_leaf_normal_fallback_matches_rust() {
        let expected_lines = [
            "let tmax_v = max(lt1, lt2);",
            "let inside = tmin_v.x < 0.0 && tmin_v.y < 0.0 && tmin_v.z < 0.0;",
            "if tmax_v.x <= tmax_v.y && tmax_v.x <= tmax_v.z {",
            "normal = vec3<f32>(sign(rd.x), 0.0, 0.0);",
            "} else if tmax_v.y <= tmax_v.z {",
            "normal = vec3<f32>(0.0, sign(rd.y), 0.0);",
            "normal = vec3<f32>(0.0, 0.0, sign(rd.z));",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the inside-leaf exit-face fallback drifted from the CPU \
                 oracle in svdag.rs::cpu_trace::raycast_with_budget. Both \
                 sides must detect inside-leaf via `all(tmin_v < 0)` and \
                 pick the nearest exit face with `sign(rd)` on that axis \
                 (hash-thing-2nd). Update whichever side is wrong so they \
                 stay byte-equivalent."
            );
        }
    }

    #[test]
    fn wgsl_svdag_inside_lod_node_normal_fallback_matches_rust() {
        let expected_lines = [
            "let ntmax_v = max(nt1, nt2);",
            "let inside_node = ntmin_v.x < 0.0 && ntmin_v.y < 0.0 && ntmin_v.z < 0.0;",
            "if ntmax_v.x <= ntmax_v.y && ntmax_v.x <= ntmax_v.z {",
            "normal = vec3<f32>(sign(rd.x), 0.0, 0.0);",
            "} else if ntmax_v.y <= ntmax_v.z {",
            "normal = vec3<f32>(0.0, sign(rd.y), 0.0);",
            "normal = vec3<f32>(0.0, 0.0, sign(rd.z));",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the representative-material LOD path must detect inside-node \
                 origins via `all(ntmin_v < 0)` and pick the nearest exit face \
                 with `sign(rd)` on that axis (hash-thing-p9dd). Update \
                 whichever side is wrong so CPU and WGSL stay aligned."
            );
        }
    }

    #[test]
    fn wgsl_leaf_shading_uses_surface_hit_position() {
        let expected_lines = [
            "let hit_t = select(",
            "ro_local + rd * max(hit_t, 0.0) - normal * (INV_RES * 0.25);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — leaf shading \
                 should sample the actual surface hit position (nudged inward), \
                 not the leaf center, or collapsed leaves turn into rectangular \
                 lighting artifacts."
            );
        }
    }

    #[test]
    fn wgsl_lod_shading_uses_surface_hit_position() {
        let expected_lines = [
            "let lod_hit_t = select(",
            "ro_local + rd * max(lod_hit_t, 0.0) - normal * (INV_RES * 0.25);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — representative-\
                 material LOD hits should shade the actual surface hit position \
                 (nudged inward), not the node center, or coarse interiors turn \
                 into rectangular ghost patches."
            );
        }
    }

    #[test]
    fn wgsl_dump_debug_modes_cover_7m63_axes() {
        let expected_lines = [
            "fn normal_axis_debug(normal: vec3<f32>) -> vec3<f32>",
            "if debug_mode == 2u",
            "normal_axis_debug(normal)",
            "if debug_mode == 3u",
            "vec4<f32>(1.0, 0.25, 1.0",
            "vec4<f32>(0.15, 1.0, 0.35",
            "if debug_mode == 4u",
            "vec4<f32>(material_color(lod_mat)",
            "vec4<f32>(material_color(mat)",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — dump-frame \
                 diagnostics need normal-axis, LOD-vs-leaf, and raw-material \
                 modes for hash-thing-nznv / 7m63."
            );
        }
    }

    #[test]
    fn wgsl_stone_detail_stays_low_contrast() {
        let expected_lines = [
            "let coarse = value_noise(vox * 0.18);",
            "let fine = value_noise(vox * 1.2);",
            "let vein = smoothstep(0.50, 0.62, value_noise(vox * vec3<f32>(0.32, 0.13, 0.32)));",
            "base = base * (0.94 + 0.10 * coarse) * (0.96 + 0.06 * fine);",
            "base = mix(base, base * 0.78, vein * 0.22);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — stone chamber \
                 walls should keep low-contrast grain instead of large blurry \
                 hypertexture patches (hash-thing-hcol)."
            );
        }
    }

    #[test]
    fn wgsl_hit_alpha_tracks_scene_distance() {
        let expected_lines = [
            "vec4<f32>(lit, max(entry + max(lod_hit_t, 0.0), 1e-4))",
            "vec4<f32>(lit, max(entry + max(hit_t, 0.0), 1e-4))",
            "final_color = vec4<f32>(bg, 0.0);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — the raycast \
                 output alpha now carries scene hit distance for overlay \
                 occlusion; background pixels must keep alpha 0."
            );
        }
    }

    #[test]
    fn wgsl_particles_sample_scene_depth_for_occlusion() {
        let expected_lines = [
            "@group(0) @binding(3) var t_scene: texture_2d<f32>;",
            "let scene = textureSample(t_scene, s_scene, in.screen_uv);",
            "if scene.a > 0.0 && scene.a + depth_epsilon < in.ray_t {",
            "let size = 0.5 / u.params.x;",
        ];
        for expected in expected_lines {
            assert!(
                PARTICLE_WGSL.contains(expected),
                "particle.wgsl must contain `{expected}` — billboard overlays \
                 should sample the raycast texture's scene depth and discard \
                 particles hidden behind voxel geometry."
            );
        }
    }

    #[test]
    fn wgsl_svdag_octant_of_tiebreak_matches_rust() {
        let expected_lines = [
            "if pos.x > mid.x || (pos.x == mid.x && rd.x >= 0.0) { idx |= 1u; }",
            "if pos.y > mid.y || (pos.y == mid.y && rd.y >= 0.0) { idx |= 2u; }",
            "if pos.z > mid.z || (pos.z == mid.z && rd.z >= 0.0) { idx |= 4u; }",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the octant_of midpoint tiebreak drifted from the CPU \
                 oracle in svdag.rs::cpu_trace::octant_of. Both sides \
                 must use strict `>` with `rd >= 0.0` tiebreak on exact \
                 midpoint matches (hash-thing-6hd). Update whichever \
                 side is wrong so they stay byte-equivalent."
            );
        }
    }

    #[test]
    fn wgsl_material_palette_uses_buffer_lookup() {
        let expected = "palette[mat_id].xyz";
        assert!(
            SVDAG_RAYCAST_WGSL.contains(expected),
            "svdag_raycast.wgsl must contain `{expected}` — material_color \
             should read from the GPU palette buffer, not a hardcoded switch \
             (hash-thing-5bb.7)."
        );
    }

    #[test]
    fn wgsl_svdag_traversal_constants_match_rust() {
        use crate::svdag::{cpu_trace, CHILD_BASE_WORD, GRAND_MASK_HI_WORD, GRAND_MASK_LO_WORD};

        let expected_depth = format!("const MAX_DEPTH: u32 = {}u;", cpu_trace::MAX_DEPTH);
        let expected_min = format!(
            "const MIN_STEP_BUDGET: u32 = {}u;",
            cpu_trace::MIN_STEP_BUDGET
        );
        let expected_fudge = format!(
            "const STEP_BUDGET_FUDGE: u32 = {}u;",
            cpu_trace::STEP_BUDGET_FUDGE
        );

        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_depth),
            "svdag_raycast.wgsl must contain `{expected_depth}` — \
             MAX_DEPTH drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_min),
            "svdag_raycast.wgsl must contain `{expected_min}` — \
             MIN_STEP_BUDGET drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_fudge),
            "svdag_raycast.wgsl must contain `{expected_fudge}` — \
             STEP_BUDGET_FUDGE drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );

        let expected_stack = format!("const MAX_STACK: u32 = {}u;", cpu_trace::MAX_STACK);
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_stack),
            "svdag_raycast.wgsl must contain `{expected_stack}` — \
             MAX_STACK drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );

        let expected_grand_lo = format!("const GRAND_MASK_LO_WORD: u32 = {}u;", GRAND_MASK_LO_WORD);
        let expected_grand_hi = format!("const GRAND_MASK_HI_WORD: u32 = {}u;", GRAND_MASK_HI_WORD);
        let expected_child_base = format!("const CHILD_BASE_WORD: u32 = {}u;", CHILD_BASE_WORD);
        for expected in [expected_grand_lo, expected_grand_hi, expected_child_base] {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(&expected),
                "svdag_raycast.wgsl must contain `{expected}` — SVDAG slot \
                 layout constants drifted from svdag.rs."
            );
        }
    }

    #[test]
    fn svdag_raycast_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(SVDAG_RAYCAST_WGSL)
            .expect("svdag_raycast.wgsl must parse as WGSL");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("svdag_raycast.wgsl must pass naga validation");
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RaycastUniforms {
        camera_pos: [f32; 4],
        camera_dir: [f32; 4],
        camera_up: [f32; 4],
        camera_right: [f32; 4],
        params: [f32; 4],
        debug: [f32; 4],
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn mat(id: u16) -> u16 {
        Cell::pack(id, 0).raw()
    }

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: Default::default(),
            display: Default::default(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("svdag parity test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()
    }

    fn shader_hits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dag: &crate::Svdag,
        ro: [f32; 3],
        rd: [f32; 3],
    ) -> bool {
        let uniforms = RaycastUniforms {
            camera_pos: [ro[0], ro[1], ro[2], 0.0],
            camera_dir: [rd[0], rd[1], rd[2], 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            params: [(1u32 << dag.root_level) as f32, 1.0, 1.0, 1.0],
            debug: [0.0, 1.0, 1.0, 1.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svdag parity uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dag_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svdag parity dag"),
            contents: bytemuck::cast_slice(&dag.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let palette = [[1.0f32, 1.0, 1.0, 1.0]; 16];
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svdag parity palette"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("svdag parity output"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output.create_view(&Default::default());
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("svdag parity layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("svdag parity bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dag_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: palette_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svdag parity shader"),
            source: wgpu::ShaderSource::Wgsl(SVDAG_RAYCAST_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svdag parity pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svdag parity pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let padded_row = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svdag parity readback"),
            size: padded_row,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("svdag parity encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svdag parity pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row as u32),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("svdag parity map callback")
            .expect("svdag parity map");
        let mapped = slice.get_mapped_range();
        let alpha_half_bits = u16::from_le_bytes([mapped[6], mapped[7]]);
        drop(mapped);
        readback.unmap();
        alpha_half_bits != 0
    }

    fn assert_shader_cpu_hit_parity(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dag: &crate::Svdag,
        ro: [f32; 3],
        rd: [f32; 3],
        label: &str,
    ) {
        let cpu_hit = crate::cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false)
            .hit_cell
            .is_some();
        let shader_hit = shader_hits(device, queue, dag, ro, rd);
        assert_eq!(shader_hit, cpu_hit, "{label}: shader/CPU hit parity");
    }

    #[test]
    fn svdag_raycast_shader_matches_cpu_for_grandchild_masks() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("skipping GPU parity test: no headless adapter");
            return;
        };

        let mut store = NodeStore::new();
        let empty_root = store.empty(2);
        let empty = crate::Svdag::build(&store, empty_root, 2);
        assert_shader_cpu_hit_parity(
            &device,
            &queue,
            &empty,
            [-1.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            "empty root",
        );

        let uniform_root = store.leaf(mat(3));
        let uniform = crate::Svdag::build(&store, uniform_root, 2);
        assert_shader_cpu_hit_parity(
            &device,
            &queue,
            &uniform,
            [-1.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            "uniform root",
        );

        let mat1 = mat(1);
        let mut root = store.empty(2);
        root = store.set_cell(root, 1, 0, 0, mat1);
        let dag = crate::Svdag::build(&store, root, 2);
        assert_shader_cpu_hit_parity(
            &device,
            &queue,
            &dag,
            [-1.0, 0.125, 0.125],
            [1.0, 0.0, 0.0],
            "empty grandchild then occupied sibling",
        );

        let mat2 = mat(2);
        let mut root = store.empty(2);
        root = store.set_cell(root, 2, 3, 3, mat2);
        let dag = crate::Svdag::build(&store, root, 2);
        let ro = [2.0, 2.0, 2.0];
        let target = [2.5 / 4.0, 3.5 / 4.0, 3.5 / 4.0];
        let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
        assert_shader_cpu_hit_parity(
            &device,
            &queue,
            &dag,
            ro,
            rd,
            "negative mirrored grandchild",
        );

        let mut root = store.empty(2);
        root = store.set_cell(root, 0, 0, 2, mat2);
        let dag = crate::Svdag::build(&store, root, 2);
        assert_shader_cpu_hit_parity(
            &device,
            &queue,
            &dag,
            [0.125, 0.125, -1.0],
            [0.0, 0.0, 1.0],
            "high-word grandchild",
        );
    }
}
