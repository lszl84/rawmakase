//! GPU port of the per-pixel color and tone stage (`develop.wgsl`). Samples from the
//! stage cache stay on the device while only the recipe changes, and so does their tone
//! stage while only the stages after it change (see `Kept`); parameters and tables are
//! uploaded per render. The result is read back for the CPU finishing steps.
use super::Processor;
use crate::develop::pipeline::{Samples, pixel_params::PixelParams};
use crate::rendered::Rendered;
use anyhow::{Context, Result, ensure};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use wgpu::util::DeviceExt;

/// Invocations per workgroup, as declared in `develop.wgsl`.
const GROUP: u32 = 256;
/// Stand-in position for samples outside the photo; `develop.wgsl` tests against it
/// rather than NaN, which shader compilers may assume never occurs.
const OUTSIDE: f32 = -3e38;

pub(super) struct Developer {
    pub(super) layout: wgpu::BindGroupLayout,
    /// The layout without the mask weights, for passes that only share the camera
    /// stage (`logs.wgsl`) and must stay within eight storage buffers.
    pub(super) camera_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    /// Uploaded sample sets, most recently used first, so switching between Fit and
    /// 100% does not upload again.
    samples: Vec<Uploaded>,
    /// Bound in place of the kept tone stage by passes that do not keep it.
    no_tone: wgpu::Buffer,
}
/// The tone stage's output for one set of samples and the recipe as the tone stage
/// reads it (`PixelParams::tone`), so renders that change only the stages after it
/// (the Basic tone sliders, curves, colour) read it instead of running it again.
pub(super) struct Kept {
    pixels: wgpu::Buffer,
    positions: wgpu::Buffer,
    recipe: crate::model::recipe::Recipe,
    toned: wgpu::Buffer,
    /// A submitted pass wrote `toned` for these samples and `recipe`. Until then it is
    /// written again: a pass recorded but never submitted (a cancelled render) left it
    /// as it was. `recipe` is the tone of the last render of these samples, written or
    /// not.
    written: bool,
    /// The last recorded pass writes it; marked written once submitted.
    pending: bool,
}
/// Uploaded sample sets kept, and their device memory budget.
const UPLOADS: usize = 3;
const UPLOAD_BUDGET: u64 = 512 << 20;
struct Uploaded {
    source: Arc<Samples>,
    pixels: wgpu::Buffer,
    positions: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
}
impl Developer {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let source =
            crate::develop::pipeline::pixel_params::wgsl_prelude() + include_str!("develop.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Develop"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 2 && binding != 7,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Develop buffers"),
            entries: &entries,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera stage buffers"),
            entries: &entries[..6],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Develop layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Develop"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("develop"),
            compilation_options: Default::default(),
            cache: None,
        });
        let no_tone = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("No kept tone stage"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            layout,
            camera_layout,
            pipeline,
            samples: Vec::new(),
            no_tone,
        }
    }
}
/// Samples made on the device (`resident.rs`), and the develop stage's output.
pub(crate) struct DeviceSamples {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: wgpu::Buffer,
    pub(super) positions: wgpu::Buffer,
    pub(super) output: wgpu::Buffer,
}
/// What the develop stage reads: samples prepared on the CPU, or on the device.
#[derive(Clone, Copy)]
pub(crate) enum Input<'a> {
    Cpu(&'a Arc<Samples>),
    Device(&'a DeviceSamples),
}
impl Input<'_> {
    pub(super) fn size(&self) -> (u32, u32) {
        match self {
            Input::Cpu(s) => (s.width, s.height),
            Input::Device(s) => (s.width, s.height),
        }
    }
}
impl Developer {
    /// Makes `samples` the most recent upload, uploading them unless they are there.
    fn upload(&mut self, device: &wgpu::Device, samples: &Arc<Samples>) {
        let developer = self;
        let n = samples.pixels.len() as u64;
        if let Some(i) = developer
            .samples
            .iter()
            .position(|u| Arc::ptr_eq(&u.source, samples))
        {
            let uploaded = developer.samples.remove(i);
            developer.samples.insert(0, uploaded);
        } else {
            // Release the least recently used buffers before allocating more.
            let bytes = |u: &Uploaded| u.source.pixels.len() as u64 * 44;
            developer.samples.truncate(UPLOADS - 1);
            while developer.samples.iter().map(bytes).sum::<u64>() + n * 44 > UPLOAD_BUDGET
                && developer.samples.pop().is_some()
            {}
            let positions: Vec<[f32; 2]> = samples
                .positions
                .iter()
                .map(|p| if p[0].is_nan() { [OUTSIDE; 2] } else { *p })
                .collect();
            let init = |label, contents: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let buffer = |label, usage| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: n * 12,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let uploaded = Uploaded {
                source: samples.clone(),
                pixels: init("Develop samples", bytemuck::cast_slice(&samples.pixels)),
                positions: init("Develop positions", bytemuck::cast_slice(&positions)),
                output: buffer(
                    "Developed pixels",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                ),
                staging: buffer(
                    "Develop readback",
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                ),
            };
            developer.samples.insert(0, uploaded);
        }
    }
}
/// What a develop pass does with the kept tone stage of its samples; the value is the
/// shader's `size.z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Keep {
    /// Runs the tone stage.
    Run = 0,
    /// Reads the kept tone stage.
    Read = 1,
    /// Runs the tone stage and keeps it.
    Write = 2,
}
impl Keep {
    /// From the samples' entry, when there is one: whether its tone is this render's,
    /// and whether a submitted pass wrote it. The tone is kept once a second render of
    /// the samples has the same tone: while Exposure itself moves, every value is new
    /// and writing it would only add work.
    pub(super) fn step(entry: Option<(bool, bool)>) -> Self {
        match entry {
            Some((true, true)) => Self::Read,
            Some((true, false)) => Self::Write,
            _ => Self::Run,
        }
    }
}
/// Sample sets whose tone stage is kept: the Fit and its draft, and a 100% region.
const KEPT: usize = 3;
/// Marks the kept tone stages written once the pass that writes them is submitted.
pub(super) fn submitted(kept: &mut [Kept]) {
    for k in kept.iter_mut().filter(|k| k.pending) {
        k.pending = false;
        k.written = true;
    }
}
impl Processor {
    /// Uploads CPU samples unless they are on the device already, and records the
    /// develop pass; returns the buffer that holds its result.
    pub(super) fn record_develop(
        &mut self,
        input: Input,
        params: &PixelParams,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Buffer {
        let device = &self.device;
        let developer = self.developer.get_or_insert_with(|| Developer::new(device));
        let (w, h) = input.size();
        let n = w as u64 * h as u64;
        let (pixels, positions, output) = match input {
            Input::Device(s) => (s.pixels.clone(), s.positions.clone(), s.output.clone()),
            Input::Cpu(samples) => {
                developer.upload(device, samples);
                let u = &developer.samples[0];
                (u.pixels.clone(), u.positions.clone(), u.output.clone())
            }
        };
        // 1 reads the kept tone stage, 2 runs it and keeps it, 0 runs it.
        let (mode, toned) = match &params.tone {
            Some(recipe) if params.weights.is_empty() => {
                let bytes = n * 12;
                // The entry for these samples, most recently used first.
                let found = self
                    .kept
                    .iter()
                    .position(|k| k.pixels == pixels && k.positions == positions);
                let entry = found.map(|i| self.kept.remove(i));
                let step = Keep::step(
                    entry
                        .as_ref()
                        .filter(|k| k.toned.size() >= bytes)
                        .map(|k| (k.recipe == *recipe, k.written)),
                );
                let toned = match entry {
                    Some(k) if k.toned.size() >= bytes => k.toned,
                    _ => device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Kept tone stage"),
                        size: bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                };
                self.kept.insert(
                    0,
                    Kept {
                        pixels: pixels.clone(),
                        positions: positions.clone(),
                        recipe: recipe.clone(),
                        toned: toned.clone(),
                        written: step == Keep::Read,
                        pending: step == Keep::Write,
                    },
                );
                self.kept.truncate(KEPT);
                (step as u32, Some(toned))
            }
            _ => (0, None),
        };
        let developer = self.developer.get_or_insert_with(|| Developer::new(device));
        let toned = toned.unwrap_or_else(|| developer.no_tone.clone());
        let storage = |label, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let params_buffer = storage("Develop parameters", &params.params);
        let tables = storage("Develop tables", &params.tables);
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask weights"),
            contents: bytemuck::cast_slice(if params.weights.is_empty() {
                &[0u32; 4]
            } else {
                &params.weights[..]
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let groups = (n as u32).div_ceil(GROUP);
        let max = device.limits().max_compute_workgroups_per_dimension;
        let (gx, gy) = (groups.min(max), groups.div_ceil(groups.min(max)));
        let size = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Develop size"),
            contents: bytemuck::cast_slice(&[n as u32, gx, mode, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let entries: Vec<_> = [
            &pixels,
            &positions,
            &output,
            &params_buffer,
            &tables,
            &size,
            &weights,
            &toned,
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Develop bindings"),
            layout: &developer.layout,
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Develop pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&developer.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        output
    }
    /// The per-pixel stage over `samples`. Does not change CPU state on failure.
    pub(crate) fn develop(
        &mut self,
        samples: &Arc<Samples>,
        params: &PixelParams,
        cancel: &AtomicBool,
    ) -> Result<Rendered> {
        self.check_develop(Input::Cpu(samples), params, cancel)?;
        self.scoped(|gpu| gpu.develop_inner(samples, params, cancel))
    }
    /// The per-pixel stage over `pixels` (at no particular position), in buffers of their
    /// own rather than the upload cache: toning the reduced photo for the
    /// Shadows/Highlights map (`pixel_params::tone_params`).
    pub(crate) fn develop_pixels(
        &mut self,
        pixels: &[[f32; 3]],
        params: &PixelParams,
        cancel: &AtomicBool,
    ) -> Result<Vec<[f32; 3]>> {
        let n = pixels.len() as u64;
        ensure!(n > 0, "Empty develop region");
        let device = self.device.clone();
        let init = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: n * 12,
                usage,
                mapped_at_creation: false,
            })
        };
        let input = DeviceSamples {
            width: n as u32,
            height: 1,
            pixels: init("Map pixels", bytemuck::cast_slice(pixels)),
            positions: init("Map positions", &vec![0; n as usize * 8]),
            output: buffer(
                "Toned map pixels",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        };
        self.check_develop(Input::Device(&input), params, cancel)?;
        let staging = buffer(
            "Toned map readback",
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        let output = self.record_develop(Input::Device(&input), params, &mut encoder);
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, n * 12);
        let submission = super::submit(&self.queue, encoder);
        submitted(&mut self.kept);
        let (tx, rx) = mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        super::wait(&device, submission)?;
        rx.recv_timeout(std::time::Duration::from_secs(1))
            .context("GPU readback timed out")??;
        let pixels =
            bytemuck::cast_slice::<u8, [f32; 3]>(&staging.slice(..).get_mapped_range()?).to_vec();
        staging.unmap();
        Ok(pixels)
    }
    /// Whether the device can develop `samples` at all.
    pub(super) fn check_develop(
        &self,
        input: Input,
        params: &PixelParams,
        cancel: &AtomicBool,
    ) -> Result<()> {
        ensure!(!cancel.load(Ordering::Relaxed), "Render superseded");
        let (w, h) = input.size();
        let n = w as u64 * h as u64;
        ensure!(n > 0, "Empty develop region");
        let limits = self.device.limits();
        ensure!(
            n * 12 <= limits.max_storage_buffer_binding_size
                && n * 12 <= limits.max_buffer_size
                && (params.tables.len() as u64 * 4) <= limits.max_storage_buffer_binding_size,
            "Region exceeds GPU buffer limits"
        );
        ensure!(
            n * 44 <= 1024 * 1024 * 1024,
            "Region exceeds GPU memory budget"
        );
        Ok(())
    }
    /// Runs `work` inside validation, internal and memory error scopes; a device error
    /// releases the uploaded samples and resident buffers and fails.
    pub(crate) fn scoped<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let allocation = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let result = work(self);
        let memory_error = pollster::block_on(allocation.pop());
        let internal_error = pollster::block_on(internal.pop());
        let validation_error = pollster::block_on(validation.pop());
        if let Some(error) = memory_error.or(internal_error).or(validation_error) {
            if let Some(d) = &mut self.developer {
                d.samples.clear();
            }
            self.release_resident();
            anyhow::bail!("{error}");
        }
        result
    }
    fn develop_inner(
        &mut self,
        samples: &Arc<Samples>,
        params: &PixelParams,
        cancel: &AtomicBool,
    ) -> Result<Rendered> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.record_develop(Input::Cpu(samples), params, &mut encoder);
        ensure!(!cancel.load(Ordering::Relaxed), "Render superseded");
        let device = &self.device;
        let n = samples.pixels.len() as u64;
        let uploaded = &self.developer.as_ref().unwrap().samples[0];
        encoder.copy_buffer_to_buffer(&uploaded.output, 0, &uploaded.staging, 0, n * 12);
        let submission = super::submit(&self.queue, encoder);
        submitted(&mut self.kept);
        let (tx, rx) = mpsc::sync_channel(1);
        uploaded
            .staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let poll = super::wait(device, submission);
        if let Err(error) = poll {
            uploaded.staging.unmap();
            return Err(error.into());
        }
        rx.recv_timeout(std::time::Duration::from_secs(1))
            .context("GPU readback timed out")??;
        if cancel.load(Ordering::Relaxed) {
            uploaded.staging.unmap();
            anyhow::bail!("Render superseded");
        }
        let mapped = uploaded.staging.slice(..).get_mapped_range()?;
        let pixels = bytemuck::cast_slice::<u8, [f32; 3]>(&mapped).to_vec();
        drop(mapped);
        uploaded.staging.unmap();
        Ok(Rendered {
            width: samples.width,
            height: samples.height,
            pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Keep;

    /// A render reads the kept tone stage only after a submitted pass wrote it for the
    /// same tone; a pass recorded but never submitted (a cancelled render) leaves it to
    /// be written again.
    #[test]
    fn the_kept_tone_stage_is_read_only_once_a_submitted_pass_wrote_it() {
        // (same tone as the entry, written): new samples, a new tone, the second
        // render of a tone, then the third.
        assert_eq!(Keep::step(None), Keep::Run);
        assert_eq!(Keep::step(Some((false, true))), Keep::Run);
        assert_eq!(Keep::step(Some((true, false))), Keep::Write);
        assert_eq!(Keep::step(Some((true, true))), Keep::Read);
        // Only `submitted` marks an entry written, so after a writing pass that was
        // cancelled before submission the entry is still unwritten: written again.
        assert_eq!(Keep::step(Some((true, false))), Keep::Write);
    }
}
