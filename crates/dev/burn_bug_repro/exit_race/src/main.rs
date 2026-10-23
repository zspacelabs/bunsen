//! # Exiting with work queued on a wgpu device can segfault.
//!
//! On the wgpu backends (`vulkan`, `wgpu`), a device's work runs on the
//! runtime's own device thread. A program that frees its tensors and returns
//! from `main` can leave that thread busy (releasing memory-pool pages
//! through `wgpu_hal::vulkan::Device::destroy_buffer`) while libc's `exit()`
//! runs the NVIDIA driver's exit handler, which tears the driver down beneath
//! it: the process dies with `SIGSEGV` after its output is written.
//!
//! Seen on Linux, NVIDIA 595.99.02 (RTX 3090), burn 0.22. CUDA and the CPU
//! (`flex`) don't crash. Waiting for the device before returning
//! (`Device::sync`, the `--sync` flag here) avoids it.
//!
//! **Status: this doesn't reproduce the crash yet** (0 of 40 Vulkan runs over
//! four workloads; see the README). bunsen's own Whisper binary does.
//!
//! The crash is a race: it depends on how much is freed at exit and on the
//! parent's timing (it shows with the child's output on pipes, not files, and
//! never under gdb). So `--runs N` runs this binary N times as a child, with
//! piped output, and counts the crashes.
//!
//! ```sh
//! cargo run --release --features vulkan -- --device vulkan --runs 10
//! cargo run --release --features vulkan -- --device vulkan --runs 10 --sync
//! ```
//!
//! Flags: `--device vulkan|wgpu|cuda|metal|flex` (default `vulkan`),
//! `--tensors N` (default 400) tensors of 1 to `--max-mib M` (default 16) MiB
//! each, `--steps N` (default 200) decode-like steps over `--layers N`
//! (default 24) growing caches, `--tail N` (default 4000) allocations queued
//! after the weights are freed, `--sync`, `--runs N`.

use std::{
    process::{
        Command,
        ExitCode,
    },
    time::Instant,
};

use burn::tensor::{
    Device,
    Distribution,
    Tensor,
    TensorData,
};

struct Args {
    device: String,
    tensors: usize,
    max_mib: usize,
    steps: usize,
    layers: usize,
    tail: usize,
    sync: bool,
    runs: Option<usize>,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        device: "vulkan".into(),
        tensors: 400,
        max_mib: 16,
        steps: 200,
        layers: 24,
        tail: 4000,
        sync: false,
        runs: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--device" => args.device = value()?,
            "--tensors" => {
                args.tensors = value()?.parse().map_err(|e| format!("--tensors: {e}"))?
            }
            "--max-mib" => {
                args.max_mib = value()?.parse().map_err(|e| format!("--max-mib: {e}"))?
            }
            "--tail" => args.tail = value()?.parse().map_err(|e| format!("--tail: {e}"))?,
            "--steps" => args.steps = value()?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--layers" => args.layers = value()?.parse().map_err(|e| format!("--layers: {e}"))?,
            "--runs" => args.runs = Some(value()?.parse().map_err(|e| format!("--runs: {e}"))?),
            "--sync" => args.sync = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

fn device(name: &str) -> Result<Device, String> {
    #[allow(unused_imports)]
    use burn::tensor::DeviceKind;
    match name {
        "flex" => Ok(Device::flex()),
        #[cfg(feature = "vulkan")]
        "vulkan" => Ok(Device::vulkan(DeviceKind::DefaultDevice)),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(Device::wgpu(DeviceKind::DefaultDevice)),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::metal(DeviceKind::DefaultDevice)),
        other => Err(format!(
            "device {other} is not built in (features: vulkan, wgpu, cuda, metal; flex always)"
        )),
    }
}

/// One run: load weight-sized tensors, use them, print a result, free them,
/// and return, as a model-serving command does.
fn child(args: &Args) -> Result<(), String> {
    let device = device(&args.device)?;

    // The "weights", uploaded from host memory as a checkpoint's are:
    // `tensors` f32 tensors of 1 to `max_mib` MiB, then one embedding-sized
    // 200 MiB tensor; about 3.6 GB at the defaults (Whisper `medium` is 3 GB).
    //
    // The order matters. The memory pools grow their page size to the
    // largest allocation, and each growth retires the pool before it: the
    // weights allocated before the 200 MiB tensor sit in retired pools, which
    // give a page back to the driver (`destroy_buffer`) the moment it empties,
    // on the next reserve.
    let upload = |mib: usize| {
        let n = mib << 18;
        Tensor::<1>::from_data(TensorData::new(vec![0.5f32; n], [n]), &device)
    };
    let mut weights: Vec<Tensor<1>> = (0..args.tensors)
        .map(|i| upload(i % args.max_mib + 1))
        .collect();
    weights.push(upload(200));

    // The "decode": `steps` steps over `layers` layers, each layer's cache
    // growing by a row a step, as a KV cache does by concatenation, with a
    // read back per step, as a token is. This fills the memory pools with
    // pages of many sizes.
    const WIDTH: usize = 1024;
    let mut caches: Vec<Option<Tensor<2>>> = vec![None; args.layers];
    let mut token = 0.0f32;
    for _ in 0..args.steps {
        let mut x = Tensor::<2>::random([1, WIDTH], Distribution::Default, &device);
        for cache in caches.iter_mut() {
            let grown = match cache.take() {
                Some(c) => Tensor::cat(vec![c, x.clone()], 0),
                None => x.clone(),
            };
            x = (grown.clone() * x).sum_dim(0) / WIDTH as f32;
            *cache = Some(grown);
        }
        token += x.slice([0..1, 0..1]).into_scalar::<f32>();
    }

    // The "transcript": a result read back, and written to stdout.
    let total: f32 = weights
        .iter()
        .map(|w| w.clone().mean().into_scalar::<f32>())
        .sum();
    println!(
        "{} tensors, {} steps over {} layers, {:.3} {token:.3}",
        weights.len(),
        args.steps,
        args.layers,
        total / weights.len() as f32
    );
    drop(caches);

    // Free the weights, then queue `tail` small allocations and hand them to
    // the device thread without waiting (`flush`, not `sync`). Each
    // allocation reserves, every reserve runs the pools' periodic cleanup,
    // and enough of them pass the cleanup period of the freed weights' pages:
    // the device thread is releasing those pages (`destroy_buffer`) as `main`
    // returns.
    drop(weights);
    let tail: Vec<Tensor<1>> = (0..args.tail)
        .map(|i| Tensor::zeros([256 + i % 256], &device) + 1.0)
        .collect();
    device.flush().map_err(|e| e.to_string())?;

    if args.sync {
        device.sync().map_err(|e| e.to_string())?;
    }
    drop(tail);
    Ok(())
}

/// `runs` children with piped output; how many died of a signal.
#[cfg(unix)]
fn parent(
    args: &Args,
    runs: usize,
) -> Result<usize, String> {
    use std::os::unix::process::ExitStatusExt;

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut flags = vec![
        "--device".to_string(),
        args.device.clone(),
        "--tensors".into(),
        args.tensors.to_string(),
        "--max-mib".into(),
        args.max_mib.to_string(),
        "--steps".into(),
        args.steps.to_string(),
        "--layers".into(),
        args.layers.to_string(),
        "--tail".into(),
        args.tail.to_string(),
    ];
    if args.sync {
        flags.push("--sync".into());
    }
    let mut crashed = 0;
    for i in 0..runs {
        let t0 = Instant::now();
        let out = Command::new(&exe)
            .args(&flags)
            .output()
            .map_err(|e| e.to_string())?;
        let printed = !out.stdout.is_empty();
        let outcome = match (out.status.code(), out.status.signal()) {
            (_, Some(signal)) => {
                crashed += 1;
                format!("killed by signal {signal}")
            }
            (Some(code), _) => format!("exit {code}"),
            _ => "unknown".to_string(),
        };
        println!(
            "run {i}: {outcome}, stdout written: {printed}, {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "{crashed} of {runs} crashed (device {}, sync {})",
        args.device, args.sync
    );
    Ok(crashed)
}

#[cfg(not(unix))]
fn parent(
    _args: &Args,
    _runs: usize,
) -> Result<usize, String> {
    Err("--runs needs a unix host".into())
}

fn main() -> ExitCode {
    let result = parse().and_then(|args| match args.runs {
        Some(runs) => parent(&args, runs).map(|crashed| crashed > 0),
        None => child(&args).map(|()| false),
    });
    match result {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
