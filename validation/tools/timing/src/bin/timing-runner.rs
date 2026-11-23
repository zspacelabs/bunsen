//! The timing runner: built for one backend by `cargo timing run`, it loads
//! each row through its kit's factory and times its decode's phases.
//!
//! Not run by hand as a rule (`cargo timing run` builds it with the backend's
//! feature and passes the rows); it can be:
//!
//! ```sh
//! cargo run --profile validate -p timing --bin timing-runner --features cuda -- \
//!     --row qwen3:qwen3-0.6b --dtype f32 --out target/timing/run.jsonl
//! ```

use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::Arc,
    time::{
        Instant,
        SystemTime,
        UNIX_EPOCH,
    },
};

use bunsen::{
    data::{
        builds::{
            BuildFactory,
            ConstructInit,
        },
        formats::LoadPolicy,
        resources::{
            ResourceCache,
            ResourceCacheOptions,
        },
    },
    kits::{
        language::generation::{
            gemma::pretrained::default_gemma_factory,
            nanochat::pretrained::default_nanochat_factory,
            qwen3::pretrained::default_qwen3_factory,
            qwen3_5::pretrained::default_qwen3_5_factory,
        },
        support::{
            construct::LmConstruct,
            timing::TimedDecode,
        },
    },
    ops::transformers::attention::KVCacheStorage,
    support::testing::performance_device,
};
use burn::tensor::{
    Device,
    FloatDType,
    TensorData,
    read_sync,
};
use clap::{
    Parser,
    ValueEnum,
};
use timing::{
    Bucket,
    Memory,
    Phase,
    Stats,
    TimingRecord,
    Window,
};
use validate::schedule::Dtype;

/// Times each row's decode, phase by phase, appending a JSON line per row
/// to `--out`.
#[derive(Parser, Debug)]
#[command(name = "timing-runner")]
struct Args {
    /// A row to time, `crate:model` as `validation/models.yml` names them
    /// (`qwen3:qwen3-0.6b`); the model may be a full catalog name.
    #[arg(long = "row", required = true)]
    rows_to_time: Vec<String>,

    /// The dtype every row loads in.
    #[arg(long, value_enum, default_value = "f32")]
    dtype: Dtype,

    /// Rows decoded at once.
    #[arg(long, default_value_t = 1)]
    rows: usize,

    /// Positions each row's prompt prefills.
    #[arg(long, default_value_t = 128)]
    prefill: usize,

    /// Steps each row decodes after the prompt.
    #[arg(long, default_value_t = 128)]
    steps: usize,

    /// Warm prefills timed after the cold one, each into a fresh state.
    #[arg(long, default_value_t = 3)]
    prefill_reps: usize,

    /// Groups the warm steps are reported in, by cache position.
    #[arg(long, default_value_t = 4)]
    buckets: usize,

    /// The KV cache's storage.
    #[arg(long, value_enum, default_value = "auto")]
    storage: Storage,

    /// What the run is labeled in its records.
    #[arg(long, default_value = "")]
    label: String,

    /// The JSON-lines file each row's record is appended to.
    #[arg(long)]
    out: PathBuf,
}

/// [`KVCacheStorage`], by name.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Storage {
    Auto,
    Concat,
    PerLayer,
    MultiLayer,
}

impl Storage {
    fn storage(self) -> KVCacheStorage {
        match self {
            Storage::Auto => KVCacheStorage::Auto,
            Storage::Concat => KVCacheStorage::Concat,
            Storage::PerLayer => KVCacheStorage::PerLayer,
            Storage::MultiLayer => KVCacheStorage::MultiLayer,
        }
    }
}

fn main() {
    let args = Args::parse();
    let device = performance_device();
    let cache = ResourceCache::new(
        ResourceCacheOptions::default()
            .with_offline(std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some()),
    )
    .expect("open the resource cache");
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir).expect("make the output directory");
    }
    let mut failed = false;
    for row in &args.rows_to_time {
        match time_row(row, &args, &cache, &device) {
            Ok(record) => {
                print_record(&record);
                let mut out = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&args.out)
                    .expect("open the output file");
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string(&record).expect("a record serializes")
                )
                .expect("write the record");
            }
            Err(err) => {
                eprintln!("{row}: {err}");
                failed = true;
            }
        }
        device.memory_cleanup();
    }
    if failed {
        std::process::exit(1);
    }
}

/// Loads `row` through its kit's factory and times it.
fn time_row(
    row: &str,
    args: &Args,
    cache: &ResourceCache,
    device: &Device,
) -> Result<TimingRecord, String> {
    let (krate, model) = row
        .split_once(':')
        .ok_or_else(|| format!("a row is `crate:model`, not `{row}`"))?;
    let lm = |e: bunsen::errors::BunsenError| e.to_string();
    match krate {
        "qwen3" => time_lm(
            krate,
            model,
            &default_qwen3_factory().map_err(lm)?,
            args,
            cache,
            device,
        ),
        "qwen3-5" => time_lm(
            krate,
            model,
            &default_qwen3_5_factory().map_err(lm)?,
            args,
            cache,
            device,
        ),
        "gemma" => time_lm(
            krate,
            model,
            &default_gemma_factory().map_err(lm)?,
            args,
            cache,
            device,
        ),
        "nanochat" => time_lm(
            krate,
            model,
            &default_nanochat_factory().map_err(lm)?,
            args,
            cache,
            device,
        ),
        other => Err(format!(
            "{other} has no timing adapter yet (qwen3, qwen3-5, gemma and nanochat do)"
        )),
    }
}

/// The catalog build `model` names in `factory`: itself when it is one, else
/// the build whose last `/` part it is (`models.yml`'s names).
fn build_id<H: ConstructInit>(
    factory: &BuildFactory<H>,
    model: &str,
) -> Result<String, String> {
    let ids = factory.ids();
    ids.iter()
        .find(|id| id.as_str() == model)
        .or_else(|| ids.iter().find(|id| id.rsplit('/').next() == Some(model)))
        .cloned()
        .ok_or_else(|| format!("no catalog build is named {model}"))
}

/// A language model's row: loaded by `factory`, timed through its
/// [`TimedDecode`].
fn time_lm<H, M>(
    krate: &str,
    model: &str,
    factory: &BuildFactory<H>,
    args: &Args,
    cache: &ResourceCache,
    device: &Device,
) -> Result<TimingRecord, String>
where
    H: ConstructInit<Construct = LmConstruct<M>>,
    M: TimedDecode + Sync,
    M::State: Send,
{
    let build = build_id(factory, model)?;
    let policy = match args.dtype {
        Dtype::F32 => LoadPolicy::min_precision(FloatDType::F32),
        Dtype::Bf16 => LoadPolicy::performance(),
    };
    let start = Instant::now();
    let construct: Arc<LmConstruct<M>> = factory
        .load_construct(&build, cache, device, policy)
        .map_err(|e| format!("load {build}: {e}"))?;
    let load_s = start.elapsed().as_secs_f64();
    let timed = time_decode(&construct.model, args, device);
    let storage = args.storage.storage();
    let resolved = match storage {
        KVCacheStorage::Auto => KVCacheStorage::auto_for(device),
        other => other,
    };
    Ok(TimingRecord {
        krate: krate.to_string(),
        model: model.to_string(),
        build,
        dtype: args.dtype.name().to_string(),
        backend: backend_stack(device),
        storage: format!("{storage:?} -> {resolved:?}"),
        label: args.label.clone(),
        rows: args.rows,
        prefill: args.prefill,
        steps: args.steps,
        unit: construct.model.step_unit().to_string(),
        load_s,
        ..timed
    })
}

/// The backend stack the device dispatches to: burn's name for it.
fn backend_stack(device: &Device) -> String {
    use burn_backend::Backend;
    burn_dispatch::Dispatch::name(device.as_dispatch())
}

/// The largest of the device pool's readings so far.
fn sample_memory(
    device: &Device,
    memory: &mut Option<Memory>,
) {
    if let Some(usage) = device.memory_pool_usage() {
        let m = memory.get_or_insert_with(Memory::default);
        m.bytes_in_use = m.bytes_in_use.max(usage.bytes_in_use);
        m.bytes_reserved = m.bytes_reserved.max(usage.bytes_reserved);
    }
}

/// `f` in one window: from an idle device to `f`'s read, on the wall clock
/// and the device's.
fn window(
    device: &Device,
    f: impl FnOnce() -> TensorData + Send,
) -> Window {
    device.sync().expect("sync the device");
    let start = Instant::now();
    let (_read, duration) = device.profile(f).expect("open a profiling window");
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let device_ms =
        read_sync(duration.resolve()).map(|ticks| ticks.duration().as_secs_f64() * 1000.0);
    Window { wall_ms, device_ms }
}

/// The phases of `model`'s decode: a cold prefill and its first step, the
/// warm steps after them, then warm prefills into fresh states. Every window
/// closes on a read of the phase's tokens (the argmax of its logits), as a
/// decode reads its token. The load fields are left for the caller.
fn time_decode<M>(
    model: &M,
    args: &Args,
    device: &Device,
) -> TimingRecord
where
    M: TimedDecode + Sync,
    M::State: Send,
{
    let (rows, prefill, steps) = (args.rows, args.prefill, args.steps);
    let positions = prefill + steps + 1;
    let storage = args.storage.storage();
    let mut memory = None;

    let mut state = model.timed_state(rows, positions, storage, device);
    let prefill_cold = window(device, || {
        model
            .timed_prefill(&mut state, rows, prefill, device)
            .argmax(1)
            .into_data()
    });
    sample_memory(device, &mut memory);
    let step_cold = window(device, || {
        model
            .timed_step(&mut state, rows, device)
            .argmax(1)
            .into_data()
    });
    let steps_warm: Vec<Window> = (1..steps)
        .map(|_| {
            window(device, || {
                model
                    .timed_step(&mut state, rows, device)
                    .argmax(1)
                    .into_data()
            })
        })
        .collect();
    sample_memory(device, &mut memory);
    drop(state);

    let prefills_warm: Vec<Window> = (0..args.prefill_reps)
        .map(|_| {
            let mut state = model.timed_state(rows, positions, storage, device);
            window(device, || {
                model
                    .timed_prefill(&mut state, rows, prefill, device)
                    .argmax(1)
                    .into_data()
            })
        })
        .collect();
    sample_memory(device, &mut memory);

    // Warm step `i` ran with `prefill + 1 + i` positions cached before it.
    let first = prefill + 1;
    let per = steps_warm.len().div_ceil(args.buckets.max(1)).max(1);
    let step_buckets = steps_warm
        .chunks(per)
        .enumerate()
        .map(|(b, chunk)| Bucket {
            from: first + b * per,
            to: first + b * per + chunk.len(),
            wall: Stats::of(chunk.iter().map(|w| w.wall_ms)),
        })
        .collect();

    TimingRecord {
        krate: String::new(),
        model: String::new(),
        build: String::new(),
        dtype: String::new(),
        backend: String::new(),
        storage: String::new(),
        label: String::new(),
        rows,
        prefill,
        steps,
        unit: String::new(),
        load_s: 0.0,
        prefill_cold,
        prefill_warm: Phase::of(&prefills_warm),
        step_cold,
        step_warm: Phase::of(&steps_warm),
        step_buckets,
        step_wall_ms: steps_warm.iter().map(|w| w.wall_ms).collect(),
        memory,
    }
}

/// `record`, as a table on stdout.
fn print_record(r: &TimingRecord) {
    let device = |ms: Option<f64>| ms.map_or("-".to_string(), |ms| format!("{ms:.2}"));
    let phase_device = |p: &Phase| p.device.map(|d| d.median_ms);
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    println!(
        "{} ({}) {} rows={} prefill={} steps={} [{}] storage {} @{unix}",
        r.build, r.krate, r.dtype, r.rows, r.prefill, r.steps, r.backend, r.storage
    );
    println!("  {:<14} {:>10} {:>10}", "phase", "wall ms", "device ms*");
    println!("  {:<14} {:>10.1} s", "load", r.load_s);
    println!(
        "  {:<14} {:>10.2} {:>10}",
        "prefill cold",
        r.prefill_cold.wall_ms,
        device(r.prefill_cold.device_ms)
    );
    println!(
        "  {:<14} {:>10.2} {:>10}   median of {}, {:.0} {}/s",
        "prefill warm",
        r.prefill_warm.wall.median_ms,
        device(phase_device(&r.prefill_warm)),
        r.prefill_warm.count,
        r.prefill_per_s(),
        r.unit
    );
    println!(
        "  {:<14} {:>10.2} {:>10}",
        "step cold",
        r.step_cold.wall_ms,
        device(r.step_cold.device_ms)
    );
    println!(
        "  {:<14} {:>10.2} {:>10}   median of {}, {:.1} {}/s (min {:.2}, max {:.2})",
        "step warm",
        r.step_warm.wall.median_ms,
        device(phase_device(&r.step_warm)),
        r.step_warm.count,
        r.decode_per_s(),
        r.unit,
        r.step_warm.wall.min_ms,
        r.step_warm.wall.max_ms
    );
    for b in &r.step_buckets {
        println!(
            "    positions {:>5}..{:<5} {:>8.2}",
            b.from, b.to, b.wall.median_ms
        );
    }
    let slowest = r
        .step_wall_ms
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1));
    if let Some((i, ms)) = slowest {
        println!(
            "    slowest warm step: {ms:.2} ms, with {} positions cached",
            r.prefill + 1 + i
        );
    }
    if let Some(m) = r.memory {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        println!(
            "  memory         {:.2} GiB in use, {:.2} GiB reserved (largest reading)",
            gib(m.bytes_in_use),
            gib(m.bytes_reserved)
        );
    }
    println!("  * burn's device window; not pure kernel time (it reads above wall)");
}
