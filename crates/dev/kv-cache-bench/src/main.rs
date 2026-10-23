//! # KV cache growth benchmark
//!
//! Times a decode-shaped loop under each way of storing a self-attention KV
//! cache's history: how [`KVCacheStorage::auto_for`]'s table is measured,
//! and how it is retuned as machines and burn change.
//!
//! The library's own [`KVCache`], pinned to each storage:
//!
//! - `auto`: [`KVCacheStorage::Auto`], whatever it resolves to here.
//! - `concat`: [`KVCacheStorage::Concat`], one `Tensor::cat` per layer per
//!   step; it copies the history every step.
//! - `per-layer`: [`KVCacheStorage::PerLayer`], per-layer preallocated buffers,
//!   written in place.
//! - `multi-layer`: [`KVCacheStorage::MultiLayer`], one preallocated `[layers,
//!   2, batch, heads, seq, head_dim]` buffer.
//!
//! And this benchmark's own copies, whose writes it can check (the
//! library's buffers are private): before each write to a preallocated
//! buffer, [`Tensor::can_mut`] says whether the buffer is uniquely owned, so
//! the backend may write it in place; the "in-place writes" column counts the
//! writes that were, and `--require-in-place` panics on the first that isn't.
//!
//! - `checked-per-layer`: `per-layer`, written through [`Tensor::inplace`].
//! - `checked-per-layer-clone`: the same, written on a `clone()` of the buffer,
//!   which keeps a second reference alive and forces a copy.
//! - `checked-multi-layer`: `multi-layer`'s scheme.
//!
//! Each step appends one position to every layer, attends over the layer's
//! history (scores, softmax, values), feeds the result to the next layer, and
//! reads a value back at the end of the step, as a decode reads its token.
//! The geometry defaults to Whisper `base`'s decoder (6 layers, 8 heads of
//! 64) over its 448-token text context.
//!
//! ```sh
//! cargo run --release -p kv-cache-bench --features vulkan -- --device vulkan
//! ```

use std::time::{
    Duration,
    Instant,
};

use bunsen::ops::transformers::attention::{
    KVCache,
    KVCacheConfig,
    KVCacheStorage,
};
use bunsen_app::device::BackendArgs;
use burn::{
    Tensor,
    prelude::s,
    tensor::{
        Device,
        Distribution,
        activation::softmax,
    },
};
use clap::{
    Parser,
    ValueEnum,
};

/// How a layer's history grows.
/// What runs on the device before each layer's write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SyncMode {
    /// Nothing: queued work may still hold the buffer's handle.
    None,
    /// `Device::sync`: wait for queued work, so it drops its handles.
    Sync,
    /// `Device::flush`: hand queued work to the device, without waiting.
    Flush,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Strategy {
    /// The library's `KVCache`, `KVCacheStorage::Auto`.
    Auto,
    /// The library's `KVCache`, `KVCacheStorage::Concat`.
    Concat,
    /// The library's `KVCache`, `KVCacheStorage::PerLayer`.
    PerLayer,
    /// The library's `KVCache`, `KVCacheStorage::MultiLayer`.
    MultiLayer,
    /// Per-layer buffers, written through `Tensor::inplace`, checked.
    CheckedPerLayer,
    /// Per-layer buffers, written on a clone, checked.
    CheckedPerLayerClone,
    /// The single multi-layer buffer, checked.
    CheckedMultiLayer,
}

impl Strategy {
    /// The library storage this strategy pins, if it's the library's cache.
    fn storage(self) -> Option<KVCacheStorage> {
        match self {
            Self::Auto => Some(KVCacheStorage::Auto),
            Self::Concat => Some(KVCacheStorage::Concat),
            Self::PerLayer => Some(KVCacheStorage::PerLayer),
            Self::MultiLayer => Some(KVCacheStorage::MultiLayer),
            _ => None,
        }
    }
}

/// Times decode-shaped KV cache growth.
#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    backend: BackendArgs,

    /// The strategies to time.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [Strategy::Auto, Strategy::Concat, Strategy::PerLayer, Strategy::MultiLayer, Strategy::CheckedPerLayer, Strategy::CheckedPerLayerClone, Strategy::CheckedMultiLayer])]
    strategies: Vec<Strategy>,

    /// Batch sizes to time (a beam search's beams are its batch).
    #[arg(long, value_delimiter = ',', default_values_t = [1, 5])]
    batches: Vec<usize>,

    /// Decoder layers.
    #[arg(long, default_value_t = 6)]
    layers: usize,

    /// KV heads.
    #[arg(long, default_value_t = 8)]
    heads: usize,

    /// Head width.
    #[arg(long, default_value_t = 64)]
    head_dim: usize,

    /// Decode steps (positions).
    #[arg(long, default_value_t = 448)]
    steps: usize,

    /// Timed repetitions of each configuration, after one untimed warm-up;
    /// round-robin across configurations, and reported by their median.
    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// Panic on the first preallocated write whose buffer isn't uniquely
    /// owned (`Tensor::can_mut` false), rather than counting it.
    #[arg(long)]
    require_in_place: bool,

    /// What runs before each layer's write: queued work can hold a buffer's
    /// handle, and so make the write copy.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [SyncMode::None])]
    sync_modes: Vec<SyncMode>,
}

/// One timed run.
struct Run {
    elapsed: Duration,
    /// Each step's time; a step syncs, reading its token back.
    steps: Vec<Duration>,
    peak_in_use: Option<u64>,
    peak_reserved: Option<u64>,
    /// `(in place, checked)` writes; `None` for a strategy that isn't checked.
    in_place: Option<(u64, u64)>,
    /// The library cache's storage, as it resolved.
    storage: Option<KVCacheStorage>,
}

impl Run {
    /// Mean milliseconds per step over `range` of the decode, as fractions.
    fn ms_per_step(
        &self,
        from: f64,
        to: f64,
    ) -> f64 {
        let n = self.steps.len();
        let (a, b) = (
            (n as f64 * from) as usize,
            ((n as f64 * to) as usize).max(1),
        );
        let span = &self.steps[a..b.max(a + 1).min(n)];
        span.iter().map(Duration::as_secs_f64).sum::<f64>() * 1000.0 / span.len() as f64
    }
}

/// Counts the preallocated writes whose buffer [`Tensor::can_mut`] says is
/// uniquely owned.
struct InPlace {
    require: bool,
    in_place: u64,
    checked: u64,
}

impl InPlace {
    /// Notes a write to `buffer`, before it happens.
    fn check<const D: usize>(
        &mut self,
        what: &str,
        buffer: &Tensor<D>,
    ) {
        self.checked += 1;
        if buffer.can_mut() {
            self.in_place += 1;
        } else if self.require {
            panic!(
                "{what}: write {} would copy: the buffer has another reference",
                self.checked
            );
        }
    }
}

/// The history store a strategy grows.
enum History {
    KvCache(KVCache),
    PerLayer {
        keys: Vec<Tensor<4>>,
        values: Vec<Tensor<4>>,
        pos: usize,
        inplace: bool,
    },
    MultiLayer {
        buffer: Option<Tensor<6>>,
        layers: usize,
        pos: usize,
    },
}

impl History {
    fn new(
        strategy: Strategy,
        args: &Args,
        batch: usize,
        device: &Device,
    ) -> Self {
        if let Some(storage) = strategy.storage() {
            return Self::KvCache(
                KVCacheConfig::new(batch, args.heads, args.steps, args.head_dim, args.layers)
                    .with_max_seq_len(Some(args.steps))
                    .with_storage(storage)
                    .init(),
            );
        }
        match strategy {
            Strategy::CheckedPerLayer | Strategy::CheckedPerLayerClone => {
                let shape = [batch, args.heads, args.steps, args.head_dim];
                Self::PerLayer {
                    keys: (0..args.layers)
                        .map(|_| Tensor::zeros(shape, device))
                        .collect(),
                    values: (0..args.layers)
                        .map(|_| Tensor::zeros(shape, device))
                        .collect(),
                    pos: 0,
                    inplace: strategy == Strategy::CheckedPerLayer,
                }
            }
            Strategy::CheckedMultiLayer => Self::MultiLayer {
                buffer: Some(Tensor::zeros(
                    [args.layers, 2, batch, args.heads, args.steps, args.head_dim],
                    device,
                )),
                layers: args.layers,
                pos: 0,
            },
            _ => unreachable!("the library's strategies return above"),
        }
    }

    /// Appends one position's `(k, v)` to `layer`; the layer's whole history.
    fn insert(
        &mut self,
        layer: usize,
        k: Tensor<4>,
        v: Tensor<4>,
        check: &mut InPlace,
    ) -> (Tensor<4>, Tensor<4>) {
        match self {
            Self::KvCache(cache) => cache.insert_kv(layer, k, v),
            Self::PerLayer {
                keys,
                values,
                pos,
                inplace,
            } => {
                let t = *pos;
                if *inplace {
                    keys[layer].inplace(|key| {
                        check.check("per-layer-inplace key", &key);
                        key.slice_assign(s![.., .., t..t + 1], k)
                    });
                    values[layer].inplace(|value| {
                        check.check("per-layer-inplace value", &value);
                        value.slice_assign(s![.., .., t..t + 1], v)
                    });
                } else {
                    let key = keys[layer].clone();
                    check.check("per-layer key", &key);
                    keys[layer] = key.slice_assign(s![.., .., t..t + 1], k);
                    let value = values[layer].clone();
                    check.check("per-layer value", &value);
                    values[layer] = value.slice_assign(s![.., .., t..t + 1], v);
                }
                if layer + 1 == keys.len() {
                    *pos = t + 1;
                }
                (
                    keys[layer].clone().slice(s![.., .., ..t + 1]),
                    values[layer].clone().slice(s![.., .., ..t + 1]),
                )
            }
            Self::MultiLayer {
                buffer,
                layers,
                pos,
            } => {
                let t = *pos;
                let mut b = buffer.take().expect("the buffer is in its slot");
                check.check("multi-layer", &b);
                b = b.slice_assign(s![layer, 0, .., .., t..t + 1], k.unsqueeze());
                check.check("multi-layer", &b);
                b = b.slice_assign(s![layer, 1, .., .., t..t + 1], v.unsqueeze());
                let key = b
                    .clone()
                    .slice(s![layer, 0, .., .., ..t + 1])
                    .squeeze_dims::<4>(&[0, 1]);
                let value = b
                    .clone()
                    .slice(s![layer, 1, .., .., ..t + 1])
                    .squeeze_dims::<4>(&[0, 1]);
                *buffer = Some(b);
                if layer + 1 == *layers {
                    *pos = t + 1;
                }
                (key, value)
            }
        }
    }
}

fn run(
    strategy: Strategy,
    sync: SyncMode,
    args: &Args,
    batch: usize,
    device: &Device,
) -> Run {
    let scale = (args.head_dim as f64).sqrt().recip();
    let step = [batch, args.heads, 1, args.head_dim];
    let mut history = History::new(strategy, args, batch, device);
    let mut check = InPlace {
        require: args.require_in_place,
        in_place: 0,
        checked: 0,
    };
    let usage = || device.memory_pool_usage();
    let (mut peak_in_use, mut peak_reserved) = (None::<u64>, None::<u64>);
    let mut steps = Vec::with_capacity(args.steps);

    let mut x = Tensor::<4>::random(step, Distribution::Default, device);
    device.sync().expect("the device syncs");
    let t0 = Instant::now();
    for _ in 0..args.steps {
        let ts = Instant::now();
        for layer in 0..args.layers {
            let (k, v) = (x.clone(), x.clone() * 0.5);
            match sync {
                SyncMode::None => {}
                SyncMode::Sync => device.sync().expect("the device syncs"),
                SyncMode::Flush => device.flush().expect("the device flushes"),
            }
            let (k, v) = history.insert(layer, k, v, &mut check);
            let scores = x.clone().matmul(k.swap_dims(2, 3)) * scale;
            x = softmax(scores, 3).matmul(v);
        }
        // A decode reads its token back each step.
        let _token: f32 = x.clone().slice(s![0..1, 0..1, 0..1, 0..1]).into_scalar();
        steps.push(ts.elapsed());
        if let Some(u) = usage() {
            peak_in_use = Some(peak_in_use.unwrap_or(0).max(u.bytes_in_use));
            peak_reserved = Some(peak_reserved.unwrap_or(0).max(u.bytes_reserved));
        }
    }
    device.sync().expect("the device syncs");
    let elapsed = t0.elapsed();
    let storage = match &history {
        History::KvCache(cache) => Some(cache.storage()),
        _ => None,
    };
    drop(history);
    device.memory_cleanup();
    Run {
        elapsed,
        steps,
        peak_in_use,
        peak_reserved,
        in_place: (check.checked > 0).then_some((check.in_place, check.checked)),
        storage,
    }
}

fn mib(bytes: Option<u64>) -> String {
    bytes.map_or("-".to_string(), |b| {
        format!("{:.1}", b as f64 / (1 << 20) as f64)
    })
}

fn main() {
    let args = Args::parse();
    let device = args.backend.open().unwrap_or_else(|e| panic!("{e}"));
    println!(
        "device {:?}, Auto is {:?}, {} layers x {} heads x {} head_dim, {} steps",
        *device,
        KVCacheStorage::auto_for(&device),
        args.layers,
        args.heads,
        args.head_dim,
        args.steps
    );
    println!(
        "| strategy | sync | batch | median ms | range ms (spread) | ms/step | first quarter ms/step | last quarter ms/step | last / first | peak in use MiB | peak reserved MiB | in-place writes |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &batch in &args.batches {
        let configs: Vec<(Strategy, SyncMode)> = args
            .strategies
            .iter()
            .flat_map(|&strategy| args.sync_modes.iter().map(move |&sync| (strategy, sync)))
            .collect();
        // One warm-up each, then the repetitions round-robin, each round
        // starting one configuration later, so order and warm-up effects
        // spread evenly rather than favouring whichever runs first.
        for &(strategy, sync) in &configs {
            run(strategy, sync, &args, batch, &device);
        }
        let mut runs: Vec<Vec<Run>> = configs.iter().map(|_| Vec::new()).collect();
        for rep in 0..args.reps {
            for i in 0..configs.len() {
                let c = (i + rep) % configs.len();
                let (strategy, sync) = configs[c];
                runs[c].push(run(strategy, sync, &args, batch, &device));
            }
        }
        for (&(strategy, sync), mut runs) in configs.iter().zip(runs) {
            runs.sort_by_key(|r| r.elapsed);
            let ms = |r: &Run| r.elapsed.as_secs_f64() * 1000.0;
            let median = &runs[runs.len() / 2];
            let (min, max) = (ms(&runs[0]), ms(&runs[runs.len() - 1]));
            let (first, last) = (median.ms_per_step(0.0, 0.25), median.ms_per_step(0.75, 1.0));
            println!(
                "| {} | {sync:?} | {batch} | {:.1} | {min:.1}-{max:.1} ({:.0}%) | {:.3} | {first:.3} | {last:.3} | {:.2} | {} | {} | {} |",
                match (strategy, median.storage) {
                    (Strategy::Auto, Some(storage)) => format!("Auto ({storage:?})"),
                    _ => format!("{strategy:?}"),
                },
                ms(median),
                100.0 * (max - min) / ms(median),
                ms(median) / args.steps as f64,
                last / first,
                mib(median.peak_in_use),
                mib(median.peak_reserved),
                median.in_place.map_or("-".to_string(), |(n, of)| format!(
                    "{:.0}% ({n} of {of})",
                    100.0 * n as f64 / of as f64
                )),
            );
        }
    }
}
