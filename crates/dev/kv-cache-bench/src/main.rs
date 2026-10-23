//! # KV cache growth benchmark
//!
//! Times a decode-shaped loop under three ways of growing a self-attention
//! KV cache, to decide how bunsen's unified cache should store its history:
//!
//! - `concat`: Whisper's today: each layer's [`AttnKvPair`] grows by
//!   [`concat`](AttnKvPair::concat), one `Tensor::cat` per step, so it copies
//!   the whole history every step.
//! - `kv-cache`: the shared [`KVCache`]: one preallocated `[layers, 2, batch,
//!   heads, seq, head_dim]` tensor, written by `slice_assign`. Cheap only if
//!   `slice_assign` updates in place; if a backend copies, it copies every
//!   layer's storage on every write.
//! - `per-layer`: per-layer preallocated key and value tensors, written by
//!   `slice_assign`: the fallback if the single buffer copies.
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
    AttnKvPair,
    KVCache,
    KVCacheConfig,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Strategy {
    /// `AttnKvPair::concat`, one `Tensor::cat` per layer per step.
    Concat,
    /// The shared multi-layer `KVCache`.
    KvCache,
    /// Per-layer preallocated buffers, written by `slice_assign`.
    PerLayer,
}

/// Times decode-shaped KV cache growth.
#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    backend: BackendArgs,

    /// The strategies to time.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [Strategy::Concat, Strategy::KvCache, Strategy::PerLayer])]
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

    /// Timed repetitions, after one untimed warm-up.
    #[arg(long, default_value_t = 3)]
    reps: usize,
}

/// One timed run.
struct Run {
    elapsed: Duration,
    /// Each step's time; a step syncs, reading its token back.
    steps: Vec<Duration>,
    peak_in_use: Option<u64>,
    peak_reserved: Option<u64>,
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

/// The history store a strategy grows.
enum History {
    Concat(Vec<Option<AttnKvPair>>),
    KvCache(KVCache),
    PerLayer {
        keys: Vec<Tensor<4>>,
        values: Vec<Tensor<4>>,
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
        match strategy {
            Strategy::Concat => Self::Concat(vec![None; args.layers]),
            Strategy::KvCache => Self::KvCache(
                KVCacheConfig::new(batch, args.heads, args.steps, args.head_dim, args.layers)
                    .with_max_seq_len(Some(args.steps))
                    .init(),
            ),
            Strategy::PerLayer => {
                let shape = [batch, args.heads, args.steps, args.head_dim];
                Self::PerLayer {
                    keys: (0..args.layers)
                        .map(|_| Tensor::zeros(shape, device))
                        .collect(),
                    values: (0..args.layers)
                        .map(|_| Tensor::zeros(shape, device))
                        .collect(),
                    pos: 0,
                }
            }
        }
    }

    /// Appends one position's `(k, v)` to `layer`; the layer's whole history.
    fn insert(
        &mut self,
        layer: usize,
        k: Tensor<4>,
        v: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        match self {
            Self::Concat(pairs) => {
                let step = AttnKvPair { key: k, value: v };
                let grown = match pairs[layer].take() {
                    Some(pair) => pair.concat(step),
                    None => step,
                };
                let out = (grown.key.clone(), grown.value.clone());
                pairs[layer] = Some(grown);
                out
            }
            Self::KvCache(cache) => cache.insert_kv(layer, k, v),
            Self::PerLayer { keys, values, pos } => {
                let t = *pos;
                let key = keys[layer].clone().slice_assign(s![.., .., t..t + 1], k);
                let value = values[layer].clone().slice_assign(s![.., .., t..t + 1], v);
                keys[layer] = key.clone();
                values[layer] = value.clone();
                if layer + 1 == keys.len() {
                    *pos = t + 1;
                }
                (
                    key.slice(s![.., .., ..t + 1]),
                    value.slice(s![.., .., ..t + 1]),
                )
            }
        }
    }
}

fn run(
    strategy: Strategy,
    args: &Args,
    batch: usize,
    device: &Device,
) -> Run {
    let scale = (args.head_dim as f64).sqrt().recip();
    let step = [batch, args.heads, 1, args.head_dim];
    let mut history = History::new(strategy, args, batch, device);
    let usage = || device.memory_pool_usage();
    let (mut peak_in_use, mut peak_reserved) = (None::<u64>, None::<u64>);
    let mut steps = Vec::with_capacity(args.steps);

    let mut x = Tensor::<4>::random(step, Distribution::Default, device);
    device.sync().expect("the device syncs");
    let t0 = Instant::now();
    for _ in 0..args.steps {
        let ts = Instant::now();
        for layer in 0..args.layers {
            let (k, v) = history.insert(layer, x.clone(), x.clone() * 0.5);
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
    drop(history);
    device.memory_cleanup();
    Run {
        elapsed,
        steps,
        peak_in_use,
        peak_reserved,
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
        "device {:?}, fusion {}, {} layers x {} heads x {} head_dim, {} steps",
        *device,
        cfg!(feature = "fusion"),
        args.layers,
        args.heads,
        args.head_dim,
        args.steps
    );
    println!(
        "| strategy | batch | total ms | ms/step | first quarter ms/step | last quarter ms/step | last / first | peak in use MiB | peak reserved MiB |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for &batch in &args.batches {
        for &strategy in &args.strategies {
            run(strategy, &args, batch, &device);
            let runs: Vec<Run> = (0..args.reps)
                .map(|_| run(strategy, &args, batch, &device))
                .collect();
            let best = runs
                .iter()
                .min_by_key(|r| r.elapsed)
                .expect("at least one repetition");
            let (first, last) = (best.ms_per_step(0.0, 0.25), best.ms_per_step(0.75, 1.0));
            println!(
                "| {strategy:?} | {batch} | {:.1} | {:.3} | {first:.3} | {last:.3} | {:.2} | {} | {} |",
                best.elapsed.as_secs_f64() * 1000.0,
                best.elapsed.as_secs_f64() * 1000.0 / args.steps as f64,
                last / first,
                mib(best.peak_in_use),
                mib(best.peak_reserved),
            );
        }
    }
}
