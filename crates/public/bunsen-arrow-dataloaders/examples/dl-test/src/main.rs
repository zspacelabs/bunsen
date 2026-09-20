use std::sync::{
    Arc,
    Mutex,
};

use bunsen::{
    data::cache::BunsenDiskCache,
    kits::gpts::nanochat::datasets::NANOCHAT_SHARD_SETS,
    support::testing::{
        PerformanceBackend,
        backend_device,
    },
};
use bunsen_arrow_dataloaders::{
    dataloaders::chat::ChatDataLoader,
    tokens::{
        DenseTokenBlocksOptions,
        TokenBatchIteratorOptions,
    },
};
use clap::Parser;
use clap_common::shards::ShardArgs;
use rand::{
    SeedableRng,
    rngs::StdRng,
};
use wordchipper::{
    Tokenizer,
    UnifiedTokenVocab,
    VocabIndex,
    disk_cache::WordchipperDiskCache,
};
use wordchipper_cli_util::logging::LogArgs;

#[derive(Debug, Clone, clap::Args)]
pub struct TokenBatchOptionsArgs {
    /// The number of sequences to load per batch.
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// The maximum number of tokens in a sequence.
    #[arg(long, default_value_t = 2048)]
    pub batch_seq_len: usize,

    /// The minimum number of sequences to keep in the buffer
    /// before loading more sequences.
    #[arg(long, default_value_t = 1024)]
    pub min_buffer: usize,
}

impl TokenBatchOptionsArgs {
    pub fn options(&self) -> TokenBatchIteratorOptions {
        TokenBatchIteratorOptions {
            batch_size: self.batch_size,
            batch_seq_len: self.batch_seq_len,
            min_buffer: self.min_buffer,
        }
    }
}

/// Example Nanochat Data Loader.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The shards to load, and how to fetch them.
    #[clap(flatten)]
    pub shards: ShardArgs,

    /// The vocab model to use.
    #[arg(long, default_value = "openai:p50k_base")]
    pub vocab_model: String,

    #[arg(long, default_value = "<|bos|>")]
    pub bos_token: String,

    #[command(flatten)]
    pub token_batch_options: TokenBatchOptionsArgs,

    /// Logging configuration.
    #[clap(flatten)]
    logging: LogArgs,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    args.logging.setup_logging(3).unwrap();

    log::info!("ARGS: {:#?}", args);

    type T = u32;

    let shard_cache = BunsenDiskCache::default();
    let shard_paths = args.shards.fetch_paths(
        &shard_cache,
        NANOCHAT_SHARD_SETS.expect_lookup("fineweb-edu-100b-shuffle"),
    )?;

    let mut wc_disk_cache: WordchipperDiskCache = Default::default();

    log::info!("Loading Vocab: {:?}", args.vocab_model);
    let mut vocab: UnifiedTokenVocab<T> =
        wordchipper::load_vocab(&args.vocab_model, &mut wc_disk_cache)?
            .vocab()
            .to_token_type()?;

    let max_token = vocab.max_token().unwrap();

    let bos_token: T = {
        let specials = vocab.special_vocab_mut();
        if let Some(tok) = specials.lookup_token(args.bos_token.as_bytes()) {
            tok
        } else {
            let tok = max_token + 1;
            specials.add_str_word(&args.bos_token, tok);
            tok
        }
    };

    let tok: Arc<Tokenizer<T>> = wordchipper::TokenizerOptions::default()
        .with_parallel(true)
        .with_accelerated_lexers(true)
        .build(vocab.into());

    let block_options = DenseTokenBlocksOptions {
        batch_size: args.token_batch_options.batch_size,
        batch_seq_len: args.token_batch_options.batch_seq_len,
        min_buffer: args.token_batch_options.min_buffer,
        bos: vec![bos_token],
        eos: vec![],
    };

    type B = PerformanceBackend;

    let device = backend_device::<B>();

    let data_loader: ChatDataLoader<B> = ChatDataLoader::new(
        shard_paths,
        Some(Arc::new(Mutex::new(StdRng::seed_from_u64(0)))),
        &device,
        tok.clone(),
        block_options,
    );

    let dl_iter = data_loader.start_epoch();
    let stats = dl_iter.stats().clone();

    let shape = [
        args.token_batch_options.batch_size,
        args.token_batch_options.batch_seq_len,
    ];

    let mut last_idx = 0;
    let t0 = std::time::Instant::now();
    for (idx, tensor) in dl_iter.enumerate() {
        assert_eq!(&tensor.dims(), &shape);
        last_idx = idx;
    }
    let elapsed = t0.elapsed();
    println!("elapsed: {:.2?}", elapsed);

    println!("shape: {last_idx} x {shape:?}");

    let human_opts = humansize::FormatSizeOptions::from(humansize::BINARY).decimal_places(1);

    println!(
        "bps: {}/s",
        humansize::format_size_i(
            stats.byte_count() as f64 / elapsed.as_secs_f64(),
            human_opts
        )
    );

    let tps = humansize::format_size_i(
        stats.token_count() as f64 / elapsed.as_secs_f64(),
        human_opts,
    );
    println!("tps: {}T/s", &tps[..tps.len() - 1]);

    Ok(())
}
