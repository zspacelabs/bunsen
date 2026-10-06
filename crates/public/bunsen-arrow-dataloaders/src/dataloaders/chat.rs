//! Chat Data Loader
//!
//! A preview implementation of a streaming chat data loader for training
//! LLM-style models on top of the [`burn`] tensor library.
//!
//! The pipeline reads Parquet shards, selects a text column, tokenizes the
//! text with a [`wordchipper::Tokenizer`], packs the tokens into dense
//! fixed-shape blocks, optionally shuffles the blocks through a bounded
//! reservoir, and finally materializes each block as a 2D integer
//! [`burn::tensor::Tensor`].
//!
//! ## Pipeline overview
//!
//! ```text
//! shard paths
//!     -> read_parquet_shards          (Iterator<ArrowResult<RecordBatch>>)
//!     -> select_text_column           (Iterator<ArrowResult<Vec<String>>>)
//!     -> tokenize_text_batches        (Iterator<ArrowResult<Vec<Vec<u32>>>>)
//!     -> DenseTokenBlockBatcher       (Iterator<ArrowResult<Vec<Vec<u32>>>>)
//!     -> ShuffleIter (optional)
//!     -> Tensor<2, Int>
//! ```
//!
//! Counters layered into the pipeline via [`crate::iterators::IterWatcher`]
//! feed [`EpochStats`], which drives the [`burn::data::dataloader::Progress`]
//! reporting expected by the burn training loop.
//!
//! ## Example Use
//!
//! The following example builds a data loader for a chat dataset consisting
//! of a training and validation set of Parquet shards.
//!
//! The batch items are `Tensor<2, Int>`, where `B` is the burn backend
//! type (e.g. `Cuda` or `Cpu`) and `Int` is burn's integer tensor kind; the
//! element type is the backend's integer element.
//!
//! ```rust,ignore
//! let training_data_loader: ChatDataLoader = ChatDataLoader::new(
//!     training_paths,
//!     Some(Arc::new(Mutex::new(StdRng::seed_from_u64(0)))),
//!     &device,
//!     tok.clone(),
//!     dl_config.clone(),
//! );
//! let validation_data_loader: ChatDataLoader =
//!     ChatDataLoader::new(validation_paths, None, &device, tok.clone(), dl_config);
//! ```
use std::{
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
        atomic::AtomicUsize,
    },
};

use arrow::error::ArrowError;
use burn::{
    Tensor,
    data::dataloader::{
        DataLoader,
        DataLoaderIterator,
        Progress,
    },
    prelude::TensorData,
};
use rand::prelude::SliceRandom;
use wordchipper::Tokenizer;

use crate::{
    arrow::{
        read_parquet_shards,
        select_text_column,
    },
    iterators::{
        IterWatcher,
        ShuffleIterOptions,
    },
    tokens::{
        DenseTokenBlocksOptions,
        tokenize_text_batches,
    },
};

/// A single-epoch iterator over packed 2D integer token tensors.
///
/// Built by [`ChatDataLoader::start_epoch`]; assembles the full pipeline
/// (Parquet read -> column select -> tokenize -> dense pack -> optional
/// shuffle -> tensor materialization) and exposes the shared
/// [`EpochStats`] used for progress reporting.
pub struct ChatDataLoaderIterator {
    stats: Arc<EpochStats>,
    inner: Box<dyn Iterator<Item = Tensor<2, burn::prelude::Int>>>,
}

impl ChatDataLoaderIterator {
    /// Builds the streaming pipeline for one epoch.
    ///
    /// ## Arguments
    /// * `device` - Target burn device for emitted tensors.
    /// * `tokenizer` - Shared tokenizer used to encode the text column.
    /// * `shard_paths` - Ordered list of Parquet shards to consume this epoch.
    /// * `block_options` - Packing configuration (batch shape, BOS / EOS
    ///   markers).
    /// * `shuffle_options` - Optional reservoir-shuffle configuration applied
    ///   to the packed blocks; `None` preserves source order.
    /// * `text_column` - Name of the UTF-8 column in each shard to tokenize.
    pub fn new(
        device: Device,
        tokenizer: Arc<Tokenizer<u32>>,
        shard_paths: Vec<PathBuf>,
        block_options: DenseTokenBlocksOptions,
        shuffle_options: Option<ShuffleIterOptions>,
        text_column: &str,
    ) -> Self {
        let stats = EpochStats {
            items_total: shard_paths.len(),
            ..Default::default()
        };

        let file_counter = stats.file_counter.clone();
        let shard_counter = IterWatcher::new(
            shard_paths.into_iter(),
            Box::new(move |_| {
                file_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }),
        );

        // Iterator<ArrowResult<RecordBatch>>
        let parquet_batches = read_parquet_shards(shard_counter);

        // Iterator<ArrowResult<Vec<String>>>
        let byte_counter = stats.byte_counter.clone();
        let sample_batches = IterWatcher::new(
            select_text_column(text_column, parquet_batches),
            Box::new(move |result| {
                if let Ok(batch) = &result {
                    let bytes = batch.iter().map(|s| s.as_str().len()).sum::<usize>();
                    byte_counter.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                }
            }),
        );

        // Iterator<ArrowResult<Vec<Vec<u32>>>>
        let token_batches = tokenize_text_batches(tokenizer, sample_batches);

        let shape = [block_options.batch_size, block_options.batch_seq_len];

        // Iterator<ArrowResult<Vec<Vec<u32>>>> (batch_size x batch_seq_len)
        let token_counter = stats.token_counter.clone();
        let dense_blocks = IterWatcher::new(
            block_options.build_dense_blocks(token_batches),
            Box::new(move |result| {
                if let Ok(batch) = &result {
                    let tokens = batch.iter().map(|ts| ts.len()).sum::<usize>();
                    token_counter.fetch_add(tokens, std::sync::atomic::Ordering::Relaxed);
                }
            }),
        );

        let shuffle: Box<dyn Iterator<Item = Result<Vec<Vec<u32>>, ArrowError>>> =
            if let Some(shuffle_options) = shuffle_options {
                Box::new(shuffle_options.init(dense_blocks))
            } else {
                Box::new(dense_blocks)
            };

        let tensors = shuffle.map(move |result| {
            let batch = &result.unwrap();
            let tensor: Tensor<2, burn::prelude::Int> = Tensor::from_ints(
                TensorData::new(batch.iter().flatten().copied().collect(), shape),
                &device,
            );
            tensor
        });

        Self {
            stats: Arc::new(stats),
            inner: Box::new(tensors),
        }
    }

    /// Returns the shared [`EpochStats`] handle.
    ///
    /// The handle is updated as the pipeline runs and can be cloned cheaply
    /// to observe progress from another thread.
    pub fn stats(&self) -> &Arc<EpochStats> {
        &self.stats
    }
}

impl Iterator for ChatDataLoaderIterator {
    type Item = Tensor<2, burn::prelude::Int>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl DataLoaderIterator<Tensor<2, burn::prelude::Int>> for ChatDataLoaderIterator {
    fn progress(&self) -> Progress {
        self.stats.progress()
    }
}

/// A [`burn::data::dataloader::DataLoader`] that streams packed token
/// tensors out of a set of Parquet shards.
///
/// When constructed with an `rng`, [`start_epoch`](Self::start_epoch)
/// shuffles the shard order and applies a reservoir shuffle to the packed
/// blocks, both drawn from that `rng`; without an `rng` the loader returns
/// shards (and packed blocks) in their input order.
#[derive(Clone)]
pub struct ChatDataLoader {
    shard_paths: Vec<PathBuf>,
    rng: Option<Arc<Mutex<dyn rand::Rng + Send>>>,
    device: Device,
    tokenizer: Arc<Tokenizer<u32>>,
    block_options: DenseTokenBlocksOptions,
}

impl ChatDataLoader {
    /// Builds a new chat data loader.
    ///
    /// ## Arguments
    /// * `files` - Parquet shard paths consumed each epoch.
    /// * `rng` - Optional shared rng; presence enables both shard-order
    ///   shuffling and the reservoir block shuffle, and each epoch draws both
    ///   from it.
    /// * `device` - Target burn device for emitted tensors.
    /// * `tokenizer` - Shared tokenizer used to encode the text column.
    /// * `block_options` - Packing configuration (batch shape, BOS / EOS
    ///   markers).
    pub fn new(
        files: Vec<PathBuf>,
        rng: Option<Arc<Mutex<dyn rand::Rng + Send>>>,
        device: &Device,
        tokenizer: Arc<Tokenizer<u32>>,
        block_options: DenseTokenBlocksOptions,
    ) -> Self {
        Self {
            shard_paths: files,
            rng,
            device: device.clone(),
            tokenizer,
            block_options,
        }
    }

    /// Starts a new epoch.
    ///
    /// With an rng, the epoch draws its shard order and its block-shuffle
    /// seed from it, so successive epochs shuffle differently, and a loader
    /// built with the same seeded rng repeats them.
    pub fn start_epoch(&self) -> ChatDataLoaderIterator {
        let mut shard_paths = self.shard_paths.clone();
        let shuffle_options = self.rng.as_ref().map(|mutex| {
            let mut rng = mutex.lock().unwrap();
            shard_paths.shuffle(&mut *rng);

            ShuffleIterOptions::default()
                .with_fill_rate(2)
                .with_buffer_size(128)
                .with_seed(rng.next_u64())
        });

        ChatDataLoaderIterator::new(
            self.device.clone(),
            self.tokenizer.clone(),
            shard_paths,
            self.block_options.clone(),
            shuffle_options,
            "text",
        )
    }
}

impl DataLoader<Tensor<2, burn::prelude::Int>> for ChatDataLoader {
    fn iter(&self) -> Box<dyn DataLoaderIterator<Tensor<2, burn::prelude::Int>>> {
        Box::new(self.start_epoch())
    }

    fn num_items(&self) -> usize {
        self.shard_paths.len()
    }

    fn to_device(
        &self,
        device: &Device,
    ) -> Arc<dyn DataLoader<Tensor<2, burn::prelude::Int>>> {
        Arc::new(Self {
            shard_paths: self.shard_paths.clone(),
            rng: self.rng.clone(),
            device: device.clone(),
            tokenizer: self.tokenizer.clone(),
            block_options: self.block_options.clone(),
        })
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Arc<dyn DataLoader<Tensor<2, burn::prelude::Int>>> {
        Arc::new(Self {
            shard_paths: self.shard_paths[start..end].to_vec(),
            rng: self.rng.clone(),
            device: self.device.clone(),
            tokenizer: self.tokenizer.clone(),
            block_options: self.block_options.clone(),
        })
    }
}

/// Live, shared statistics for one iteration over the data loader.
///
/// The counters are stored as shared atomics so that
/// [`IterWatcher`]-style callbacks layered into the streaming pipeline can
/// update them as data flows through, while the training loop reads them
/// to drive [`Progress`] reporting.
#[derive(Debug, Default, Clone)]
pub struct EpochStats {
    /// Count every file opened.
    pub file_counter: Arc<AtomicUsize>,
    /// Count every byte read.
    pub byte_counter: Arc<AtomicUsize>,
    /// Count every token read.
    pub token_counter: Arc<AtomicUsize>,
    /// Lists the number of shards.
    pub items_total: usize,
}

impl EpochStats {
    /// Builds an `EpochStats` from pre-existing atomic counters.
    ///
    /// Primarily intended for testing; the typical construction path is to
    /// let [`ChatDataLoaderIterator::new`] allocate fresh counters.
    pub fn new(
        file_counter: Arc<AtomicUsize>,
        byte_counter: Arc<AtomicUsize>,
        token_counter: Arc<AtomicUsize>,
        items_total: usize,
    ) -> Self {
        Self {
            file_counter,
            byte_counter,
            token_counter,
            items_total,
        }
    }

    /// Number of Parquet shards opened so far this epoch.
    pub fn file_count(&self) -> usize {
        self.file_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total number of source text bytes pulled out of Parquet so far.
    pub fn byte_count(&self) -> usize {
        self.byte_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total number of tokens emitted by the packer so far.
    pub fn token_count(&self) -> usize {
        self.token_counter
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total number of shards in the epoch (the denominator for
    /// [`progress`](Self::progress)).
    pub fn items_total(&self) -> usize {
        self.items_total
    }

    /// Returns a [`Progress`] snapshot suitable for the burn training loop.
    pub fn progress(&self) -> Progress {
        Progress {
            items_processed: self.file_count(),
            items_total: self.items_total(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        path::Path,
    };

    use arrow::{
        array::{
            RecordBatch,
            StringArray,
        },
        datatypes::{
            DataType,
            Field,
            Schema,
        },
    };
    use parquet::arrow::ArrowWriter;
    use rand::{
        SeedableRng,
        rngs::StdRng,
    };
    use wordchipper::{
        UnifiedTokenVocab,
        vocab::utility::testing::build_test_vocab,
    };

    use super::*;

    /// Writes `texts` as the `text` column of a one-shard Parquet file.
    fn write_shard(
        path: &Path,
        texts: &[String],
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(texts.to_vec()))],
        )
        .unwrap();

        let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// One epoch's blocks, in order, from a loader with an rng seeded `seed`.
    fn epoch_blocks(
        shard: &Path,
        seed: u64,
    ) -> Vec<Vec<i64>> {
        let vocab: UnifiedTokenVocab<u32> = build_test_vocab(
            Default::default(),
            wordchipper::pretrained::openai::oa_p50k_edit_spanning_config(),
        );
        let tokenizer = wordchipper::TokenizerOptions::default().build(vocab.into());

        let loader: ChatDataLoader = ChatDataLoader::new(
            vec![shard.to_path_buf()],
            Some(Arc::new(Mutex::new(StdRng::seed_from_u64(seed)))),
            &Default::default(),
            tokenizer,
            DenseTokenBlocksOptions {
                batch_size: 1,
                batch_seq_len: 8,
                min_buffer: 1,
                bos: vec![],
                eos: vec![],
            },
        );

        loader
            .start_epoch()
            .map(|tensor| tensor.into_data().iter::<i64>().collect())
            .collect()
    }

    #[test]
    fn test_block_shuffle_draws_from_the_loader_rng() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("shard.parquet");
        // Each text is longer than a row, so each block is one text's first
        // 8 tokens (bytes, with the test vocab), which differ.
        let texts = (0..64)
            .map(|i| format!("{i:04} says hello"))
            .collect::<Vec<_>>();
        write_shard(&shard, &texts);

        // One shard, so the shard-order shuffle cannot change anything; the
        // block order comes from the block shuffle alone.
        let first = epoch_blocks(&shard, 1);
        assert!(first.len() > 16, "{} blocks", first.len());
        assert_eq!(epoch_blocks(&shard, 1), first);

        let second = epoch_blocks(&shard, 2);
        assert_ne!(second, first);

        let mut first_sorted = first.clone();
        first_sorted.sort();
        let mut second_sorted = second.clone();
        second_sorted.sort();
        assert_eq!(second_sorted, first_sorted);
    }
}
