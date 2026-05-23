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
    data::dataloader::{
        DataLoader,
        DataLoaderIterator,
        Progress,
    },
    prelude::Backend,
    tensor::{
        Tensor,
        TensorData,
    },
};
use rand::seq::SliceRandom;
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

#[derive(Debug, Default, Clone)]
pub struct EpochStats {
    file_counter: Arc<AtomicUsize>,
    byte_counter: Arc<AtomicUsize>,
    token_counter: Arc<AtomicUsize>,
    items_total: usize,
}

impl EpochStats {
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

    pub fn file_count(&self) -> usize {
        self.file_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn byte_count(&self) -> usize {
        self.byte_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn token_count(&self) -> usize {
        self.token_counter
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn items_total(&self) -> usize {
        self.items_total
    }

    pub fn progress(&self) -> Progress {
        Progress {
            items_processed: self.file_count(),
            items_total: self.items_total(),
        }
    }
}

pub struct ChatDataLoaderIterator<B: Backend> {
    stats: Arc<EpochStats>,
    inner: Box<dyn Iterator<Item = Tensor<B, 2, burn::prelude::Int>>>,
}

impl<B: Backend> ChatDataLoaderIterator<B> {
    pub fn new(
        device: B::Device,
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
            let tensor: Tensor<B, 2, burn::prelude::Int> = Tensor::from_ints(
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

    pub fn stats(&self) -> &Arc<EpochStats> {
        &self.stats
    }
}

impl<B: Backend> Iterator for ChatDataLoaderIterator<B> {
    type Item = Tensor<B, 2, burn::prelude::Int>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<B: Backend> DataLoaderIterator<Tensor<B, 2, burn::prelude::Int>>
    for ChatDataLoaderIterator<B>
{
    fn progress(&self) -> Progress {
        self.stats.progress()
    }
}

#[derive(Clone)]
pub struct ChatDataLoader<B: Backend> {
    shard_paths: Vec<PathBuf>,
    rng: Option<Arc<Mutex<dyn rand::Rng + Send>>>,
    device: B::Device,
    tokenizer: Arc<Tokenizer<u32>>,
    block_options: DenseTokenBlocksOptions,
}

impl<B: Backend> ChatDataLoader<B> {
    pub fn new(
        files: Vec<PathBuf>,
        rng: Option<Arc<Mutex<dyn rand::Rng + Send>>>,
        device: &B::Device,
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
    pub fn start_epoch(&self) -> ChatDataLoaderIterator<B> {
        let mut shard_paths = self.shard_paths.clone();
        if let Some(mutex) = &self.rng {
            let mut rng = mutex.lock().unwrap();
            shard_paths.shuffle(&mut *rng);
        }

        let shuffle_options = if self.rng.is_none() {
            None
        } else {
            Some(
                ShuffleIterOptions::default()
                    .with_fill_rate(2)
                    .with_buffer_size(128),
            )
        };

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

impl<B: Backend> DataLoader<B, Tensor<B, 2, burn::prelude::Int>> for ChatDataLoader<B>
where
    B: Backend,
{
    fn iter(&self) -> Box<dyn DataLoaderIterator<Tensor<B, 2, burn::prelude::Int>>> {
        Box::new(self.start_epoch())
    }

    fn num_items(&self) -> usize {
        self.shard_paths.len()
    }

    fn to_device(
        &self,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, Tensor<B, 2, burn::prelude::Int>>> {
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
    ) -> Arc<dyn DataLoader<B, Tensor<B, 2, burn::prelude::Int>>> {
        Arc::new(Self {
            shard_paths: self.shard_paths[start..end].to_vec(),
            rng: self.rng.clone(),
            device: self.device.clone(),
            tokenizer: self.tokenizer.clone(),
            block_options: self.block_options.clone(),
        })
    }
}
