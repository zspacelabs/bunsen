use std::{
    cmp::max,
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
    },
};

use bunsen::{
    burner::{
        module::{
            ModuleInit,
            reflection::XmlModuleTree,
        },
        optim::{
            GroupOptimizerPlan,
            OptimizerGroup,
        },
    },
    data::cache::BunsenDiskCache,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
        sys_at,
    },
    kits::gpts::nanochat::{
        NanoChatGpt,
        NanoChatGptContractConfig,
        NanoChatGptMeta,
        datasets::NANOCHAT_SHARD_SETS,
    },
    public::hashbrown::HashSet,
};
use bunsen_app::{
    device::{
        DeviceArgs,
        DevicePrefs,
    },
    shards::ShardArgs,
};
use bunsen_arrow_dataloaders::{
    dataloaders::chat::ChatDataLoader,
    tokens::{
        DenseTokenBlocksOptions,
        TokenBatchIteratorOptions,
    },
};
use burn::{
    lr_scheduler::linear::LinearLrSchedulerConfig,
    module::{
        Module,
        ParamId,
    },
    nn::loss::CrossEntropyLossConfig,
    optim::{
        AdamWConfig,
        MuonConfig,
        decay::WeightDecayConfig,
    },
    tensor::{
        Device,
        Tensor,
        s,
    },
    train::{
        ClassificationOutput,
        InferenceStep,
        Learner,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
        metric::{
            LearningRateMetric,
            LossMetric,
        },
    },
};
use clap::Parser;
use num_traits::Pow;
use rand::{
    SeedableRng,
    rngs::StdRng,
};
use wordchipper::{
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

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The embedding dimension size.
    #[clap(long, default_value = "768")]
    pub n_embed: usize,

    /// The number of layers.
    #[clap(long, default_value = "8")]
    pub n_layer: usize,

    /// The pretrained vocabulary.
    #[clap(long, default_value = "openai:p50k_edit")]
    pub pretrained_vocab: String,

    /// Beginning-of-Sequence token.
    #[arg(long, default_value = "<|bos|>")]
    pub bos_token: String,

    /// The shards to train on, and how to fetch them.
    #[clap(flatten)]
    pub shards: ShardArgs,

    #[arg(long, default_value_t = 0.008)]
    pub unembedding_lr: f64,

    #[arg(long, default_value_t = 0.3)]
    pub embedding_lr: f64,

    #[arg(long, default_value_t = 0.02)]
    pub matrix_lr: f64,

    #[arg(long, default_value_t = 0.5)]
    pub scalar_lr: f64,

    /// Warm-up steps.
    #[arg(long, default_value_t = 300)]
    pub warmup_steps: usize,

    /// Optimizer Weight decay.
    #[arg(long, default_value_t = 0.28)]
    pub weight_decay: f32,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "100")]
    pub num_epochs: usize,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 4)]
    pub batch_size: usize,

    /// The training seq len.
    #[clap(long, default_value = "2048")]
    pub seq_len: usize,

    /// Grads accumulation size for processing
    #[arg(short, long, default_value_t = 1)]
    pub grads_accumulation: usize,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/chat")]
    pub artifact_dir: String,

    /// The device to train on.
    #[command(flatten)]
    pub device: DeviceArgs,
}

fn ensure_artifact_dir(artifact_dir: &str) -> BunsenResult<()> {
    let _ignored = std::fs::remove_dir_all(artifact_dir);
    std::fs::create_dir_all(artifact_dir).map_err(sys_at("create", artifact_dir))?;
    Ok(())
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();

    run(&args)
}

fn run(args: &Args) -> BunsenResult<()> {
    type T = u32;

    println!("{:#?}", args);

    // Remove existing artifacts before to get an accurate learner summary
    let artifact_dir: &str = args.artifact_dir.as_ref();
    ensure_artifact_dir(artifact_dir)?;

    // Training records gradients: autodiff before the model and inputs.
    let device: Device = args
        .device
        .init(&DevicePrefs::training())
        .map_err(BunsenError::unsupported)?;

    let shard_cache = BunsenDiskCache::default();
    let shard_paths = args.shards.fetch_paths(
        &shard_cache,
        NANOCHAT_SHARD_SETS.expect_lookup("fineweb-edu-100b-shuffle"),
    )?;

    let mut disk_cache = WordchipperDiskCache::default();

    let validation_ratio = 0.10;
    let num_validation_shards: usize = max(
        ((shard_paths.len() as f64) * validation_ratio).ceil() as usize,
        1,
    );
    let num_training_shards = shard_paths.len() - num_validation_shards;

    let training_paths: Vec<PathBuf> = shard_paths[..num_training_shards].to_vec();
    let validation_paths: Vec<PathBuf> = shard_paths[num_training_shards..].to_vec();

    let mut vocab: UnifiedTokenVocab<T> =
        wordchipper::load_vocab(&args.pretrained_vocab, &mut disk_cache)
            .map_err(BunsenError::other)
            .context("loading the vocabulary")?
            .vocab()
            .to_token_type()
            .map_err(BunsenError::other)?;

    let max_token = vocab.max_token().unwrap();

    // This is a stupid hack.
    let mut vocab_size = vocab.len();
    let bos_token: T = {
        let specials = vocab.special_vocab_mut();
        if let Some(tok) = specials.lookup_token(args.bos_token.as_bytes()) {
            tok
        } else {
            let tok = max_token + 1;
            specials.add_str_word(&args.bos_token, tok);
            vocab_size += 1;
            tok
        }
    };
    let vocab = Arc::new(vocab);

    let tok = wordchipper::TokenizerOptions::default()
        .with_accelerated_lexers(true)
        .with_parallel(true)
        .build(vocab);

    let gpt_config = NanoChatGptContractConfig::new()
        .with_n_embed(args.n_embed)
        .with_n_layer(args.n_layer)
        .with_vocab_size(vocab_size);

    let gpt: NanoChatGpt = gpt_config.init(&device);

    let host = GptHost { gpt };

    let dl_config = DenseTokenBlocksOptions {
        batch_seq_len: args.seq_len,
        batch_size: args.batch_size,
        bos: vec![bos_token],
        ..Default::default()
    };

    let training_data_loader: ChatDataLoader = ChatDataLoader::new(
        training_paths,
        Some(Arc::new(Mutex::new(StdRng::seed_from_u64(0)))),
        &device,
        tok.clone(),
        dl_config.clone(),
    );

    let validation_data_loader: ChatDataLoader =
        ChatDataLoader::new(validation_paths, None, &device, tok.clone(), dl_config);

    let training = SupervisedTraining::new(
        artifact_dir,
        Arc::new(training_data_loader),
        Arc::new(validation_data_loader),
    )
    .grads_accumulation(args.grads_accumulation)
    .num_epochs(args.num_epochs)
    .metrics((LossMetric::new(), LearningRateMetric::new()))
    .with_default_checkpointers()
    .summary();

    let ParamGroups {
        matrix: matrix_params,
        embedding: embedding_params,
        lm_head: lm_head_params,
        remnant: remnant_params,
    } = ParamGroups::select(&host)?;

    let model_dim = gpt_config.n_embed();
    let dmodel_lr_scale: f64 = (model_dim as f64 / 768.0_f64).pow(-0.5);

    let lm_head_lr = args.unembedding_lr * dmodel_lr_scale;
    let embedding_lr = args.embedding_lr * dmodel_lr_scale;
    let scalar_lr = args.scalar_lr;
    let matrix_lr = args.matrix_lr;

    // This is only used to scale the learning rates for each group below.
    // This implements warmup scheduling for learning rate.
    let warmup_scheduler = LinearLrSchedulerConfig::new(1e-10, 1.0, args.warmup_steps)
        .init()
        .expect("Failed to initialize learning rate scheduler");

    // TODO: per-group GradientClipping.

    // `try_new` checks that the groups partition the model's float
    // parameters; `ParamGroups::select`'s remnant group covers whatever the
    // others don't.
    let plan = GroupOptimizerPlan::try_new(
        &host,
        vec![
            OptimizerGroup::new(
                lm_head_params,
                AdamWConfig::new()
                    .with_beta_1(0.8)
                    .with_beta_2(0.96)
                    .with_epsilon(1e-10)
                    .with_weight_decay(0.01)
                    .build(),
            )
            .with_lr_selector(move |lr| lr * lm_head_lr),
            OptimizerGroup::new(
                embedding_params,
                AdamWConfig::new()
                    .with_beta_1(0.8)
                    .with_beta_2(0.995)
                    .with_epsilon(1e-10)
                    .with_weight_decay(0.001)
                    .build(),
            )
            .with_lr_selector(move |lr| lr * embedding_lr),
            OptimizerGroup::new(
                remnant_params,
                AdamWConfig::new()
                    .with_beta_1(0.8)
                    .with_beta_2(0.96)
                    .with_epsilon(1e-10)
                    .with_weight_decay(0.01)
                    .build(),
            )
            .with_lr_selector(move |lr| lr * scalar_lr),
            OptimizerGroup::new(
                matrix_params,
                MuonConfig::new()
                    // .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
                    .with_weight_decay(Some(WeightDecayConfig {
                        penalty: args.weight_decay,
                    }))
                    .build(),
            )
            .with_lr_selector(move |lr| lr * matrix_lr),
        ],
    )?;

    let result = training.launch(Learner::new(
        host,
        plan.optimizer(),
        plan.lr_scheduler(warmup_scheduler),
    ));
    if let Some(error) = result.error {
        return Err(BunsenError::other(error).context("training"));
    }

    result
        .model
        .save_file(format!("{artifact_dir}/model.bpk"))
        .map_err(BunsenError::other)
        .context("saving the model")?;

    Ok(())
}

#[derive(Module, Debug)]
pub struct GptHost {
    pub gpt: NanoChatGpt,
}

impl GptHost {
    fn loss_step(
        &self,
        input: Tensor<2, burn::prelude::Int>,
    ) -> ClassificationOutput {
        let inputs: Tensor<2, burn::prelude::Int> = input.clone().slice(s![.., ..-1]);
        let targets: Tensor<2, burn::prelude::Int> = input.slice(s![.., 1..]);

        let mut kv_cache = None;

        // Logits.
        let outputs: Tensor<3> = self.gpt.forward(inputs, &mut kv_cache);

        let output_flatten: Tensor<2> = outputs.flatten(0, 1);
        let targets_flatten: Tensor<1, burn::prelude::Int> = targets.flatten(0, 1);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_flatten.device())
            .forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl TrainStep for GptHost {
    type Input = Tensor<2, burn::prelude::Int>;
    type Output = ClassificationOutput;

    fn step(
        &self,
        input: Self::Input,
    ) -> TrainOutput<Self::Output> {
        let classification_output = self.loss_step(input);
        let grads = classification_output.loss.backward();

        TrainOutput::new(self, grads, classification_output)
    }
}

impl InferenceStep for GptHost {
    type Input = Tensor<2, burn::prelude::Int>;
    type Output = ClassificationOutput;

    fn step(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        self.loss_step(input)
    }
}

/// The optimizer groups of the `NanoChat` recipe, as disjoint `ParamId` sets.
///
/// See `GPT.setup_optimizer`:
/// <https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L374>
#[derive(Debug)]
pub struct ParamGroups {
    /// Rank-2 `Linear` weights inside the transformer blocks (Muon).
    pub matrix: HashSet<ParamId>,

    /// The token embedding table (`AdamW`).
    pub embedding: HashSet<ParamId>,

    /// The output head (`AdamW`).
    pub lm_head: HashSet<ParamId>,

    /// Every parameter no other group claims, e.g. the norms (`AdamW`).
    pub remnant: HashSet<ParamId>,
}

impl ParamGroups {
    /// Selects the groups from the module structure.
    ///
    /// Element names are module *type* names, and field names are `@name`:
    /// the `gpt` field is `GptHost/NanoChatGpt`. `h` is a `Vec` of
    /// `NanoChatGptBlock`, whose `Linear`s sit in `attn` and `mlp`, hence
    /// `//Linear`. Stacked predicates (`[a][b]`) mean "a and b".
    pub fn select(host: &GptHost) -> BunsenResult<Self> {
        let mut mtree = XmlModuleTree::build(host);
        let mut select = |expr: &str| -> BunsenResult<HashSet<ParamId>> {
            Ok(mtree.select_param_ids(expr)?.into_iter().collect())
        };

        let matrix = select("GptHost/NanoChatGpt/*[@name='h']//Linear/*[@name='weight'][@rank=2]")?;

        // TODO: value_embeds
        // TODO: resid
        // TODO: x0
        // TODO: smear, smear_gate, blackout

        let embedding = select("GptHost/NanoChatGpt/*[@name='wte']")?;
        let lm_head = select("GptHost/NanoChatGpt/*[@name='lm_head']")?;

        let remnant: HashSet<ParamId> = mtree
            .param_ids()?
            .into_iter()
            .filter(|id| !matrix.contains(id) && !embedding.contains(id) && !lm_head.contains(id))
            .collect();

        Ok(Self {
            matrix,
            embedding,
            lm_head,
            remnant,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    /// A `GptHost` small enough to build in milliseconds.
    fn tiny_host() -> GptHost {
        let gpt: NanoChatGpt = NanoChatGptContractConfig::new()
            .with_vocab_size(32)
            .with_n_layer(2)
            .with_n_head(2)
            .with_n_kv_head(2)
            .with_n_embed(8)
            .with_init_seq_len(16)
            .init(&bunsen::support::testing::cpu_device());
        GptHost { gpt }
    }

    /// Each group selects something, no two overlap, and together they cover
    /// the model. A selector that names no element matches nothing, and its
    /// parameters fall silently into `remnant`, so Muon would never step.
    #[test]
    fn test_param_groups_partition_the_model() {
        let host = tiny_host();
        let mut mtree = XmlModuleTree::build(&host);
        let xml = mtree.to_xml(true);

        let groups = ParamGroups::select(&host).unwrap();
        let named = [
            ("matrix", &groups.matrix),
            ("embedding", &groups.embedding),
            ("lm_head", &groups.lm_head),
            ("remnant", &groups.remnant),
        ];

        for (name, group) in named {
            assert!(
                !group.is_empty(),
                "group `{name}` selected nothing in:\n{xml}"
            );
        }

        for (i, (a_name, a)) in named.iter().enumerate() {
            for (b_name, b) in &named[i + 1..] {
                assert!(a.is_disjoint(b), "groups `{a_name}` and `{b_name}` overlap");
            }
        }

        let all: HashSet<ParamId> = mtree.param_ids().unwrap().into_iter().collect();
        let covered: HashSet<ParamId> = named
            .iter()
            .flat_map(|(_, group)| group.iter().copied())
            .collect();
        assert_eq!(covered, all);

        // Muon gets every matrix in the blocks, not just some of them.
        let block_matrices: HashSet<ParamId> = mtree
            .select_params("GptHost/NanoChatGpt/*[@name='h']")
            .to_param_descs()
            .unwrap()
            .iter()
            .filter(|desc| desc.rank() == 2)
            .map(|desc| desc.param_id())
            .collect();
        assert_eq!(groups.matrix, block_matrices);
    }
}
