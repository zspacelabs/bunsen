//! The model a chat holds, whichever kit it is.
use std::{
    fmt::Debug,
    sync::Arc,
};

use bunsen::{
    blocks::transformers::decoders::stack::DecoderStackCache,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::{
            driver::CachedHistory,
            gemma::{
                GemmaConstruct,
                GemmaMeta,
                GemmaModel,
            },
            qwen3::{
                Qwen3Construct,
                Qwen3Meta,
                Qwen3Model,
            },
        },
        support::{
            construct::LmConstruct,
            decoding::{
                CachedStep,
                LogitFilter,
                StackedLm,
                SuppressTokens,
            },
            model_config::GenerationPolicy,
            tokens::Detokenizer,
        },
    },
    ops::transformers::attention::KVCacheStorage,
};
use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
    },
    tensor::{
        DType,
        Device,
    },
};

use crate::chat::{
    ChatMessage,
    ChatTemplate,
    ChatTokenizer,
    HfTokenizer,
    Qwen3Tokenizer,
};

/// A checkpoint a [`ChatModel`](crate::chat::ChatModel) holds a conversation
/// with: its model, as a cached step, and what turns a conversation into the
/// ids it reads and its ids back into text.
///
/// [`LmChat`] is a stacked language model's: [`Qwen3Chat`], [`GemmaChat`].
pub trait ChatBackend: Debug + Send + Sync + 'static {
    /// The decode state a session keeps between turns.
    type State: Debug + Send;

    /// The model, stepped one token at a time.
    type Model: CachedStep<State = Self::State>;

    /// The model.
    fn model(&self) -> &Self::Model;

    /// The prompt for the assistant's next turn of `messages`: the ids the
    /// model reads. `enable_thinking` is a thinking switch, for a checkpoint
    /// that has one.
    ///
    /// # Errors
    /// As the checkpoint's renderer.
    fn render(
        &self,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>>;

    /// What turns the reply's ids into text.
    fn detokenizer(&self) -> Arc<dyn Detokenizer>;

    /// The checkpoint's generation policy: the stop tokens, and the
    /// sampling and template kwargs a reply uses unless told otherwise.
    fn generation(&self) -> &GenerationPolicy;

    /// Filters every reply's logits pass before the search chooses: ids the
    /// tokenizer cannot spell, say.
    fn filters(&self) -> Vec<Arc<dyn LogitFilter>>;

    /// The positions the checkpoint was trained on, when it says.
    fn trained_context(&self) -> Option<usize>;

    /// The float dtype the model's weights are held, and computed, in.
    fn float_dtype(&self) -> DType;

    /// A decode state for one row of at most `positions` positions.
    fn decode_state(
        &self,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> Self::State;

    /// Rewinds `state`, whose cache holds `history`, to the prefix it shares
    /// with `next` ([`CachedHistory::rewind`]), and returns the rest of
    /// `next`: what is left to prefill.
    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut Self::State,
        next: &'a [i64],
        device: &Device,
    ) -> &'a [i64];

    /// Runs `ids` through the model into `state`, after what it holds.
    ///
    /// # Returns
    /// `[1, vocab]`, the logits that follow.
    fn prefill(
        &self,
        ids: &[i64],
        state: &mut Self::State,
        device: &Device,
    ) -> Tensor<2>;
}

/// `ids` as a `[1, len]` tensor on `device`.
pub fn row(
    ids: &[i64],
    device: &Device,
) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
}

/// A language model a chat drives: a [`StackedLm`], with what the chat reads
/// of it beside.
pub trait ChatLm: StackedLm + Debug + Send + Sync + 'static {
    /// The head's width: the ids a reply may choose among.
    fn n_vocab(&self) -> usize;

    /// The float dtype its weights are held, and computed, in.
    fn float_dtype(&self) -> DType;
}

impl ChatLm for Qwen3Model {
    fn n_vocab(&self) -> usize {
        Qwen3Meta::n_vocab(self)
    }

    fn float_dtype(&self) -> DType {
        bunsen::burner::tensor::param_float_dtype(&self.embed_tokens.weight)
    }
}

impl ChatLm for GemmaModel {
    fn n_vocab(&self) -> usize {
        GemmaMeta::n_vocab(self)
    }

    fn float_dtype(&self) -> DType {
        bunsen::burner::tensor::param_float_dtype(&self.embed_tokens.weight)
    }
}

/// A loaded stacked language model with what a conversation needs around
/// it: its tokenizer, chat template and generation policy.
///
/// A model's head may be wider than its tokenizer (Qwen3's 151936 rows over
/// 151669 tokens; Gemma 3 4B's 262208 over 262144); a reply never chooses an
/// id the tokenizer cannot spell.
#[derive(Debug)]
pub struct LmChat<M> {
    bundle: Arc<LmConstruct<M>>,
    tokenizer: Arc<dyn ChatTokenizer>,
    template: ChatTemplate,
    generation: GenerationPolicy,
    unspelled: Arc<dyn LogitFilter>,
}

/// A chat over a loaded Qwen3.
pub type Qwen3Chat = LmChat<Qwen3Model>;

/// A chat over a loaded Gemma.
pub type GemmaChat = LmChat<GemmaModel>;

/// An `InvalidResource` for a checkpoint without `what`.
fn missing(what: &str) -> BunsenError {
    BunsenError::invalid_resource(format!("a checkpoint without {what}"))
}

/// The refusal of a checkpoint, `name`, none of whose presets names a chat
/// template.
pub fn no_chat_template(name: &str) -> BunsenError {
    BunsenError::invalid_resource(format!(
        "{name} has no chat template: none of its presets names one (it is a pretrained model, \
         say); give one with `NAME{{generation: {{chat_template: ...}}}}`"
    ))
}

/// The chat template `bundle`'s policy names
/// ([`GenerationPolicy::chat_template`]), its special tokens its
/// `tokenizer_config.json`'s.
fn template_of<M>(bundle: &LmConstruct<M>) -> BunsenResult<ChatTemplate> {
    let config = bundle
        .tokenizer_config()
        .ok_or_else(|| missing("tokenizer_config.json"))?;
    let source = bundle
        .policy
        .generation
        .chat_template
        .as_ref()
        .ok_or_else(|| no_chat_template("the checkpoint"))?;
    ChatTemplate::from_source(source, |key| bundle.file(key), config)
}

impl<M: ChatLm> LmChat<M> {
    /// A chat over `bundle`, through `tokenizer` and `template`, under
    /// `generation`.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when `generation` names no stop token.
    pub fn new(
        bundle: Arc<LmConstruct<M>>,
        tokenizer: Arc<dyn ChatTokenizer>,
        template: ChatTemplate,
        generation: GenerationPolicy,
    ) -> BunsenResult<Self> {
        if generation.stop_ids.is_empty() {
            return Err(missing("a stop token (no generation_config.json)"));
        }
        let unspelled = (0..bundle.model.n_vocab() as i64).filter(|&id| !tokenizer.contains(id));
        Ok(Self {
            unspelled: Arc::new(SuppressTokens::new(unspelled)),
            bundle,
            tokenizer,
            template,
            generation,
        })
    }

    /// The model and its files.
    pub fn bundle(&self) -> &Arc<LmConstruct<M>> {
        &self.bundle
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Arc<dyn ChatTokenizer> {
        &self.tokenizer
    }

    /// The chat template.
    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }
}

impl LmChat<Qwen3Model> {
    /// A chat over a loaded Qwen3, through its tokenizer files
    /// ([`Qwen3Tokenizer`]) and the chat template and stop tokens its presets
    /// resolved to.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the checkpoint lacks `tokenizer.json` or `tokenizer_config.json`,
    /// or its policy names no chat template or no stop token (a repo without
    /// a `generation_config.json`); as each file's reader.
    pub fn from_construct(bundle: Arc<Qwen3Construct>) -> BunsenResult<Self> {
        let config = bundle
            .tokenizer_config()
            .ok_or_else(|| missing("tokenizer_config.json"))?;
        let tokenizer = Qwen3Tokenizer::from_files(
            bundle
                .tokenizer()
                .ok_or_else(|| missing("tokenizer.json"))?,
            Some(config),
        )?;
        let template = template_of(&bundle)?;
        let generation = bundle.policy.generation.clone();
        Self::new(bundle, Arc::new(tokenizer), template, generation)
    }
}

impl LmChat<GemmaModel> {
    /// A chat over a loaded Gemma, through its `tokenizer.json` and
    /// `tokenizer_config.json` ([`HfTokenizer`]) and the chat template and stop
    /// tokens its presets resolved to: its own template where it has one, else
    /// its family's.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the checkpoint lacks `tokenizer.json` or `tokenizer_config.json`,
    /// or its policy names no chat template or no stop token; as each
    /// file's reader.
    pub fn from_construct(bundle: Arc<GemmaConstruct>) -> BunsenResult<Self> {
        let tokenizer = HfTokenizer::from_files(
            bundle
                .tokenizer()
                .ok_or_else(|| missing("tokenizer.json"))?,
            Some(
                bundle
                    .tokenizer_config()
                    .ok_or_else(|| missing("tokenizer_config.json"))?,
            ),
        )?;
        let template = template_of(&bundle)?;
        let generation = bundle.policy.generation.clone();
        Self::new(bundle, Arc::new(tokenizer), template, generation)
    }
}

impl<M: ChatLm> ChatBackend for LmChat<M> {
    type Model = M;
    type State = DecoderStackCache;

    fn model(&self) -> &M {
        &self.bundle.model
    }

    fn render(
        &self,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>> {
        let text = self.template.render(messages, true, enable_thinking)?;
        self.tokenizer.encode(&text)
    }

    fn detokenizer(&self) -> Arc<dyn Detokenizer> {
        self.tokenizer.clone()
    }

    fn generation(&self) -> &GenerationPolicy {
        &self.generation
    }

    fn filters(&self) -> Vec<Arc<dyn LogitFilter>> {
        vec![self.unspelled.clone()]
    }

    fn trained_context(&self) -> Option<usize> {
        self.bundle.max_positions
    }

    fn float_dtype(&self) -> DType {
        self.bundle.model.float_dtype()
    }

    fn decode_state(
        &self,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        self.bundle
            .model
            .decode_state(1, positions, storage, device)
    }

    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut DecoderStackCache,
        next: &'a [i64],
        _device: &Device,
    ) -> &'a [i64] {
        history.rewind(&mut state.cache, next)
    }

    fn prefill(
        &self,
        ids: &[i64],
        state: &mut DecoderStackCache,
        device: &Device,
    ) -> Tensor<2> {
        let model = &self.bundle.model;
        model.prefill(model.embed(row(ids, device)), state)
    }
}
