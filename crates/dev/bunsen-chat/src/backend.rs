//! The model a chat holds: what [`ChatSession`](crate::ChatSession) needs of
//! a checkpoint, whichever kit it is.
use std::{
    fmt::Debug,
    sync::Arc,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        decoding::{
            CachedStep,
            LogitFilter,
            SuppressTokens,
        },
        gpts::{
            driver_support::CachedHistory,
            qwen3::{
                Qwen3Bundle,
                Qwen3DecodeState,
                Qwen3Meta,
                Qwen3Model,
            },
        },
        tokens::Detokenizer,
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

use crate::{
    ChatMessage,
    ChatTemplate,
    GenerationDefaults,
    Qwen3Tokenizer,
};

/// A checkpoint a [`ChatModel`](crate::ChatModel) holds a conversation with:
/// its model, as a cached step, and what turns a conversation into the ids
/// it reads and its ids back into text.
///
/// [`Qwen3Chat`] is Qwen3's.
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

    /// The stop tokens and sampling a reply uses unless told otherwise.
    fn defaults(&self) -> &GenerationDefaults;

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

/// A loaded Qwen3 with what a conversation needs around it: its tokenizer,
/// chat template and generation defaults.
///
/// A Qwen3's head is wider than its tokenizer (151936 rows, 151669 tokens);
/// a reply never chooses an id the tokenizer cannot spell.
#[derive(Debug)]
pub struct Qwen3Chat {
    bundle: Arc<Qwen3Bundle>,
    tokenizer: Arc<Qwen3Tokenizer>,
    template: ChatTemplate,
    defaults: GenerationDefaults,
    unspelled: Arc<dyn LogitFilter>,
}

impl Qwen3Chat {
    /// A chat over a loaded checkpoint, from the tokenizer, template and
    /// generation files it carries.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the checkpoint lacks `tokenizer.json`, `tokenizer_config.json`
    /// or `generation_config.json`; as each file's reader.
    pub fn from_bundle(bundle: Arc<Qwen3Bundle>) -> BunsenResult<Self> {
        let missing =
            |what: &str| BunsenError::invalid_resource(format!("a checkpoint without {what}"));
        let config = bundle
            .tokenizer_config()
            .ok_or_else(|| missing("tokenizer_config.json"))?;
        let tokenizer = Qwen3Tokenizer::from_files(
            bundle
                .tokenizer()
                .ok_or_else(|| missing("tokenizer.json"))?,
            Some(config),
        )?;
        let template = ChatTemplate::from_file(config)?;
        let defaults = GenerationDefaults::from_file(
            bundle
                .generation_config()
                .ok_or_else(|| missing("generation_config.json"))?,
        )?;
        let unspelled = (0..bundle.model.n_vocab() as i64).filter(|&id| !tokenizer.contains(id));
        Ok(Self {
            unspelled: Arc::new(SuppressTokens::new(unspelled)),
            bundle,
            tokenizer: Arc::new(tokenizer),
            template,
            defaults,
        })
    }

    /// The model and its files.
    pub fn bundle(&self) -> &Arc<Qwen3Bundle> {
        &self.bundle
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Arc<Qwen3Tokenizer> {
        &self.tokenizer
    }

    /// The chat template.
    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }
}

impl ChatBackend for Qwen3Chat {
    type Model = Qwen3Model;
    type State = Qwen3DecodeState;

    fn model(&self) -> &Qwen3Model {
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

    fn defaults(&self) -> &GenerationDefaults {
        &self.defaults
    }

    fn filters(&self) -> Vec<Arc<dyn LogitFilter>> {
        vec![self.unspelled.clone()]
    }

    fn trained_context(&self) -> Option<usize> {
        self.bundle.max_positions
    }

    fn float_dtype(&self) -> DType {
        bunsen::burner::tensor::param_float_dtype(&self.bundle.model.embed_tokens.weight)
    }

    fn decode_state(
        &self,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> Qwen3DecodeState {
        self.bundle
            .model
            .decode_state(1, positions, storage, device)
    }

    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut Qwen3DecodeState,
        next: &'a [i64],
        _device: &Device,
    ) -> &'a [i64] {
        history.rewind(&mut state.cache, next)
    }

    fn prefill(
        &self,
        ids: &[i64],
        state: &mut Qwen3DecodeState,
        device: &Device,
    ) -> Tensor<2> {
        let model = &self.bundle.model;
        model.prefill(model.embed(row(ids, device)), state)
    }
}
