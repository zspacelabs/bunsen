//! The model a chat holds, whichever kit it is.
use std::{
    fmt::Debug,
    sync::Arc,
};

use bunsen::{
    blocks::transformers::models::language_decoder::{
        LmDecoderMeta,
        LmDecoderOps,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::{
            driver::{
                CachedHistory,
                CachedLm,
            },
            gemma::GemmaModel,
            qwen3::Qwen3Model,
            qwen3_5::Qwen3_5Model,
        },
        support::{
            construct::LmConstruct,
            decoding::{
                CachedStep,
                LogitFilter,
                SuppressTokens,
            },
            model_config::GenerationPolicy,
            tokens::{
                BunsenTokenizer,
                Detokenizer,
                Tokenizer,
            },
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
};

/// A checkpoint a [`ChatModel`](crate::chat::ChatModel) holds a conversation
/// with: its model, as a cached step, and what turns a conversation into the
/// ids it reads and its ids back into text.
///
/// [`LmChat`] is a stacked language model's: [`Qwen3Chat`], [`GemmaChat`],
/// [`Qwen3_5Chat`].
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

    /// Where, in the prompt [`render`](Self::render) gives for `messages`,
    /// the decode state should be [snapshot](Self::snapshot): the end of the
    /// conversation before the assistant's turn opens, where the next turn's
    /// rendering will part from what the state holds (a template that
    /// drops a reply's thinking parts inside it). `None` (the default) for a
    /// backend whose state rewinds to any position, or a prompt that does
    /// not split there.
    ///
    /// # Errors
    /// As the checkpoint's renderer.
    fn snapshot_point(
        &self,
        _messages: &[ChatMessage],
        _enable_thinking: Option<bool>,
    ) -> BunsenResult<Option<usize>> {
        Ok(None)
    }

    /// Keeps `state` as it is now, for a later [`rewind`](Self::rewind) to
    /// go back to; a no-op by default.
    fn snapshot(
        &self,
        _state: &mut Self::State,
    ) {
    }

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

/// How a chat turns a conversation into the ids its model reads, through
/// its tokenizer: a chat template ([`ChatTemplate`]), or a model's own
/// layout (nanochat's).
pub trait ChatRenderer: Debug + Send + Sync {
    /// The prompt for the assistant's next turn of `messages`.
    /// `enable_thinking` is a thinking switch, for a renderer that has one.
    ///
    /// # Errors
    /// As the renderer and the tokenizer.
    fn render_prompt(
        &self,
        tokenizer: &dyn Tokenizer,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>>;

    /// The length, in ids, of `messages` rendered without the assistant's
    /// turn opening, where [`render_prompt`](Self::render_prompt)'s prompt is
    /// that and more; `None` (the default) where the prompt does not split
    /// there.
    ///
    /// # Errors
    /// As the renderer and the tokenizer.
    fn closed_len(
        &self,
        _tokenizer: &dyn Tokenizer,
        _messages: &[ChatMessage],
        _enable_thinking: Option<bool>,
    ) -> BunsenResult<Option<usize>> {
        Ok(None)
    }
}

impl ChatRenderer for ChatTemplate {
    fn render_prompt(
        &self,
        tokenizer: &dyn Tokenizer,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>> {
        tokenizer.encode(&self.render(messages, true, enable_thinking)?)
    }

    fn closed_len(
        &self,
        tokenizer: &dyn Tokenizer,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Option<usize>> {
        let open = self.render(messages, true, enable_thinking)?;
        let closed = self.render(messages, false, enable_thinking)?;
        if !open.starts_with(&closed) || open.len() == closed.len() {
            return Ok(None);
        }
        let (open, closed) = (tokenizer.encode(&open)?, tokenizer.encode(&closed)?);
        Ok(open.starts_with(&closed).then_some(closed.len()))
    }
}

/// What a chat reads a conversation and its replies through: the
/// tokenizer, the renderer, and the generation policy (its stop tokens
/// among it).
#[derive(Debug)]
pub struct ChatParts {
    /// The tokenizer.
    pub tokenizer: Arc<dyn Tokenizer>,

    /// The renderer.
    pub renderer: Arc<dyn ChatRenderer>,

    /// The generation policy.
    pub generation: GenerationPolicy,
}

/// A language model a chat drives: a [`CachedLm`], with its meta and its
/// forward-free API, and the [`ChatParts`] its checkpoint's files give.
pub trait ChatLm:
    CachedLm + LmDecoderMeta + LmDecoderOps + Debug + Send + Sync + Sized + 'static
{
    /// The tokenizer, renderer and generation policy a chat over `bundle`
    /// reads through.
    ///
    /// # Errors
    /// The kit's: a checkpoint without the files they are read from.
    fn chat_parts(bundle: &LmConstruct<Self>) -> BunsenResult<ChatParts>;
}

/// The [`ChatParts`] of a `transformers` checkpoint: its `tokenizer.json`
/// and `tokenizer_config.json` ([`BunsenTokenizer`]), and the chat template
/// and stop tokens its presets resolved to: its own template where it has
/// one, else its family's.
///
/// # Errors
/// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
/// when the checkpoint lacks `tokenizer.json` or `tokenizer_config.json`, or
/// its policy names no chat template; as each file's reader.
pub fn template_chat_parts<M>(bundle: &LmConstruct<M>) -> BunsenResult<ChatParts> {
    let tokenizer = BunsenTokenizer::from_files(
        bundle
            .tokenizer()
            .ok_or_else(|| missing("tokenizer.json"))?,
        Some(
            bundle
                .tokenizer_config()
                .ok_or_else(|| missing("tokenizer_config.json"))?,
        ),
    )?;
    Ok(ChatParts {
        tokenizer: Arc::new(tokenizer),
        renderer: Arc::new(template_of(bundle)?),
        generation: bundle.policy.generation.clone(),
    })
}

impl ChatLm for Qwen3Model {
    fn chat_parts(bundle: &LmConstruct<Self>) -> BunsenResult<ChatParts> {
        template_chat_parts(bundle)
    }
}

impl ChatLm for GemmaModel {
    fn chat_parts(bundle: &LmConstruct<Self>) -> BunsenResult<ChatParts> {
        template_chat_parts(bundle)
    }
}

impl ChatLm for Qwen3_5Model {
    fn chat_parts(bundle: &LmConstruct<Self>) -> BunsenResult<ChatParts> {
        template_chat_parts(bundle)
    }
}

/// A loaded language model with what a conversation needs around it: its
/// tokenizer, its renderer and its generation policy ([`ChatParts`]).
///
/// A model's head may be wider than its tokenizer (Qwen3's 151936 rows over
/// 151669 tokens; Gemma 3 4B's 262208 over 262144); a reply never chooses an
/// id the tokenizer cannot spell.
#[derive(Debug)]
pub struct LmChat<M> {
    bundle: Arc<LmConstruct<M>>,
    tokenizer: Arc<dyn Tokenizer>,
    renderer: Arc<dyn ChatRenderer>,
    generation: GenerationPolicy,
    unspelled: Arc<dyn LogitFilter>,
}

/// A chat over a loaded Qwen3.
pub type Qwen3Chat = LmChat<Qwen3Model>;

/// A chat over a loaded Gemma.
pub type GemmaChat = LmChat<GemmaModel>;

/// A chat over a loaded dense Qwen3.5: its decode state carries the linear
/// layers' states, so a turn whose rendering parts from the last one's
/// prefills the whole conversation again ([`CachedHistory::rewind`]).
pub type Qwen3_5Chat = LmChat<Qwen3_5Model>;

/// The filter that keeps a reply off the ids of a head `n_vocab` wide that
/// `tokenizer` cannot spell: a head may be wider than its tokenizer.
pub fn unspelled(
    n_vocab: usize,
    tokenizer: &dyn Tokenizer,
) -> Arc<dyn LogitFilter> {
    Arc::new(SuppressTokens::new(
        (0..n_vocab as i64).filter(|&id| !tokenizer.contains(id)),
    ))
}

/// An `InvalidResource` for a checkpoint without `what`.
pub(crate) fn missing(what: &str) -> BunsenError {
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
    /// A chat over `bundle`, through `parts`.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the generation policy names no stop token.
    pub fn new(
        bundle: Arc<LmConstruct<M>>,
        parts: ChatParts,
    ) -> BunsenResult<Self> {
        let ChatParts {
            tokenizer,
            renderer,
            generation,
        } = parts;
        if generation.stop_ids.is_empty() {
            return Err(missing("a stop token (no generation_config.json)"));
        }
        Ok(Self {
            unspelled: unspelled(bundle.model.n_logits(), tokenizer.as_ref()),
            bundle,
            tokenizer,
            renderer,
            generation,
        })
    }

    /// A chat over a loaded model, through the parts its files give
    /// ([`ChatLm::chat_parts`]).
    ///
    /// # Errors
    /// As [`ChatLm::chat_parts`] and [`new`](Self::new).
    pub fn from_construct(bundle: Arc<LmConstruct<M>>) -> BunsenResult<Self> {
        let parts = M::chat_parts(&bundle)?;
        Self::new(bundle, parts)
    }

    /// The model and its files.
    pub fn bundle(&self) -> &Arc<LmConstruct<M>> {
        &self.bundle
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }

    /// The renderer.
    pub fn renderer(&self) -> &Arc<dyn ChatRenderer> {
        &self.renderer
    }
}

impl<M: ChatLm> ChatBackend for LmChat<M>
where
    M::State: Debug + Send,
{
    type Model = M;
    type State = M::State;

    fn model(&self) -> &M {
        &self.bundle.model
    }

    fn render(
        &self,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>> {
        self.renderer
            .render_prompt(self.tokenizer.as_ref(), messages, enable_thinking)
    }

    /// For a model whose state cannot rewind to any position (one with
    /// linear-attention layers): the length of the conversation rendered
    /// without the assistant's turn opening, where the prompt is that and
    /// more.
    fn snapshot_point(
        &self,
        messages: &[ChatMessage],
        enable_thinking: Option<bool>,
    ) -> BunsenResult<Option<usize>> {
        if self.bundle.model.rewinds_anywhere() {
            return Ok(None);
        }
        self.renderer
            .closed_len(self.tokenizer.as_ref(), messages, enable_thinking)
    }

    fn snapshot(
        &self,
        state: &mut M::State,
    ) {
        CachedLm::snapshot(&self.bundle.model, state);
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
        LmDecoderOps::float_dtype(&self.bundle.model)
    }

    fn decode_state(
        &self,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> M::State {
        CachedLm::decode_state(&self.bundle.model, 1, positions, storage, device)
    }

    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut M::State,
        next: &'a [i64],
        device: &Device,
    ) -> &'a [i64] {
        CachedLm::rewind(&self.bundle.model, history, state, next, device)
    }

    fn prefill(
        &self,
        ids: &[i64],
        state: &mut M::State,
        device: &Device,
    ) -> Tensor<2> {
        self.bundle.model.prefill_ids(row(ids, device), state)
    }
}
