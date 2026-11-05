//! A nanochat checkpoint as a chat's model.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        language::{
            driver::CachedHistory,
            nanochat::{
                NanoChatConstruct,
                NanoChatDecodeState,
                NanoChatGpt,
            },
        },
        support::{
            decoding::LogitFilter,
            model_config::{
                GenerationPolicy,
                SamplingPolicy,
            },
            tokens::Detokenizer,
        },
    },
    ops::transformers::attention::KVCacheStorage,
};
use burn::{
    Tensor,
    tensor::{
        DType,
        Device,
    },
};

use crate::chat::{
    ChatBackend,
    ChatMessage,
    NanoChatLayout,
    NanoChatTokenizer,
    row,
};

/// A loaded nanochat with what a conversation needs around it: its
/// tokenizer, its conversation layout, and upstream's chat defaults.
///
/// nanochat's checkpoints carry no generation config; the defaults are
/// upstream's chat scripts': temperature 0.6, top-k 50, a reply ending at
/// `<|assistant_end|>` or `<|bos|>`. It has no thinking switch.
#[derive(Debug)]
pub struct NanoChatChat {
    bundle: Arc<NanoChatConstruct>,
    tokenizer: Arc<NanoChatTokenizer>,
    layout: NanoChatLayout,
    generation: GenerationPolicy,
}

impl NanoChatChat {
    /// A chat over a loaded checkpoint, from the `tokenizer.pkl` it carries.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the checkpoint lacks `tokenizer.pkl`, or its tokenizer the
    /// layout's special tokens; as [`NanoChatTokenizer::from_file`].
    pub fn from_construct(bundle: Arc<NanoChatConstruct>) -> BunsenResult<Self> {
        let path = bundle
            .tokenizer()
            .ok_or_else(|| BunsenError::invalid_resource("a checkpoint without tokenizer.pkl"))?;
        let tokenizer = NanoChatTokenizer::from_file(path)?;
        let layout = NanoChatLayout::from_tokenizer(&tokenizer)?;
        let generation = GenerationPolicy {
            stop_ids: layout.stop_ids(),
            sampling: Some(SamplingPolicy {
                temperature: 0.6,
                top_k: Some(50),
                top_p: None,
                min_p: None,
            }),
            ..Default::default()
        };
        Ok(Self {
            bundle,
            tokenizer: Arc::new(tokenizer),
            layout,
            generation,
        })
    }

    /// The model and its files.
    pub fn bundle(&self) -> &Arc<NanoChatConstruct> {
        &self.bundle
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Arc<NanoChatTokenizer> {
        &self.tokenizer
    }

    /// The conversation layout.
    pub fn layout(&self) -> &NanoChatLayout {
        &self.layout
    }
}

impl ChatBackend for NanoChatChat {
    type Model = NanoChatGpt;
    type State = NanoChatDecodeState;

    fn model(&self) -> &NanoChatGpt {
        &self.bundle.model
    }

    fn render(
        &self,
        messages: &[ChatMessage],
        _enable_thinking: Option<bool>,
    ) -> BunsenResult<Vec<i64>> {
        self.layout.render(&self.tokenizer, messages)
    }

    fn detokenizer(&self) -> Arc<dyn Detokenizer> {
        self.tokenizer.clone()
    }

    fn generation(&self) -> &GenerationPolicy {
        &self.generation
    }

    fn filters(&self) -> Vec<Arc<dyn LogitFilter>> {
        Vec::new()
    }

    fn trained_context(&self) -> Option<usize> {
        Some(self.bundle.sequence_len())
    }

    fn float_dtype(&self) -> DType {
        self.bundle.model.float_dtype()
    }

    fn decode_state(
        &self,
        positions: usize,
        storage: KVCacheStorage,
        _device: &Device,
    ) -> NanoChatDecodeState {
        self.bundle.model.decode_state(1, positions, storage)
    }

    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut NanoChatDecodeState,
        next: &'a [i64],
        device: &Device,
    ) -> &'a [i64] {
        let todo = history.rewind(&mut state.cache, next);
        let shared = next.len() - todo.len();
        // The token before the prefill, which a smeared model mixes in.
        state.last = (shared > 0).then(|| row(&next[shared - 1..shared], device));
        todo
    }

    fn prefill(
        &self,
        ids: &[i64],
        state: &mut NanoChatDecodeState,
        device: &Device,
    ) -> Tensor<2> {
        self.bundle.model.prefill(row(ids, device), state)
    }
}
