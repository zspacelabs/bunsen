//! A nanochat checkpoint as a chat's model.
use std::sync::Arc;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        decoding::{
            LogitFilter,
            SearchConfig,
        },
        gpts::{
            driver_support::CachedHistory,
            nanochat::{
                NanoChatBundle,
                NanoChatDecodeState,
                NanoChatGpt,
            },
        },
        tokens::Detokenizer,
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

use crate::{
    ChatBackend,
    ChatMessage,
    GenerationDefaults,
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
    bundle: Arc<NanoChatBundle>,
    tokenizer: Arc<NanoChatTokenizer>,
    layout: NanoChatLayout,
    defaults: GenerationDefaults,
}

impl NanoChatChat {
    /// A chat over a loaded checkpoint, from the `tokenizer.pkl` it carries.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the checkpoint lacks `tokenizer.pkl`, or its tokenizer the
    /// layout's special tokens; as [`NanoChatTokenizer::from_file`].
    pub fn from_bundle(bundle: Arc<NanoChatBundle>) -> BunsenResult<Self> {
        let path = bundle
            .tokenizer()
            .ok_or_else(|| BunsenError::invalid_resource("a checkpoint without tokenizer.pkl"))?;
        let tokenizer = NanoChatTokenizer::from_file(path)?;
        let layout = NanoChatLayout::from_tokenizer(&tokenizer)?;
        let defaults = GenerationDefaults {
            stop_ids: layout.stop_ids(),
            search: SearchConfig::new()
                .with_temperature(0.6)
                .with_top_k(Some(50)),
        };
        Ok(Self {
            bundle,
            tokenizer: Arc::new(tokenizer),
            layout,
            defaults,
        })
    }

    /// The model and its files.
    pub fn bundle(&self) -> &Arc<NanoChatBundle> {
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

    fn defaults(&self) -> &GenerationDefaults {
        &self.defaults
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
