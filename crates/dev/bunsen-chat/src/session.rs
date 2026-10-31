//! The chat driver: a loaded model and the conversations it holds.
use std::{
    ops::ControlFlow,
    sync::Arc,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        decoding::{
            SearchConfig,
            decode_cached_observed,
        },
        gpts::{
            driver_support::{
                CachedHistory,
                StopStrings,
                TextStream,
            },
            qwen3::Qwen3Bundle,
        },
    },
    ops::transformers::attention::KVCacheStorage,
};
use burn::{
    config::Config,
    tensor::Device,
};

use crate::{
    ChatBackend,
    ChatMessage,
    GenerationDefaults,
    Qwen3Chat,
};

/// The positions a session's cache holds when neither its config nor the
/// checkpoint says.
pub const DEFAULT_CONTEXT: usize = 8192;

/// A loaded checkpoint with what a conversation needs around it, its
/// [`ChatBackend`] (Qwen3's, [`Qwen3Chat`], unless named): the shared,
/// immutable half of a chat. [`session`](Self::session) opens any number of
/// [`ChatSession`]s over one.
#[derive(Debug)]
pub struct ChatModel<B: ChatBackend = Qwen3Chat> {
    backend: B,
}

impl ChatModel<Qwen3Chat> {
    /// A chat over a loaded Qwen3 checkpoint, from the tokenizer, template
    /// and generation files it carries ([`Qwen3Chat::from_bundle`]).
    ///
    /// # Errors
    /// As [`Qwen3Chat::from_bundle`].
    pub fn from_bundle(bundle: Arc<Qwen3Bundle>) -> BunsenResult<Self> {
        Ok(Self::new(Qwen3Chat::from_bundle(bundle)?))
    }
}

impl<B: ChatBackend> ChatModel<B> {
    /// A chat over `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// The checkpoint, and what reads and writes its ids.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The checkpoint's generation defaults.
    pub fn defaults(&self) -> &GenerationDefaults {
        self.backend.defaults()
    }

    /// The float dtype the model's weights are held, and computed, in: the
    /// device's default when it loaded.
    pub fn float_dtype(&self) -> burn::tensor::DType {
        self.backend.float_dtype()
    }

    /// Opens a conversation under `config`, its cache on `device`.
    pub fn session(
        self: &Arc<Self>,
        config: ChatSessionConfig,
        device: &Device,
    ) -> ChatSession<B> {
        let context = config
            .context
            .or(self.backend.trained_context())
            .unwrap_or(DEFAULT_CONTEXT);
        let state = self.backend.decode_state(context, config.storage, device);
        ChatSession {
            model: self.clone(),
            config,
            context,
            device: device.clone(),
            messages: Vec::new(),
            state,
            history: CachedHistory::new(),
        }
    }
}

/// How a [`ChatSession`] generates.
#[derive(Config, Debug)]
pub struct ChatSessionConfig {
    /// The most tokens one reply generates, its stop token included.
    #[config(default = "1024")]
    pub max_new_tokens: usize,

    /// Qwen3's thinking switch, as its chat template reads it; `None` leaves
    /// it to the template (Qwen3 thinks by default).
    #[config(default = "None")]
    pub enable_thinking: Option<bool>,

    /// The search; `None` is the checkpoint's generation defaults. Sessions
    /// decode one row, so a beam search is not one of them.
    #[config(default = "None")]
    pub search: Option<SearchConfig>,

    /// Text that ends a reply where it appears, beyond the stop tokens.
    #[config(default = "Vec::new()")]
    pub stop_strings: Vec<String>,

    /// The positions the session's cache holds: the whole conversation and
    /// a reply; `None` is the checkpoint's trained length, or
    /// [`DEFAULT_CONTEXT`].
    #[config(default = "None")]
    pub context: Option<usize>,

    /// How the cache stores its history.
    #[config(default = "KVCacheStorage::Auto")]
    pub storage: KVCacheStorage,
}

/// Why a reply ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted this stop token.
    StopToken(i64),

    /// This stop string appeared.
    StopString(String),

    /// The reply reached `max_new_tokens`.
    Length,
}

/// One reply, and what it took.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatReply {
    /// The reply, as the model wrote it (its thinking included), cut at a
    /// stop string.
    pub text: String,

    /// The ids generated, the stop token included when one ended them.
    pub ids: Vec<i64>,

    /// Why it ended.
    pub stop: StopReason,

    /// The rendered conversation's length, in tokens.
    pub prompt_tokens: usize,

    /// The tokens of it the turn ran through the model; the rest the cache
    /// already held.
    pub prefilled_tokens: usize,

    /// Time spent on the prefill.
    pub prefill_time: Duration,

    /// Time spent generating.
    pub decode_time: Duration,
}

impl ChatReply {
    /// The reply's thinking (what precedes `</think>`, without the
    /// `<think>` tag), and its answer: the text after it, or all of it.
    pub fn split_thinking(&self) -> (Option<&str>, &str) {
        match self.text.split_once("</think>") {
            Some((thinking, answer)) => (
                Some(
                    thinking
                        .trim_start()
                        .trim_start_matches("<think>")
                        .trim_matches('\n'),
                ),
                answer.trim_start_matches('\n'),
            ),
            None => (None, &self.text),
        }
    }
}

/// One conversation with a [`ChatModel`]: its messages, and the KV cache
/// that holds them.
///
/// Every turn renders the whole conversation through the template, but runs
/// only what the cache does not already hold through the model: the cache
/// is rewound to the prefix the rendering shares with what it holds
/// ([`CachedHistory`]), so a turn's prefill is its new messages, and
/// whatever of the last reply the template re-renders.
#[derive(Debug)]
pub struct ChatSession<B: ChatBackend = Qwen3Chat> {
    model: Arc<ChatModel<B>>,
    config: ChatSessionConfig,
    context: usize,
    device: Device,
    messages: Vec<ChatMessage>,
    state: B::State,
    history: CachedHistory,
}

impl<B: ChatBackend> ChatSession<B> {
    /// The chat this session holds a conversation with.
    pub fn model(&self) -> &Arc<ChatModel<B>> {
        &self.model
    }

    /// The conversation so far.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// The positions the cache holds.
    pub fn context(&self) -> usize {
        self.context
    }

    /// The session's config.
    pub fn config(&self) -> &ChatSessionConfig {
        &self.config
    }

    /// Adds a message without replying: a system prompt, say.
    pub fn push(
        &mut self,
        message: ChatMessage,
    ) {
        self.messages.push(message);
    }

    /// Adds a user message and generates the reply, calling `on_text` with
    /// each piece of it as it is decoded. The reply joins the conversation.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when the
    /// conversation and a reply's budget do not fit the context; as the
    /// template and the tokenizer.
    pub fn send(
        &mut self,
        user: &str,
        on_text: &mut dyn FnMut(&str),
    ) -> BunsenResult<ChatReply> {
        self.messages.push(ChatMessage::user(user));
        let reply = self.generate(on_text);
        match reply {
            Ok(reply) => {
                self.messages
                    .push(ChatMessage::assistant(reply.text.clone()));
                Ok(reply)
            }
            Err(e) => {
                self.messages.pop();
                Err(e)
            }
        }
    }

    /// Generates the assistant's next turn from the conversation as it is.
    fn generate(
        &mut self,
        on_text: &mut dyn FnMut(&str),
    ) -> BunsenResult<ChatReply> {
        let chat = self.model.clone();
        let backend = &chat.backend;
        let prompt = backend.render(&self.messages, self.config.enable_thinking)?;
        let budget = self.config.max_new_tokens;
        if prompt.len() + budget > self.context {
            return Err(BunsenError::policy(format!(
                "the conversation ({} tokens) and a reply ({budget}) overrun the context ({})",
                prompt.len(),
                self.context
            )));
        }

        let start = Instant::now();
        let todo = backend
            .rewind(&mut self.history, &mut self.state, &prompt, &self.device)
            .to_vec();
        let first = backend.prefill(&todo, &mut self.state, &self.device);
        self.history.extend(&todo);
        let prefill_time = start.elapsed();

        let search = self
            .config
            .search
            .clone()
            .unwrap_or_else(|| backend.defaults().search.clone());
        let stops = &backend.defaults().stop_ids;
        let mut searcher = search.init_search(stops, stops[0]);
        let mut filters = backend.filters();
        filters.extend(search.sampling_filters());

        let start = Instant::now();
        let mut stream = TextStream::new(backend.detokenizer());
        let mut stop_strings = StopStrings::new(self.config.stop_strings.clone());
        let mut shown = String::new();
        let mut failed = None;
        let out = decode_cached_observed(
            backend.model(),
            &mut self.state,
            vec![Vec::new()],
            first,
            budget,
            searcher.as_mut(),
            &filters,
            &mut |tokens| {
                let Some(&id) = tokens[0].get(stream.ids().len()) else {
                    return ControlFlow::Continue(());
                };
                match stream.push(id) {
                    Ok(delta) => {
                        let safe = stop_strings.push(&delta);
                        on_text(&safe);
                        shown.push_str(&safe);
                    }
                    Err(e) => {
                        failed = Some(e);
                        return ControlFlow::Break(());
                    }
                }
                if stop_strings.stopped().is_some() {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        if let Some(e) = failed {
            return Err(e);
        }
        let candidate = out
            .into_iter()
            .next()
            .and_then(|group| group.into_iter().next())
            .expect("one row, one candidate");
        if stop_strings.stopped().is_none() {
            let tail = stop_strings.push(&stream.finish()?);
            let tail = tail + &stop_strings.finish();
            on_text(&tail);
            shown.push_str(&tail);
        }
        let decode_time = start.elapsed();

        // A stop token is chosen but never stepped; a reply cut by the
        // budget or a stop string leaves its last token unstepped too.
        let stepped = match candidate.stop {
            Some(_) => candidate.tokens.len(),
            None => candidate.tokens.len().saturating_sub(1),
        };
        self.history.extend(&candidate.tokens[..stepped]);

        let stop = match (stop_strings.stopped(), candidate.stop) {
            (Some(s), _) => StopReason::StopString(s.to_string()),
            (None, Some(id)) => StopReason::StopToken(id),
            (None, None) => StopReason::Length,
        };
        Ok(ChatReply {
            text: shown,
            ids: candidate.tokens_with_stop(),
            stop,
            prompt_tokens: prompt.len(),
            prefilled_tokens: todo.len(),
            prefill_time,
            decode_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use bunsen::{
        burner::module::ModuleInit,
        kits::gpts::qwen3::{
            GENERATION_CONFIG,
            Qwen3Config,
            Qwen3Model,
            TOKENIZER,
            TOKENIZER_CONFIG,
        },
        support::testing::cpu_device,
    };

    use super::*;
    use crate::tokenizer::toy_tokenizer_json;

    /// A chat over an untrained toy Qwen3, with the toy tokenizer, a `ChatML`
    /// template, and greedy defaults, its files in `dir`.
    fn toy_chat(dir: &std::path::Path) -> Arc<ChatModel> {
        let template = "{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
        let files = [
            (TOKENIZER, "tokenizer.json", toy_tokenizer_json()),
            (
                TOKENIZER_CONFIG,
                "tokenizer_config.json",
                serde_json::json!({ "chat_template": template }).to_string(),
            ),
            (
                GENERATION_CONFIG,
                "generation_config.json",
                r#"{"eos_token_id": 301, "do_sample": false}"#.to_string(),
            ),
        ];
        let device = cpu_device();
        let model: Qwen3Model = Qwen3Config::new(310, 12, 2, 3, 1, 6, 20).init(&device);
        let mut bundle = Qwen3Bundle::new(model).with_max_positions(Some(256));
        for (key, name, body) in files {
            let path = dir.join(name);
            std::fs::write(&path, body).unwrap();
            bundle = bundle.with_file(key, path);
        }
        Arc::new(ChatModel::from_bundle(Arc::new(bundle)).unwrap())
    }

    /// Two turns: each reply streams as it is decoded, joins the
    /// conversation, and the second turn prefills only what the cache does
    /// not hold.
    #[test]
    fn test_turns_stream_and_reuse_the_cache() {
        let dir = std::env::temp_dir().join(format!("bunsen-chat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let chat = toy_chat(&dir);
        let config = ChatSessionConfig::new().with_max_new_tokens(6);
        let mut session = chat.session(config, &cpu_device());
        assert_eq!(session.context(), 256);

        let mut streamed = String::new();
        let first = session
            .send("hello", &mut |delta| streamed.push_str(delta))
            .unwrap();
        assert_eq!(streamed, first.text, "the deltas are the reply");
        assert_eq!(first.prefilled_tokens, first.prompt_tokens, "a cold cache");
        assert!(!first.ids.is_empty() && first.ids.len() <= 6);
        assert_eq!(session.messages().len(), 2);

        let second = session.send("again", &mut |_| {}).unwrap();
        assert!(
            second.prefilled_tokens < second.prompt_tokens,
            "{} of {} prefilled",
            second.prefilled_tokens,
            second.prompt_tokens
        );
        assert_eq!(session.messages().len(), 4);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A conversation that cannot fit is refused, and leaves no trace.
    #[test]
    fn test_overrun_is_refused() {
        let dir = std::env::temp_dir().join(format!("bunsen-chat-overrun-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let chat = toy_chat(&dir);
        let config = ChatSessionConfig::new()
            .with_max_new_tokens(16)
            .with_context(Some(20));
        let mut session = chat.session(config, &cpu_device());
        bunsen::errors::testing::ErrorMatcher::kind(bunsen::errors::BunsenErrorKind::Policy)
            .message_contains("overrun")
            .assert_err(&session.send("hello hello hello", &mut |_| {}));
        assert!(session.messages().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_split_thinking() {
        let reply = |text: &str| ChatReply {
            text: text.to_string(),
            ids: Vec::new(),
            stop: StopReason::Length,
            prompt_tokens: 0,
            prefilled_tokens: 0,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
        };
        assert_eq!(
            reply("<think>\nhm\n</think>\n\nBlue.").split_thinking(),
            (Some("hm"), "Blue.")
        );
        assert_eq!(reply("Blue.").split_thinking(), (None, "Blue."));
    }
}
