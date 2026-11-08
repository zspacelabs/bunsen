//! The flags `chat run` and `chat repl` share.

use std::{
    io::{
        BufRead,
        Write,
    },
    sync::Arc,
};

use bunsen::{
    errors::{
        BunsenResult,
        sys_at,
    },
    kits::{
        language::qwen3::pretrained::DEFAULT_QWEN3_MODEL,
        support::decoding::SearchConfig,
    },
    ops::transformers::attention::KVCacheStorage,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::{
        BackendArgs,
        DeviceSession,
    },
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::{
    chat::{
        ChatBackend,
        ChatMessage,
        ChatModel,
        ChatModelFamily,
        ChatModelFamilyFlags,
        ChatReply,
        ChatSession,
        ChatSessionConfig,
        LoadedChat,
        search_config,
    },
    support::{
        PresetArgs,
        with_presets,
    },
};

/// How the checkpoint is loaded and a reply generated.
#[derive(clap::Args, Debug)]
pub struct ChatGenerateFlags {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device. The checkpoint loads at the precision it ships in where
    /// the device accelerates it, else in f32: Qwen3's bf16 on cuda and
    /// metal, f32 on vulkan, wgpu and the CPU (`LoadPolicy::performance`).
    #[clap(flatten)]
    device: BackendArgs,

    /// The checkpoint: a name `list` shows, or `hf:org/repo`, a Qwen3. The
    /// family's default when omitted; outside `family`, the name picks the
    /// family, and Qwen3's default is the default.
    #[arg(long)]
    model: Option<String>,

    #[clap(flatten)]
    presets: PresetArgs,

    /// A system prompt.
    #[arg(long)]
    system: Option<String>,

    /// The most tokens a reply generates.
    #[arg(long, default_value_t = 1024)]
    max_new_tokens: usize,

    /// Decode greedily, whatever the checkpoint's generation defaults say.
    #[arg(long)]
    greedy: bool,

    /// The sampling temperature, over the checkpoint's.
    #[arg(long)]
    temperature: Option<f64>,

    /// Sample from the top-k tokens only, over the checkpoint's.
    #[arg(long)]
    top_k: Option<usize>,

    /// Sample from the top-p nucleus only, over the checkpoint's.
    #[arg(long)]
    top_p: Option<f64>,

    /// Seed the device's random numbers, for a repeatable sampled reply.
    #[arg(long)]
    seed: Option<u64>,

    /// The positions the KV cache holds; the checkpoint's trained length by
    /// default.
    #[arg(long)]
    context: Option<usize>,

    /// How the KV cache stores its history.
    #[arg(long, value_enum, default_value_t = ChatKvStorage::Auto)]
    kv_storage: ChatKvStorage,

    /// Print, to stderr after each reply, the model's lookup expression
    /// (its name and presets) on a line of its own, then the reply's token
    /// counts and speeds.
    #[arg(long)]
    stats: bool,
}

/// `--kv-storage`.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum ChatKvStorage {
    Auto,
    Concat,
    PerLayer,
    MultiLayer,
}

impl ChatKvStorage {
    fn storage(self) -> KVCacheStorage {
        match self {
            Self::Auto => KVCacheStorage::Auto,
            Self::Concat => KVCacheStorage::Concat,
            Self::PerLayer => KVCacheStorage::PerLayer,
            Self::MultiLayer => KVCacheStorage::MultiLayer,
        }
    }
}

impl ChatGenerateFlags {
    /// The checkpoint these flags name, and its family: `family`'s (its
    /// default when `--model` is not given), or outside `family` the one
    /// `--model` names.
    pub(super) fn checkpoint(
        &self,
        family: Option<ChatModelFamily>,
    ) -> BunsenResult<(ChatModelFamily, String)> {
        Ok(match family {
            Some(family) => (
                family,
                self.model
                    .clone()
                    .unwrap_or_else(|| family.default_model().to_string()),
            ),
            None => {
                let name = self.model.as_deref().unwrap_or(DEFAULT_QWEN3_MODEL);
                (ChatModelFamily::of(name)?, name.to_string())
            }
        })
    }

    /// Opens the device, and loads the checkpoint on it: `family`'s, or
    /// outside `family` whichever `--model` names. Also returns the lookup
    /// expression it loaded: the checkpoint's name, with the presets
    /// `--preset` and `family`'s flags add.
    pub(super) fn load(
        &self,
        family: Option<ChatModelFamily>,
        presets: &[&str],
    ) -> BunsenResult<(DeviceSession, LoadedChat, String)> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        if let Some(seed) = self.seed {
            device.seed(seed);
        }
        let (family, name) = self.checkpoint(family)?;
        let mut name = self.presets.apply(&name)?;
        // nanochat has no thinking switch: the flags select only Qwen3's
        // presets.
        if family == ChatModelFamily::Qwen3 && !presets.is_empty() {
            name = with_presets(&name, presets)?;
        }
        let chat = family.load(&name, &cache, &device)?;
        Ok((device, chat, name))
    }

    /// The session config these flags and `family`'s give, over `chat`'s
    /// defaults.
    fn config<B: ChatBackend>(
        &self,
        chat: &ChatModel<B>,
    ) -> ChatSessionConfig {
        let defaults = search_config(chat.generation());
        let search = if self.greedy {
            SearchConfig::new()
        } else {
            SearchConfig {
                temperature: self.temperature.unwrap_or(defaults.temperature),
                top_k: self.top_k.or(defaults.top_k),
                top_p: self.top_p.or(defaults.top_p),
                ..defaults
            }
        };
        ChatSessionConfig::new()
            .with_max_new_tokens(self.max_new_tokens)
            .with_search(Some(search))
            .with_context(self.context)
            .with_storage(self.kv_storage.storage())
    }

    /// A session over `chat`, opened with the system prompt.
    fn session<B: ChatBackend>(
        &self,
        chat: &Arc<ChatModel<B>>,
        config: ChatSessionConfig,
        device: &DeviceSession,
    ) -> ChatSession<B> {
        let mut session = chat.session(config, device);
        if let Some(system) = &self.system {
            session.push(ChatMessage::system(system.clone()));
        }
        session
    }

    /// Sends `user`, streaming the reply to stdout (its thinking only when
    /// `family`'s flags show it), and reports it under `--stats`: `model`,
    /// the lookup expression loaded, on a line of its own, then the reply's
    /// numbers.
    fn reply<B: ChatBackend>(
        &self,
        session: &mut ChatSession<B>,
        family: &impl ChatModelFamilyFlags,
        model: &str,
        user: &str,
    ) -> BunsenResult<ChatReply> {
        let mut out = std::io::stdout().lock();
        let mut hidden = ChatThinkingFilter::new(family.hide_thinking());
        let mut failed = None;
        let reply = session.send(user, &mut |delta| {
            let shown = hidden.push(delta);
            if let Err(e) = out.write_all(shown.as_bytes()).and_then(|()| out.flush()) {
                failed.get_or_insert(e);
            }
        })?;
        writeln!(out).map_err(sys_at("write to", "stdout"))?;
        if !hidden.done {
            eprintln!("[the reply ran out of tokens while thinking; --show-thinking shows it]");
        }
        if let Some(e) = failed {
            return Err(sys_at("write to", "stdout")(e));
        }
        if self.stats {
            let rate = |n: usize, t: std::time::Duration| n as f64 / t.as_secs_f64().max(1e-9);
            eprintln!("[model {model}]");
            eprintln!(
                "[{:?}; prompt {} tokens, {} prefilled in {:.2} s ({:.0} tok/s); reply {} tokens in {:.2} s ({:.1} tok/s); {:?}]",
                session.model().float_dtype(),
                reply.prompt_tokens,
                reply.prefilled_tokens,
                reply.prefill_time.as_secs_f64(),
                rate(reply.prefilled_tokens, reply.prefill_time),
                reply.ids.len(),
                reply.decode_time.as_secs_f64(),
                rate(reply.ids.len(), reply.decode_time),
                reply.stop,
            );
        }
        Ok(reply)
    }

    /// Answers `prompt` with `model`, the lookup expression loaded, once.
    pub(super) fn run_once<B: ChatBackend>(
        &self,
        chat: &Arc<ChatModel<B>>,
        family: &impl ChatModelFamilyFlags,
        device: &DeviceSession,
        model: &str,
        prompt: &str,
    ) -> BunsenResult<()> {
        let mut session = self.session(chat, self.config(chat), device);
        self.reply(&mut session, family, model, prompt)?;
        Ok(())
    }

    /// Holds a conversation on stdin with `model`, the lookup expression
    /// loaded.
    pub(super) fn converse<B: ChatBackend>(
        &self,
        chat: &Arc<ChatModel<B>>,
        family: &impl ChatModelFamilyFlags,
        device: &DeviceSession,
        model: &str,
    ) -> BunsenResult<()> {
        let config = self.config(chat);
        let mut session = self.session(chat, config.clone(), device);
        let stdin = std::io::stdin();
        let mut lines = stdin.lock().lines();
        loop {
            eprint!("> ");
            let Some(line) = lines.next() else { break };
            let line = line.map_err(sys_at("read", "stdin"))?;
            match line.trim() {
                "" => continue,
                "/quit" => break,
                "/reset" => {
                    session = self.session(chat, config.clone(), device);
                    eprintln!("[a new conversation]");
                }
                user => {
                    self.reply(&mut session, family, model, user)?;
                }
            }
        }
        Ok(())
    }
}

/// Hides a reply's thinking from the stream: everything up to and through
/// `</think>`, when the reply opens with `<think>`.
struct ChatThinkingFilter {
    seen: String,
    done: bool,
    /// The thinking just ended: the newlines before the answer are dropped.
    trim: bool,
}

impl ChatThinkingFilter {
    fn new(hide: bool) -> Self {
        Self {
            seen: String::new(),
            done: !hide,
            trim: false,
        }
    }

    fn push(
        &mut self,
        delta: &str,
    ) -> String {
        if self.done {
            if !self.trim {
                return delta.to_string();
            }
            let shown = delta.trim_start_matches('\n');
            self.trim = shown.is_empty();
            return shown.to_string();
        }
        self.seen.push_str(delta);
        let opening = self.seen.trim_start();
        if !"<think>".starts_with(opening) && !opening.starts_with("<think>") {
            // Not a thinking reply after all.
            self.done = true;
            return std::mem::take(&mut self.seen);
        }
        match self.seen.split_once("</think>") {
            Some((_, answer)) => {
                self.done = true;
                let answer = answer.trim_start_matches('\n').to_string();
                self.trim = answer.is_empty();
                self.seen.clear();
                answer
            }
            None => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use bunsen::kits::language::nanochat::pretrained::DEFAULT_NANOCHAT_MODEL;
    use clap::Parser;

    use super::*;

    /// The flags alone, as `chat run` and `chat repl` flatten them.
    #[derive(clap::Parser, Debug)]
    struct Args {
        #[clap(flatten)]
        flags: ChatGenerateFlags,
    }

    #[test]
    fn test_thinking_filter() {
        let mut hide = ChatThinkingFilter::new(true);
        let shown: String = ["<thi", "nk>\nhm", "\n</think>", "\n\nBlue."]
            .iter()
            .map(|d| hide.push(d))
            .collect();
        assert_eq!(shown, "Blue.");

        let mut plain = ChatThinkingFilter::new(true);
        let shown: String = ["Bl", "ue."].iter().map(|d| plain.push(d)).collect();
        assert_eq!(shown, "Blue.");

        let mut show = ChatThinkingFilter::new(false);
        assert_eq!(show.push("<think>x"), "<think>x");
    }

    /// In `family`, an omitted `--model` is the family's default; outside
    /// it, Qwen3's.
    #[test]
    fn test_checkpoint() {
        let Args { flags } = Args::try_parse_from(["run"]).unwrap();
        assert_eq!(
            flags.checkpoint(None).unwrap(),
            (ChatModelFamily::Qwen3, DEFAULT_QWEN3_MODEL.to_string())
        );
        assert_eq!(
            flags.checkpoint(Some(ChatModelFamily::NanoChat)).unwrap(),
            (
                ChatModelFamily::NanoChat,
                DEFAULT_NANOCHAT_MODEL.to_string()
            )
        );
    }
}
