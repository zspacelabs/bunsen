//! `bunsen-chat`: chat with a Qwen3 checkpoint, on bunsen.
use std::{
    io::{
        BufRead,
        Read,
        Write,
    },
    sync::Arc,
};

use bunsen::{
    data::pretrained::PretrainedCache,
    errors::{
        BunsenResult,
        sys_at,
    },
    kits::{
        decoding::SearchConfig,
        gpts::qwen3::pretrained::{
            DEFAULT_QWEN3_MODEL,
            QWEN3_KIT,
            default_qwen3_factory,
        },
    },
    ops::transformers::attention::KVCacheStorage,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::{
        DeviceArgs,
        DevicePrefs,
        DeviceSession,
        Precision,
    },
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};
use bunsen_chat::{
    ChatMessage,
    ChatModel,
    ChatReply,
    ChatSession,
    ChatSessionConfig,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// The Qwen3 checkpoints bunsen knows by name, and what of each is
    /// cached.
    List(ListCmd),

    /// Brings every file of the named checkpoints into the weights cache.
    Fetch(FetchCmd),

    /// Answers one prompt, from the command line or stdin.
    Run(RunCmd),

    /// Holds a conversation on stdin: `/reset` starts over, `/quit` (or end
    /// of input) leaves.
    Chat(ChatCmd),

    /// The weights cache the checkpoints load through: where it is, what it
    /// holds (Qwen3's files are the `qwen3` kit's), and clearing it.
    Cache(bunsen_app::cache::CacheCommand),
}

#[derive(clap::Args, Debug)]
struct ListCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,
}

#[derive(clap::Args, Debug)]
struct FetchCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Checkpoint names, as `--model` takes them (`list` shows them).
    #[arg(required = true)]
    names: Vec<String>,
}

/// How the checkpoint is loaded and a reply generated.
#[derive(clap::Args, Debug)]
struct GenerateFlags {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device, and the precision the model loads and runs in:
    /// `--precision half` is f16 where the device runs it.
    #[clap(flatten)]
    device: DeviceArgs,

    /// The checkpoint: a name `list` shows, or `hf:org/repo`.
    #[arg(long, default_value = DEFAULT_QWEN3_MODEL)]
    model: String,

    /// A system prompt.
    #[arg(long)]
    system: Option<String>,

    /// Let the model think first (Qwen3's default).
    #[arg(long, conflicts_with = "no_think")]
    think: bool,

    /// Ask the model not to think.
    #[arg(long)]
    no_think: bool,

    /// Show the model's thinking as it streams, not just its answer.
    #[arg(long)]
    show_thinking: bool,

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
    #[arg(long, value_enum, default_value_t = Storage::Auto)]
    kv_storage: Storage,

    /// Print each reply's token counts and speeds to stderr.
    #[arg(long)]
    stats: bool,
}

/// `--kv-storage`.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum Storage {
    Auto,
    Concat,
    PerLayer,
    MultiLayer,
}

impl Storage {
    fn storage(self) -> KVCacheStorage {
        match self {
            Self::Auto => KVCacheStorage::Auto,
            Self::Concat => KVCacheStorage::Concat,
            Self::PerLayer => KVCacheStorage::PerLayer,
            Self::MultiLayer => KVCacheStorage::MultiLayer,
        }
    }
}

#[derive(clap::Args, Debug)]
struct RunCmd {
    #[clap(flatten)]
    flags: GenerateFlags,

    /// The prompt; stdin when omitted.
    prompt: Option<String>,
}

#[derive(clap::Args, Debug)]
struct ChatCmd {
    #[clap(flatten)]
    flags: GenerateFlags,
}

impl GenerateFlags {
    /// Loads the checkpoint, and opens a session on it.
    fn open(&self) -> BunsenResult<(DeviceSession, Arc<ChatModel>, ChatSession)> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let device = self
            .device
            .open(&DevicePrefs::new().with_half(Precision::AnyHalf))?;
        if let Some(seed) = self.seed {
            device.seed(seed);
        }
        let bundle = default_qwen3_factory()?.load_bundle(&self.model, &cache, &device)?;
        let chat = Arc::new(ChatModel::from_bundle(bundle)?);
        let defaults = chat.defaults().search.clone();
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
        let thinking = match (self.think, self.no_think) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        };
        let config = ChatSessionConfig::new()
            .with_max_new_tokens(self.max_new_tokens)
            .with_enable_thinking(thinking)
            .with_search(Some(search))
            .with_context(self.context)
            .with_storage(self.kv_storage.storage());
        let session = self.session(&chat, config, &device);
        Ok((device, chat, session))
    }

    /// A session over `chat`, opened with the system prompt.
    fn session(
        &self,
        chat: &Arc<ChatModel>,
        config: ChatSessionConfig,
        device: &DeviceSession,
    ) -> ChatSession {
        let mut session = chat.session(config, device);
        if let Some(system) = &self.system {
            session.push(ChatMessage::system(system.clone()));
        }
        session
    }

    /// Sends `user`, streaming the reply to stdout (its thinking only under
    /// `--show-thinking`), and reports it under `--stats`.
    fn reply(
        &self,
        session: &mut ChatSession,
        user: &str,
    ) -> BunsenResult<ChatReply> {
        let mut out = std::io::stdout().lock();
        let mut hidden = ThinkingFilter::new(!self.show_thinking);
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
            eprintln!(
                "[prompt {} tokens, {} prefilled in {:.2} s ({:.0} tok/s); reply {} tokens in {:.2} s ({:.1} tok/s); {:?}]",
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
}

/// Hides a reply's thinking from the stream: everything up to and through
/// `</think>`, when the reply opens with `<think>`.
struct ThinkingFilter {
    seen: String,
    done: bool,
    /// The thinking just ended: the newlines before the answer are dropped.
    trim: bool,
}

impl ThinkingFilter {
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

fn list(cmd: &ListCmd) -> BunsenResult<()> {
    cmd.logging
        .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
    let cache = cmd.cache.init()?;
    let factory = default_qwen3_factory()?;
    println!("qwen3 checkpoints; cache: {}", cache.cache_dir().display());
    for id in factory.ids() {
        let (_, row) = factory.lookup(&id)?;
        let default = if row.name == DEFAULT_QWEN3_MODEL {
            " (default)"
        } else {
            ""
        };
        println!();
        println!("{id}{default}");
        println!("  {}", row.description);
        for (key, r) in &row.resources.resources {
            println!("  {key}: {} [{}]", r.file, cache.status(QWEN3_KIT, r));
        }
    }
    println!();
    println!("hf:org/repo: any Hugging Face repo in transformers' Qwen3ForCausalLM layout");
    Ok(())
}

fn fetch(cmd: &FetchCmd) -> BunsenResult<()> {
    cmd.logging
        .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
    let cache = cmd.cache.init()?;
    let factory = default_qwen3_factory()?;
    for name in &cmd.names {
        let model = factory.resolve(name, &cache)?;
        let map = model.plan(&cache)?;
        let loaded = load_all(&cache, &map)?;
        println!("{}:", model.id());
        for (key, part) in loaded.iter() {
            println!("  {key}: {}", part.path.display());
        }
    }
    Ok(())
}

fn load_all(
    cache: &PretrainedCache,
    map: &bunsen::data::pretrained::ResourceMap,
) -> BunsenResult<bunsen::data::pretrained::LoadedResources> {
    cache.load(QWEN3_KIT, map)
}

fn run(cmd: &RunCmd) -> BunsenResult<()> {
    let prompt = match &cmd.prompt {
        Some(prompt) => prompt.clone(),
        None => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(sys_at("read", "stdin"))?;
            text
        }
    };
    let (device, _, mut session) = cmd.flags.open()?;
    cmd.flags.reply(&mut session, prompt.trim_end())?;
    drop(session);
    device.close()
}

fn chat(cmd: &ChatCmd) -> BunsenResult<()> {
    let (device, chat, mut session) = cmd.flags.open()?;
    let config = session.config().clone();
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
                session = cmd.flags.session(&chat, config.clone(), &device);
                eprintln!("[a new conversation]");
            }
            user => {
                cmd.flags.reply(&mut session, user)?;
            }
        }
    }
    drop(session);
    device.close()
}

fn main() -> BunsenResult<()> {
    match Args::parse().command {
        Command::List(cmd) => list(&cmd),
        Command::Fetch(cmd) => fetch(&cmd),
        Command::Run(cmd) => run(&cmd),
        Command::Chat(cmd) => chat(&cmd),
        Command::Cache(cmd) => cmd.run(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_filter() {
        let mut hide = ThinkingFilter::new(true);
        let shown: String = ["<thi", "nk>\nhm", "\n</think>", "\n\nBlue."]
            .iter()
            .map(|d| hide.push(d))
            .collect();
        assert_eq!(shown, "Blue.");

        let mut plain = ThinkingFilter::new(true);
        let shown: String = ["Bl", "ue."].iter().map(|d| plain.push(d)).collect();
        assert_eq!(shown, "Blue.");

        let mut show = ThinkingFilter::new(false);
        assert_eq!(show.push("<think>x"), "<think>x");
    }

    #[test]
    fn test_args() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }
}
