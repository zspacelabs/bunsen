//! `bunsen-tool chat`: chat with a Qwen3 or nanochat checkpoint. `chat
//! <list|fetch|run|chat>` takes a checkpoint of any family, by its name;
//! `chat family <qwen3|nanochat> <list|fetch|run|chat>` runs one family's,
//! its own default by default, with only the flags that apply to it.
use std::{
    io::{
        BufRead,
        Read,
        Write,
    },
    sync::Arc,
};

use bunsen::{
    data::pretrained::{
        Construct,
        Pretrained,
        PretrainedCache,
        PretrainedFactory,
    },
    errors::{
        BunsenResult,
        sys_at,
    },
    kits::{
        decoding::SearchConfig,
        gpts::{
            nanochat::pretrained::{
                DEFAULT_NANOCHAT_MODEL,
                NANOCHAT_KIT,
                default_nanochat_factory,
            },
            qwen3::pretrained::{
                DEFAULT_QWEN3_MODEL,
                QWEN3_KIT,
                default_qwen3_factory,
            },
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
use bunsen_tool::chat::{
    ChatBackend,
    ChatMessage,
    ChatModel,
    ChatReply,
    ChatSession,
    ChatSessionConfig,
    NanoChatChat,
    Qwen3Chat,
};

/// `bunsen-tool chat`'s commands.
#[derive(clap::Args, Debug)]
pub struct ChatCmd {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// The checkpoints bunsen knows by name, every family's: each one's
    /// origin, and each of its files' path in its repo and whether it is
    /// cached.
    List(ListCmd),

    /// Brings every file of the named checkpoints into the weights cache;
    /// each name is its own family's.
    Fetch(FetchCmd),

    /// Answers one prompt, from the command line or stdin, with the
    /// checkpoint `--model` names, of whichever family.
    Run(RunCmd<ThinkFlags>),

    /// Holds a conversation on stdin, with the checkpoint `--model` names,
    /// of whichever family: `/reset` starts over, `/quit` (or end of input)
    /// leaves.
    Chat(ConverseCmd<ThinkFlags>),

    /// The model families, each with its own `list`, `fetch`, `run` and
    /// `chat`, its own default checkpoint, and only the flags that apply to
    /// it.
    Family(FamilyCmd),
}

/// The model families' commands.
#[derive(clap::Args, Debug)]
struct FamilyCmd {
    #[command(subcommand)]
    family: FamilySub,
}

/// The model families, each with its own `list`, `fetch`, `run` and `chat`.
#[derive(clap::Subcommand, Debug)]
enum FamilySub {
    /// Qwen's Qwen3: `qwen/qwen3-0.6b` by default, or any `hf:org/repo` in
    /// its layout.
    Qwen3(KitCmd<ThinkFlags>),

    /// Andrej Karpathy's nanochat: community checkpoints of upstream's 2025
    /// and 2026 models, `nanochat-students/rl-d20` by default.
    #[command(name = "nanochat")]
    NanoChat(KitCmd<NanoChatFlags>),
}

impl FamilyCmd {
    fn run(&self) -> BunsenResult<()> {
        match &self.family {
            FamilySub::Qwen3(cmd) => cmd.action.run(Family::Qwen3),
            FamilySub::NanoChat(cmd) => cmd.action.run(Family::NanoChat),
        }
    }
}

/// One family's commands; `F` is its own generation flags.
#[derive(clap::Args, Debug)]
struct KitCmd<F: clap::Args> {
    #[command(subcommand)]
    action: Action<F>,
}

#[derive(clap::Subcommand, Debug)]
enum Action<F: clap::Args> {
    /// The family's checkpoints: each one's origin, and each of its files'
    /// path in its repo and whether it is cached.
    List(ListCmd),

    /// Brings every file of the family's named checkpoints into the weights
    /// cache.
    Fetch(FetchCmd),

    /// Answers one prompt, from the command line or stdin.
    Run(RunCmd<F>),

    /// Holds a conversation on stdin: `/reset` starts over, `/quit` (or end
    /// of input) leaves.
    Chat(ConverseCmd<F>),
}

impl<F: FamilyFlags> Action<F> {
    fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        match self {
            Action::List(cmd) => cmd.run(&[family]),
            Action::Fetch(cmd) => cmd.run(Some(family)),
            Action::Run(cmd) => cmd.run(Some(family)),
            Action::Chat(cmd) => cmd.run(Some(family)),
        }
    }
}

/// A model family: one kit, its factory, and its default checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// Qwen's Qwen3.
    Qwen3,

    /// Andrej Karpathy's nanochat.
    NanoChat,
}

impl Family {
    /// Every family.
    const ALL: [Family; 2] = [Family::Qwen3, Family::NanoChat];

    /// The family whose checkpoint `name` is: nanochat's when the nanochat
    /// factory knows it, Qwen3's otherwise.
    fn of(name: &str) -> BunsenResult<Family> {
        Ok(if default_nanochat_factory()?.lookup(name).is_ok() {
            Family::NanoChat
        } else {
            Family::Qwen3
        })
    }

    /// The family's command name.
    fn name(self) -> &'static str {
        match self {
            Family::Qwen3 => "qwen3",
            Family::NanoChat => "nanochat",
        }
    }

    /// The checkpoint `--model` names when it is not given.
    fn default_model(self) -> &'static str {
        match self {
            Family::Qwen3 => DEFAULT_QWEN3_MODEL,
            Family::NanoChat => DEFAULT_NANOCHAT_MODEL,
        }
    }

    /// The kit's segment of the pretrained cache.
    fn cache_kit(self) -> &'static str {
        match self {
            Family::Qwen3 => QWEN3_KIT,
            Family::NanoChat => NANOCHAT_KIT,
        }
    }

    /// The family's checkpoints by name: each row, by its id.
    fn rows(self) -> BunsenResult<Vec<(String, Pretrained)>> {
        fn rows_of<H: Construct>(
            factory: &PretrainedFactory<H>
        ) -> BunsenResult<Vec<(String, Pretrained)>> {
            factory
                .ids()
                .into_iter()
                .map(|id| Ok((id.clone(), factory.lookup(&id)?.1)))
                .collect()
        }
        match self {
            Family::Qwen3 => rows_of(&default_qwen3_factory()?),
            Family::NanoChat => rows_of(&default_nanochat_factory()?),
        }
    }

    /// Loads `name` on `device`, through `cache`.
    fn load(
        self,
        name: &str,
        cache: &PretrainedCache,
        device: &DeviceSession,
    ) -> BunsenResult<Chat> {
        Ok(match self {
            Family::Qwen3 => {
                let bundle = default_qwen3_factory()?.load_bundle(name, cache, device)?;
                Chat::Qwen3(Arc::new(ChatModel::from_bundle(bundle)?))
            }
            Family::NanoChat => {
                let bundle = default_nanochat_factory()?.load_bundle(name, cache, device)?;
                Chat::NanoChat(Arc::new(ChatModel::new(NanoChatChat::from_bundle(bundle)?)))
            }
        })
    }

    /// Brings every file of `name` into `cache`, and prints where each one
    /// is.
    fn fetch(
        self,
        name: &str,
        cache: &PretrainedCache,
    ) -> BunsenResult<()> {
        let (id, loaded) = match self {
            Family::Qwen3 => {
                let model = default_qwen3_factory()?.resolve(name, cache)?;
                let map = model.plan(cache)?;
                (model.id(), cache.load(QWEN3_KIT, &map)?)
            }
            Family::NanoChat => {
                let model = default_nanochat_factory()?.resolve(name, cache)?;
                let map = model.plan(cache)?;
                (model.id(), cache.load(NANOCHAT_KIT, &map)?)
            }
        };
        println!("{id}:");
        for (key, part) in loaded.iter() {
            println!("  {key}: {}", part.path.display());
        }
        Ok(())
    }
}

#[derive(clap::Args, Debug)]
struct ListCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Also print each file's first source, where it is fetched from (as
    /// `url: ...`).
    #[arg(long)]
    urls: bool,
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
    /// `--precision half` is bf16 where the device runs it (cuda, metal),
    /// the dtype Qwen3 ships and nanochat trains in; f16 overflows Qwen3.
    #[clap(flatten)]
    device: DeviceArgs,

    /// The checkpoint: a name `list` shows, or `hf:org/repo`, a Qwen3. The
    /// family's default when omitted; outside `family`, the name picks the
    /// family, and Qwen3's default is the default.
    #[arg(long)]
    model: Option<String>,

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
    #[arg(long, value_enum, default_value_t = Storage::Auto)]
    kv_storage: Storage,

    /// Print each reply's token counts and speeds to stderr.
    #[arg(long)]
    stats: bool,
}

/// A family's own generation flags.
trait FamilyFlags: clap::Args + std::fmt::Debug {
    /// Whether the model thinks first; `None` for the checkpoint's default.
    fn thinking(&self) -> Option<bool> {
        None
    }

    /// Whether a reply's thinking is hidden from the stream.
    fn hide_thinking(&self) -> bool {
        false
    }
}

/// Thinking: Qwen3's own flags.
#[derive(clap::Args, Debug)]
struct ThinkFlags {
    /// Let the model think first (Qwen3's default; nanochat does not
    /// think).
    #[arg(long, conflicts_with = "no_think")]
    think: bool,

    /// Ask the model not to think.
    #[arg(long)]
    no_think: bool,

    /// Show the model's thinking as it streams, not just its answer.
    #[arg(long)]
    show_thinking: bool,
}

impl FamilyFlags for ThinkFlags {
    fn thinking(&self) -> Option<bool> {
        match (self.think, self.no_think) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        }
    }

    fn hide_thinking(&self) -> bool {
        !self.show_thinking
    }
}

/// nanochat's own flags: none; it does not think.
#[derive(clap::Args, Debug)]
struct NanoChatFlags {}

impl FamilyFlags for NanoChatFlags {}

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
struct RunCmd<F: clap::Args> {
    #[clap(flatten)]
    flags: GenerateFlags,

    #[clap(flatten)]
    family: F,

    /// The prompt; stdin when omitted.
    prompt: Option<String>,
}

#[derive(clap::Args, Debug)]
struct ConverseCmd<F: clap::Args> {
    #[clap(flatten)]
    flags: GenerateFlags,

    #[clap(flatten)]
    family: F,
}

/// A loaded chat, of whichever kit its checkpoint is.
enum Chat {
    Qwen3(Arc<ChatModel<Qwen3Chat>>),
    NanoChat(Arc<ChatModel<NanoChatChat>>),
}

impl GenerateFlags {
    /// The checkpoint these flags name, and its family: `family`'s (its
    /// default when `--model` is not given), or outside `family` the one
    /// `--model` names.
    fn checkpoint(
        &self,
        family: Option<Family>,
    ) -> BunsenResult<(Family, String)> {
        Ok(match family {
            Some(family) => (
                family,
                self.model
                    .clone()
                    .unwrap_or_else(|| family.default_model().to_string()),
            ),
            None => {
                let name = self.model.as_deref().unwrap_or(DEFAULT_QWEN3_MODEL);
                (Family::of(name)?, name.to_string())
            }
        })
    }

    /// Opens the device, and loads the checkpoint on it: `family`'s, or
    /// outside `family` whichever `--model` names.
    fn load(
        &self,
        family: Option<Family>,
    ) -> BunsenResult<(DeviceSession, Chat)> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let device = self
            .device
            .open(&DevicePrefs::new().with_half(Precision::BF16Half))?;
        if let Some(seed) = self.seed {
            device.seed(seed);
        }
        let (family, name) = self.checkpoint(family)?;
        let chat = family.load(&name, &cache, &device)?;
        Ok((device, chat))
    }

    /// The session config these flags and `family`'s give, over `chat`'s
    /// defaults.
    fn config<B: ChatBackend>(
        &self,
        chat: &ChatModel<B>,
        family: &impl FamilyFlags,
    ) -> ChatSessionConfig {
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
        ChatSessionConfig::new()
            .with_max_new_tokens(self.max_new_tokens)
            .with_enable_thinking(family.thinking())
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
    /// `family`'s flags show it), and reports it under `--stats`.
    fn reply<B: ChatBackend>(
        &self,
        session: &mut ChatSession<B>,
        family: &impl FamilyFlags,
        user: &str,
    ) -> BunsenResult<ChatReply> {
        let mut out = std::io::stdout().lock();
        let mut hidden = ThinkingFilter::new(family.hide_thinking());
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

    /// Answers `prompt`, once.
    fn run_once<B: ChatBackend>(
        &self,
        chat: &Arc<ChatModel<B>>,
        family: &impl FamilyFlags,
        device: &DeviceSession,
        prompt: &str,
    ) -> BunsenResult<()> {
        let mut session = self.session(chat, self.config(chat, family), device);
        self.reply(&mut session, family, prompt)?;
        Ok(())
    }

    /// Holds a conversation on stdin.
    fn converse<B: ChatBackend>(
        &self,
        chat: &Arc<ChatModel<B>>,
        family: &impl FamilyFlags,
        device: &DeviceSession,
    ) -> BunsenResult<()> {
        let config = self.config(chat, family);
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
                    self.reply(&mut session, family, user)?;
                }
            }
        }
        Ok(())
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

impl ListCmd {
    /// Prints each of `families`' checkpoints, as [`print_row`] does.
    fn run(
        &self,
        families: &[Family],
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        println!("cache: {}", cache.cache_dir().display());
        for &family in families {
            let default = family.default_model();
            println!();
            println!("# {} checkpoints (the default: {default})", family.name());
            for (id, row) in family.rows()? {
                print_row(&cache, family.cache_kit(), &id, &row, default, self.urls);
            }
            if family == Family::Qwen3 {
                println!();
                println!(
                    "hf:org/repo: any Hugging Face repo in transformers' Qwen3ForCausalLM layout"
                );
            }
        }
        Ok(())
    }
}

/// One row of `list`: its id, aliases, description and origin, and each
/// file's state in `cache` and path in its repo; under `urls`, its first
/// source too.
///
/// ```text
/// well-known:nanochat-students/rl-d20 (default)
///   aliases: nanochat-d20-rl
///   nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat
///   origin: https://huggingface.co/nanochat-students/rl-d20
///   checkpoint: [cached]
///     path: model_000650.pt
///     url: https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a8.../model_000650.pt
/// ```
fn print_row(
    cache: &PretrainedCache,
    kit: &str,
    id: &str,
    row: &Pretrained,
    default: &str,
    urls: bool,
) {
    let default = if row.name == default {
        " (default)"
    } else {
        ""
    };
    println!();
    println!("{id}{default}");
    if !row.aliases.is_empty() {
        println!("  aliases: {}", row.aliases.join(", "));
    }
    println!("  {}", row.description);
    if let Some(origin) = &row.origin {
        println!("  origin: {origin}");
    }
    for (key, r) in &row.resources.resources {
        println!("  {key}: [{}]", cache.status(kit, r));
        println!("    path: {}", r.file);
        if urls && !r.sources.is_empty() {
            println!("    {}", r.sources[0]);
        }
    }
}

impl FetchCmd {
    /// Fetches each name: `family`'s, or outside `family` its own family's.
    fn run(
        &self,
        family: Option<Family>,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        for name in &self.names {
            let family = match family {
                Some(family) => family,
                None => Family::of(name)?,
            };
            family.fetch(name, &cache)?;
        }
        Ok(())
    }
}

impl<F: FamilyFlags> RunCmd<F> {
    /// Answers the prompt with `family`'s checkpoint, or outside `family`
    /// whichever `--model` names.
    fn run(
        &self,
        family: Option<Family>,
    ) -> BunsenResult<()> {
        let prompt = match &self.prompt {
            Some(prompt) => prompt.clone(),
            None => {
                let mut text = String::new();
                std::io::stdin()
                    .read_to_string(&mut text)
                    .map_err(sys_at("read", "stdin"))?;
                text
            }
        };
        let prompt = prompt.trim_end();
        let (device, chat) = self.flags.load(family)?;
        match &chat {
            Chat::Qwen3(chat) => self.flags.run_once(chat, &self.family, &device, prompt)?,
            Chat::NanoChat(chat) => self.flags.run_once(chat, &self.family, &device, prompt)?,
        }
        drop(chat);
        device.close()
    }
}

impl<F: FamilyFlags> ConverseCmd<F> {
    /// Holds a conversation with `family`'s checkpoint, or outside `family`
    /// whichever `--model` names.
    fn run(
        &self,
        family: Option<Family>,
    ) -> BunsenResult<()> {
        let (device, chat) = self.flags.load(family)?;
        match &chat {
            Chat::Qwen3(chat) => self.flags.converse(chat, &self.family, &device)?,
            Chat::NanoChat(chat) => self.flags.converse(chat, &self.family, &device)?,
        }
        drop(chat);
        device.close()
    }
}

impl ChatCmd {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.command {
            Command::List(cmd) => cmd.run(&Family::ALL),
            Command::Fetch(cmd) => cmd.run(None),
            Command::Run(cmd) => cmd.run(None),
            Command::Chat(cmd) => cmd.run(None),
            Command::Family(cmd) => cmd.run(),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    /// `chat`'s commands alone, as `bunsen-tool chat` parses them.
    #[derive(clap::Parser, Debug)]
    struct Args {
        #[clap(subcommand)]
        command: Command,
    }

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

    /// Thinking is Qwen3's: its family and the family-agnostic commands
    /// take the flags, nanochat's does not.
    #[test]
    fn test_family_flags() {
        let parse = |args: &[&str]| Args::try_parse_from([&["chat"], args].concat());
        assert!(parse(&["run", "--think", "hi"]).is_ok());
        assert!(parse(&["family", "qwen3", "run", "--no-think", "hi"]).is_ok());
        assert!(parse(&["family", "nanochat", "run", "hi"]).is_ok());
        assert!(parse(&["family", "nanochat", "run", "--think", "hi"]).is_err());
        assert!(parse(&["family", "nanochat", "chat", "--show-thinking"]).is_err());
    }

    /// Each family's default is its own; a nanochat name or alias is
    /// nanochat's, and any other name Qwen3's.
    #[test]
    fn test_family_of() {
        for family in Family::ALL {
            assert_eq!(Family::of(family.default_model()).unwrap(), family);
        }
        assert_eq!(Family::of("nanochat-d20-rl").unwrap(), Family::NanoChat);
        assert_eq!(Family::of("hf:Qwen/Qwen3-0.6B").unwrap(), Family::Qwen3);
    }

    /// In `family`, an omitted `--model` is the family's default; outside
    /// it, Qwen3's.
    #[test]
    fn test_checkpoint() {
        let Args {
            command: Command::Run(cmd),
        } = Args::try_parse_from(["chat", "run", "hi"]).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            cmd.flags.checkpoint(None).unwrap(),
            (Family::Qwen3, DEFAULT_QWEN3_MODEL.to_string())
        );
        assert_eq!(
            cmd.flags.checkpoint(Some(Family::NanoChat)).unwrap(),
            (Family::NanoChat, DEFAULT_NANOCHAT_MODEL.to_string())
        );
    }
}
