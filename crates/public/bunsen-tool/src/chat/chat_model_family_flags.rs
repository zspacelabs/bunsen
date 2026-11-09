//! The chat model families, and each one's own generation flags.

use std::sync::Arc;

use bunsen::{
    burner::device::DeviceSession,
    data::pretrained::{
        Pretrained,
        PretrainedCache,
        WELL_KNOWN,
    },
    errors::BunsenResult,
    kits::{
        language::{
            gemma::pretrained::{
                DEFAULT_GEMMA_MODEL,
                GEMMA_KIT,
                default_gemma_factory,
            },
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
            qwen3_5::pretrained::{
                DEFAULT_QWEN3_5_MODEL,
                QWEN3_5_KIT,
                default_qwen3_5_factory,
            },
        },
        support::construct::{
            ConstructInit,
            LmPolicy,
            LoadPolicy,
            PretrainedFactory,
        },
    },
};

use crate::chat::{
    ChatModel,
    GemmaChat,
    NanoChatChat,
    Qwen3_5Chat,
    Qwen3Chat,
    no_chat_template,
};

/// Refuses a well-known checkpoint none of whose presets names a chat
/// template, before anything of it is fetched. An `hf:` repo's template is
/// found in its files, so it is checked when it loads.
pub fn require_template<H: ConstructInit>(
    factory: &PretrainedFactory<H>,
    name: &str,
) -> BunsenResult<()> {
    let (provider, _) = factory.lookup(name)?;
    if provider != WELL_KNOWN {
        return Ok(());
    }
    let resolved = factory.resolve_presets(name, &LmPolicy::default())?;
    match resolved.config.generation.chat_template {
        Some(_) => Ok(()),
        None => Err(no_chat_template(name)),
    }
}

/// A loaded chat, of whichever kit its checkpoint is.
pub enum LoadedChat {
    Qwen3(Arc<ChatModel<Qwen3Chat>>),
    Qwen3_5(Arc<ChatModel<Qwen3_5Chat>>),
    Gemma(Arc<ChatModel<GemmaChat>>),
    NanoChat(Arc<ChatModel<NanoChatChat>>),
}

/// A model family: one kit, its factory, and its default checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatModelFamily {
    /// Qwen's Qwen3.
    Qwen3,

    /// Qwen's dense Qwen3.5.
    Qwen3_5,

    /// Google's Gemma.
    Gemma,

    /// Andrej Karpathy's nanochat.
    NanoChat,
}

impl ChatModelFamily {
    /// Every family.
    pub const ALL: [ChatModelFamily; 4] = [
        ChatModelFamily::Qwen3,
        ChatModelFamily::Qwen3_5,
        ChatModelFamily::Gemma,
        ChatModelFamily::NanoChat,
    ];

    /// The family's command name.
    pub fn name(self) -> &'static str {
        match self {
            ChatModelFamily::Qwen3 => "qwen3",
            ChatModelFamily::Qwen3_5 => "qwen3.5",
            ChatModelFamily::Gemma => "gemma",
            ChatModelFamily::NanoChat => "nanochat",
        }
    }

    /// The checkpoint `--model` names when it is not given.
    pub fn default_model(self) -> &'static str {
        match self {
            ChatModelFamily::Qwen3 => DEFAULT_QWEN3_MODEL,
            ChatModelFamily::Qwen3_5 => DEFAULT_QWEN3_5_MODEL,
            ChatModelFamily::Gemma => DEFAULT_GEMMA_MODEL,
            ChatModelFamily::NanoChat => DEFAULT_NANOCHAT_MODEL,
        }
    }

    /// The kit's segment of the pretrained cache.
    pub fn cache_kit(self) -> &'static str {
        match self {
            ChatModelFamily::Qwen3 => QWEN3_KIT,
            ChatModelFamily::Qwen3_5 => QWEN3_5_KIT,
            ChatModelFamily::Gemma => GEMMA_KIT,
            ChatModelFamily::NanoChat => NANOCHAT_KIT,
        }
    }

    /// The family's checkpoints by name: each row, by its id.
    pub fn rows(self) -> BunsenResult<Vec<(String, Pretrained)>> {
        fn rows_of<H: ConstructInit>(
            factory: &PretrainedFactory<H>
        ) -> BunsenResult<Vec<(String, Pretrained)>> {
            factory
                .ids()
                .into_iter()
                .map(|id| Ok((id.clone(), factory.lookup(&id)?.1)))
                .collect()
        }
        match self {
            ChatModelFamily::Qwen3 => rows_of(&default_qwen3_factory()?),
            ChatModelFamily::Qwen3_5 => rows_of(&default_qwen3_5_factory()?),
            ChatModelFamily::Gemma => rows_of(&default_gemma_factory()?),
            ChatModelFamily::NanoChat => rows_of(&default_nanochat_factory()?),
        }
    }

    /// Loads `name` on `device`, through `cache`.
    pub fn load(
        self,
        name: &str,
        cache: &PretrainedCache,
        device: &DeviceSession,
    ) -> BunsenResult<LoadedChat> {
        Ok(match self {
            ChatModelFamily::Qwen3 => {
                require_template(&default_qwen3_factory()?, name)?;
                let bundle = default_qwen3_factory()?.load_construct(
                    name,
                    cache,
                    device,
                    LoadPolicy::performance(),
                )?;
                LoadedChat::Qwen3(Arc::new(ChatModel::from_construct(bundle)?))
            }
            ChatModelFamily::Qwen3_5 => {
                require_template(&default_qwen3_5_factory()?, name)?;
                let bundle = default_qwen3_5_factory()?.load_construct(
                    name,
                    cache,
                    device,
                    LoadPolicy::performance(),
                )?;
                LoadedChat::Qwen3_5(Arc::new(ChatModel::new(Qwen3_5Chat::from_construct(
                    bundle,
                )?)))
            }
            ChatModelFamily::Gemma => {
                require_template(&default_gemma_factory()?, name)?;
                let bundle = default_gemma_factory()?.load_construct(
                    name,
                    cache,
                    device,
                    LoadPolicy::performance(),
                )?;
                LoadedChat::Gemma(Arc::new(ChatModel::new(GemmaChat::from_construct(bundle)?)))
            }
            ChatModelFamily::NanoChat => {
                let bundle = default_nanochat_factory()?.load_construct(
                    name,
                    cache,
                    device,
                    LoadPolicy::performance(),
                )?;
                LoadedChat::NanoChat(Arc::new(ChatModel::new(NanoChatChat::from_construct(
                    bundle,
                )?)))
            }
        })
    }

    /// Brings every file of `name` into `cache`, and prints where each one
    /// is.
    pub fn fetch(
        self,
        name: &str,
        cache: &PretrainedCache,
    ) -> BunsenResult<()> {
        let (id, loaded) = match self {
            ChatModelFamily::Qwen3 => {
                let model = default_qwen3_factory()?.resolve(name, cache)?;
                let map = model.plan(cache)?;
                (model.id(), cache.load(QWEN3_KIT, &map)?)
            }
            ChatModelFamily::Qwen3_5 => {
                let model = default_qwen3_5_factory()?.resolve(name, cache)?;
                let map = model.plan(cache)?;
                (model.id(), cache.load(QWEN3_5_KIT, &map)?)
            }
            ChatModelFamily::Gemma => {
                let model = default_gemma_factory()?.resolve(name, cache)?;
                let map = model.plan(cache)?;
                (model.id(), cache.load(GEMMA_KIT, &map)?)
            }
            ChatModelFamily::NanoChat => {
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

/// A family's own generation flags.
pub trait ChatModelFamilyFlags: clap::Args + std::fmt::Debug {
    /// The presets the flags select, added to a Qwen3 checkpoint's lookup
    /// expression: none for the checkpoint's default.
    fn presets(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Whether a reply's thinking is hidden from the stream.
    fn hide_thinking(&self) -> bool {
        false
    }
}

/// Thinking: Qwen3's own flags.
#[derive(clap::Args, Debug)]
pub struct ChatThinkFlags {
    /// Let the model think first: a Qwen3 or Qwen3.5 checkpoint's
    /// `[thinking]` preset (Qwen3's hybrid models' default; Qwen3.5 answers at
    /// once by default), under the card's thinking sampling. nanochat does
    /// not think.
    #[arg(long, conflicts_with = "no_think")]
    think: bool,

    /// Ask the model not to think: its `[no-thinking]` preset, under the
    /// card's non-thinking sampling.
    #[arg(long)]
    no_think: bool,

    /// Show the model's thinking as it streams, not just its answer.
    #[arg(long)]
    show_thinking: bool,
}

impl ChatModelFamilyFlags for ChatThinkFlags {
    /// `[thinking]` or `[no-thinking]`: the `thinking-mode` axis, which also
    /// sets the model card's sampling for the mode.
    fn presets(&self) -> Vec<&'static str> {
        match (self.think, self.no_think) {
            (true, _) => vec!["thinking"],
            (_, true) => vec!["no-thinking"],
            _ => Vec::new(),
        }
    }

    fn hide_thinking(&self) -> bool {
        !self.show_thinking
    }
}

/// The flags of a family that does not think (Gemma, nanochat): none.
#[derive(clap::Args, Debug)]
pub struct ChatPlainFlags {}

impl ChatModelFamilyFlags for ChatPlainFlags {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every well-known chat checkpoint has a chat template, a pretrained
    /// Gemma its family's; a lookup's patch that takes it away is refused
    /// before anything is fetched.
    #[test]
    fn test_require_template() {
        let gemma = default_gemma_factory().unwrap();
        for id in gemma.ids() {
            require_template(&gemma, &id).unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let qwen3 = default_qwen3_factory().unwrap();
        for id in qwen3.ids() {
            require_template(&qwen3, &id).unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let qwen3_5 = default_qwen3_5_factory().unwrap();
        for id in qwen3_5.ids() {
            require_template(&qwen3_5, &id).unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let err =
            require_template(&gemma, "gemma-2b{generation: {chat_template: null}}").unwrap_err();
        assert!(err.to_string().contains("no chat template"), "{err}");
    }
}
