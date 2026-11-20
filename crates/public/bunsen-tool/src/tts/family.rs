//! The TTS families `bunsen-tool tts` serves.

use std::sync::Arc;

use bunsen::{
    data::{
        builds::BuildProvider,
        catalog::{
            CatalogName,
            resolve::ResolvedBuild,
        },
        formats::LoadPolicy,
        resources::ResourceCache,
    },
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsDriver,
                TtsDriverConfig,
                TtsSynthesizer,
            },
            qwen3_tts::{
                driver::{
                    Qwen3TtsEngine,
                    Qwen3TtsEngineConfig,
                },
                pretrained::{
                    DEFAULT_QWEN3_TTS_MODEL,
                    default_qwen3_tts_factory,
                },
            },
            zonos3::{
                driver::{
                    Zonos3Engine,
                    Zonos3EngineConfig,
                    zonos3_release_normalization,
                },
                pretrained::{
                    DEFAULT_ZONOS3_MODEL,
                    default_zonos3_factory,
                },
            },
        },
        support::{
            text::TextNormalizationConfig,
            tokens::BunsenTokenizer,
        },
    },
};
use burn::tensor::Device;

/// A TTS model family: one kit, its factory, its default model, and its
/// release's text normalization.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TtsFamily {
    /// Zyphra's ZONOS3-Mini.
    Zonos3,

    /// Alibaba's Qwen3-TTS.
    #[value(name = "qwen3-tts")]
    Qwen3Tts,
}

impl TtsFamily {
    /// Every family, in listing order.
    pub const ALL: [Self; 2] = [Self::Zonos3, Self::Qwen3Tts];

    /// Its name, as `--family` takes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Zonos3 => "zonos3",
            Self::Qwen3Tts => "qwen3-tts",
        }
    }

    /// Its default checkpoint.
    pub fn default_model(self) -> &'static str {
        match self {
            Self::Zonos3 => DEFAULT_ZONOS3_MODEL,
            Self::Qwen3Tts => DEFAULT_QWEN3_TTS_MODEL,
        }
    }

    /// How its release normalizes a text: ZONOS3's, Zyphra's normalizer
    /// after line breaks are collapsed; Qwen3-TTS's, none (the text as
    /// typed).
    pub fn release_normalization(self) -> TextNormalizationConfig {
        match self {
            Self::Zonos3 => zonos3_release_normalization(),
            Self::Qwen3Tts => TextNormalizationConfig::None,
        }
    }

    /// Its factory's listed rows, by provider.
    ///
    /// # Errors
    /// As the factory's.
    pub fn rows(self) -> BunsenResult<Vec<(String, ResolvedBuild)>> {
        Ok(match self {
            Self::Zonos3 => default_zonos3_factory()?.list(),
            Self::Qwen3Tts => default_qwen3_tts_factory()?.list(),
        })
    }

    /// Its factory's providers, in search order.
    ///
    /// # Errors
    /// As the factory's.
    pub fn providers(self) -> BunsenResult<Vec<Arc<dyn BuildProvider>>> {
        Ok(match self {
            Self::Zonos3 => default_zonos3_factory()?.providers().to_vec(),
            Self::Qwen3Tts => default_qwen3_tts_factory()?.providers().to_vec(),
        })
    }

    /// Whether its factory knows `name`.
    ///
    /// # Errors
    /// As the factory's.
    pub fn knows(
        self,
        name: &str,
    ) -> BunsenResult<bool> {
        Ok(match self {
            Self::Zonos3 => default_zonos3_factory()?.lookup(name).is_ok(),
            Self::Qwen3Tts => default_qwen3_tts_factory()?.lookup(name).is_ok(),
        })
    }

    /// Every file of `name` brought into `cache`: each key and its path.
    ///
    /// # Errors
    /// As the factory's and the cache's.
    pub fn fetch(
        self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<(String, Vec<(String, std::path::PathBuf)>)> {
        let fetched = |id: String, map| -> BunsenResult<_> {
            let loaded = cache.load(&map)?;
            let parts = loaded
                .iter()
                .map(|(key, part)| (key.to_string(), part.path.clone()))
                .collect();
            Ok((id, parts))
        };
        match self {
            Self::Zonos3 => {
                let model = default_zonos3_factory()?.resolve(name, cache)?;
                fetched(model.id().to_string(), model.plan(cache)?)
            }
            Self::Qwen3Tts => {
                let model = default_qwen3_tts_factory()?.resolve(name, cache)?;
                fetched(model.id().to_string(), model.plan(cache)?)
            }
        }
    }

    /// `model` loaded on `device` under `policy`, and its driver over it,
    /// configured by `driver`: the family's engine at the release's
    /// settings.
    ///
    /// # Errors
    /// As the factory's, the tokenizer's, the engine's and the driver's.
    pub fn load(
        self,
        model: &str,
        cache: &ResourceCache,
        device: &Device,
        policy: LoadPolicy,
        driver: TtsDriverConfig,
    ) -> BunsenResult<Arc<dyn TtsSynthesizer>> {
        match self {
            Self::Zonos3 => {
                let construct =
                    default_zonos3_factory()?.load_construct(model, cache, device, policy)?;
                let tokenizer = BunsenTokenizer::from_file(&construct.tokenizer)?;
                let config = Zonos3EngineConfig::from_generation(&construct.generation);
                let engine = Zonos3Engine::new(construct, tokenizer, config)?;
                Ok(Arc::new(TtsDriver::new(engine, driver)?))
            }
            Self::Qwen3Tts => {
                let construct =
                    default_qwen3_tts_factory()?.load_construct(model, cache, device, policy)?;
                let tokenizer = construct.tokenizer.load()?;
                let engine =
                    Qwen3TtsEngine::new(construct, tokenizer, Qwen3TtsEngineConfig::new())?;
                Ok(Arc::new(TtsDriver::new(engine, driver)?))
            }
        }
    }
}

/// Every TTS family's listed builds, each with its family.
///
/// # Errors
/// As the factories'.
pub fn tts_builds() -> BunsenResult<Vec<(CatalogName, TtsFamily)>> {
    let mut builds = Vec::new();
    for family in TtsFamily::ALL {
        for (_, row) in family.rows()? {
            builds.push((row.name, family));
        }
    }
    Ok(builds)
}

/// The family whose factory knows `model`: a listed build's, by name.
///
/// # Errors
/// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup), with the families,
/// for a name no family knows (a provider ref such as `hf:org/repo` names
/// no family: name it with `--family`).
pub fn tts_family_of(model: &str) -> BunsenResult<TtsFamily> {
    for family in TtsFamily::ALL {
        if family.knows(model)? {
            return Ok(family);
        }
    }
    Err(BunsenError::lookup(
        LookupError::missing("TTS checkpoint", model)
            .with_candidates(TtsFamily::ALL.iter().map(|f| f.name())),
    )
    .with_details("a provider ref names no family: pass --family"))
}
