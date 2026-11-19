//! The TTS families `bunsen-tool tts` serves.

use std::{
    collections::BTreeMap,
    sync::Arc,
};

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
                prompt::Qwen3TtsTextPlacement,
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
    /// settings, with `flags` over them.
    ///
    /// # Errors
    /// As [`TtsFamilyArgs::check`]; as the factory's, the tokenizer's, the
    /// engine's and the driver's.
    pub fn load(
        self,
        model: &str,
        cache: &ResourceCache,
        device: &Device,
        policy: LoadPolicy,
        driver: TtsDriverConfig,
        flags: &TtsFamilyArgs,
    ) -> BunsenResult<Arc<dyn TtsSynthesizer>> {
        flags.check(self)?;
        match self {
            Self::Zonos3 => {
                let construct =
                    default_zonos3_factory()?.load_construct(model, cache, device, policy)?;
                let tokenizer = BunsenTokenizer::from_file(&construct.tokenizer)?;
                let mut config = Zonos3EngineConfig::from_generation(&construct.generation)
                    .with_conditions(flags.condition_values()?);
                if let Some(frames) = flags.silence_prefill_frames {
                    config.silence_prefill_frames = frames;
                }
                let engine = Zonos3Engine::new(construct, tokenizer, config)?;
                Ok(Arc::new(TtsDriver::new(engine, driver)?))
            }
            Self::Qwen3Tts => {
                let construct =
                    default_qwen3_tts_factory()?.load_construct(model, cache, device, policy)?;
                let tokenizer = construct.tokenizer.load()?;
                let mut config = Qwen3TtsEngineConfig::new();
                if let Some(placement) = flags.placement {
                    config.placement = match placement {
                        TtsPlacement::Upfront => Qwen3TtsTextPlacement::Upfront,
                        TtsPlacement::Interleaved => Qwen3TtsTextPlacement::Interleaved,
                    };
                }
                let mut take = construct.take.clone();
                let predictor = &mut take.code_predictor;
                if let Some(t) = flags.predictor_temperature {
                    predictor.temperature = t;
                    predictor.sample = t > 0.0;
                }
                if let Some(k) = flags.predictor_top_k {
                    predictor.top_k = k;
                }
                if let Some(p) = flags.predictor_top_p {
                    predictor.top_p = p;
                }
                config.take = Some(take);
                let engine = Qwen3TtsEngine::new(construct, tokenizer, config)?;
                Ok(Arc::new(TtsDriver::new(engine, driver)?))
            }
        }
    }
}

/// Where Qwen3-TTS puts the text in its prompt, as `--placement` takes it.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsPlacement {
    /// The whole text before the first frame (upstream's default).
    Upfront,

    /// A text token joins each generated frame (upstream's "streaming").
    Interleaved,
}

/// The flags one family reads: paths into its engine's config, not policy.
/// A flag of a family other than the chosen checkpoint's is refused.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct TtsFamilyArgs {
    /// ZONOS3: a condition's value, as `NAME=VALUE`; repeated, one per
    /// condition.
    ///
    /// `loudness`, `fmax` or `speaking_rate`, bucketed by the release's
    /// edges. A condition not given is the unknown bucket.
    #[arg(long = "condition", value_name = "NAME=VALUE")]
    pub conditions: Vec<String>,

    /// ZONOS3: the silence prefill's frames (the release's 3).
    #[arg(long)]
    pub silence_prefill_frames: Option<usize>,

    /// Qwen3-TTS: where the text goes in the prompt.
    #[arg(long, value_enum)]
    pub placement: Option<TtsPlacement>,

    /// Qwen3-TTS: the code predictor's temperature (the release's 0.9).
    #[arg(long)]
    pub predictor_temperature: Option<f64>,

    /// Qwen3-TTS: the code predictor's top-k (the release's 50).
    ///
    /// The codes kept per choice, the most probable.
    #[arg(long)]
    pub predictor_top_k: Option<usize>,

    /// Qwen3-TTS: the code predictor's top-p (the release's 1.0).
    ///
    /// The probability mass kept per choice.
    #[arg(long)]
    pub predictor_top_p: Option<f64>,
}

impl TtsFamilyArgs {
    /// The flags given, each with its family.
    fn given(&self) -> Vec<(&'static str, TtsFamily)> {
        let z = TtsFamily::Zonos3;
        let q = TtsFamily::Qwen3Tts;
        [
            ("--condition", !self.conditions.is_empty(), z),
            (
                "--silence-prefill-frames",
                self.silence_prefill_frames.is_some(),
                z,
            ),
            ("--placement", self.placement.is_some(), q),
            (
                "--predictor-temperature",
                self.predictor_temperature.is_some(),
                q,
            ),
            ("--predictor-top-k", self.predictor_top_k.is_some(), q),
            ("--predictor-top-p", self.predictor_top_p.is_some(), q),
        ]
        .into_iter()
        .filter(|(_, given, _)| *given)
        .map(|(flag, _, family)| (flag, family))
        .collect()
    }

    /// Checks that every flag given is `family`'s.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) for a flag of
    /// another family, naming both.
    pub fn check(
        &self,
        family: TtsFamily,
    ) -> BunsenResult<()> {
        match self.given().into_iter().find(|(_, of)| *of != family) {
            Some((flag, of)) => Err(BunsenError::illegal(format!(
                "{flag} is a {} flag; the checkpoint is a {}",
                of.name(),
                family.name()
            ))),
            None => Ok(()),
        }
    }

    /// The `--condition` values, by name.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) for one not
    /// `NAME=VALUE`, a value not a number, or a name given twice.
    pub fn condition_values(&self) -> BunsenResult<BTreeMap<String, f64>> {
        let mut values = BTreeMap::new();
        for spec in &self.conditions {
            let (name, value) = spec.split_once('=').ok_or_else(|| {
                BunsenError::illegal(format!("--condition {spec:?}: not NAME=VALUE"))
            })?;
            let value: f64 = value.trim().parse().map_err(|_| {
                BunsenError::illegal(format!("--condition {spec:?}: {value:?} is not a number"))
            })?;
            if values.insert(name.trim().to_string(), value).is_some() {
                return Err(BunsenError::illegal(format!(
                    "--condition {name} given twice"
                )));
            }
        }
        Ok(values)
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

#[cfg(test)]
mod tests {
    use bunsen::errors::BunsenErrorKind;
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        flags: TtsFamilyArgs,
    }

    fn flags(argv: &[&str]) -> TtsFamilyArgs {
        Cli::try_parse_from([&["tts"], argv].concat())
            .unwrap()
            .flags
    }

    /// A family's own flags pass; another's are refused, naming both.
    #[test]
    fn test_check() {
        let zonos3 = flags(&[
            "--condition",
            "speaking_rate=14",
            "--silence-prefill-frames",
            "2",
        ]);
        zonos3.check(TtsFamily::Zonos3).unwrap();
        let error = zonos3.check(TtsFamily::Qwen3Tts).unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Illegal);
        assert!(
            error.to_string().contains("--condition is a zonos3 flag"),
            "{error}"
        );
        let qwen = flags(&["--placement", "interleaved", "--predictor-top-k", "20"]);
        qwen.check(TtsFamily::Qwen3Tts).unwrap();
        assert!(qwen.check(TtsFamily::Zonos3).is_err());
        TtsFamilyArgs::default().check(TtsFamily::Zonos3).unwrap();
    }

    /// `--condition` values, by name; malformed ones refused.
    #[test]
    fn test_condition_values() {
        let values = flags(&[
            "--condition",
            "loudness=-20",
            "--condition",
            " fmax = 12000",
        ])
        .condition_values()
        .unwrap();
        assert_eq!(
            values,
            BTreeMap::from([
                ("fmax".to_string(), 12000.0),
                ("loudness".to_string(), -20.0)
            ])
        );
        for bad in [["--condition", "loud"], ["--condition", "loud=x"]] {
            assert!(flags(&bad).condition_values().is_err(), "{bad:?}");
        }
        let twice = flags(&["--condition", "a=1", "--condition", "a=2"]);
        assert!(twice.condition_values().is_err());
    }
}
