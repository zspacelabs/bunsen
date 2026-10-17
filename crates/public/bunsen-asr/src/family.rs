//! The model families, and loading one of their models with its info card.
use std::{
    sync::Arc,
    time::Instant,
};

use bunsen::{
    data::pretrained::{
        Construct,
        Pretrained,
        PretrainedCache,
        PretrainedFactory,
    },
    errors::BunsenResult,
    kits::speech::{
        parakeet::{
            ParakeetBundle,
            pretrained::{
                DEFAULT_PARAKEET_MODEL,
                PARAKEET_KIT,
                default_parakeet_factory,
            },
        },
        qwen3_asr::{
            Qwen3AsrBundle,
            pretrained::{
                DEFAULT_QWEN3_ASR_MODEL,
                QWEN3_ASR_KIT,
                default_qwen3_asr_factory,
            },
        },
        whisper::{
            driver::WhisperBundle,
            pretrained::{
                WHISPER_KIT,
                default_whisper_factory,
            },
        },
    },
};
use burn::tensor::Device;

/// A model family: one kit, its factory, and its default model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// NVIDIA's Parakeet TDT.
    Parakeet,

    /// Qwen's Qwen3-ASR.
    Qwen3Asr,

    /// `OpenAI`'s Whisper.
    Whisper,
}

/// What a family's factory offers: its listed rows, with their providers,
/// and the providers that list nothing but resolve a ref (`hf`).
#[derive(Clone, Debug)]
pub struct Listing {
    /// The listed rows, by provider.
    pub rows: Vec<(String, Pretrained)>,

    /// The providers that list nothing.
    pub unlisted: Vec<String>,
}

/// A loaded model of one family.
#[derive(Clone, Debug)]
pub enum Loaded {
    /// A Parakeet TDT bundle.
    Parakeet(Arc<ParakeetBundle>),

    /// A Qwen3-ASR bundle.
    Qwen3Asr(Arc<Qwen3AsrBundle>),

    /// A Whisper bundle, which `transcribe` drives through a stream driver.
    Whisper(Arc<WhisperBundle>),
}

impl Family {
    /// The family's command name.
    pub fn name(self) -> &'static str {
        match self {
            Family::Parakeet => "parakeet",
            Family::Qwen3Asr => "qwen3-asr",
            Family::Whisper => "whisper",
        }
    }

    /// The model `--model` names when it is not given.
    pub fn default_model(self) -> &'static str {
        match self {
            Family::Parakeet => DEFAULT_PARAKEET_MODEL,
            Family::Qwen3Asr => DEFAULT_QWEN3_ASR_MODEL,
            Family::Whisper => "openai/base",
        }
    }

    /// The kit's segment of the pretrained cache.
    pub fn cache_kit(self) -> &'static str {
        match self {
            Family::Parakeet => PARAKEET_KIT,
            Family::Qwen3Asr => QWEN3_ASR_KIT,
            Family::Whisper => WHISPER_KIT,
        }
    }

    /// What the family's factory offers.
    pub fn listing(self) -> BunsenResult<Listing> {
        fn split<H: Construct>(factory: PretrainedFactory<H>) -> Listing {
            let rows = factory.list();
            let listed: Vec<&str> = rows.iter().map(|(p, _)| p.as_str()).collect();
            let unlisted = factory
                .providers()
                .iter()
                .map(|p| p.name().to_string())
                .filter(|p| !listed.contains(&p.as_str()))
                .collect();
            Listing { rows, unlisted }
        }
        Ok(match self {
            Family::Parakeet => split(default_parakeet_factory()?),
            Family::Qwen3Asr => split(default_qwen3_asr_factory()?),
            Family::Whisper => split(default_whisper_factory()?),
        })
    }

    /// Loads `spec` through `cache`, and logs its info card at `INFO`.
    pub fn load(
        self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Loaded> {
        Ok(match self {
            Family::Parakeet => Loaded::Parakeet(load_logged(
                default_parakeet_factory()?,
                spec,
                cache,
                device,
            )?),
            Family::Qwen3Asr => Loaded::Qwen3Asr(load_logged(
                default_qwen3_asr_factory()?,
                spec,
                cache,
                device,
            )?),
            Family::Whisper => Loaded::Whisper(load_logged(
                default_whisper_factory()?,
                spec,
                cache,
                device,
            )?),
        })
    }
}

/// Resolves and loads `spec`, logging what it is and where its files came
/// from: the model's info card.
fn load_logged<H: Construct>(
    factory: PretrainedFactory<H>,
    spec: &str,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<Arc<H::Built>> {
    let t0 = Instant::now();
    let deferred = factory.resolve(spec, cache)?;
    let loaded = deferred.load(cache, device)?;
    log::info!("model: {}", deferred.id());
    if let Some((_, row)) = deferred.model.named() {
        log::info!("  {}", row.description);
        if let Some(license) = &row.license {
            log::info!("  license: {license}");
        }
        if let Some(origin) = &row.origin {
            log::info!("  origin: {origin}");
        }
    }
    for (key, part) in &loaded.resources.parts {
        log::info!("  {key}: {} ({})", part.path.display(), part.provenance);
    }
    log::info!("  device: {device:?}; loaded in {:.1?}", t0.elapsed());
    Ok(loaded.handle)
}
