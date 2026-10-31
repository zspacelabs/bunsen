//! The model families and kits, and loading a model with its info card.
use std::{
    io::Write,
    sync::Arc,
    time::Instant,
};

use bunsen::{
    data::{
        cache::verify_sha256,
        pretrained::{
            Construct,
            Deferred,
            Pretrained,
            PretrainedCache,
            PretrainedFactory,
        },
    },
    errors::{
        BunsenResult,
        sys_at,
    },
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
        sortformer::pretrained::{
            DEFAULT_SORTFORMER_MODEL,
            SORTFORMER_KIT,
            default_sortformer_factory,
        },
        voxtral::{
            VoxtralBundle,
            pretrained::{
                DEFAULT_VOXTRAL_MODEL,
                VOXTRAL_KIT,
                default_voxtral_factory,
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

use crate::{
    transcribe_cmd::FamilyFlags,
    whisper,
};

/// A model family: one kit, its factory, and its default model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// NVIDIA's Parakeet TDT.
    Parakeet,

    /// Qwen's Qwen3-ASR.
    Qwen3Asr,

    /// Mistral's Voxtral Realtime.
    Voxtral,

    /// `OpenAI`'s Whisper.
    Whisper,
}

/// What `list` and `fetch` serve: a speech recognizer's family, or the
/// diarizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kit {
    /// A speech recognizer's family.
    Asr(Family),

    /// The speaker diarizer: Sortformer.
    Diarizer,
}

impl From<Family> for Kit {
    fn from(family: Family) -> Self {
        Kit::Asr(family)
    }
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

    /// A Voxtral Realtime bundle.
    Voxtral(Arc<VoxtralBundle>),

    /// A Whisper bundle, which `transcribe` drives through a stream driver.
    Whisper(Arc<WhisperBundle>),
}

impl Family {
    /// Every family.
    pub const ALL: [Family; 4] = [
        Family::Parakeet,
        Family::Qwen3Asr,
        Family::Voxtral,
        Family::Whisper,
    ];

    /// The family whose command name is `name`.
    pub fn from_name(name: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The family's command name.
    pub fn name(self) -> &'static str {
        match self {
            Family::Parakeet => "parakeet",
            Family::Qwen3Asr => "qwen3-asr",
            Family::Voxtral => "voxtral",
            Family::Whisper => "whisper",
        }
    }

    /// The model `--model` names when it is not given.
    pub fn default_model(self) -> &'static str {
        match self {
            Family::Parakeet => DEFAULT_PARAKEET_MODEL,
            Family::Qwen3Asr => DEFAULT_QWEN3_ASR_MODEL,
            Family::Voxtral => DEFAULT_VOXTRAL_MODEL,
            Family::Whisper => "openai/base",
        }
    }

    /// The kit's segment of the pretrained cache.
    pub fn cache_kit(self) -> &'static str {
        match self {
            Family::Parakeet => PARAKEET_KIT,
            Family::Qwen3Asr => QWEN3_ASR_KIT,
            Family::Voxtral => VOXTRAL_KIT,
            Family::Whisper => WHISPER_KIT,
        }
    }

    /// Loads `spec` through `cache`, and logs its info card at `INFO`:
    /// a row of the family's factory, or for Whisper a checkpoint's path,
    /// with `flags`' `--vocab` over it.
    pub fn load<F: FamilyFlags>(
        self,
        spec: &str,
        flags: &F,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Loaded> {
        Ok(match self {
            Family::Parakeet => Loaded::Parakeet(load_logged(
                default_parakeet_factory()?.resolve(spec, cache)?,
                cache,
                device,
            )?),
            Family::Qwen3Asr => Loaded::Qwen3Asr(load_logged(
                default_qwen3_asr_factory()?.resolve(spec, cache)?,
                cache,
                device,
            )?),
            Family::Voxtral => Loaded::Voxtral(load_logged(
                default_voxtral_factory()?.resolve(spec, cache)?,
                cache,
                device,
            )?),
            Family::Whisper => Loaded::Whisper(load_logged(
                whisper::resolve(spec, flags.whisper(), cache)?,
                cache,
                device,
            )?),
        })
    }
}

impl Kit {
    /// The kit's command name.
    pub fn name(self) -> &'static str {
        match self {
            Kit::Asr(family) => family.name(),
            Kit::Diarizer => "diarize",
        }
    }

    /// The model `--model` names when it is not given.
    pub fn default_model(self) -> &'static str {
        match self {
            Kit::Asr(family) => family.default_model(),
            Kit::Diarizer => DEFAULT_SORTFORMER_MODEL,
        }
    }

    /// The kit's segment of the pretrained cache.
    pub fn cache_kit(self) -> &'static str {
        match self {
            Kit::Asr(family) => family.cache_kit(),
            Kit::Diarizer => SORTFORMER_KIT,
        }
    }

    /// What the kit's factory offers.
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
            Kit::Asr(Family::Parakeet) => split(default_parakeet_factory()?),
            Kit::Asr(Family::Qwen3Asr) => split(default_qwen3_asr_factory()?),
            Kit::Asr(Family::Voxtral) => split(default_voxtral_factory()?),
            Kit::Asr(Family::Whisper) => split(default_whisper_factory()?),
            Kit::Diarizer => split(default_sortformer_factory()?),
        })
    }

    /// Brings every resource of each of `names` into `cache`, fetching what
    /// is not local, and writes where each one is to `out`.
    ///
    /// # Arguments
    /// * `verify` - re-hash each pinned file once it is local, the cached ones
    ///   included.
    ///
    /// # Errors
    /// As the kit's factory resolving a name, its plan, and the cache's
    /// load; [`DigestMismatch`](bunsen::errors::DigestMismatch) when a file
    /// verified does not hash to its pin.
    pub fn fetch(
        self,
        names: &[String],
        cache: &PretrainedCache,
        verify: bool,
        out: &mut impl Write,
    ) -> BunsenResult<()> {
        match self {
            Kit::Asr(Family::Parakeet) => {
                fetch_with(default_parakeet_factory()?, names, cache, verify, out)
            }
            Kit::Asr(Family::Qwen3Asr) => {
                fetch_with(default_qwen3_asr_factory()?, names, cache, verify, out)
            }
            Kit::Asr(Family::Voxtral) => {
                fetch_with(default_voxtral_factory()?, names, cache, verify, out)
            }
            Kit::Asr(Family::Whisper) => {
                fetch_with(default_whisper_factory()?, names, cache, verify, out)
            }
            Kit::Diarizer => fetch_with(default_sortformer_factory()?, names, cache, verify, out),
        }
    }
}

/// [`Kit::fetch`] through one kit's factory: each name's plan (a
/// path's vocabulary is the one its checkpoint's layout selects, and a
/// name's checkpoint must scan as its prefab), then every resource of it.
pub fn fetch_with<H: Construct>(
    factory: PretrainedFactory<H>,
    names: &[String],
    cache: &PretrainedCache,
    verify: bool,
    out: &mut impl Write,
) -> BunsenResult<()> {
    let w = |e| sys_at("write to", "stdout")(e);
    for name in names {
        let model = factory.resolve(name, cache)?;
        let map = model.plan(cache)?;
        let loaded = cache.load(H::KIT, &map)?;
        writeln!(out, "{}:", model.id()).map_err(w)?;
        for (key, part) in loaded.iter() {
            writeln!(
                out,
                "  {key}: {} ({})",
                part.path.display(),
                part.provenance
            )
            .map_err(w)?;
            if verify {
                match map.get(key).and_then(|r| r.sha256.as_deref()) {
                    Some(sha256) => {
                        verify_sha256(&part.path, sha256)?;
                        writeln!(out, "    sha256 {sha256} ok").map_err(w)?;
                    }
                    None => {
                        writeln!(out, "    (unpinned; no digest to verify against)").map_err(w)?
                    }
                }
            }
        }
    }
    Ok(())
}

/// Resolves and loads `spec`, logging what it is and where its files came
/// from: the model's info card.
pub fn load_logged<H: Construct>(
    deferred: Deferred<H>,
    cache: &PretrainedCache,
    device: &Device,
) -> BunsenResult<Arc<H::Built>> {
    let t0 = Instant::now();
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

#[cfg(test)]
mod tests {
    use bunsen_app::cache::WeightsCacheArgs;

    use super::*;

    /// Offline, over an empty cache, a name the family does not have is a
    /// lookup error, and one it has cannot be fetched; nothing is written.
    #[test]
    fn test_fetch_offline() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WeightsCacheArgs {
            cache_dir: Some(dir.path().to_path_buf()),
            offline: true,
        }
        .init()
        .unwrap();
        let mut out = Vec::new();
        for kit in [
            Kit::Asr(Family::Parakeet),
            Kit::Asr(Family::Qwen3Asr),
            Kit::Asr(Family::Voxtral),
            Kit::Asr(Family::Whisper),
            Kit::Diarizer,
        ] {
            let unknown = kit.fetch(&["no/such-model".into()], &cache, false, &mut out);
            assert!(unknown.is_err(), "{}", kit.name());
            let known = kit.fetch(&[kit.default_model().into()], &cache, false, &mut out);
            assert!(known.is_err(), "{} fetched offline", kit.name());
        }
        assert!(cache.cached_files(&Default::default()).unwrap().is_empty());
    }
}
