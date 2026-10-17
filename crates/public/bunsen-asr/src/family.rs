//! The model families, and loading one of their models with its info card.
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

impl Family {
    /// Brings every resource of each of `names` into `cache`, fetching what
    /// is not local, and writes where each one is to `out`.
    ///
    /// # Arguments
    /// * `verify` - re-hash each pinned file once it is local, the cached ones
    ///   included.
    ///
    /// # Errors
    /// As the family's factory resolving a name, its plan, and the cache's
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
            Family::Parakeet => fetch_with(default_parakeet_factory()?, names, cache, verify, out),
            Family::Qwen3Asr => fetch_with(default_qwen3_asr_factory()?, names, cache, verify, out),
            Family::Whisper => fetch_with(default_whisper_factory()?, names, cache, verify, out),
        }
    }
}

/// [`Family::fetch`] through one family's factory: each name's plan (a
/// path's vocabulary is the one its checkpoint's layout selects, and a
/// name's checkpoint must scan as its prefab), then every resource of it.
fn fetch_with<H: Construct>(
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
        for family in [Family::Parakeet, Family::Qwen3Asr, Family::Whisper] {
            let unknown = family.fetch(&["no/such-model".into()], &cache, false, &mut out);
            assert!(unknown.is_err(), "{}", family.name());
            let known = family.fetch(&[family.default_model().into()], &cache, false, &mut out);
            assert!(known.is_err(), "{} fetched offline", family.name());
        }
        assert!(cache.cached_files(&Default::default()).unwrap().is_empty());
    }
}
