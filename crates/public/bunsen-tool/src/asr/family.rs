//! The model families and kits, and loading a model with its info card.
use std::{
    io::Write,
    sync::Arc,
    time::Instant,
};

use bunsen::{
    data::{
        builds::{
            BuildFactory,
            BuildProvider,
            ConstructInit,
            Deferred,
        },
        cache::verify_sha256,
        catalog::{
            ResolvedBuild,
            lookup::{
                LeafMatch,
                name_matches,
            },
        },
        formats::LoadPolicy,
        resources::{
            ResourceCache,
            ResourceCacheOptions,
        },
    },
    errors::{
        BunsenResult,
        sys_at,
    },
    kits::speech::{
        asr::{
            parakeet::{
                ParakeetConstruct,
                pretrained::{
                    DEFAULT_PARAKEET_MODEL,
                    default_parakeet_factory,
                },
            },
            qwen3_asr::{
                Qwen3AsrConstruct,
                pretrained::{
                    DEFAULT_QWEN3_ASR_MODEL,
                    default_qwen3_asr_factory,
                },
            },
            voxtral::{
                VoxtralConstruct,
                pretrained::{
                    DEFAULT_VOXTRAL_MODEL,
                    default_voxtral_factory,
                },
            },
            whisper::{
                driver::WhisperConstruct,
                pretrained::default_whisper_factory,
            },
        },
        diarization::sortformer::pretrained::{
            DEFAULT_SORTFORMER_MODEL,
            default_sortformer_factory,
        },
    },
};
use burn::tensor::Device;

use crate::asr::{
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    voxtral::VoxtralFlags,
    whisper,
};

/// A model family: one kit, its factory, and its default model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AsrFamily {
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
pub enum AsrKit {
    /// A speech recognizer's family.
    Asr(AsrFamily),

    /// The speaker diarizer: Sortformer.
    Diarizer,
}

impl From<AsrFamily> for AsrKit {
    fn from(family: AsrFamily) -> Self {
        AsrKit::Asr(family)
    }
}

/// What a family's factory offers: its listed rows, with their providers,
/// and the providers that list nothing but resolve a ref (`hf`).
#[derive(Clone, Debug)]
pub struct AsrFamilyListing {
    /// The listed rows, by provider.
    pub rows: Vec<(String, ResolvedBuild)>,

    /// Every provider, in search order.
    pub providers: Vec<Arc<dyn BuildProvider>>,

    /// The providers that list nothing.
    pub unlisted: Vec<String>,
}

/// A loaded model of one family.
#[derive(Clone, Debug)]
pub enum AsrConstructWrapper {
    /// A Parakeet TDT construct.
    Parakeet(Arc<ParakeetConstruct>),

    /// A Qwen3-ASR construct.
    Qwen3Asr(Arc<Qwen3AsrConstruct>),

    /// A Voxtral Realtime construct.
    Voxtral(Arc<VoxtralConstruct>),

    /// A Whisper construct, which `transcribe` drives through a stream driver.
    Whisper(Arc<WhisperConstruct>),
}

impl<'de> serde::Deserialize<'de> for AsrFamily {
    /// By [`name`](AsrFamily::name): `parakeet`, `qwen3-asr`, `voxtral`,
    /// `whisper`.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        AsrFamily::from_name(&name).ok_or_else(|| {
            let names: Vec<&str> = AsrFamily::ALL.iter().map(|f| f.name()).collect();
            serde::de::Error::custom(format!("`{name}` is not a family: {}", names.join(", ")))
        })
    }
}

impl AsrFamily {
    /// Every family.
    pub const ALL: [AsrFamily; 4] = [
        AsrFamily::Parakeet,
        AsrFamily::Qwen3Asr,
        AsrFamily::Voxtral,
        AsrFamily::Whisper,
    ];

    /// The family whose command name is `name`.
    pub fn from_name(name: &str) -> Option<AsrFamily> {
        AsrFamily::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The family's command name.
    pub fn name(self) -> &'static str {
        match self {
            AsrFamily::Parakeet => "parakeet",
            AsrFamily::Qwen3Asr => "qwen3-asr",
            AsrFamily::Voxtral => "voxtral",
            AsrFamily::Whisper => "whisper",
        }
    }

    /// The model `--model` names when it is not given.
    pub fn default_model(self) -> &'static str {
        match self {
            AsrFamily::Parakeet => DEFAULT_PARAKEET_MODEL,
            AsrFamily::Qwen3Asr => DEFAULT_QWEN3_ASR_MODEL,
            AsrFamily::Voxtral => DEFAULT_VOXTRAL_MODEL,
            AsrFamily::Whisper => "openai/base",
        }
    }

    /// Loads `spec` through `cache`, and logs its info card at `INFO`:
    /// a row of the family's factory, or for Whisper a checkpoint's path,
    /// with `flags`' `--vocab` over it.
    pub fn load<F: FamilyFlags>(
        self,
        spec: &str,
        flags: &F,
        cache: &ResourceCache,
        device: &Device,
    ) -> BunsenResult<AsrConstructWrapper> {
        Ok(match self {
            AsrFamily::Parakeet => AsrConstructWrapper::Parakeet(load_logged(
                default_parakeet_factory()?.resolve(spec, cache)?,
                cache,
                device,
            )?),
            AsrFamily::Qwen3Asr => AsrConstructWrapper::Qwen3Asr(load_logged(
                default_qwen3_asr_factory()?.resolve(spec, cache)?,
                cache,
                device,
            )?),
            AsrFamily::Voxtral => AsrConstructWrapper::Voxtral(load_logged(
                default_voxtral_factory()?.resolve(spec, cache)?,
                cache,
                device,
            )?),
            AsrFamily::Whisper => AsrConstructWrapper::Whisper(load_logged(
                whisper::resolve(spec, flags.whisper(), cache)?,
                cache,
                device,
            )?),
        })
    }
}

impl AsrKit {
    /// The kit's command name.
    pub fn name(self) -> &'static str {
        match self {
            AsrKit::Asr(family) => family.name(),
            AsrKit::Diarizer => "diarize",
        }
    }

    /// The model `--model` names when it is not given.
    pub fn default_model(self) -> &'static str {
        match self {
            AsrKit::Asr(family) => family.default_model(),
            AsrKit::Diarizer => DEFAULT_SORTFORMER_MODEL,
        }
    }

    /// Whether the kit's factory lists `name`, a row's name or alias, with
    /// or without its provider (`catalog:openai/small`), case aside: a
    /// row named in the build catalog's form answers its shorter spellings too
    /// (`openai/small` for `openai/whisper/small`), as its provider's lookup
    /// takes them.
    ///
    /// # Errors
    /// As [`listing`](Self::listing).
    pub fn lists(
        self,
        name: &str,
    ) -> BunsenResult<bool> {
        let listing = self.listing()?;
        Ok(listing.rows.iter().any(|(provider, row)| {
            let bare = match name.split_once(':') {
                Some((p, bare)) if p.eq_ignore_ascii_case(provider) => bare,
                Some(_) => return false,
                None => name,
            };
            name_matches(bare, &row.name, LeafMatch::Exact) || row.answers_to(bare)
        }))
    }

    /// What the kit's factory offers.
    pub fn listing(self) -> BunsenResult<AsrFamilyListing> {
        fn split<H: ConstructInit>(factory: BuildFactory<H>) -> AsrFamilyListing {
            let rows = factory.list();
            let listed: Vec<&str> = rows.iter().map(|(p, _)| p.as_str()).collect();
            let unlisted = factory
                .providers()
                .iter()
                .map(|p| p.name().to_string())
                .filter(|p| !listed.contains(&p.as_str()))
                .collect();
            AsrFamilyListing {
                rows,
                providers: factory.providers().to_vec(),
                unlisted,
            }
        }
        Ok(match self {
            AsrKit::Asr(AsrFamily::Parakeet) => split(default_parakeet_factory()?),
            AsrKit::Asr(AsrFamily::Qwen3Asr) => split(default_qwen3_asr_factory()?),
            AsrKit::Asr(AsrFamily::Voxtral) => split(default_voxtral_factory()?),
            AsrKit::Asr(AsrFamily::Whisper) => split(default_whisper_factory()?),
            AsrKit::Diarizer => split(default_sortformer_factory()?),
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
        cache: &ResourceCache,
        verify: bool,
        out: &mut impl Write,
    ) -> BunsenResult<()> {
        match self {
            AsrKit::Asr(AsrFamily::Parakeet) => {
                fetch_with(default_parakeet_factory()?, names, cache, verify, out)
            }
            AsrKit::Asr(AsrFamily::Qwen3Asr) => {
                fetch_with(default_qwen3_asr_factory()?, names, cache, verify, out)
            }
            AsrKit::Asr(AsrFamily::Voxtral) => {
                fetch_with(default_voxtral_factory()?, names, cache, verify, out)
            }
            AsrKit::Asr(AsrFamily::Whisper) => {
                fetch_with(default_whisper_factory()?, names, cache, verify, out)
            }
            AsrKit::Diarizer => {
                fetch_with(default_sortformer_factory()?, names, cache, verify, out)
            }
        }
    }
}

/// [`AsrKit::fetch`] through one kit's factory: each name's plan (a
/// path's vocabulary is the one its checkpoint's layout selects, and a
/// name's checkpoint must scan as its prefab), then every resource of it.
pub fn fetch_with<H: ConstructInit>(
    factory: BuildFactory<H>,
    names: &[String],
    cache: &ResourceCache,
    verify: bool,
    out: &mut impl Write,
) -> BunsenResult<()> {
    let w = |e| sys_at("write to", "stdout")(e);
    for name in names {
        let model = factory.resolve(name, cache)?;
        let map = model.plan(cache)?;
        let loaded = cache.load(&map)?;
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
pub fn load_logged<H: ConstructInit>(
    deferred: Deferred<H>,
    cache: &ResourceCache,
    device: &Device,
) -> BunsenResult<Arc<H::Construct>> {
    let t0 = Instant::now();
    let loaded = deferred.load(cache, device, LoadPolicy::performance())?;
    log::info!("model: {}", deferred.id());
    if let Some((_, row)) = deferred.model.build() {
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

/// A family's own transcription flags: paths, not policy.
pub trait FamilyFlags: std::fmt::Debug {
    /// Whisper's, when these are Whisper's.
    fn whisper(&self) -> Option<&whisper::WhisperFlags> {
        None
    }

    /// The cache options these flags call for, over `options`.
    fn cache_options(
        &self,
        options: ResourceCacheOptions,
    ) -> ResourceCacheOptions {
        match self.whisper() {
            Some(whisper) => whisper.cache_options(options),
            None => options,
        }
    }
}

impl FamilyFlags for whisper::WhisperFlags {
    fn whisper(&self) -> Option<&whisper::WhisperFlags> {
        Some(self)
    }
}

impl FamilyFlags for ParakeetFlags {}

impl FamilyFlags for Qwen3AsrFlags {}

impl FamilyFlags for VoxtralFlags {}

/// `$body` with `$flags` the family `$family`'s own flags: their defaults, as
/// a preset sets no paths. A macro, as each family's flags are their own
/// type.
macro_rules! with_flags {
    ($family:expr, $flags:ident => $body:expr) => {
        match $family {
            $crate::asr::AsrFamily::Parakeet => {
                let $flags = $crate::asr::parakeet::ParakeetFlags::default();
                $body
            }
            $crate::asr::AsrFamily::Qwen3Asr => {
                let $flags = $crate::asr::qwen3_asr::Qwen3AsrFlags::default();
                $body
            }
            $crate::asr::AsrFamily::Voxtral => {
                let $flags = $crate::asr::voxtral::VoxtralFlags::default();
                $body
            }
            $crate::asr::AsrFamily::Whisper => {
                let $flags = $crate::asr::whisper::WhisperFlags::default();
                $body
            }
        }
    };
}
pub(crate) use with_flags;

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
            AsrKit::Asr(AsrFamily::Parakeet),
            AsrKit::Asr(AsrFamily::Qwen3Asr),
            AsrKit::Asr(AsrFamily::Voxtral),
            AsrKit::Asr(AsrFamily::Whisper),
            AsrKit::Diarizer,
        ] {
            let unknown = kit.fetch(&["no/such-model".into()], &cache, false, &mut out);
            assert!(unknown.is_err(), "{}", kit.name());
            let known = kit.fetch(&[kit.default_model().into()], &cache, false, &mut out);
            assert!(known.is_err(), "{} fetched offline", kit.name());
        }
        assert!(cache.cached_files(&Default::default()).unwrap().is_empty());
    }
}
