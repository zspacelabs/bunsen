//! # Constructing a Silero VAD pretrained

use std::sync::Arc;

use burn::tensor::Device;

use crate::{
    data::pretrained::{
        LoadedResources,
        PretrainedRef,
        ResourceBundle,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::silero_vad::{
            SileroConstruct,
            SileroPolicy,
            SileroVadCollection,
            pretrained::{
                BURNPACK,
                BURNPACK_KIND,
                CHECKPOINT,
                SAFETENSORS_KIND,
                SILERO_KIT,
            },
        },
        support::{
            construct::ConstructInit,
            model_config::resolve_ref_presets,
        },
    },
};

/// How a Silero checkpoint is read, by its resource bundle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SileroReader {
    /// A burnpack under [`BURNPACK`] ([`BURNPACK_KIND`]): both sample-rate
    /// branches, keyed as `burn-onnx` names the ONNX graph's nodes.
    #[default]
    Burnpack,

    /// Upstream's 16 kHz state dict under [`CHECKPOINT`]
    /// ([`SAFETENSORS_KIND`]), read through
    /// [`upstream_rekey`](super::load::upstream_rekey).
    /// Needs the `store_safetensors` feature.
    Safetensors,
}

/// How a Silero pretrained is built: its checkpoint read by the
/// [`SileroReader`] its map calls for, into a [`SileroVadCollection`]: both
/// branches from a burnpack, the 16 kHz branch from upstream's safetensors;
/// and its [`SileroPolicy`] resolved through the row's presets, as the
/// lookup asked. Together, a [`SileroConstruct`].
///
/// The Silero kit's construct config: its [`ConstructInit`] hook.
/// [`default_silero_factory`](super::default_silero_factory) attaches it;
/// a caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct SileroConstructConfig {
    /// The checkpoint's reader.
    pub reader: SileroReader,
}

impl SileroConstructConfig {
    /// The hook, reading a burnpack.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ConstructInit for SileroConstructConfig {
    type Construct = SileroConstruct;

    const KIT: &'static str = SILERO_KIT;
    const RESOLVES_PRESETS: bool = true;

    /// The reader for the map's checkpoint: a burnpack under [`BURNPACK`],
    /// or upstream's safetensors under [`CHECKPOINT`].
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// map with neither, or a checkpoint of another kind.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        let reader = if let Some(r) = map.get(BURNPACK) {
            match r.kind.as_deref() {
                None | Some(BURNPACK_KIND) => SileroReader::Burnpack,
                other => {
                    return Err(BunsenError::unsupported(format!(
                        "no Silero reader for a burnpack of kind {other:?}"
                    )));
                }
            }
        } else if let Some(r) = map.get(CHECKPOINT) {
            match r.kind.as_deref() {
                Some(SAFETENSORS_KIND) if cfg!(feature = "store_safetensors") => {
                    SileroReader::Safetensors
                }
                Some(SAFETENSORS_KIND) => {
                    return Err(BunsenError::unsupported(
                        "a safetensors checkpoint needs bunsen's `store_safetensors` feature",
                    ));
                }
                other => {
                    return Err(BunsenError::unsupported(format!(
                        "no Silero reader for a checkpoint of kind {other:?}"
                    )));
                }
            }
        } else {
            return Err(BunsenError::unsupported(format!(
                "a Silero resource bundle needs a `{BURNPACK}` or a `{CHECKPOINT}`"
            )));
        };
        Ok(Self { reader })
    }

    /// Reads the checkpoint: a burnpack into the standard 16 kHz and 8 kHz
    /// models, upstream's safetensors into the 16 kHz one. The policy is the
    /// row's presets over [`SileroPolicy`]'s default, as the lookup asked
    /// ([`resolve_ref_presets`]).
    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<SileroConstruct>> {
        let policy = resolve_ref_presets(&SileroPolicy::default(), model)?;
        let vads = match self.reader {
            SileroReader::Burnpack => {
                SileroVadCollection::load_from_burnpack_file(loaded.expect(BURNPACK)?, device)?
            }
            #[cfg(feature = "store_safetensors")]
            SileroReader::Safetensors => SileroVadCollection::from_16khz(
                crate::kits::speech::silero_vad::SileroVad::load_16khz_from_safetensors_file(
                    loaded.expect(CHECKPOINT)?,
                    device,
                )?,
            ),
            #[cfg(not(feature = "store_safetensors"))]
            SileroReader::Safetensors => {
                return Err(BunsenError::unsupported(
                    "a safetensors checkpoint needs bunsen's `store_safetensors` feature",
                ));
            }
        };
        Ok(Arc::new(SileroConstruct {
            presets: policy.applied(),
            policy: policy.config,
            vads,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hook names its kit, and follows the map's checkpoint.
    #[test]
    fn test_the_hook_follows_the_map() {
        assert_eq!(<SileroConstructConfig as ConstructInit>::KIT, "silero_vad");
        assert_eq!(SileroConstructConfig::new().reader, SileroReader::Burnpack);

        #[cfg(feature = "store_safetensors")]
        assert_eq!(
            SileroConstructConfig::for_resource_bundle(
                &super::super::SILERO_CATALOG.resource_bundle("silero/vad-16k")
            )
            .unwrap()
            .reader,
            SileroReader::Safetensors
        );
        // A burnpack given by path: unlabeled, read as a burnpack.
        let mut burnpack = crate::data::pretrained::ResourceBundle::new("a burnpack");
        burnpack.insert(crate::data::pretrained::Resource::given(
            BURNPACK,
            "/models/silero_vad.bpk",
        ));
        assert_eq!(
            SileroConstructConfig::for_resource_bundle(&burnpack)
                .unwrap()
                .reader,
            SileroReader::Burnpack
        );
    }

    /// `silero/vad-16k`'s `vad-policy` axis: upstream's policy by default,
    /// `faster-whisper`'s by name, resolved with nothing fetched.
    #[test]
    fn test_the_vad_policy_presets() {
        use crate::kits::speech::{
            asr_driver::VoiceActivityFilterConfig,
            silero_vad::pretrained::default_silero_factory,
        };

        let factory = default_silero_factory().unwrap();
        let policy = |spec: &str| {
            factory
                .resolve_presets(spec, &SileroPolicy::default())
                .unwrap()
        };

        let silero = policy("silero/vad-16k");
        assert_eq!(silero.presets, ["silero"]);
        assert_eq!(
            silero.config.vad,
            VoiceActivityFilterConfig::new()
                .with_min_speech_ms(250)
                .with_min_silence_ms(100)
                .with_speech_pad_ms(30)
        );
        assert_eq!(policy("silero/vad-16k[silero]"), silero);

        let fw = policy("Silero/VAD-16K[Faster-Whisper]");
        assert_eq!(fw.presets, ["faster-whisper"]);
        assert_eq!(fw.config.vad, VoiceActivityFilterConfig::new());

        let patched = policy("silero/vad-16k{vad: {threshold: 0.6}}");
        assert_eq!(patched.presets, ["silero"]);
        assert_eq!(patched.config.vad.threshold, 0.6);
        assert_eq!(patched.config.vad.min_silence_ms, 100);

        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Policy)
            .message_contains("vad-policy")
            .assert_err(&factory.resolve_presets(
                "silero/vad-16k[silero,faster-whisper]",
                &SileroPolicy::default(),
            ));
    }
}
