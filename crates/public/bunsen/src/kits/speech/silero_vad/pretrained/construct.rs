//! # Constructing a Silero VAD pretrained

use alloc::sync::Arc;

use burn::tensor::Device;

use crate::{
    data::pretrained::{
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::silero_vad::{
        SileroVadCollection,
        pretrained::{
            BURNPACK,
            BURNPACK_KIND,
            CHECKPOINT,
            SAFETENSORS_KIND,
            SILERO_KIT,
        },
    },
};

/// How a Silero checkpoint is read, by its resource map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SileroReader {
    /// A burnpack under [`BURNPACK`] ([`BURNPACK_KIND`]): both sample-rate
    /// branches, keyed as `burn-onnx` names the ONNX graph's nodes.
    #[default]
    Burnpack,

    /// Upstream's 16 kHz state dict under [`CHECKPOINT`]
    /// ([`SAFETENSORS_KIND`]), read through
    /// [`upstream_to_bunsen_path`](super::load::upstream_to_bunsen_path).
    /// Needs the `store_safetensors` feature.
    Safetensors,
}

/// How a Silero pretrained is built: its checkpoint read by the
/// [`SileroReader`] its map calls for, into a [`SileroVadCollection`]: both
/// branches from a burnpack, the 16 kHz branch from upstream's safetensors.
///
/// The [`Construct`] hook for the Silero kit.
/// [`default_silero_factory`](super::default_silero_factory) attaches it;
/// a caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct SileroConstruct {
    /// The checkpoint's reader.
    pub reader: SileroReader,
}

impl SileroConstruct {
    /// The hook, reading a burnpack.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Construct for SileroConstruct {
    type Built = SileroVadCollection;

    const KIT: &'static str = SILERO_KIT;

    /// The reader for the map's checkpoint: a burnpack under [`BURNPACK`],
    /// or upstream's safetensors under [`CHECKPOINT`].
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// map with neither, or a checkpoint of another kind.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
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
                "a Silero resource map needs a `{BURNPACK}` or a `{CHECKPOINT}`"
            )));
        };
        Ok(Self { reader })
    }

    /// Reads the checkpoint: a burnpack into the standard 16 kHz and 8 kHz
    /// models, upstream's safetensors into the 16 kHz one.
    fn construct(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<SileroVadCollection>> {
        Ok(Arc::new(match self.reader {
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
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hook names its kit, and follows the map's checkpoint.
    #[test]
    fn test_the_hook_follows_the_map() {
        assert_eq!(<SileroConstruct as Construct>::KIT, "silero_vad");
        assert_eq!(SileroConstruct::new().reader, SileroReader::Burnpack);

        #[cfg(feature = "store_safetensors")]
        assert_eq!(
            SileroConstruct::for_map(&super::super::SILERO_VAD_16K_FILES.to_map())
                .unwrap()
                .reader,
            SileroReader::Safetensors
        );
        // A burnpack given by path: unlabeled, read as a burnpack.
        let mut burnpack = crate::data::pretrained::ResourceMap::new("a burnpack");
        burnpack.insert(crate::data::pretrained::Resource::given(
            BURNPACK,
            "/models/silero_vad.bpk",
        ));
        assert_eq!(
            SileroConstruct::for_map(&burnpack).unwrap().reader,
            SileroReader::Burnpack
        );
    }
}
