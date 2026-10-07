//! # Constructing a Silero VAD pretrained

use std::sync::Arc;

use burn::tensor::Device;

use crate::{
    data::pretrained::{
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
    },
    errors::BunsenResult,
    kits::speech::silero_vad::{
        SileroVadCollection,
        pretrained::{
            BURNPACK,
            SILERO_KIT,
        },
    },
};

/// How a Silero pretrained is built: the burnpack, read into both
/// branches.
///
/// The [`Construct`] hook for the Silero kit: from a resolved model to a
/// [`SileroVadCollection`], both sample-rate branches read from the one
/// burnpack with upstream's keying.
/// [`default_silero_factory`](super::default_silero_factory) attaches it;
/// a caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct SileroConstruct;

impl SileroConstruct {
    /// The hook.
    pub fn new() -> Self {
        Self
    }
}

impl Construct for SileroConstruct {
    type Built = SileroVadCollection;

    const KIT: &'static str = SILERO_KIT;

    /// The one hook: every row is a burnpack.
    fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
        Ok(Self)
    }

    /// Reads the burnpack into the standard 16 kHz and 8 kHz models.
    fn construct(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<SileroVadCollection>> {
        Ok(Arc::new(SileroVadCollection::load_from_burnpack_file(
            loaded.expect(BURNPACK)?,
            device,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hook names its kit.
    #[test]
    fn test_the_hook_names_its_kit() {
        assert_eq!(<SileroConstruct as Construct>::KIT, "silero_vad");
        assert!(format!("{:?}", SileroConstruct::new()).contains("SileroConstruct"));
    }
}
