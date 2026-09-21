//! # Constructing a Silero VAD pretrained
//!
//! The [`Construct`] hook for the Silero kit: from a resolved model to a
//! [`SileroVadCollection`], both sample-rate branches read from the one
//! burnpack with upstream's keying.

use std::sync::Arc;

use burn::prelude::Backend;

use crate::{
    data::pretrained::{
        Construct,
        LoadedResources,
        PretrainedRef,
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
#[derive(Clone, Debug, Default)]
pub struct SileroConstruct;

impl SileroConstruct {
    /// The hook.
    pub fn new() -> Self {
        Self
    }
}

impl Construct for SileroConstruct {
    type Built<B: Backend> = SileroVadCollection<B>;

    const GIVEN_KEY: Option<&'static str> = Some(BURNPACK);
    const KIT: &'static str = SILERO_KIT;

    /// Reads the burnpack into the standard 16 kHz and 8 kHz models.
    fn construct<B: Backend>(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &B::Device,
    ) -> BunsenResult<Arc<SileroVadCollection<B>>> {
        Ok(Arc::new(SileroVadCollection::load_from_burnpack_file(
            loaded.expect(BURNPACK)?,
            device,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hook names its kit and the key a bare burnpack path fills.
    #[test]
    fn test_the_hook_names_its_kit_and_key() {
        assert_eq!(<SileroConstruct as Construct>::KIT, "silero_vad");
        assert_eq!(<SileroConstruct as Construct>::GIVEN_KEY, Some("burnpack"));
        assert!(format!("{:?}", SileroConstruct::new()).contains("SileroConstruct"));
    }
}
