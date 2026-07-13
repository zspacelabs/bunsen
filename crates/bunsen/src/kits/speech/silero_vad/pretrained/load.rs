use std::path::Path;

use burn::prelude::Backend;
use burn_store::{
    BurnpackStore,
    KeyRemapper,
    ModuleSnapshot,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::silero_vad::{
        SileroVad,
        SileroVadCollection,
        SileroVadSignalConfig,
        reference,
    },
};

impl<B: Backend> SileroVad<B> {
    /// Load the pretrained 16khz model.
    pub fn load_16khz_pretrained(device: &B::Device) -> BunsenResult<Self> {
        Self::load_16khz_from_burnpack_bytes(reference::burnpack_as_burn_bytes(), device)
    }

    /// Load the pretrained 8khz model.
    pub fn load_8khz_pretrained(device: &B::Device) -> BunsenResult<Self> {
        Self::load_8khz_from_burnpack_bytes(reference::burnpack_as_burn_bytes(), device)
    }

    /// Load the 16khz model from pretrained burnpack bytes.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_16khz_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        Self::load_from_burnpack(
            BurnpackStore::from_bytes(Some(bytes)),
            SileroVadSignalConfig::standard_16khz(),
            Self::pretrained_16khz_remapper(),
            device,
        )
    }

    /// Load the 16khz model from a burnpack file.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_16khz_from_burnpack_file(
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        Self::load_from_burnpack(
            BurnpackStore::from_file(path),
            SileroVadSignalConfig::standard_16khz(),
            Self::pretrained_16khz_remapper(),
            device,
        )
    }

    /// Load the 8khz model from pretrained burnpack bytes.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_8khz_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        Self::load_from_burnpack(
            BurnpackStore::from_bytes(Some(bytes)),
            SileroVadSignalConfig::standard_8khz(),
            Self::pretrained_8khz_remapper(),
            device,
        )
    }

    /// Load the 8khz model from a burnpack file.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_8khz_from_burnpack_file(
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        Self::load_from_burnpack(
            BurnpackStore::from_file(path),
            SileroVadSignalConfig::standard_8khz(),
            Self::pretrained_8khz_remapper(),
            device,
        )
    }

    /// The key remapping for the 16khz model.
    pub fn pretrained_16khz_remapper() -> KeyRemapper {
        KeyRemapper::from_patterns(vec![
            ("conv1d37", "stft"),
            ("conv1d38", "encoder.blocks.0.conv"),
            ("conv1d39", "encoder.blocks.1.conv"),
            ("conv1d40", "encoder.blocks.2.conv"),
            ("conv1d41", "encoder.blocks.3.conv"),
            ("linear13", "lstm_hidden"),
            ("linear14", "lstm_features"),
            ("conv1d42", "decoder"),
        ])
        .ok_or_panic()
    }

    /// The key remapping for the 8khz model.
    pub fn pretrained_8khz_remapper() -> KeyRemapper {
        KeyRemapper::from_patterns(vec![
            ("conv1d43", "stft"),
            ("conv1d44", "encoder.blocks.0.conv"),
            ("conv1d45", "encoder.blocks.1.conv"),
            ("conv1d46", "encoder.blocks.2.conv"),
            ("conv1d47", "encoder.blocks.3.conv"),
            ("linear15", "lstm_hidden"),
            ("linear16", "lstm_features"),
            ("conv1d48", "decoder"),
        ])
        .ok_or_panic()
    }

    /// Load from a burnpack store.
    pub fn load_from_burnpack<C>(
        store: BurnpackStore,
        cfg: C,
        remapper: KeyRemapper,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>>
    where
        C: ModuleInit<B, SileroVad<B>>,
    {
        let mut store = store.remap(remapper);
        let mut module = cfg.try_init(device)?;
        module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;

        Ok(module)
    }
}

impl<B: Backend> SileroVadCollection<B> {
    fn new_common_collection(
        vad_16: SileroVad<B>,
        vad_8: SileroVad<B>,
    ) -> Self {
        Self {
            branches: vec![(16000, vad_16), (8000, vad_8)],
        }
    }

    /// Load the standard 16khz/8khz pretrained models.
    pub fn load_pretrained(device: &B::Device) -> BunsenResult<Self> {
        Ok(Self::new_common_collection(
            SileroVad::load_16khz_pretrained(device)?,
            SileroVad::load_8khz_pretrained(device)?,
        ))
    }

    /// Load the standard 16khz/8khz pretrained models from burnpack bytes.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        Ok(Self::new_common_collection(
            SileroVad::load_16khz_from_burnpack_bytes(bytes.clone(), device)?,
            SileroVad::load_8khz_from_burnpack_bytes(bytes, device)?,
        ))
    }

    /// Load the standard 16khz/8khz pretrained models from a burnpack file.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_from_burnpack_file<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        let path = path.as_ref();
        Ok(Self::new_common_collection(
            SileroVad::load_16khz_from_burnpack_file(path, device)?,
            SileroVad::load_8khz_from_burnpack_file(path, device)?,
        ))
    }
}
