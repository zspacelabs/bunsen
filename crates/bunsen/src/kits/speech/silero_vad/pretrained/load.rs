use std::path::{
    Path,
    PathBuf,
};

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

impl<B: Backend> SileroVadCollection<B> {
    /// Load from a burnpack file.
    pub fn load_pretrained(device: &B::Device) -> BunsenResult<Self> {
        Self::load_from_burnpack_bytes(reference::burnpack_as_burn_bytes(), device)
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

    fn load_from_burnpack(
        store: BurnpackStore,
        cfg: SileroVadSignalConfig,
        remapper: KeyRemapper,
        device: &B::Device,
    ) -> BunsenResult<SileroVad<B>> {
        let mut store = store.remap(remapper);

        let mut module = cfg.try_init(device)?;
        module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;

        Ok(module)
    }

    /// Load from a burnpack file.
    pub fn load_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        let vad16: SileroVad<B> = Self::load_from_burnpack(
            BurnpackStore::from_bytes(Some(bytes.clone())),
            SileroVadSignalConfig::standard_16khz(),
            Self::pretrained_16khz_remapper(),
            device,
        )?;

        let vad8: SileroVad<B> = Self::load_from_burnpack(
            BurnpackStore::from_bytes(Some(bytes.clone())),
            SileroVadSignalConfig::standard_8khz(),
            Self::pretrained_8khz_remapper(),
            device,
        )?;

        Ok(SileroVadCollection {
            branches: vec![(16000, vad16), (8000, vad8)],
        })
    }

    /// Load from a burnpack file.
    pub fn load_from_burnpack_path<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        let path = path.as_ref();
        let path: PathBuf = path.to_path_buf();

        let vad16: SileroVad<B> = Self::load_from_burnpack(
            BurnpackStore::from_file(path.clone()),
            SileroVadSignalConfig::standard_16khz(),
            Self::pretrained_16khz_remapper(),
            device,
        )?;

        let vad8: SileroVad<B> = Self::load_from_burnpack(
            BurnpackStore::from_file(path.clone()),
            SileroVadSignalConfig::standard_8khz(),
            Self::pretrained_8khz_remapper(),
            device,
        )?;

        Ok(SileroVadCollection {
            branches: vec![(16000, vad16), (8000, vad8)],
        })
    }
}
