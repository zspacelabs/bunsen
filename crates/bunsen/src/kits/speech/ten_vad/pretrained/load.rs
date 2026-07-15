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
    kits::speech::ten_vad::{
        TenVad,
        TenVadStructureConfig,
        reference,
    },
};

impl<B: Backend> TenVad<B> {
    /// Load the common pretrained `TenVAD` model.
    pub fn load_pretrained(device: &B::Device) -> BunsenResult<Self> {
        Self::load_from_burnpack_bytes(reference::burnpack_as_burn_bytes(), device)
    }

    /// The key remapping for the pretrained model.
    pub fn pretrained_mapper() -> KeyRemapper {
        KeyRemapper::from_patterns(vec![
            ("conv2d1", "cs1.blocks.0.conv"),
            ("conv2d2", "cs1.blocks.1.conv"),
            ("maxpool2d", "maxpool"),
            ("conv2d3", "cs2.blocks.0.conv"),
            ("conv2d4", "cs2.blocks.1.conv"),
            ("conv2d5", "cs2.blocks.2.conv"),
            ("conv2d6", "cs2.blocks.3.conv"),
            ("constant23", "linear1.bias"),
            ("constant27", "linear2.bias"),
        ])
        .ok_or_panic()
    }

    /// Load from pretrained burnpack bytes.
    pub fn load_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        Self::load_from_burnpack(
            BurnpackStore::from_bytes(Some(bytes)),
            TenVadStructureConfig::default(),
            Self::pretrained_mapper(),
            device,
        )
    }

    /// Load from a burnpack file.
    pub fn load_from_burnpack_file(
        path: impl AsRef<std::path::Path>,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        Self::load_from_burnpack(
            BurnpackStore::from_file(path),
            TenVadStructureConfig::default(),
            Self::pretrained_mapper(),
            device,
        )
    }

    /// Load from a burnpack store.
    pub fn load_from_burnpack<C>(
        store: BurnpackStore,
        cfg: C,
        remapper: KeyRemapper,
        device: &B::Device,
    ) -> BunsenResult<Self>
    where
        C: ModuleInit<B, Self>,
    {
        let mut store = store.remap(remapper);
        let mut module = cfg.try_init(device)?;
        module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;

        Ok(module)
    }
}
