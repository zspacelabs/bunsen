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

#[cfg(test)]
mod tests {
    use bunsen_silero_onnx::silero_vad_op18_ifless::Model as ReferenceModel;
    use burn::{
        Tensor,
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use crate::{
        errors::*,
        kits::speech::silero_vad::{
            SileroVadCollection,
            SileroVadMeta,
        },
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial_test::serial]
    fn test_load_forward_pretrained() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();

        let sc: SileroVadCollection<B> =
            SileroVadCollection::load_pretrained(&device).ok_or_panic();

        let r_mod: ReferenceModel<B> = ReferenceModel::from_bytes(
            bunsen_silero_onnx::silero_vad_op18_ifless::burnpack_as_burn_bytes(),
            &device,
        );

        let batch = 8;

        for sample_rate in [16000, 8000] {
            let vad = sc.expect_branch(sample_rate);

            if sample_rate == 16000 {
                assert_eq!(vad.chunk_size(), 512)
            }

            let input =
                Tensor::<B, 2>::random([batch, vad.chunk_size()], Distribution::Default, &device);
            let state = vad.init_state(batch, &device);

            // ([batch], [2, batch, d_hidden])
            let input1 = input.clone();
            let state1 = state.clone();
            let (s_out, s_state) = vad.forward(input1, state1);

            // ([batch, 1], [2, batch, d_hidden])
            let (r_out, r_state) = r_mod.forward(input, sample_rate as i64, state.clone());

            s_out
                .reshape([batch, 1])
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }
    }
}
