//! Pretrained models

use std::path::{
    Path,
    PathBuf,
};

use burn::prelude::Backend;
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::silero_vad::{
        SileroVad,
        SileroVad16x8,
        SileroVadSignalConfig,
    },
};

impl<B: Backend> SileroVad16x8<B> {
    /// Load from a burnpack file.
    pub fn load_from_burnpack<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        let path = path.as_ref();
        let path: PathBuf = path.to_path_buf();

        let vad16: SileroVad<B> = {
            let cfg = SileroVadSignalConfig::standard_16khz();

            let mut store = BurnpackStore::from_file(path.clone())
                .with_remap_pattern("conv1d37", "stft")
                .with_remap_pattern("conv1d38", "encoder.blocks.0.conv")
                .with_remap_pattern("conv1d39", "encoder.blocks.1.conv")
                .with_remap_pattern("conv1d40", "encoder.blocks.2.conv")
                .with_remap_pattern("conv1d41", "encoder.blocks.3.conv")
                .with_remap_pattern("linear13", "lstm_hidden")
                .with_remap_pattern("linear14", "lstm_features")
                .with_remap_pattern("conv1d42", "decoder");

            // println!("keys: {:#?}", store.keys());

            let mut module = cfg.try_init(device)?;
            module
                .load_from(&mut store)
                .map_err(BunsenError::external)?;

            module
        };

        let vad8: SileroVad<B> = {
            let cfg = SileroVadSignalConfig::standard_8khz();

            let mut store = BurnpackStore::from_file(path.clone())
                .with_remap_pattern("conv1d43", "stft")
                .with_remap_pattern("conv1d44", "encoder.blocks.0.conv")
                .with_remap_pattern("conv1d45", "encoder.blocks.1.conv")
                .with_remap_pattern("conv1d46", "encoder.blocks.2.conv")
                .with_remap_pattern("conv1d47", "encoder.blocks.3.conv")
                .with_remap_pattern("linear15", "lstm_hidden")
                .with_remap_pattern("linear16", "lstm_features")
                .with_remap_pattern("conv1d48", "decoder");

            // println!("keys: {:#?}", store.keys());

            let mut module = cfg.try_init(device)?;
            module
                .load_from(&mut store)
                .map_err(BunsenError::external)?;

            module
        };

        Ok(SileroVad16x8 { vad16, vad8 })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use burn::{
        Tensor,
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;
    use crate::{
        errors::*,
        kits::speech::silero_vad::ReferenceVAD,
        support::testing::PerformanceBackend,
    };

    /// A valid chunk length for the given sample rate (standard Silero chunk).
    pub fn chunk_samples(sample_rate: usize) -> usize {
        match sample_rate {
            16000 => 512,
            8000 => 256,
            other => panic!("no test chunk for {other}"),
        }
    }

    fn silero_burnpack_path() -> PathBuf {
        PathBuf::from("/home/crutcher/git/fast-whisper-burn/src/vad/silero_vad_op18_ifless.bpk")
    }

    #[test]
    #[ignore]
    fn test_load_forward_pretrained() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let path = silero_burnpack_path();
        let device = Default::default();

        let s_mod: SileroVad16x8<B> =
            SileroVad16x8::load_from_burnpack(&path, &device).ok_or_panic();

        let r_mod: ReferenceVAD<B> = ReferenceVAD::from_file(path.to_str().unwrap(), &device);

        let batch = 2;
        let state = Tensor::zeros([2, batch, 128], &device);

        // 16khz
        {
            let sample_rate = 16000;
            let input = Tensor::<B, 2>::random(
                [batch, chunk_samples(sample_rate)],
                Distribution::Default,
                &device,
            );

            let (s_out, s_state) = s_mod.vad16.forward(input.clone(), state.clone());
            let (r_out, r_state) = r_mod.forward_16khz(input.clone(), state.clone());

            s_out
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }

        // 8khz
        {
            let sample_rate = 8000;

            let input = Tensor::<B, 2>::random(
                [batch, chunk_samples(sample_rate)],
                Distribution::Default,
                &device,
            );

            let (s_out, s_state) = s_mod.vad8.forward(input.clone(), state.clone());
            let (r_out, r_state) = r_mod.forward_8khz(input.clone(), state.clone());

            s_out
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }
    }
}
