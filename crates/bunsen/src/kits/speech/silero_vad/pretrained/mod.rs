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
        SileroVadAbstractConfig,
    },
};

/// Load a pretrained Silero VAD model from a file.
///
/// Returns (vad16, vad8)
pub fn load_pretrained_silero_vad<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> BunsenResult<(SileroVad<B>, SileroVad<B>)> {
    let path = path.as_ref();
    let path: PathBuf = path.to_path_buf();

    let vad16: SileroVad<B> = {
        let cfg = SileroVadAbstractConfig::standard_16khz();

        let mut store = BurnpackStore::from_file(path.clone())
            .with_remap_pattern("conv1d37", "stft")
            .with_remap_pattern("conv1d38", "encoder.blocks.0.conv")
            .with_remap_pattern("conv1d39", "encoder.blocks.1.conv")
            .with_remap_pattern("conv1d40", "encoder.blocks.2.conv")
            .with_remap_pattern("conv1d41", "encoder.blocks.3.conv")
            .with_remap_pattern("linear13", "hidden_gate")
            .with_remap_pattern("linear14", "input_gate")
            .with_remap_pattern("conv1d42", "decoder");

        // println!("keys: {:#?}", store.keys());

        let mut module = cfg.try_init(device)?;
        module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;

        module
    };

    let vad8: SileroVad<B> = {
        let cfg = SileroVadAbstractConfig::standard_8khz();

        let mut store = BurnpackStore::from_file(path.clone())
            .with_remap_pattern("conv1d43", "stft")
            .with_remap_pattern("conv1d44", "encoder.blocks.0.conv")
            .with_remap_pattern("conv1d45", "encoder.blocks.1.conv")
            .with_remap_pattern("conv1d46", "encoder.blocks.2.conv")
            .with_remap_pattern("conv1d47", "encoder.blocks.3.conv")
            .with_remap_pattern("linear15", "hidden_gate")
            .with_remap_pattern("linear16", "input_gate")
            .with_remap_pattern("conv1d48", "decoder");

        // println!("keys: {:#?}", store.keys());

        let mut module = cfg.try_init(device)?;
        module
            .load_from(&mut store)
            .map_err(BunsenError::external)?;

        module
    };

    Ok((vad16, vad8))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        errors::*,
        support::testing::CpuBackend,
    };

    #[test]
    #[ignore]
    fn test_load_pretrained() {
        type B = CpuBackend;

        let path = PathBuf::from(
            "/home/crutcher/git/fast-whisper-burn/src/vad/silero_vad_op18_ifless.bpk",
        );
        let device = Default::default();

        let (_vad16, _vad8) = load_pretrained_silero_vad::<B, _>(path, &device).ok_or_panic();
    }
}
