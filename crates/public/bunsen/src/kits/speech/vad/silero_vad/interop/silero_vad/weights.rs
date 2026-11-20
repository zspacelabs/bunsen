//! Loading upstream's `silero_vad_16k.safetensors`.
use std::path::Path;

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::Device,
};

use super::silero_vad_rekey;
use crate::{
    burner::module::ModuleInit,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
        sys_at,
    },
    kits::speech::vad::silero_vad::{
        SileroVad,
        SileroVadSignalConfig,
    },
};

/// Loads tensors named as upstream's `silero_vad_16k.safetensors` names them
/// into `model`: [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to)
/// under [`silero_vad_rekey`].
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to):
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
/// when a tensor does not fit, or a parameter is missing.
pub fn load_silero_vad_tensors(
    model: &mut SileroVad,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    silero_vad_rekey().apply_to(model, tensors)
}

/// The 16 kHz model of upstream's `silero_vad_16k.safetensors` at `path`.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
/// when the file is not a safetensors file; otherwise as
/// [`load_silero_vad_tensors`]; under a frame naming the file.
pub fn load_silero_vad_16khz_from_path(
    path: impl AsRef<Path>,
    device: &Device,
) -> BunsenResult<SileroVad> {
    use burn::store::{
        ModuleStore,
        SafetensorsStore,
    };

    let path = path.as_ref();
    let read = || -> BunsenResult<SileroVad> {
        std::fs::metadata(path).map_err(sys_at("open", path))?;
        let tensors: Vec<_> = SafetensorsStore::from_file(path)
            .get_all_tensors()
            .map_err(|e| {
                BunsenError::invalid_resource("the store refused the checkpoint").with_cause(e)
            })?
            .values()
            .cloned()
            .collect();
        let mut vad: SileroVad = SileroVadSignalConfig::standard_16khz().try_init(device)?;
        load_silero_vad_tensors(&mut vad, tensors)?;
        Ok(vad)
    };
    read().with_context(|| format!("loading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state dict with upstream's names and shapes loads into every
    /// parameter, the LSTM's linears transposed as `PyTorch`'s are.
    #[test]
    fn test_load_upstream_tensors() {
        use burn::{
            prelude::TensorData,
            store::bridge,
        };

        use crate::support::testing::cpu_device;

        let shapes: [(&str, &[usize]); 15] = [
            ("stft_conv.weight", &[258, 1, 256]),
            ("conv1.weight", &[128, 129, 3]),
            ("conv1.bias", &[128]),
            ("conv2.weight", &[64, 128, 3]),
            ("conv2.bias", &[64]),
            ("conv3.weight", &[64, 64, 3]),
            ("conv3.bias", &[64]),
            ("conv4.weight", &[128, 64, 3]),
            ("conv4.bias", &[128]),
            ("lstm_cell.weight_ih", &[512, 128]),
            ("lstm_cell.weight_hh", &[512, 128]),
            ("lstm_cell.bias_ih", &[512]),
            ("lstm_cell.bias_hh", &[512]),
            ("final_conv.weight", &[1, 128, 1]),
            ("final_conv.bias", &[1]),
        ];
        let tensors = shapes.iter().map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|k| k as f32 / n as f32).collect();
            bridge::from_data(
                TensorData::new(values, shape.to_vec()),
                name.to_string(),
                None,
            )
        });
        let mut vad: SileroVad = SileroVadSignalConfig::standard_16khz()
            .try_init(&cpu_device())
            .unwrap();
        load_silero_vad_tensors(&mut vad, tensors).unwrap();

        // `weight_ih[1, 0]` is the 128th value; burn's linear holds its
        // transpose.
        let w: Vec<f32> = vad
            .lstm
            .features
            .weight
            .val()
            .slice([0..1, 1..2])
            .into_data()
            .try_into_vec_as()
            .unwrap();
        let n = 512 * 128;
        assert_eq!(w, [128.0 / n as f32]);
        let b: Vec<f32> = vad
            .decoder
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_data()
            .try_into_vec_as()
            .unwrap();
        assert_eq!(b, [0.0]);
    }
}
