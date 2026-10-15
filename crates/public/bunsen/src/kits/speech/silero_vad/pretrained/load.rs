//! Load pretrained models.
use std::path::Path;

use burn::{
    store::{
        BurnpackStore,
        KeyRemapper,
        ModuleSnapshot,
        burn_pack::Error as BurnpackError,
    },
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        ResultContext,
        WithOkOrPanic,
        sys_at,
    },
    kits::speech::silero_vad::{
        SileroVad,
        SileroVadCollection,
        SileroVadSignalConfig,
    },
};

#[cfg(feature = "silero-weights")]
mod with_weights {
    use bunsen_bundled_silero::burnpack_as_burn_bytes;

    use super::*;

    impl SileroVad {
        /// Load the pretrained 16khz model.
        pub fn load_16khz_pretrained(device: &Device) -> BunsenResult<Self> {
            Self::load_16khz_from_burnpack_bytes(burnpack_as_burn_bytes(), device)
        }

        /// Load the pretrained 8khz model.
        pub fn load_8khz_pretrained(device: &Device) -> BunsenResult<Self> {
            Self::load_8khz_from_burnpack_bytes(burnpack_as_burn_bytes(), device)
        }
    }

    impl SileroVadCollection {
        /// Load the standard 16khz/8khz pretrained models.
        pub fn load_pretrained(device: &Device) -> BunsenResult<Self> {
            Ok(Self::new_common_collection(
                SileroVad::load_16khz_pretrained(device)?,
                SileroVad::load_8khz_pretrained(device)?,
            ))
        }
    }
}

impl SileroVad {
    /// Load the 16khz model from pretrained burnpack bytes.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_16khz_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &Device,
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
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when `path` is
    /// missing or forbidden; otherwise as
    /// [`load_from_burnpack`](Self::load_from_burnpack), under a frame
    /// naming the file.
    pub fn load_16khz_from_burnpack_file(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> BunsenResult<Self> {
        Self::load_from_burnpack_path(
            path,
            SileroVadSignalConfig::standard_16khz(),
            Self::pretrained_16khz_remapper(),
            device,
        )
    }

    /// Load the 8khz model from pretrained burnpack bytes.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_8khz_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &Device,
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
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when `path` is
    /// missing or forbidden; otherwise as
    /// [`load_from_burnpack`](Self::load_from_burnpack), under a frame
    /// naming the file.
    pub fn load_8khz_from_burnpack_file(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> BunsenResult<Self> {
        Self::load_from_burnpack_path(
            path,
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
            ("linear13", "lstm.hidden"),
            ("linear14", "lstm.features"),
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
            ("linear15", "lstm.hidden"),
            ("linear16", "lstm.features"),
            ("conv1d48", "decoder"),
        ])
        .ok_or_panic()
    }

    /// Load from a burnpack file, under a frame naming it.
    fn load_from_burnpack_path<C>(
        path: impl AsRef<Path>,
        cfg: C,
        remapper: KeyRemapper,
        device: &Device,
    ) -> BunsenResult<Self>
    where
        C: ModuleInit<Self>,
    {
        let path = path.as_ref();
        std::fs::metadata(path).map_err(sys_at("open", path))?;
        Self::load_from_burnpack(BurnpackStore::from_file(path), cfg, remapper, device)
            .with_context(|| format!("loading {}", path.display()))
    }

    /// Load from a burnpack store.
    ///
    /// # Errors
    /// - as `cfg`'s [`try_init`](ModuleInit::try_init);
    /// - [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    ///   with the [`BurnpackError`] as its cause, when the store is not a
    ///   burnpack or lacks a tensor the model needs;
    ///   [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    ///   burnpack version this build does not read;
    ///   [`Sys`](crate::errors::BunsenErrorKind::Sys) when reading it fails.
    pub fn load_from_burnpack<C>(
        store: BurnpackStore,
        cfg: C,
        remapper: KeyRemapper,
        device: &Device,
    ) -> BunsenResult<Self>
    where
        C: ModuleInit<Self>,
    {
        let mut store = store.remap(remapper);
        let mut module = cfg.try_init(device)?;
        module.load_from(&mut store).map_err(burnpack_load_error)?;

        Ok(module)
    }
}

/// Sorts a [`BurnpackError`] from loading a module by what it means.
fn burnpack_load_error(error: BurnpackError) -> BunsenError {
    let kind = match &error {
        BurnpackError::IoError(_) => BunsenErrorKind::Sys,
        BurnpackError::InvalidVersion => BunsenErrorKind::Unsupported,
        _ => BunsenErrorKind::InvalidResource,
    };
    BunsenError::from_cause(kind, error)
}

impl SileroVadCollection {
    pub(super) fn new_common_collection(
        vad_16: SileroVad,
        vad_8: SileroVad,
    ) -> Self {
        Self {
            branches: vec![(16000, vad_16), (8000, vad_8)],
        }
    }

    /// Load the standard 16khz/8khz pretrained models from burnpack bytes.
    /// Uses the upstream `silero_vad` keying.
    pub fn load_from_burnpack_bytes(
        bytes: burn::tensor::Bytes,
        device: &Device,
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
        device: &Device,
    ) -> BunsenResult<Self> {
        let path = path.as_ref();
        Ok(Self::new_common_collection(
            SileroVad::load_16khz_from_burnpack_file(path, device)?,
            SileroVad::load_8khz_from_burnpack_file(path, device)?,
        ))
    }
}

/// The bunsen module path of one tensor of upstream's
/// `silero_vad_16k.safetensors`, the 16 kHz model's `PyTorch` state dict, or
/// `None` for a name the model has no parameter for.
///
/// | upstream | bunsen |
/// |---|---|
/// | `stft_conv` | `stft` |
/// | `conv1` … `conv4` | `encoder.blocks.0` … `3.conv` |
/// | `lstm_cell.{weight,bias}_ih` | `lstm.features.{weight,bias}` |
/// | `lstm_cell.{weight,bias}_hh` | `lstm.hidden.{weight,bias}` |
/// | `final_conv` | `decoder` |
pub fn upstream_to_bunsen_path(name: &str) -> Option<String> {
    let (module, param) = name.rsplit_once('.')?;
    let path = match module {
        "stft_conv" => "stft".to_string(),
        "final_conv" => "decoder".to_string(),
        "lstm_cell" => {
            let (param, side) = param.rsplit_once('_')?;
            let linear = match side {
                "ih" => "features",
                "hh" => "hidden",
                _ => return None,
            };
            return Some(format!("lstm.{linear}.{param}"));
        }
        conv => {
            let n: usize = conv.strip_prefix("conv")?.parse().ok()?;
            format!("encoder.blocks.{}.conv", n.checked_sub(1)?)
        }
    };
    Some(format!("{path}.{param}"))
}

#[cfg(feature = "store_safetensors")]
impl SileroVad {
    /// Load the 16khz model from tensors named as upstream's
    /// `silero_vad_16k.safetensors` names them ([`upstream_to_bunsen_path`]).
    ///
    /// # Errors
    /// As [`apply_named_tensors`](crate::data::pretrained::apply_named_tensors):
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when a tensor does not fit, or a parameter is missing.
    pub fn load_16khz_from_upstream_tensors(
        tensors: impl IntoIterator<Item = burn::store::burn_pack::Tensor>,
        device: &Device,
    ) -> BunsenResult<Self> {
        let mut vad: Self = SileroVadSignalConfig::standard_16khz().try_init(device)?;
        crate::data::pretrained::apply_named_tensors(&mut vad, tensors, upstream_to_bunsen_path)?;
        Ok(vad)
    }

    /// Load the 16khz model from upstream's `silero_vad_16k.safetensors`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when the file is not a safetensors file; otherwise as
    /// [`load_16khz_from_upstream_tensors`](Self::load_16khz_from_upstream_tensors),
    /// under a frame naming the file.
    pub fn load_16khz_from_safetensors_file(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> BunsenResult<Self> {
        use burn::store::{
            ModuleStore,
            SafetensorsStore,
        };

        let path = path.as_ref();
        let read = || -> BunsenResult<Self> {
            std::fs::metadata(path).map_err(sys_at("open", path))?;
            let tensors: Vec<_> = SafetensorsStore::from_file(path)
                .get_all_tensors()
                .map_err(|e| {
                    BunsenError::invalid_resource("the store refused the checkpoint").with_cause(e)
                })?
                .values()
                .cloned()
                .collect();
            Self::load_16khz_from_upstream_tensors(tensors, device)
        };
        read().with_context(|| format!("loading {}", path.display()))
    }
}

impl SileroVadCollection {
    /// A collection of one 16khz model: what upstream's
    /// `silero_vad_16k.safetensors` holds.
    pub fn from_16khz(vad_16: SileroVad) -> Self {
        Self {
            branches: vec![(16000, vad_16)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upstream_to_bunsen_path() {
        for (upstream, bunsen) in [
            ("stft_conv.weight", Some("stft.weight")),
            ("conv1.weight", Some("encoder.blocks.0.conv.weight")),
            ("conv4.bias", Some("encoder.blocks.3.conv.bias")),
            ("lstm_cell.weight_ih", Some("lstm.features.weight")),
            ("lstm_cell.bias_ih", Some("lstm.features.bias")),
            ("lstm_cell.weight_hh", Some("lstm.hidden.weight")),
            ("lstm_cell.bias_hh", Some("lstm.hidden.bias")),
            ("final_conv.bias", Some("decoder.bias")),
            ("conv0.weight", None),
            ("lstm_cell.weight_xx", None),
            ("unknown", None),
        ] {
            assert_eq!(
                upstream_to_bunsen_path(upstream).as_deref(),
                bunsen,
                "{upstream}"
            );
        }
    }

    /// A state dict with upstream's names and shapes loads into every
    /// parameter, the LSTM's linears transposed as `PyTorch`'s are.
    #[cfg(feature = "store_safetensors")]
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
        let vad = SileroVad::load_16khz_from_upstream_tensors(tensors, &cpu_device()).unwrap();

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
        assert_eq!(SileroVadCollection::from_16khz(vad).branches.len(), 1);
    }
}
