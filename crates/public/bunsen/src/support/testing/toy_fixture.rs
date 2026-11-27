//! Toy-model fixtures.

use alloc::{
    collections::BTreeMap,
    format,
    string::ToString,
};
use std::path::Path;

use burn::{
    prelude::*,
    store::{
        ModuleStore,
        SafetensorsStore,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::Tolerance,
};

/// A toy model's fixture, read from a safetensors file: the reference's
/// inputs and outputs (the `io.*` tensors, by the name after `io.`) and the
/// model's weights (the rest, under their checkpoint names), read onto a
/// device.
///
/// `tools/toy_fixture.py` writes them.
pub struct ToyFixture {
    io: BTreeMap<String, PackTensor>,
    weights: Vec<PackTensor>,
    device: Device,
}

impl ToyFixture {
    /// The fixture at `path`, its `io.*` tensors apart from its weights.
    ///
    /// # Panics
    /// When the file does not read.
    pub fn load(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Self {
        let (io, weights): (Vec<_>, Vec<_>) = Self::read(path.as_ref())
            .into_iter()
            .partition(|(name, _)| name.starts_with("io."));
        Self {
            io: io
                .into_iter()
                .map(|(name, t)| (name["io.".len()..].to_string(), t))
                .collect(),
            weights: weights.into_iter().map(|(_, t)| t).collect(),
            device: device.clone(),
        }
    }

    /// As [`load`](Self::load), for a fixture that stores its weights under
    /// `weights_prefix` (`"model."`): the prefix is stripped from each
    /// weight's name.
    ///
    /// # Panics
    /// When the file does not read, or holds a tensor under neither `io.` nor
    /// `weights_prefix`.
    pub fn load_prefixed(
        path: impl AsRef<Path>,
        weights_prefix: &str,
        device: &Device,
    ) -> Self {
        let mut fixture = Self::load(path, device);
        for tensor in &mut fixture.weights {
            let name = tensor.name.strip_prefix(weights_prefix).unwrap_or_else(|| {
                panic!(
                    "the toy fixture's {:?} is under neither \"io.\" nor {weights_prefix:?}",
                    tensor.name
                )
            });
            tensor.name = name.to_string();
        }
        fixture
    }

    /// Answers alone, at `path`: every tensor is an io tensor, under its own
    /// name (a fixture whose weights are a checkpoint of their own).
    ///
    /// # Panics
    /// When the file does not read.
    pub fn load_io(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Self {
        Self {
            io: Self::read(path.as_ref()).into_iter().collect(),
            weights: Vec::new(),
            device: device.clone(),
        }
    }

    fn read(path: &Path) -> BTreeMap<String, PackTensor> {
        SafetensorsStore::from_file(path)
            .get_all_tensors()
            .unwrap_or_else(|e| panic!("read the toy fixture {}: {e}", path.display()))
            .clone()
    }

    /// The device the io tensors are read onto.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The weights, for a loader; leaves none behind.
    pub fn take_weights(&mut self) -> Vec<PackTensor> {
        core::mem::take(&mut self.weights)
    }

    /// Whether the fixture has the io tensor `key`.
    pub fn has(
        &self,
        key: &str,
    ) -> bool {
        self.io.contains_key(key)
    }

    /// The io tensor `key`, on the fixture's device, in its default float.
    ///
    /// # Panics
    /// When there is no such tensor, or it is not of rank `D`.
    pub fn io<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        let tensor = self
            .io
            .get(key)
            .unwrap_or_else(|| panic!("the toy fixture has no io tensor {key:?}"));
        Tensor::from_data(
            bridge::to_data(tensor).expect("read an io tensor"),
            &self.device,
        )
    }

    /// The io tensor `key`, which a fixture stores as floats, as integers.
    ///
    /// # Panics
    /// As [`io`](Self::io).
    pub fn int<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D, Int> {
        self.io::<D>(key).int()
    }

    /// `actual` is the io tensor `key`, as [`assert_io_close`]; a failure is
    /// labelled `"{what} {key}"`, or `key` alone when `what` is empty.
    ///
    /// # Panics
    /// When they differ, or as [`io`](Self::io).
    #[track_caller]
    pub fn assert_io<const D: usize>(
        &self,
        what: &str,
        key: &str,
        actual: Tensor<D>,
    ) {
        let label = match what {
            "" => key.to_string(),
            what => format!("{what} {key}"),
        };
        assert_io_close(&label, actual, self.io(key));
    }

    /// The `input_ids` io tensor, which a fixture stores as floats, as ids.
    pub fn ids(&self) -> Tensor<2, Int> {
        self.int("input_ids")
    }
}

/// `actual` is the fixture's `expected`: the same shape, and every value
/// within the toy checks' tolerance, relative `1e-3` and absolute `1e-4` (a
/// float64 reference, stored as float32, against a run on a test device).
///
/// # Panics
/// When they differ; the message names `what`.
pub fn assert_io_close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    assert_io_close_within(what, actual, expected, Tolerance::rel_abs(1e-3, 1e-4));
}

/// [`assert_io_close`] within `tolerance`: for a toy whose reference is
/// computed partly in float32 (a `transformers` model that widens some steps
/// to float32 only). Prints the largest absolute difference, the measure a
/// tolerance is set from.
///
/// # Returns
/// The largest absolute difference.
///
/// # Panics
/// When they differ; the message names `what`.
pub fn assert_io_close_within<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
    tolerance: Tolerance<f32>,
) -> f32 {
    assert_eq!(actual.dims(), expected.dims(), "{what}: shape");
    let (actual, expected) = (actual.to_data_as::<f32>(), expected.to_data_as::<f32>());
    let gap = actual
        .iter::<f32>()
        .zip(expected.iter::<f32>())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("{what}: off by at most {gap:.2e}");
    actual.assert_approx_eq::<f32>(&expected, tolerance);
    gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::{
        cpu_device,
        testdata,
    };

    /// The prefix is stripped from every weight, and the io tensors read
    /// as floats and as integers.
    #[test]
    fn test_load_prefixed() {
        let mut fixture = ToyFixture::load_prefixed(
            testdata!("parakeet/toy.safetensors"),
            "model.",
            &cpu_device(),
        );
        let weights = fixture.take_weights();
        assert!(!weights.is_empty());
        assert!(weights.iter().all(|w| !w.name.starts_with("model.")));
        assert!(fixture.take_weights().is_empty());
        let x = fixture.io::<3>("subsampling");
        fixture.assert_io("the toy", "subsampling", x.clone());
        fixture.assert_io("", "subsampling", x);
    }

    /// A weight under another prefix is refused, by name.
    #[test]
    #[should_panic(expected = "is under neither \"io.\" nor \"weights.\"")]
    fn test_load_prefixed_refuses_a_stray_tensor() {
        ToyFixture::load_prefixed(
            testdata!("parakeet/toy.safetensors"),
            "weights.",
            &cpu_device(),
        );
    }
}
