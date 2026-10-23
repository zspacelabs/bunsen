//! # Contiguous index mapping moves the entries after a missing one.
//!
//! Affects `burn-store`'s `map_indices_contiguous`, which `PytorchStore` turns
//! on by default and `SafetensorsStore` offers.
//!
//! ## Expected semantics
//!
//! The mapping exists for a `PyTorch` `nn.Sequential` whose parameter-free
//! entries leave gaps in its indices (`Conv2d` at 0, 2, 4, `ReLU` at 1, 3, 5),
//! loaded into a Burn list without those entries. A checkpoint whose list
//! already has the module's indices, but lacks one entry, should load every
//! entry it has at its own index and report the missing one as missing:
//! `layers.{0,1,3}` into a four-entry list is `layers.2` missing.
//!
//! ## The defect
//!
//! The renumbering is computed from the file's names alone, never from the
//! module's. It cannot tell a structural gap from a missing entry, so it
//! closes this one too: `layers.3` loads into `layers.2`, and `layers.3` is
//! the one reported missing. With `allow_partial(true)` the load succeeds
//! with a layer in the wrong place; without it, the error names the wrong
//! layer. Any list in the file is renumbered, including one a key remapping
//! introduced, since the mapping runs after the remapping.
//!
//! ## Why it matters
//!
//! A truncated or partly converted checkpoint loads silently misaligned, and
//! the mapping is on by default for `PyTorch` files. bunsen's loaders turn it
//! off where they can (Whisper's `.pt` loader) and map indices themselves
//! (`bunsen::burner::module::Rekey`). `ResNet`'s loader relies on it the
//! other way: its remapping writes torchvision's
//! 1-based `layer1` … `layer4` as `layers.1` … `layers.4`, and the mapping is
//! what lands them at 0 … 3.
//!
//! ## Recovery
//!
//! Renumber against the module's own paths: an index the module has is kept,
//! and only gaps the module does not have are closed. Until then, turn the
//! mapping off for a file whose indices already match the module.

#[cfg(test)]
mod tests {
    use bunsen::support::testing::cpu_device;
    use burn::{
        module::Module,
        nn::{
            Linear,
            LinearConfig,
        },
        prelude::*,
        store::{
            ApplyResult,
            ModuleSnapshot,
            SafetensorsStore,
        },
    };

    #[derive(Module, Debug)]
    struct Stack {
        layers: Vec<Linear>,
    }

    /// Four one-weight layers, layer `i`'s weight `i + 1`.
    fn stack(device: &Device) -> Stack {
        Stack {
            layers: (0..4)
                .map(|i| {
                    let mut l = LinearConfig::new(1, 1).with_bias(false).init(device);
                    l.weight = l.weight.map(|w| w.zeros_like().add_scalar(i as f32 + 1.0));
                    l
                })
                .collect(),
        }
    }

    /// `stack`'s weights in a safetensors file that lacks `layers.2`, loaded
    /// into a fresh stack with contiguous index mapping on; and each fresh
    /// layer's weight, `0.0` where nothing loaded.
    fn load_without_layer_2() -> (ApplyResult, Vec<f32>) {
        let device = cpu_device();
        let mut save = SafetensorsStore::from_bytes(None).with_regex(r"^layers\.[013]\.");
        stack(&device).save_into(&mut save).unwrap();
        let bytes = save.get_bytes().unwrap();

        let mut fresh = stack(&device);
        for l in &mut fresh.layers {
            l.weight = l.weight.clone().map(|w| w.zeros_like());
        }
        let mut load = SafetensorsStore::from_bytes(Some(bytes))
            .map_indices_contiguous(true)
            .allow_partial(true);
        let result = fresh.load_from(&mut load).unwrap();
        let weights = fresh
            .layers
            .iter()
            .map(|l| l.weight.val().into_scalar())
            .collect();
        (result, weights)
    }

    /// The reproduction: layers 0, 1 and 3 load where they were, and layer 2
    /// is missing.
    #[test]
    #[ignore = "burn defect: contiguous index mapping closes a missing entry's gap; see the module docs"]
    fn test_a_missing_entry_stays_missing() {
        let (result, weights) = load_without_layer_2();
        assert_eq!(weights, [1.0, 2.0, 0.0, 4.0]);
        assert!(
            result.missing.iter().any(|(p, _)| p == "layers.2.weight"),
            "{:?}",
            result.missing
        );
    }

    /// The behaviour pin: layer 3 loads into layer 2, and layer 3 is the one
    /// missing.
    #[test]
    fn test_pin_the_gap_is_closed() {
        let (result, weights) = load_without_layer_2();
        assert_eq!(
            weights,
            [1.0, 2.0, 4.0, 0.0],
            "a missing entry no longer shifts the ones after it: the contiguous-gap defect \
             is fixed. Un-ignore `test_a_missing_entry_stays_missing`; `ResNet`'s loader, which \
             relies on the renumbering for torchvision's 1-based `layer1`..`layer4`, may need \
             its own index map."
        );
        assert!(
            result.missing.iter().any(|(p, _)| p == "layers.3.weight"),
            "{:?}",
            result.missing
        );
    }
}
