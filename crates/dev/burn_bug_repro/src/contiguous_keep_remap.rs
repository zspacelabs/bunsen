//! # Keeping a list's indices stops working once its name is remapped.
//!
//! Affects `burn-store`'s `map_indices_contiguous_except`, on `PytorchStore`
//! and `SafetensorsStore`.
//!
//! ## Expected semantics
//!
//! `map_indices_contiguous_except(pattern)` keeps the indices of one list in
//! a file while every other list is renumbered. Its documentation describes
//! the pattern against the file's names (`^model_g\.flow\.flows$`, "`flows`
//! for `flows.2.weight`"), so a pattern naming a list in the file should keep
//! that list whatever `with_key_remapping` renames it to.
//!
//! ## The defect
//!
//! The store remaps the names first, and then matches the pattern against
//! the remapped prefix. A pattern written against the file's name stops
//! matching as soon as a remapping renames that list, and the list is
//! renumbered after all, silently. Only a pattern written against the
//! remapped name (the module's) keeps it.
//!
//! ## Why it matters
//!
//! A loader's remappings and its kept lists are usually written together,
//! both from the checkpoint's names; adding a remapping later silently
//! un-keeps a list, and its entries load one slot early. bunsen's loaders
//! don't use the keep pattern; bunsen's own key maps
//! (`bunsen::burner::module::Rekey`) never renumber.
//!
//! ## Recovery
//!
//! Match the pattern against the file's names, before remapping, as the
//! documentation describes; or document that it names the remapped list.
//! Until then, write the pattern against the remapped name.

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
            ModuleSnapshot,
            SafetensorsStore,
        },
    };

    #[derive(Module, Debug)]
    struct Checkpoint {
        flows: Vec<Linear>,
    }

    #[derive(Module, Debug)]
    struct Model {
        blocks: Vec<Linear>,
    }

    /// Three one-weight layers, layer `i`'s weight `i + 1`.
    fn layers(device: &Device) -> Vec<Linear> {
        (0..3)
            .map(|i| {
                let mut l = LinearConfig::new(1, 1).with_bias(false).init(device);
                l.weight = l.weight.map(|w| w.zeros_like().add_scalar(i as f32 + 1.0));
                l
            })
            .collect()
    }

    /// A file holding `flows.0` and `flows.2` (the second list entry
    /// parameter-free, say), loaded into `blocks` with `flows.` remapped to
    /// `blocks.`, contiguous index mapping on, and `keep` kept; each block's
    /// weight, `0.0` where nothing loaded.
    fn load_keeping(keep: &str) -> Vec<f32> {
        let device = cpu_device();
        let mut save = SafetensorsStore::from_bytes(None).with_regex(r"^flows\.[02]\.");
        Checkpoint {
            flows: layers(&device),
        }
        .save_into(&mut save)
        .unwrap();
        let bytes = save.get_bytes().unwrap();

        let mut model = Model {
            blocks: layers(&device)
                .into_iter()
                .map(|mut l| {
                    l.weight = l.weight.map(|w| w.zeros_like());
                    l
                })
                .collect(),
        };
        let mut load = SafetensorsStore::from_bytes(Some(bytes))
            .with_key_remapping(r"^flows\.", "blocks.")
            .map_indices_contiguous(true)
            .map_indices_contiguous_except(keep)
            .allow_partial(true);
        model.load_from(&mut load).unwrap();
        model
            .blocks
            .iter()
            .map(|l| l.weight.val().into_scalar())
            .collect()
    }

    /// The reproduction: a pattern naming the file's list keeps it.
    #[test]
    #[ignore = "burn defect: the keep pattern is matched after remapping; see the module docs"]
    fn test_the_files_list_name_keeps_it() {
        assert_eq!(load_keeping(r"^flows$"), [1.0, 0.0, 3.0]);
    }

    /// The behaviour pin: the file's name keeps nothing, and the list is
    /// renumbered; the remapped name keeps it.
    #[test]
    fn test_pin_only_the_remapped_name_keeps_it() {
        assert_eq!(
            load_keeping(r"^flows$"),
            [1.0, 3.0, 0.0],
            "a pattern naming the file's list now keeps it: the keep-after-remap defect is \
             fixed. Un-ignore `test_the_files_list_name_keeps_it`."
        );
        assert_eq!(load_keeping(r"^blocks$"), [1.0, 0.0, 3.0]);
    }
}
