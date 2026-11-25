//! # `ViT` pretrained providers

use alloc::{
    sync::Arc,
    vec,
};

use crate::data::builds::{
    BuildProvider,
    HfProvider,
};

/// The `ViT` kit's name: its rows' `kit` tag.
pub const VIT_KIT: &str = "vit";

/// The key of the checkpoint in a `ViT` resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// What the vit kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const VIT_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const VIT_CATALOG_CONTRACT: &str = "vit-classifier";

/// `ViT`'s compiled-in providers, in search order: the catalog provider, then
/// `hf:org/repo`, a timm hub repo of a `ViT` the kit builds
/// (`hf:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_vit_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            VIT_CATALOG_CONTRACT,
        )),
        Arc::new(HfProvider::new().with_checkpoint_key(CHECKPOINT)),
    ]
}
