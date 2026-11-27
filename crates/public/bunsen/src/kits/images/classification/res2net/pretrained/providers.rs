//! # `Res2Net` pretrained providers

use alloc::sync::Arc;

use crate::data::builds::BuildProvider;

/// The `Res2Net` kit's name: its rows' `kit` tag.
pub const RES2NET_KIT: &str = "res2net";

/// The key of the checkpoint in a `Res2Net` resource set.
pub use crate::data::builds::CHECKPOINT;

/// What the res2net kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const RES2NET_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const RES2NET_CATALOG_CONTRACT: &str = "res2net-classifier";

/// `Res2Net`'s compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a timm hub repo of a `Res2Net` the kit builds
/// (`hf:timm/res2net50_26w_4s.in1k`): its
/// `model.safetensors` and its `config.json`, whose architecture names the
/// geometry.
pub fn default_res2net_providers() -> Vec<Arc<dyn BuildProvider>> {
    crate::kits::images::support::pretrained::timm_providers(RES2NET_CATALOG_CONTRACT)
}

/// `Res2Net`'s factory: [`default_res2net_providers`] behind
/// [`Res2NetConstructConfig`](super::Res2NetConstructConfig).
///
/// # Errors
/// As [`BuildFactory::register`](crate::data::builds::BuildFactory::register), if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_res2net_factory()
-> crate::errors::BunsenResult<crate::data::builds::BuildFactory<super::Res2NetConstructConfig>> {
    crate::data::builds::BuildFactory::new().with_providers(default_res2net_providers())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::kits::images::classification::res2net::Res2NetContractConfig;

    /// Every row is a timm hub repo of a prefab's architecture, pinned: its
    /// checkpoint, its `config.json`, and `ImageNet`-1k's labels.
    #[test]
    fn test_every_row_is_a_pinned_timm_repo() {
        use crate::kits::images::support::pretrained::testing::{
            LabelRule,
            assert_timm_rows,
        };
        assert_timm_rows::<Res2NetContractConfig>(
            &super::super::default_res2net_factory().unwrap(),
            9,
            LabelRule::Counts(&[("imagenet/synsets-1k", 9)]),
        );
    }

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        crate::kits::images::support::pretrained::testing::assert_timm_defaults_register(
            &default_res2net_factory().unwrap(),
        );
    }
}
