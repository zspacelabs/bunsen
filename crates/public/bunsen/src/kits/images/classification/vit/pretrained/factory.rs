//! # The `ViT` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::vit::pretrained::{
        ViTConstructConfig,
        default_vit_providers,
    },
};

/// `ViT`'s factory: [`default_vit_providers`] behind [`ViTConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_vit_factory() -> BunsenResult<BuildFactory<ViTConstructConfig>> {
    BuildFactory::new().with_providers(default_vit_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::vit::pretrained::VIT_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_vit_factory().unwrap();
        assert_eq!(factory.kit(), VIT_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
