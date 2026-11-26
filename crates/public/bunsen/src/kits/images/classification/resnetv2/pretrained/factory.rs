//! # The `ResNetV2` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::resnetv2::pretrained::{
        ResNetV2ConstructConfig,
        default_resnetv2_providers,
    },
};

/// `ResNetV2`'s factory: [`default_resnetv2_providers`] behind
/// [`ResNetV2ConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_resnetv2_factory() -> BunsenResult<BuildFactory<ResNetV2ConstructConfig>> {
    BuildFactory::new().with_providers(default_resnetv2_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::resnetv2::pretrained::RESNETV2_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_resnetv2_factory().unwrap();
        assert_eq!(factory.kit(), RESNETV2_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
