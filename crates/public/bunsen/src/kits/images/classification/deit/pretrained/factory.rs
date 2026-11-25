//! # The `DeiT` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::deit::pretrained::{
        DeiTConstructConfig,
        default_deit_providers,
    },
};

/// `DeiT`'s factory: [`default_deit_providers`] behind [`DeiTConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_deit_factory() -> BunsenResult<BuildFactory<DeiTConstructConfig>> {
    BuildFactory::new().with_providers(default_deit_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::deit::pretrained::DEIT_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_deit_factory().unwrap();
        assert_eq!(factory.kit(), DEIT_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
