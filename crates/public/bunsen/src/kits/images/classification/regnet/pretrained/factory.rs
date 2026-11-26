//! # The `RegNet` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::regnet::pretrained::{
        RegNetConstructConfig,
        default_regnet_providers,
    },
};

/// `RegNet`'s factory: [`default_regnet_providers`] behind
/// [`RegNetConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_regnet_factory() -> BunsenResult<BuildFactory<RegNetConstructConfig>> {
    BuildFactory::new().with_providers(default_regnet_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::regnet::pretrained::REGNET_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_regnet_factory().unwrap();
        assert_eq!(factory.kit(), REGNET_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
