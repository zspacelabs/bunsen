//! # The `SKNet` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::sknet::pretrained::{
        SKNetConstructConfig,
        default_sknet_providers,
    },
};

/// `SKNet`'s factory: [`default_sknet_providers`] behind
/// [`SKNetConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_sknet_factory() -> BunsenResult<BuildFactory<SKNetConstructConfig>> {
    BuildFactory::new().with_providers(default_sknet_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::sknet::pretrained::SKNET_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_sknet_factory().unwrap();
        assert_eq!(factory.kit(), SKNET_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
