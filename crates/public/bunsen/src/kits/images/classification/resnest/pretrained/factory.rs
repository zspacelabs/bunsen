//! # The `ResNeSt` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::resnest::pretrained::{
        ResNeStConstructConfig,
        default_resnest_providers,
    },
};

/// `ResNeSt`'s factory: [`default_resnest_providers`] behind
/// [`ResNeStConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_resnest_factory() -> BunsenResult<BuildFactory<ResNeStConstructConfig>> {
    BuildFactory::new().with_providers(default_resnest_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::resnest::pretrained::RESNEST_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_resnest_factory().unwrap();
        assert_eq!(factory.kit(), RESNEST_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
