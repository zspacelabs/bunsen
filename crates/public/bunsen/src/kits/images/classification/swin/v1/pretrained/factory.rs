//! # The Swin V1 pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::swin::v1::pretrained::{
        SwinV1ConstructConfig,
        default_swin_v1_providers,
    },
};

/// Swin V1's factory: [`default_swin_v1_providers`] behind
/// [`SwinV1ConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_swin_v1_factory() -> BunsenResult<BuildFactory<SwinV1ConstructConfig>> {
    BuildFactory::new().with_providers(default_swin_v1_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::swin::v1::pretrained::SWIN_V1_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_swin_v1_factory().unwrap();
        assert_eq!(factory.kit(), SWIN_V1_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
