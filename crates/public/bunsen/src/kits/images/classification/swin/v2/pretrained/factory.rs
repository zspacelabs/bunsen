//! # The Swin V2 pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::swin::v2::pretrained::{
        SwinV2ConstructConfig,
        default_swin_v2_providers,
    },
};

/// Swin V2's factory: [`default_swin_v2_providers`] behind
/// [`SwinV2ConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_swin_v2_factory() -> BunsenResult<BuildFactory<SwinV2ConstructConfig>> {
    BuildFactory::new().with_providers(default_swin_v2_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::swin::v2::pretrained::SWIN_V2_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_swin_v2_factory().unwrap();
        assert_eq!(factory.kit(), SWIN_V2_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
