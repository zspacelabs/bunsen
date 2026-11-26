//! # The `Res2Net` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::res2net::pretrained::{
        Res2NetConstructConfig,
        default_res2net_providers,
    },
};

/// `Res2Net`'s factory: [`default_res2net_providers`] behind
/// [`Res2NetConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_res2net_factory() -> BunsenResult<BuildFactory<Res2NetConstructConfig>> {
    BuildFactory::new().with_providers(default_res2net_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::classification::res2net::pretrained::RES2NET_KIT;

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_res2net_factory().unwrap();
        assert_eq!(factory.kit(), RES2NET_KIT);
        assert_eq!(factory.providers().len(), 2);
    }
}
