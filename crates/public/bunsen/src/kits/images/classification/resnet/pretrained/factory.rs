//! # The `ResNet` pretrained factory

use crate::{
    data::builds::BuildFactory,
    errors::BunsenResult,
    kits::images::classification::resnet::pretrained::{
        ResNetConstructConfig,
        default_resnet_providers,
    },
};

/// `ResNet`'s factory: [`default_resnet_providers`] behind
/// [`ResNetConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::register`], if two of the defaults share a name,
/// which the tests pin they do not.
pub fn default_resnet_factory() -> BunsenResult<BuildFactory<ResNetConstructConfig>> {
    BuildFactory::new().with_providers(default_resnet_providers())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{
        data::pretrained::Prefabs,
        kits::images::classification::resnet::{
            ResNetContractConfig,
            pretrained::RESNET_KIT,
        },
    };

    /// The default factory is the catalog provider, then `hf:`, serving the
    /// kit; a torchvision row states its prefab's geometry, and a timm row
    /// takes its geometry from its `config.json`.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_resnet_factory().unwrap();
        assert_eq!(factory.kit(), RESNET_KIT);
        assert_eq!(factory.providers().len(), 2);
        assert_eq!(factory.ids().len(), 145);
        let (provider, row) = factory.lookup("resnet18_a1").unwrap();
        assert_eq!(provider, "catalog");
        // A legacy name is an alias of the hub repo with its weights.
        assert_eq!(row.name, "timm/resnet18.a1_in1k");
        // The rows that state resnet34's geometry.
        let resnet34 =
            serde_json::to_value(ResNetContractConfig::expect_prefab("resnet34")).unwrap();
        let for_34: Vec<String> = factory
            .list()
            .iter()
            .filter(|(_, row)| {
                row.args
                    .get(crate::data::builds::GEOMETRY)
                    .map(|a| &a.value)
                    == Some(&resnet34)
            })
            .map(|(p, row)| format!("{p}:{}", row.name))
            .collect();
        assert_eq!(for_34, ["catalog:torchvision/resnet34"]);
        assert!(
            BuildFactory::<ResNetConstructConfig>::new()
                .with_providers(default_resnet_providers())
                .unwrap()
                .with_providers(default_resnet_providers())
                .is_err()
        );
    }
}
