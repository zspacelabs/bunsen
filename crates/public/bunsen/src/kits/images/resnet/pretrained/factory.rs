//! # The `ResNet` pretrained factory

use crate::{
    data::pretrained::PretrainedFactory,
    errors::BunsenResult,
    kits::images::resnet::pretrained::{
        ResNetConstruct,
        default_resnet_providers,
    },
};

/// `ResNet`'s factory: [`default_resnet_providers`] behind
/// [`ResNetConstruct`].
///
/// # Errors
/// [`BunsenError::Invalid`](crate::errors::BunsenError::Invalid) if two of
/// the defaults share a name, which the tests pin they do not.
pub fn default_resnet_factory() -> BunsenResult<PretrainedFactory<ResNetConstruct>> {
    PretrainedFactory::new().with_providers(default_resnet_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::images::resnet::pretrained::RESNET_KIT;

    /// The default factory is the well-known table alone, serving the
    /// kit; one prefab has many rows across both groups.
    #[test]
    fn test_the_defaults_register() {
        let factory = default_resnet_factory().unwrap();
        assert_eq!(factory.kit(), RESNET_KIT);
        assert_eq!(factory.providers().len(), 1);
        assert_eq!(factory.ids().len(), 14);
        let (provider, row) = factory.lookup("resnet18_a1").unwrap();
        assert_eq!(provider, "well-known");
        assert_eq!(row.name, "timm/resnet18_a1");
        let for_34: Vec<String> = factory
            .for_prefab("resnet34")
            .iter()
            .map(|(p, row)| format!("{p}:{}", row.name))
            .collect();
        assert_eq!(
            for_34,
            [
                "well-known:torchvision/resnet34",
                "well-known:timm/resnet34_a1",
                "well-known:timm/resnet34_a2",
                "well-known:timm/resnet34_a3",
                "well-known:timm/resnet34",
            ]
        );
        assert!(
            PretrainedFactory::<ResNetConstruct>::new()
                .with_providers(default_resnet_providers())
                .unwrap()
                .with_providers(default_resnet_providers())
                .is_err()
        );
    }
}
