//! The image classification kits' table.

use alloc::{
    boxed::Box,
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec,
    vec::Vec,
};

use burn::tensor::Device;

use crate::{
    data::{
        builds::{
            BuildFactory,
            BuildProvider,
            BuildRef,
        },
        catalog::resolve::ResolvedBuild,
        formats::LoadPolicy,
        pretrained::Prefabs,
        resources::{
            LoadedResourceSet,
            ResourceCache,
        },
    },
    errors::BunsenResult,
    kits::images::{
        classification::{
            deit::pretrained::default_deit_factory,
            driver::ImageClassifier,
            regnet::pretrained::default_regnet_factory,
            res2net::pretrained::default_res2net_factory,
            resnest::pretrained::default_resnest_factory,
            resnet::pretrained::default_resnet_factory,
            resnetv2::pretrained::default_resnetv2_factory,
            sknet::pretrained::default_sknet_factory,
            swin::{
                v1::pretrained::default_swin_v1_factory,
                v2::pretrained::default_swin_v2_factory,
            },
            vit::pretrained::default_vit_factory,
        },
        support::{
            card::ImageModelCard,
            pretrained::{
                TimmConstructConfig,
                TimmImageContract,
            },
        },
    },
};

/// One image classification kit's pretrained pathway, type-erased: what a
/// tool needs to list a kit's checkpoints and load one, whatever its model.
///
/// Every kit's [`BuildFactory`] is one ([`image_kit_factories`] lists them).
pub trait ImageKitFactory {
    /// The kit's name: its rows' `kit` tag.
    fn kit(&self) -> &'static str;

    /// The kit's architectures, by timm's names: its prefabs'.
    fn architectures(&self) -> Vec<&'static str>;

    /// The factory's providers, in search order.
    fn providers(&self) -> Vec<Arc<dyn BuildProvider>>;

    /// The factory's rows, by id, in order.
    ///
    /// # Errors
    /// When a listed id doesn't look up.
    fn rows(&self) -> BunsenResult<Vec<(String, ResolvedBuild)>>;

    /// The provider that names `name`, and its build.
    ///
    /// # Errors
    /// As [`BuildFactory::lookup`].
    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<(String, ResolvedBuild)>;

    /// The build `name` resolves to through the factory.
    ///
    /// # Errors
    /// As [`BuildFactory::resolve`].
    fn resolve(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<BuildRef>;

    /// The checkpoint `name`'s files, fetched into `cache`: its id, and the
    /// loaded set.
    ///
    /// # Errors
    /// As resolving, planning and loading the build.
    fn fetch(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<(String, LoadedResourceSet)>;

    /// The checkpoint `name`, loaded on `device` under `policy`: the model as
    /// any kit's classifier, and its card.
    ///
    /// # Errors
    /// As [`BuildFactory::load_construct`].
    fn load(
        &self,
        name: &str,
        cache: &ResourceCache,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<(Box<dyn ImageClassifier>, ImageModelCard)>;
}

impl<C> ImageKitFactory for BuildFactory<TimmConstructConfig<C>>
where
    C: TimmImageContract + Prefabs,
    C::Model: ImageClassifier + Clone + 'static,
{
    fn kit(&self) -> &'static str {
        C::KIT
    }

    fn architectures(&self) -> Vec<&'static str> {
        C::prefab_names()
    }

    fn providers(&self) -> Vec<Arc<dyn BuildProvider>> {
        BuildFactory::providers(self).to_vec()
    }

    fn rows(&self) -> BunsenResult<Vec<(String, ResolvedBuild)>> {
        self.ids()
            .into_iter()
            .map(|id| Ok((id.clone(), BuildFactory::lookup(self, &id)?.1)))
            .collect()
    }

    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<(String, ResolvedBuild)> {
        BuildFactory::lookup(self, name)
    }

    fn resolve(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<BuildRef> {
        Ok(BuildFactory::resolve(self, name, cache)?.model)
    }

    fn fetch(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<(String, LoadedResourceSet)> {
        let model = BuildFactory::resolve(self, name, cache)?;
        let map = model.plan(cache)?;
        Ok((model.id().to_string(), cache.load(&map)?))
    }

    fn load(
        &self,
        name: &str,
        cache: &ResourceCache,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<(Box<dyn ImageClassifier>, ImageModelCard)> {
        let construct = Arc::unwrap_or_clone(self.load_construct(name, cache, device, policy)?);
        Ok((Box::new(construct.model), construct.card))
    }
}

/// Every image classification kit's factory, in the order a listing shows
/// them: the conv nets, then the transformers. A new kit is one entry here.
///
/// # Errors
/// When a kit's factory doesn't build.
pub fn image_kit_factories() -> BunsenResult<Vec<Box<dyn ImageKitFactory>>> {
    Ok(vec![
        Box::new(default_resnet_factory()?),
        Box::new(default_resnetv2_factory()?),
        Box::new(default_res2net_factory()?),
        Box::new(default_resnest_factory()?),
        Box::new(default_sknet_factory()?),
        Box::new(default_regnet_factory()?),
        Box::new(default_vit_factory()?),
        Box::new(default_deit_factory()?),
        Box::new(default_swin_v1_factory()?),
        Box::new(default_swin_v2_factory()?),
    ])
}

/// The kit a checkpoint name loads through, among `kits`: the one whose
/// catalog names it; else, for a hub repo (`hf:timm/vit_base_patch16_224.x`),
/// the one with a prefab of the repo's architecture; else the first
/// (`ResNet`, in [`image_kit_factories`]' order), whose factory then says
/// what it is missing.
///
/// # Panics
/// When `kits` is empty.
pub fn image_kit_for<'k>(
    kits: &'k [Box<dyn ImageKitFactory>],
    name: &str,
) -> &'k dyn ImageKitFactory {
    if let Some(kit) = kits.iter().find(|kit| {
        kit.lookup(name)
            .is_ok_and(|(provider, _)| provider == "catalog")
    }) {
        return kit.as_ref();
    }
    let repo = name.rsplit('/').next().unwrap_or(name);
    let architecture = repo.split('.').next().unwrap_or(repo);
    kits.iter()
        .find(|kit| kit.architectures().contains(&architecture))
        .unwrap_or_else(|| kits.first().expect("an image kit"))
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A catalog name loads through the kit whose catalog has it; a hub repo
    /// through the kit of its architecture; anything else through `ResNet`.
    #[test]
    fn test_image_kit_for() {
        let kits = image_kit_factories().unwrap();
        for (name, kit) in [
            ("timm/resnet50.a1_in1k", "resnet"),
            ("resnet50", "resnet"),
            ("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit"),
            ("hf:timm/vit_base_patch16_224.mine", "vit"),
            ("timm/deit_base_distilled_patch16_224.fb_in1k", "deit"),
            ("hf:timm/deit3_small_patch16_224.mine", "deit"),
            ("timm/swin_s3_tiny_224.ms_in1k", "swin-v1"),
            ("hf:timm/swin_tiny_patch4_window7_224.mine", "swin-v1"),
            ("timm/swinv2_tiny_window8_256.ms_in1k", "swin-v2"),
            ("hf:timm/swinv2_base_window12_192.mine", "swin-v2"),
            ("timm/regnety_032.ra_in1k", "regnet"),
            ("timm/regnety_320.seer", "regnet"),
            ("hf:timm/regnetx_002.mine", "regnet"),
            ("timm/resnetv2_50x1_bit.goog_in21k_ft_in1k", "resnetv2"),
            ("hf:timm/resnetv2_50.mine", "resnetv2"),
            ("timm/res2net50_26w_4s.in1k", "res2net"),
            ("hf:timm/res2next50.mine", "res2net"),
            ("timm/resnest50d.in1k", "resnest"),
            ("hf:timm/resnest14d.mine", "resnest"),
            ("timm/skresnet18.ra_in1k", "sknet"),
            ("hf:timm/skresnext50_32x4d.mine", "sknet"),
            ("hf:timm/resnet18.mine", "resnet"),
            ("hf:someone/unknown", "resnet"),
        ] {
            assert_eq!(image_kit_for(&kits, name).kit(), kit, "{name}");
        }
    }

    /// Every kit is listed once, by its own name, with rows.
    #[test]
    fn test_every_kit_once() {
        let kits = image_kit_factories().unwrap();
        let names: Vec<&str> = kits.iter().map(|k| k.kit()).collect();
        assert_eq!(
            names,
            [
                "resnet", "resnetv2", "res2net", "resnest", "sknet", "regnet", "vit", "deit",
                "swin-v1", "swin-v2"
            ]
        );
        for kit in &kits {
            assert!(!kit.rows().unwrap().is_empty(), "{}", kit.kit());
        }
    }
}
