//! # `DeiT` prefabs

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::classification::{
        deit::DeiTContractConfig,
        vit::ViTContractConfig,
    },
};

impl DeiTContractConfig {
    /// A plain `DeiT` of 1000 classes: timm's default `ViT` of these sizes.
    fn deit(
        image_size: usize,
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
    ) -> Self {
        Self::new(ViTContractConfig::new(
            image_size, patch_size, embed_dim, depth, num_heads, 1000,
        ))
    }

    /// A `DeiT` III of 1000 classes: the position embedding over the patches
    /// alone (`no_embed_class`), and layer scale.
    fn deit3(
        image_size: usize,
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
    ) -> Self {
        let deit = Self::deit(image_size, patch_size, embed_dim, depth, num_heads);
        Self {
            vit: deit.vit.with_no_embed_class(true).with_layer_scale(true),
            ..deit
        }
    }

    /// The `deit_tiny_patch16_224` prefab: `DeiT-Ti` at 224 pixels in 16 pixel
    /// patches: 12 layers, 192 wide, 3 heads.
    pub fn deit_tiny_patch16_224() -> Self {
        Self::deit(224, 16, 192, 12, 3)
    }

    /// The `deit_small_patch16_224` prefab: `DeiT-S` at 224 pixels in 16 pixel
    /// patches: 12 layers, 384 wide, 6 heads.
    pub fn deit_small_patch16_224() -> Self {
        Self::deit(224, 16, 384, 12, 6)
    }

    /// The `deit_base_patch16_224` prefab: `DeiT-B` at 224 pixels in 16 pixel
    /// patches: 12 layers, 768 wide, 12 heads.
    pub fn deit_base_patch16_224() -> Self {
        Self::deit(224, 16, 768, 12, 12)
    }

    /// The `deit_base_patch16_384` prefab: `DeiT-B` at 384 pixels in 16 pixel
    /// patches: 12 layers, 768 wide, 12 heads.
    pub fn deit_base_patch16_384() -> Self {
        Self::deit(384, 16, 768, 12, 12)
    }

    /// The `deit_tiny_distilled_patch16_224` prefab: distilled `DeiT-Ti` at 224
    /// pixels in 16 pixel patches: 12 layers, 192 wide, 3 heads; a distillation
    /// token and its classifier.
    pub fn deit_tiny_distilled_patch16_224() -> Self {
        Self::deit(224, 16, 192, 12, 3).with_distilled(true)
    }

    /// The `deit_small_distilled_patch16_224` prefab: distilled `DeiT-S` at 224
    /// pixels in 16 pixel patches: 12 layers, 384 wide, 6 heads; a distillation
    /// token and its classifier.
    pub fn deit_small_distilled_patch16_224() -> Self {
        Self::deit(224, 16, 384, 12, 6).with_distilled(true)
    }

    /// The `deit_base_distilled_patch16_224` prefab: distilled `DeiT-B` at 224
    /// pixels in 16 pixel patches: 12 layers, 768 wide, 12 heads; a
    /// distillation token and its classifier.
    pub fn deit_base_distilled_patch16_224() -> Self {
        Self::deit(224, 16, 768, 12, 12).with_distilled(true)
    }

    /// The `deit_base_distilled_patch16_384` prefab: distilled `DeiT-B` at 384
    /// pixels in 16 pixel patches: 12 layers, 768 wide, 12 heads; a
    /// distillation token and its classifier.
    pub fn deit_base_distilled_patch16_384() -> Self {
        Self::deit(384, 16, 768, 12, 12).with_distilled(true)
    }

    /// The `deit3_small_patch16_224` prefab: `DeiT` III-S at 224 pixels in 16
    /// pixel patches: 12 layers, 384 wide, 6 heads; the position embedding over
    /// the patches alone, and layer scale.
    pub fn deit3_small_patch16_224() -> Self {
        Self::deit3(224, 16, 384, 12, 6)
    }

    /// The `deit3_small_patch16_384` prefab: `DeiT` III-S at 384 pixels in 16
    /// pixel patches: 12 layers, 384 wide, 6 heads; the position embedding over
    /// the patches alone, and layer scale.
    pub fn deit3_small_patch16_384() -> Self {
        Self::deit3(384, 16, 384, 12, 6)
    }

    /// The `deit3_medium_patch16_224` prefab: `DeiT` III-M at 224 pixels in 16
    /// pixel patches: 12 layers, 512 wide, 8 heads; the position embedding over
    /// the patches alone, and layer scale.
    pub fn deit3_medium_patch16_224() -> Self {
        Self::deit3(224, 16, 512, 12, 8)
    }

    /// The `deit3_base_patch16_224` prefab: `DeiT` III-B at 224 pixels in 16
    /// pixel patches: 12 layers, 768 wide, 12 heads; the position embedding
    /// over the patches alone, and layer scale.
    pub fn deit3_base_patch16_224() -> Self {
        Self::deit3(224, 16, 768, 12, 12)
    }

    /// The `deit3_base_patch16_384` prefab: `DeiT` III-B at 384 pixels in 16
    /// pixel patches: 12 layers, 768 wide, 12 heads; the position embedding
    /// over the patches alone, and layer scale.
    pub fn deit3_base_patch16_384() -> Self {
        Self::deit3(384, 16, 768, 12, 12)
    }

    /// The `deit3_large_patch16_224` prefab: `DeiT` III-L at 224 pixels in 16
    /// pixel patches: 24 layers, 1024 wide, 16 heads; the position embedding
    /// over the patches alone, and layer scale.
    pub fn deit3_large_patch16_224() -> Self {
        Self::deit3(224, 16, 1024, 24, 16)
    }

    /// The `deit3_large_patch16_384` prefab: `DeiT` III-L at 384 pixels in 16
    /// pixel patches: 24 layers, 1024 wide, 16 heads; the position embedding
    /// over the patches alone, and layer scale.
    pub fn deit3_large_patch16_384() -> Self {
        Self::deit3(384, 16, 1024, 24, 16)
    }

    /// The `deit3_huge_patch14_224` prefab: `DeiT` III-H at 224 pixels in 14
    /// pixel patches: 32 layers, 1280 wide, 16 heads; the position embedding
    /// over the patches alone, and layer scale.
    pub fn deit3_huge_patch14_224() -> Self {
        Self::deit3(224, 14, 1280, 32, 16)
    }
}

/// The `DeiT` geometries by name, timm's architecture names: 16
/// [`DeiTContractConfig`]s, each with 1000 classes: the plain and distilled
/// `DeiT`s (`deit_tiny_patch16_224` to `deit_base_distilled_patch16_384`)
/// and `DeiT` III (`deit3_small_patch16_224` to `deit3_huge_patch14_224`).
///
/// A prefab has no weights, and its size is the architecture's: a timm row
/// takes the classes and the size from its `config.json`.
impl Prefabs for DeiTContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("deit_tiny_patch16_224", Self::deit_tiny_patch16_224),
        ("deit_small_patch16_224", Self::deit_small_patch16_224),
        ("deit_base_patch16_224", Self::deit_base_patch16_224),
        ("deit_base_patch16_384", Self::deit_base_patch16_384),
        (
            "deit_tiny_distilled_patch16_224",
            Self::deit_tiny_distilled_patch16_224,
        ),
        (
            "deit_small_distilled_patch16_224",
            Self::deit_small_distilled_patch16_224,
        ),
        (
            "deit_base_distilled_patch16_224",
            Self::deit_base_distilled_patch16_224,
        ),
        (
            "deit_base_distilled_patch16_384",
            Self::deit_base_distilled_patch16_384,
        ),
        ("deit3_small_patch16_224", Self::deit3_small_patch16_224),
        ("deit3_small_patch16_384", Self::deit3_small_patch16_384),
        ("deit3_medium_patch16_224", Self::deit3_medium_patch16_224),
        ("deit3_base_patch16_224", Self::deit3_base_patch16_224),
        ("deit3_base_patch16_384", Self::deit3_base_patch16_384),
        ("deit3_large_patch16_224", Self::deit3_large_patch16_224),
        ("deit3_large_patch16_384", Self::deit3_large_patch16_384),
        ("deit3_huge_patch14_224", Self::deit3_huge_patch14_224),
    ];
    const PREFAB_TABLE: &'static str = "deit";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burner::module::ToStructureConfig;

    /// Every prefab lowers, at its name's size; the distilled ones are named
    /// so.
    #[test]
    fn test_prefabs_lower() {
        for name in DeiTContractConfig::prefab_names() {
            let config = DeiTContractConfig::expect_prefab(name);
            config
                .try_to_structure()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let size = name.rsplit('_').next().unwrap();
            assert_eq!(config.vit.image_size.to_string(), size, "{name}");
            assert_eq!(config.distilled, name.contains("distilled"), "{name}");
        }
        assert_eq!(DeiTContractConfig::prefab_names().len(), 16);
    }
}
