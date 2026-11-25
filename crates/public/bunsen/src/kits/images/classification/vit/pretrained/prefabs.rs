//! # `ViT` prefabs

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::classification::vit::ViTContractConfig,
};

impl ViTContractConfig {
    /// A `ViT` of timm's defaults with 1000 classes: `image_size` pixel
    /// images in `patch_size` patches, `depth` layers `embed_dim` wide of
    /// `num_heads` heads.
    fn vit(
        image_size: usize,
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
    ) -> Self {
        Self::new(image_size, patch_size, embed_dim, depth, num_heads, 1000)
    }

    /// The `vit_tiny_patch16_224` prefab: `ViT-Ti/16` at 224 pixels: 12 layers,
    /// 192 wide, 3 heads.
    pub fn vit_tiny_patch16_224() -> Self {
        Self::vit(224, 16, 192, 12, 3)
    }

    /// The `vit_tiny_patch16_384` prefab: `ViT-Ti/16` at 384 pixels: 12 layers,
    /// 192 wide, 3 heads.
    pub fn vit_tiny_patch16_384() -> Self {
        Self::vit(384, 16, 192, 12, 3)
    }

    /// The `vit_small_patch32_224` prefab: `ViT-S/32` at 224 pixels: 12 layers,
    /// 384 wide, 6 heads.
    pub fn vit_small_patch32_224() -> Self {
        Self::vit(224, 32, 384, 12, 6)
    }

    /// The `vit_small_patch32_384` prefab: `ViT-S/32` at 384 pixels: 12 layers,
    /// 384 wide, 6 heads.
    pub fn vit_small_patch32_384() -> Self {
        Self::vit(384, 32, 384, 12, 6)
    }

    /// The `vit_small_patch16_224` prefab: `ViT-S/16` at 224 pixels: 12 layers,
    /// 384 wide, 6 heads.
    pub fn vit_small_patch16_224() -> Self {
        Self::vit(224, 16, 384, 12, 6)
    }

    /// The `vit_small_patch16_384` prefab: `ViT-S/16` at 384 pixels: 12 layers,
    /// 384 wide, 6 heads.
    pub fn vit_small_patch16_384() -> Self {
        Self::vit(384, 16, 384, 12, 6)
    }

    /// The `vit_small_patch8_224` prefab: `ViT-S/8` at 224 pixels: 12 layers,
    /// 384 wide, 6 heads.
    pub fn vit_small_patch8_224() -> Self {
        Self::vit(224, 8, 384, 12, 6)
    }

    /// The `vit_base_patch32_224` prefab: `ViT-B/32` at 224 pixels: 12 layers,
    /// 768 wide, 12 heads.
    pub fn vit_base_patch32_224() -> Self {
        Self::vit(224, 32, 768, 12, 12)
    }

    /// The `vit_base_patch32_384` prefab: `ViT-B/32` at 384 pixels: 12 layers,
    /// 768 wide, 12 heads.
    pub fn vit_base_patch32_384() -> Self {
        Self::vit(384, 32, 768, 12, 12)
    }

    /// The `vit_base_patch16_224` prefab: `ViT-B/16` at 224 pixels: 12 layers,
    /// 768 wide, 12 heads.
    pub fn vit_base_patch16_224() -> Self {
        Self::vit(224, 16, 768, 12, 12)
    }

    /// The `vit_base_patch16_384` prefab: `ViT-B/16` at 384 pixels: 12 layers,
    /// 768 wide, 12 heads.
    pub fn vit_base_patch16_384() -> Self {
        Self::vit(384, 16, 768, 12, 12)
    }

    /// The `vit_base_patch8_224` prefab: `ViT-B/8` at 224 pixels: 12 layers,
    /// 768 wide, 12 heads.
    pub fn vit_base_patch8_224() -> Self {
        Self::vit(224, 8, 768, 12, 12)
    }

    /// The `vit_large_patch32_224` prefab: `ViT-L/32` at 224 pixels: 24 layers,
    /// 1024 wide, 16 heads.
    pub fn vit_large_patch32_224() -> Self {
        Self::vit(224, 32, 1024, 24, 16)
    }

    /// The `vit_large_patch32_384` prefab: `ViT-L/32` at 384 pixels: 24 layers,
    /// 1024 wide, 16 heads.
    pub fn vit_large_patch32_384() -> Self {
        Self::vit(384, 32, 1024, 24, 16)
    }

    /// The `vit_large_patch16_224` prefab: `ViT-L/16` at 224 pixels: 24 layers,
    /// 1024 wide, 16 heads.
    pub fn vit_large_patch16_224() -> Self {
        Self::vit(224, 16, 1024, 24, 16)
    }

    /// The `vit_large_patch16_384` prefab: `ViT-L/16` at 384 pixels: 24 layers,
    /// 1024 wide, 16 heads.
    pub fn vit_large_patch16_384() -> Self {
        Self::vit(384, 16, 1024, 24, 16)
    }

    /// The `vit_huge_patch14_224` prefab: `ViT-H/14` at 224 pixels: 32 layers,
    /// 1280 wide, 16 heads.
    pub fn vit_huge_patch14_224() -> Self {
        Self::vit(224, 14, 1280, 32, 16)
    }

    /// The `vit_base_patch16_224_miil` prefab: `ViT-B/16` at 224 pixels: 12
    /// layers, 768 wide, 12 heads; MIIL's, with no `qkv` bias.
    pub fn vit_base_patch16_224_miil() -> Self {
        Self::vit(224, 16, 768, 12, 12).with_qkv_bias(false)
    }

    /// The `vit_base_patch32_clip_224` prefab: `ViT-B/32` at 224 pixels: 12
    /// layers, 768 wide, 12 heads; CLIP's pre-norm image tower.
    pub fn vit_base_patch32_clip_224() -> Self {
        Self::vit(224, 32, 768, 12, 12).with_clip()
    }

    /// The `vit_base_patch32_clip_384` prefab: `ViT-B/32` at 384 pixels: 12
    /// layers, 768 wide, 12 heads; CLIP's pre-norm image tower.
    pub fn vit_base_patch32_clip_384() -> Self {
        Self::vit(384, 32, 768, 12, 12).with_clip()
    }

    /// The `vit_base_patch16_clip_224` prefab: `ViT-B/16` at 224 pixels: 12
    /// layers, 768 wide, 12 heads; CLIP's pre-norm image tower.
    pub fn vit_base_patch16_clip_224() -> Self {
        Self::vit(224, 16, 768, 12, 12).with_clip()
    }

    /// The `vit_base_patch16_clip_384` prefab: `ViT-B/16` at 384 pixels: 12
    /// layers, 768 wide, 12 heads; CLIP's pre-norm image tower.
    pub fn vit_base_patch16_clip_384() -> Self {
        Self::vit(384, 16, 768, 12, 12).with_clip()
    }

    /// The `vit_base_patch16_plus_clip_240` prefab: `ViT-B/16+` at 240 pixels:
    /// 12 layers, 896 wide, 14 heads; CLIP's pre-norm image tower.
    pub fn vit_base_patch16_plus_clip_240() -> Self {
        Self::vit(240, 16, 896, 12, 14).with_clip()
    }

    /// The `vit_large_patch14_clip_224` prefab: `ViT-L/14` at 224 pixels: 24
    /// layers, 1024 wide, 16 heads; CLIP's pre-norm image tower.
    pub fn vit_large_patch14_clip_224() -> Self {
        Self::vit(224, 14, 1024, 24, 16).with_clip()
    }

    /// The `vit_large_patch14_clip_336` prefab: `ViT-L/14` at 336 pixels: 24
    /// layers, 1024 wide, 16 heads; CLIP's pre-norm image tower.
    pub fn vit_large_patch14_clip_336() -> Self {
        Self::vit(336, 14, 1024, 24, 16).with_clip()
    }
}

/// The `ViT` geometries by name, timm's architecture names: 25
/// [`ViTContractConfig`]s, each with 1000 classes, every `ViT` architecture
/// with weights from a paper before Swin V2: `vit_tiny_patch16_224` to
/// `vit_huge_patch14_224`, the MIIL `vit_base_patch16_224_miil`, and the
/// CLIP image towers (`vit_base_patch32_clip_224`, …,
/// `vit_large_patch14_clip_336`).
///
/// A prefab has no weights, and its size is the architecture's: a timm row
/// takes the classes and the size from its `config.json`.
impl Prefabs for ViTContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("vit_tiny_patch16_224", Self::vit_tiny_patch16_224),
        ("vit_tiny_patch16_384", Self::vit_tiny_patch16_384),
        ("vit_small_patch32_224", Self::vit_small_patch32_224),
        ("vit_small_patch32_384", Self::vit_small_patch32_384),
        ("vit_small_patch16_224", Self::vit_small_patch16_224),
        ("vit_small_patch16_384", Self::vit_small_patch16_384),
        ("vit_small_patch8_224", Self::vit_small_patch8_224),
        ("vit_base_patch32_224", Self::vit_base_patch32_224),
        ("vit_base_patch32_384", Self::vit_base_patch32_384),
        ("vit_base_patch16_224", Self::vit_base_patch16_224),
        ("vit_base_patch16_384", Self::vit_base_patch16_384),
        ("vit_base_patch8_224", Self::vit_base_patch8_224),
        ("vit_large_patch32_224", Self::vit_large_patch32_224),
        ("vit_large_patch32_384", Self::vit_large_patch32_384),
        ("vit_large_patch16_224", Self::vit_large_patch16_224),
        ("vit_large_patch16_384", Self::vit_large_patch16_384),
        ("vit_huge_patch14_224", Self::vit_huge_patch14_224),
        ("vit_base_patch16_224_miil", Self::vit_base_patch16_224_miil),
        ("vit_base_patch32_clip_224", Self::vit_base_patch32_clip_224),
        ("vit_base_patch32_clip_384", Self::vit_base_patch32_clip_384),
        ("vit_base_patch16_clip_224", Self::vit_base_patch16_clip_224),
        ("vit_base_patch16_clip_384", Self::vit_base_patch16_clip_384),
        (
            "vit_base_patch16_plus_clip_240",
            Self::vit_base_patch16_plus_clip_240,
        ),
        (
            "vit_large_patch14_clip_224",
            Self::vit_large_patch14_clip_224,
        ),
        (
            "vit_large_patch14_clip_336",
            Self::vit_large_patch14_clip_336,
        ),
    ];
    const PREFAB_TABLE: &'static str = "vit";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burner::module::ToStructureConfig;

    /// Every prefab lowers, at its name's size.
    #[test]
    fn test_prefabs_lower() {
        for name in ViTContractConfig::prefab_names() {
            let config = ViTContractConfig::expect_prefab(name);
            config
                .try_to_structure()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let size = name.trim_end_matches("_miil").rsplit('_').next().unwrap();
            assert_eq!(config.image_size.to_string(), size, "{name}");
        }
        assert_eq!(ViTContractConfig::prefab_names().len(), 25);
    }
}
