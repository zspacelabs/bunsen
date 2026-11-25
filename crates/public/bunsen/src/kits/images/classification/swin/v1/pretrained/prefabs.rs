//! # Swin V1 prefabs

use alloc::vec;

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::classification::swin::v1::SwinV1ContractConfig,
};

impl SwinV1ContractConfig {
    /// The `swin_tiny_patch4_window7_224` prefab: Swin-T at 224 pixels in 7x7
    /// windows.
    pub fn swin_tiny_patch4_window7_224() -> Self {
        Self::new(224, 96, vec![2, 2, 6, 2], vec![3, 6, 12, 24], 7, 1000)
    }

    /// The `swin_small_patch4_window7_224` prefab: Swin-S at 224 pixels in 7x7
    /// windows.
    pub fn swin_small_patch4_window7_224() -> Self {
        Self::new(224, 96, vec![2, 2, 18, 2], vec![3, 6, 12, 24], 7, 1000)
    }

    /// The `swin_base_patch4_window7_224` prefab: Swin-B at 224 pixels in 7x7
    /// windows.
    pub fn swin_base_patch4_window7_224() -> Self {
        Self::new(224, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 7, 1000)
    }

    /// The `swin_base_patch4_window12_384` prefab: Swin-B at 384 pixels in
    /// 12x12 windows.
    pub fn swin_base_patch4_window12_384() -> Self {
        Self::new(384, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 12, 1000)
    }

    /// The `swin_large_patch4_window7_224` prefab: Swin-L at 224 pixels in 7x7
    /// windows.
    pub fn swin_large_patch4_window7_224() -> Self {
        Self::new(224, 192, vec![2, 2, 18, 2], vec![6, 12, 24, 48], 7, 1000)
    }

    /// The `swin_large_patch4_window12_384` prefab: Swin-L at 384 pixels in
    /// 12x12 windows.
    pub fn swin_large_patch4_window12_384() -> Self {
        Self::new(384, 192, vec![2, 2, 18, 2], vec![6, 12, 24, 48], 12, 1000)
    }

    /// The `swin_s3_tiny_224` prefab: Swin-S3-T at 224 pixels, its stages in
    /// 7x7, 7x7, 14x14, 7x7 windows (a searched geometry, arXiv:2111.14725).
    pub fn swin_s3_tiny_224() -> Self {
        Self::new(224, 96, vec![2, 2, 6, 2], vec![3, 6, 12, 24], 7, 1000)
            .with_stage_window_sizes(vec![7, 7, 14, 7])
    }

    /// The `swin_s3_small_224` prefab: Swin-S3-S at 224 pixels, its stages in
    /// 14x14, 14x14, 14x14, 7x7 windows (a searched geometry,
    /// arXiv:2111.14725).
    pub fn swin_s3_small_224() -> Self {
        Self::new(224, 96, vec![2, 2, 18, 2], vec![3, 6, 12, 24], 7, 1000)
            .with_stage_window_sizes(vec![14, 14, 14, 7])
    }

    /// The `swin_s3_base_224` prefab: Swin-S3-B at 224 pixels, its stages in
    /// 7x7, 7x7, 14x14, 7x7 windows (a searched geometry, arXiv:2111.14725).
    pub fn swin_s3_base_224() -> Self {
        Self::new(224, 96, vec![2, 2, 30, 2], vec![3, 6, 12, 24], 7, 1000)
            .with_stage_window_sizes(vec![7, 7, 14, 7])
    }
}

/// The Swin (V1) geometries by name, timm's architecture names: 9
/// [`SwinV1ContractConfig`]s, each with 1000 classes at its name's size:
/// tiny, small, base and large in `7x7` windows at 224 pixels, base and large
/// in `12x12` at 384, and the three Swin-S3 geometries, a window per stage.
///
/// A prefab has no weights: a timm row takes the classes and the size from
/// its `config.json`.
impl Prefabs for SwinV1ContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        (
            "swin_tiny_patch4_window7_224",
            Self::swin_tiny_patch4_window7_224,
        ),
        (
            "swin_small_patch4_window7_224",
            Self::swin_small_patch4_window7_224,
        ),
        (
            "swin_base_patch4_window7_224",
            Self::swin_base_patch4_window7_224,
        ),
        (
            "swin_base_patch4_window12_384",
            Self::swin_base_patch4_window12_384,
        ),
        (
            "swin_large_patch4_window7_224",
            Self::swin_large_patch4_window7_224,
        ),
        (
            "swin_large_patch4_window12_384",
            Self::swin_large_patch4_window12_384,
        ),
        ("swin_s3_tiny_224", Self::swin_s3_tiny_224),
        ("swin_s3_small_224", Self::swin_s3_small_224),
        ("swin_s3_base_224", Self::swin_s3_base_224),
    ];
    const PREFAB_TABLE: &'static str = "swin-v1";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burner::module::ToStructureConfig;

    /// Every prefab lowers, at its name's size.
    #[test]
    fn test_prefabs_lower() {
        for name in SwinV1ContractConfig::prefab_names() {
            let config = SwinV1ContractConfig::expect_prefab(name);
            config
                .try_to_structure()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let size = name.rsplit('_').next().unwrap();
            assert_eq!(config.image_size.to_string(), size, "{name}");
        }
        assert_eq!(SwinV1ContractConfig::prefab_names().len(), 9);
    }
}
