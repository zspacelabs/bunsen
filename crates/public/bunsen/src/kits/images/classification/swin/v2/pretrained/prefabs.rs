//! # Swin V2 prefabs

use alloc::vec;

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::classification::swin::v2::SwinV2ContractConfig,
};

impl SwinV2ContractConfig {
    /// The `swinv2_tiny_window16_256` prefab: Swin V2-Tiny at 256 pixels in
    /// 16x16 windows.
    pub fn swinv2_tiny_window16_256() -> Self {
        Self::new(256, 96, vec![2, 2, 6, 2], vec![3, 6, 12, 24], 16, 1000)
    }

    /// The `swinv2_tiny_window8_256` prefab: Swin V2-Tiny at 256 pixels in 8x8
    /// windows.
    pub fn swinv2_tiny_window8_256() -> Self {
        Self::new(256, 96, vec![2, 2, 6, 2], vec![3, 6, 12, 24], 8, 1000)
    }

    /// The `swinv2_small_window16_256` prefab: Swin V2-Small at 256 pixels in
    /// 16x16 windows.
    pub fn swinv2_small_window16_256() -> Self {
        Self::new(256, 96, vec![2, 2, 18, 2], vec![3, 6, 12, 24], 16, 1000)
    }

    /// The `swinv2_small_window8_256` prefab: Swin V2-Small at 256 pixels in
    /// 8x8 windows.
    pub fn swinv2_small_window8_256() -> Self {
        Self::new(256, 96, vec![2, 2, 18, 2], vec![3, 6, 12, 24], 8, 1000)
    }

    /// The `swinv2_base_window16_256` prefab: Swin V2-Base at 256 pixels in
    /// 16x16 windows.
    pub fn swinv2_base_window16_256() -> Self {
        Self::new(256, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 16, 1000)
    }

    /// The `swinv2_base_window8_256` prefab: Swin V2-Base at 256 pixels in 8x8
    /// windows.
    pub fn swinv2_base_window8_256() -> Self {
        Self::new(256, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 8, 1000)
    }

    /// The `swinv2_base_window12_192` prefab: Swin V2-Base at 192 pixels in
    /// 12x12 windows.
    pub fn swinv2_base_window12_192() -> Self {
        Self::new(192, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 12, 1000)
    }

    /// The `swinv2_base_window12to16_192to256` prefab: Swin V2-Base at 256
    /// pixels in 16x16 windows, its weights trained at 192 pixels in 12x12
    /// windows.
    pub fn swinv2_base_window12to16_192to256() -> Self {
        Self::new(256, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 16, 1000)
            .with_pretrained_window_sizes(vec![12, 12, 12, 6])
    }

    /// The `swinv2_base_window12to24_192to384` prefab: Swin V2-Base at 384
    /// pixels in 24x24 windows, its weights trained at 192 pixels in 12x12
    /// windows.
    pub fn swinv2_base_window12to24_192to384() -> Self {
        Self::new(384, 128, vec![2, 2, 18, 2], vec![4, 8, 16, 32], 24, 1000)
            .with_pretrained_window_sizes(vec![12, 12, 12, 6])
    }

    /// The `swinv2_large_window12_192` prefab: Swin V2-Large at 192 pixels in
    /// 12x12 windows.
    pub fn swinv2_large_window12_192() -> Self {
        Self::new(192, 192, vec![2, 2, 18, 2], vec![6, 12, 24, 48], 12, 1000)
    }

    /// The `swinv2_large_window12to16_192to256` prefab: Swin V2-Large at 256
    /// pixels in 16x16 windows, its weights trained at 192 pixels in 12x12
    /// windows.
    pub fn swinv2_large_window12to16_192to256() -> Self {
        Self::new(256, 192, vec![2, 2, 18, 2], vec![6, 12, 24, 48], 16, 1000)
            .with_pretrained_window_sizes(vec![12, 12, 12, 6])
    }

    /// The `swinv2_large_window12to24_192to384` prefab: Swin V2-Large at 384
    /// pixels in 24x24 windows, its weights trained at 192 pixels in 12x12
    /// windows.
    pub fn swinv2_large_window12to24_192to384() -> Self {
        Self::new(384, 192, vec![2, 2, 18, 2], vec![6, 12, 24, 48], 24, 1000)
            .with_pretrained_window_sizes(vec![12, 12, 12, 6])
    }
}

/// The Swin V2 geometries by name, timm's architecture names: 12
/// [`SwinV2ContractConfig`]s, each with 1000 classes at its name's size:
/// tiny, small, base and large, in `8x8`, `12x12` or `16x16` windows, and
/// the `12to16` and `12to24` fine-tunes of the 192-pixel pretrainings.
///
/// A prefab has no weights: a timm row takes the classes and the size from
/// its `config.json`.
impl Prefabs for SwinV2ContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("swinv2_tiny_window16_256", Self::swinv2_tiny_window16_256),
        ("swinv2_tiny_window8_256", Self::swinv2_tiny_window8_256),
        ("swinv2_small_window16_256", Self::swinv2_small_window16_256),
        ("swinv2_small_window8_256", Self::swinv2_small_window8_256),
        ("swinv2_base_window16_256", Self::swinv2_base_window16_256),
        ("swinv2_base_window8_256", Self::swinv2_base_window8_256),
        ("swinv2_base_window12_192", Self::swinv2_base_window12_192),
        (
            "swinv2_base_window12to16_192to256",
            Self::swinv2_base_window12to16_192to256,
        ),
        (
            "swinv2_base_window12to24_192to384",
            Self::swinv2_base_window12to24_192to384,
        ),
        ("swinv2_large_window12_192", Self::swinv2_large_window12_192),
        (
            "swinv2_large_window12to16_192to256",
            Self::swinv2_large_window12to16_192to256,
        ),
        (
            "swinv2_large_window12to24_192to384",
            Self::swinv2_large_window12to24_192to384,
        ),
    ];
    const PREFAB_TABLE: &'static str = "swin-v2";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burner::module::ToStructureConfig;

    /// Every prefab lowers, at its name's size.
    #[test]
    fn test_prefabs_lower() {
        for name in SwinV2ContractConfig::prefab_names() {
            let config = SwinV2ContractConfig::expect_prefab(name);
            config
                .try_to_structure()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let size = name.rsplit('_').next().unwrap();
            let size = size.rsplit("to").next().unwrap();
            assert_eq!(config.image_size.to_string(), size, "{name}");
        }
        assert_eq!(SwinV2ContractConfig::prefab_names().len(), 12);
    }
}
