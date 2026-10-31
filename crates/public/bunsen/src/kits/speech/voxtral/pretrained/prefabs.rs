//! # Voxtral Realtime prefabs
use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::speech::voxtral::VoxtralRealtimeConfig,
};

/// The Voxtral Realtime prefabs: the geometries bunsen's rows instantiate, by
/// name, without weights.
pub static VOXTRAL_PREFABS: StaticPreFabMap<VoxtralRealtimeConfig> = StaticPreFabMap {
    name: "voxtral",
    description: "Voxtral Realtime geometries, as transformers' VoxtralRealtimeConfig has them",
    items: &[&StaticPreFabConfig {
        name: "voxtral-mini-4b-realtime-2602",
        description: "a 32-layer 1280-wide causal audio encoder and a 26-layer 3072-wide Ministral-3 decoder",
        builder: VoxtralRealtimeConfig::voxtral_mini_4b_realtime,
    }],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::voxtral::blocks::VoxtralRealtimeMeta;

    #[test]
    fn test_prefabs() {
        let config = VOXTRAL_PREFABS
            .expect_lookup_prefab("voxtral-mini-4b-realtime-2602")
            .to_config();
        assert_eq!(config.d_audio(), 1280);
        assert_eq!(config.d_text(), 3072);
        assert!(VOXTRAL_PREFABS.try_lookup_prefab("voxtral-9b").is_err());
    }
}
