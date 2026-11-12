//! # Voxtral Realtime prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::speech::asr::voxtral::VoxtralRealtimeConfig,
};

/// The Voxtral Realtime prefabs: the geometries bunsen's builds state, by
/// name, without weights; `voxtral-mini-4b-realtime-2602` is
/// [`VoxtralRealtimeConfig::voxtral_mini_4b_realtime`], a 32-layer 1280-wide
/// causal audio encoder and a 26-layer 3072-wide Ministral-3 decoder.
impl Prefabs for VoxtralRealtimeConfig {
    const PREFABS: PrefabTable<Self> = &[(
        "voxtral-mini-4b-realtime-2602",
        Self::voxtral_mini_4b_realtime,
    )];
    const PREFAB_TABLE: &'static str = "voxtral";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::asr::voxtral::blocks::VoxtralRealtimeMeta;

    #[test]
    fn test_prefabs() {
        let config = VoxtralRealtimeConfig::expect_prefab("voxtral-mini-4b-realtime-2602");
        assert_eq!(config.d_audio(), 1280);
        assert_eq!(config.d_text(), 3072);
        assert!(VoxtralRealtimeConfig::try_prefab("voxtral-9b").is_err());
    }
}
