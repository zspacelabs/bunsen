//! # Sortformer pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::SortformerConstruct;
use crate::{
    data::pretrained::{
        HfProvider,
        PretrainedCache,
        PretrainedFactory,
        PretrainedProvider,
        StaticBase,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResource,
        StaticResourceMap,
        WELL_KNOWN,
    },
    errors::BunsenResult,
    kits::speech::sortformer::SortformerBundle,
};

/// The kit segment of a Sortformer resource's path in the cache:
/// `<cache>/pretrained/sortformer/<namespace>/<sha256>/<file>`.
pub const SORTFORMER_KIT: &str = "sortformer";

/// The key of the checkpoint in a Sortformer resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `NeMo` `.nemo` checkpoint: its config and its weights in
/// one archive.
pub const NEMO: &str = "nemo";

const NVIDIA_ORIGIN: &str = "https://huggingface.co/nvidia/Nemotron-3-Diarization";

/// `nvidia/Nemotron-3-Diarization`'s `transformers` checkpoint, at a fixed
/// revision.
pub static NEMOTRON_3_DIARIZATION_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nvidia/nemotron-3-diarization",
    description: "Nemotron-3-Diarization: model.safetensors (f32)",
    license: Some("OpenMDW-1.1"),
    origin: Some(NVIDIA_ORIGIN),
    namespace: "nvidia",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e",
    )],
    resources: &[StaticResource {
        key: CHECKPOINT,
        file: "model.safetensors",
        sha256: Some("c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"),
        kind: Some(SAFETENSORS),
        sources: &[],
    }],
};

static NEMOTRON_3_DIARIZATION: StaticPretrained<'static> = StaticPretrained {
    name: "nemotron-3-diarization",
    aliases: &[],
    description: "Nemotron-3-Diarization: 100 M parameters, up to 8 speakers, 10 ms frames, streaming",
    license: Some("OpenMDW-1.1"),
    origin: Some(NVIDIA_ORIGIN),
    prefab: None,
    maps: &[&NEMOTRON_3_DIARIZATION_FILES],
};

const V2_ORIGIN: &str = "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2";

const V2_1_ORIGIN: &str = "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1";

const NVIDIA_OPEN_MODEL_LICENSE: &str = "NVIDIA Open Model License";

/// `nvidia/diar_streaming_sortformer_4spk-v2`'s `.nemo`, at a fixed revision.
pub static SORTFORMER_V2_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nvidia/diar_streaming_sortformer_4spk-v2",
    description: "Streaming Sortformer v2: diar_streaming_sortformer_4spk-v2.nemo (f32)",
    license: Some(NVIDIA_OPEN_MODEL_LICENSE),
    origin: Some(V2_ORIGIN),
    namespace: "nvidia",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/resolve/84edd514b8ef68004c10086918cd62f2148cbd59",
    )],
    resources: &[StaticResource {
        key: CHECKPOINT,
        file: "diar_streaming_sortformer_4spk-v2.nemo",
        sha256: Some("b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329"),
        kind: Some(NEMO),
        sources: &[],
    }],
};

/// `nvidia/diar_streaming_sortformer_4spk-v2.1`'s `.nemo`, at a fixed
/// revision.
pub static SORTFORMER_V2_1_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "nvidia/diar_streaming_sortformer_4spk-v2.1",
    description: "Streaming Sortformer v2.1: diar_streaming_sortformer_4spk-v2.1.nemo (f32)",
    license: Some(NVIDIA_OPEN_MODEL_LICENSE),
    origin: Some(V2_1_ORIGIN),
    namespace: "nvidia",
    bases: &[StaticBase::Url(
        "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/resolve/cd03eee90fbec18297ac31b8c21546e596b7f71c",
    )],
    resources: &[StaticResource {
        key: CHECKPOINT,
        file: "diar_streaming_sortformer_4spk-v2.1.nemo",
        sha256: Some("8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"),
        kind: Some(NEMO),
        sources: &[],
    }],
};

static SORTFORMER_V2: StaticPretrained<'static> = StaticPretrained {
    name: "diar_streaming_sortformer_4spk-v2",
    aliases: &["sortformer-v2"],
    description: "Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms frames",
    license: Some(NVIDIA_OPEN_MODEL_LICENSE),
    origin: Some(V2_ORIGIN),
    prefab: None,
    maps: &[&SORTFORMER_V2_FILES],
};

static SORTFORMER_V2_1: StaticPretrained<'static> = StaticPretrained {
    name: "diar_streaming_sortformer_4spk-v2.1",
    aliases: &["sortformer-v2.1"],
    description: "Streaming Sortformer v2.1: 117 M parameters, up to 4 speakers, 80 ms frames",
    license: Some(NVIDIA_OPEN_MODEL_LICENSE),
    origin: Some(V2_1_ORIGIN),
    prefab: None,
    maps: &[&SORTFORMER_V2_1_FILES],
};

/// NVIDIA's Sortformer diarizers.
pub static NVIDIA: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "nvidia",
    description: "NVIDIA's Sortformer speaker diarizers: Nemotron-3-Diarization as transformers ships it, the streaming Sortformer v2 and v2.1 as NeMo does",
    license: None,
    origin: Some("https://huggingface.co/nvidia"),
    items: &[&NEMOTRON_3_DIARIZATION, &SORTFORMER_V2, &SORTFORMER_V2_1],
};

/// The Sortformer checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:nvidia/nemotron-3-diarization`,
/// `nvidia/nemotron-3-diarization`, or `nemotron-3-diarization`; and so
/// `nvidia/diar_streaming_sortformer_4spk-v2` (`sortformer-v2`) and
/// `nvidia/diar_streaming_sortformer_4spk-v2.1` (`sortformer-v2.1`).
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the Sortformer diarizers bunsen knows by name",
    groups: &[&NVIDIA],
};

/// The name [`default_sortformer_factory`] loads when a caller names none.
pub const DEFAULT_SORTFORMER_MODEL: &str = "nvidia/nemotron-3-diarization";

/// Sortformer's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, and its `config.json`).
pub fn default_sortformer_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(WELL_KNOWN_TABLE.to_table()),
        Arc::new(HfProvider::new().with_checkpoint_key(CHECKPOINT)),
    ]
}

/// Sortformer's factory over [`default_sortformer_providers`], building
/// through [`SortformerConstruct`].
///
/// [`load_bundle`](PretrainedFactory::load_bundle) with `"[provider:]name"`
/// is the whole pathway from a name to a [`SortformerBundle`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_sortformer_factory() -> BunsenResult<PretrainedFactory<SortformerConstruct>> {
    PretrainedFactory::new().with_providers(default_sortformer_providers())
}

impl PretrainedFactory<SortformerConstruct> {
    /// A name to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<SortformerBundle>> {
        Ok(self.load(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is pinned, and answers by its qualified and short names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        for (files, kind) in [
            (&NEMOTRON_3_DIARIZATION_FILES, SAFETENSORS),
            (&SORTFORMER_V2_FILES, NEMO),
            (&SORTFORMER_V2_1_FILES, NEMO),
        ] {
            let map = files.to_map();
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{}", files.name);
            assert_eq!(checkpoint.urls().len(), 1, "{}", files.name);
            assert!(checkpoint.urls()[0].ends_with(&checkpoint.file));
            assert_eq!(checkpoint.kind.as_deref(), Some(kind), "{}", files.name);
        }

        let factory = default_sortformer_factory().unwrap();
        assert_eq!(factory.kit(), SORTFORMER_KIT);
        assert_eq!(
            factory.ids(),
            [
                "well-known:nvidia/nemotron-3-diarization",
                "well-known:nvidia/diar_streaming_sortformer_4spk-v2",
                "well-known:nvidia/diar_streaming_sortformer_4spk-v2.1",
            ]
        );
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
        for (spec, name) in [
            (DEFAULT_SORTFORMER_MODEL, "nvidia/nemotron-3-diarization"),
            ("nemotron-3-diarization", "nvidia/nemotron-3-diarization"),
            (
                "well-known:nvidia/nemotron-3-diarization",
                "nvidia/nemotron-3-diarization",
            ),
            (
                "nvidia/diar_streaming_sortformer_4spk-v2",
                "nvidia/diar_streaming_sortformer_4spk-v2",
            ),
            ("sortformer-v2", "nvidia/diar_streaming_sortformer_4spk-v2"),
            (
                "diar_streaming_sortformer_4spk-v2.1",
                "nvidia/diar_streaming_sortformer_4spk-v2.1",
            ),
            (
                "sortformer-v2.1",
                "nvidia/diar_streaming_sortformer_4spk-v2.1",
            ),
        ] {
            assert_eq!(factory.lookup(spec).unwrap().1.name, name, "{spec}");
        }
    }
}
