//! # Hugging Face as a Whisper provider
//!
//! `hf:{org}/{repo}` is a Hugging Face repo in `transformers`' Whisper
//! layout: `hf:openai/whisper-large-v3` is
//! <https://huggingface.co/openai/whisper-large-v3>. [`HfWhisperProvider`]
//! answers such a ref with a one-resource map, the repo's
//! `model.safetensors` at a revision, and lists nothing: the hub is not
//! enumerable, and a bare name never reaches it.
//!
//! The row is unpinned. Resolving a name touches no network, so there is
//! no digest to pin the file by; the cache keys it by its URL, revision
//! included, and a repo whose `main` moves is fetched again only when the
//! cache is cleared or the revision is named. The checkpoint's `kind` is
//! [`SAFETENSORS`], which is what makes
//! [`WhisperConstruct`](super::WhisperConstruct) read it through the
//! safetensors reader (feature `store_safetensors`). The vocabulary is not
//! the repo's: `transformers` ships a tokenizer in its own format, and the
//! checkpoint's token layout selects one of `OpenAI`'s rank files, as it
//! does for a checkpoint given by path.

use crate::{
    data::pretrained::{
        Pretrained,
        PretrainedProvider,
        Resource,
        ResourceMap,
        Source,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::pretrained::{
        CHECKPOINT,
        SAFETENSORS,
    },
};

/// The provider's name: the `hf` of `hf:openai/whisper-tiny`, and the
/// namespace its files are cached under.
pub const HF: &str = "hf";

/// Where the hub is.
pub const HF_ORIGIN: &str = "https://huggingface.co";

/// The revision a ref resolves at unless another is named.
pub const HF_MAIN: &str = "main";

/// The file a `transformers` Whisper repo keeps its weights in.
pub const HF_MODEL_FILE: &str = "model.safetensors";

/// Hugging Face repos in `transformers`' Whisper layout, by ref.
#[derive(Clone, Debug)]
pub struct HfWhisperProvider {
    /// The revision every ref resolves at: a branch, a tag, or a commit.
    pub revision: String,
}

impl Default for HfWhisperProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl HfWhisperProvider {
    /// Repos at [`HF_MAIN`].
    pub fn new() -> Self {
        Self {
            revision: HF_MAIN.to_string(),
        }
    }

    /// Repos at `revision`.
    pub fn with_revision(
        mut self,
        revision: impl Into<String>,
    ) -> Self {
        self.revision = revision.into();
        self
    }

    /// The URL `file` in `org/repo` is served at, at this provider's
    /// revision.
    pub fn url(
        &self,
        org: &str,
        repo: &str,
        file: &str,
    ) -> String {
        format!("{HF_ORIGIN}/{org}/{repo}/resolve/{}/{file}", self.revision)
    }

    /// The `org` and `repo` of a ref, which is exactly `org/repo`.
    fn split(name: &str) -> BunsenResult<(&str, &str)> {
        match name.split_once('/') {
            Some((org, repo)) if !org.is_empty() && !repo.is_empty() && !repo.contains('/') => {
                Ok((org, repo))
            }
            _ => Err(BunsenError::Invalid(format!(
                "{HF}:{name}: a Hugging Face ref is org/repo"
            ))),
        }
    }
}

impl PretrainedProvider for HfWhisperProvider {
    fn name(&self) -> &str {
        HF
    }

    fn description(&self) -> &str {
        "a Hugging Face repo in transformers' Whisper layout, by ref: `hf:org/repo` is its model.safetensors, unpinned; lists nothing"
    }

    fn origin(&self) -> Option<&str> {
        Some(HF_ORIGIN)
    }

    /// Nothing: the hub is not enumerable.
    fn list(&self) -> Vec<Pretrained> {
        Vec::new()
    }

    /// `org/repo` as a row: one unpinned resource, the repo's
    /// `model.safetensors` at this provider's revision, labeled
    /// [`SAFETENSORS`].
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] for a ref that is not `org/repo`. Nothing
    /// here asks the hub whether the repo exists; a fetch will.
    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<Pretrained>> {
        let (org, repo) = Self::split(name)?;
        let origin = format!("{HF_ORIGIN}/{org}/{repo}");
        let description = format!("{org}/{repo} on Hugging Face, at {}", self.revision);
        let mut resources = ResourceMap::new(name);
        resources.description.clone_from(&description);
        resources.origin = Some(origin.clone());
        let resources = resources.with_resource(Resource {
            key: CHECKPOINT.to_string(),
            file: HF_MODEL_FILE.to_string(),
            sha256: None,
            kind: Some(SAFETENSORS.to_string()),
            namespace: HF.to_string(),
            sources: vec![Source::Url(self.url(org, repo, HF_MODEL_FILE))],
        });
        Ok(Some(Pretrained {
            name: name.to_string(),
            aliases: Vec::new(),
            description,
            license: None,
            origin: Some(origin),
            prefab: None,
            resources,
        }))
    }

    /// Never: a bare name means a row bunsen knows.
    fn answers_bare_names(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ref becomes a row over one unpinned resource: the repo's
    /// `model.safetensors` at `main`, labeled for the safetensors reader,
    /// cached under the provider's namespace.
    #[test]
    fn test_lookup_builds_the_row() {
        let hf = HfWhisperProvider::new();
        let row = hf.lookup("openai/whisper-large-v3").unwrap().unwrap();
        assert_eq!(row.name, "openai/whisper-large-v3");
        assert_eq!(
            row.origin.as_deref(),
            Some("https://huggingface.co/openai/whisper-large-v3")
        );
        assert_eq!(row.prefab, None);
        assert_eq!(hf.id(&row), "hf:openai/whisper-large-v3");

        row.resources.validate().unwrap();
        assert_eq!(row.resources.keys(), [CHECKPOINT]);
        let checkpoint = row.resources.get(CHECKPOINT).unwrap();
        assert_eq!(checkpoint.file, "model.safetensors");
        assert_eq!(checkpoint.sha256, None);
        assert_eq!(checkpoint.kind.as_deref(), Some("safetensors"));
        assert_eq!(checkpoint.namespace, "hf");
        assert_eq!(
            checkpoint.sources,
            [Source::Url(
                "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors"
                    .to_string()
            )]
        );
    }

    /// A revision is in the URL, and in the description.
    #[test]
    fn test_a_revision_is_in_the_url() {
        let hf = HfWhisperProvider::new().with_revision("06f233fe06e710322aca913c1bc4249a0d71fce1");
        let row = hf.lookup("openai/whisper-large-v3").unwrap().unwrap();
        let checkpoint = row.resources.get(CHECKPOINT).unwrap();
        assert_eq!(
            checkpoint.sources,
            [Source::Url("https://huggingface.co/openai/whisper-large-v3/resolve/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors".to_string())]
        );
        assert!(
            row.description
                .ends_with("at 06f233fe06e710322aca913c1bc4249a0d71fce1")
        );
    }

    /// A ref that is not `org/repo` is refused, not looked up.
    #[test]
    fn test_a_malformed_ref_is_invalid() {
        let hf = HfWhisperProvider::new();
        for bad in [
            "whisper-tiny",
            "openai/",
            "/whisper-tiny",
            "openai/whisper/tiny",
            "",
        ] {
            let err = hf.lookup(bad).unwrap_err();
            assert!(
                matches!(&err, BunsenError::Invalid(m) if m.contains("org/repo")),
                "{bad:?}: {err}"
            );
        }
    }

    /// The hub lists nothing and declines bare names.
    #[test]
    fn test_lists_nothing_and_declines_bare_names() {
        let hf = HfWhisperProvider::new();
        assert_eq!(hf.name(), "hf");
        assert_eq!(hf.origin(), Some("https://huggingface.co"));
        assert!(hf.list().is_empty());
        assert!(hf.ids().is_empty());
        assert!(hf.for_prefab("tiny").is_empty());
        assert!(!hf.answers_bare_names());
    }

    /// Through the default factory: `hf:org/repo` resolves to a deferred
    /// model with the safetensors reader, its checkpoint remote; the bare
    /// `org/repo` never reaches the hub.
    #[cfg(all(feature = "store_pytorch", feature = "cache"))]
    #[test]
    fn test_through_the_default_factory() {
        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    CacheStatus,
                    PretrainedCache,
                    PretrainedCacheOptions,
                    PretrainedRef,
                },
            },
            kits::speech::whisper::pretrained::default_whisper_factory,
        };
        let factory = default_whisper_factory().unwrap();
        assert_eq!(factory.provider(HF).unwrap().name(), "hf");

        let (provider, row) = factory.lookup("hf:openai/whisper-tiny").unwrap();
        assert_eq!(provider, "hf");
        assert_eq!(row.name, "openai/whisper-tiny");
        assert!(matches!(
            factory.lookup("openai/whisper-tiny"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert!(matches!(
            factory.lookup("hf:whisper-tiny"),
            Err(BunsenError::Invalid(_))
        ));

        let resolved = factory.resolve("hf:openai/whisper-tiny");
        #[cfg(not(feature = "store_safetensors"))]
        {
            assert!(
                matches!(&resolved, Err(BunsenError::Invalid(m)) if m.contains("store_safetensors")),
                "{resolved:?}"
            );
        }
        #[cfg(feature = "store_safetensors")]
        {
            use crate::kits::speech::whisper::pretrained::WhisperReader;
            let model = resolved.unwrap();
            assert_eq!(model.id(), "hf:openai/whisper-tiny");
            assert!(matches!(model.hook.reader, WhisperReader::Safetensors(_)));
            assert!(model.hook.expected_for(&model.model).is_none());
            assert!(
                matches!(&model.model, PretrainedRef::Named { provider, .. } if provider == "hf")
            );

            let dir = tempfile::tempdir().unwrap();
            let cache = PretrainedCache::new(
                PretrainedCacheOptions::default()
                    .with_disk(
                        BunsenDiskCacheOptions::default()
                            .with_cache_dir(Some(dir.path().join("cache")))
                            .without_transfer_observers(),
                    )
                    .with_offline(true),
            )
            .unwrap();
            let status = model.status(&cache);
            assert_eq!(status[CHECKPOINT], CacheStatus::Remote);
            assert!(
                model.plan(&cache).is_err(),
                "offline: the checkpoint cannot come local"
            );
            assert!(!dir.path().join("cache").exists(), "nothing was written");
        }
    }
}

/// Against the hub: the smallest repo, fetched once into the default
/// cache, read whole, and checked against `OpenAI`'s own `tiny.pt`, which
/// it was converted from.
#[cfg(all(test, feature = "store_safetensors", feature = "fetch"))]
mod hub_tests {
    use burn::tensor::{
        Tensor,
        Tolerance,
    };

    use crate::{
        data::pretrained::{
            PretrainedCache,
            PretrainedCacheOptions,
        },
        kits::speech::whisper::{
            WhisperMeta,
            pretrained::{
                WHISPER_PREFABS,
                default_whisper_factory,
            },
        },
        support::testing::{
            CpuBackend,
            assert_tensors_close,
            default_device,
        },
    };

    /// `hf:openai/whisper-tiny` loads to the `tiny` geometry with the
    /// multilingual layout and vocabulary, and its weights are
    /// `openai/tiny`'s: the same numbers through two files, two layouts and
    /// two readers.
    #[test]
    fn test_hf_whisper_tiny_is_openai_tiny() {
        let device = default_device();
        let cache = PretrainedCache::new(PretrainedCacheOptions::default()).unwrap();
        let factory = default_whisper_factory().unwrap();

        let hf = factory
            .load_bundle::<CpuBackend>("hf:openai/whisper-tiny", &cache, &device)
            .unwrap();
        let tiny = WHISPER_PREFABS
            .expect_lookup_prefab("tiny")
            .to_config()
            .geometry();
        assert_eq!(hf.model.n_mels(), tiny.n_mels);
        assert_eq!(hf.model.vocab_size(), tiny.vocab_size);
        assert_eq!(hf.model.d_model(), tiny.d_model);
        assert_eq!(hf.model.max_audio_ctx(), tiny.max_audio_ctx);
        assert_eq!(hf.model.max_text_ctx(), tiny.max_text_ctx);
        assert_eq!(hf.model.encoder.blocks.len(), tiny.n_encoder_layers);
        assert_eq!(hf.model.decoder.blocks.len(), tiny.n_decoder_layers);
        assert!(hf.layout.ids().is_multilingual());
        assert_eq!(hf.ranks.as_ref().map(|r| r.len()), Some(50257));

        let openai = factory
            .load_bundle::<CpuBackend>("openai/tiny", &cache, &device)
            .unwrap();
        let close = |a: Tensor<CpuBackend, 2>, b: Tensor<CpuBackend, 2>| {
            assert_tensors_close(&a, &b, Tolerance::<f32>::default());
        };
        close(
            hf.model.encoder.blocks[0].attn.query.weight.val(),
            openai.model.encoder.blocks[0].attn.query.weight.val(),
        );
        close(
            hf.model.decoder.blocks[3].cross_attn.output.weight.val(),
            openai.model.decoder.blocks[3]
                .cross_attn
                .output
                .weight
                .val(),
        );
        close(
            hf.model.decoder.blocks[1].mlp.linear2.weight.val(),
            openai.model.decoder.blocks[1].mlp.linear2.weight.val(),
        );
        close(
            hf.model.encoder.positional_embedding.val(),
            openai.model.encoder.positional_embedding.val(),
        );
        close(
            hf.model.decoder.token_embedding.weight.val(),
            openai.model.decoder.token_embedding.weight.val(),
        );
        close(
            hf.model.decoder.ln.gamma.val().unsqueeze(),
            openai.model.decoder.ln.gamma.val().unsqueeze(),
        );
    }
}
