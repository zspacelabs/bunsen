//! What a pretrained image model comes with: its eval transform and labels.

use super::{
    labels::ImageLabelSet,
    transform::ImageEvalTransformConfig,
};

/// The key of a classifier's label set (its synsets) in an image build's
/// resource set.
pub const LABELS: &str = "labels";

/// The key of the synsets' lemmas, their readable names, in an image build's
/// resource set.
pub const LABEL_NAMES: &str = "label_names";

/// The key of an image build's stated eval transform: an
/// [`ImageEvalTransformConfig`], as it serializes.
pub const EVAL_TRANSFORM: &str = "eval_transform";

/// What a pretrained image model comes with, beside its weights: the eval
/// transform its images go through, and the labels of its classes.
///
/// A build states them: its [`EVAL_TRANSFORM`] argument, else its timm
/// `config.json`'s `pretrained_cfg` (bound as `config`), else the transform's
/// defaults; its [`LABELS`] and [`LABEL_NAMES`] resources, else no labels.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageModelCard {
    /// The eval transform the model's images go through.
    pub transform: ImageEvalTransformConfig,

    /// The classes' labels, when the build binds them.
    pub labels: Option<ImageLabelSet>,
}

impl ImageModelCard {
    /// A card with `transform` and no labels.
    pub fn new(transform: ImageEvalTransformConfig) -> Self {
        Self {
            transform,
            labels: None,
        }
    }

    /// The card with `labels`.
    pub fn with_labels(
        self,
        labels: Option<ImageLabelSet>,
    ) -> Self {
        Self { labels, ..self }
    }

    /// The card a build states: its [`EVAL_TRANSFORM`] argument, else its
    /// loaded timm `config.json`'s (`config`), else the transform's defaults;
    /// and its loaded [`LABELS`] and [`LABEL_NAMES`] files, else no labels.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    /// An [`EVAL_TRANSFORM`] that does not deserialize or validate; a
    /// `config.json` that is not timm's, or states a transform bunsen does
    /// not build
    /// ([`eval_transform_config`](super::timm::interop::TimmPretrainedCfgMirror::eval_transform_config));
    /// a label
    /// file that does not read or parse
    /// ([`ImageLabelSet::try_from_files`]); [`LABEL_NAMES`] without
    /// [`LABELS`], [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource).
    #[cfg(feature = "cache")]
    pub fn try_from_build(
        model: &crate::data::builds::BuildRef,
        loaded: &crate::data::resources::LoadedResourceSet,
    ) -> crate::errors::BunsenResult<Self> {
        use crate::errors::BunsenError;

        let transform = match model.stated::<ImageEvalTransformConfig>(EVAL_TRANSFORM)? {
            Some(transform) => transform,
            None => match loaded.get(crate::data::builds::CONFIG) {
                Some(config) => super::timm::interop::timm_model_config_from_path(&config.path)?
                    .pretrained_cfg
                    .eval_transform_config()?,
                None => ImageEvalTransformConfig::new(),
            },
        };
        transform.try_validate()?;
        let names = loaded.get(LABEL_NAMES).map(|r| &r.path);
        let labels = match loaded.get(LABELS) {
            Some(synsets) => Some(ImageLabelSet::try_from_files(&synsets.path, names)?),
            None if names.is_some() => {
                return Err(BunsenError::invalid_resource(format!(
                    "{}: binds {LABEL_NAMES} without {LABELS}",
                    model.id()
                )));
            }
            None => None,
        };
        Ok(Self { transform, labels })
    }

    /// The card, checked against a head of `n_classes` classes: its labels,
    /// when it has some, name each class.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for labels of another count (a build binding the wrong `ImageNet`
    /// subset).
    pub fn try_for_classes(
        self,
        n_classes: usize,
    ) -> crate::errors::BunsenResult<Self> {
        match &self.labels {
            Some(labels) if labels.len() != n_classes => {
                Err(crate::errors::BunsenError::invalid_resource(format!(
                    "{} labels for a head of {n_classes} classes",
                    labels.len()
                )))
            }
            _ => Ok(self),
        }
    }
}

/// A pretrained image model as a factory loads it: the model and its
/// [`ImageModelCard`].
///
/// A kit's construct, for every image kit: the model is whatever Module the
/// kit builds.
#[derive(Debug, Clone)]
pub struct ImageModelConstruct<M> {
    /// The model, its weights loaded.
    pub model: M,

    /// The model's eval transform and labels.
    pub card: ImageModelCard,
}

#[cfg(all(test, feature = "cache"))]
mod tests {
    use std::{
        collections::BTreeMap,
        path::Path,
    };

    use super::*;
    use crate::{
        data::{
            builds::BuildRef,
            resources::{
                LoadedResourceSet,
                Provenance,
                ResolvedResource,
                ResourceSet,
            },
        },
        errors::testing::ErrorMatcher,
    };

    /// Labels must name the head's classes; a card without labels fits any
    /// head.
    #[test]
    fn test_try_for_classes() {
        let card = ImageModelCard::new(ImageEvalTransformConfig::new());
        assert!(card.clone().try_for_classes(7).is_ok());
        let labelled =
            card.with_labels(Some(ImageLabelSet::try_parse("n01\nn02\n", None).unwrap()));
        assert!(labelled.clone().try_for_classes(2).is_ok());
        ErrorMatcher::invalid_resource()
            .message_contains("2 labels for a head of 1000 classes")
            .assert_err(&labelled.try_for_classes(1000));
    }

    /// A loaded set of the given parts, by key.
    fn loaded(parts: &[(&str, &Path)]) -> LoadedResourceSet {
        LoadedResourceSet {
            map: ResourceSet::given("labels", LABELS, parts[0].1),
            parts: parts
                .iter()
                .map(|(key, path)| {
                    (
                        key.to_string(),
                        ResolvedResource {
                            path: path.to_path_buf(),
                            provenance: Provenance::LocalDir,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        }
    }

    /// A build that states nothing has the transform's defaults; its label
    /// files, when bound, are read.
    #[test]
    fn test_try_from_build_defaults_and_reads_labels() {
        let dir = tempfile::tempdir().unwrap();
        let synsets = dir.path().join("synsets.txt");
        std::fs::write(&synsets, "n1\nn2\n").unwrap();
        let lemmas = dir.path().join("lemmas.txt");
        std::fs::write(&lemmas, "n1\tone\nn2\ttwo\n").unwrap();
        let build = BuildRef::from(ResourceSet::given("mine", "checkpoint", &synsets));

        let card = ImageModelCard::try_from_build(
            &build,
            &loaded(&[(LABELS, &synsets), (LABEL_NAMES, &lemmas)]),
        )
        .unwrap();
        assert_eq!(card.transform, ImageEvalTransformConfig::new());
        let labels = card.labels.unwrap();
        assert_eq!(labels.name(1), "two");

        let unlabeled =
            ImageModelCard::try_from_build(&build, &loaded(&[("checkpoint", &synsets)])).unwrap();
        assert_eq!(unlabeled.labels, None);
    }

    /// A bound timm `config.json` states the transform.
    #[test]
    fn test_try_from_build_reads_a_timm_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architecture": "resnet50", "num_classes": 1000, "num_features": 2048,
                "pretrained_cfg": {"input_size": [3, 160, 160], "interpolation": "bilinear",
                "crop_pct": 0.95, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}"#,
        )
        .unwrap();
        let build = BuildRef::from(ResourceSet::given("mine", "checkpoint", &config));
        let card = ImageModelCard::try_from_build(
            &build,
            &loaded(&[(crate::data::builds::CONFIG, &config)]),
        )
        .unwrap();
        assert_eq!(
            card.transform,
            ImageEvalTransformConfig::new()
                .with_size([160, 160])
                .with_interpolation(super::super::transform::ImageInterpolation::Bilinear)
                .with_crop_pct(0.95)
                .with_mean(vec![0.5; 3])
                .with_std(vec![0.5; 3])
        );
    }

    #[test]
    fn test_try_from_build_rejects_names_without_labels() {
        let dir = tempfile::tempdir().unwrap();
        let lemmas = dir.path().join("lemmas.txt");
        std::fs::write(&lemmas, "n1\tone\n").unwrap();
        let build = BuildRef::from(ResourceSet::given("mine", "checkpoint", &lemmas));
        ErrorMatcher::invalid_resource()
            .message_contains("binds label_names without labels")
            .assert_err(&ImageModelCard::try_from_build(
                &build,
                &loaded(&[(LABEL_NAMES, &lemmas)]),
            ));
    }
}
