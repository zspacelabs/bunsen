//! The classifier seam, and the top classes of its logits.

use alloc::{
    string::String,
    vec::Vec,
};

use burn::{
    Tensor,
    tensor::activation::softmax,
};

use crate::{
    blocks::{
        images::models::conv_net::{
            ConvNetMeta,
            ConvNetModel,
        },
        transformers::models::{
            staged_vision_encoder::{
                StagedVisionEncoderMeta,
                StagedVisionEncoderModel,
            },
            vision_encoder::{
                VisionEncoderMeta,
                VisionEncoderModel,
            },
        },
    },
    errors::{
        ConstraintError,
        Rule,
    },
    kits::images::support::{
        encoder::ImageEncoder,
        labels::ImageLabelSet,
    },
};

/// An image classifier: an [`ImageEncoder`] with a head that scores each
/// class.
///
/// Object-safe, as [`ImageEncoder`] is.
pub trait ImageClassifier: ImageEncoder {
    /// The number of classes the head scores.
    fn n_classes(&self) -> usize;

    /// The `[batch, n_classes]` logits of a normalised
    /// `[batch, channels, height, width]` batch.
    fn forward_logits(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2>;
}

/// The generic conv net as an image classifier: its head's logits
/// ([`ConvNetModel::forward`]).
impl ImageClassifier for ConvNetModel {
    fn n_classes(&self) -> usize {
        self.num_classes()
    }

    fn forward_logits(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        self.forward(images)
    }
}

/// The generic vision encoder as an image classifier: its head's logits
/// ([`VisionEncoderModel::forward`]). A headless model has 0 classes: its
/// output is its pooled features, not logits.
impl ImageClassifier for VisionEncoderModel {
    fn n_classes(&self) -> usize {
        self.num_classes()
    }

    fn forward_logits(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        self.forward(images)
    }
}

/// The generic staged vision encoder as an image classifier: its head's
/// logits ([`StagedVisionEncoderModel::forward`]).
impl ImageClassifier for StagedVisionEncoderModel {
    fn n_classes(&self) -> usize {
        self.num_classes()
    }

    fn forward_logits(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        self.forward(images)
    }
}

/// One of a row's top classes.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassPrediction {
    /// The class's index in the head.
    pub index: usize,

    /// The softmax probability of the class.
    pub probability: f32,

    /// The class's synset id, when the model has labels.
    pub synset: Option<String>,

    /// The class's readable name, when the model has labels: its lemmas, or
    /// its synset id where the labels have no lemmas.
    pub name: Option<String>,
}

/// The `k` most probable classes of each row of `[batch, n_classes]` logits,
/// most probable first, named by `labels` where it has them.
///
/// `k` is clamped to the class count.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
/// [`ConstraintError`](crate::errors::ConstraintError) on `labels`, when
/// `labels` does not have the logits' class count.
pub fn try_top_k(
    logits: Tensor<2>,
    k: usize,
    labels: Option<&ImageLabelSet>,
) -> crate::errors::BunsenResult<Vec<Vec<ClassPrediction>>> {
    let [batch, n_classes] = crate::contracts::unpack_shape_contract!(
        ["batch", "n_classes"],
        &logits,
        &["batch", "n_classes"],
    );
    if let Some(labels) = labels
        && labels.len() != n_classes
    {
        return Err(ConstraintError::new(
            "try_top_k",
            "labels",
            Rule::Relation {
                lhs: ("labels.len()".into(), labels.len().to_string()),
                op: "==",
                rhs: ("the logits' classes".into(), n_classes.to_string()),
            },
        )
        .into());
    }
    let k = k.min(n_classes);
    if k == 0 {
        return Ok((0..batch).map(|_| Vec::new()).collect());
    }
    let probabilities = softmax(logits.cast(burn::tensor::DType::F32), 1);
    let (values, indices) = probabilities.topk_with_indices(k, 1);
    let values = values
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap();
    let indices = indices
        .into_data()
        .convert::<i64>()
        .try_into_vec::<i64>()
        .unwrap();
    Ok((0..batch)
        .map(|row| {
            (0..k)
                .map(|j| {
                    let index = indices[row * k + j] as usize;
                    ClassPrediction {
                        index,
                        probability: values[row * k + j],
                        synset: labels.map(|l| l.synset(index).into()),
                        name: labels.map(|l| l.name(index).into()),
                    }
                })
                .collect()
        })
        .collect())
}

/// [`try_top_k`], panicking with its error's report.
///
/// # Panics
/// When `labels` does not have the logits' class count.
pub fn top_k(
    logits: Tensor<2>,
    k: usize,
    labels: Option<&ImageLabelSet>,
) -> Vec<Vec<ClassPrediction>> {
    use crate::errors::WithOkOrPanic;
    try_top_k(logits, k, labels).ok_or_panic()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::images::models::conv_net::{
            ConvNetContractConfig,
            ConvStageContractConfig,
            ConvStemContractConfig,
            ConvUnitKind,
        },
        burner::module::ModuleInit,
        kits::images::support::encoder::ImageFeatures,
        support::testing::{
            assert_scalar_close,
            cpu_device,
            expect_variant,
        },
    };

    /// The conv net through the seam: the stem's and the stages' maps at
    /// timm's strides, pooled features of the head's width, the head's
    /// classes, all through a `dyn ImageClassifier`.
    #[test]
    fn test_the_conv_net_is_an_image_classifier() {
        let device = cpu_device();
        let model: ConvNetModel = ConvNetContractConfig::new(
            ConvStemContractConfig::new(vec![8]),
            vec![
                ConvStageContractConfig::new(1, ConvUnitKind::Basic, 8, 8).with_stride(1),
                ConvStageContractConfig::new(1, ConvUnitKind::Basic, 16, 16),
            ],
            7,
        )
        .init(&device);
        let classifier: &dyn ImageClassifier = &model;
        let images = Tensor::<4>::zeros([2, 3, 32, 32], &device);

        assert_eq!(classifier.input_channels(), 3);
        assert_eq!(classifier.n_classes(), 7);
        assert_eq!(classifier.d_features(), 16);
        let stages: Vec<_> = classifier
            .forward_intermediates(images.clone())
            .iter()
            .map(|x| x.dims())
            .collect();
        assert_eq!(stages, [[2, 8, 16, 16], [2, 8, 8, 8], [2, 16, 4, 4]]);
        expect_variant!(
            let ImageFeatures::Spatial(features) = classifier.forward_features(images.clone()),
            "a conv net's features are spatial",
        );
        assert_eq!(features.dims(), [2, 16, 4, 4]);
        assert_eq!(classifier.forward_pooled(images.clone()).dims(), [2, 16]);
        classifier
            .forward_logits(images.clone())
            .to_data()
            .assert_eq(&model.forward(images).to_data(), true);
    }

    /// The top classes, most probable first, with their softmax
    /// probabilities and their labels.
    #[test]
    fn test_top_k_orders_and_names() {
        let device = cpu_device();
        let logits = Tensor::<2>::from_floats([[0.0, 2.0, 1.0], [3.0, 0.0, 0.0]], &device);
        let labels =
            ImageLabelSet::try_parse("n0\nn1\nn2", Some("n0\tzero\nn1\tone\nn2\ttwo")).unwrap();
        let top = top_k(logits, 2, Some(&labels));

        let softmax = |xs: [f32; 3]| {
            let total: f32 = xs.iter().map(|x| x.exp()).sum();
            xs.map(|x| x.exp() / total)
        };
        let (p0, p1) = (softmax([0.0, 2.0, 1.0]), softmax([3.0, 0.0, 0.0]));
        assert_eq!(top[0].iter().map(|p| p.index).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(top[1][0].index, 0);
        assert_eq!(top[0][0].name.as_deref(), Some("one"));
        assert_eq!(top[0][1].synset.as_deref(), Some("n2"));
        assert_scalar_close(top[0][0].probability, p0[1], 1e-6);
        assert_scalar_close(top[1][0].probability, p1[0], 1e-6);
    }

    /// `k` past the class count is the class count; no labels, no names.
    #[test]
    fn test_top_k_clamps_k() {
        let device = cpu_device();
        let logits = Tensor::<2>::from_floats([[1.0, 0.0]], &device);
        let top = top_k(logits, 5, None);
        assert_eq!(top[0].len(), 2);
        assert_eq!(top[0][0].name, None);
    }
}
