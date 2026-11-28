//! # The classifier head
//!
//! [`ClassifierHead`]: what every image model does after its pool, timm's
//! `forward_head` past the pool: a norm on the pooled features (timm's
//! `fc_norm`), a dropout, and a classifier, none on a headless model; for a
//! distilled model (`DeiT`'s), a second classifier on its distillation token,
//! the two logits averaged. The pool is the model's own (a conv net's over
//! a map, a vision transformer's over tokens); the head reads its
//! `[batch, d_features]` output.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout,
        DropoutConfig,
        Linear,
        LinearConfig,
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::norm::{
        BunsenNormalization,
        BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    support::validators::try_probability,
};

/// [`ClassifierHead`]'s config: the pooled width, the classes, the norm, the
/// dropout, and whether it is distilled.
#[derive(Config, Debug)]
pub struct ClassifierHeadConfig {
    /// The width of the pooled features it reads.
    pub d_features: usize,

    /// The classes; 0 for a headless model, whose output is its pooled
    /// features (timm's identity head).
    pub num_classes: usize,

    /// The norm on the pooled features (timm's `fc_norm`), its width the
    /// features'; `None` for none.
    #[config(default = "None")]
    pub fc_norm: Option<BunsenNormalizationConfig>,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = 0.0)]
    pub drop_rate: f64,

    /// Whether a second classifier reads a distillation token (timm's
    /// `head_dist`).
    #[config(default = false)]
    pub distilled: bool,
}

impl ModuleInit<ClassifierHead> for ClassifierHeadConfig {
    /// Builds the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a zero
    /// width, a norm on a distilled head, or a dropout that is not a
    /// probability.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ClassifierHead> {
        if self.d_features == 0 {
            return Err(
                ConstraintError::zero_or_empty("ClassifierHeadConfig", "d_features").into(),
            );
        }
        if self.distilled && self.fc_norm.is_some() {
            return Err(ConstraintError::custom(
                "ClassifierHeadConfig",
                "fc_norm",
                "a distilled head has no norm on its pooled features",
            )
            .into());
        }
        try_probability(self.drop_rate)?;
        let linear = || LinearConfig::new(self.d_features, self.num_classes).init(device);
        Ok(ClassifierHead {
            d_features: self.d_features,
            fc_norm: self
                .fc_norm
                .clone()
                .map(|norm| norm.with_num_features(self.d_features).try_init(device))
                .transpose()?,
            drop: (self.drop_rate != 0.0).then(|| DropoutConfig::new(self.drop_rate).init()),
            fc: (self.num_classes > 0).then(linear),
            fc_dist: (self.num_classes > 0 && self.distilled).then(linear),
        })
    }
}

/// An image model's classifier head: a norm on the pooled features, a
/// dropout, then a linear layer; on a distilled model, a second linear layer
/// on the distillation token, the two logits averaged. Built by
/// [`ClassifierHeadConfig`].
///
/// The parameter paths are `fc`, `fc_norm` and `fc_dist`, under the model's
/// `head`.
#[derive(Module, Debug)]
pub struct ClassifierHead {
    /// The width of the pooled features it reads.
    pub d_features: usize,

    /// The norm on the pooled features (timm's `fc_norm`).
    pub fc_norm: Option<BunsenNormalization>,

    /// The dropout before the classifier, when its rate is not 0.
    pub drop: Option<Dropout>,

    /// The classifier (timm's `head`, a conv net's `fc`); `None` for a
    /// headless model.
    pub fc: Option<Linear>,

    /// A distilled head's classifier on the distillation token (timm's
    /// `head_dist`).
    pub fc_dist: Option<Linear>,
}

impl ClassifierHead {
    /// The classes; 0 for a headless model.
    pub fn num_classes(&self) -> usize {
        self.fc.as_ref().map_or(0, |fc| fc.weight.dims()[1])
    }

    /// The width of what [`forward`](Self::forward) returns: the classes, or
    /// the features for a headless model.
    pub fn d_output(&self) -> usize {
        match self.num_classes() {
            0 => self.d_features,
            classes => classes,
        }
    }

    /// Whether a second classifier reads a distillation token.
    pub fn is_distilled(&self) -> bool {
        self.fc_dist.is_some()
    }

    /// The pooled features through the norm: the classifier's input (timm's
    /// `forward_head(..., pre_logits=True)`, but for the dropout, a no-op in
    /// eval).
    ///
    /// # Arguments
    ///
    /// - `pooled`: `[batch, d_features]`.
    ///
    /// # Returns
    ///
    /// `[batch, d_features]`.
    pub fn pre_logits(
        &self,
        pooled: Tensor<2>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "d_features"],
            &pooled.dims(),
            &["batch"],
            &[("d_features", self.d_features)]
        );
        let x = match &self.fc_norm {
            Some(norm) => norm.forward(pooled),
            None => pooled,
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_features"],
            &x.dims(),
            &[("batch", batch), ("d_features", self.d_features)]
        );
        x
    }

    /// `fc`'s logits of `x`, through the dropout; `x` itself with no `fc`.
    fn classify(
        &self,
        fc: Option<&Linear>,
        x: Tensor<2>,
    ) -> Tensor<2> {
        let x = match &self.drop {
            Some(drop) => drop.forward(x),
            None => x,
        };
        match fc {
            Some(fc) => fc.forward(x),
            None => x,
        }
    }

    /// The logits of pooled features: [`pre_logits`](Self::pre_logits),
    /// the dropout, the classifier; the features for a headless model.
    ///
    /// # Arguments
    ///
    /// - `pooled`: `[batch, d_features]`.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`, or `[batch, d_features]` for a headless model.
    ///
    /// # Panics
    ///
    /// On a distilled head with classifiers, whose logits read two tokens
    /// ([`forward_distilled`](Self::forward_distilled)).
    pub fn forward(
        &self,
        pooled: Tensor<2>,
    ) -> Tensor<2> {
        assert!(
            !self.is_distilled(),
            "a distilled head's logits read its two tokens: forward_distilled"
        );
        #[cfg(any(debug_assertions, test))]
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "d_features"],
            &pooled.dims(),
            &["batch"],
            &[("d_features", self.d_features)]
        );
        let logits = self.classify(self.fc.as_ref(), self.pre_logits(pooled));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_output"],
            &logits.dims(),
            &[("batch", batch), ("d_output", self.d_output())]
        );
        logits
    }

    /// A distilled head's logits: the mean of the classifier over the class
    /// token and the distillation classifier over the distillation token,
    /// each through the dropout (timm's `VisionTransformerDistilled`).
    ///
    /// # Arguments
    ///
    /// - `class`, `dist`: `[batch, d_features]`, the two tokens.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`.
    ///
    /// # Panics
    ///
    /// On a head that is not distilled, or is headless.
    pub fn forward_distilled(
        &self,
        class: Tensor<2>,
        dist: Tensor<2>,
    ) -> Tensor<2> {
        let (Some(fc), Some(fc_dist)) = (&self.fc, &self.fc_dist) else {
            panic!("forward_distilled: a distilled head with classifiers");
        };
        #[cfg(any(debug_assertions, test))]
        let [batch] = {
            let bindings = [("d_features", self.d_features)];
            let [batch] = crate::contracts::unpack_shape_contract!(
                ["batch", "d_features"],
                &class.dims(),
                &["batch"],
                &bindings
            );
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "d_features"],
                &dist.dims(),
                &[("batch", batch), ("d_features", self.d_features)]
            );
            [batch]
        };
        let logits = (self.classify(Some(fc), class) + self.classify(Some(fc_dist), dist)) * 0.5;
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "classes"],
            &logits.dims(),
            &[("batch", batch), ("classes", self.num_classes())]
        );
        logits
    }

    /// The head re-initialized for `num_classes` classes (0: headless), on
    /// the same norm; a new classifier is on `device`.
    pub fn with_classes(
        self,
        num_classes: usize,
        device: &Device,
    ) -> Self {
        let distilled = self.is_distilled();
        let linear = || LinearConfig::new(self.d_features, num_classes).init(device);
        Self {
            fc: (num_classes > 0).then(linear),
            fc_dist: (num_classes > 0 && distilled).then(linear),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            cpu_device,
            tensor_to_vec_as,
        },
    };

    /// A `[2, 1]` linear layer of weights `w` and bias `b`.
    fn linear(
        w: [f32; 2],
        b: f32,
        device: &Device,
    ) -> Linear {
        let mut fc = LinearConfig::new(2, 1).init(device);
        fc.weight = burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(w.to_vec(), [2, 1]),
            device,
        ));
        fc.bias = Some(burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(alloc::vec![b], [1]),
            device,
        )));
        fc
    }

    fn row(
        values: [f32; 2],
        device: &Device,
    ) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values.to_vec(), [1, 2]), device)
    }

    /// A headless head passes its features through; one with classes
    /// classifies them; re-heading builds a classifier.
    #[test]
    fn test_headless_and_with_classes() {
        let device = cpu_device();
        let head: ClassifierHead = ClassifierHeadConfig::new(2, 0).init(&device);
        assert_eq!(head.num_classes(), 0);
        assert_eq!(
            tensor_to_vec_as::<f32, _>(&head.forward(row([1., 2.], &device))),
            [1., 2.]
        );
        let head = head.with_classes(3, &device);
        assert_eq!(head.num_classes(), 3);
        assert_eq!(head.forward(row([1., 2.], &device)).dims(), [1, 3]);
    }

    /// A distilled head's logits are the mean of its two classifiers'.
    #[test]
    fn test_distilled_logits() {
        let device = cpu_device();
        let mut head: ClassifierHead = ClassifierHeadConfig::new(2, 1)
            .with_distilled(true)
            .init(&device);
        assert!(head.is_distilled());
        head.fc = Some(linear([1., 0.], 0., &device));
        head.fc_dist = Some(linear([0., 1.], 1., &device));
        // (1 + (6 + 1)) / 2
        assert_eq!(
            tensor_to_vec_as::<f32, _>(
                &head.forward_distilled(row([1., 2.], &device), row([5., 6.], &device))
            ),
            [4.]
        );
        let headless: ClassifierHead = ClassifierHeadConfig::new(2, 0)
            .with_distilled(true)
            .init(&device);
        assert!(!headless.is_distilled());
        assert!(
            ClassifierHeadConfig::new(2, 4)
                .with_distilled(true)
                .init(&device)
                .with_classes(5, &device)
                .is_distilled()
        );
    }

    #[test]
    fn test_illegal_heads() {
        let device = cpu_device();
        let illegal = |config: ClassifierHeadConfig| {
            ErrorMatcher::illegal()
                .assert_err(&ModuleInit::<ClassifierHead>::try_init(&config, &device));
        };
        illegal(ClassifierHeadConfig::new(0, 3));
        illegal(ClassifierHeadConfig::new(2, 3).with_drop_rate(1.5));
        illegal(
            ClassifierHeadConfig::new(2, 3)
                .with_distilled(true)
                .with_fc_norm(Some(BunsenNormalizationConfig::Layer(
                    crate::blocks::norm::BunsenLayerNormConfig::new(0),
                ))),
        );
    }
}
