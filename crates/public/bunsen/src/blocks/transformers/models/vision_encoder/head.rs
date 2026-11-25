//! The token head: a pool, a norm, a dropout, a classifier.

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
    tensor::{
        Device,
        s,
    },
};

use crate::{
    blocks::norm::{
        BunsenNormalization,
        BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
    },
    support::validators::try_probability,
};

/// The pool over a vision transformer's tokens, timm's `global_pool_nlc`
/// kinds: the config of a [`TokenPool`].
///
/// The reducing kinds take the patch tokens alone, the prefix tokens left
/// out.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum TokenPoolConfig {
    /// The class token, the sequence's first (timm's `token`).
    Token,

    /// The mean of the patch tokens (timm's `avg`).
    Avg,

    /// The max of the patch tokens (timm's `max`).
    Max,

    /// The mean of the two (timm's `avgmax`).
    AvgMax,

    /// The class and distillation tokens, the first two: their mean pooled,
    /// each through its own classifier, the two logits averaged (timm's
    /// `VisionTransformerDistilled`, `DeiT`'s distilled models).
    Distilled,
}

impl TokenPoolConfig {
    /// The kind timm names `name` (`token`, `avg`, `max`, `avgmax`); `None`
    /// for a kind the head doesn't build (`map`, `prr`, `''`).
    pub fn from_timm(name: &str) -> Option<Self> {
        Some(match name {
            "token" => Self::Token,
            "avg" => Self::Avg,
            "max" => Self::Max,
            "avgmax" => Self::AvgMax,
            _ => return None,
        })
    }

    /// Whether timm normalises after this pool rather than before it: its
    /// `fc_norm` default, on for the reducing pools.
    pub fn timm_fc_norm(self) -> bool {
        matches!(self, Self::Avg | Self::Max | Self::AvgMax)
    }

    /// The prefix tokens the pool reads: the class token for
    /// [`Token`](Self::Token), it and the distillation token for
    /// [`Distilled`](Self::Distilled), none for the reducing pools.
    pub fn prefix_read(self) -> usize {
        match self {
            Self::Token => 1,
            Self::Distilled => 2,
            Self::Avg | Self::Max | Self::AvgMax => 0,
        }
    }
}

/// A pool over the tokens; built by [`VisionHeadConfig`]. No kind has
/// parameters.
#[derive(Module, Debug)]
pub struct TokenPool {
    /// The kind.
    #[module(skip)]
    pub kind: TokenPoolConfig,

    /// The prefix tokens a reducing pool leaves out.
    pub n_prefix: usize,
}

impl TokenPool {
    /// The pooled tokens.
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, n_prefix + patches, d]`.
    ///
    /// # Returns
    ///
    /// `[batch, d]`.
    pub fn forward(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<2> {
        let n = tokens.dims()[1];
        let patches = || tokens.clone().slice(s![.., self.n_prefix..n]);
        let pooled: Tensor<3> = match self.kind {
            TokenPoolConfig::Token => tokens.clone().slice(s![.., 0..1]),
            TokenPoolConfig::Distilled => tokens.clone().slice(s![.., 0..2]).mean_dim(1),
            TokenPoolConfig::Avg => patches().mean_dim(1),
            TokenPoolConfig::Max => patches().max_dim(1),
            TokenPoolConfig::AvgMax => (patches().max_dim(1) + patches().mean_dim(1)) * 0.5,
        };
        pooled.squeeze_dim(1)
    }
}

/// [`VisionHead`]'s config: the pool, the norm after it, the dropout, and
/// the classes.
#[derive(Config, Debug)]
pub struct VisionHeadConfig {
    /// The token width.
    pub d_model: usize,

    /// The classes; 0 for a headless model, whose output is its pooled
    /// features (timm's identity head).
    pub num_classes: usize,

    /// The prefix tokens before the patch tokens.
    pub n_prefix: usize,

    /// The pool.
    #[config(default = "TokenPoolConfig::Token")]
    pub pool: TokenPoolConfig,

    /// The norm after the pool (timm's `fc_norm`), its width the token's;
    /// `None` for none.
    #[config(default = "None")]
    pub fc_norm: Option<BunsenNormalizationConfig>,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = "0.0")]
    pub drop_rate: f64,
}

impl ModuleInit<VisionHead> for VisionHeadConfig {
    /// Builds the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a zero
    /// width, a pool reading more prefix tokens than there are, a norm after
    /// a distilled pool, or a dropout that is not a probability.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VisionHead> {
        if self.d_model == 0 {
            return Err(ConstraintError::zero_or_empty("VisionHeadConfig", "d_model").into());
        }
        let read = self.pool.prefix_read();
        if read > self.n_prefix {
            return Err(BunsenError::illegal(alloc::format!(
                "a {:?} pool reads {read} prefix tokens: the head has {}",
                self.pool,
                self.n_prefix
            )));
        }
        if self.pool == TokenPoolConfig::Distilled && self.fc_norm.is_some() {
            return Err(BunsenError::illegal(
                "a distilled head has no norm after its pool",
            ));
        }
        try_probability(self.drop_rate)?;
        Ok(VisionHead {
            d_model: self.d_model,
            pool: TokenPool {
                kind: self.pool,
                n_prefix: self.n_prefix,
            },
            fc_norm: self
                .fc_norm
                .clone()
                .map(|norm| norm.with_num_features(self.d_model).try_init(device))
                .transpose()?,
            drop: (self.drop_rate != 0.0).then(|| DropoutConfig::new(self.drop_rate).init()),
            fc: (self.num_classes > 0)
                .then(|| LinearConfig::new(self.d_model, self.num_classes).init(device)),
            fc_dist: (self.num_classes > 0 && self.pool == TokenPoolConfig::Distilled)
                .then(|| LinearConfig::new(self.d_model, self.num_classes).init(device)),
        })
    }
}

/// A vision transformer's head: a pool over the tokens, a norm, a dropout,
/// then a linear layer, timm's `VisionTransformer.forward_head`; for a
/// [`Distilled`](TokenPoolConfig::Distilled) pool, a classifier on each of
/// the class and distillation tokens, averaged. Built by
/// [`VisionHeadConfig`].
#[derive(Module, Debug)]
pub struct VisionHead {
    /// The token width: the pooled features'.
    pub d_model: usize,

    /// The pool.
    pub pool: TokenPool,

    /// The norm after the pool (timm's `fc_norm`).
    pub fc_norm: Option<BunsenNormalization>,

    /// The dropout before the classifier, when its rate is not 0.
    pub drop: Option<Dropout>,

    /// The classifier (timm's `head`); `None` for a headless model.
    pub fc: Option<Linear>,

    /// A distilled head's classifier on the distillation token (timm's
    /// `head_dist`).
    pub fc_dist: Option<Linear>,
}

impl VisionHead {
    /// The classes; 0 for a headless model.
    pub fn num_classes(&self) -> usize {
        self.fc.as_ref().map_or(0, |fc| fc.weight.dims()[1])
    }

    /// The pooled features: the pool, then the norm (timm's
    /// `forward_head(..., pre_logits=True)`, but for the dropout, which is a
    /// no-op in eval).
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, n_prefix + patches, d]`.
    ///
    /// # Returns
    ///
    /// `[batch, d]`.
    pub fn pool(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<2> {
        let pooled = self.pool.forward(tokens);
        match &self.fc_norm {
            Some(norm) => norm.forward(pooled),
            None => pooled,
        }
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

    /// The logits: the classifier over the pooled features, through the
    /// dropout; for a distilled head, the mean of the classifier over the
    /// class token and the distillation classifier over the distillation
    /// token; the pooled features for a headless model.
    ///
    /// # Arguments
    ///
    /// - `tokens`: `[batch, n_prefix + patches, d]`.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`, or `[batch, d]` for a headless model.
    pub fn forward(
        &self,
        tokens: Tensor<3>,
    ) -> Tensor<2> {
        match (&self.fc, &self.fc_dist) {
            (Some(fc), Some(fc_dist)) => {
                let token = |i: usize| tokens.clone().slice(s![.., i..i + 1]).squeeze_dim::<2>(1);
                (self.classify(Some(fc), token(0)) + self.classify(Some(fc_dist), token(1))) * 0.5
            }
            (fc, _) => self.classify(fc.as_ref(), self.pool(tokens)),
        }
    }

    /// The head re-initialized for `num_classes` classes (0: headless), on
    /// the same pool and norm; a new classifier is on `device`.
    pub fn with_classes(
        self,
        num_classes: usize,
        device: &Device,
    ) -> Self {
        let linear = || LinearConfig::new(self.d_model, num_classes).init(device);
        Self {
            fc: (num_classes > 0).then(linear),
            fc_dist: (num_classes > 0 && self.pool.kind == TokenPoolConfig::Distilled).then(linear),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    /// Each pool over `[1, 3, 2]` tokens, one prefix token `(9, 9)` and the
    /// patches `(1, 4)`, `(3, 2)`.
    #[test]
    fn test_pools() {
        let device = cpu_device();
        let tokens = Tensor::<3>::from_data(
            TensorData::new(alloc::vec![9f32, 9., 1., 4., 3., 2.], [1, 3, 2]),
            &device,
        );
        for (kind, want) in [
            (TokenPoolConfig::Token, [9f32, 9.]),
            (TokenPoolConfig::Avg, [2., 3.]),
            (TokenPoolConfig::Max, [3., 4.]),
            (TokenPoolConfig::AvgMax, [2.5, 3.5]),
        ] {
            let pool = TokenPool { kind, n_prefix: 1 };
            let got = pool.forward(tokens.clone()).into_data();
            assert_eq!(got.try_to_vec::<f32>().unwrap(), want, "{kind:?}");
        }
        assert_eq!(
            TokenPoolConfig::from_timm("avg"),
            Some(TokenPoolConfig::Avg)
        );
        assert_eq!(TokenPoolConfig::from_timm("map"), None);
    }

    /// The distilled pool is the mean of the first two tokens; its logits the
    /// mean of its two classifiers'.
    #[test]
    fn test_distilled_head() {
        let device = cpu_device();
        let tokens = Tensor::<3>::from_data(
            TensorData::new(alloc::vec![1f32, 2., 5., 6., 9., 9.], [1, 3, 2]),
            &device,
        );
        let mut head: VisionHead = VisionHeadConfig::new(2, 1, 2)
            .with_pool(TokenPoolConfig::Distilled)
            .init(&device);
        assert_eq!(
            head.pool(tokens.clone())
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [3., 4.]
        );
        let linear = |w: [f32; 2], b: f32| {
            let mut fc = LinearConfig::new(2, 1).init(&device);
            fc.weight = burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::new(w.to_vec(), [2, 1]),
                &device,
            ));
            fc.bias = Some(burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::new(alloc::vec![b], [1]),
                &device,
            )));
            fc
        };
        head.fc = Some(linear([1., 0.], 0.));
        head.fc_dist = Some(linear([0., 1.], 1.));
        // (1 + (6 + 1)) / 2
        assert_eq!(
            head.forward(tokens)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [4.]
        );
        let headless: VisionHead = VisionHeadConfig::new(2, 0, 2)
            .with_pool(TokenPoolConfig::Distilled)
            .init(&device);
        assert!(headless.fc_dist.is_none());
    }

    /// A headless head passes its pooled features through; a class-token
    /// pool needs a prefix.
    #[test]
    fn test_headless_and_illegal_heads() {
        let device = cpu_device();
        let head: VisionHead = VisionHeadConfig::new(2, 0, 1).init(&device);
        assert_eq!(head.num_classes(), 0);
        let tokens = Tensor::<3>::from_data(
            TensorData::new(alloc::vec![1f32, 2., 5., 6.], [1, 2, 2]),
            &device,
        );
        assert_eq!(
            head.forward(tokens)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [1., 2.]
        );
        let head: VisionHead = VisionHeadConfig::new(2, 5, 1).init(&device);
        assert_eq!(head.num_classes(), 5);
        assert_eq!(head.with_classes(3, &device).num_classes(), 3);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("reads 1 prefix tokens: the head has 0")
            .assert_err(&ModuleInit::<VisionHead>::try_init(
                &VisionHeadConfig::new(2, 5, 0),
                &device,
            ));
    }
}
