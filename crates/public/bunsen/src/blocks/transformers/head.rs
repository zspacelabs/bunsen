//! The language-model head.

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::{
        DType,
        Device,
        FloatDType,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::embedding::unembed,
};

/// Common meta for [`LmHead`] and [`LmHeadConfig`].
pub trait LmHeadMeta {
    /// The logits' width: the vocabulary, after any slice.
    fn n_vocab(&self) -> usize;

    /// Whether the head is the token embedding's table, transposed.
    fn is_tied(&self) -> bool;
}

/// Config for [`LmHead`].
#[derive(Config, Debug)]
pub struct LmHeadConfig {
    /// The hidden width.
    pub d_model: usize,

    /// The projection's rows: the vocabulary, as padded in the table.
    pub n_rows: usize,

    /// Whether the head is the token embedding's table (no weights of its
    /// own), or a projection of its own.
    #[config(default = true)]
    pub tied: bool,

    /// Whether a projection of its own carries a bias (nanochat's may).
    #[config(default = false)]
    pub bias: bool,

    /// A final soft-cap, `c · tanh(logits / c)` (nanochat's 15, Gemma 2's
    /// 30); `None`, none.
    #[config(default = "None")]
    pub softcap: Option<f64>,

    /// The logits kept: the first `vocab` of the `n_rows` (a vocabulary
    /// padded for the table's sake); `None`, all of them.
    #[config(default = "None")]
    pub vocab: Option<usize>,

    /// The dtype the logits leave in; `None`, the hidden state's.
    #[config(default = "None")]
    pub logits_dtype: Option<FloatDType>,
}

impl LmHeadMeta for LmHeadConfig {
    fn n_vocab(&self) -> usize {
        self.vocab.unwrap_or(self.n_rows)
    }

    fn is_tied(&self) -> bool {
        self.tied
    }
}

impl ModuleInit<LmHead> for LmHeadConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LmHead> {
        if let Some(vocab) = self.vocab
            && vocab > self.n_rows
        {
            return Err(ConstraintError::new(
                "LmHeadConfig",
                "",
                Rule::Relation {
                    lhs: ("vocab".into(), vocab.to_string()),
                    op: "<=",
                    rhs: ("n_rows".into(), self.n_rows.to_string()),
                },
            )
            .into());
        }
        Ok(LmHead {
            proj: (!self.tied).then(|| {
                LinearConfig::new(self.d_model, self.n_rows)
                    .with_bias(self.bias)
                    .init(device)
            }),
            n_rows: self.n_rows,
            softcap: self.softcap,
            vocab: self.vocab,
            logits_dtype: self.logits_dtype,
        })
    }
}

/// A language model's head: the hidden state to logits over the vocabulary.
///
/// - The projection is its own ([`proj`](Self::proj), a `Linear`), or tied: the
///   token embedding's table, transposed, which [`forward`](Self::forward) is
///   handed.
/// - The logits are then sliced to the vocabulary (a table padded past it),
///   soft-capped (`c · tanh(logits / c)`), and cast to the logits' dtype, each
///   as configured.
///
/// Built by [`LmHeadConfig`].
#[derive(Module, Debug)]
pub struct LmHead {
    /// The projection of its own; `None`, tied.
    pub proj: Option<Linear>,

    /// The projection's rows.
    #[module(skip)]
    pub n_rows: usize,

    /// The final soft-cap.
    #[module(skip)]
    pub softcap: Option<f64>,

    /// The logits kept, of the rows.
    #[module(skip)]
    pub vocab: Option<usize>,

    /// The logits' dtype.
    #[module(skip)]
    pub logits_dtype: Option<FloatDType>,
}

impl LmHeadMeta for LmHead {
    fn n_vocab(&self) -> usize {
        self.vocab.unwrap_or(self.n_rows)
    }

    fn is_tied(&self) -> bool {
        self.proj.is_none()
    }
}

impl LmHead {
    /// The logits.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, d_model]`.
    /// * `table` - the `[n_rows, d_model]` token table a tied head projects
    ///   onto; a head of its own ignores it.
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`.
    pub fn forward(
        &self,
        hidden: Tensor<3>,
        table: &Param<Tensor<2>>,
    ) -> Tensor<3> {
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &hidden.dims(),
            &["batch", "seq"],
        );
        let y = match &self.proj {
            Some(proj) => proj.forward(hidden),
            None => unembed(table.val(), hidden),
        };
        let y = match self.vocab {
            Some(vocab) if vocab < y.dims()[2] => y.slice_dim(2, 0..vocab),
            _ => y,
        };
        let y = match self.softcap {
            Some(cap) => y.div_scalar(cap).tanh().mul_scalar(cap),
            None => y,
        };
        let y = match self.logits_dtype {
            Some(dtype) => y.cast(DType::from(dtype)),
            None => y,
        };
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "n_vocab"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("n_vocab", self.n_vocab())]
        );
        y
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    fn table(device: &Device) -> Param<Tensor<2>> {
        Param::from_tensor(seeded_tensor::<2>(
            1,
            [12, 4],
            Distribution::Default,
            device,
        ))
    }

    fn hidden(device: &Device) -> Tensor<3> {
        seeded_tensor::<3>(2, [2, 3, 4], Distribution::Default, device)
    }

    /// A tied head projects onto the table: `hidden · tableᵀ`.
    #[test]
    fn test_tied() {
        let device = cpu_device();
        let head: LmHead = LmHeadConfig::new(4, 12).init(&device);
        assert!(head.is_tied() && head.proj.is_none());
        assert_eq!(head.n_vocab(), 12);
        let (table, hidden) = (table(&device), hidden(&device));
        assert_tensors_close(
            &head.forward(hidden.clone(), &table),
            &unembed(table.val(), hidden),
            Tolerance::default(),
        );
    }

    /// A head of its own is its projection, bias and all; the table is not
    /// read.
    #[test]
    fn test_own() {
        let device = cpu_device();
        let head: LmHead = LmHeadConfig::new(4, 12)
            .with_tied(false)
            .with_bias(true)
            .init(&device);
        let proj = head.proj.as_ref().expect("its own projection");
        assert!(proj.bias.is_some());
        let hidden = hidden(&device);
        assert_tensors_close(
            &head.forward(hidden.clone(), &table(&device)),
            &proj.forward(hidden),
            Tolerance::default(),
        );
    }

    /// The slice keeps the first `vocab` logits, then the soft-cap bounds
    /// them by `c`, then they leave in the logits' dtype.
    #[test]
    fn test_slice_softcap_and_dtype() {
        let device = cpu_device();
        let (table, hidden) = (table(&device), hidden(&device));
        let cap = 0.5;
        let head: LmHead = LmHeadConfig::new(4, 12)
            .with_vocab(Some(10))
            .with_softcap(Some(cap))
            .with_logits_dtype(Some(FloatDType::F32))
            .init(&device);
        assert_eq!(head.n_vocab(), 10);
        let logits = head.forward(hidden.clone(), &table);
        assert_eq!(logits.dims(), [2, 3, 10]);
        assert_eq!(logits.dtype(), DType::F32);
        let raw = unembed(table.val(), hidden).slice_dim(2, 0..10);
        assert!(
            raw.clone().abs().max().into_scalar::<f32>() > cap as f32,
            "the cap binds"
        );
        assert_tensors_close(
            &logits,
            &raw.div_scalar(cap).tanh().mul_scalar(cap),
            Tolerance::default(),
        );
    }

    /// A slice wider than the rows is refused.
    #[test]
    fn test_vocab_within_rows() {
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Illegal)
            .message_contains("vocab")
            .assert_err(
                &LmHeadConfig::new(4, 12)
                    .with_vocab(Some(13))
                    .try_init(&cpu_device()),
            );
    }
}
