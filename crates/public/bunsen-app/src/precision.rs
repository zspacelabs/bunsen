//! The precision flags: `--min-precision`, a floor under the dtypes a loaded
//! model is placed in, and `--init-dtype`, the dtype a program creates its own
//! tensors in.
//!
//! Neither sets the device's default dtypes, which stay f32, and neither can
//! narrow a loaded model: [`MinPrecisionArgs`] only ever adds precision, and
//! [`InitDTypeArgs`] names the dtype of what the program makes itself (a
//! simulation's state, a model trained from scratch, a new head), never of
//! what it loads.
//!
//! A program that both loads and creates takes both, and refuses an
//! `--init-dtype` narrower than its `--min-precision`
//! ([`InitDTypeArgs::resolve_within`]): the new parts would come out narrower
//! than the floor the loaded ones were held to.

use bunsen::{
    burner::tensor::float_width,
    data::formats::LoadPolicy,
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use burn::tensor::{
    Device,
    FloatDType,
};

/// `--min-precision`: the narrowest float a loaded model is placed in.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinPrecision {
    /// 32-bit float.
    F32,

    /// 64-bit float.
    F64,
}

impl From<MinPrecision> for FloatDType {
    fn from(min: MinPrecision) -> Self {
        match min {
            MinPrecision::F32 => FloatDType::F32,
            MinPrecision::F64 => FloatDType::F64,
        }
    }
}

/// The load floor's flag, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct MinPrecisionArgs {
    /// The narrowest float the model's weights are placed in.
    ///
    /// It only adds precision: a weight stored narrower is widened to it.
    /// Without it, a half dtype is kept where the device accelerates it, else
    /// f32 (`LoadPolicy::performance`).
    #[arg(long, value_enum)]
    pub min_precision: Option<MinPrecision>,
}

impl MinPrecisionArgs {
    /// The load policy the flag asks for: [`LoadPolicy::min_precision`] with
    /// it, [`LoadPolicy::performance`] without.
    pub fn load_policy(&self) -> LoadPolicy {
        match self.min_precision {
            Some(min) => LoadPolicy::min_precision(min.into()),
            None => LoadPolicy::performance(),
        }
    }
}

/// `--init-dtype`: an exact float dtype for what a program creates.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitDType {
    /// 64-bit float.
    F64,

    /// 32-bit float.
    F32,

    /// 16-bit float.
    F16,

    /// 16-bit brain float.
    Bf16,
}

impl From<InitDType> for FloatDType {
    fn from(init: InitDType) -> Self {
        match init {
            InitDType::F64 => FloatDType::F64,
            InitDType::F32 => FloatDType::F32,
            InitDType::F16 => FloatDType::F16,
            InitDType::Bf16 => FloatDType::BF16,
        }
    }
}

/// The creation dtype's flag, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct InitDTypeArgs {
    /// The float dtype the program creates its own tensors and new modules in.
    ///
    /// The program's own default when omitted (f32 where the device lacks
    /// it). Never the dtype a loaded model is placed in (`--min-precision` is
    /// that floor).
    #[arg(long, value_enum)]
    pub init_dtype: Option<InitDType>,
}

impl InitDTypeArgs {
    /// The creation dtype: the flag's, else the program's `default` where
    /// `device` supports it, else f32.
    ///
    /// # Errors
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) for a
    /// flag naming a dtype `device` does not support.
    pub fn resolve(
        &self,
        default: FloatDType,
        device: &Device,
    ) -> BunsenResult<FloatDType> {
        let supports = |dtype: FloatDType| device.supports_dtype(burn::tensor::DType::from(dtype));
        match self.init_dtype.map(FloatDType::from) {
            Some(dtype) if !supports(dtype) => Err(BunsenError::unsupported(format!(
                "the device does not support {dtype:?} (`--init-dtype`)"
            ))),
            Some(dtype) => Ok(dtype),
            None if supports(default) => Ok(default),
            None => Ok(FloatDType::F32),
        }
    }

    /// [`resolve`](Self::resolve), for a program that also loads under `min`:
    /// the creation dtype may not be narrower than the load floor.
    ///
    /// # Errors
    /// As [`resolve`](Self::resolve);
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) for a creation
    /// dtype narrower than `--min-precision`.
    pub fn resolve_within(
        &self,
        default: FloatDType,
        device: &Device,
        min: &MinPrecisionArgs,
    ) -> BunsenResult<FloatDType> {
        let dtype = self.resolve(default, device)?;
        if let Some(floor) = min.min_precision.map(FloatDType::from)
            && float_width(dtype) < float_width(floor)
        {
            return Err(BunsenError::illegal(format!(
                "`--init-dtype` {dtype:?} is narrower than `--min-precision` {floor:?}: \
                 the new parts would be narrower than the loaded ones"
            )));
        }
        Ok(dtype)
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::BunsenErrorKind;
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        min: MinPrecisionArgs,

        #[command(flatten)]
        init: InitDTypeArgs,
    }

    fn cli(argv: &[&str]) -> Cli {
        Cli::try_parse_from([&["cli"], argv].concat()).unwrap()
    }

    /// The floor is a load policy; without one, the performance policy.
    #[test]
    fn test_min_precision() {
        assert_eq!(cli(&[]).min.load_policy(), LoadPolicy::performance());
        assert_eq!(
            cli(&["--min-precision", "f32"]).min.load_policy(),
            LoadPolicy::min_precision(FloatDType::F32)
        );
        assert!(
            Cli::try_parse_from(["cli", "--min-precision", "bf16"]).is_err(),
            "a floor only adds precision"
        );
    }

    /// The creation dtype is the flag's or the program's; never narrower than
    /// the floor when there is one.
    #[test]
    fn test_init_dtype() {
        let device = Device::default();
        assert_eq!(
            cli(&[]).init.resolve(FloatDType::F32, &device).unwrap(),
            FloatDType::F32
        );
        let args = cli(&["--init-dtype", "f64", "--min-precision", "f32"]);
        assert_eq!(
            args.init
                .resolve_within(FloatDType::F32, &device, &args.min)
                .unwrap(),
            FloatDType::F64
        );

        let args = cli(&["--init-dtype", "f16", "--min-precision", "f32"]);
        let narrower = args
            .init
            .resolve_within(FloatDType::F32, &device, &args.min);
        if device.supports_dtype(burn::tensor::DType::from(FloatDType::F16)) {
            assert_eq!(narrower.unwrap_err().kind(), BunsenErrorKind::Illegal);
        } else {
            assert_eq!(narrower.unwrap_err().kind(), BunsenErrorKind::Unsupported);
        }
    }
}
