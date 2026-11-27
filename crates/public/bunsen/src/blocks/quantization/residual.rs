//! [`ResidualVectorQuantizer`] and [`SplitResidualVectorQuantizer`], to decode.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    nn::{
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::*,
};

use super::{
    EuclideanCodebook,
    EuclideanCodebookConfig,
    EuclideanCodebookMeta,
};
use crate::{
    burner::module::ModuleInit,
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// Common meta for [`ResidualVectorQuantizer`] and
/// [`ResidualVectorQuantizerConfig`].
pub trait ResidualVectorQuantizerMeta {
    /// The codebooks, `K`.
    fn codebooks(&self) -> usize;

    /// Each codebook's codes, `S`.
    fn size(&self) -> usize;

    /// The codebooks' vectors' width, `d`.
    fn dim(&self) -> usize;

    /// The latent's width: the projection's, or `d` without one.
    fn output_dim(&self) -> usize;
}

/// Config for [`ResidualVectorQuantizer`].
#[derive(Config, Debug)]
pub struct ResidualVectorQuantizerConfig {
    /// The codebooks, `K`, at least 1.
    pub codebooks: usize,

    /// Each codebook's codes, `S`.
    pub size: usize,

    /// The codebooks' vectors' width, `d`.
    pub dim: usize,

    /// The projection's width, a pointwise convolution without a bias from
    /// `d`; `None`, no projection.
    #[config(default = "None")]
    pub output_dim: Option<usize>,

    /// The floor each code's usage is clamped to before it divides.
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl ResidualVectorQuantizerMeta for ResidualVectorQuantizerConfig {
    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn size(&self) -> usize {
        self.size
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim.unwrap_or(self.dim)
    }
}

/// The quantizer; its codebooks as [`EuclideanCodebookConfig`] builds them.
impl ModuleInit<ResidualVectorQuantizer> for ResidualVectorQuantizerConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no codebooks.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ResidualVectorQuantizer> {
        if self.codebooks == 0 {
            return Err(BunsenError::illegal(
                "a residual quantizer of no codebooks; at least one",
            ));
        }
        let codebook = EuclideanCodebookConfig::new(self.size, self.dim).with_epsilon(self.epsilon);
        Ok(ResidualVectorQuantizer {
            codebooks: (0..self.codebooks)
                .map(|_| codebook.try_init(device))
                .collect::<BunsenResult<_>>()?,
            output_proj: self.output_dim.map(|out| {
                Conv1dConfig::new(self.dim, out, 1)
                    .with_bias(false)
                    .with_padding(PaddingConfig1d::Valid)
                    .init(device)
            }),
        })
    }
}

/// A residual vector quantizer's decode: codebooks in sequence, each having
/// coded what the ones before left, so a frame's latent is the sum of its
/// codes' vectors, then projected (a pointwise convolution without a bias):
///
/// ```text
/// latent[:, :, t] = output_proj(Σ_k codebooks[k].vector[codes[:, k, t]])
/// ```
///
/// A decode may take fewer codes than codebooks, the first `K'`: the coarser
/// latent a codec's lower bitrate reads.
///
/// Built by [`ResidualVectorQuantizerConfig`].
#[derive(Module, Debug)]
pub struct ResidualVectorQuantizer {
    /// The codebooks, in order.
    pub codebooks: Vec<EuclideanCodebook>,

    /// The projection, `[out, d, 1]`, when there is one.
    pub output_proj: Option<Conv1d>,
}

impl ResidualVectorQuantizerMeta for ResidualVectorQuantizer {
    fn codebooks(&self) -> usize {
        self.codebooks.len()
    }

    fn size(&self) -> usize {
        self.codebooks[0].size()
    }

    fn dim(&self) -> usize {
        self.codebooks[0].dim()
    }

    fn output_dim(&self) -> usize {
        match &self.output_proj {
            Some(proj) => proj.weight.dims()[0],
            None => self.dim(),
        }
    }
}

impl ResidualVectorQuantizer {
    /// Codes to the latent they stand for.
    ///
    /// # Arguments
    /// - `codes`: `[B, K', T]`, `1 ≤ K' ≤ K`, codebook `k`'s in `0..S`.
    ///
    /// # Returns
    /// `[B, D, T]`, `D` the
    /// [`output_dim`](ResidualVectorQuantizerMeta::output_dim).
    pub fn decode(
        &self,
        codes: Tensor<3, Int>,
    ) -> Tensor<3> {
        let [b, k, t] = unpack_shape_contract!(["B", "K", "T"], &codes.dims(), &["B", "K", "T"]);
        assert!(
            (1..=self.codebooks()).contains(&k),
            "codes of 1..={} codebooks; got {k}",
            self.codebooks()
        );
        let latent = self.codebooks[..k]
            .iter()
            .enumerate()
            .map(|(i, codebook)| {
                codebook.decode(codes.clone().slice(s![.., i..=i, ..]).reshape([b, t]))
            })
            .reduce(|sum, vectors| sum + vectors)
            .expect("at least one codebook");
        let latent = match &self.output_proj {
            Some(proj) => proj.forward(latent),
            None => latent,
        };
        assert_shape_contract_periodically!(
            ["B", "D", "T"],
            &latent.dims(),
            &[("B", b), ("D", self.output_dim()), ("T", t)]
        );
        latent
    }
}

/// Common meta for [`SplitResidualVectorQuantizer`] and
/// [`SplitResidualVectorQuantizerConfig`].
pub trait SplitResidualVectorQuantizerMeta {
    /// The codebooks in all, `K`.
    fn codebooks(&self) -> usize;

    /// The first codebooks, apart from the rest.
    fn first_codebooks(&self) -> usize;

    /// The latent's width.
    fn output_dim(&self) -> usize;
}

/// Config for [`SplitResidualVectorQuantizer`].
#[derive(Config, Debug)]
pub struct SplitResidualVectorQuantizerConfig {
    /// The codebooks in all, `K`; more than
    /// [`first_codebooks`](Self::first_codebooks).
    pub codebooks: usize,

    /// Each codebook's codes, `S`.
    pub size: usize,

    /// The codebooks' vectors' width, `d`.
    pub dim: usize,

    /// The latent's width: each half's projection's.
    pub output_dim: usize,

    /// The first codebooks, apart from the rest (Mimi's semantic ones).
    #[config(default = 1)]
    pub first_codebooks: usize,

    /// The floor each code's usage is clamped to before it divides.
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl SplitResidualVectorQuantizerMeta for SplitResidualVectorQuantizerConfig {
    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn first_codebooks(&self) -> usize {
        self.first_codebooks
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl ModuleInit<SplitResidualVectorQuantizer> for SplitResidualVectorQuantizerConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) unless
    /// `1 ≤ first_codebooks < codebooks`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SplitResidualVectorQuantizer> {
        if self.first_codebooks == 0 || self.first_codebooks >= self.codebooks {
            return Err(BunsenError::illegal(format!(
                "a split of {} codebooks after {}; the split is in 1..{}",
                self.codebooks, self.first_codebooks, self.codebooks
            )));
        }
        let half = |codebooks| {
            ResidualVectorQuantizerConfig::new(codebooks, self.size, self.dim)
                .with_output_dim(Some(self.output_dim))
                .with_epsilon(self.epsilon)
                .try_init(device)
        };
        Ok(SplitResidualVectorQuantizer {
            rvq_first: half(self.first_codebooks)?,
            rvq_rest: half(self.codebooks - self.first_codebooks)?,
        })
    }
}

/// A residual vector quantizer split in two (Mimi's and Qwen3-TTS's codecs'):
/// the first codebooks (Mimi's "semantic" ones, distilled toward a speech
/// model's features) and the rest (the "acoustic" ones), each a
/// [`ResidualVectorQuantizer`] with its own projection, their latents added.
///
/// The halves keep the checkpoints' names, `rvq_first` and `rvq_rest`.
///
/// Built by [`SplitResidualVectorQuantizerConfig`].
#[derive(Module, Debug)]
pub struct SplitResidualVectorQuantizer {
    /// The first codebooks.
    pub rvq_first: ResidualVectorQuantizer,

    /// The rest.
    pub rvq_rest: ResidualVectorQuantizer,
}

impl SplitResidualVectorQuantizerMeta for SplitResidualVectorQuantizer {
    fn codebooks(&self) -> usize {
        self.rvq_first.codebooks() + self.rvq_rest.codebooks()
    }

    fn first_codebooks(&self) -> usize {
        self.rvq_first.codebooks()
    }

    fn output_dim(&self) -> usize {
        self.rvq_first.output_dim()
    }
}

impl SplitResidualVectorQuantizer {
    /// Codes to the latent they stand for: the first half's latent, plus the
    /// rest's when the codes reach past the first codebooks.
    ///
    /// # Arguments
    /// - `codes`: `[B, K', T]`, `first_codebooks ≤ K' ≤ K`.
    ///
    /// # Returns
    /// `[B, D, T]`.
    pub fn decode(
        &self,
        codes: Tensor<3, Int>,
    ) -> Tensor<3> {
        let [k] = unpack_shape_contract!(["B", "K", "T"], &codes.dims(), &["K"]);
        let first = self.first_codebooks();
        assert!(
            (first..=self.codebooks()).contains(&k),
            "codes of {first}..={} codebooks; got {k}",
            self.codebooks()
        );
        let latent = self
            .rvq_first
            .decode(codes.clone().slice(s![.., 0..first, ..]));
        if k > first {
            latent + self.rvq_rest.decode(codes.slice(s![.., first..k, ..]))
        } else {
            latent
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    fn random_codebooks(
        rvq: &mut ResidualVectorQuantizer,
        seed: u64,
    ) {
        let device = cpu_device();
        for (i, codebook) in rvq.codebooks.iter_mut().enumerate() {
            let [s, d] = codebook.embedding_sum.dims();
            codebook.embedding_sum = Param::from_tensor(seeded_tensor(
                seed + i as u64,
                [s, d],
                Distribution::Normal(0.0, 1.0),
                &device,
            ));
            codebook.cluster_usage = Param::from_tensor(seeded_tensor(
                seed + 100 + i as u64,
                [s],
                Distribution::Uniform(0.5, 1.5),
                &device,
            ));
        }
    }

    /// The config and the module built from it answer the meta alike.
    #[test]
    fn test_meta() {
        let device = cpu_device();
        let config = ResidualVectorQuantizerConfig::new(3, 8, 4).with_output_dim(Some(6));
        let rvq = config.init(&device);
        for meta in [&config as &dyn ResidualVectorQuantizerMeta, &rvq] {
            assert_eq!(
                [meta.codebooks(), meta.size(), meta.dim(), meta.output_dim()],
                [3, 8, 4, 6]
            );
        }
        assert_eq!(
            ResidualVectorQuantizerConfig::new(2, 8, 4)
                .init(&device)
                .output_dim(),
            4
        );

        let config = SplitResidualVectorQuantizerConfig::new(4, 8, 4, 6);
        let split = config.init(&device);
        for meta in [&config as &dyn SplitResidualVectorQuantizerMeta, &split] {
            assert_eq!(
                [meta.codebooks(), meta.first_codebooks(), meta.output_dim()],
                [4, 1, 6]
            );
        }
    }

    /// The decode is the sum of the codebooks' vectors, then the projection;
    /// fewer codes read fewer codebooks.
    #[test]
    fn test_decode_is_the_sum() {
        let device = cpu_device();
        let mut rvq = ResidualVectorQuantizerConfig::new(3, 5, 2)
            .with_output_dim(Some(3))
            .init(&device);
        random_codebooks(&mut rvq, 1);
        let codes = Tensor::<3, Int>::from_ints([[[0, 4], [3, 1], [2, 2]]], &device);
        let vectors = |k: usize, codes: Tensor<3, Int>| {
            rvq.codebooks[k].decode(codes.slice(s![.., k..=k, ..]).reshape([1, 2]))
        };
        let proj = rvq.output_proj.as_ref().expect("a projection");
        let sum = vectors(0, codes.clone()) + vectors(1, codes.clone()) + vectors(2, codes.clone());
        assert_tensors_close(
            &rvq.decode(codes.clone()),
            &proj.forward(sum),
            Tolerance::default(),
        );
        let coarse = vectors(0, codes.clone()) + vectors(1, codes.clone());
        assert_tensors_close(
            &rvq.decode(codes.slice(s![.., 0..2, ..])),
            &proj.forward(coarse),
            Tolerance::default(),
        );
    }

    /// The split decode: the first half's latent, plus the rest's.
    #[test]
    fn test_split_decode() {
        let device = cpu_device();
        let mut split = SplitResidualVectorQuantizerConfig::new(3, 5, 2, 3).init(&device);
        random_codebooks(&mut split.rvq_first, 1);
        random_codebooks(&mut split.rvq_rest, 11);
        let codes = Tensor::<3, Int>::from_ints([[[0, 4], [3, 1], [2, 2]]], &device);
        let first = split
            .rvq_first
            .decode(codes.clone().slice(s![.., 0..1, ..]));
        let rest = split.rvq_rest.decode(codes.clone().slice(s![.., 1..3, ..]));
        assert_tensors_close(
            &split.decode(codes.clone()),
            &(first.clone() + rest),
            Tolerance::default(),
        );
        assert_tensors_close(
            &split.decode(codes.slice(s![.., 0..1, ..])),
            &first,
            Tolerance::default(),
        );
    }

    #[test]
    #[should_panic(expected = "codes of 1..=2 codebooks")]
    fn test_too_many_codes() {
        let device = cpu_device();
        ResidualVectorQuantizerConfig::new(2, 4, 2)
            .init(&device)
            .decode(Tensor::zeros([1, 3, 2], &device));
    }

    /// The Qwen3-TTS toy codec's split quantizer
    /// (`tools/gen_qwen3_tts_toy_fixtures.py`) on its codes: one codebook
    /// first, three after, a code whose usage is under the clamp among them.
    #[cfg(feature = "store")]
    #[test]
    fn test_qwen3_tts_toy_quantizer() {
        use crate::support::testing::{
            ToyFixture,
            assert_io_close,
        };

        let device = cpu_device();
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_tts/toy.safetensors");
        let fixture = ToyFixture::load_io(path, &device);
        let codes = fixture.io::<3>("io.codec.codes").int();
        let [_, k, _] = codes.dims();
        let at = |half: &str, part: &str| format!("decoder.quantizer.{half}.{part}");
        let [s, d] = fixture
            .io::<2>(&at("rvq_first", "vq.layers.0._codebook.embedding_sum"))
            .dims();
        let [out, _, _] = fixture
            .io::<3>(&at("rvq_first", "output_proj.weight"))
            .dims();
        let mut split = SplitResidualVectorQuantizerConfig::new(k, s, d, out).init(&device);
        for (half, rvq) in [
            ("rvq_first", &mut split.rvq_first),
            ("rvq_rest", &mut split.rvq_rest),
        ] {
            for (i, codebook) in rvq.codebooks.iter_mut().enumerate() {
                let book = |part: &str| at(half, &format!("vq.layers.{i}._codebook.{part}"));
                codebook.embedding_sum = Param::from_tensor(fixture.io(&book("embedding_sum")));
                codebook.cluster_usage = Param::from_tensor(fixture.io(&book("cluster_usage")));
            }
            rvq.output_proj.as_mut().expect("a projection").weight =
                Param::from_tensor(fixture.io(&at(half, "output_proj.weight")));
        }
        assert_io_close(
            "decoder.quantizer",
            split.decode(codes),
            fixture.io("io.codec.quantized"),
        );
    }

    /// No codebooks, or a split that leaves either side empty, is refused.
    #[test]
    fn test_illegal_codebooks() {
        let device = cpu_device();
        ErrorMatcher::illegal()
            .message_contains("no codebooks")
            .assert_err(
                &ModuleInit::<ResidualVectorQuantizer>::try_init(
                    &ResidualVectorQuantizerConfig::new(0, 4, 2),
                    &device,
                )
                .map(|_| ()),
            );
        for first in [0, 3] {
            ErrorMatcher::illegal()
                .message_contains("the split is in 1..3")
                .assert_err(
                    &ModuleInit::<SplitResidualVectorQuantizer>::try_init(
                        &SplitResidualVectorQuantizerConfig::new(3, 4, 2, 3)
                            .with_first_codebooks(first),
                        &device,
                    )
                    .map(|_| ()),
                );
        }
    }
}
