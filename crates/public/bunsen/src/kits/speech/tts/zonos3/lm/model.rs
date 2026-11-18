//! The language model.

use alloc::vec::Vec;

use burn::{
    Tensor,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        Linear,
        LinearConfig,
    },
    prelude::{
        Int,
        s,
    },
    tensor::Device,
};

use super::Zonos3LmConfig;
use crate::{
    blocks::transformers::decoders::{
        attn_res::AttnResStack,
        stack::DecoderStackCache,
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    ops::transformers::attention::KVCacheStorage,
};

impl ModuleInit<Zonos3Lm> for Zonos3LmConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Zonos3Lm> {
        let d = self.d_model;
        let audio_vocab = self.audio_vocab();
        let embedders = (0..self.n_codebooks)
            .map(|_| audio_vocab)
            .chain([self.text_rows])
            .map(|rows| EmbeddingConfig::new(rows, d).init(device))
            .collect();
        Ok(Zonos3Lm {
            embedders,
            speaker_projection: LinearConfig::new(self.speaker_dim, d).init(device),
            condition_embeddings: self
                .condition_buckets
                .iter()
                .map(|&n| EmbeddingConfig::new(n + 1, d).init(device))
                .collect(),
            stack: self.stack().try_init(device)?,
            head: LinearConfig::new(d, self.n_codebooks * audio_vocab)
                .with_bias(false)
                .init(device),
            n_codebooks: self.n_codebooks,
            audio_vocab,
            softcap: self.softcap,
        })
    }
}

/// The rows of a prompt whose embedding is replaced rather than looked up:
/// a speaker row, and the condition rows.
#[derive(Clone, Debug, Default)]
pub struct Zonos3RowConditioning {
    /// A speaker row: its absolute position, and the `[B, S]` speaker
    /// embedding projected in its place, in any float dtype (cast to the
    /// model's before the projection).
    pub speaker: Option<(usize, Tensor<2>)>,

    /// The condition rows: the first one's absolute position, and `[B, C]`
    /// bucket ids; condition `i`'s row, `first + i`, is its table's row for
    /// its bucket.
    pub conditions: Option<(usize, Tensor<2, Int>)>,
}

/// What [`Zonos3Lm::forward_traced`] computed on the way to the logits.
#[derive(Clone, Debug)]
pub struct Zonos3LmTrace {
    /// `[B, T, D]`: the summed row embeddings.
    pub emb_sum: Tensor<3>,

    /// `[B, T, D]`: with the speaker and condition rows replaced.
    pub emb_cond: Tensor<3>,

    /// The stack's bank: the normalized embedding, then each layer's
    /// attention and feed-forward outputs, each `[B, T, D]`.
    pub bank: Vec<Tensor<3>>,

    /// `[B, T, D]`: the stack's output, after its final norm.
    pub hidden: Tensor<3>,
}

/// ZONOS3-Mini's language model: rows of ids in, each row's logits over
/// every codebook's vocabulary out.
///
/// A row is `n_codebooks` audio ids and a text id; its embedding is the sum
/// of each id's row in its own table (the text table last). A speaker row's
/// and the condition rows' embeddings are replaced instead
/// ([`Zonos3RowConditioning`]). The rows go through an [`AttnResStack`] of
/// gated `DeltaNet` and global attention layers, the first with a dense
/// `SwiGLU`, the rest with a latent mixture of experts; one linear then
/// gives each codebook's logits, soft-capped by `c · tanh(x / c)`.
///
/// Built by [`Zonos3LmConfig`].
#[derive(Module, Debug)]
pub struct Zonos3Lm {
    /// The row embeddings: each codebook's, `[V_a, D]`, then the text's.
    pub embedders: Vec<Embedding>,

    /// The speaker embedding's projection, `[S, D]`, with a bias.
    pub speaker_projection: Linear,

    /// Each condition's table, its buckets and an unknown bucket last.
    pub condition_embeddings: Vec<Embedding>,

    /// The layers.
    pub stack: AttnResStack,

    /// Every codebook's logits, `[D, K V_a]`.
    pub head: Linear,

    /// The audio codebooks, `K`.
    #[module(skip)]
    pub n_codebooks: usize,

    /// Each codebook's vocabulary, `V_a`.
    #[module(skip)]
    pub audio_vocab: usize,

    /// The logits' soft-cap.
    #[module(skip)]
    pub softcap: f64,
}

impl Zonos3Lm {
    /// A decode cache for `rows` batch rows of at most `positions` positions
    /// ([`AttnResStack::decode_cache`]).
    pub fn decode_cache(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        self.stack.decode_cache(rows, positions, storage, device)
    }

    /// The rows' embeddings at absolute positions `start..start + T`: the
    /// summed lookups, and the same with the conditioned rows replaced.
    ///
    /// # Arguments
    /// - `ids`: `[B, T, K + 1]`.
    /// - `rows`: the conditioned rows; those outside the span are ignored.
    /// - `start`: the first row's position.
    ///
    /// # Returns
    /// - `([B, T, D], [B, T, D])`.
    pub fn embed(
        &self,
        ids: Tensor<3, Int>,
        rows: &Zonos3RowConditioning,
        start: usize,
    ) -> (Tensor<3>, Tensor<3>) {
        let [b, t] = unpack_shape_contract!(
            ["B", "T", "C"],
            &ids.dims(),
            &["B", "T"],
            &[("C", self.n_codebooks + 1)]
        );
        let sum = self
            .embedders
            .iter()
            .enumerate()
            .map(|(c, table)| {
                table.forward(ids.clone().slice(s![.., .., c..c + 1]).reshape([b, t]))
            })
            .reduce(|a, b| a + b)
            .expect("at least the text table");
        let in_span = |row: usize| (start..start + t).contains(&row).then(|| row - start);

        let mut x = sum.clone();
        if let Some((row, speaker)) = &rows.speaker
            && let Some(at) = in_span(*row)
        {
            let projected = self
                .speaker_projection
                .forward(speaker.clone().cast(sum.dtype()));
            x = x.slice_assign(s![.., at..at + 1, ..], projected.unsqueeze_dim(1));
        }
        if let Some((first, buckets)) = &rows.conditions {
            for (i, table) in self.condition_embeddings.iter().enumerate() {
                if let Some(at) = in_span(first + i) {
                    let looked_up = table.forward(buckets.clone().slice(s![.., i..i + 1]));
                    x = x.slice_assign(s![.., at..at + 1, ..], looked_up);
                }
            }
        }
        (sum, x)
    }

    /// Each row's logits over every codebook's vocabulary.
    ///
    /// # Arguments
    /// - `ids`: `[B, T, K + 1]`, at the cache's position (0 without one).
    /// - `rows`: the conditioned rows, by absolute position.
    /// - `cache`: a decode cache ([`decode_cache`](Self::decode_cache)),
    ///   advanced by `T`; `None` runs the rows alone.
    ///
    /// # Returns
    /// - `[B, T, K, V_a]`, soft-capped.
    pub fn forward(
        &self,
        ids: Tensor<3, Int>,
        rows: &Zonos3RowConditioning,
        cache: Option<&mut DecoderStackCache>,
    ) -> Tensor<4> {
        self.forward_traced(ids, rows, cache).0
    }

    /// [`forward`](Self::forward), with what it computed on the way.
    pub fn forward_traced(
        &self,
        ids: Tensor<3, Int>,
        rows: &Zonos3RowConditioning,
        cache: Option<&mut DecoderStackCache>,
    ) -> (Tensor<4>, Zonos3LmTrace) {
        let [b, t, _] = ids.dims();
        let start = cache.as_ref().map_or(0, |c| c.pos());
        let (emb_sum, emb_cond) = self.embed(ids, rows, start);
        let (hidden, bank) = match cache {
            Some(cache) => self.stack.forward_bank(
                emb_cond.clone(),
                &cache.rotary,
                Some(&mut cache.cache),
                Some(&mut cache.linear),
            ),
            None => {
                let rotary = self.stack.stack.rotary(t, &emb_cond.device());
                self.stack
                    .forward_bank(emb_cond.clone(), &rotary, None, None)
            }
        };
        let c = self.softcap;
        let logits = self
            .head
            .forward(hidden.clone())
            .div_scalar(c)
            .tanh()
            .mul_scalar(c)
            .reshape([b, t, self.n_codebooks, self.audio_vocab]);
        (
            logits,
            Zonos3LmTrace {
                emb_sum,
                emb_cond,
                bank,
                hidden,
            },
        )
    }
}

impl CanonicalKeys for Zonos3Lm {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}
