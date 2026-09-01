//! # Decoding: from mel windows to token ids.
//!
//! One loop, two seams. The loop encodes the windows, feeds the prompt,
//! then steps the text decoder one token at a time against its KV cache
//! until the search says it is complete or the cap is hit. The seams are
//! the search ([`TokenDecoder`]: [`GreedyDecoder`] or
//! [`BeamSearchDecoder`]) and the filters ([`LogitFilter`]) consulted before
//! it; the search returns finished candidates and a [`SequenceRanker`] picks
//! one per audio.
//!
//! A beam search widens the batch: with `k` beams the encoder runs once per
//! audio and its output is repeated `k` times (`row = audio * k + beam`),
//! so the cross-attention cache is built at the wide width and the
//! self-attention cache is permuted as beams branch. That is the plain
//! layout; a cache that shares cross-attention across beams is a later
//! optimization the seam allows.
//!
//! [`GreedyDecodeConfig`] and the `decode_window*` methods are the greedy
//! path as it was before the seams existed, and decode exactly as they
//! did: a [`DecodeConfig`] with a beam size of one is the same search.

use std::{
    fmt::Debug,
    sync::Arc,
};

use burn::{
    Tensor,
    config::Config,
    prelude::{
        Backend,
        Int,
        TensorData,
    },
};

use crate::{
    kits::speech::whisper::blocks::{
        Whisper,
        WhisperMeta,
    },
    ops::{
        repeat::repeat_interleave,
        split::split_padded,
    },
};

mod beam;
mod filters;
mod greedy;
mod rank;

pub use beam::BeamSearchDecoder;
pub use filters::{
    LogitFilter,
    SuppressBlank,
    SuppressTokens,
    blank_token,
    default_filters,
    default_suppress_tokens,
    non_speech_tokens,
};
pub use greedy::GreedyDecoder;
pub use rank::{
    MaximumLikelihoodRanker,
    SequenceRanker,
};

/// Splits a mel spectrogram into fixed-width windows along the frame axis,
/// zero-padding the last window to full width.
///
/// # Arguments
/// * `mels` - `[batch, n_mels, frames]`.
/// * `window` - frames per window; the model's audio context width.
pub fn mel_windows<B: Backend>(
    mels: Tensor<B, 3>,
    window: usize,
) -> Vec<Tensor<B, 3>> {
    split_padded(mels, window, 2)
}

/// The search: what the next token of every row is, given the logits.
///
/// Rows are laid out `row = audio * group_size + member`; a greedy search
/// has one member per audio, a beam search `k`. The loop owns the tensors
/// and the cache; the search owns the sequences' bookkeeping and tells the
/// loop, through `reorder`, how to permute the self-attention cache when
/// its members branch.
pub trait TokenDecoder<B: Backend>: Send + Debug {
    /// Rows per audio.
    fn group_size(&self) -> usize;

    /// Forgets any state from a previous decode.
    fn reset(&mut self);

    /// Consumes one step of logits.
    ///
    /// # Arguments
    /// * `tokens` - one sequence per row, prompt included; extended (or
    ///   replaced, as beams branch) in place.
    /// * `logits` - `[rows, vocab]`, the last position's, filtered.
    /// * `sum_logprobs` - one cumulative log probability per row, updated in
    ///   place.
    /// * `reorder` - permutes the self-attention cache: row `r` becomes row
    ///   `sources[r]`.
    ///
    /// # Returns
    /// The token to feed each row next, and whether the search is complete.
    fn update(
        &mut self,
        tokens: &mut Vec<Vec<i64>>,
        logits: Tensor<B, 2>,
        sum_logprobs: &mut [f32],
        reorder: &mut dyn FnMut(&[usize]),
    ) -> (Vec<i64>, bool);

    /// The finished candidates of every audio, each the generated ids after
    /// `prompt_len` with no stop token, and its cumulative log probability.
    fn finalize(
        &mut self,
        tokens: Vec<Vec<i64>>,
        sum_logprobs: Vec<f32>,
        prompt_len: usize,
    ) -> Vec<Vec<(Vec<i64>, f32)>>;
}

/// Greedy decoding of one mel window.
///
/// The prompt is what upstream calls the initial tokens: the SOT sequence
/// (`<|startoftranscript|>`, and for a multilingual model the language and
/// task tokens), with `<|notimestamps|>` when timestamps are not wanted,
/// optionally preceded by `<|startofprev|>` and the previous window's text.
#[derive(Config, Debug)]
pub struct GreedyDecodeConfig {
    /// The tokens fed before the first sampled one; at least one.
    pub prompt: Vec<i64>,

    /// The token that ends a sequence: `<|endoftext|>`.
    pub eot_token: i64,

    /// The most tokens to sample per window.
    #[config(default = "224")]
    pub max_tokens: usize,
}

/// Decoding of mel windows: the prompt, the stop token, the cap, and the
/// width of the search.
#[derive(Config, Debug)]
pub struct DecodeConfig {
    /// The tokens fed before the first sampled one; at least one.
    pub prompt: Vec<i64>,

    /// The token that ends a sequence: `<|endoftext|>`.
    pub eot_token: i64,

    /// The most tokens to sample per window.
    #[config(default = "224")]
    pub max_tokens: usize,

    /// Beams per audio; one is greedy.
    #[config(default = "1")]
    pub beam_size: usize,

    /// Finished candidates to collect before a beam search stops, as a
    /// multiple of the beam size; `None` is one.
    #[config(default = "None")]
    pub patience: Option<f64>,

    /// The exponent of the ranker's length penalty; `None` normalizes by
    /// length.
    #[config(default = "None")]
    pub length_penalty: Option<f64>,
}

impl DecodeConfig {
    /// The search this config asks for.
    pub fn decoder<B: Backend>(&self) -> Box<dyn TokenDecoder<B>> {
        if self.beam_size == 1 {
            Box::new(GreedyDecoder::new(self.eot_token, self.prompt[0]))
        } else {
            Box::new(BeamSearchDecoder::new(
                self.beam_size,
                self.eot_token,
                self.patience,
            ))
        }
    }

    /// The ranker this config asks for.
    pub fn ranker(&self) -> MaximumLikelihoodRanker {
        MaximumLikelihoodRanker {
            length_penalty: self.length_penalty,
        }
    }
}

impl From<&GreedyDecodeConfig> for DecodeConfig {
    fn from(config: &GreedyDecodeConfig) -> Self {
        Self::new(config.prompt.clone(), config.eot_token).with_max_tokens(config.max_tokens)
    }
}

/// The `[rows, step_len]` tensor that feeds a step.
fn feed_tensor<B: Backend>(
    feed: &[i64],
    step_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let rows = feed.len() / step_len;
    Tensor::from_data(TensorData::new(feed.to_vec(), [rows, step_len]), device)
}

impl<B: Backend> Whisper<B> {
    /// Decodes a batch of mel windows.
    ///
    /// # Arguments
    /// * `mels` - `[batch, n_mels, frames]`, `frames` the audio context width;
    ///   every row is decoded against its own audio.
    /// * `config` - the prompt, stop token, cap, and search width.
    /// * `filters` - applied to the logits every step, in order, before the
    ///   search sees them.
    ///
    /// # Returns
    /// The generated ids per row, after the prompt and without the stop
    /// token.
    ///
    /// # Panics
    /// If the batch is empty, the window is not the audio context width, or
    /// the prompt is empty.
    pub fn decode_windows(
        &self,
        mels: Tensor<B, 3>,
        config: &DecodeConfig,
        filters: &[Arc<dyn LogitFilter<B>>],
    ) -> Vec<Vec<i64>> {
        let mut decoder = config.decoder::<B>();
        self.decode_windows_with(mels, config, decoder.as_mut(), filters)
    }

    /// [`Self::decode_windows`] with an explicit search, whose group size
    /// decides the width; the config's `beam_size` and `patience` are then
    /// the search's own business.
    pub fn decode_windows_with(
        &self,
        mels: Tensor<B, 3>,
        config: &DecodeConfig,
        decoder: &mut dyn TokenDecoder<B>,
        filters: &[Arc<dyn LogitFilter<B>>],
    ) -> Vec<Vec<i64>> {
        let [n_audio, _, frames] = mels.dims();
        assert!(n_audio > 0, "decode needs at least one row");
        assert_eq!(
            frames,
            self.max_audio_ctx(),
            "window must be the model's audio context width",
        );
        assert!(!config.prompt.is_empty(), "prompt must not be empty");

        decoder.reset();
        let k = decoder.group_size();
        let rows = n_audio * k;
        let prompt_len = config.prompt.len();

        let device = mels.device();
        let mut xa = self.forward_encoder(mels);
        if k > 1 {
            xa = repeat_interleave::<B, 3, 4, _>(xa, k, 0);
        }
        let mut cache = self.decoder.new_cache(xa);

        let mut tokens: Vec<Vec<i64>> = vec![config.prompt.clone(); rows];
        let mut sum_logprobs = vec![0f32; rows];

        let mut feed: Vec<i64> = config.prompt.repeat(rows);
        let mut step_len = prompt_len;

        for _ in 0..config.max_tokens {
            let logits = self
                .decoder
                .forward_cached(feed_tensor(&feed, step_len, &device), &mut cache);
            let [_, positions, vocab] = logits.dims();
            let last = positions - 1;
            let mut logits: Tensor<B, 2> = logits
                .slice_dim(1, last as isize..(last + 1) as isize)
                .reshape([rows, vocab]);

            for filter in filters {
                logits = filter.apply(logits, &tokens, prompt_len);
            }

            let (next, completed) =
                decoder.update(&mut tokens, logits, &mut sum_logprobs, &mut |sources| {
                    cache.reorder(sources)
                });
            if completed {
                break;
            }

            feed = next;
            step_len = 1;

            if cache.pos() >= self.max_text_ctx() {
                break;
            }
        }

        let ranker = config.ranker();
        decoder
            .finalize(tokens, sum_logprobs, prompt_len)
            .into_iter()
            .map(|candidates| {
                let best = ranker.rank(&candidates);
                candidates
                    .into_iter()
                    .nth(best)
                    .expect("the ranker picked a candidate")
                    .0
            })
            .collect()
    }

    /// Greedily decodes a batch of mel windows, each against its own audio.
    ///
    /// Rows finish independently: a row that emits the stop token stops
    /// growing while the rest continue, and the batch stops when every row
    /// has finished or `max_tokens` is reached.
    ///
    /// # Arguments
    /// * `mels` - `[batch, n_mels, frames]`, `frames` the audio context width.
    ///
    /// # Returns
    /// The generated ids per row, without the stop token.
    ///
    /// # Panics
    /// If the batch is empty, the window is not the audio context width, or
    /// the prompt is empty.
    pub fn decode_window_batched(
        &self,
        mels: Tensor<B, 3>,
        config: &GreedyDecodeConfig,
    ) -> Vec<Vec<i64>> {
        self.decode_windows(mels, &DecodeConfig::from(config), &[])
    }

    /// Greedily decodes one mel window.
    ///
    /// # Arguments
    /// * `mels` - `[1, n_mels, frames]`, `frames` the audio context width.
    ///
    /// # Panics
    /// If the batch is not one.
    pub fn decode_window(
        &self,
        mels: Tensor<B, 3>,
        config: &GreedyDecodeConfig,
    ) -> Vec<i64> {
        assert_eq!(mels.dims()[0], 1, "decode_window handles a batch of one");

        self.decode_window_batched(mels, config)
            .pop()
            .expect("one row in, one row out")
    }

    /// Greedily decodes a whole spectrogram window by window, with the same
    /// prompt for every window.
    ///
    /// # Arguments
    /// * `mels` - `[1, n_mels, frames]`, any length; split with
    ///   [`mel_windows`].
    ///
    /// # Returns
    /// One id sequence per window.
    pub fn decode_chunked(
        &self,
        mels: Tensor<B, 3>,
        config: &GreedyDecodeConfig,
    ) -> Vec<Vec<i64>> {
        mel_windows(mels, self.max_audio_ctx())
            .into_iter()
            .map(|window| self.decode_window(window, config))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::whisper::blocks::WhisperApiConfig,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    /// A tiny model, sized so the attention has whole heads.
    fn tiny_model(device: &burn::prelude::Device<B>) -> Whisper<B> {
        WhisperApiConfig::new(
            /* n_mels */ 8, /* vocab_size */ 32, /* d_model */ 64,
            /* max_audio_ctx */ 16, /* n_encoder_layers */ 1, /* max_text_ctx */ 12,
            /* n_decoder_layers */ 1,
        )
        .init(device)
    }

    #[test]
    fn test_mel_windows_splits_and_pads() {
        let device = Default::default();
        let (batch, n_mels, window) = (1, 4, 10);

        // Exactly two windows.
        let exact: Tensor<B, 3> =
            Tensor::random([batch, n_mels, 20], Distribution::Default, &device);
        let windows = mel_windows(exact, window);
        assert_eq!(windows.len(), 2);
        assert!(windows.iter().all(|w| w.dims() == [batch, n_mels, window]));

        // A ragged tail is padded up, not dropped.
        let ragged: Tensor<B, 3> =
            Tensor::random([batch, n_mels, 25], Distribution::Default, &device);
        let windows = mel_windows(ragged, window);
        assert_eq!(windows.len(), 3);
        assert!(windows.iter().all(|w| w.dims() == [batch, n_mels, window]));

        // Shorter than one window still gives one padded window.
        let short: Tensor<B, 3> =
            Tensor::random([batch, n_mels, 3], Distribution::Default, &device);
        assert_eq!(mel_windows(short, window).len(), 1);
    }

    /// The padded tail must be zeros, and the kept part must be untouched.
    #[test]
    fn test_mel_windows_preserves_content() {
        let device = Default::default();
        let frames = 7;

        let mels: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(
                (0..frames).map(|f| (f + 1) as f64).collect::<Vec<_>>(),
                [1, 1, frames],
            ),
            &device,
        );

        let windows = mel_windows(mels, 10);
        assert_eq!(windows.len(), 1);

        let got: Vec<f64> = windows[0]
            .clone()
            .cast(burn::tensor::DType::F64)
            .to_data()
            .to_vec()
            .unwrap();

        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[serial]
    fn test_decode_window_respects_the_token_cap() {
        let device = Default::default();
        let model = tiny_model(&device);

        let mels: Tensor<B, 3> = Tensor::random(
            [1, model.n_mels(), model.max_audio_ctx()],
            Distribution::Default,
            &device,
        );

        // An unreachable stop token, so only the cap can end this.
        let config = GreedyDecodeConfig::new(vec![1, 2], -1).with_max_tokens(5);
        let out = model.decode_window(mels, &config);

        assert!(
            out.len() <= 5,
            "generated {} tokens against a cap of 5",
            out.len(),
        );
        assert!(
            out.iter()
                .all(|&t| t >= 0 && (t as usize) < model.vocab_size()),
            "generated an id outside the vocabulary",
        );
    }

    /// A stop token the model must emit immediately: every id is a stop.
    #[test]
    #[serial]
    fn test_decode_window_stops_on_eot() {
        let device = Default::default();
        let model = tiny_model(&device);

        let mels: Tensor<B, 3> = Tensor::random(
            [1, model.n_mels(), model.max_audio_ctx()],
            Distribution::Default,
            &device,
        );

        // Whatever the argmax picks, run it once to learn it, then make that
        // id the stop token: the decode must then produce nothing.
        let probe = GreedyDecodeConfig::new(vec![1, 2], -1).with_max_tokens(1);
        let first = model.decode_window(mels.clone(), &probe);
        assert_eq!(first.len(), 1);

        let config = GreedyDecodeConfig::new(vec![1, 2], first[0]).with_max_tokens(5);
        assert!(
            model.decode_window(mels, &config).is_empty(),
            "decoding did not stop on the first emitted token",
        );
    }

    #[test]
    #[serial]
    fn test_decode_chunked_covers_every_window() {
        let device = Default::default();
        let model = tiny_model(&device);
        let window = model.max_audio_ctx();

        // Two and a half windows.
        let mels: Tensor<B, 3> = Tensor::random(
            [1, model.n_mels(), window * 2 + window / 2],
            Distribution::Default,
            &device,
        );

        let config = GreedyDecodeConfig::new(vec![1], -1).with_max_tokens(3);
        let per_window = model.decode_chunked(mels, &config);

        assert_eq!(
            per_window.len(),
            3,
            "the ragged tail must get its own window"
        );
    }
    /// **The batching contract.** A batched decode must give each row exactly
    /// what decoding that row alone gives.
    ///
    /// This is what a mis-laid-out prompt, a transposed argmax, or leaking one
    /// row's tokens into another would break — and none of those would change
    /// a shape.
    #[test]
    #[serial]
    fn test_batched_decode_matches_individual() {
        let device = Default::default();
        let model = tiny_model(&device);
        let (batch, window) = (3, model.max_audio_ctx());

        // Distinct rows, so agreement is not trivially satisfied.
        let mels: Tensor<B, 3> = Tensor::random(
            [batch, model.n_mels(), window],
            Distribution::Default,
            &device,
        );

        // Unreachable stop token: every row runs to the cap, which exercises
        // the stepping rather than the early exit.
        let config = GreedyDecodeConfig::new(vec![1, 2], -1).with_max_tokens(6);

        let batched = model.decode_window_batched(mels.clone(), &config);
        assert_eq!(batched.len(), batch);

        for row in 0..batch {
            let alone = model.decode_window(
                mels.clone().slice_dim(0, row as isize..(row + 1) as isize),
                &config,
            );
            assert_eq!(
                batched[row], alone,
                "row {row} differs from its solo decode"
            );
        }
    }

    /// Rows stop independently: one row's `eot_token` must not truncate the
    /// others.
    #[test]
    #[serial]
    fn test_batched_rows_finish_independently() {
        let device = Default::default();
        let model = tiny_model(&device);
        let (batch, window) = (3, model.max_audio_ctx());

        let mels: Tensor<B, 3> = Tensor::random(
            [batch, model.n_mels(), window],
            Distribution::Default,
            &device,
        );

        // Learn what row 0 emits first, then make that its stop token. Row 0
        // finishes immediately; the others should be unaffected.
        let probe = GreedyDecodeConfig::new(vec![1, 2], -1).with_max_tokens(1);
        let first_of_row0 = model.decode_window(mels.clone().slice_dim(0, 0..1), &probe)[0];

        let config = GreedyDecodeConfig::new(vec![1, 2], first_of_row0).with_max_tokens(6);
        let batched = model.decode_window_batched(mels.clone(), &config);

        assert!(
            batched[0].is_empty(),
            "row 0 should stop on its first token"
        );

        // Every row must still match its solo decode under the same config.
        for row in 0..batch {
            let alone = model.decode_window(
                mels.clone().slice_dim(0, row as isize..(row + 1) as isize),
                &config,
            );
            assert_eq!(
                batched[row], alone,
                "row {row} was affected by another row finishing",
            );
        }
    }
}

/// The search seam: the beam decoder at width one is the greedy decoder
/// (I7), the cache permutes as the search asks, and the filters reach it.
#[cfg(test)]
mod search_tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
        ops::FloatElem,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::whisper::blocks::WhisperApiConfig,
        prelude::TensorElemOpExt,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type F = FloatElem<B>;

    const VOCAB: usize = 32;
    const EOT: i64 = 31;

    /// A tiny model, seeded so a near-tie ranks the same way every run.
    fn tiny_model(device: &burn::prelude::Device<B>) -> Whisper<B> {
        B::seed(device, 11);
        WhisperApiConfig::new(
            /* n_mels */ 8, /* vocab_size */ VOCAB, /* d_model */ 64,
            /* max_audio_ctx */ 16, /* n_encoder_layers */ 1, /* max_text_ctx */ 12,
            /* n_decoder_layers */ 1,
        )
        .init(device)
    }

    fn windows(
        n: usize,
        device: &burn::prelude::Device<B>,
    ) -> Tensor<B, 3> {
        Tensor::random([n, 8, 16], Distribution::Default, device)
    }

    /// I7: a beam search of width one returns exactly what greedy returns,
    /// row for row, with and without a cap in the way.
    #[test]
    #[serial]
    fn test_beam_of_one_is_greedy() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mels = windows(3, &device);

        for max_tokens in [2, 224] {
            let config = DecodeConfig::new(vec![1, 2], EOT).with_max_tokens(max_tokens);
            let greedy = model.decode_windows(mels.clone(), &config, &[]);

            let mut beam = BeamSearchDecoder::new(1, EOT, None);
            let wide = model.decode_windows_with(mels.clone(), &config, &mut beam, &[]);
            assert_eq!(wide, greedy, "beam of one at a cap of {max_tokens}");
            assert!(greedy.iter().all(|row| row.len() <= max_tokens));
        }
    }

    /// A wider beam decodes every audio to something within the cap, and
    /// the same thing every time.
    #[test]
    #[serial]
    fn test_beam_search_runs_wider() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mels = windows(2, &device);

        let config = DecodeConfig::new(vec![1, 2], EOT)
            .with_max_tokens(6)
            .with_beam_size(3)
            .with_patience(Some(2.0));
        let first = model.decode_windows(mels.clone(), &config, &[]);
        assert_eq!(first.len(), 2, "one answer per audio, not per beam");
        assert!(first.iter().all(|row| row.len() <= 6));
        assert!(first.iter().all(|row| !row.contains(&EOT)));

        let again = model.decode_windows(mels, &config, &[]);
        assert_eq!(again, first);
    }

    /// Reordering the cache is the same as having built it in that order:
    /// two rows with the same audio, swapped, continue as the swapped
    /// build would.
    #[test]
    #[serial]
    fn test_reorder_permutes_the_self_attention_cache() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mels = windows(1, &device).repeat_dim(0, 2);

        let prompts = |rows: Vec<i64>| feed_tensor::<B>(&rows, 2, &device);
        let next = feed_tensor::<B>(&[5, 5], 1, &device);

        let mut swapped = model.decoder.new_cache(model.forward_encoder(mels.clone()));
        model
            .decoder
            .forward_cached(prompts(vec![1, 2, 3, 4]), &mut swapped);
        swapped.reorder(&[1, 0]);
        let after_swap = model.decoder.forward_cached(next.clone(), &mut swapped);

        let mut built = model.decoder.new_cache(model.forward_encoder(mels.clone()));
        model
            .decoder
            .forward_cached(prompts(vec![3, 4, 1, 2]), &mut built);
        let as_built = model.decoder.forward_cached(next.clone(), &mut built);

        after_swap
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&as_built.to_data_as::<F>(), Tolerance::permissive());

        // Duplicating a row is a branch: both rows continue identically.
        let mut branched = model.decoder.new_cache(model.forward_encoder(mels));
        model
            .decoder
            .forward_cached(prompts(vec![1, 2, 3, 4]), &mut branched);
        branched.reorder(&[0, 0]);
        let both = model.decoder.forward_cached(next, &mut branched);
        let [_, positions, vocab] = both.dims();
        let row0 = both.clone().slice_dim(0, 0..1).reshape([positions, vocab]);
        let row1 = both.slice_dim(0, 1..2).reshape([positions, vocab]);
        row0.to_data_as::<F>()
            .assert_approx_eq::<F>(&row1.to_data_as::<F>(), Tolerance::permissive());
    }

    /// Filters are consulted every step: with everything but one id
    /// suppressed, greedy emits that id to the cap, and a beam of two
    /// emits only its two allowed ids.
    #[test]
    #[serial]
    fn test_filters_reach_the_search() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mels = windows(1, &device);

        let only = |allowed: &[i64]| -> Vec<Arc<dyn LogitFilter<B>>> {
            let ids = (0..VOCAB as i64).filter(|id| !allowed.contains(id));
            vec![Arc::new(SuppressTokens::new(ids))]
        };

        let config = DecodeConfig::new(vec![1, 2], EOT).with_max_tokens(4);
        let out = model.decode_windows(mels.clone(), &config, &only(&[7]));
        assert_eq!(out, vec![vec![7, 7, 7, 7]]);

        let config = config.with_beam_size(2);
        let out = model.decode_windows(mels, &config, &only(&[7, 9]));
        assert_eq!(out[0].len(), 4);
        assert!(out[0].iter().all(|t| *t == 7 || *t == 9), "{out:?}");
    }
}
