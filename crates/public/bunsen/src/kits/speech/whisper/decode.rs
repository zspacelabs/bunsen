//! # Chunked greedy decoding.
//!
//! Whisper sees a fixed 30 s window. Longer audio is cut into windows, each
//! encoded once and then decoded against a key/value cache — which is the
//! arrangement that makes the cache pay: one encoder pass and one
//! cross-attention projection serve every token in the window.
//!
//! This produces **token ids**, not text. Turning ids into text needs
//! Whisper's tokenizer, which lives outside this crate.

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
    ops::split::split_padded,
};

/// Splits `[batch, n_mels, frames]` into fixed-width windows.
///
/// The final window is zero-padded out to `window`, matching how Whisper pads
/// short audio to its full 30 s context. Audio shorter than one window yields
/// exactly one padded window, and empty audio yields none.
///
/// Names the frame axis and the padding policy; the split itself is
/// [`split_padded`].
///
/// # Arguments
/// * `mels`: `[batch, n_mels, frames]` log-mels.
/// * `window`: frames per window — the model's
///   [`max_audio_ctx`](WhisperMeta::max_audio_ctx).
///
/// # Returns
/// Windows of `[batch, n_mels, window]`, in order.
pub fn mel_windows<B: Backend>(
    mels: Tensor<B, 3>,
    window: usize,
) -> Vec<Tensor<B, 3>> {
    split_padded(mels, window, 2)
}

/// How to drive a greedy decode.
#[derive(Config, Debug)]
pub struct GreedyDecodeConfig {
    /// Tokens prefixed to every window's decode.
    ///
    /// Whisper's are the special ids that select task and language, e.g.
    /// `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`. Their
    /// values come from the tokenizer, which this crate does not own.
    pub prompt: Vec<i64>,

    /// The end-of-text token. Decoding a window stops when it is emitted.
    pub eot_token: i64,

    /// Cap on tokens generated per window, excluding the prompt.
    ///
    /// A greedy decode can fail to emit `eot_token` at all, so this is what
    /// guarantees termination.
    #[config(default = "224")]
    pub max_tokens: usize,
}

impl<B: Backend> Whisper<B> {
    /// Greedily decodes one already-windowed spectrogram, for a batch.
    ///
    /// Encodes `mels` once, builds the key/value cache from that encoding,
    /// prefills the prompt for every row, then steps a token at a time across
    /// the whole batch.
    ///
    /// Rows finish independently. A row that emits `eot_token` stops
    /// contributing, but the batch keeps stepping until every row has finished
    /// or `max_tokens` is reached — so a finished row is fed a filler token
    /// whose output is discarded. That filler is the first prompt token rather
    /// than `eot_token`, because `eot_token` need not be a valid embedding
    /// index and would fault the lookup.
    ///
    /// # Arguments
    /// * `mels`: `[batch, n_mels, window]` — one window per row.
    /// * `config`: prompt, stop token and generation cap, shared by all rows.
    ///
    /// # Returns
    /// The generated ids per row, prompt excluded, each truncated at its own
    /// `eot_token`.
    ///
    /// # Panics
    ///
    /// If the batch is empty, if the prompt is empty, or if the window is not
    /// the model's audio context width.
    pub fn decode_window_batched(
        &self,
        mels: Tensor<B, 3>,
        config: &GreedyDecodeConfig,
    ) -> Vec<Vec<i64>> {
        let [batch, _, frames] = mels.dims();
        assert!(batch > 0, "decode needs at least one row");
        assert_eq!(
            frames,
            self.max_audio_ctx(),
            "window must be the model's audio context width",
        );
        assert!(!config.prompt.is_empty(), "prompt must not be empty");

        let device = mels.device();
        let xa = self.forward_encoder(mels);
        let mut cache = self.decoder.new_cache(xa);

        let mut generated: Vec<Vec<i64>> = vec![Vec::new(); batch];
        let mut finished = vec![false; batch];

        // Valid for every row, and only ever consumed by finished ones.
        let filler = config.prompt[0];

        // Row-major `[batch, prompt_len]`: each row gets the same prompt.
        let mut next: Vec<i64> = config.prompt.repeat(batch);
        let mut step_len = config.prompt.len();

        for _ in 0..config.max_tokens {
            let tokens: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(next, [batch, step_len]), &device);

            let logits = self.decoder.forward_cached(tokens, &mut cache);

            // The prediction for the next token is the last position's.
            let last = logits.dims()[1] - 1;
            let picked: Vec<i64> = logits
                .slice_dim(1, last as isize..(last + 1) as isize)
                .argmax(2)
                .into_data()
                .convert::<i64>()
                .to_vec()
                .unwrap();

            next = Vec::with_capacity(batch);
            for (row, token) in picked.into_iter().enumerate() {
                if finished[row] {
                    next.push(filler);
                } else if token == config.eot_token {
                    finished[row] = true;
                    next.push(filler);
                } else {
                    generated[row].push(token);
                    next.push(token);
                }
            }

            if finished.iter().all(|&done| done) {
                break;
            }

            step_len = 1;

            // Whisper's decoder cannot see past its text context.
            if cache.pos() >= self.max_text_ctx() {
                break;
            }
        }

        generated
    }

    /// Greedily decodes one already-windowed spectrogram.
    ///
    /// The batch-of-one case of
    /// [`decode_window_batched`](Self::decode_window_batched).
    ///
    /// # Arguments
    /// * `mels`: `[1, n_mels, window]` — one window, batch of one.
    /// * `config`: prompt, stop token and generation cap.
    ///
    /// # Returns
    /// The generated ids, prompt excluded, without `eot_token`.
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

    /// Splits a spectrogram into windows and greedily decodes each.
    ///
    /// Each window is independent: it gets its own encoder pass, its own cache
    /// and its own prompt. Carrying text across windows as a prompt is a
    /// transcription-level concern and is not done here.
    ///
    /// Windows are decoded **one at a time**, which bounds memory to a single
    /// encoder output and cache. Because they are independent, stacking them
    /// and calling [`decode_window_batched`](Self::decode_window_batched)
    /// gives the same ids and is faster — at the cost of holding every
    /// window's encoding at once. Long audio is exactly where that trade bites,
    /// so the sequential form is the default here.
    ///
    /// # Arguments
    /// * `mels`: `[1, n_mels, frames]` log-mels of any length.
    /// * `config`: as [`decode_window`](Self::decode_window).
    ///
    /// # Returns
    /// One vector of ids per window, in order.
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
