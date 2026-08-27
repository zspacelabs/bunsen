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

use crate::kits::speech::whisper::blocks::{
    Whisper,
    WhisperMeta,
};

/// Splits `[batch, n_mels, frames]` into fixed-width windows.
///
/// The final window is zero-padded out to `window`, matching how Whisper pads
/// short audio to its full 30 s context. Audio shorter than one window yields
/// exactly one padded window, and empty audio yields none.
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
    assert!(window > 0, "window must be non-zero");

    let [batch, n_mels, frames] = mels.dims();
    let device = mels.device();

    let mut windows = Vec::new();
    let mut seek = 0;

    while seek < frames {
        let end = (seek + window).min(frames);
        let chunk = mels.clone().slice_dim(2, seek as isize..end as isize);

        windows.push(if end - seek == window {
            chunk
        } else {
            let pad: Tensor<B, 3> = Tensor::zeros([batch, n_mels, window - (end - seek)], &device);
            Tensor::cat(vec![chunk, pad], 2)
        });

        seek = end;
    }

    windows
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
    /// Greedily decodes one already-windowed spectrogram.
    ///
    /// Encodes `mels` once, builds the key/value cache from that encoding, then
    /// steps one token at a time. The prompt is fed as a single prefill.
    ///
    /// # Arguments
    /// * `mels`: `[1, n_mels, window]` — one window, batch of one.
    /// * `config`: prompt, stop token and generation cap.
    ///
    /// # Returns
    /// The generated ids, prompt excluded, without `eot_token`.
    ///
    /// # Panics
    ///
    /// If the batch is not 1, if the prompt is empty, or if the window is not
    /// the model's audio context width.
    pub fn decode_window(
        &self,
        mels: Tensor<B, 3>,
        config: &GreedyDecodeConfig,
    ) -> Vec<i64> {
        let [batch, _, frames] = mels.dims();
        assert_eq!(batch, 1, "decode_window handles a batch of one");
        assert_eq!(
            frames,
            self.max_audio_ctx(),
            "window must be the model's audio context width",
        );
        assert!(!config.prompt.is_empty(), "prompt must not be empty");

        let device = mels.device();
        let xa = self.forward_encoder(mels);

        let mut cache = self.decoder.new_cache(xa);
        let mut generated = Vec::new();

        // Prefill the prompt in one pass, then step.
        let mut next: Vec<i64> = config.prompt.clone();

        for _ in 0..config.max_tokens {
            let len = next.len();
            let tokens: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(next, [1, len]), &device);

            let logits = self.decoder.forward_cached(tokens, &mut cache);

            // The prediction for the next token is the last position's.
            let last = logits.dims()[1] - 1;
            let token = logits
                .slice_dim(1, last as isize..(last + 1) as isize)
                .argmax(2)
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .unwrap()[0];

            if token == config.eot_token {
                break;
            }

            generated.push(token);
            next = vec![token];

            // Whisper's decoder cannot see past its text context.
            if cache.pos() >= self.max_text_ctx() {
                break;
            }
        }

        generated
    }

    /// Splits a spectrogram into windows and greedily decodes each.
    ///
    /// Each window is independent: it gets its own encoder pass, its own cache
    /// and its own prompt. Carrying text across windows as a prompt is a
    /// transcription-level concern and is not done here.
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
}
