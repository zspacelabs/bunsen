//! The Qwen3-ASR model.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Int,
        TensorData,
    },
    tensor::Device,
};

use super::{
    Qwen3AsrAudioEncoder,
    Qwen3AsrAudioEncoderConfig,
    Qwen3AsrAudioEncoderMeta,
    Qwen3AsrProjector,
    Qwen3AsrProjectorConfig,
};
use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::gpts::qwen3::{
        Qwen3Config,
        Qwen3Meta,
        Qwen3Model,
    },
};

/// The audio placeholder of `Qwen3-ASR-0.6B`'s vocabulary, `<|audio_pad|>`.
pub const QWEN3_ASR_AUDIO_TOKEN: usize = 151_676;

/// Common meta for [`Qwen3Asr`] and [`Qwen3AsrConfig`].
pub trait Qwen3AsrMeta {
    /// The audio placeholder token.
    fn audio_token(&self) -> usize;

    /// The audio encoder's width.
    fn d_audio(&self) -> usize;

    /// The text model's width.
    fn d_text(&self) -> usize;

    /// The token count.
    fn n_vocab(&self) -> usize;
}

/// Config for [`Qwen3Asr`].
///
/// Implements [`Qwen3AsrMeta`].
#[derive(Config, Debug)]
pub struct Qwen3AsrConfig {
    /// The audio encoder.
    pub audio: Qwen3AsrAudioEncoderConfig,

    /// The text model.
    pub text: Qwen3Config,

    /// The token whose positions the audio frames take in a prompt.
    #[config(default = "QWEN3_ASR_AUDIO_TOKEN")]
    pub audio_token: usize,
}

impl Qwen3AsrConfig {
    /// `Qwen/Qwen3-ASR-0.6B`, and `FermionResearch`'s Phonon-1 builds of it.
    pub fn qwen3_asr_0_6b() -> Self {
        Self::new(
            Qwen3AsrAudioEncoderConfig::qwen3_asr_0_6b(),
            Qwen3Config::qwen3_0_6b(),
        )
    }

    /// `Qwen/Qwen3-ASR-1.7B`.
    pub fn qwen3_asr_1_7b() -> Self {
        Self::new(
            Qwen3AsrAudioEncoderConfig::qwen3_asr_1_7b(),
            Qwen3Config::qwen3_1_7b(),
        )
    }
}

impl Qwen3AsrMeta for Qwen3AsrConfig {
    fn audio_token(&self) -> usize {
        self.audio_token
    }

    fn d_audio(&self) -> usize {
        self.audio.d_model()
    }

    fn d_text(&self) -> usize {
        self.text.d_model()
    }

    fn n_vocab(&self) -> usize {
        self.text.n_vocab()
    }
}

impl ModuleInit<Qwen3Asr> for Qwen3AsrConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3Asr> {
        Ok(Qwen3Asr {
            audio_tower: self.audio.try_init(device)?,
            multi_modal_projector: Qwen3AsrProjectorConfig::new(
                self.audio.d_model(),
                self.text.d_model(),
            )
            .try_init(device)?,
            language_model: self.text.try_init(device)?,
            audio_token: self.audio_token,
        })
    }
}

/// Qwen3-ASR (Qwen team 2025): a Whisper-style audio encoder whose frames,
/// projected, take the place of placeholder tokens in a chat prompt, and a
/// Qwen3 language model that answers it with the transcript.
///
/// The fields carry the names `transformers`'
/// `Qwen3ASRForConditionalGeneration` gives them, its `model.` dropped.
///
/// Built by [`Qwen3AsrConfig`].
/// Implements [`Qwen3AsrMeta`].
#[derive(Module, Debug)]
pub struct Qwen3Asr {
    /// The audio encoder.
    pub audio_tower: Qwen3AsrAudioEncoder,

    /// Audio frames into the text model's embedding space.
    pub multi_modal_projector: Qwen3AsrProjector,

    /// The text model.
    pub language_model: Qwen3Model,

    /// The audio placeholder token.
    pub audio_token: usize,
}

impl Qwen3AsrMeta for Qwen3Asr {
    fn audio_token(&self) -> usize {
        self.audio_token
    }

    fn d_audio(&self) -> usize {
        self.audio_tower.d_model()
    }

    fn d_text(&self) -> usize {
        self.language_model.d_model()
    }

    fn n_vocab(&self) -> usize {
        self.language_model.n_vocab()
    }
}

impl Qwen3Asr {
    /// Encodes one utterance's log-mels into text-model embeddings.
    ///
    /// # Arguments
    /// * `features` - `[n_mels, frames]` log-mels.
    /// * `valid` - the utterance's valid mel frames.
    ///
    /// # Returns
    /// `[n, d_text]`, `n` the audio tower's
    /// [`output_len`](Qwen3AsrAudioEncoderMeta::output_len) of `valid`.
    pub fn encode_audio(
        &self,
        features: Tensor<2>,
        valid: usize,
    ) -> Tensor<2> {
        self.multi_modal_projector
            .forward(self.audio_tower.forward(features, valid))
    }

    /// The embeddings of a prompt, its audio placeholders replaced by
    /// `audio`, in order.
    ///
    /// # Arguments
    /// * `ids` - the prompt's tokens.
    /// * `audio` - `[n, d_text]`, from [`encode_audio`](Self::encode_audio).
    ///
    /// # Returns
    /// `[1, ids.len(), d_text]`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the prompt's
    /// placeholders are not `n`.
    pub fn prompt_embeds(
        &self,
        ids: &[usize],
        audio: Tensor<2>,
    ) -> BunsenResult<Tensor<3>> {
        let n = audio.dims()[0];
        let placeholders = ids.iter().filter(|&&t| t == self.audio_token).count();
        if placeholders != n {
            return Err(BunsenError::illegal(format!(
                "a prompt with {placeholders} audio placeholders for {n} audio frames"
            )));
        }
        let device = audio.device();
        let embed = |run: &[usize]| {
            let run: Vec<i64> = run.iter().map(|&t| t as i64).collect();
            let len = run.len();
            self.language_model.embed(Tensor::<2, Int>::from_data(
                TensorData::new(run, [1, len]),
                &device,
            ))
        };
        let mut parts = Vec::new();
        let (mut start, mut audio_at) = (0, 0);
        while start < ids.len() {
            let is_audio = ids[start] == self.audio_token;
            let end = ids[start..]
                .iter()
                .position(|&t| (t == self.audio_token) != is_audio)
                .map_or(ids.len(), |p| start + p);
            if is_audio {
                let len = end - start;
                parts.push(
                    audio
                        .clone()
                        .slice_dim(0, audio_at..audio_at + len)
                        .unsqueeze_dim(0),
                );
                audio_at += len;
            } else {
                parts.push(embed(&ids[start..end]));
            }
            start = end;
        }
        Ok(Tensor::cat(parts, 1))
    }

    /// Logits for every position of a prompt, without a cache.
    ///
    /// # Arguments
    /// * `embeds` - `[1, seq, d_text]`, from
    ///   [`prompt_embeds`](Self::prompt_embeds).
    ///
    /// # Returns
    /// `[1, seq, n_vocab]`.
    pub fn forward_logits(
        &self,
        embeds: Tensor<3>,
    ) -> Tensor<3> {
        let seq = embeds.dims()[1];
        let r_emb = self.language_model.rotary(seq, &embeds.device());
        let hidden = self.language_model.forward_hidden(embeds, &r_emb, None);
        self.language_model.logits(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// A config non-default in every field, and the model built from it,
    /// answer the meta alike.
    #[test]
    fn test_meta() {
        let config = Qwen3AsrConfig::new(
            Qwen3AsrAudioEncoderConfig::new(16, 32, 4, 64, 1, 8),
            Qwen3Config::new(50, 24, 1, 4, 2, 8, 48),
        )
        .with_audio_token(7);
        let model: Qwen3Asr = config.init(&cpu_device());
        assert_eq!(model.audio_token(), config.audio_token());
        assert_eq!(model.d_audio(), config.d_audio());
        assert_eq!(model.d_text(), config.d_text());
        assert_eq!(model.n_vocab(), config.n_vocab());
    }
}
