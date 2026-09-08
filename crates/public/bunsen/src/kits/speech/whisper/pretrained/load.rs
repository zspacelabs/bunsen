//! Load the pretrained checkpoint that `bunsen-bundled-whisper` fetched.

use burn::prelude::Backend;

use crate::{
    errors::BunsenResult,
    kits::{
        speech::whisper::{
            blocks::{
                Whisper,
                WhisperApiConfig,
            },
            driver::WhisperSpecialIds,
            pretrained::{
                PytorchWhisperScanner,
                bundled,
            },
        },
        tokens::TiktokenRanks,
    },
};

impl<B: Backend> Whisper<B> {
    /// Loads `OpenAI`'s multilingual Whisper *base* checkpoint.
    ///
    /// The checkpoint is not in this crate: `bunsen-bundled-whisper`
    /// fetches it at build time, pins it to a SHA-256 and caches it. It is
    /// 145 MB, so it stays a file on disk rather than bytes in the binary —
    /// which is the one way this differs from
    /// [`SileroVad::load_16khz_pretrained`](crate::kits::speech::silero_vad::SileroVad::load_16khz_pretrained),
    /// whose weights are small enough to ship inline.
    ///
    /// The returned config is **scanned from the checkpoint**, not assumed, so
    /// this also reports the geometry a caller needs — `n_mels` for the mel
    /// front end, `vocab_size` to tell a multilingual model from an
    /// English-only one. What a checkpoint cannot report is its audio front
    /// end and its token layout; the scanner declares those, upstream's for
    /// `OpenAI`'s.
    ///
    /// # Returns
    /// The loaded model, and the configuration inferred from its weights.
    ///
    /// # Errors
    /// [`BunsenError`](crate::errors::BunsenError) if the checkpoint cannot be
    /// read or does not scan as a Whisper model. A missing file means the
    /// cached asset was deleted after the build.
    ///
    /// # Note
    /// `OpenAI` ships these checkpoints in **fp16**, and the weights load at
    /// that precision, which is what the model then computes in. It is not
    /// what its interface speaks: the mel front end's log-mels are cast
    /// down on the way in and the logits are cast back up on the way out
    /// (see [`Whisper::dtype`]), so a caller stays in the backend's float
    /// and nothing here needs re-typing.
    ///
    /// To run the model itself at another precision — comparing against an
    /// f32 reference graph, or on a device without fp16 — map it:
    ///
    /// ```no_run
    /// # use burn::{module::Module, tensor::DType, backend::Wgpu};
    /// # use bunsen::{burner::module::DTypeMapper, kits::speech::whisper::Whisper};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Default::default();
    /// let (model, cfg) = Whisper::<Wgpu>::load_pretrained(&device)?;
    /// let model = model.map(&mut DTypeMapper::new(DType::F32));
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_pretrained(device: &B::Device) -> BunsenResult<(Self, WhisperApiConfig)> {
        PytorchWhisperScanner::new().load::<B, _>(bundled::base_pt(), device)
    }
}

/// The bundled vocabulary that matches a token layout: `multilingual.tiktoken`
/// for a multilingual layout, `gpt2.tiktoken` for an English-only one.
///
/// The two files number their tokens differently, and a checkpoint decoded
/// through the wrong one produces text that is wrong without being
/// obviously so; taking the layout, which comes from the checkpoint's
/// vocabulary size, keeps the pairing out of the caller's hands.
///
/// # Errors
/// [`BunsenError`](crate::errors::BunsenError) if the rank file cannot be
/// read or parsed. A missing file means the cached asset was deleted after
/// the build.
pub fn bundled_vocabulary(ids: &WhisperSpecialIds) -> BunsenResult<TiktokenRanks> {
    let path = if ids.is_multilingual() {
        bundled::multilingual_tiktoken()
    } else {
        bundled::gpt2_tiktoken()
    };
    TiktokenRanks::load(path)
}
