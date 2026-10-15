//! The models `phonon` serves, behind one seam.
use std::sync::Arc;

use bunsen::{
    data::pretrained::{
        Pretrained,
        PretrainedCache,
    },
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::speech::{
        parakeet::{
            ParakeetBundle,
            ParakeetFrontEndMeta,
            pretrained::{
                PARAKEET_KIT,
                default_parakeet_factory,
            },
        },
        qwen3_asr::{
            Qwen3AsrBundle,
            Qwen3AsrFrontEndMeta,
            pretrained::{
                QWEN3_ASR_KIT,
                default_qwen3_asr_factory,
            },
        },
    },
};
use burn::tensor::Device;

/// The model `--model` names when it is not given: Phonon-2.
pub const DEFAULT_MODEL: &str = bunsen::kits::speech::parakeet::pretrained::DEFAULT_PARAKEET_MODEL;

/// The kit a model name belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kit {
    /// NVIDIA's Parakeet TDT: Phonon-2, and its base model.
    Parakeet,

    /// Qwen's Qwen3-ASR: Phonon-1's three builds, and their base model.
    Qwen3Asr,
}

impl Kit {
    /// Every kit, in the order a name is looked up.
    pub const ALL: [Kit; 2] = [Kit::Parakeet, Kit::Qwen3Asr];

    /// The kit's segment of the pretrained cache.
    pub fn cache_kit(self) -> &'static str {
        match self {
            Kit::Parakeet => PARAKEET_KIT,
            Kit::Qwen3Asr => QWEN3_ASR_KIT,
        }
    }

    /// The kit's rows, with their providers.
    pub fn list(self) -> BunsenResult<Vec<(String, Pretrained)>> {
        Ok(match self {
            Kit::Parakeet => default_parakeet_factory()?.list(),
            Kit::Qwen3Asr => default_qwen3_asr_factory()?.list(),
        })
    }

    /// The row `name` names in this kit, if it names one.
    pub fn lookup(
        self,
        name: &str,
    ) -> BunsenResult<Option<Pretrained>> {
        let found = match self {
            Kit::Parakeet => default_parakeet_factory()?.lookup(name),
            Kit::Qwen3Asr => default_qwen3_asr_factory()?.lookup(name),
        };
        Ok(found.ok().map(|(_, row)| row))
    }

    /// The kit `name` belongs to, and its row.
    ///
    /// # Errors
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when no kit knows
    /// the name.
    pub fn of(name: &str) -> BunsenResult<(Kit, Pretrained)> {
        for kit in Self::ALL {
            if let Some(row) = kit.lookup(name)? {
                return Ok((kit, row));
            }
        }
        let mut names = Vec::new();
        for kit in Self::ALL {
            names.extend(
                kit.list()?
                    .into_iter()
                    .map(|(p, row)| format!("{p}:{}", row.name)),
            );
        }
        Err(BunsenError::lookup(
            LookupError::missing("model", name).with_candidates(names),
        ))
    }
}

/// A transcript, whichever model made it.
#[derive(Clone, Debug)]
pub struct Transcript {
    /// The text.
    pub text: String,

    /// The language the model named, if it names one.
    pub language: Option<String>,

    /// Each word with its start and end in seconds, if the model times them.
    pub words: Option<Vec<(String, f64, f64)>>,
}

/// A loaded model of either kit.
#[derive(Clone, Debug)]
pub enum Transcriber {
    /// A Parakeet TDT bundle.
    Parakeet(Arc<ParakeetBundle>),

    /// A Qwen3-ASR bundle.
    Qwen3Asr(Arc<Qwen3AsrBundle>),
}

impl Transcriber {
    /// Loads `name` through `cache`, from whichever kit knows it.
    pub fn load(
        name: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Self> {
        Ok(match Kit::of(name)?.0 {
            Kit::Parakeet => {
                Self::Parakeet(default_parakeet_factory()?.load_bundle(name, cache, device)?)
            }
            Kit::Qwen3Asr => {
                Self::Qwen3Asr(default_qwen3_asr_factory()?.load_bundle(name, cache, device)?)
            }
        })
    }

    /// The sample rate the model reads, in Hz.
    pub fn sample_rate(&self) -> usize {
        match self {
            Self::Parakeet(b) => b.front_end.sample_rate(),
            Self::Qwen3Asr(b) => b.front_end.sample_rate(),
        }
    }

    /// The longest window the model takes well, in seconds: Parakeet's
    /// encoder grows with the square of its input, and Qwen3-ASR was
    /// trained on at most 30 s.
    pub fn default_window_seconds(&self) -> f64 {
        match self {
            Self::Parakeet(_) => 60.0,
            Self::Qwen3Asr(_) => Qwen3AsrBundle::MAX_WINDOW_SECONDS as f64,
        }
    }

    /// Transcribes one recording in windows of at most `window` samples,
    /// `batch` windows at a time where the model batches.
    pub fn transcribe(
        &self,
        samples: &[f32],
        window: usize,
        batch: usize,
        device: &Device,
    ) -> BunsenResult<Transcript> {
        match self {
            Self::Parakeet(b) => {
                let t = b.transcribe_long(samples, window, batch, device)?;
                let words = t
                    .words(&b.vocab, b.frame_seconds())?
                    .into_iter()
                    .map(|w| (w.text, w.start, w.end))
                    .collect();
                Ok(Transcript {
                    text: t.text,
                    language: None,
                    words: Some(words),
                })
            }
            Self::Qwen3Asr(b) => {
                let t = b.transcribe_long(samples, window, device)?;
                Ok(Transcript {
                    text: t.text,
                    language: t.language,
                    words: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of every kit is found by its qualified name in its own kit,
    /// and in no other; the bare default is Phonon-2.
    #[test]
    fn test_names_route_to_their_kit() {
        for kit in Kit::ALL {
            for (provider, row) in kit.list().unwrap() {
                let qualified = format!("{provider}:{}", row.name);
                assert_eq!(Kit::of(&qualified).unwrap().0, kit, "{qualified}");
                for other in Kit::ALL.into_iter().filter(|&k| k != kit) {
                    assert!(other.lookup(&row.name).unwrap().is_none(), "{}", row.name);
                }
            }
        }
        assert_eq!(Kit::of(DEFAULT_MODEL).unwrap().0, Kit::Parakeet);
        assert_eq!(Kit::of("phonon-1").unwrap().0, Kit::Qwen3Asr);
        assert!(Kit::of("whisper/tiny").is_err());
    }
}
