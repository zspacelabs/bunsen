//! # The Whisper bundle
//!
//! What a Whisper pretrained builds, and what the driver holds behind an
//! `Arc`: the model, its token layout, and, when the pretrained has one,
//! the base vocabulary a decode needs for text and for upstream's default
//! suppress list. Several drivers over one loaded model share the `Arc`.
//!
//! A bundle without a vocabulary is ids only, which is still a complete
//! result: the layout needs no file, and a driver over it emits ids and
//! applies no suppress list.

use std::{
    fmt,
    sync::Arc,
};

use burn::prelude::Backend;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::whisper::{
            Whisper,
            WhisperMeta,
            blocks::{
                AudioEncoderMeta,
                TextDecoderMeta,
            },
            driver::WhisperTokenLayout,
            logit_filters::{
                LogitFilter,
                default_filters,
            },
        },
        tokens::TiktokenRanks,
    },
};

/// A loaded Whisper model with its token layout and, when it has one, its
/// vocabulary.
#[derive(Debug)]
pub struct WhisperBundle<B: Backend> {
    /// The model.
    pub model: Whisper<B>,

    /// The token layout the model's vocabulary follows.
    pub layout: WhisperTokenLayout,

    /// The base vocabulary, when the bundle has one. `None` is ids only: no
    /// text, and no default suppress list.
    pub ranks: Option<TiktokenRanks>,
}

impl<B: Backend> WhisperBundle<B> {
    /// A bundle over a model with an explicit token layout, and no
    /// vocabulary.
    pub fn new(
        model: Whisper<B>,
        layout: WhisperTokenLayout,
    ) -> Self {
        Self {
            model,
            layout,
            ranks: None,
        }
    }

    /// A bundle over a model, deriving the token layout from the model's
    /// vocabulary size, and no vocabulary.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the vocabulary size is not a Whisper
    /// layout.
    pub fn from_model(model: Whisper<B>) -> BunsenResult<Self> {
        let layout = model.token_layout().policy_for_vocab(model.vocab_size())?;
        Ok(Self::new(model, layout))
    }

    /// Attaches the base vocabulary.
    pub fn with_ranks(
        mut self,
        ranks: TiktokenRanks,
    ) -> Self {
        self.ranks = Some(ranks);
        self
    }

    /// Checks the parts agree: every id of the layout is inside the
    /// model's vocabulary, and the vocabulary, if there is one, has the
    /// layout's base ranks.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] naming the disagreement.
    pub fn validate(&self) -> BunsenResult<()> {
        let ids = self.layout.ids();
        if ids.n_vocab() > self.model.vocab_size() {
            return Err(BunsenError::Invalid(format!(
                "the token layout has {} ids but the model's vocabulary has {}",
                ids.n_vocab(),
                self.model.vocab_size(),
            )));
        }
        if let Some(ranks) = &self.ranks {
            self.layout.token_spans(ranks)?;
        }
        Ok(())
    }

    /// Upstream's default logit filters over the vocabulary: the blank and
    /// the non-speech suppress list. None without a vocabulary.
    pub fn default_filters(&self) -> Vec<Arc<dyn LogitFilter<B>>> {
        match &self.ranks {
            Some(ranks) => default_filters::<B>(ranks, self.layout.ids()),
            None => Vec::new(),
        }
    }

    /// A detokenizer over the vocabulary, if there is one.
    ///
    /// # Errors
    /// As [`WhisperTokenLayout::detokenizer`].
    #[cfg(feature = "tokenizer")]
    pub fn detokenizer(
        &self
    ) -> BunsenResult<Option<crate::kits::tokens::WordchipperDetokenizer<u16>>> {
        self.ranks
            .as_ref()
            .map(|ranks| self.layout.detokenizer(ranks))
            .transpose()
    }
}

impl<B: Backend> fmt::Display for WhisperBundle<B> {
    /// One line of what was loaded: `80 mels, vocabulary 51865, d_model
    /// 512, 6 + 6 layers, 50257 ranks`, or `ids only` in place of the
    /// ranks for a bundle without a vocabulary.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "{} mels, vocabulary {}, d_model {}, {} + {} layers",
            self.model.n_mels(),
            self.model.vocab_size(),
            self.model.d_model(),
            self.model.encoder().n_layers(),
            self.model.decoder().n_layers(),
        )?;
        match &self.ranks {
            Some(ranks) => write!(f, ", {} ranks", ranks.len()),
            None => f.write_str(", ids only"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::whisper::{
            WhisperGeometry,
            blocks::WhisperTokenLayoutConfig,
        },
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    /// A model with a 64-id vocabulary, which is not a Whisper layout.
    fn tiny_model() -> Whisper<CpuBackend> {
        WhisperGeometry::openai(8, 64, 64, 1, 1)
            .to_api_config()
            .try_init(&default_device())
            .unwrap()
    }

    /// A layout that fits 64 ids: two base ranks, one language, one
    /// timestamp.
    fn tiny_layout() -> WhisperTokenLayout {
        WhisperTokenLayoutConfig::new()
            .with_languages(vec!["en".to_string()])
            .with_timestamp_tokens(1)
            .policy(2, 1)
            .unwrap()
    }

    /// Without a vocabulary the bundle is ids only: no filters, no text.
    #[test]
    fn test_ids_only() {
        let bundle = WhisperBundle::new(tiny_model(), tiny_layout());
        let shown = bundle.to_string();
        assert!(shown.contains(" mels, vocabulary "), "{shown}");
        assert!(shown.ends_with(" layers, ids only"), "{shown}");
        bundle.validate().unwrap();
        assert!(bundle.ranks.is_none());
        assert!(bundle.default_filters().is_empty());
        #[cfg(feature = "tokenizer")]
        assert!(bundle.detokenizer().unwrap().is_none());
    }

    /// A vocabulary with the layout's base ranks gives filters and text; one
    /// with another count is refused.
    #[test]
    fn test_with_a_vocabulary() {
        let ranks = TiktokenRanks::parse("IA== 0\nb2s= 1\n").unwrap();
        let bundle = WhisperBundle::new(tiny_model(), tiny_layout()).with_ranks(ranks);
        assert!(bundle.to_string().ends_with(" ranks"), "{bundle}");
        bundle.validate().unwrap();
        assert!(!bundle.default_filters().is_empty());
        #[cfg(feature = "tokenizer")]
        {
            use crate::kits::tokens::Detokenizer;
            assert_eq!(
                bundle
                    .detokenizer()
                    .unwrap()
                    .unwrap()
                    .detokenize(&[1])
                    .unwrap(),
                "ok"
            );
        }

        let short = WhisperBundle::new(tiny_model(), tiny_layout())
            .with_ranks(TiktokenRanks::parse("IA== 0\n").unwrap());
        assert!(matches!(short.validate(), Err(BunsenError::Invalid(_))));
    }

    /// A layout with more ids than the model has is refused, and a model
    /// whose vocabulary is not a Whisper size cannot derive one.
    #[test]
    fn test_layout_must_fit_the_model() {
        let too_big = WhisperBundle::new(
            tiny_model(),
            WhisperTokenLayout::from_vocab_size(51865).unwrap(),
        );
        assert!(matches!(too_big.validate(), Err(BunsenError::Invalid(m)) if m.contains("51865")));
        assert!(matches!(
            WhisperBundle::from_model(tiny_model()),
            Err(BunsenError::Invalid(_))
        ));
    }
}
