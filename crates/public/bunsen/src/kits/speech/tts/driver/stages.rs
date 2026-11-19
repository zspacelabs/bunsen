//! Request stages, run before the engine.

use alloc::{
    borrow::Cow,
    format,
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};
use core::fmt::Debug;

use super::TtsRequest;
use crate::{
    errors::BunsenResult,
    kits::support::text::{
        BunsenLanguageTag,
        TextNormalizer,
    },
};

/// What a stage did to a request that its caller should know: a change
/// beyond the text's normalization, such as input dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsStageNote {
    /// The stage's name.
    pub stage: String,

    /// What it did.
    pub note: String,
}

/// One step a [`TtsDriver`](super::TtsDriver) runs a request through before
/// the engine, in order: text normalization ([`TtsTextStage`]), dropping
/// input ([`StripInstruction`]), or a caller's own.
pub trait TtsRequestStage: Debug + Send + Sync {
    /// What it is called, for notes and reports.
    fn name(&self) -> String;

    /// Checks that the stage can run in `language`, before any request
    /// does.
    ///
    /// # Errors
    /// As the stage refuses the language.
    fn check_language(
        &self,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<()> {
        let _ = language;
        Ok(())
    }

    /// `request`, as this stage leaves it, and what it did worth saying.
    ///
    /// # Errors
    /// As the stage fails.
    fn apply(
        &self,
        request: TtsRequest,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<(TtsRequest, Vec<TtsStageNote>)>;
}

/// The text stage: a request's text through a [`TextNormalizer`], in the
/// request's language. An instruction's text is left as it is.
#[derive(Debug, Clone)]
pub struct TtsTextStage {
    normalizer: Arc<dyn TextNormalizer>,
}

impl TtsTextStage {
    /// The stage of `normalizer`.
    pub fn new(normalizer: Arc<dyn TextNormalizer>) -> Self {
        Self { normalizer }
    }

    /// Its normalizer.
    pub fn normalizer(&self) -> &Arc<dyn TextNormalizer> {
        &self.normalizer
    }
}

impl TtsRequestStage for TtsTextStage {
    fn name(&self) -> String {
        format!("text: {}", self.normalizer.name())
    }

    fn check_language(
        &self,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<()> {
        self.normalizer.check_language(language)
    }

    fn apply(
        &self,
        mut request: TtsRequest,
        language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<(TtsRequest, Vec<TtsStageNote>)> {
        if let Cow::Owned(text) = self.normalizer.normalize(&request.text, language)? {
            request.text = text;
        }
        Ok((request, Vec::new()))
    }
}

/// Drops a request's instruction, with a note saying so: the driver's
/// opt-in for an engine that takes none
/// ([`TtsDriverConfig::strip_instruction`](super::TtsDriverConfig::strip_instruction)).
#[derive(Debug, Clone, Copy, Default)]
pub struct StripInstruction;

impl TtsRequestStage for StripInstruction {
    fn name(&self) -> String {
        "strip-instruction".into()
    }

    fn apply(
        &self,
        mut request: TtsRequest,
        _language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<(TtsRequest, Vec<TtsStageNote>)> {
        let notes = match request.instruction.take() {
            Some(instruction) => vec![TtsStageNote {
                stage: self.name(),
                note: format!("dropped the instruction {instruction:?}"),
            }],
            None => Vec::new(),
        };
        Ok((request, notes))
    }
}
