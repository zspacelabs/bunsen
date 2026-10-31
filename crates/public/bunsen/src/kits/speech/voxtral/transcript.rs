//! A decoded Voxtral Realtime utterance, as timed words.
use super::VoxtralTokenLayout;
use crate::{
    errors::BunsenResult,
    kits::tokens::TekkenVocabulary,
};

/// One word of a [`VoxtralTranscript`]: the text tokens after a
/// `[STREAMING_WORD]`, up to the next.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxtralWord {
    /// The word, its leading space included as the vocabulary spells it.
    pub text: String,

    /// Its text ids.
    pub ids: Vec<i64>,

    /// The sequence index of its `[STREAMING_WORD]`: the decoder position
    /// it is written at.
    pub position: usize,

    /// When it starts in the audio, in seconds
    /// ([`VoxtralTokenLayout::word_start_seconds`]): to within a position
    /// (80 ms).
    pub start: f64,
}

/// What a Voxtral Realtime decode wrote: its ids, the words they spell with
/// their times, and the text.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxtralTranscript {
    /// The whole sequence, the prompt first.
    pub ids: Vec<i64>,

    /// The words, in order. Text the model writes before its first
    /// `[STREAMING_WORD]` is a word of its own.
    pub words: Vec<VoxtralWord>,

    /// The text, without its leading space.
    pub text: String,
}

impl VoxtralTranscript {
    /// Reads a decode's ids: everything after the prompt, the control
    /// tokens dropped, split into words at each `[STREAMING_WORD]`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if a text id is outside the vocabulary.
    pub fn parse(
        ids: Vec<i64>,
        layout: &VoxtralTokenLayout,
        vocab: &TekkenVocabulary,
    ) -> BunsenResult<Self> {
        let start = layout.prompt().len().min(ids.len());
        let word = layout.config().streaming_word as i64;
        let mut words: Vec<VoxtralWord> = Vec::new();
        let mut open: Option<(usize, Vec<i64>)> = None;
        let mut close = |open: &mut Option<(usize, Vec<i64>)>| -> BunsenResult<()> {
            if let Some((position, ids)) = open.take()
                && !ids.is_empty()
            {
                let text = vocab.text(&ids.iter().map(|&i| i as usize).collect::<Vec<_>>())?;
                words.push(VoxtralWord {
                    text,
                    ids,
                    position,
                    start: layout.word_start_seconds(position),
                });
            }
            Ok(())
        };
        for (position, &id) in ids.iter().enumerate().skip(start) {
            if id == word {
                close(&mut open)?;
                open = Some((position, Vec::new()));
            } else if !layout.is_control(id) {
                open.get_or_insert_with(|| (position, Vec::new()))
                    .1
                    .push(id);
            }
        }
        close(&mut open)?;
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<String>()
            .trim_start()
            .to_string();
        Ok(Self { ids, words, text })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::voxtral::VoxtralTokenLayoutConfig;

    #[test]
    fn test_parse() {
        let vocab = TekkenVocabulary::from_tekken_json(
            r#"{
                "config": {"default_vocab_size": 37, "default_num_special_tokens": 34},
                "vocab": [
                    {"rank": 0, "token_bytes": "IElu"},
                    {"rank": 1, "token_bytes": "IHRoZQ=="},
                    {"rank": 2, "token_bytes": "LA=="}
                ],
                "special_tokens": [
                    {"rank": 1, "token_str": "<s>"}, {"rank": 2, "token_str": "</s>"},
                    {"rank": 32, "token_str": "[STREAMING_PAD]"},
                    {"rank": 33, "token_str": "[STREAMING_WORD]"}
                ]
            }"#,
        )
        .unwrap();
        let layout = VoxtralTokenLayoutConfig::new()
            .with_left_pad_positions(1)
            .with_delay_ms(80)
            .init();
        // The prompt (<s> and two pads), then "In", a pad, "the" and ",".
        let ids = vec![1, 32, 32, 33, 34, 32, 33, 35, 36, 2];
        let t = VoxtralTranscript::parse(ids.clone(), &layout, &vocab).unwrap();
        assert_eq!(t.ids, ids);
        assert_eq!(t.text, "In the,");
        assert_eq!(t.words.len(), 2);
        assert_eq!(t.words[0].text, " In");
        assert_eq!(t.words[0].position, 3);
        assert_eq!(t.words[1].ids, [35, 36]);
        assert_eq!(t.words[1].position, 6);
        // Position 6, less a left pad and a delay of one: the fifth position.
        assert!((t.words[1].start - 0.32).abs() < 1e-9);
    }
}
