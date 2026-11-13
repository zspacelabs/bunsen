//! ZONOS3-Mini's prompt: the rows the language model reads before it
//! generates, and the delay pattern its audio rows follow.
//!
//! A row is `n_codebooks` audio ids and a text id. In order:
//!
//! 1. the speaker: an all-pad row whose embedding is replaced by the projected
//!    speaker embedding ([`Zonos3Speaker::Embedding`]), or a reference prefix,
//!    raw codec codes laid out as audio rows ([`Zonos3Speaker::Reference`]), or
//!    nothing;
//! 2. one all-pad row per condition, each replaced by its table's row for its
//!    bucket (the unknown bucket when the request gives none);
//! 3. the text: `[BOS] + ids + [EOS]`, the audio columns pad;
//! 4. the silence prefill: copies of the codec's silence frame, as the first
//!    audio rows of the stream.
//!
//! Audio rows are delayed ([`shear`]): codebook `j` of frame `f` sits in row
//! `f + j`, pad above the diagonal. A reference prefix is flushed with `n -
//! 1` pad rows so its last frame is whole; the silence prefill is not, and
//! the generated rows continue its pattern. The text column of every
//! non-text row is the text pad. Text is tokenized already normalized: the
//! kit does no text normalization.

use alloc::vec::Vec;

use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
    },
    tensor::Device,
};

use super::lm::{
    Zonos3LmConfig,
    Zonos3RowConditioning,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::tokens::Tokenizer,
};

/// The text's begin-of-sequence id.
pub const ZONOS3_BOS: u32 = 2;

/// The text's end-of-sequence id.
pub const ZONOS3_EOS: u32 = 3;

/// `frames` (`[F][n]`) delayed: `F` rows, row `t` holding codebook `j` of
/// frame `t - j`, `pad` where `t < j`.
pub fn shear(
    frames: &[Vec<u32>],
    pad: u32,
) -> Vec<Vec<u32>> {
    (0..frames.len())
        .map(|t| {
            (0..frames[t].len())
                .map(|j| if t >= j { frames[t - j][j] } else { pad })
                .collect()
        })
        .collect()
}

/// `rows` (`[R][n]`) undelayed: `R` frames, frame `f` holding codebook `j`
/// of row `f + j`, `pad` past the last row. The inverse of [`shear`] on the
/// frames whose codebooks are all in `rows`.
pub fn unshear(
    rows: &[Vec<u32>],
    pad: u32,
) -> Vec<Vec<u32>> {
    (0..rows.len())
        .map(|f| {
            (0..rows[f].len())
                .map(|j| rows.get(f + j).map_or(pad, |r| r[j]))
                .collect()
        })
        .collect()
}

/// The bucket of a condition's value: how many of its ascending `edges`
/// are at or below it (`bisect_right`).
pub fn condition_bucket(
    edges: &[f64],
    value: f64,
) -> usize {
    edges.iter().filter(|&&e| e <= value).count()
}

/// The text rows' ids of normalized `text`: `[BOS] + encode(text) + [EOS]`,
/// through the `-hf` `tokenizer.json`, whose ids are already offset past the
/// four special ids.
///
/// # Errors
/// As the tokenizer's.
pub fn zonos3_text_ids(
    tokenizer: &impl Tokenizer,
    text: &str,
) -> BunsenResult<Vec<u32>> {
    let ids = tokenizer.encode(text)?;
    let mut out = Vec::with_capacity(ids.len() + 2);
    out.push(ZONOS3_BOS);
    for id in ids {
        out.push(
            u32::try_from(id).map_err(|_| {
                BunsenError::invalid_resource(format!("a text id {id} outside u32"))
            })?,
        );
    }
    out.push(ZONOS3_EOS);
    Ok(out)
}

/// The prompt's speaker.
#[derive(Clone, Debug, PartialEq)]
pub enum Zonos3Speaker {
    /// None: an unconditioned voice.
    None,

    /// A speaker embedding, projected into one row.
    Embedding(Vec<f32>),

    /// A reference prefix: raw codec codes, `[F][n]`, kept to the first
    /// [`max_reference_frames`](Zonos3PromptConfig::max_reference_frames).
    Reference(Vec<Vec<u32>>),
}

/// What [`Zonos3PromptConfig::build`] lays out.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3PromptRequest {
    /// The text rows' ids, `[BOS] + ids + [EOS]` ([`zonos3_text_ids`]).
    pub text_ids: Vec<u32>,

    /// The speaker.
    pub speaker: Zonos3Speaker,

    /// Each condition's bucket; `None` (or a short list) is the unknown
    /// bucket.
    pub condition_buckets: Vec<Option<usize>>,

    /// The silence prefill's frames.
    pub silence_prefill_frames: usize,
}

/// A laid-out prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Prompt {
    /// The rows, `[R][n + 1]`.
    pub rows: Vec<Vec<u32>>,

    /// The speaker row and its embedding, in embedding mode.
    pub speaker: Option<(usize, Vec<f32>)>,

    /// The first condition row.
    pub first_condition_row: usize,

    /// Each condition's bucket, the unknown bucket resolved.
    pub condition_buckets: Vec<usize>,

    /// The silence prefill's rows, at the end.
    pub silence_rows: usize,
}

impl Zonos3Prompt {
    /// The rows as the language model's input, `[1, R, n + 1]`.
    pub fn ids(
        &self,
        device: &Device,
    ) -> Tensor<3, Int> {
        let width = self.rows.first().map_or(0, Vec::len);
        let flat: Vec<i64> = self
            .rows
            .iter()
            .flat_map(|r| r.iter().map(|&id| i64::from(id)))
            .collect();
        Tensor::from_data(TensorData::new(flat, [1, self.rows.len(), width]), device)
    }

    /// The replaced rows, for the language model.
    pub fn conditioning(
        &self,
        device: &Device,
    ) -> Zonos3RowConditioning {
        let n = self.condition_buckets.len();
        Zonos3RowConditioning {
            speaker: self.speaker.as_ref().map(|(row, embedding)| {
                (
                    *row,
                    Tensor::from_data(
                        TensorData::new(embedding.clone(), [1, embedding.len()]),
                        device,
                    ),
                )
            }),
            conditions: (n > 0).then(|| {
                let buckets: Vec<i64> = self.condition_buckets.iter().map(|&b| b as i64).collect();
                (
                    self.first_condition_row,
                    Tensor::from_data(TensorData::new(buckets, [1, n]), device),
                )
            }),
        }
    }
}

/// What a prompt's layout needs to know of the model and its release.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3PromptConfig {
    /// The audio codebooks, `n`.
    pub n_codebooks: usize,

    /// Codes per codebook.
    pub codebook_size: usize,

    /// The audio pad id.
    pub audio_pad: u32,

    /// The text pad id: the text vocabulary's size.
    pub text_pad: u32,

    /// Each condition's unknown bucket: its bucket count.
    pub unknown_buckets: Vec<usize>,

    /// The codec's silence frame, `[n]`.
    pub silence_frame: Vec<u32>,

    /// The most frames of a reference prefix.
    pub max_reference_frames: usize,
}

impl Zonos3PromptConfig {
    /// The layout for a model of `lm`'s geometry: its text pad, `text_pad`
    /// (the `lm` section's `text_vocab`); the release's silence frame and
    /// reference limit.
    pub fn new(
        lm: &Zonos3LmConfig,
        text_pad: u32,
        silence_frame: Vec<u32>,
        max_reference_frames: usize,
    ) -> Self {
        Self {
            n_codebooks: lm.n_codebooks,
            codebook_size: lm.codebook_size,
            audio_pad: lm.audio_pad_id() as u32,
            text_pad,
            unknown_buckets: lm.condition_buckets.clone(),
            silence_frame,
            max_reference_frames,
        }
    }

    /// A row of pads: every audio column the audio pad, the text column the
    /// text pad.
    fn pad_row(&self) -> Vec<u32> {
        let mut row = vec![self.audio_pad; self.n_codebooks];
        row.push(self.text_pad);
        row
    }

    /// `audio` (`[R][n]`) rows with the text pad appended.
    fn audio_rows(
        &self,
        audio: Vec<Vec<u32>>,
    ) -> impl Iterator<Item = Vec<u32>> + '_ {
        audio.into_iter().map(|mut row| {
            row.push(self.text_pad);
            row
        })
    }

    /// The prompt of `request`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for reference
    /// codes that are empty, not `n` wide, or past the codebooks' range; a
    /// bucket past its condition's; more buckets than conditions; or a
    /// silence frame not `n` wide.
    pub fn build(
        &self,
        request: &Zonos3PromptRequest,
    ) -> BunsenResult<Zonos3Prompt> {
        let n = self.n_codebooks;
        let mut rows = Vec::new();
        let mut speaker = None;
        match &request.speaker {
            Zonos3Speaker::None => {}
            Zonos3Speaker::Embedding(embedding) => {
                speaker = Some((rows.len(), embedding.clone()));
                rows.push(self.pad_row());
            }
            Zonos3Speaker::Reference(codes) => {
                if codes.is_empty()
                    || codes.iter().any(|f| {
                        f.len() != n || f.iter().any(|&c| c as usize >= self.codebook_size)
                    })
                {
                    return Err(BunsenError::illegal(format!(
                        "reference codes must be non-empty frames of {n} codes below {}",
                        self.codebook_size
                    )));
                }
                let mut frames: Vec<Vec<u32>> = codes
                    .iter()
                    .take(self.max_reference_frames)
                    .cloned()
                    .collect();
                frames.extend((0..n - 1).map(|_| vec![self.audio_pad; n]));
                rows.extend(self.audio_rows(shear(&frames, self.audio_pad)));
            }
        }

        if request.condition_buckets.len() > self.unknown_buckets.len() {
            return Err(BunsenError::illegal(format!(
                "{} condition buckets for {} conditions",
                request.condition_buckets.len(),
                self.unknown_buckets.len()
            )));
        }
        let first_condition_row = rows.len();
        let mut condition_buckets = Vec::with_capacity(self.unknown_buckets.len());
        for (i, &unknown) in self.unknown_buckets.iter().enumerate() {
            let bucket = request
                .condition_buckets
                .get(i)
                .copied()
                .flatten()
                .unwrap_or(unknown);
            if bucket > unknown {
                return Err(BunsenError::illegal(format!(
                    "condition {i}'s bucket {bucket} of {unknown}"
                )));
            }
            condition_buckets.push(bucket);
            rows.push(self.pad_row());
        }

        rows.extend(request.text_ids.iter().map(|&id| {
            let mut row = vec![self.audio_pad; n];
            row.push(id);
            row
        }));

        let silence_rows = request.silence_prefill_frames;
        if silence_rows > 0 {
            if self.silence_frame.len() != n {
                return Err(BunsenError::illegal(format!(
                    "a silence frame of {} codes for {n} codebooks",
                    self.silence_frame.len()
                )));
            }
            let frames = vec![self.silence_frame.clone(); silence_rows];
            rows.extend(self.audio_rows(shear(&frames, self.audio_pad)));
        }

        Ok(Zonos3Prompt {
            rows,
            speaker,
            first_condition_row,
            condition_buckets,
            silence_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: u32 = 9;

    fn config() -> Zonos3PromptConfig {
        Zonos3PromptConfig {
            n_codebooks: 3,
            codebook_size: 8,
            audio_pad: PAD,
            text_pad: 20,
            unknown_buckets: vec![4, 6],
            silence_frame: vec![1, 2, 3],
            max_reference_frames: 2,
        }
    }

    /// Codebook `j` of frame `f` sits in row `f + j`, pad above the
    /// diagonal; unshearing gives the frames back.
    #[test]
    fn test_shear() {
        let frames = vec![vec![10, 11, 12], vec![20, 21, 22], vec![30, 31, 32]];
        let rows = shear(&frames, PAD);
        assert_eq!(
            rows,
            [vec![10, PAD, PAD], vec![20, 11, PAD], vec![30, 21, 12]]
        );
        assert_eq!(unshear(&rows, PAD)[0], [10, 11, 12]);
        assert_eq!(unshear(&rows, PAD)[1], [20, 21, PAD]);
    }

    #[test]
    fn test_condition_bucket() {
        let edges = [-30.0, -20.0, -10.0];
        assert_eq!(condition_bucket(&edges, -40.0), 0);
        assert_eq!(condition_bucket(&edges, -20.0), 2);
        assert_eq!(condition_bucket(&edges, -18.5), 2);
        assert_eq!(condition_bucket(&edges, 0.0), 3);
    }

    /// An embedding prompt: the speaker row, the conditions (the second
    /// unknown), the text, the silence prefill sheared without a flush.
    #[test]
    fn test_embedding_prompt() {
        let prompt = config()
            .build(&Zonos3PromptRequest {
                text_ids: vec![2, 7, 3],
                speaker: Zonos3Speaker::Embedding(vec![0.5, -0.5]),
                condition_buckets: vec![Some(1)],
                silence_prefill_frames: 2,
            })
            .unwrap();
        let p = vec![PAD, PAD, PAD, 20];
        assert_eq!(
            prompt.rows,
            [
                p.clone(),
                p.clone(),
                p.clone(),
                vec![PAD, PAD, PAD, 2],
                vec![PAD, PAD, PAD, 7],
                vec![PAD, PAD, PAD, 3],
                vec![1, PAD, PAD, 20],
                vec![1, 2, PAD, 20],
            ]
        );
        assert_eq!(prompt.speaker, Some((0, vec![0.5, -0.5])));
        assert_eq!(prompt.first_condition_row, 1);
        assert_eq!(prompt.condition_buckets, [1, 6]);
        assert_eq!(prompt.silence_rows, 2);
        let device = crate::support::testing::cpu_device();
        assert_eq!(prompt.ids(&device).dims(), [1, 8, 4]);
        let rows = prompt.conditioning(&device);
        assert_eq!(rows.speaker.unwrap().0, 0);
        assert_eq!(
            rows.conditions
                .unwrap()
                .1
                .into_data()
                .try_into_vec_as::<i64>()
                .unwrap(),
            [1, 6]
        );
    }

    /// A reference prefix: the first frames only, sheared and flushed with
    /// pad rows so its last frame is whole; no speaker row; no silence.
    #[test]
    fn test_reference_prompt() {
        let prompt = config()
            .build(&Zonos3PromptRequest {
                text_ids: vec![2, 3],
                speaker: Zonos3Speaker::Reference(vec![
                    vec![1, 2, 3],
                    vec![4, 5, 6],
                    vec![7, 0, 7],
                ]),
                condition_buckets: vec![],
                silence_prefill_frames: 0,
            })
            .unwrap();
        assert_eq!(
            prompt.rows[..4],
            [
                vec![1, PAD, PAD, 20],
                vec![4, 2, PAD, 20],
                vec![PAD, 5, 3, 20],
                vec![PAD, PAD, 6, 20],
            ]
        );
        assert_eq!(prompt.first_condition_row, 4);
        assert_eq!(prompt.rows.len(), 4 + 2 + 2);
        assert!(prompt.speaker.is_none());
        assert_eq!(prompt.condition_buckets, [4, 6]);
    }

    #[test]
    fn test_refusals() {
        let request = |speaker, buckets| Zonos3PromptRequest {
            text_ids: vec![2, 3],
            speaker,
            condition_buckets: buckets,
            silence_prefill_frames: 0,
        };
        for bad in [
            request(Zonos3Speaker::Reference(vec![]), vec![]),
            request(Zonos3Speaker::Reference(vec![vec![1, 2]]), vec![]),
            request(Zonos3Speaker::Reference(vec![vec![1, 2, 8]]), vec![]),
            request(Zonos3Speaker::None, vec![Some(5)]),
            request(Zonos3Speaker::None, vec![None, None, None]),
        ] {
            assert!(config().build(&bad).is_err(), "{bad:?}");
        }
    }
}
