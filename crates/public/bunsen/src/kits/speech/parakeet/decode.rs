//! Greedy token-and-duration transducer decoding.
use burn::{
    Tensor,
    config::Config,
    prelude::{
        Bool,
        Int,
        TensorData,
    },
};

use super::blocks::{
    ParakeetTdt,
    ParakeetTdtMeta,
};

/// One decoded item: the tokens a decode emitted, and when.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TdtHypothesis {
    /// The emitted tokens, blanks dropped.
    pub tokens: Vec<usize>,

    /// The encoder frame each token was emitted at.
    pub frames: Vec<usize>,

    /// The duration, in encoder frames, the joint chose with each token.
    pub durations: Vec<usize>,
}

/// Config for [`TdtGreedyDecoder`].
#[derive(Config, Debug)]
pub struct TdtGreedyDecoderConfig {
    /// The most tokens emitted at one frame before the decode moves on by
    /// one frame. `None` lets the joint's durations alone decide; a joint
    /// that keeps choosing duration 0 then never finishes, which `NeMo` and
    /// `onnx-asr` guard against with 10.
    #[config(default = "Some(10)")]
    pub max_symbols_per_frame: Option<usize>,
}

impl TdtGreedyDecoderConfig {
    /// Builds the decoder.
    pub fn init(&self) -> TdtGreedyDecoder {
        TdtGreedyDecoder {
            config: self.clone(),
        }
    }
}

/// Greedy decoding of a token-and-duration transducer, batched.
///
/// Each item keeps a frame index `t`, starting at 0, and the prediction
/// network's output for the tokens emitted so far, starting from the blank.
/// At each step the joint scores frame `t` against that output: the argmax
/// over the tokens (blank included) is emitted unless it is the blank, which
/// is when the prediction network advances; the argmax over the durations
/// advances `t`, by at least one frame when the token is the blank. An item
/// is done when `t` passes its length. This is the decode `transformers`'
/// `ParakeetForTDT.generate` and `NeMo`'s greedy TDT perform.
///
/// Built by [`TdtGreedyDecoderConfig`].
#[derive(Clone, Debug)]
pub struct TdtGreedyDecoder {
    config: TdtGreedyDecoderConfig,
}

impl TdtGreedyDecoder {
    /// The config this was built from.
    pub fn config(&self) -> &TdtGreedyDecoderConfig {
        &self.config
    }

    /// Decodes a batch.
    ///
    /// # Arguments
    /// * `model` - the model whose prediction network and joint decode.
    /// * `encoded` - `[batch, frames, d_pred]`, the projected encoder output
    ///   ([`ParakeetTdt::encode`]).
    /// * `lengths` - each item's valid frame count.
    ///
    /// # Returns
    /// One [`TdtHypothesis`] per item.
    pub fn decode(
        &self,
        model: &ParakeetTdt,
        encoded: Tensor<3>,
        lengths: &[usize],
    ) -> Vec<TdtHypothesis> {
        let [batch, frames, d] = encoded.dims();
        assert_eq!(lengths.len(), batch, "one length per batch item");
        let device = encoded.device();
        let blank = model.blank_id();
        let n_vocab = model.n_vocab();
        let durations = model.durations().to_vec();

        let mut hypotheses = vec![TdtHypothesis::default(); batch];
        let mut t = vec![0usize; batch];
        let mut at_frame = vec![0usize; batch];
        let ids = |v: &[usize]| {
            let v: Vec<i64> = v.iter().map(|&x| x as i64).collect();
            Tensor::<1, Int>::from_data(TensorData::new(v, [batch]), &device)
        };

        let (mut prediction, mut state) = model.decoder.step(
            ids(&vec![blank; batch]),
            model.decoder.new_state(batch, &device),
        );

        while (0..batch).any(|b| t[b] < lengths[b]) {
            let index: Vec<usize> = (0..batch)
                .map(|b| t[b].min(lengths[b].max(1) - 1).min(frames - 1))
                .collect();
            let frame = encoded
                .clone()
                .gather(1, ids(&index).reshape([batch, 1, 1]).expand([batch, 1, d]))
                .reshape([batch, d]);
            let logits = model.joint.forward(frame, prediction.clone());
            let tokens = logits
                .clone()
                .slice_dim(1, 0..n_vocab)
                .argmax(1)
                .into_data()
                .try_into_vec_as::<i64>()
                .expect("token argmax");
            let choices = logits
                .slice_dim(1, n_vocab..)
                .argmax(1)
                .into_data()
                .try_into_vec_as::<i64>()
                .expect("duration argmax");

            let mut emitted = vec![false; batch];
            for b in 0..batch {
                if t[b] >= lengths[b] {
                    continue;
                }
                let token = tokens[b] as usize;
                let mut duration = durations[choices[b] as usize];
                if token != blank {
                    emitted[b] = true;
                    let hypothesis = &mut hypotheses[b];
                    hypothesis.tokens.push(token);
                    hypothesis.frames.push(t[b]);
                    hypothesis.durations.push(duration);
                    at_frame[b] += 1;
                }
                if duration == 0
                    && (token == blank
                        || self
                            .config
                            .max_symbols_per_frame
                            .is_some_and(|max| at_frame[b] >= max))
                {
                    duration = 1;
                }
                if duration > 0 {
                    at_frame[b] = 0;
                }
                t[b] += duration;
            }

            if emitted.iter().any(|&e| e) {
                let previous: Vec<usize> = (0..batch)
                    .map(|b| {
                        if emitted[b] {
                            tokens[b] as usize
                        } else {
                            blank
                        }
                    })
                    .collect();
                let (next_prediction, next_state) =
                    model.decoder.step(ids(&previous), state.clone());
                let mask = Tensor::<1, Bool>::from_data(TensorData::new(emitted, [batch]), &device);
                let width = prediction.dims()[1];
                prediction = prediction.mask_where(
                    mask.clone().reshape([batch, 1]).expand([batch, width]),
                    next_prediction,
                );
                state = state.select(mask, next_state);
            }
        }
        hypotheses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::encoders::conformer::{
            ConformerBlockConfig,
            ConformerEncoderConfig,
            DwStridingSubsamplingConfig,
        },
        burner::module::ModuleInit,
        kits::speech::parakeet::blocks::ParakeetTdtConfig,
        support::testing::cpu_device,
    };

    /// Every item finishes, tokens are in the vocabulary, and frames stay
    /// inside each item's length, whatever the (random) weights choose.
    #[test]
    fn test_decode_terminates_in_bounds() {
        let device = cpu_device();
        let config = ParakeetTdtConfig::new(
            ConformerEncoderConfig::new(
                DwStridingSubsamplingConfig::new(24, 4, 16),
                ConformerBlockConfig::new(16, 2, 32, 5),
                1,
            ),
            11,
            12,
        );
        let model: ParakeetTdt = config.init(&device);
        let encoded = Tensor::<3>::random(
            [2, 9, 12],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let decoder = TdtGreedyDecoderConfig::new().init();
        let out = decoder.decode(&model, encoded, &[9, 4]);
        assert_eq!(out.len(), 2);
        for (h, len) in out.iter().zip([9, 4]) {
            assert_eq!(h.tokens.len(), h.frames.len());
            assert!(h.tokens.iter().all(|&t| t < 10));
            assert!(h.frames.iter().all(|&f| f < len));
            assert!(h.frames.windows(2).all(|w| w[0] <= w[1]));
            // At most max_symbols tokens at any one frame.
            for f in 0..len {
                assert!(h.frames.iter().filter(|&&x| x == f).count() <= 10);
            }
        }
    }
}
