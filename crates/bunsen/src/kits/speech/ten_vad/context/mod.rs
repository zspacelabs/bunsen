//! # ten-vad pre-processing driver.
//!
//! Everything between raw audio and the `[batch, d_ctx, n_freq]` feature
//! stack [`TenVad::forward`] consumes, plus the mutable context that carries
//! it across calls.
//!
//! [`TenVad::forward`] is the stateless call through the network;
//! [`TenVad::context_forward`] is what turns audio into the widened,
//! normalized input that call expects. The `_sequence` forms are equivalent
//! to iterating their single-step counterparts, but run the whole front end
//! batched across the sequence.
//!
//! ## The pipeline
//!
//! Per 256-sample hop, reproducing the reference C driver
//! (`ALGO_TRACE.md` §3.3 - §3.7, §5):
//!
//! ```text
//! raw    = hop * 32768                              # [-1, 1] -> int16 scale
//! emph   = raw[n] - 0.97 * raw[n-1]                 # carry = previous raw sample
//! binpow = |rfft(hann768 * queue768, n=1024)|^2     # 513 bins, queue = last 3 hops
//! pitch  = pitch_estimate(raw, binpow)              # Hz, 0 = unvoiced
//! mel    = ln(melbank40(binpow / 32768^2) + 1e-20)  # 40 triangular filters
//! feat   = (concat(mel, [pitch]) - MEANS) / (STDS + 1e-20)
//! stack  = concat(stack[:, 1:], feat)               # [1, 3, 41]
//! ```
//!
//! Two orderings are load-bearing and easy to get backwards:
//!
//! * **Pitch runs before the power normalization**, and reads the raw,
//!   un-pre-emphasized hop. The reference keeps two parallel FIFOs for exactly
//!   this reason.
//! * **The `1 / 32768^2` division happens before the filterbank matmul.** It
//!   commutes algebraically with the matmul but not in `f32`.
//!
//! ## The pieces
//!
//! * [`coeff`](self) — the reference constants and normalization tables.
//! * [`PreEmphasisContext`] — the first-order high-pass, with carry.
//! * [`TenVadMelBank`] — the 40-band triangular filterbank.
//! * [`PitchSource`] — the pitch seam, tensor-in and tensor-out;
//!   [`HostPitchEstimator`] behind [`HostPitch`] is the reference estimator,
//!   [`ZeroPitch`] the constant stub.
//! * [`TenVadFeatureContext`] — the 41-dim feature extractor and its state.
//! * [`TenVadContext`] — the driving context: features, frame stack, and both
//!   LSTM states.
//!
//! The sliding STFT itself is [`SlidingStftContext`], which already ports the
//! reference analyzer.
//!
//! ## Feeding it audio
//!
//! [`TenVad::context_forward`] and `_sequence` take hops as tensors, for
//! callers already holding device-resident audio.
//! [`TenVad::context_forward_audio`] and `_audio_sequence` take host `&[f32]`
//! rows, frame and upload them, and take the device from the context — the
//! usual path for a decoded file or a capture buffer.
//!
//! [`TenVad::context_forward_audio`]: crate::kits::speech::ten_vad::TenVad::context_forward_audio
//! [`TenVad::context_forward_audio_sequence`]: crate::kits::speech::ten_vad::TenVad::context_forward_audio_sequence
//!
//! ## Choosing a pitch source
//!
//! Feature `40` is reached through the [`PitchSource`] seam, selected by
//! [`PitchSourceConfig`] on [`TenVadContextConfig::pitch`]:
//!
//! | variant | what it is |
//! |---|---|
//! | `Tensor(..)` | the device estimator. **The default.** Keeps the whole front end resident; nothing synchronizes. |
//! | `Host` | the host scalar port — the reference oracle. Costs a device-to-host readback per call. |
//! | `Zero` | pins feature `40` to a constant and skips the branch. Features `0..40` are exact regardless. |
//!
//! The device variant has two tiers, differing only in how the anti-alias
//! filter before decimation is realized. The default folds the filter and its
//! decimation into one GEMM against a truncated impulse response;
//! [`TensorPitchConfig::reference`] instead selects a literal transcription of
//! the reference's IIR cascade. That tier is sample-sequential — five sections
//! stepping one sample at a time — so it is a correctness reference for short
//! inputs rather than a workload path. It is not a fidelity trade in the usual
//! direction: the truncated FIR is measurably *more* accurate than the
//! recurrence, being a better-conditioned realization of the same filter.
//!
//! All three implementations are cross-tested against each other, and the host
//! oracle is pinned to the C reference.
//!
//! ### Cost shape
//!
//! The device path is built for sequences. `forward_sequence` runs stages 1
//! and 3 over the whole run in one pass, and threads a carry through 2 and 4;
//! a single-hop `forward` pays the same setup for one frame's worth of work.
//! Callers stepping hop by hop through the device path should expect that,
//! and the `Host` variant may well be cheaper for them.
//!
//! Internally it works in fixed-size passes
//! ([`TensorPitchConfig::chunk_steps`]), which does not change the answer but
//! matters a great deal for cost: cubecl selects kernels per shape, so a
//! pinned pass size is tuned once instead of once per distinct input length.
//!
//! ## The periodic state reset
//!
//! The reference driver zeroes both LSTM states every `resetFrameNum = 1875`
//! model calls — 30 s of audio — and this driver reproduces that, on
//! [`TenVadContextConfig::reset_frames`]. `None` disables it.
//!
//! Three details matter, and all three are the reference's
//! (`src/aed.cc:476-481`, `ALGO_TRACE.md` §5):
//!
//! * **Only the recurrence is zeroed.** The feature stack, the STFT queue and
//!   the pre-emphasis carry keep running, so the frame after a reset still sees
//!   its two predecessors in the context stack. That is what separates
//!   [`TenVadContext::reset_states`] from [`TenVadContext::reset`].
//! * **The counter fires after the model call**, on `>=`. Call 1875 runs with
//!   the states it inherited; call 1876 runs from zero.
//! * **Both drivers tick identically**, so `context_forward_sequence` stays
//!   exactly "iterating `context_forward`" — a reset lands on the same hops
//!   whether or not the caller batches, and whatever chunk boundaries it uses.
//!
//! The reference zeroes lazily, at the top of its next call, via a
//! `clear_hidden` flag; this driver zeroes eagerly at the end of the current
//! one. Those are observationally identical for any continuation.
//!
//! The reference marks the constant `// TODO` (`src/aed.cc:640`), so treat it
//! as reproduced behavior rather than a recommendation — but it is on by
//! default, because parity with the pretrained weights is what this kit is
//! for, and the shipped golden covers 3750 hops, exactly two periods.
//!
//! ## Known deviations from the reference driver
//!
//! * **Batch size 1.** The stock ONNX graph pins its LSTM batch to 1; see
//!   [`TenVad::forward`] for what the leading axis actually means.
//!
//! ## What is pinned numerically
//!
//! * **Feature `40`** — `testdata/ten/pitch.json` holds one reference pitch per
//!   hop over the 60 s fixture, dumped from the C `AUP_PE_proc` driven by the C
//!   STFT. The kit's cross test asserts the voicing decision matches on every
//!   frame and the voiced estimates agree to within f32 rounding.
//! * **Features `0..40`** — [`TenVadFeatureContext`]'s own tests pin the mel
//!   path against an independent host implementation written from the reference
//!   ordering.
//! * **The driver as a whole** — the kit's cross test pins it against the ONNX
//!   graph over real audio.
//!
//! [`PitchSource`]: crate::kits::speech::pitch::PitchSource
//! [`PitchSourceConfig`]: crate::kits::speech::pitch::PitchSourceConfig
//! [`HostPitchEstimator`]: crate::kits::speech::pitch::HostPitchEstimator
//! [`HostPitch`]: crate::kits::speech::pitch::HostPitch
//! [`ZeroPitch`]: crate::kits::speech::pitch::ZeroPitch
//! [`TenVadContextConfig::pitch`]: crate::kits::speech::ten_vad::TenVadContextConfig
//! [`TenVadContextConfig::reset_frames`]: crate::kits::speech::ten_vad::TenVadContextConfig
//! [`TensorPitchConfig::reference`]: crate::kits::speech::pitch::tensor::TensorPitchConfig::reference
//! [`TensorPitchConfig::chunk_steps`]: crate::kits::speech::pitch::tensor::TensorPitchConfig
//! [`TenVad::forward`]: crate::kits::speech::ten_vad::TenVad::forward
//! [`TenVad::context_forward`]: crate::kits::speech::ten_vad::TenVad::context_forward
//! [`SlidingStftContext`]: crate::ops::signal::SlidingStftContext

pub mod coeff;

mod driver;
mod features;
mod mel;
mod pre_emphasis;

#[doc(inline)]
pub use coeff::*;
#[doc(inline)]
pub use driver::*;
#[doc(inline)]
pub use features::*;
#[doc(inline)]
pub use mel::*;
#[doc(inline)]
pub use pre_emphasis::*;
