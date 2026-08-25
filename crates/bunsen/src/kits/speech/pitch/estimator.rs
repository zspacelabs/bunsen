//! # The ten-vad pitch estimator.
//!
//! A port of the reference `AUP_PE_proc` (`src/pitch_est.cc`), producing
//! feature `40` of the ten-vad feature vector.
//!
//! The estimator is an RNNoise-derived tracker. Per 256-sample hop it runs
//! four stages, each depending on state the previous hops left behind:
//!
//! 1. **Pre-filter design.** The bin powers are folded into 18 bands, log
//!    compressed with a decay clamp, taken through a cepstrum, and fitted with
//!    a 16th-order all-pole model ([`super::lpc`]).
//! 2. **Excitation.** The raw hop — delayed by
//!    [`XCORR_TRAINING_OFFSET`](super::coeff::XCORR_TRAINING_OFFSET) samples so
//!    it lines up with the correlation window — is whitened by that filter,
//!    smoothed by a 2-tap FIR (`y[n] = w[n] + 0.7·w[n-1]`), anti-alias
//!    filtered, and decimated 16 kHz → 4 kHz. Whitening is what leaves a clean
//!    impulse train at the pitch period.
//! 3. **Correlation.** Two half-hop windows per hop are correlated against 64
//!    candidate lags, each normalized by the energy under the lagged window,
//!    then sharpened against their own half-lag to suppress octave doubling.
//! 4. **Tracking.** A Viterbi pass over the last `2 * n_feat` half-hops picks a
//!    period path, penalizing `PITCH_MAX_PATH_W * step²` for jumping. The
//!    backtrace yields both a voicing score and a weighted linear regression
//!    over the recovered periods, extrapolated half a frame forward.
//!
//! ## Ordering that is load-bearing
//!
//! * The estimator reads the **raw** hop, never the pre-emphasized one, and the
//!   **un-normalized** bin powers. The reference keeps two parallel FIFOs for
//!   exactly this (`ALGO_TRACE.md` §3.3).
//! * The Viterbi accumulator carries **across hops**. It is renormalized by
//!   subtracting the running maximum each step rather than being cleared, so
//!   clearing it per hop would discard the tracker's whole memory.
//! * The correlation buffer is a circular buffer of `2 * n_feat` half-hops
//!   indexed off [`xcorr_offset`](HostPitchEstimator), and the DP walks it in
//!   stream order, not slot order.
//!
//! ## Divergences from the reference
//!
//! * The reference's inverse FFT in the LPC fit is replaced by the equivalent
//!   closed-form cosine sum; see [`super::lpc`].
//! * The reference copies its correlation buffer to a scratch buffer before the
//!   DP, commenting that the DP may modify it. Nothing does, so the copy is
//!   dropped and the DP reads the buffer directly.
//! * Where the reference divides by an unguarded `sw`, this guards against a
//!   `0/0`. That case is reachable only on digital silence, where the frame is
//!   already unvoiced and the quotient is discarded, so no output changes.

use burn::prelude::Backend;

use super::{
    coeff::{
        ANTI_ALIAS_A_4KHZ,
        ANTI_ALIAS_B_4KHZ,
        ANTI_ALIAS_G_4KHZ,
        ANTI_ALIAS_SECTIONS,
        FEAT_MAX_NFRM,
        FEAT_TIME_WINDOW_MS,
        HOP_SIZE,
        LPC_ORDER,
        MAX_PERIOD_16KHZ,
        MIN_PERIOD_16KHZ,
        PITCH_MAX_PATH_W,
        PROC_FS,
        PROC_RESAMPLE_RATE,
        SAMPLE_RATE,
        VOICED_THRESHOLD,
        XCORR_TRAINING_OFFSET,
    },
    host::{
        HostPitch,
        HostPitchInit,
    },
    lpc::{
        DctTable,
        band_energy,
        lpc_from_cepstrum,
    },
    source::{
        PitchScalarSource,
        PitchSourceInit,
    },
};
use crate::{
    ops::signal::{
        Autocorrelator,
        BiquadCascade,
    },
    prelude::BunsenResult,
};

/// The ten-vad STFT size, which sets the bin count the estimator expects.
const FFT_SIZE: usize = 1024;

/// The ten-vad STFT analysis window, which sets the LPC noise floor.
const WINDOW_SIZE: usize = 768;

/// The reference pitch estimator.
///
/// A [`PitchScalarSource`]: one instance per stream, stepped one hop at
/// a time. It reaches the driver's tensor seam through
/// [`HostPitch`](super::HostPitch), which this type's
/// [`PitchSourceInit`] impl wraps it in — so it can be named directly
/// where a pitch source is expected:
///
/// ```rust,ignore
/// let ctx = vad.init_context_with(&cfg, HostPitchEstimator::new(), &device)?;
/// ```
///
/// This is also the permanent oracle a device-side port is validated against:
/// it is pinned to the C reference by `testdata/ten/pitch.json`, and
/// [`frame_pitch_with_lpc`](Self::frame_pitch_with_lpc) exposes the one stage
/// boundary that carries no state.
///
/// Built by [`new`](Self::new), rewound by
/// [`reset`](PitchScalarSource::reset).
#[derive(Debug, Clone)]
pub struct HostPitchEstimator {
    // --- Fixed geometry, all derived from the ten-vad configuration. ---
    /// The shortest candidate period, in samples at [`PROC_FS`].
    min_period: usize,

    /// The longest candidate period, in samples at [`PROC_FS`]; also the width
    /// of the correlation search.
    max_period: usize,

    /// `max_period - min_period`: the width of the tracker's state space.
    dif_period: usize,

    /// The number of whole hops of correlation history; each contributes two
    /// half-hop slots.
    n_feat: usize,

    /// Half a hop, in samples at [`PROC_FS`]; the correlation window length.
    corr_half_hop: usize,

    // --- Tables. ---
    dct: DctTable,
    autocorrelator: Autocorrelator,

    // --- Stage 1: pre-filter design. ---
    /// The current whitening filter.
    lpc: [f32; LPC_ORDER],

    /// `[fft_size / 2 + 1]` scratch for the band-gain interpolation.
    bin_scratch: Vec<f32>,

    // --- Stage 2: excitation. ---
    /// The raw-sample FIFO the delayed hop is read out of.
    input_q: Vec<f32>,

    /// The delayed hop: `input_q` read `XCORR_TRAINING_OFFSET` behind the tail.
    aligned_in: Vec<f32>,

    /// The whitening filter's sample memory.
    pitch_mem: [f32; LPC_ORDER],

    /// The previous whitened sample, carried across samples and hops.
    pitch_filt: f32,

    /// The whitened hop, before anti-aliasing.
    lpc_filter_out: Vec<f32>,

    /// The anti-alias cascade run before decimation.
    biquad: BiquadCascade<ANTI_ALIAS_SECTIONS>,

    /// The decimated excitation history the correlation reads.
    exc_buf: Vec<f32>,

    /// `exc_buf`, squared.
    exc_buf_sq: Vec<f32>,

    // --- Stage 3: correlation. ---
    /// The oldest slot in the circular correlation buffer, in whole hops.
    xcorr_offset: usize,

    /// The un-normalized correlations of the half-hop being processed.
    xcorr_inst: Vec<f32>,

    /// `[2 * n_feat][max_period]` normalized correlations, circular.
    xcorr: Vec<Vec<f32>>,

    /// `[2 * n_feat]` per-half-hop energies, in stream order.
    frm_weight: Vec<f32>,

    /// `frm_weight`, scaled to average 1.
    frm_weight_norm: Vec<f32>,

    // --- Stage 4: tracking. ---
    /// The Viterbi accumulator: `[previous, current]`, each `[dif_period]`.
    path_score: [Vec<f32>; 2],

    /// `[2 * n_feat][dif_period]` backpointers, in stream order.
    path_prev: Vec<Vec<usize>>,

    /// The best path score reached so far, carried across hops.
    path_best_all: f32,

    /// The best period index reached so far, carried across hops.
    best_period: usize,

    /// The recovered period per half-hop, filled by the backtrace.
    period_local: Vec<usize>,

    /// Whether the last hop was voiced.
    voiced: bool,

    /// The last hop's pitch, in Hz.
    pitch_hz: f32,
}

impl Default for HostPitchEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl HostPitchEstimator {
    /// Builds an estimator for the ten-vad front end, in start-of-stream state.
    ///
    /// The geometry is fixed: 16 kHz in, a 256-sample hop, a 1024-point STFT
    /// with a 768-sample window, and a 4 kHz correlation branch.
    pub fn new() -> Self {
        let hop_size = HOP_SIZE;
        let min_period = MIN_PERIOD_16KHZ / PROC_RESAMPLE_RATE;
        let max_period = MAX_PERIOD_16KHZ / PROC_RESAMPLE_RATE;
        let dif_period = max_period - min_period;
        let corr_half_hop = hop_size / (PROC_RESAMPLE_RATE * 2);

        let n_feat = FEAT_MAX_NFRM.min(
            ((FEAT_TIME_WINDOW_MS * SAMPLE_RATE) as f32 / (hop_size * 1000) as f32).ceil() as usize,
        );
        let slots = n_feat * 2;

        // The FIFO must hold the hop plus the correlation delay ahead of it.
        let input_q_len = XCORR_TRAINING_OFFSET.max(hop_size) + hop_size;
        // One decimated hop of headroom past the search width, plus the slot
        // the reference over-allocates.
        let exc_shift = hop_size.div_ceil(PROC_RESAMPLE_RATE);
        let exc_buf_len = max_period + exc_shift + 1;

        Self {
            min_period,
            max_period,
            dif_period,
            n_feat,
            corr_half_hop,

            dct: DctTable::new(),
            autocorrelator: Autocorrelator::new(FFT_SIZE),

            lpc: [0.0; LPC_ORDER],
            bin_scratch: vec![0.0; FFT_SIZE / 2 + 1],

            input_q: vec![0.0; input_q_len],
            aligned_in: vec![0.0; hop_size],
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            lpc_filter_out: vec![0.0; hop_size],
            biquad: BiquadCascade::from_tables(
                &ANTI_ALIAS_B_4KHZ,
                &ANTI_ALIAS_A_4KHZ,
                &ANTI_ALIAS_G_4KHZ,
            ),
            exc_buf: vec![0.0; exc_buf_len],
            exc_buf_sq: vec![0.0; exc_buf_len],

            xcorr_offset: 0,
            xcorr_inst: vec![0.0; max_period],
            xcorr: vec![vec![0.0; max_period]; slots],
            frm_weight: vec![0.0; slots],
            frm_weight_norm: vec![0.0; slots],

            path_score: [vec![0.0; dif_period], vec![0.0; dif_period]],
            path_prev: vec![vec![0; dif_period]; slots],
            path_best_all: 0.0,
            best_period: 0,
            period_local: vec![0; slots],
            voiced: false,
            pitch_hz: 0.0,
        }
    }

    /// The hop size, in samples at 16 kHz.
    pub fn hop_size(&self) -> usize {
        self.aligned_in.len()
    }

    /// The number of frequency bins the estimator expects.
    pub fn n_bins(&self) -> usize {
        self.bin_scratch.len()
    }

    /// The number of half-hop slots the tracker spans.
    pub fn slots(&self) -> usize {
        self.n_feat * 2
    }

    /// Whether the most recent hop was judged voiced.
    ///
    /// [`pitch_hz`](Self::pitch_hz) is zero whenever this is `false`.
    pub fn voiced(&self) -> bool {
        self.voiced
    }

    /// The most recent hop's pitch estimate, in Hz; `0.0` when unvoiced.
    pub fn pitch_hz(&self) -> f32 {
        self.pitch_hz
    }

    /// The current whitening filter coefficients.
    pub fn lpc(&self) -> &[f32; LPC_ORDER] {
        &self.lpc
    }

    /// Estimates the pitch of one hop against an externally supplied
    /// pre-filter, skipping the stage that would design one.
    ///
    /// [`frame_pitch`](PitchScalarSource::frame_pitch) is exactly
    /// "design the pre-filter from `bin_power`, then this". The stage that
    /// designs it carries no state, so the split is exact: feeding this the
    /// coefficients that stage would have produced reproduces `frame_pitch`
    /// bit for bit.
    ///
    /// Exposed so a device-side pre-filter can be validated end to end against
    /// the reference golden with the remaining stages held fixed.
    ///
    /// # Arguments
    /// * `raw_hop`: the hop's samples at the reference's int16 scale.
    /// * `lpc`: the whitening filter to use for this hop.
    ///
    /// # Returns
    /// The pitch in Hz, or `0.0` when nothing voiced was detected.
    ///
    /// # Panics
    /// If `raw_hop` is not [`hop_size`](Self::hop_size) long.
    pub fn frame_pitch_with_lpc(
        &mut self,
        raw_hop: &[f32],
        lpc: &[f32; LPC_ORDER],
    ) -> f32 {
        assert_eq!(
            raw_hop.len(),
            self.hop_size(),
            "HostPitchEstimator expects a {}-sample hop",
            self.hop_size(),
        );

        self.lpc = *lpc;
        self.extract_excitation(raw_hop);
        self.correlate();
        self.track()
    }

    /// Stage 1: fits the whitening filter to this hop's spectrum.
    fn design_prefilter(
        &mut self,
        bin_power: &[f32],
    ) {
        let mut band = band_energy(bin_power, FFT_SIZE);

        // Log compression with a two-sided floor: nothing may sit more than
        // 8 decades below the running peak, nor fall faster than 2.5 decades
        // per band. Without it, near-empty bands would dominate the cepstrum.
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        for b in band.iter_mut() {
            let ly = (1e-2 + *b).log10();
            let ly = (log_max - 8.0).max((follow - 2.5).max(ly));
            log_max = log_max.max(ly);
            follow = (follow - 2.5).max(ly);
            *b = ly;
        }

        let cepstrum = self.dct.dct(&band);
        self.lpc = lpc_from_cepstrum(
            &cepstrum,
            &self.dct,
            WINDOW_SIZE,
            &self.autocorrelator,
            &mut self.bin_scratch,
        );
    }

    /// Stage 2: whitens, anti-aliases, and decimates the delayed hop.
    fn extract_excitation(
        &mut self,
        raw_hop: &[f32],
    ) {
        let hop = self.hop_size();

        // Slide the raw FIFO and append this hop.
        self.input_q.copy_within(hop.., 0);
        let tail = self.input_q.len() - hop;
        self.input_q[tail..].copy_from_slice(raw_hop);

        // Read out the hop that sits `XCORR_TRAINING_OFFSET` behind the tail.
        let offset = self
            .input_q
            .len()
            .saturating_sub(hop)
            .saturating_sub(XCORR_TRAINING_OFFSET);
        self.aligned_in
            .copy_from_slice(&self.input_q[offset..offset + hop]);

        // FIR whitening, then a 2-tap smoother that keeps the excitation from
        // going fully impulsive. Both are FIR, so both are convolutions.
        for i in 0..hop {
            let sample = self.aligned_in[i];
            let mut whitened = sample;
            for j in 0..LPC_ORDER {
                whitened += self.lpc[j] * self.pitch_mem[j];
            }

            self.pitch_mem.copy_within(0..LPC_ORDER - 1, 1);
            self.pitch_mem[0] = sample;

            self.lpc_filter_out[i] = whitened + 0.7 * self.pitch_filt;
            self.pitch_filt = whitened;
        }

        self.biquad.process_in_place(&mut self.lpc_filter_out);

        // Decimate and push into the excitation history.
        let shift = hop.div_ceil(PROC_RESAMPLE_RATE);
        self.exc_buf.copy_within(shift.., 0);
        let tail = self.exc_buf.len() - shift;
        for n in 0..shift {
            self.exc_buf[tail + n] = self.lpc_filter_out[n * PROC_RESAMPLE_RATE];
        }
    }

    /// Stage 3: correlates both half-hops against every candidate lag.
    fn correlate(&mut self) {
        for (dst, &x) in self.exc_buf_sq.iter_mut().zip(self.exc_buf.iter()) {
            *dst = x * x;
        }

        // Slide the energy history left by one hop's worth of slots.
        for idx in 0..(self.n_feat - 1) {
            self.frm_weight[2 * idx] = self.frm_weight[2 * (idx + 1)];
            self.frm_weight[2 * idx + 1] = self.frm_weight[2 * (idx + 1) + 1];
        }

        let half = self.corr_half_hop;
        for sub in 0..2 {
            let slot = 2 * self.xcorr_offset + sub;
            let base = sub * half;

            // Correlate the newest half-hop against every lag behind it.
            for (lag, dst) in self.xcorr_inst.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for j in 0..half {
                    sum += self.exc_buf[self.max_period + base + j] * self.exc_buf[base + lag + j];
                }
                *dst = sum;
            }

            // The reference window's energy, which is also this slot's weight
            // in the tracker.
            let mut reference_energy = 0.0f32;
            for j in 0..half {
                reference_energy += self.exc_buf_sq[self.max_period + base + j];
            }
            self.frm_weight[2 * (self.n_feat - 1) + sub] = reference_energy;

            // Normalize each lag by the energy under the lagged window, kept
            // as a sliding sum. The `1 +` keeps silence from amplifying noise.
            let mut lagged_energy = 0.0f32;
            for j in 0..half {
                lagged_energy += self.exc_buf_sq[base + j];
            }

            let mut denom = (lagged_energy + (1.0 + reference_energy)).max(1e-12);
            self.xcorr[slot][0] = 2.0 * self.xcorr_inst[0] / denom;
            for lag in 1..self.max_period {
                lagged_energy = (lagged_energy - self.exc_buf_sq[base + lag - 1]).max(0.0);
                lagged_energy += self.exc_buf_sq[base + lag + half - 1];
                denom = (lagged_energy + (1.0 + reference_energy)).max(1e-12);
                self.xcorr[slot][lag] = 2.0 * self.xcorr_inst[lag] / denom;
            }

            // Octave suppression: a lag that does not clearly beat its own
            // half-lag neighborhood is a likely period double, so discount it.
            for lag in 0..(self.max_period - 2 * self.min_period) {
                let mut rival = self.xcorr[slot][(self.max_period + lag) / 2];
                rival = rival.max(self.xcorr[slot][(self.max_period + lag + 2) / 2]);
                rival = rival.max(self.xcorr[slot][(self.max_period + lag - 1) / 2]);
                if self.xcorr[slot][lag] < rival * 1.1 {
                    self.xcorr[slot][lag] *= 0.8;
                }
            }
        }

        self.xcorr_offset = (self.xcorr_offset + 1) % self.n_feat;
    }

    /// The circular correlation slot holding stream position `sub`.
    fn slot_of(
        &self,
        sub: usize,
    ) -> usize {
        (sub + self.xcorr_offset * 2) % self.slots()
    }

    /// Stage 4: tracks a period path and regresses it to a pitch.
    fn track(&mut self) -> f32 {
        let slots = self.slots();

        // Scale the energies to average 1, so the tracker's correlation term
        // and its jump penalty stay commensurable regardless of level.
        let mut total = 1e-15f32;
        for sub in 0..slots {
            total += self.frm_weight[sub];
        }
        for sub in 0..slots {
            self.frm_weight_norm[sub] = self.frm_weight[sub] * (slots as f32 / total);
        }

        // Slide the backpointer history left by this hop's two slots.
        for sub in 0..(slots - 2) {
            let (head, tail) = self.path_prev.split_at_mut(sub + 2);
            head[sub].copy_from_slice(&tail[0]);
        }

        // Forward Viterbi over the two new slots. `path_score[0]` carries over
        // from the previous hop; it is renormalized, never cleared.
        for sub in (slots - 2)..slots {
            let slot = self.slot_of(sub);

            for idx in 0..self.dif_period {
                let mut best_score = self.path_best_all - 1e10;
                let mut best_prev = self.best_period;

                // Candidates reachable in one step. The window is asymmetric
                // near the short-period end, exactly as the reference has it.
                let first = 0.min(4 - idx as i32);
                for jdx in first..=4 {
                    let cand = idx as i32 + jdx;
                    debug_assert!(cand >= 0, "candidate period index went negative");
                    let cand = cand as usize;
                    if cand >= self.dif_period {
                        break;
                    }
                    let penalty = PITCH_MAX_PATH_W * (jdx.abs() as f32) * (jdx.abs() as f32);
                    let score = self.path_score[0][cand] - penalty;
                    if score > best_score {
                        best_score = score;
                        best_prev = cand;
                    }
                }

                self.path_prev[sub][idx] = best_prev;
                self.path_score[1][idx] =
                    best_score + self.frm_weight_norm[sub] * self.xcorr[slot][idx];
            }

            let mut max_score = -1e15f32;
            let mut arg_max = 0usize;
            for idx in 0..self.dif_period {
                if self.path_score[1][idx] > max_score {
                    max_score = self.path_score[1][idx];
                    arg_max = idx;
                }
            }
            self.path_best_all = max_score;
            self.best_period = arg_max;

            // Roll current into previous, renormalized so the peak sits at 0
            // and the accumulator cannot drift out of range.
            let (previous, current) = self.path_score.split_at_mut(1);
            previous[0].copy_from_slice(&current[0]);
            for score in self.path_score[0].iter_mut() {
                *score -= max_score;
            }
        }

        // Backtrace, collecting both the path and its correlation score.
        let mut cursor = self.best_period;
        let mut frame_corr = 0.0f32;
        for sub in (0..slots).rev() {
            self.period_local[sub] = self.max_period - cursor;
            let slot = self.slot_of(sub);
            frame_corr += self.frm_weight_norm[sub] * self.xcorr[slot][cursor];
            cursor = self.path_prev[sub][cursor];
        }
        frame_corr = (frame_corr / slots as f32).max(0.0);
        self.voiced = frame_corr >= VOICED_THRESHOLD;

        // Weighted least squares of period against slot index.
        let (mut sw, mut sx, mut sxx, mut sxy, mut sy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for sub in 0..slots {
            let w = self.frm_weight_norm[sub];
            let x = sub as f32;
            let y = self.period_local[sub] as f32;
            sw += w;
            sx += w * x;
            sxx += w * x * x;
            sxy += w * x * y;
            sy += w * y;
        }

        let denom = sw * sxx - sx * sx;
        let mut slope = if denom == 0.0 {
            (sw * sxy - sx * sy) / 1e-15
        } else {
            (sw * sxy - sx * sy) / denom
        };

        if self.voiced {
            // Cap the contour slope so one bad slot cannot swing the estimate.
            let limit = (sy / sw) / (4.0 * 2.0 * self.n_feat as f32);
            slope = slope.max(-limit).min(limit);
        } else {
            slope = 0.0;
        }

        // Extrapolate half a hop past the last slot, where this hop's output
        // nominally sits.
        let intercept = (sy - slope * sx) / sw;
        let period = intercept + 5.5 * slope;

        self.pitch_hz = if self.voiced {
            PROC_FS as f32 / period.max(1.0)
        } else {
            0.0
        };
        self.pitch_hz
    }
}

/// The oracle surface: read access to the stage boundaries a device-side port
/// is checked against.
///
/// Deliberately test-only and `pub(crate)`. These expose the estimator's
/// internal layout — `exc_buf`'s length, `path_prev`'s slot shape — which
/// should not become public API just to let a differential test read them.
#[cfg(test)]
impl HostPitchEstimator {
    /// The decimated excitation history.
    ///
    /// Part of the oracle surface: this is what a device-side excitation stage
    /// is checked against.
    pub(crate) fn exc_buf(&self) -> &[f32] {
        &self.exc_buf
    }

    /// The normalized correlations of stream slot `sub`, `[max_period]`.
    ///
    /// `sub` is a **stream** index, oldest at `0`. The circular-buffer
    /// translation is applied here, so callers never see the ring offset —
    /// which is exactly the bookkeeping a tensor port stores away.
    pub(crate) fn xcorr_slot(
        &self,
        sub: usize,
    ) -> &[f32] {
        &self.xcorr[self.slot_of(sub)]
    }

    /// The per-half-hop energies, in stream order.
    pub(crate) fn frm_weight(&self) -> &[f32] {
        &self.frm_weight
    }

    /// The Viterbi accumulator carried into the next hop, `[dif_period]`.
    pub(crate) fn path_score(&self) -> &[f32] {
        &self.path_score[0]
    }

    /// The backpointers of stream slot `sub`, `[dif_period]`.
    ///
    /// `sub` is a stream index, oldest at `0`; unlike the correlation ring
    /// this buffer is already kept in stream order.
    pub(crate) fn path_prev_slot(
        &self,
        sub: usize,
    ) -> &[usize] {
        &self.path_prev[sub]
    }

    /// The best period index reached so far, carried across hops.
    pub(crate) fn best_period(&self) -> usize {
        self.best_period
    }

    /// The best path score reached so far, carried across hops.
    pub(crate) fn path_best_all(&self) -> f32 {
        self.path_best_all
    }
}

impl PitchScalarSource for HostPitchEstimator {
    /// Estimates the pitch of one hop.
    ///
    /// # Panics
    /// If `raw_hop` is not [`hop_size`](Self::hop_size) long, or `bin_power`
    /// is not [`n_bins`](Self::n_bins) long.
    fn frame_pitch(
        &mut self,
        raw_hop: &[f32],
        bin_power: &[f32],
    ) -> f32 {
        assert_eq!(
            raw_hop.len(),
            self.hop_size(),
            "HostPitchEstimator expects a {}-sample hop",
            self.hop_size(),
        );
        assert_eq!(
            bin_power.len(),
            self.n_bins(),
            "HostPitchEstimator expects {} bins",
            self.n_bins(),
        );

        self.design_prefilter(bin_power);
        self.extract_excitation(raw_hop);
        self.correlate();
        self.track()
    }

    fn reset(&mut self) {
        self.lpc = [0.0; LPC_ORDER];
        self.bin_scratch.fill(0.0);

        self.input_q.fill(0.0);
        self.aligned_in.fill(0.0);
        self.pitch_mem = [0.0; LPC_ORDER];
        self.pitch_filt = 0.0;
        self.lpc_filter_out.fill(0.0);
        self.biquad.reset();
        self.exc_buf.fill(0.0);
        self.exc_buf_sq.fill(0.0);

        self.xcorr_offset = 0;
        self.xcorr_inst.fill(0.0);
        for row in &mut self.xcorr {
            row.fill(0.0);
        }
        self.frm_weight.fill(0.0);
        self.frm_weight_norm.fill(0.0);

        for reg in &mut self.path_score {
            reg.fill(0.0);
        }
        for row in &mut self.path_prev {
            row.fill(0);
        }
        self.path_best_all = 0.0;
        self.best_period = 0;
        self.period_local.fill(0);
        self.voiced = false;
        self.pitch_hz = 0.0;
    }
}

/// Builds the estimator behind a [`HostPitch`] adapter, one instance per
/// stream.
///
/// This is what lets `init_context_with(cfg, HostPitchEstimator::new(), dev)`
/// read naturally while the driver's seam stays tensor-in, tensor-out.
impl<B: Backend> PitchSourceInit<B> for HostPitchEstimator {
    type Source = HostPitch<HostPitchEstimator>;

    fn try_init_source(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> BunsenResult<Self::Source> {
        PitchSourceInit::<B>::try_init_source(&HostPitchInit(self.clone()), batch_size, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::CpuBackend;

    /// Constructing a `HostPitch` is setup plumbing with no numerics in it, so
    /// it does not want a device lock.
    type B = CpuBackend;

    const N_BINS: usize = FFT_SIZE / 2 + 1;

    /// A 16 kHz tone at `freq_hz`, at the reference's int16 scale.
    fn tone(
        freq_hz: f32,
        samples: usize,
        phase_at: usize,
    ) -> Vec<f32> {
        (0..samples)
            .map(|i| {
                let t = (phase_at + i) as f32 / SAMPLE_RATE as f32;
                8000.0 * (core::f32::consts::TAU * freq_hz * t).sin()
            })
            .collect()
    }

    /// A glottal-like pulse train at `f0`, which is what the estimator is
    /// actually built to find.
    fn pulse_train(
        f0: f32,
        samples: usize,
        phase_at: usize,
    ) -> Vec<f32> {
        let period = SAMPLE_RATE as f32 / f0;
        (0..samples)
            .map(|i| {
                let pos = (phase_at + i) as f32 % period;
                // A short, decaying burst once per period.
                8000.0 * (-pos / (period * 0.08)).exp()
            })
            .collect()
    }

    /// The bin powers of one hop, via the same path the driver uses:
    /// pre-emphasis, Hann-768 window over the last three hops, 1024-point
    /// real FFT, `re² + im²`.
    struct HostStft {
        queue: Vec<f32>,
        window: Vec<f32>,
        prev_raw: f32,
    }

    impl HostStft {
        fn new() -> Self {
            let window = (0..WINDOW_SIZE)
                .map(|n| 0.5 - 0.5 * (core::f32::consts::TAU * n as f32 / WINDOW_SIZE as f32).cos())
                .collect();
            Self {
                queue: vec![0.0; WINDOW_SIZE],
                window,
                prev_raw: 0.0,
            }
        }

        fn push(
            &mut self,
            raw_hop: &[f32],
        ) -> Vec<f32> {
            let hop = raw_hop.len();
            self.queue.copy_within(hop.., 0);
            let tail = self.queue.len() - hop;
            for (i, &x) in raw_hop.iter().enumerate() {
                self.queue[tail + i] = x - 0.97 * self.prev_raw;
                self.prev_raw = x;
            }

            let windowed: Vec<f32> = self
                .queue
                .iter()
                .zip(self.window.iter())
                .map(|(&x, &w)| x * w)
                .collect();

            (0..N_BINS)
                .map(|k| {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for (n, &x) in windowed.iter().enumerate() {
                        let ang = -core::f64::consts::TAU * k as f64 * n as f64 / FFT_SIZE as f64;
                        re += x as f64 * ang.cos();
                        im += x as f64 * ang.sin();
                    }
                    (re * re + im * im) as f32
                })
                .collect()
        }
    }

    /// The signal's whole hops, in order; any trailing partial hop is dropped.
    fn hops(signal: &[f32]) -> impl Iterator<Item = &[f32]> {
        signal
            .as_chunks::<HOP_SIZE>()
            .0
            .iter()
            .map(|hop| hop.as_slice())
    }

    /// Runs `signal` through the estimator hop by hop, returning per-hop pitch.
    fn run(signal: &[f32]) -> Vec<f32> {
        let mut est = HostPitchEstimator::new();
        let mut stft = HostStft::new();
        hops(signal)
            .map(|hop| {
                let power = stft.push(hop);
                est.frame_pitch(hop, &power)
            })
            .collect()
    }

    #[test]
    fn test_geometry_matches_the_reference_configuration() {
        let est = HostPitchEstimator::new();
        assert_eq!(PROC_RESAMPLE_RATE, 4);
        assert_eq!(est.hop_size(), 256);
        assert_eq!(est.n_bins(), 513);
        assert_eq!(est.min_period, 8);
        assert_eq!(est.max_period, 64);
        assert_eq!(est.dif_period, 56);
        assert_eq!(est.corr_half_hop, 32);
        // ceil(40 ms * 16 kHz / 256 samples) = ceil(2.5) = 3 hops of history.
        assert_eq!(est.n_feat, 3);
        assert_eq!(est.slots(), 6);
    }

    #[test]
    fn test_default_matches_new() {
        // `Default` lets the estimator sit in a struct field without a
        // builder; it must not diverge from the real constructor.
        let a = HostPitchEstimator::default();
        let b = HostPitchEstimator::new();
        assert_eq!(a.hop_size(), b.hop_size());
        assert_eq!(a.n_bins(), b.n_bins());
        assert_eq!(a.max_period, b.max_period);
        assert_eq!(a.min_period, b.min_period);
        assert_eq!(a.n_feat, b.n_feat);
    }

    #[test]
    fn test_estimator_is_directly_usable_as_a_source_init() {
        // `HostPitchEstimator` implements `PitchSourceInit` itself, so a
        // caller with a configured estimator can hand it straight to a
        // pipeline instead of wrapping it in `HostPitchInit` by hand. The two
        // routes must agree.
        let device = Default::default();

        let direct: HostPitch<HostPitchEstimator> =
            PitchSourceInit::<B>::init_source(&HostPitchEstimator::new(), 2, &device);
        let wrapped: HostPitch<HostPitchEstimator> = PitchSourceInit::<B>::init_source(
            &HostPitchInit(HostPitchEstimator::new()),
            2,
            &device,
        );

        assert_eq!(direct.batch_size(), wrapped.batch_size());
        assert_eq!(direct.batch_size(), 2);
    }

    #[test]
    fn test_silence_is_unvoiced() {
        let pitches = run(&vec![0.0f32; HOP_SIZE * 20]);
        assert_eq!(pitches.len(), 20);
        for (i, p) in pitches.iter().enumerate() {
            assert_eq!(*p, 0.0, "hop {i} of silence reported {p} Hz");
        }
    }

    #[test]
    fn test_silence_leaves_no_nan_behind() {
        // The reference divides by an unguarded weight sum here; make sure the
        // guarded form still reports a clean zero rather than a NaN.
        let mut est = HostPitchEstimator::new();
        for _ in 0..8 {
            let p = est.frame_pitch(&[0.0; HOP_SIZE], &[0.0; N_BINS]);
            assert!(p.is_finite(), "pitch went non-finite on silence");
            assert_eq!(p, 0.0);
        }
        assert!(!est.voiced());
    }

    #[test]
    fn test_pulse_train_recovers_its_fundamental() {
        // The tracker needs a few hops of history before its regression is
        // meaningful, so judge the settled tail.
        for f0 in [110.0f32, 160.0, 220.0] {
            let signal = pulse_train(f0, HOP_SIZE * 40, 0);
            let pitches = run(&signal);
            let tail = &pitches[20..];

            let voiced: Vec<f32> = tail.iter().copied().filter(|p| *p > 0.0).collect();
            assert!(
                voiced.len() * 2 > tail.len(),
                "f0 {f0}: only {} of {} tail hops voiced",
                voiced.len(),
                tail.len(),
            );

            let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
            let rel = (mean - f0).abs() / f0;
            assert!(rel < 0.12, "f0 {f0}: estimated {mean} Hz (rel err {rel})");
        }
    }

    #[test]
    fn test_a_harmonic_rich_signal_does_not_report_the_octave() {
        // What `suppress_octaves` exists for. A pulse train whose alternate
        // pulses are attenuated correlates strongly at *twice* its period, so
        // a tracker without octave suppression reports f0/2. This is the
        // behaviour the reference golden used to cover implicitly.
        let f0 = 150.0f32;
        let period = SAMPLE_RATE as f32 / f0;

        let signal: Vec<f32> = (0..HOP_SIZE * 40)
            .map(|i| {
                let cycle = (i as f32 / period).floor() as usize;
                let pos = i as f32 % period;
                let amp = if cycle % 2 == 0 { 8000.0 } else { 5200.0 };
                amp * (-pos / (period * 0.08)).exp()
            })
            .collect();

        let pitches = run(&signal);
        let voiced: Vec<f32> = pitches[20..].iter().copied().filter(|p| *p > 0.0).collect();
        assert!(!voiced.is_empty(), "fixture should be voiced");

        let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
        assert!(
            (mean - f0).abs() / f0 < 0.15,
            "expected ~{f0} Hz, got {mean} Hz; ~{} Hz would be the octave error",
            f0 / 2.0,
        );
    }

    #[test]
    fn test_noise_is_not_reported_as_a_stable_pitch() {
        // Aperiodic input has no period to find. The estimator may call some
        // frames voiced -- noise does correlate by chance -- but it must not
        // settle on one period the way it does for a real pulse train.
        let mut seed = 0x5eed_1234u32;
        let noise: Vec<f32> = (0..HOP_SIZE * 40)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 6000.0
            })
            .collect();

        let noisy: Vec<f32> = run(&noise)[20..]
            .iter()
            .copied()
            .filter(|p| *p > 0.0)
            .collect();
        let tonal: Vec<f32> = run(&pulse_train(150.0, HOP_SIZE * 40, 0))[20..]
            .iter()
            .copied()
            .filter(|p| *p > 0.0)
            .collect();

        let spread = |v: &[f32]| -> f32 {
            if v.len() < 2 {
                return 0.0;
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt() / mean
        };

        // Either noise is mostly rejected, or what survives is far less
        // consistent than a real period.
        assert!(
            noisy.len() < tonal.len() / 2 || spread(&noisy) > 4.0 * spread(&tonal).max(1e-3),
            "noise gave {} voiced frames (spread {:.3}) against {} tonal (spread {:.3})",
            noisy.len(),
            spread(&noisy),
            tonal.len(),
            spread(&tonal),
        );
    }

    #[test]
    fn test_pitch_is_bounded_by_the_search_range() {
        // The regression can extrapolate, but never far outside the lags the
        // correlation actually searched.
        let signal = pulse_train(150.0, HOP_SIZE * 30, 0);
        for p in run(&signal) {
            if p > 0.0 {
                assert!(
                    (40.0..=600.0).contains(&p),
                    "{p} Hz is outside the plausible search range",
                );
            }
        }
    }

    #[test]
    fn test_reset_rewinds_to_start_of_stream() {
        let signal = pulse_train(140.0, HOP_SIZE * 12, 0);

        let mut est = HostPitchEstimator::new();
        let mut stft = HostStft::new();
        let first: Vec<f32> = hops(&signal)
            .map(|hop| est.frame_pitch(hop, &stft.push(hop)))
            .collect();

        // Drive it somewhere else entirely, then rewind.
        let other = tone(300.0, HOP_SIZE * 8, 0);
        let mut other_stft = HostStft::new();
        for hop in hops(&other) {
            est.frame_pitch(hop, &other_stft.push(hop));
        }

        est.reset();
        let mut stft = HostStft::new();
        let second: Vec<f32> = hops(&signal)
            .map(|hop| est.frame_pitch(hop, &stft.push(hop)))
            .collect();

        assert_eq!(first, second);
    }

    #[test]
    fn test_state_carries_across_hops() {
        // Feeding the same hop twice must not give the same answer, or the
        // tracker's history is not doing anything.
        let signal = pulse_train(150.0, HOP_SIZE * 16, 0);
        let pitches = run(&signal);

        let first = pitches[0];
        assert!(
            pitches.iter().any(|p| *p != first),
            "every hop produced the same estimate",
        );
    }

    #[test]
    fn test_clone_is_an_independent_stream() {
        let signal = pulse_train(130.0, HOP_SIZE * 16, 0);
        let mut a = HostPitchEstimator::new();
        let mut stft = HostStft::new();

        let mut powers = Vec::new();
        for hop in hops(&signal).take(8) {
            let power = stft.push(hop);
            a.frame_pitch(hop, &power);
            powers.push(power);
        }

        // A clone mid-stream continues identically to the original.
        let mut b = a.clone();
        let tail: Vec<&[f32]> = hops(&signal).skip(8).take(4).collect();
        let mut next_powers = Vec::new();
        for hop in &tail {
            next_powers.push(stft.push(hop));
        }

        let from_a: Vec<f32> = tail
            .iter()
            .zip(next_powers.iter())
            .map(|(hop, power)| a.frame_pitch(hop, power))
            .collect();
        let from_b: Vec<f32> = tail
            .iter()
            .zip(next_powers.iter())
            .map(|(hop, power)| b.frame_pitch(hop, power))
            .collect();

        assert_eq!(from_a, from_b);
    }

    #[test]
    fn test_voiced_flag_agrees_with_the_reported_pitch() {
        let signal = pulse_train(180.0, HOP_SIZE * 24, 0);
        let mut est = HostPitchEstimator::new();
        let mut stft = HostStft::new();

        for hop in hops(&signal) {
            let p = est.frame_pitch(hop, &stft.push(hop));
            assert_eq!(p > 0.0, est.voiced(), "voiced flag disagrees with {p} Hz");
            assert_eq!(p, est.pitch_hz());
        }
    }

    #[test]
    #[should_panic(expected = "expects a 256-sample hop")]
    fn test_wrong_hop_length_panics() {
        HostPitchEstimator::new().frame_pitch(&[0.0; 128], &[0.0; N_BINS]);
    }

    #[test]
    #[should_panic(expected = "expects 513 bins")]
    fn test_wrong_bin_count_panics() {
        HostPitchEstimator::new().frame_pitch(&[0.0; HOP_SIZE], &[0.0; 257]);
    }

    #[test]
    fn test_frame_pitch_with_lpc_reproduces_frame_pitch() {
        // The hybrid decomposition the device-side pre-filter is validated
        // through: designing the filter carries no state, so replaying a hop
        // against the filter that hop produced must be exact.
        let signal = pulse_train(150.0, HOP_SIZE * 24, 0);

        let mut whole = HostPitchEstimator::new();
        let mut split = HostPitchEstimator::new();
        let mut stft = HostStft::new();

        for hop in hops(&signal) {
            let power = stft.push(hop);
            let from_whole = whole.frame_pitch(hop, &power);
            // The filter `whole` designed for this hop; nothing after the
            // design stage writes it.
            let lpc = *whole.lpc();
            let from_split = split.frame_pitch_with_lpc(hop, &lpc);
            assert_eq!(from_whole, from_split);
        }

        // The residual state must agree too, not just the per-hop output.
        assert_eq!(whole.exc_buf(), split.exc_buf());
        assert_eq!(whole.path_score(), split.path_score());
        assert_eq!(whole.best_period(), split.best_period());
    }

    #[test]
    fn test_oracle_accessors_report_the_documented_shapes() {
        let signal = pulse_train(140.0, HOP_SIZE * 8, 0);
        let mut est = HostPitchEstimator::new();
        let mut stft = HostStft::new();
        for hop in hops(&signal) {
            est.frame_pitch(hop, &stft.push(hop));
        }

        assert_eq!(est.exc_buf().len(), est.max_period + 64 + 1);
        assert_eq!(est.frm_weight().len(), est.slots());
        assert_eq!(est.path_score().len(), est.dif_period);
        for sub in 0..est.slots() {
            assert_eq!(est.xcorr_slot(sub).len(), est.max_period);
            assert_eq!(est.path_prev_slot(sub).len(), est.dif_period);
        }
        assert!(est.best_period() < est.dif_period);
    }

    #[test]
    fn test_xcorr_slot_walks_in_stream_order() {
        // Each hop appends two half-hop slots, so the newest pair lands at
        // `slots-2` and `slots-1`, and everything older shifts left by two.
        // This is the translation the ring offset exists to hide.
        let signal = pulse_train(160.0, HOP_SIZE * 12, 0);
        let mut est = HostPitchEstimator::new();
        let mut stft = HostStft::new();

        let mut hop_iter = hops(&signal);
        for hop in hop_iter.by_ref().take(6) {
            est.frame_pitch(hop, &stft.push(hop));
        }

        let slots = est.slots();
        let before: Vec<Vec<f32>> = (0..slots).map(|s| est.xcorr_slot(s).to_vec()).collect();

        let hop = hop_iter.next().unwrap();
        est.frame_pitch(hop, &stft.push(hop));

        for sub in 0..(slots - 2) {
            assert_eq!(
                est.xcorr_slot(sub),
                before[sub + 2].as_slice(),
                "slot {sub} should hold what slot {} held before the hop",
                sub + 2,
            );
        }
    }

    #[test]
    fn test_path_score_carries_across_hops() {
        // The Viterbi accumulator is renormalized, never cleared — so after a
        // voiced run it must be non-trivial, with its peak pinned at zero.
        let signal = pulse_train(170.0, HOP_SIZE * 20, 0);
        let mut est = HostPitchEstimator::new();
        let mut stft = HostStft::new();
        for hop in hops(&signal) {
            est.frame_pitch(hop, &stft.push(hop));
        }

        let score = est.path_score();
        let peak = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (peak - 0.0).abs() < 1e-5,
            "peak should be renormalized to 0, got {peak}"
        );
        assert!(
            score.iter().any(|v| *v < -1e-6),
            "the accumulator should spread once the tracker has history",
        );
        assert!(est.path_best_all().is_finite());
    }

    #[test]
    fn test_usable_as_a_scalar_trait_object() {
        let mut source: Box<dyn PitchScalarSource> = Box::new(HostPitchEstimator::new());
        let p = source.frame_pitch(&[0.0; HOP_SIZE], &[0.0; N_BINS]);
        assert_eq!(p, 0.0);
        source.reset();
    }
}
