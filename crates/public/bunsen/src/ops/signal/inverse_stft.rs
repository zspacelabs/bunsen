//! Inverse STFT synthesis, at any FFT size.

use alloc::vec::Vec;
use core::f64::consts::PI;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        TensorData,
        s,
    },
    tensor::Device,
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::kernels::window::{
        SamplingWindowBuilder,
        StftWindowConfig,
    },
};

/// Common meta for [`InverseStftConfig`] and [`InverseStft`].
pub trait InverseStftMeta {
    /// The FFT size, `N`: each frame's samples.
    fn n_fft(&self) -> usize;

    /// The hop between frames, in samples.
    fn hop(&self) -> usize;

    /// The frequency bins a frame holds, `N / 2 + 1`.
    fn n_bins(&self) -> usize {
        self.n_fft() / 2 + 1
    }
}

/// Config for [`InverseStft`].
#[derive(Config, Debug)]
pub struct InverseStftConfig {
    /// The FFT size, at least 2; any size, not only a power of two.
    pub n_fft: usize,

    /// The hop between frames, in samples; at least 1, at most `n_fft`.
    pub hop: usize,

    /// The synthesis window, `n_fft` wide.
    #[config(default = "StftWindowConfig::Hann { periodic: true }")]
    pub window: StftWindowConfig,

    /// Whether the frames were centred on their positions (the analysis
    /// padded `n_fft / 2` on each side): the synthesis then drops the first
    /// `n_fft / 2` samples.
    #[config(default = true)]
    pub center: bool,
}

impl InverseStftMeta for InverseStftConfig {
    fn n_fft(&self) -> usize {
        self.n_fft
    }

    fn hop(&self) -> usize {
        self.hop
    }
}

impl ModuleInit<InverseStft> for InverseStftConfig {
    /// The inverse real DFT's bases and the window, computed in f64 on the
    /// host and kept in f32.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for an FFT size
    /// under 2, or a hop of 0 or over the FFT size.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<InverseStft> {
        let n = self.n_fft;
        if n < 2 || self.hop == 0 || self.hop > n {
            return Err(BunsenError::illegal(format!(
                "an inverse STFT of n_fft {n} and hop {}",
                self.hop
            )));
        }
        let bins = self.n_bins();
        // irfft: x[t] = Σ_k c_k (Re X_k cos(2π k t / N) - Im X_k sin(2π k t /
        // N)), c_k = 1 / N for the DC bin and (N even) the Nyquist bin, 2 / N
        // between.
        let weight = |k: usize| {
            if k == 0 || 2 * k == n {
                1.0 / n as f64
            } else {
                2.0 / n as f64
            }
        };
        let mut cos = Vec::with_capacity(bins * n);
        let mut sin = Vec::with_capacity(bins * n);
        for k in 0..bins {
            for t in 0..n {
                let angle = 2.0 * PI * ((k * t) % n) as f64 / n as f64;
                cos.push((weight(k) * angle.cos()) as f32);
                sin.push((-weight(k) * angle.sin()) as f32);
            }
        }
        let window: Vec<f32> = self
            .window
            .to_vec_window(n)
            .into_iter()
            .map(|w| w as f32)
            .collect();
        Ok(InverseStft {
            cos_basis: Tensor::from_data(TensorData::new(cos, [bins, n]), device),
            sin_basis: Tensor::from_data(TensorData::new(sin, [bins, n]), device),
            window: Tensor::from_data(TensorData::new(window, [n]), device),
            hop: self.hop,
            center: self.center,
        })
    }
}

/// Inverse STFT synthesis: spectra back to a waveform, as `torch.istft`
/// computes it, at any FFT size (burn-signal's is power-of-two only).
///
/// Each frame's spectrum is inverted by a real inverse DFT, written as two
/// matmuls against precomputed cos and sin bases (the imaginary parts of the
/// DC and Nyquist bins are ignored, as `irfft` ignores them); the frames are
/// windowed and overlap-added at the hop, and the sum divided by the summed
/// squared window at each sample. Centred frames drop the first `N / 2`
/// samples. The tables are bare tensors, not parameters.
///
/// Built by [`InverseStftConfig`].
#[derive(Module, Debug)]
pub struct InverseStft {
    /// `[N/2 + 1, N]`: the inverse DFT's cosine basis, bin weights folded in.
    pub cos_basis: Tensor<2>,

    /// `[N/2 + 1, N]`: its negated sine basis, bin weights folded in.
    pub sin_basis: Tensor<2>,

    /// `[N]`: the synthesis window.
    pub window: Tensor<1>,

    /// The hop between frames.
    #[module(skip)]
    pub hop: usize,

    /// Whether the frames are centred.
    #[module(skip)]
    pub center: bool,
}

impl InverseStftMeta for InverseStft {
    fn n_fft(&self) -> usize {
        self.window.dims()[0]
    }

    fn hop(&self) -> usize {
        self.hop
    }
}

/// `frames` `[B, F, N]` overlap-added at `hop`: `[B, (F - 1) hop + N]`. Each
/// frame is cut into hop-wide chunks (zero-padded to a whole number of
/// them), and chunk `j` of frame `f` lands on output chunk `f + j`.
fn overlap_add(
    frames: Tensor<3>,
    hop: usize,
) -> Tensor<2> {
    let [b, f, n] = crate::contracts::unpack_shape_contract!(["B", "F", "N"], &frames.dims());
    let chunks = n.div_ceil(hop);
    let padded = chunks * hop;
    let frames = if padded > n {
        let zeros = Tensor::zeros([b, f, padded - n], (&frames.device(), frames.dtype()));
        Tensor::cat(vec![frames, zeros], 2)
    } else {
        frames
    };
    let frames = frames.reshape([b, f, chunks, hop]);
    let mut sum = Tensor::<3>::zeros([b, f + chunks - 1, hop], (&frames.device(), frames.dtype()));
    for j in 0..chunks {
        let chunk = frames
            .clone()
            .slice(s![.., .., j..j + 1, ..])
            .reshape([b, f, hop]);
        let at = s![.., j..j + f, ..];
        sum = sum.clone().slice_assign(at, sum.slice(at) + chunk);
    }
    let signal = sum
        .reshape([b, (f + chunks - 1) * hop])
        .slice(s![.., ..(f - 1) * hop + n]);
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["B", "samples"],
        &signal.dims(),
        &[("B", b), ("samples", (f - 1) * hop + n)]
    );
    signal
}

impl InverseStft {
    /// The waveform of a sequence of spectra.
    ///
    /// # Arguments
    /// - `real`, `imag`: `[B, N/2 + 1, F]`, each frame's spectrum.
    /// - `length`: the samples to return (zero-padded past the synthesis' end);
    ///   `None`, all of them (less `N / 2` at each end when centred).
    ///
    /// # Returns
    /// - `[B, L]`.
    ///
    /// # Panics
    /// If the shapes disagree, or there are no frames.
    pub fn istft(
        &self,
        real: Tensor<3>,
        imag: Tensor<3>,
        length: Option<usize>,
    ) -> Tensor<2> {
        let (n, hop) = (self.n_fft(), self.hop);
        let [b, f] = unpack_shape_contract!(
            ["B", "K", "F"],
            &real.dims(),
            &["B", "F"],
            &[("K", self.n_bins())]
        );
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "K", "F"],
            &imag.dims(),
            &[("B", b), ("K", self.n_bins()), ("F", f)]
        );
        assert!(f > 0, "an inverse STFT of no frames");
        // [B, F, N]
        let frames = real
            .swap_dims(1, 2)
            .matmul(self.cos_basis.clone().unsqueeze())
            + imag
                .swap_dims(1, 2)
                .matmul(self.sin_basis.clone().unsqueeze());
        let window = self.window.clone().reshape([1, 1, n]);
        let y = overlap_add(frames * window.clone(), hop);
        let envelope = overlap_add(window.square().expand([1, f, n]), hop);
        let y = y / envelope.clamp_min(1e-11);

        let total = (f - 1) * hop + n;
        let start = if self.center { n / 2 } else { 0 };
        let end = match length {
            Some(length) => start + length,
            None if self.center => total - n / 2,
            None => total,
        };
        if end <= total {
            y.slice(s![.., start..end])
        } else {
            let zeros = Tensor::zeros([b, end - total], (&y.device(), y.dtype()));
            Tensor::cat(vec![y.slice(s![.., start..]), zeros], 1)
        }
    }

    /// The first output sample's position in the synthesis: `N / 2` when
    /// centred, else 0.
    fn output_start(&self) -> usize {
        if self.center { self.n_fft() / 2 } else { 0 }
    }

    /// A stream through the synthesis, not yet started.
    ///
    /// # Panics
    /// For a hop past the FFT size (frames that do not overlap or touch).
    pub fn new_stream_context(&self) -> InverseStftStreamContext {
        assert!(
            self.hop <= self.n_fft(),
            "a streamed inverse STFT's hop is at most its FFT size; got {} > {}",
            self.hop,
            self.n_fft()
        );
        InverseStftStreamContext {
            numerator: None,
            envelope: None,
            batch: 0,
            dtype: None,
            frames: 0,
            next: self.output_start(),
            finished: false,
        }
    }

    /// The stream's next frames' spectra, to the samples they make final.
    ///
    /// A sample is final once no later frame reaches it: after `F` frames,
    /// the synthesis' first `F · hop` samples, the first `N / 2` of them
    /// skipped when centred. Joined with [`finish_stream`](Self::finish_stream)'s,
    /// the stream's samples are [`istft`](Self::istft)'s over all the frames,
    /// for any output that reaches the last frame's hop: every `length` of
    /// at least `F · hop` less the skip, and `None` at a hop of at most
    /// `N / 2`.
    ///
    /// # Arguments
    /// - `real`, `imag`: `[B, N/2 + 1, T]`, `T ≥ 1`, the next frames' spectra.
    ///
    /// # Returns
    /// `[B, n]`, the `n ≥ 1` samples now final; `None` when none is (a
    /// centred stream's first `N / 2`).
    ///
    /// # Panics
    /// As [`istft`](Self::istft); or for a finished stream.
    pub fn istft_stream(
        &self,
        real: Tensor<3>,
        imag: Tensor<3>,
        ctx: &mut InverseStftStreamContext,
    ) -> Option<Tensor<2>> {
        assert!(!ctx.finished, "a push to a finished inverse STFT stream");
        let (n, hop) = (self.n_fft(), self.hop);
        let [b, t] = unpack_shape_contract!(
            ["B", "K", "F"],
            &real.dims(),
            &["B", "F"],
            &[("K", self.n_bins())]
        );
        assert!(t > 0, "an inverse STFT stream's push of no frames");
        ctx.batch = b;
        ctx.dtype = Some(real.dtype());
        let frames = real
            .swap_dims(1, 2)
            .matmul(self.cos_basis.clone().unsqueeze())
            + imag
                .swap_dims(1, 2)
                .matmul(self.sin_basis.clone().unsqueeze());
        let window = self.window.clone().reshape([1, 1, n]);
        let mut y = overlap_add(frames * window.clone(), hop);
        let mut envelope = overlap_add(window.square().expand([1, t, n]), hop);
        let tail = n - hop;
        if let (Some(num), Some(env)) = (ctx.numerator.take(), ctx.envelope.take()) {
            let head = s![.., 0..tail];
            y = y.clone().slice_assign(head, y.slice(head) + num);
            envelope = envelope
                .clone()
                .slice_assign(head, envelope.slice(head) + env);
        }
        let done = t * hop;
        let len = (t - 1) * hop + n;
        if tail > 0 {
            ctx.numerator = Some(y.clone().slice(s![.., done..len]));
            ctx.envelope = Some(envelope.clone().slice(s![.., done..len]));
        }
        let first = ctx.frames * hop;
        ctx.frames += t;
        let last = first + done;
        if last <= ctx.next {
            return None;
        }
        let from = ctx.next - first;
        ctx.next = last;
        Some(y.slice(s![.., from..done]) / envelope.slice(s![.., from..done]).clamp_min(1e-11))
    }

    /// The end of the stream: the samples no later frame would have reached,
    /// to the length [`istft`](Self::istft) would return.
    ///
    /// # Arguments
    /// - `length`: as `istft`'s, the samples of the whole output (zero-padded
    ///   past the synthesis' end); `None`, all of them (less `N / 2` at each
    ///   end when centred).
    ///
    /// # Returns
    /// `[B, n]`, the rest; `None` when nothing is left, no frame was pushed,
    /// or the stream was finished already.
    ///
    /// # Panics
    /// For a `length` shorter than the samples already returned.
    pub fn finish_stream(
        &self,
        ctx: &mut InverseStftStreamContext,
        length: Option<usize>,
    ) -> Option<Tensor<2>> {
        if ctx.finished {
            return None;
        }
        ctx.finished = true;
        let (n, hop) = (self.n_fft(), self.hop);
        let f = ctx.frames;
        if f == 0 {
            return None;
        }
        let start = self.output_start();
        let total = (f - 1) * hop + n;
        let end = match length {
            Some(length) => start + length,
            None if self.center => total - n / 2,
            None => total,
        };
        assert!(
            end >= ctx.next,
            "an inverse STFT stream cut to {end} samples has returned {} already",
            ctx.next
        );
        let first = f * hop;
        let mut parts = Vec::with_capacity(2);
        let synthesized = end.min(total);
        if synthesized > ctx.next {
            let num = ctx.numerator.take().expect("the synthesis' tail");
            let env = ctx.envelope.take().expect("the envelope's tail");
            let range = s![.., ctx.next - first..synthesized - first];
            parts.push(num.slice(range) / env.slice(range).clamp_min(1e-11));
        }
        if end > total.max(ctx.next) {
            let pad = end - total.max(ctx.next);
            let dtype = ctx.dtype.unwrap_or_else(|| self.window.dtype());
            parts.push(Tensor::zeros(
                [ctx.batch, pad],
                (&self.window.device(), dtype),
            ));
        }
        let returned = end - ctx.next;
        ctx.next = end;
        ctx.numerator = None;
        ctx.envelope = None;
        let rest = (!parts.is_empty()).then(|| Tensor::cat(parts, 1));
        // The rest is the stream's batch, through its cut.
        #[cfg(any(debug_assertions, test))]
        if let Some(rest) = &rest {
            crate::contracts::assert_shape_contract_periodically!(
                ["B", "returned"],
                &rest.dims(),
                &[("B", ctx.batch), ("returned", returned)]
            );
        }
        #[cfg(not(any(debug_assertions, test)))]
        let _ = returned;
        rest
    }
}

/// One stream through an [`InverseStft`]: the overlap-add's sums past the
/// last final sample, and where the output stands.
///
/// Made by [`InverseStft::new_stream_context`], driven by
/// [`istft_stream`](InverseStft::istft_stream) and
/// [`finish_stream`](InverseStft::finish_stream). Each batch row is one
/// stream.
///
/// A `Module` over bare tensors: the sums move with `to_device`, and are
/// neither recorded nor visited.
#[derive(Module, Debug)]
pub struct InverseStftStreamContext {
    /// `[B, N - hop]`: the windowed frames' overlap-add past the last final
    /// sample; `None` before the first frame, or at a hop of `N`.
    pub numerator: Option<Tensor<2>>,

    /// `[1, N - hop]`: the squared window's overlap-add there.
    pub envelope: Option<Tensor<2>>,

    /// The rows, `B`; 0 before the first frame.
    pub batch: usize,

    /// The spectra's dtype; `None` before the first frame.
    #[module(skip)]
    pub dtype: Option<burn::tensor::DType>,

    /// The frames pushed so far.
    pub frames: usize,

    /// The synthesis' next sample to return.
    pub next: usize,

    /// Whether [`finish_stream`](InverseStft::finish_stream) has run.
    pub finished: bool,
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        performance_device,
        seeded_tensor,
    };

    /// `x`'s centred, windowed frames' spectra, by a direct real DFT on the
    /// host in f64: `([B, K, F], [B, K, F])`.
    fn spectra(
        x: &[Vec<f64>],
        n: usize,
        hop: usize,
        window: &[f64],
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let bins = n / 2 + 1;
        let len = x[0].len();
        let padded = len + n;
        let frames = (padded - n) / hop + 1;
        let (mut re, mut im) = (Vec::new(), Vec::new());
        for row in x {
            let mut p = vec![0.0; n / 2];
            p.extend(row);
            p.extend(vec![0.0; n - n / 2]);
            let (mut r, mut i) = (vec![0.0; bins * frames], vec![0.0; bins * frames]);
            for fr in 0..frames {
                for k in 0..bins {
                    let (mut a, mut c) = (0.0, 0.0);
                    for t in 0..n {
                        let v = p[fr * hop + t] * window[t];
                        let angle = 2.0 * PI * ((k * t) % n) as f64 / n as f64;
                        a += v * angle.cos();
                        c -= v * angle.sin();
                    }
                    r[k * frames + fr] = a;
                    i[k * frames + fr] = c;
                }
            }
            re.extend(r.into_iter().map(|v| v as f32));
            im.extend(i.into_iter().map(|v| v as f32));
        }
        (re, im, frames)
    }

    /// A centred analysis inverted gives the signal back, at sizes that are
    /// not powers of two, with a hop that does not divide the FFT size, and
    /// at ZONOS3's codec head's two sizes.
    #[test]
    fn test_round_trip() {
        let device = cpu_device();
        for (n, hop, len) in [(12, 4, 37), (10, 4, 23), (736, 184, 900), (2208, 184, 1500)] {
            let istft: InverseStft = InverseStftConfig::new(n, hop).init(&device);
            assert_eq!(
                (istft.n_fft(), istft.hop(), istft.n_bins()),
                (n, hop, n / 2 + 1)
            );
            let x: Tensor<2> = seeded_tensor(n as u64, [2, len], Distribution::Default, &device);
            let rows: Vec<Vec<f64>> = (0..2)
                .map(|r| {
                    x.clone()
                        .slice(s![r..r + 1])
                        .into_data()
                        .try_into_vec_as::<f64>()
                        .unwrap()
                })
                .collect();
            let window = StftWindowConfig::Hann { periodic: true }.to_vec_window(n);
            let (re, im, frames) = spectra(&rows, n, hop, &window);
            let bins = n / 2 + 1;
            let real = Tensor::<3>::from_data(TensorData::new(re, [2, bins, frames]), &device);
            let imag = Tensor::<3>::from_data(TensorData::new(im, [2, bins, frames]), &device);
            let y = istft.istft(real, imag, Some(len));
            assert_tensors_close(&y, &x, Tolerance::rel_abs(1e-4, 1e-4));
        }
    }

    /// The overlap-add places each frame at its hop, the frame wider than the
    /// hop or not a multiple of it; and on the performance device too.
    #[test]
    fn test_overlap_add() {
        for device in [cpu_device(), performance_device()] {
            let frames = Tensor::<3>::from_data(
                [[[1.0f32, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]],
                &device,
            );
            let y = overlap_add(frames, 2);
            assert_eq!(
                y.into_data().try_into_vec_as::<f32>().unwrap(),
                [1.0, 2.0, 13.0, 24.0, 35.0, 40.0, 50.0]
            );
        }
    }

    /// Lengths: `None` drops `N / 2` at each end when centred; a longer
    /// `length` pads with zeros.
    #[test]
    fn test_lengths() {
        let device = cpu_device();
        let istft: InverseStft = InverseStftConfig::new(8, 2).init(&device);
        let spec = Tensor::<3>::ones([1, 5, 6], &device);
        // 6 frames: (6 - 1) * 2 + 8 = 18 samples, less 4 at each end.
        assert_eq!(
            istft.istft(spec.clone(), spec.clone(), None).dims(),
            [1, 10]
        );
        let long = istft.istft(spec.clone(), spec.clone(), Some(20));
        assert_eq!(long.dims(), [1, 20]);
        assert_eq!(
            long.slice(s![.., 14..])
                .into_data()
                .try_into_vec_as::<f32>()
                .unwrap(),
            [0.0; 6]
        );
        let uncentred: InverseStft = InverseStftConfig::new(8, 2)
            .with_center(false)
            .init(&device);
        assert_eq!(uncentred.istft(spec.clone(), spec, None).dims(), [1, 18]);
        assert!(InverseStftConfig::new(8, 9).try_init(&device).is_err());
        assert!(InverseStftConfig::new(1, 1).try_init(&device).is_err());
    }

    /// A stream's samples, with its finish's, are the whole synthesis',
    /// whatever the chunks of frames: centred or not, a hop that divides the
    /// FFT size or not, one of the FFT size (no overlap), ZONOS3's head
    /// sizes; each to the whole's own length and to a longer one.
    #[test]
    fn test_stream_matches_istft() {
        let device = cpu_device();
        let cases = [
            (12, 4, true),
            (10, 4, true),
            (10, 4, false),
            (6, 6, true),
            (736, 184, true),
            (2208, 184, true),
        ];
        for (i, &(n, hop, center)) in cases.iter().enumerate() {
            let istft: InverseStft = InverseStftConfig::new(n, hop)
                .with_center(center)
                .init(&device);
            let bins = n / 2 + 1;
            for f in [1, 3, 9] {
                let real: Tensor<3> = seeded_tensor(
                    (10 * i + f) as u64,
                    [2, bins, f],
                    Distribution::Default,
                    &device,
                );
                let imag: Tensor<3> = seeded_tensor(
                    (10 * i + f + 100) as u64,
                    [2, bins, f],
                    Distribution::Default,
                    &device,
                );
                for length in [None, Some(f * hop), Some(f * hop + n)] {
                    let start = if center { n / 2 } else { 0 };
                    let end = length.map_or((f - 1) * hop + n - start, |l| start + l);
                    if end < f * hop {
                        // A cut before the last frame's hop: no stream serves
                        // it (a hop past N / 2, the length `None`).
                        continue;
                    }
                    let whole = istft.istft(real.clone(), imag.clone(), length);
                    for size in [1, 2, f] {
                        let mut ctx = istft.new_stream_context();
                        let mut parts = Vec::new();
                        for at in (0..f).step_by(size) {
                            let end = (at + size).min(f);
                            parts.extend(istft.istft_stream(
                                real.clone().slice(s![.., .., at..end]),
                                imag.clone().slice(s![.., .., at..end]),
                                &mut ctx,
                            ));
                        }
                        parts.extend(istft.finish_stream(&mut ctx, length));
                        if parts.is_empty() {
                            // One centred frame, its own length: no samples.
                            assert_eq!(whole.dims()[1], 0);
                            continue;
                        }
                        assert_tensors_close(
                            &Tensor::cat(parts, 1),
                            &whole,
                            Tolerance::rel_abs(1e-5, 1e-5),
                        );
                    }
                }
            }
        }
    }

    /// A centred stream returns nothing until a frame passes `N / 2`; then
    /// each frame's hop, `F · hop - N / 2` in all.
    #[test]
    fn test_stream_latency() {
        let device = cpu_device();
        let istft: InverseStft = InverseStftConfig::new(12, 4).init(&device);
        let mut ctx = istft.new_stream_context();
        let frame = || Tensor::<3>::ones([1, 7, 1], &device);
        assert!(istft.istft_stream(frame(), frame(), &mut ctx).is_none());
        let y = istft
            .istft_stream(frame(), frame(), &mut ctx)
            .expect("past N / 2");
        assert_eq!(y.dims(), [1, 2]);
        let y = istft
            .istft_stream(frame(), frame(), &mut ctx)
            .expect("a hop");
        assert_eq!(y.dims(), [1, 4]);
        assert_eq!(ctx.next, 3 * 4);
    }
}
