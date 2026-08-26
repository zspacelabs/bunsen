# MelConverter build plan (bunsen)

Working plan for `bunsen::ops::signal::mels`. Supersedes the draft in
`~/Downloads/mel_converter_plan.md`; Stage 0 of that draft is **resolved** (see
[Findings](#findings-stage-0-resolved)) and the `transform` API is restructured into a
**stage stack** (see [The stage stack](#the-stage-stack)).

---

## Target API

```text
MelConverterOptions   --try_init(device)-->   MelConverter<B>          // Module, precomputed tensors
MelConverter::new_context(batch)          ->  MelConversionContext<B>  // t = 0
MelConversionContext::transform(self, waves: `[batch, samples]`) -> (`[batch, frames, n_mels]`, Self)
MelConversionContext::finish(self)        ->  Option<`[batch, frames, n_mels]`>   // end padding only
```

Reference to match: Whisper / librosa — periodic Hann, power spectrum, Slaney mel with
Slaney area norm, `log10` after a `1e-10` floor, `center=True` reflect. Defaults:
`sample_rate` 16000, `n_fft` 400, `hop` 160, `n_mels` 80.

Guiding rule for every stage: **precomputed constant → pure per-frame op → carried state.**
Each stage gets (a) a CPU-side reference in plain `Vec<f64>`, (b) the Burn op, (c) a test
that they agree, before moving on. Golden values from Python (`librosa` / `torch`) are
committed as small fixtures, not regenerated in CI.

---

## Findings (Stage 0, resolved)

Read out of `burn-tensor-0.21.0/src/tensor/signal/`; no probe code needed. **The
power-of-two constraint is the decisive fact and it reshapes the plan.**

| Question | Answer |
|---|---|
| `burn::tensor::signal` exports | `hann_window`, `hamming_window`, `blackman_window`, `rfft`, `irfft`, `fft`, `hermitian_extend`, `stft`, `istft`, `StftOptions` |
| `hann_window` convention | `hann_window::<B>(size, periodic: bool, options)` — explicit flag; `0.5 - 0.5·cos(2πn/N)`, `N = size` when periodic. `size==0` → empty, `size==1` → `[1.0]` |
| `rfft` signature / layout | `rfft(signal, dim, n: Option<usize>) -> (re, im)` — **two separate tensors**, not interleaved; length along `dim` is `n/2 + 1` |
| `rfft` with `n = 400` | **Panics.** `n` must be a power of two (`fft.rs:70`); Bluestein is an upstream follow-up |
| `stft` args | `stft(signal: [batch, samples], Option<window>, StftOptions{n_fft, hop_length, win_length, center, onesided}) -> [batch, n_frames, n_freqs, 2]`. `center: true` does reflect padding of `n_fft/2` both sides |
| `stft` with `n_fft = 400` | **Panics.** `StftOptions::assert_valid` asserts `n_fft.is_power_of_two()` (`stft.rs:49`) — even though `StftOptions::default()` is `new(400)` |
| `stft` window handling | A `win_length < n_fft` window is **center**-padded to `n_fft`. `SlidingStft` works around this by pre-padding its own window and passing `win_length: None` |
| `unfold` availability | **Yes** — `Tensor::unfold::<D2>(dim, size, step) -> [pre.., windows, post.., size]` (`base.rs:2767`). burn's own `stft` uses `signal.unfold(1, n_fft, hop)` and reshapes to `1 + (len - n_fft)/hop` frames |
| autodiff | `rfft` (and therefore `stft`) has no backward. A DFT matmul **does**, so `DftMatmul` is also the only differentiable path |

### Decision gate — resolved

- **`SpectrumImpl::DftMatmul` is the default, and the only impl that reaches Whisper parity.**
  `n_fft = 400` is not a power of two, so `rfft`/`stft` are unusable at the reference geometry.
- `SpectrumImpl::Stft` (delegating to `burn::tensor::signal::stft`) stays as an opt-in fast
  path, **validated to reject non-power-of-two `n_fft`** rather than panicking inside burn.
- `pad_to_pow2` is **not** a workaround for the above. Zero-padding 400 → 512 moves the bin
  centres (`k·sr/512` instead of `k·sr/400`) and yields 257 bins, so it cannot match librosa.
  Keep the flag, but scope it honestly: it is the *Kaldi* `round_to_power_of_two` feature —
  keep `win_len` samples of window, zero-pad the frame to `fft_size = n_fft.next_power_of_two()`.
  That is exactly the geometry `SlidingStft` already implements. Default `false`.
- `RangeClamp::Running` — **drop from v1.** Whisper's clamp is `max(log, log.max() - 8)` over
  the whole clip, which is `PerCall` on a batch call and `Fixed` on a streaming one. A running
  max makes the streaming output non-reproducible against any reference and breaks the
  homomorphism test. Ship `PerCall` and `Fixed(f32)`; add `Running` later if a model wants it.

---

## Codebase integration notes

Things the draft assumed that this repo does differently.

- **Errors.** `BunsenError` has exactly four variants: `ResourceNotFound`, `ParseError`,
  `Invalid`, `External` (`crates/bunsen/src/errors/mod.rs`). There is no `HopMisaligned`.
  Use `BunsenError::Invalid(format!(...))` with a message naming the numbers, matching
  `SlidingStftConfig::validate`. Do not add a variant just for this.
- **Init lifecycle.** `MelConverterOptions` already implements
  `ModuleInit<B, MelConverter<B>>` (`try_init` + a provided `init` that panics via
  `ok_or_panic`). Keep that; do not hand-roll a second `init`.
- **File layout.** A `Config` lives in the **same file as the `Module` it builds** —
  `SlidingStftConfig` beside `SlidingStft`, `MelConverterOptions` beside `MelConverter`. Only
  genuinely separate concerns get their own file (`filterbank.rs` is the mel scale and triangle
  construction, which stands alone and has no `Module`).
- **The `{Name}Meta` trait.** Geometry that both a `Config` and a live `Module` must answer goes
  on a shared `{Name}Meta` trait — `SlidingStftMeta`, `MelConverterMeta` — so test and reflective
  code can hold either. Keep it narrow: **only** values needed for that uniform access. Everything
  else stays on the options and is reached from the module via an `options()` accessor.
- **Variant-specific behaviour belongs on the enum.** A `match` over an option enum that computes
  a value or applies a transform is a method on that enum, not an inline match at the use site:
  `FilterNorm::gain(f_lo, f_hi)`, `PaddingMode::pad_len(n_fft)`, `SpectrumKind::from_power(t)`,
  `LogBase::apply(t)`, `RangeClamp::apply(t)`, `AffineCompress::apply(t)`. The pipeline stage then
  reads as a sequence of named steps, and adding a variant changes one place.
  `SlidingStftConfig` now does the same — being a `Module` is what made it eligible, since
  `ModuleInit<B, M>` requires `M: Module<B>`. One call-site consequence to know before copying
  the pattern: `init`/`try_init` are trait methods with no generics of their own, so the old
  `cfg.init::<B>(&device)` turbofish no longer compiles. `B` has to come from the binding:
  `let coef: SlidingStft<B> = cfg.init(&device);`.
- **Everything with tensors is a `Module` — see [Module policy](#module-policy).** Both
  `MelConverter`/`MelConversionContext` **and** `SlidingStft`/`SlidingStftContext` derive
  `Module` over bare `Tensor` fields (never `Param`). `SlidingStft`'s current
  "deliberately **not** a `Module`" rustdoc is wrong and gets retracted as part of this work.
- **Prior art to mirror, not reuse.** `SlidingStft` / `SlidingStftContext`
  (`ops/signal/sliding_stft.rs`) is the existing streaming pattern: fixed coefficients +
  a context holding the sample queue, with `forward` (one hop) and `forward_sequence`
  (N hops, one `stft` call). It is **not** reusable here — it is hard-wired to
  power-of-two `fft_size` and a zero-padded window. But copy its shape: the `SlidingStftMeta`
  trait for shared geometry accessors, `batch_size()`, `reset()`, `#[cfg(any(test, debug_assertions))]`
  shape contracts around every public tensor entry point.
- **Divergence to call out.** `SlidingStftContext::forward` takes `&mut self`; this plan's
  `transform` takes `self` and returns `(out, Self)`. The by-value form is what makes the
  stage stack compose, so keep it — but say so in the module docs.
- **Whisper layout mismatch.** `WhisperModel::forward_encoder` wants `[batch, n_mels, seq]`
  (`kits/speech/whisper/blocks/whisper_model.rs:209`), this returns `[batch, frames, n_mels]`.
  Keep `[batch, frames, n_mels]` — `frames` is the growing axis, so streaming chunks
  concatenate along dim 1 and the homomorphism test is a plain `Tensor::cat`. Transpose at
  the whisper boundary with `.swap_dims(1, 2)` and document it on `transform`.
- **Style.** `STYLE.md` governs: shapes as a single backtick span in square brackets,
  shape-first phrasing, rustdoc on every public item. Formatting needs **nightly** rustfmt
  (`rustfmt.toml` sets `unstable_features`, while `rust-toolchain.toml` pins stable) — the
  IDE's **Format** run config already overrides the channel to nightly, so use it rather than
  a bare `cargo fmt`. Run tests with `--features wgpu` — a bare `cargo test` silently falls
  back to CPU. Prefer the RustRover MCP layer for format / fix / build / test and for edits.
- **Fixtures.** `crates/bunsen/testdata/` currently holds only `silero/` (1.9 MB, no LFS).
  Mel fixtures go in `crates/bunsen/testdata/mels/`. Keep the whole set under ~2 MB
  (see [Fixtures](#fixtures)).
- **Changelog.** Handled by release-plz from Conventional Commit subjects; do not hand-edit
  `CHANGELOG.md`. Use `feat(signal): ...` subjects.

---

## Module policy

**Decision: every type in `ops::signal` that owns a `Tensor` derives `Module<B>`, over bare
`Tensor` fields — never `Param`.** That includes retrofitting `SlidingStft` and
`SlidingStftContext`, whose current rustdoc claims the opposite.

The old rationale ("nothing here is a learnable parameter, so it isn't a `Module`") conflates
two things. `Param` is what marks a tensor learnable and persistent; `Module` is the
*traversal and device-mapping* trait. A struct can be a `Module` full of constants, and that
is exactly what buys the device handling.

### What this buys — verified against burn 0.21

`impl<const D, B, K> Module<B> for Tensor<B, D, K>` (`burn-core/src/module/param/constant.rs`)
provides real implementations of `to_device`, `fork` and `collect_devices`, and the derive
(`burn-derive/src/module/codegen_struct.rs`, `gen_to_device`) forwards those **per field**. So:

- `stft.to_device(&other)` moves the window / filterbank / DFT matrices — the point of the change.
- A model can hold a `SlidingStft<B>` or `MelConverter<B>` as a plain field and device
  propagation happens automatically. Today that has to be hand-written at every embedding site,
  which is the actual bug this fixes.
- `AutodiffModule` (`valid` / `from_inner`) and `ModuleDisplay` come along for free.
- `type Record = EmptyRecord`, so these constants are **not** written into checkpoints, and
  `num_params()` stays 0. Both correct: they are derived from config and rebuilt by `try_init`.

### What this does not buy — the footgun to document

⚠️ **`visit` and `map` are no-ops for a bare `Tensor`.** This is structural, not an oversight:
`ModuleVisitor`'s hooks are `visit_float(&Param<Tensor<..>>)` / `visit_int` / `visit_bool` —
there is no `visit_tensor`. And the derive's `gen_visit` only emits traversal for fields that
`is_parameter_module() || maybe_generic_module()`. Consequences:

- Any `ModuleMapper` pass — dtype casting, quantization — **silently skips these tensors.**
  Cast a model holding a `MelConverter` to f16 and the filterbank stays f32; the failure
  surfaces later as a dtype-mismatch panic inside a matmul, far from the cause.
- bunsen's own `XmlModuleTreeBuilder` (`burner/module/reflection/`) is a `ModuleVisitor`, so
  these tensors do **not** appear in the reflection XML tree.

Neither is a reason to avoid `Module` — `to_device` is the thing being bought and it works.
But say it out loud in the rustdoc of every such type, and if a dtype-mapping story is ever
needed, implement `map` by hand rather than assuming the derive covers it.

### Forward-compat note

`burn-core/src/module/param/constant.rs` carries a literal `// TODO: tensor record should
persist`, and `ConstantRecord` is already deprecated in favour of `EmptyRecord` with the note
"misleading as it doesn't persist data". If upstream makes bare-`Tensor` records persistent,
these derived constants would start bloating checkpoints. Not a blocker; worth a grep on the
next burn bump.

### Work item

Retrofitting `SlidingStft` is a small, self-contained change and should land as its own
commit **before** the mel work, so the mel types have a consistent pattern to copy:

1. Derive `Module` on `SlidingStft<B>` and `SlidingStftContext<B>`. The `usize` geometry
   fields are fine — burn's `empty!` macro implements `Module` for `usize` and friends.
2. Delete the "This is deliberately **not** a burn `Module`" paragraph; replace it with the
   constants-not-params rationale and the `visit`/`map` caveat above.
3. Add a test that `to_device` actually moves the window and the queue — the existing suite
   has no device-movement coverage at all.
4. `SlidingStftContext` holds `coef: SlidingStft<B>`, so its derive picks up the nested
   forward automatically; assert that in the same test.

Commit subject: `refactor(signal): derive Module on SlidingStft for device propagation`.

---

## The stage stack

**This is the structural change from the draft.** `transform` is not a monolith; it is the
composition of a fixed stack of stage methods, each with the same shape:

```rust
fn t_stage_<name>(self, a: A) -> (B, Self)
```

Consumes the context, returns the stage output plus the advanced context. `transform` is then
literally a fold over the stack, and every intermediate is individually observable.

```rust
impl<B: Backend> MelConversionContext<B> {
    /// Transforms `[batch, samples]` waveform chunk into `[batch, frames, n_mels]` log-mels.
    pub fn transform(self, waves: Tensor<B, 2>) -> BunsenResult<(Tensor<B, 3>, Self)> {
        let (x,    this) = self.t_stage_extend(waves)?;
        let (x,    this) = this.t_stage_preproc(x);
        let (x,    this) = this.t_stage_frame(x);
        let (x,    this) = this.t_stage_spectrum(x);
        let (x,    this) = this.t_stage_mel(x);
        let (mels, this) = this.t_stage_compress(x);
        Ok((mels, this))
    }
}
```

### Visibility

`pub(crate)` + `#[doc(hidden)]`. Private-to-the-crate keeps them off the public API surface
(no semver commitment, no `missing_docs` obligation) while still letting a
`mels::testing` helper module drive them — the same shape as
`ops::signal::window_builder::testing`, which is gated `#[cfg(any(test, feature = "testing"))]`
and `testing` is a **default** feature of `bunsen`. Document each one anyway: the shape
contract is the whole point.

### The stages

| # | Stage | In → Out | Touches state |
|---|---|---|---|
| 1 | `t_stage_extend` | `[batch, n]` → `[batch, ext]` | `carry`, `phase` |
| 2 | `t_stage_preproc` | `[batch, ext]` → `[batch, ext']` | — |
| 3 | `t_stage_frame` | `[batch, ext']` → `[batch, frames, n_fft]` | — |
| 4 | `t_stage_spectrum` | `[batch, frames, n_fft]` → `[batch, frames, n_bins]` | — |
| 5 | `t_stage_mel` | `[batch, frames, n_bins]` → `[batch, frames, n_mels]` | — |
| 6 | `t_stage_compress` | `[batch, frames, n_mels]` → `[batch, frames, n_mels]` | `clamp_ref` (Fixed only) |

**1. `t_stage_extend`** — the only genuinely stateful stage, and the only fallible one.
Validates `waves.dims()[0] == batch` and `n % hop == 0`. On `Phase::Start` it prepends the
start padding (`Reflect` → `flip(waves[.., 1..=n_fft/2])`, `Zero` → zeros, `None` → nothing),
erroring if `Reflect` and `n < n_fft/2 + 1`. On `Phase::Running` it prepends `carry`. It then
computes `frames = (ext - n_fft)/hop + 1`, sets `carry = ext_signal[.., frames*hop ..]`, and
`phase = Running`. **Derive `carry` from the frame count; do not hardcode `carry_len`** — see
[Frame arithmetic](#frame-arithmetic). The carry holds **raw** samples (pre-emphasis needs the
unfiltered history).

**2. `t_stage_preproc`** — DC removal then pre-emphasis `y[n] = x[n] - a·x[n-1]`. Both are
identity when unconfigured, so this is a no-op passthrough by default. Pre-emphasis needs one
sample of history: extend `carry` by one sample when it is enabled and drop the leading
filtered sample here. Stage 1 owns that carry adjustment; this stage owns the drop.

**3. `t_stage_frame`** — `x.unfold::<3>(1, n_fft, hop)` → `[batch, frames, n_fft]`, then
multiply by the `[1, 1, n_fft]`-broadcast window. Fold the window here, *not* into the DFT
matrices, so `SpectrumImpl::Stft` and `DftMatmul` share one framing path and one windowed
intermediate — that makes finding #1 in Stage 5's tests an apples-to-apples comparison.

**4. `t_stage_spectrum`** — dispatch on `SpectrumImpl`:
- `DftMatmul`: `[batch·frames, n_fft] @ [n_fft, n_bins]` twice (cos, sin) → `re`, `im`,
  then `re² + im²` (`Power`) or `sqrt` of that (`Magnitude`).
- `Stft`: reshape to `[batch, ext']`… — actually, call `stft` on the *unframed* signal.
  This impl therefore **bypasses stages 3–4** and is wired as a fused alternative rather than
  a drop-in stage; keep the dispatch at `transform` level with a documented comment. Only
  reachable when `n_fft.is_power_of_two()`.

**5. `t_stage_mel`** — `[batch·frames, n_bins] @ mel_t[n_bins, n_mels]` → reshape back.

**6. `t_stage_compress`** — `log_base(max(x, log_floor))`; then `RangeClamp` (`PerCall`:
`max(v, v.max() - db)` reduced over `[frames, n_mels]` **per batch row**, not across rows;
`Fixed(m)`: `max(v, m - db)`); then the affine.

### Why this shape

- Every stage is testable in isolation against a `Vec<f64>` reference without constructing a
  whole valid stream.
- Stage boundaries are exactly the tensor shapes the tests want to assert.
- Adding a stage (e.g. per-channel energy normalisation) is a one-line insert plus one test,
  not surgery on a 60-line function.
- The state mutation is confined to stages 1 and 6, which is the thing worth reviewing.

### What `finish` does

`finish(self) -> Option<Tensor<B, 3>>`: `None` when `end_padding == None`. Otherwise
reflect/zero-pad `carry` on the right by `n_fft/2` and run stages **3 → 6** on it
(no `t_stage_extend`, no new carry). Requires `carry_len > n_fft/2` for a valid reflect —
assert it, it holds for every legal geometry (see below).

---

## Frame arithmetic

Worked out and verified numerically; commit these as literal test expectations.

Let `pad` be the start padding (`n_fft/2` for Reflect/Zero, else 0), `S` the extended length,
`F = (S - n_fft)/hop + 1`, `carry_len = S - F·hop`.

Write `S - n_fft = q·hop + r`. Then `carry_len = n_fft - hop + r`, so
`carry_len ∈ [n_fft - hop, n_fft - 1]`.

Because chunks are hop-aligned, `r` depends only on `S mod hop = pad mod hop` — **constant
across calls**. So `carry_len` is invariant after the first call, and in steady state
**`F = n / hop` exactly**. Assert that.

**Whisper geometry** (`n_fft = 400`, `hop = 160`, `pad = 200`, `L = 480000` = 30 s):

| | frames | carry |
|---|---|---|
| librosa `center=True`, whole signal | 3001 | — |
| `transform` (first + only call) | **2999** | **360** |
| `finish` (Reflect end padding) | **2** | — |
| total | **3001** ✓ | |

`carry_len = n_fft - hop + ((pad - n_fft) mod hop) = 400 - 160 + 120 = 360` ✓, and
`360 > 200 = n_fft/2`, so the end reflect is valid. Note `360 ≥ 201`, which means reflecting
the carry is *bit-identical* to reflecting the full signal — the reflect only reads the last
`n_fft/2 + 1` samples. Assert this as the correctness argument for streaming `finish`.

Whisper's own `log_mel_spectrogram` then does `stft[..., :-1]` — it drops the **last** frame,
giving 3000. Model that as an explicit option or as "call `transform` and skip `finish`, then
take one tail frame"; do **not** bury a `-1` in the frame formula.

---

## Stage 1 — Options + validation (pure, no tensors)

`MelConverterOptions`, all `#[config(default = ...)]`:

| group | field | default |
|---|---|---|
| spectral | `sample_rate`, `n_fft`, `window: StftWindowConfig`, `pad_to_pow2: bool` | 16000, 400, `Hann { periodic: true }`, false |
| mel | `n_mels`, `f_min`, `f_max: Option<f32>`, `mel_scale: {Slaney, Htk}`, `filter_norm: {Slaney, None}`, `spectrum: {Power, Magnitude}` | 80, 0.0, `None` (= sr/2), Slaney, Slaney, Power |
| framing | `hop`, `start_padding: {None, Zero, Reflect}`, `end_padding: {None, Zero, Reflect}` | 160, Reflect, Reflect |
| preproc | `pre_emphasis: Option<f32>`, `remove_dc: bool` | None, false |
| compression | `log_base: {Ten, E}`, `log_floor: f32`, `range_clamp: Option<RangeClamp>`, `affine: Option<AffineCompress>` | Ten, 1e-10, `Some(PerCall { db: 8.0 })`, `Some(AffineCompress { bias: 4.0, div: 4.0 })` |
| impl | `spectrum_impl: SpectrumImpl` | `DftMatmul` |

Reuse `StftWindowConfig` (`ops/signal/stft_window.rs`) for `window` — it already covers
Hann/Hamming/Blackman/Ones with an explicit `periodic` flag and implements
`SamplingWindowBuilder` in both `Vec<f64>` and `Tensor` form. Do not introduce a second
`WindowKind`.

**Name the affine unambiguously.** Whisper's tail is `(log + 4.0) / 4.0`. A bare
`(f32, f32)` tuple invites a scale/bias mix-up; use a named struct with `bias` and `div`
fields (or `scale`/`offset` applied as `x·scale + offset` — pick one and say which in
rustdoc).

Derived, as methods not fields: `fft_len()` (`n_fft`, or `n_fft.next_power_of_two()` when
`pad_to_pow2`), `n_bins() = fft_len/2 + 1`, `f_max_hz()`.

Validation in `try_init`, returning `BunsenError::Invalid`, never panicking:
`hop > 0`, `hop <= n_fft`, `n_fft > 0`, `n_mels > 0`, `f_max <= sr/2`, `f_min < f_max`,
`sample_rate > 0`. Plus: `spectrum_impl == Stft` requires `fft_len().is_power_of_two()`
(otherwise burn panics deep inside `assert_valid`).
Record `min_first_chunk = n_fft/2 + 1` for the Reflect start check.

**Empty mel filter check**: after building the filterbank, any all-zero row is an error.
Test: `n_fft = 256, n_mels = 128, sr = 16000` must fail.

Tests: defaults construct; each validation error fires; `Config` round-trips through serde.

---

## Stage 2 — CPU reference filterbank (`Vec<f64>`, no Backend)

Module `mels/filterbank.rs`, pure functions:

- `hz_to_mel` / `mel_to_hz` for both scales, round-tripping to < 1e-9.
- `mel_points(n_mels, f_min, f_max, scale) -> Vec<f64>` (len `n_mels + 2`).
- `filterbank(opts) -> Vec<f64>`, row-major `[n_mels, n_bins]`.

**Anchors — verified, commit as literals** (the draft's Slaney 8 kHz guess was wrong):

| scale | f | mel |
|---|---|---|
| HTK `2595·log10(1 + f/700)` | 1000 Hz | `999.9855371396244` |
| HTK | 8000 Hz | `2840.023046708319` |
| Slaney (librosa) | 1000 Hz | `15.0` exactly |
| Slaney | 8000 Hz | `45.245640471924965` |

Slaney is librosa's piecewise form: `f_sp = 200/3`; linear `mel = f / f_sp` below
`min_log_hz = 1000` (so `min_log_mel = 15`); above,
`mel = 15 + ln(f/1000) / logstep` with `logstep = ln(6.4)/27 = 0.06875177742094912`.

Slaney area norm is librosa's `enorm = 2.0 / (mel_f[2..n_mels+2] - mel_f[..n_mels])`,
applied per row.

Golden test: `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)` as a fixture, max abs
diff < 1e-6. Same for HTK + no-norm vs
`torchaudio.functional.melscale_fbanks(..., mel_scale="htk", norm=None)` — **note torchaudio
returns `[n_bins, n_mels]`, transpose it.**

Also commit periodic Hann 400 from `torch.hann_window(400, periodic=True)` and cross-check it
against `StftWindowConfig::Hann { periodic: true }.to_vec_window(400)` — that also
retro-validates the existing window code against torch.

---

## Stage 3 — MelConverter holds tensors

```rust
#[derive(Module, Debug)]
pub struct MelConverter<B: Backend> {
    #[module(skip)] options: MelConverterOptions,
    window: Tensor<B, 1>,                    // `[n_fft]`
    mel_t:  Tensor<B, 2>,                    // `[n_bins, n_mels]`, stored transposed
    dft_cos: Tensor<B, 2>,                   // `[n_fft, n_bins]`
    dft_sin: Tensor<B, 2>,                   // `[n_fft, n_bins]`
}
```

**Two refinements from building it.**

*The DFT tables are not `Option`.* The draft made them optional against a future
`SpectrumImpl::Stft`, but with a one-variant enum they are statically always `Some` — an
`Option` that cannot be `None` is a lie in the type, and it costs an `unwrap` at every use.
Wrap them when a second variant actually lands.

*The tables have `n_fft` rows, not `fft_len`.* When `pad_to_pow2` widens the transform the
frame is conceptually zero-padded out to `fft_len` — and zeros contribute nothing to
`X[k] = Σ x[n]·e^(-2πikn/fft_len)`. So folding the wider angle into `n_fft` rows gives exactly
the padded transform with no padding materialized, and `t_stage_frame` never has to widen a
frame. The tables are `[n_fft, n_bins]` with angle `2πnk/fft_len`; only the bin axis grows.

Bare `Tensor` fields, no `Param` — see [Module policy](#module-policy) for why, and for the
`visit`/`map` caveat that must go in this type's rustdoc. `Option<T: Module>` is itself a
`Module` (`burn-core/src/module/param/primitive.rs:14`), so the `dft_*` fields are fine as
written.

The window is applied in `t_stage_frame`, so it is **not** folded into `dft_cos`/`dft_sin` —
that keeps the two `SpectrumImpl`s comparable on identical windowed frames.

**Build the DFT matrices in `f64` on the host, then cast.** `cos(2πnk/n_fft)` for
`n·k` up to 400·200 = 80000 loses meaningful precision if the angle is computed in f32.
Reduce `(n·k) mod n_fft` before the multiply, as `HostStft` in `sliding_stft.rs` already does.
Sign convention: `X[k] = Σ x[n]·e^(-2πikn/N)`, so `dft_sin[n,k] = -sin(2πnk/N)` — matching
the numpy/`rfft` convention the rest of `ops::signal` documents.

Memory at the default geometry: 400 × 201 × 4 B × 2 ≈ 643 KB. Fine.

Tests: shapes; `window` matches Stage 2; `mel_t` matches Stage 2 transposed; `clone` works;
**`to_device` actually moves every tensor field** — assert on `collect_devices()` rather than
eyeballing it, so an added-but-unmoved field fails.

**Validate the tables against `burn`'s `rfft`.** The sign and layout conventions are easy to
get backwards and nothing downstream will tell you. `rfft` only accepts a power-of-two `n_fft`,
so build a converter at `n_fft = 512` and assert `frames @ dft_cos == re` and
`frames @ dft_sin == im`; the default 400 geometry inherits the convention. This is Stage 5's
test #1 pulled forward to where the tables are actually built, and it is the cheapest possible
check that `dft_sin` carries the forward transform's negative sign.

---

## Stage 4 — `t_stage_frame`

`x.unfold::<3>(1, n_fft, hop)` → `[batch, frames, n_fft]`, then `mul` the broadcast window.

### ⚠️ `unfold` is broken on CubeCL backends in burn 0.21 (fixed in 0.22.0-dev)

**This is the single largest correctness hazard in the plan, and the default Whisper geometry
trips it.** There is a documented reproduction of it in this repo — `burner::repro::unfold`,
added by commit `671b494` — but it lives on branch `crutcher/tv-dev` and is **not** on `main`
or `crutcher/whisper_wip`. (The stale IDE run config
`Test burner::repro::unfold::tests::test_repro_on_performance_backend` is the only trace of it
on this branch.) Read that module before writing `t_stage_frame`.

**Upstream status: fixed.** The burn 0.22.0 dev branch is verified correct, so this is a
burn-0.21-only hazard with a known expiry. That changes the cost calculus below — the
mitigation is cheap enough to adopt anyway, but it is a bridge, not a permanent workaround.

**Confirmed in this repo, Stage 4.** With the covered-span slice removed,
`test_frame_matches_host_reference` fails on wgpu at `samples = 1000`, batch row 1
(`len` 1000, `v` 16, `1000 → 992`, so the row starts 8 samples early). With the slice it
passes. Two details the empirical rule in `burner::repro::unfold` does not capture:

* `samples = 520` (tail 120, the same `tail % 16 == 8`) **passed** — it yields a single
  window, and the truncated outer stride has nothing to stride over. The rule is a
  description of observed symptoms, so treat it as necessary-but-not-sufficient.
* Row 0 was correct in every failing case, exactly as documented. A `batch == 1` suite is
  blind to all of this.

The defect, in its own words: `unfold` **truncates its outer stride to the vectorization line
width**. When `size` and `step` share a factor of two the access vectorizes with line width
`v` = the largest power of two dividing both, and the outer stride becomes `(len/v)·v` instead
of `len`. Every row after the first is read `len % v` elements early.

* Affects all CubeCL backends — wgpu, cuda, metal, i.e. `PerformanceBackend`. `Flex` is correct.
* Wrong exactly when `size` and `step` are **both even** and `tail % v != 0`, for
  `tail = len - ((num - 1)·step + size)`.
* **Row 0 is always correct.** A batch-1 test passes; the corruption only appears once a
  second batch row exists. This is precisely how it would slip through the test plan above.

At the Whisper geometry `size = n_fft = 400`, `step = hop = 160` → `v = 16`, and the steady-state
`tail` is `carry_len - (n_fft - hop) = 360 - 240 = 120`, with `120 % 16 = 8`. So **every chunk
size trips it**:

| chunk | ext | num | tail | `tail % 16` | |
|---|---|---|---|---|---|
| 160 | 520 | 1 | 120 | 8 | ✗ |
| 1600 | 1960 | 10 | 120 | 8 | ✗ |
| 16000 | 16360 | 100 | 120 | 8 | ✗ |
| 480000 | 480360 | 3000 | 120 | 8 | ✗ |

**Mitigation — slice to the covered span before unfolding.** `t_stage_extend` already computes
`frames` and takes the carry from `ext[.., frames·hop ..]`. Have `t_stage_frame` unfold only
`ext[.., .. (frames - 1)·hop + n_fft]`. That makes `tail = 0`, hence `tail % v == 0` for any
`v`, on every geometry — the bug becomes unreachable rather than merely absent. It costs one
slice and is the natural framing anyway, since the tail belongs to the carry, not to any frame.

Do this from the first commit, with a comment naming the defect and the 0.22.0 fix. The slice
is worth keeping even after the bump — it makes "trailing samples belong to the carry, not to
a frame" explicit rather than incidental — but the comment should be deleted then.

**Prior art for the comment.** `SlidingStft::analyze` carries the same hazard and is now
documented at its padding site: there the uncovered tail works out to
`(samples - win_len) % hop_size`, so the two in-crate callers are safe by hop alignment while
a ragged `samples` is not. Match that comment's shape and its "delete me on the burn bump"
instruction.

⚠️ The `unfold` rustdoc says the window count is `max(0, (len - size).ceil_div(step))`, which
disagrees with the standard `1 + (len - size)/step`. burn's own `stft` relies on the latter
(it reshapes to `1 + (sig_len - n_fft)/hop_length`), so the doc comment is almost certainly
wrong — **but confirm it with the first test below before building on it.**

Tests:
- `len == n_fft` → 1 frame, equal to the input. *(This is the doc-vs-behaviour check.)*
- `len == n_fft + hop` → 2 frames, `frame[1] == x[hop .. hop+n_fft]`.
- Random hop-aligned `len`: every frame `t` equals `x[t·hop .. t·hop+n_fft]` against a CPU loop.
- Batch rows independent: `frame(cat(rows)) == cat(frame(rows))`.
- Windowing: unwindowed `unfold` × host window == stage output.
- **Every framing test must run at `batch >= 2` on `PerformanceBackend`, not just `CpuBackend`.**
  Row 0 is correct even when the bug fires, so a batch-1 or CPU-only suite is blind to it.
  Add a case with a deliberate non-zero tail (unfold the unsliced signal) that asserts the
  mitigation is actually load-bearing — it should fail if the slice is removed.

---

## Stage 5 — Spectrum → mel → compress (stateless stages 4–6)

Tests, in order:

1. **`DftMatmul` vs `stft`** on the same random frames at a power-of-two geometry
   (`n_fft = 512, hop = 128`): max rel diff < 1e-4. This is the only way to validate the
   `rfft` sign/layout assumption *in code*. It cannot run at the default geometry — that is
   the whole reason `DftMatmul` is the default.
2. Single Hann-windowed sine at bin centre `k·sr/n_fft`: power peaks at bin `k`, ≥ 20 dB above
   neighbours ±2.
3. Whole batch path vs golden: a fixed 1 s fixture vector, compared against `librosa` with
   `center=False` at every frame, tolerance 1e-4 in log-mel, with `range_clamp = None`.
   The Whisper-style clamp is tested separately.
4. `t_stage_compress`: floor behaves on all-zero input (no `-inf` / `NaN`); `PerCall` reduces
   over `[frames, n_mels]` per batch row, **not** across rows — test with a batch of 2 whose
   rows have deliberately different maxima.

---

## Stage 6 — MelConversionContext

```rust
pub struct MelConversionContext<B: Backend> {
    converter: MelConverter<B>,
    carry: Tensor<B, 2>,          // `[batch, carry_len]`
    phase: Phase,                 // Start | Running
}
```

No `running_max` — `RangeClamp::Running` is cut (see the decision gate).

Contract tests — the milestones:

- **Homomorphism.** For random hop-aligned splits `a ++ b ++ c` of a 2 s signal,
  `cat([transform(a).0, transform(b).0, transform(c).0], 1) == transform(a++b++c).0`.
  Same backend, same dtype. Aim for bitwise; if the `unfold`+matmul path reassociates,
  fall back to `Tolerance::permissive` and **say so in the test name**, because a tolerance
  here is a real (if small) result.
  Run this with `range_clamp = None` — `PerCall` is inherently not a homomorphism.
- **Chunk-size invariance.** chunk = `hop`, `10·hop`, `100·hop` give identical output.
- **Parity with librosa `center=True`.** Start = Reflect, End = Reflect, whole signal through
  chunks + `finish`, compared to `librosa.feature.melspectrogram(..., center=True)` → `log10`.
  Assert the frame count is **3001** for a 30 s input.
- **Parity with Whisper.** `whisper.audio.log_mel_spectrogram` on the same signal, with the
  `[:-1]` frame drop and `range_clamp = Fixed(max over the whole clip)`. Assert **3000** frames.
- Misaligned input errors (`n % hop != 0`); wrong batch errors; `Reflect` start with
  `n < n_fft/2 + 1` errors.
- `transform` after `finish` is impossible — `finish` consumes `self`, so the type system
  covers this. Add a `compile_fail` doctest to lock it in.
- Batch rows independent: `transform` on a batch of 2 == two batch-1 transforms stacked.
- **Per-stage streaming tests.** Drive `t_stage_extend` alone across three chunks and assert
  the carry contents and `carry_len` invariance directly, without going through the spectrum.
  This is the payoff for the stage stack — the state bug and the numerics bug can no longer
  hide behind each other.

---

## Stage 7 — Snapshot / clone semantics

- `MelConversionContext: Clone`; cloning then transforming both copies with the same input
  yields identical output (no hidden mutation).
- `batch_size()` accessor; document that `new_context` fixes the batch size.
- `reset(&mut self)` back to `Phase::Start` with a zeroed carry, mirroring
  `SlidingStftContext::reset`.

---

## Stage 8 — Kaldi-flavour coverage (optional, only if a target model needs it)

`pad_to_pow2 = true` (→ `fft_len = 512`, 257 bins), HTK, no norm, `log_base = E`,
`start_padding = None`, `pre_emphasis = 0.97`, `remove_dc = true`. Golden vs
`torchaudio.compliance.kaldi.fbank(..., dither=0.0, snip_edges=true)`. Expect a window
mismatch — Kaldi uses Povey (`hann^0.85`). Add `StftWindowConfig::Povey` **in
`stft_window.rs`, not here** — it belongs with the other windows.

Note that this configuration is power-of-two, so it is the natural place to also exercise
`SpectrumImpl::Stft` end to end.

---

## Stage 9 — Performance

- Bench `transform` at batch 1 and 8, chunk 1600 (100 ms), on `PerformanceBackend` and
  `CpuBackend`. Report µs/frame. Existing benches live in `crates/bunsen/benches/`.
- **Measure cold and warm separately.** cubecl autotune is shape-keyed and cold-vs-warm can
  differ by ~30×; a single-shot number is meaningless here.
- Fuse the two DFT matmuls into one `[n_fft, 2·n_bins]` matmul, then split — one kernel
  instead of two. Measure before adopting; it doubles the matrix's live memory.
- Confirm no per-call host↔device sync — no `into_data` in the hot path. The only allowed
  sync is in tests. In particular `PerCall` range clamp must stay a tensor reduce.
- Check that `re² + im²`, matmul, log and clamp fuse into few kernels (cubecl fusion logging
  via `burn.toml`).

---

## Fixtures

Generation script in `tools/gen_mel_fixtures.py` (the directory does not exist yet), Python
with pinned versions in a header comment. Run once, commit the outputs to
`crates/bunsen/testdata/mels/`. Flat little-endian `f32`.

- `hann_400_periodic.f32`
- `mel_fb_slaney_16k_400_80.f32` (librosa)
- `mel_fb_htk_16k_400_80_nonorm.f32` (torchaudio, transposed)
- `signal_2s_16k.f32` — deterministic: seeded noise + two chirps + 200 ms silence
- `logmel_center_false.f32`, `logmel_center_true.f32` (librosa, power, log10 floor 1e-10)
- `whisper_logmel.f32` — `whisper.audio.log_mel_spectrogram` on the same signal, padded to
  30 s then sliced to the signal's frame count

Size check: the two 80×201 filterbanks are 64 KB each; 2 s of audio is 128 KB; each 2 s
log-mel is ~64 KB. Total well under 500 KB — no LFS needed (the repo has no `.gitattributes`).

Loader: a `fixture(name) -> Vec<f32>` helper reading LE f32 with a length assertion, next to
the tests.

---

## Order of work / checkpoints

0. **`SlidingStft` → `Module`** (see [Module policy](#module-policy)). Standalone commit,
   lands first so the mel types have a settled pattern to copy.
1. ~~Stage 0 probes~~ — **done**, see [Findings](#findings-stage-0-resolved).
2. Stages 1–2 (no Backend). Fast, pure, and most of the numerical risk lives here.
3. Stage 3 + Stage 4 + Stage 5.1. First tensor code; **Stage 4's first test settles the
   `unfold` frame-count doc discrepancy** before anything is built on it.
4. Stage 5.3 golden batch parity — first milestone: "batch log-mel matches librosa".
5. Stage 6 — the context and the stage stack; the per-stage carry test and the homomorphism
   test are the second milestone.
6. Stage 6 parity with `center=True` / Whisper, with the 3001 / 3000 frame counts asserted —
   third milestone.
7. Stages 7–9 as time allows.

## Open decisions, settled

| Question | Answer |
|---|---|
| `SpectrumImpl` default | `DftMatmul` — forced; `rfft`/`stft` are power-of-two only and `n_fft = 400` is not |
| `hann_window` convention in Burn | `hann_window(size, periodic, opts)`, explicit flag. Use `StftWindowConfig::Hann { periodic: true }` instead — already in-repo |
| `rfft` output layout | `(re, im)` as two separate tensors, `n/2 + 1` along `dim` |
| `unfold` availability | Yes — `Tensor::unfold(dim, size, step)`. Frame count needs one confirming test |
| Is `RangeClamp::Running` needed | No — cut from v1. `PerCall` + `Fixed` only |
| Is `unfold` safe to frame with | **Only after slicing to the covered span.** It corrupts rows ≥ 1 on CubeCL backends at the default geometry — see [Stage 4](#stage-4--t_stage_frame) |

## Still open

- **`SpectrumImpl::Stft` as a stage.** It fuses framing + windowing + FFT, so it does not slot
  into `t_stage_frame` → `t_stage_spectrum`. Either dispatch above the stage stack (simple,
  slightly ugly) or give it a fused `t_stage_frame_spectrum` and make the stack a small enum
  of pipelines (cleaner, more machinery). Decide when Stage 8 actually needs it — until then,
  ship only `DftMatmul` and leave `SpectrumImpl` a one-variant enum.
- **Whisper's `[:-1]`.** An option on `MelConverterOptions`, or a caller-side concern? Leaning
  caller-side — it is a Whisper packaging detail, not a mel-spectrogram parameter.
