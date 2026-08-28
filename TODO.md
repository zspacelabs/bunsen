# Normalization burn-down

Scope: the full diff `main...crutcher/whisper_wip` (23 commits, 44 files, +7098/-57).

This document is a **survey with options**. Nothing here has been applied. Each
item states what exists, why it is a normalization problem, and 2–3 candidate
repairs. We step through them together and pick per item.

Goal: documentation that is **layer-ignorant, local, and timeless**, and an API
surface that lives at the right layer with no duplicate mechanisms.

Legend: `[N-n]` narrative, `[L-n]` layer violation, `[U-n]` new utility.

---

## Conventions settled while stepping through

Decisions made during burn-down that apply to later items, recorded so they
don't get relitigated.

**C-1. Test helpers are tiered by domain, not centralized.** `support::testing`
holds domain-agnostic primitives; `<domain>::testing` holds domain assertions
built on them, gated `#[cfg(any(test, feature = "testing"))]`. The convention
predates this branch — `ops::signal::testing` was already reachable and in use
by `cosine_window.rs`.

**C-2. Assertions unify; fixtures stay local.** An assertion's contract ("these
agree within tolerance") doesn't drift with test intent, so sharing one couples
nothing. A fixture encodes *what a particular test exercises*, and that does
drift — sharing one means the next test that needs a different distribution
either forks it (duplication behind a name that now lies) or edits it and
silently perturbs unrelated tests. Duplicated fixture code is therefore not a
defect to fix. Applies to `[U-3]`, `[U-7]`.

**C-3. A shared namespace requires a self-describing name.** A name may rely on
its file for context while it lives there; moving it to a shared module strips
that context, so the name must acquire its subject. Renames follow moves, not
the other way round.

**C-4. The store-adapter repair is deferred, not rejected.** Fixing the
strided-weight defect at the store layer — a `ModuleAdapter` that repairs
tensors matching a `PathFilter` as they are read — is expressible today:
`PathFilter` (regex / full-path / `fn(&str, &str) -> bool` over path and
container type) and `TensorSnapshot::from_closure` are both public, and the
Whisper load path already configures the store with six `with_key_remapping`
regexes. Three things block it:

1. `PytorchStore::apply_to` hardcodes `PyTorchToBurnAdapter` (burn-store 0.21,
   `store.rs:364`); there is no `with_adapter`. A custom adapter means
   abandoning `module.load_from(&mut store)` and hand-rolling
   `get_all_snapshots` → `module.apply(..)`, re-implementing the store's
   `validate` and error aggregation. That forks the load path around a missing
   upstream builder method.
2. An over-matching pattern fails **silently**. For a square weight the repair
   degenerates to a plain transpose, and Whisper's attention projections are
   `d_model x d_model`. A pattern that matches one weight too many yields a
   model that runs and is wrong — the exact failure mode that motivated this
   work. An explicit assignment is auditable by reading it; a regex's match set
   is not.
3. The repair is neither idempotent nor self-checking: applied twice to a
   non-square weight it scrambles, applied once to a correct square weight it
   transposes, and nothing in the data distinguishes the cases.

Revisit if `PytorchStore` gains `with_adapter` upstream, and only with
something that makes over-matching loud rather than silent.

**C-5. In a repro or cross-check, a named real-world instance is evidence,
not coupling.** The usual rule — a generic layer must not name a specific
caller — does not apply to an artifact whose job is to demonstrate a defect.
"96 of the 245 tensors in `base.en.pt`, exactly these names, strides `(1, N)`"
is what makes a bug report reproducible, and no code depends on it. What *is*
a violation in such a file is a pointer back into a caller
("...and its callers in the Whisper kit"), which is a real reverse dependency
and stops meaning anything once the file is lifted out. Keep the evidence;
cut the back-references. Applies to `[N-5]`.

---

## 1. Narrative discussion

Text that documents *the development arc* rather than *the thing being
documented*. It reads as stale the moment the arc ends.

### [N-1] `MEL_CONVERTER_PLAN.md` (repo root, 647 lines)

A staged implementation plan ("Stage 0 … Stage 9", "Still open", "Open
decisions, settled"). It was the working document for this branch and is now
committed at the repo root, where it reads as project-level documentation.

Three source files cite it as if it were a durable reference:
- `crates/bunsen/src/ops/signal/mels/converter.rs:99` — "See `MEL_CONVERTER_PLAN.md`."
- `crates/bunsen/src/ops/signal/mels/filterbank.rs:291` — "…the note in `MEL_CONVERTER_PLAN.md`."
- `crates/bunsen/src/ops/signal/mels/context.rs:566` — "The frame accounting worked out in `MEL_CONVERTER_PLAN.md`."

Options:
1. **Delete it, migrate the durable content.** The parts worth keeping are the
   frame-accounting derivation and the filterbank-normalization rationale;
   both belong as rustdoc on the code that implements them (`frame_count`,
   `FilterNorm::gain`). The three citations then resolve locally.
2. **Demote to a dev note.** Move to `docs/dev/` or `dev_crates/`, retitle as a
   historical design record, and strip the three source citations so no
   library rustdoc depends on a moving document.
3. **Keep as-is** and accept a root-level plan file plus three rustdoc
   references to it. (Listed for completeness; it is the status quo.)

### [N-2] "Stage N:" doc headings on the transform pipeline

`crates/bunsen/src/ops/signal/mels/context.rs`:263, :352, :368, :380, :392, :404

Every `t_stage_*` method opens with `/// Stage 1:` … `/// Stage 6:`. The
numbers describe the plan's ordering, not anything a caller can observe — and
they are a maintenance trap: inserting a stage renumbers five doc comments.

Options:
1. **Drop the ordinals, keep the prose.** `/// Prepends the start padding or
   the carry…`. The ordering is already expressed by `transform`'s body.
2. **Keep ordinals but source them from the code** — document the stage list
   once on `MelConversionContext` (or on `transform`) as the pipeline
   contract, and have each method document only its own transform.
3. **Rename to phase names** — `t_stage_extend` already carries the name;
   let the doc lead with the invariant (`in: [B, S]`, `out: [B, S']`) instead.

### [N-3] Time-bound "delete this on the burn bump" comments

- `crates/bunsen/src/ops/signal/sliding_stft.rs:338-339` — "Fixed upstream:
  burn 0.22.0-dev is correct. On the next burn bump, delete this comment and
  the alignment note in the rustdoc."
- `crates/bunsen/src/ops/signal/mels/converter.rs:752-753` — "Fixed upstream in
  burn 0.22.0-dev. Keep the slice on the bump (it states the contract), but
  delete this comment."

These are instructions to a future maintainer keyed to a version that does not
exist yet. They rot silently: nothing fails if the bump happens and the comment
stays.

Options:
1. **Restate as a timeless invariant.** Say what the code guarantees ("the
   slice pins the output length to `n_fft/2 + 1`") and drop the upstream
   history entirely.
2. **Make it enforceable.** Keep a one-line note but attach it to something
   that breaks on the bump — a `#[cfg]`-gated compile check, or a test named
   for the behaviour that will change.
3. **Move the history to the changelog** and leave only the invariant in the
   source.

### [N-4] Defect-history prose in `burner/repro/` — RESOLVED

Handled with `[L-2]`, same file, one commit.

The square-case note read "Documented here because it is the reason this went
unnoticed" — a record of how the defect escaped notice. It now states the
constraint that still applies: the repair is a silent transpose on a weight
that did not need it, so it cannot be applied blindly. Same length, but it
earns its place, and it is what `[C-4]` rests on.

**Deliberately not touched:** the `## Why this is easy to miss` section. It
reads as narrative but is the square-degeneracy explanation — the most
load-bearing paragraph in the module, and the technical basis for `[C-4]`.
Do not "clean it up" in a later pass.

### [N-5] Discovery narrative in `dev_crates/whisper-onnx-crosscheck/`

- `src/lib.rs:31-33` — "by-inspection transliteration", "turned up four defects"
- `src/lib.rs:251` — "the three this crate was built to catch"
- `README.md:109-115` — the defect list as a found-during-development story

`:251` is the sharpest case: it hard-codes a count that this branch's own
history produced, and it is already wrong-ish the moment a fifth defect lands.

Options:
1. **Reframe as coverage, not history.** "This test asserts encoder parity to
   `1e-3`; it fails on weight-layout, positional-embedding, and
   normalization-placement regressions." That survives new defects.
2. **Move the found-defect list to the branch's PR description / changelog**
   and leave the crate documenting only what it checks today.
3. **Keep the list but date it** — an explicit "as of <date>, this caught: …"
   is honest, if less durable than option 1.

### [N-6] Dev-narrative asymmetry to confirm

`crates/bunsen/src/ops/signal/mels/context.rs:825` and `:168` justify test and
API choices with "since Whisper…". These are narrative *and* layer-violating;
see `[L-4]`. Listed here only so the two lists reconcile.

---

## 2. Layer-violating discussion and APIs

Generic layers (`blocks/`, `burner/`, `ops/`) that know about a specific
caller (`kits/speech/whisper`). Ranked most to least severe.

### [L-1] `MlpConfig::repair_strided_weights` — RESOLVED

A generic transformer block carried a boolean about a serialization-library
bug in a specific checkpoint format. Removed; the Whisper kit now applies
`repair_pytorch_strided_weight` to its MLP projections directly, exactly as it
already did for its attention projections three lines above.

What decided it: **the kit was already doing this correctly for MHA**
(`encoder_block.rs`, `decoder_block.rs`), and `Mlp` exposes `pub linear1` /
`pub linear2`, so nothing blocked symmetry. The same files already do post-init
surgery for a second Whisper-specific concern (`attn.key.bias = None`). Both
blocks now carry one repair block covering all six weights under one comment.

Also in the same pass: the `bias` field doc lost its "Whisper's MLP projections
do; a GPT-style block generally does not" and became a plain statement of what
the flag does. The field itself is legitimate architecture config.

The store-wrapper alternative (option 1) was **investigated and deliberately
deferred** — see `[C-4]`.

### [L-2] Whisper named from inside `burner/` — RESOLVED

Fixed across two commits.

`param_mappers.rs` was handled during the `[L-1]` relocation: its rustdoc now
describes the storage pattern ("a checkpoint holds a view like this wherever
the saved tensor was produced by transposing another rather than by copying
it") instead of naming a model.

In `burner/repro/pytorch_strided_weights.rs`:
- the behaviour-pin's "and its callers in the Whisper kit" is **gone**. It was
  a reverse `burner/` -> `kits/` reference, and redundant: the assert message
  already names what to remove, which is where a developer actually meets the
  instruction. The doc now defers to it.
- the `## Scope` measurement **stays**, per `[C-5]`.

Also corrected `burner/repro/mod.rs`, which claimed each harness "use[s] only
public API" and can be "lifted out and taken upstream unchanged". Neither is
true of any module there — this one imports both the crate's workaround and
its test helpers.

### [L-3] Whisper rationale inside `multihead_utils` — RESOLVED

Two sites, treated differently because they were doing different jobs.

`layer_norm_cross_attn`'s rustdoc lost "Whisper's decoder attends 4 tokens over
1500 audio frames" outright. It illustrated a sentence that had already made
the point (`cross_len` is independent of `seq_len`), so there was nothing to
generalize — the example was pure redundancy plus coupling.

The regression test's doc kept its rationale, restated: "a contract that bound
both to one `seq_len` would pass every same-length unit test and still be
wrong, which is why this one uses deliberately mismatched lengths." Same move
as `[N-4]`'s — keep the constraint, drop the incident. It now explains the
`seq_len = 3, cross_len = 17` choice to a reader who never saw the bug.

Checked while here: `layer_norm_cross_attn_kv` in `kv_attention.rs` carries no
such prose, so there was no third copy. `blocks/` is now free of model names
apart from `[L-6]`.

### [L-4] `# Producing Whisper input` recipe in `mels/context.rs` — RESOLVED

The decisive finding: **the recipe already existed as working code.**
`to_whisper_mels` in `examples/whisper-dev/src/main.rs` implements it exactly —
same slice, same `db: 8.0`, same `swap_dims` — and its own doc already pointed
back to `MelConversionContext` for the caveat. `mels/cross_test.rs` implements
it a third time as a test. The Whisper *kit* never used it. So the doc block
was prose duplicating code, not a recipe needing a home, and the fix was to
delete rather than relocate.

What the section actually contained was two separable halves:
- the **caveat** — `RangeClamp::PerCall` reduces over one call's frames, so
  chunking is not transparent while it is on — which is a real property of the
  type, already pinned by `test_per_call_clamp_is_not_chunk_invariant`. Kept,
  restated model-free as `# Chunking and per-call reductions`.
- the **recipe** — code block, `[..., :-1]` slice, `db: 8.0`, encoder axis
  order. Deleted. 26 lines became 6.

Also in the file: "Whisper's geometry" was just `MelConverterOptions::default()`
(`n_fft = 400, hop = 160, sr = 16000`), so the worked frame-count example keeps
its numbers as "the default geometry"; the chunk-invariance test now explains
its silent tail by what makes silence realistic rather than by naming a model;
and `test_whisper_frame_accounting` became
`test_frame_accounting_over_a_30s_window`.

`context.rs` now contains no Whisper references at all. Its one remaining
cross-reference is to `MEL_CONVERTER_PLAN.md`, which belongs to `[N-1]`.

**Noted, not acted on:** the recipe lives only in an example, so any real
consumer of bunsen's Whisper must re-derive it. Whether `to_whisper_mels`
should be promoted into `kits::speech::whisper` is a feature question, adjacent
to `[U-4]`, not a normalization one.

### [L-5] Whisper-shaped defaults and prose throughout `mels/`

`crates/bunsen/src/ops/signal/mels/converter.rs` (:7, :71, :111, :136, :163,
:190, :221, :745, :915) and `filterbank.rs:39` ("and by Whisper").

Softer than L-1..L-4: these are mostly *examples* and *default values* that
happen to match Whisper. The defaults themselves are defensible — 16 kHz /
400 / 160 / 80 is a common speech configuration, not a Whisper invention.

Options:
1. **Attribute to the convention, not the consumer.** Cite Slaney/HTK, librosa,
   or torchaudio where the behaviour originates; those are the actual sources
   of truth and they are stable references.
2. **Leave the defaults, launder the prose.** No API change; just rewrite the
   nine doc sites so `mels` documents mel spectrograms rather than Whisper
   input.
3. **Audit defaults for neutrality.** Confirm each default is defensible on
   its own terms; where one is only defensible as "what Whisper does", move it
   to a kit-side preset (see L-4 option 3).

### [L-6] Confirmed non-issue (do not "fix")

`crates/bunsen/src/blocks/conv/conv_seq_1d.rs:198` mentions Whisper but is
**pre-existing on `main`** and outside this diff. It may still be worth a
separate cleanup, but it is not part of this branch's burn-down and must not
be attributed to it.

---

## 3. New utility methods

New surface area added by this branch. For each: does it belong at this layer,
and does bunsen or burn already have it?

### [U-1] Duplicated test helpers — RESOLVED

Three helpers were defined identically in `mels/context.rs` and
`mels/converter.rs`. Six definitions collapsed to two.

| helper | disposition |
| --- | --- |
| `to_f64` | **deleted** at both sites — `TensorDataToVecAsExt::to_vec_as` already covered it |
| `assert_matches_host` | **promoted** → `support::testing::assert_tensor_close_to_vec` |
| `assert_tensors_close` | **promoted** → `support::testing::assert_tensors_close` |
| `sample` | **left duplicated**, per `[C-2]` |

Promoted helpers are generic over `B: Backend` with `Tolerance<B::FloatElem>`
(bound: `B::FloatElem: num_traits::Float`, since
`TensorData::assert_approx_eq` requires `F: Float + Element`).

Also moved, in the same pass: `assert_builder_impls_match` out of
`window_builder.rs` — a 72-line file with no tests of its own that defined a
helper only `cosine_window.rs` called — into `ops/signal/testing/mod.rs` as
`assert_sampling_window_builder_implementation` (`[C-3]`). Its public path is
otherwise unchanged. `support/mod.rs` gained the `any(test, ...)` gate to match.

Findings that did **not** become work:
- `from_rows` is not duplicated (single definition, `context.rs`); it was in
  the original list by association.
- `sliding_stft.rs`'s `sample` is a different function that shares a name;
  since the mels `sample` stays private, there is no collision to resolve.
- `assert_close_to_vec` has 34 call sites, 18 of them pre-existing on `main`.
  Changing its scalar-tolerance signature is a main-side refactor and stays out
  of scope. Tolerance policy is documented rather than enforced:
  `Tolerance<F>`/`assert_approx_eq` for tensor-valued comparisons,
  `assert_close_to_vec` for host `Vec` where absolute tolerance is honest.

Deferred, and deliberately **not** to be fixed locally: there is no
`Tensor -> Vec<E>` extension method, only `TensorData::to_vec_as`, so the 13
former `to_f64` call sites now read `t.to_data().to_vec_as::<f64>().unwrap()`.

Do not add a `to_vec_as` to `burner::tensor`'s ext traits to shorten them.
Those traits were upstreamed into burn's dev branch with further elaboration,
and bunsen's local copies have not yet been realigned with the upstream shape.
Adding a bunsen-side method now grows divergence that the eventual back-port
has to unwind. Revisit call-site ergonomics after that alignment lands; it is
an orthogonal track from this burn-down.

### [U-2] `AttnKv` / `project_kv` vs the existing `KVCache`

New: `blocks/transformers/attention/kv_attention.rs` — `AttnKv` (`seq_len`,
`batch_size`, `concat`), `project_kv`, `layer_norm_self_attn_kv`,
`layer_norm_cross_attn_kv`.

Existing: `blocks/transformers/attention/kvcache.rs` — `KVCacheConfig`,
`KVCache` (`prefill`, `insert_kv`, `reset`, `pos`, `allocation_size`), a
pre-allocated design. Plus burn's own `MhaCache` / `MultiHeadAttention::forward_cache`.

Three K/V caching mechanisms now live in one module. `AttnKv` grows by
`concat`; `KVCache` writes into a preallocated buffer. Both are reasonable;
having both undocumented-as-alternatives is not.

Options:
1. **Build `AttnKv` on `KVCache`.** Keep the ergonomic
   `layer_norm_*_kv` entry points but back them with the existing allocation
   strategy, so there is one storage mechanism. Requires checking whether
   `KVCache`'s fixed allocation fits the chunked-decode access pattern.
2. **Keep both, document the split.** Module-level doc stating: `KVCache` for
   bounded-length decode with a known cap; `AttnKv` for growing prefixes where
   the cap is not known ahead. Cheapest, and defensible if both are real.
3. **Retire one.** If `KVCache` has no other caller on `main`, or if `AttnKv`
   is strictly more general, delete the loser rather than carry two.

Sub-question for the same item: `project_kv` and the two `*_kv` helpers
duplicate structure from `multihead_utils`'s non-`_kv` counterparts. Check
whether the cached and uncached paths can share a body.

### [U-3] `MelConverterOptions::to_vec_*` — three host-side extractors

`to_vec_dft_tables`, `to_vec_filterbank`, `to_vec_filterbank_t` on
`MelConverterOptions`.

These build host `Vec<f64>` from config and exist mainly so tests and `init`
can share table construction. `to_vec_dft_tables` in particular is a generic
DFT-table builder wearing a mel-converter method signature.

Options:
1. **Move `to_vec_dft_tables` to `ops::signal`** as a free function; a
   cos/sin DFT table has nothing to do with mel scaling and `sliding_stft`
   may want it too.
2. **Make them `pub(crate)` / `#[doc(hidden)]`.** If the only callers are
   `init` and tests, they need not be public API we support forever.
3. **Keep public, document the contract.** They are genuinely useful for
   cross-checking against librosa/torch; if we keep them public, document the
   layout (row-major, `[n_mels, n_freqs]` vs transposed) precisely.

### [U-4] `mel_windows` in the Whisper kit

`kits/speech/whisper/decode.rs:39` — slices a mel tensor into fixed windows.

Windowing a `[B, C, T]` tensor along `T` is not Whisper-specific.

Options:
1. **Generalize into `ops::signal`** as a chunking helper, leaving the
   Whisper-specific window length in the kit.
2. **Check for an existing equivalent** — burn's `chunk`/`narrow` or
   bunsen's existing framing code in `sliding_stft` may already cover it,
   in which case delete.
3. **Keep in the kit.** Defensible if the padding/partial-window policy is
   genuinely Whisper's; document that policy as the reason it lives there.

### [U-5] `StreamPhase`, `SpectrumImpl` and speculative variants

- `SpectrumImpl` (`converter.rs:101`) has exactly one variant, `DftMatmul`.
- `MelConverterOptions::validate` rejects `pre_emphasis` (`:586`) and
  `remove_dc` (`:591`) as "not implemented yet" — config surface that exists
  only to be refused.
- No enum variant of `PaddingMode` / `SpectrumKind` / `LogBase` / `RangeClamp`
  is constructed anywhere outside `converter.rs` and `context.rs` today.

Options:
1. **Cut the unimplemented options.** Delete `pre_emphasis` and `remove_dc`
   until there is a caller; a config field that always errors is worse than an
   absent one. Reclaims the `t_stage_preproc` no-op stage too.
2. **Keep the enums, cut the single-variant one.** `SpectrumImpl` with one
   variant is a placeholder for a future `rfft` path; it costs a public type
   and a match arm. Decide whether that future is close enough to reserve for.
3. **Keep everything, document intent.** If these are deliberate extension
   points, say so — but then `[N-2]`'s "Stage 2: sample-domain preprocessing"
   doc needs to stop describing a stage that does nothing.

### [U-6] `t_stage_*` visibility

`context.rs`:272/360/371/383/395 — five `#[doc(hidden)] pub(crate)` methods
exposed for the testing layer, per the original design intent.

Options:
1. **Keep as-is.** `pub(crate)` + `doc(hidden)` is the conventional way to
   expose seams to in-crate tests; it is already correct.
2. **`#[cfg(test)]`-gate them** if no non-test in-crate caller exists besides
   `transform`, shrinking the always-compiled surface.
3. **Promote to a documented pipeline trait** if we want out-of-crate users to
   compose stages — a larger commitment, listed only to be explicit.

### [U-7] Cross-check crate helpers

`dev_crates/whisper-onnx-crosscheck/` defines `read_f32`, `read_i64`,
`summarize`. These duplicate the shape of `[U-1]`'s helpers across a crate
boundary.

Options:
1. **Leave duplicated.** A dev crate that deliberately does not depend on
   bunsen's test support keeps the cross-check honest — it should not share
   code with what it validates. (Probably correct.)
2. **Share via `support::testing`** if the crate already depends on bunsen
   anyway, and the helpers are pure data plumbing with no assertion semantics.
3. **Narrow them** to the two or three call sites and inline.

---

## Suggested order

1. ~~`[U-1]` test-helper consolidation~~ — **done**.
2. ~~`[L-1]` `repair_strided_weights`~~ — **done**.
3. ~~`[L-2]`~~ (with `[N-4]`), ~~`[L-3]`~~, ~~`[L-4]`~~ — **done**.
4. `[N-1]` plan-file disposition — unblocks `[N-2]` and the three citations.
5. `[N-3]`, `[N-5]` timeless-rewrite passes.
6. `[U-2]` KV-cache convergence — needs a design call, not a cleanup.
7. `[U-3]`–`[U-7]`, `[L-5]` — judgment calls, low urgency.

---

## Last-call removals

Artifacts that exist only to carry this normalization pass and must not
outlive it. Check these off before the branch lands — an unremoved entry here
becomes exactly the kind of committed development narrative `[N-1]` is about.

- [ ] **`TODO.md`** (this file). It is a burn-down list, not documentation:
      once every item above is resolved or explicitly deferred somewhere
      durable, delete it. Anything still worth keeping at that point belongs in
      rustdoc next to the code, in `CHANGELOG.md`, or in an issue — not here.
      In particular, the **Conventions settled** section is the part most
      likely to deserve a real home; decide where it goes rather than keeping
      this file alive to hold it.
- [ ] **`MEL_CONVERTER_PLAN.md`** — conditional on `[N-1]`. Listed so the
      decision is not lost, not because it is settled; `[N-1]` option 2 keeps
      it as a relocated dev note instead.

Deferred removals with an *external* trigger belong to `[N-3]`, not here —
they are keyed to a burn version bump rather than to this branch finishing.
