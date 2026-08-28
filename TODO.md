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
defect to fix.

A related limit: a cross-check or repro should not share *test helpers* with
the code it validates, even when sharing would be convenient — the point of an
independent check is that it fails independently. This constrains helpers, not
dependencies; such a crate may of course depend on what it tests.

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

### [N-1] `MEL_CONVERTER_PLAN.md` — RESOLVED (kept, relocated)

The plan is **not** narrative to be deleted: more mel-filter work is planned,
and Stage 8, Stage 9, and the questions under "Still open" are live. Kept, and
co-hosted with the module it documents as
`crates/bunsen/src/ops/signal/mels/PLAN.md` — out of the repo root, where it
sat beside README/CONTRIBUTING/STYLE and read as project documentation. The
path supplies the subject, so the filename no longer repeats it.

It is deliberately **not** wired into rustdoc with `include_str!`: publishing a
working design document as API documentation is the problem this disposition
was about.

Its header now states what is built and what remains, so a reader can tell
settled decisions from pending tasks, and no longer points at a draft under a
`~/Downloads` path that exists on one machine.

Correction to this entry as first written: the file is **737 lines**, not 647,
and the "migrate the durable content" work was already done. All three rustdoc
citations turned out vestigial:
- `converter.rs` already gave the whole reason `SpectrumImpl` has one variant.
- the frame accounting already lives in `transform`'s `# Frame count` section,
  so the test points there now.
- `filterbank.rs` already said its constants come from a transcription of
  `librosa.filters.mel`.

**Stale fact fixed while here.** The filterbank constants' doc promised that
"a fixture generated by real `librosa` is still owed and would supersede
them". It landed: `cross_test::test_filterbank_matches_librosa` compares the
whole bank against `testdata/mels/mel_fb_slaney_16k_400_80.f32`. The constants
are kept — they need no fixture file and fail with a different signature,
catching a bank that is transposed, shifted by a bin, or normalized along the
wrong axis even when its element values are right — and their doc now says so.

No source file cites the plan any more.

### [N-2] "Stage N:" doc headings on the transform pipeline — RESOLVED

**Pipeline stages are named, not numbered.** The ordinals carried nothing the
method name and shape transition did not already say, and the ordering they
described is expressed by `transform`'s body — the only place it cannot drift.

The finding that settled it was stronger than the maintenance-trap argument
this entry was opened with: the numbers were already **ambiguous, and in one
case contradictory**. `PLAN.md` used "Stage N" for two different things — the
build sequence (`## Stage 4 — t_stage_frame`) and the runtime pipeline (where
`t_stage_frame` was stage 3). Same function, two live numbers, in files that
now sit in the same directory. One plan heading mixed both in a single line:
"Stage 5 — Spectrum -> mel -> compress (stateless stages 4-6)".

Applied to both files, since unnumbering only the code would have preserved the
mismatch. In `PLAN.md` the stage table lost its ordinal column, the six prose
headers lost their numbers, and numbered cross-references became named ones
("bypasses `t_stage_frame` and `t_stage_spectrum`").

**Build stages keep their numbers.** A work plan is a sequence and its steps
have no other name; `## Stage 8`, `## Stage 9`, and the "Order of work" list
are unaffected. All surviving `Stage N` references in `PLAN.md` were checked
and are build stages.

**Left for `[U-5]`:** `t_stage_preproc`'s doc ("currently the identity ... where
pre-emphasis and DC removal will land") is accurate and honestly flagged, not
narrative. It is the same dead-option surface as `validate` rejecting
`pre_emphasis` and `remove_dc`, and belongs to that decision.

### [N-3] Time-bound "delete this on the burn bump" comments — CLOSED

Closed by fixing the code rather than the prose. The notes themselves are
**kept as they are**, by decision.

The two sites were not alike. `MelConverter::frame` already defended itself —
it trims to the covered span, so the hazard is unreachable there. `analyze`
did **not**: it relied on callers keeping `samples` hop-aligned and carried a
public rustdoc warning saying a ragged `samples` was wrong for `batch > 1`.

`analyze` now applies the same trim. It is semantically free — trailing
samples that do not fill a whole window are ignored either way — so no output
changes, and the caller-facing caveat is gone.

`analyze` also had **no direct tests**; it was reached only through `forward`
and `forward_sequence`, both hop-aligned by construction, which is precisely
why the ragged path went unexercised. It now has frame-count/shape coverage,
agreement with a naive per-frame DFT, and batched-vs-per-row agreement.

**Unresolved, recorded so it is not lost.** The defect these notes describe did
not reproduce. Disabling the new trim did not make the ragged-batch test fail,
and a direct `unfold` probe at both documented geometries — `(1024, 256)` with
tail 8, and the mel `(400, 160)` with tail 40 — showed zero difference between
batched and single-row results on wgpu, at burn 0.21.0 / cubecl 0.10.0.

That is **not** evidence the defect is gone: the probe unfolds a freshly built
contiguous tensor, while both real call sites unfold the result of a prior pad
or slice, so provenance and fusion context differ. The notes' claims are left
standing. Consequently the ragged-batch test is documented as a
row-independence test, not as a pin on that defect — there is still no test
that fails when burn fixes it, which was the original `[N-3]` concern and
remains open in principle.

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

### [N-5] Discovery narrative in the cross-check crate — RESOLVED

`[C-5]` settled most of this before it was opened: this is a cross-check
crate, so naming Whisper and citing the defects it found is its subject, not
coupling. The README's four-bullet defect list **stays** — it is what tells a
reader which class of bug the crate catches, and its concrete provenance is
what makes it credible.

What was actually fragile was narrower: the **counts**. Three deleted.
- `README` "turned up four defects" merely restated the bullet list beneath it.
- `lib.rs` "the three this crate was built to catch" pinned a number nothing
  updates — inside a comment whose real job is defending a deliberately loose
  tolerance, which is kept intact.
- `lib.rs` "wrong in three separate ways" likewise.

They also read as a discrepancy that was never explained: `lib.rs`'s three
counted encoder defects, the README's four included the decoder-side
cross-attention contract. Removing them removes the question.

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

### [L-5] Whisper-shaped defaults and prose in `mels/` — RESOLVED

Almost entirely a false positive. One doc rewrite; everything else stands, for
reasons worth recording so it is not re-opened.

**The enum docs are an index of frontends, not coupling.** Every variant names
the real implementations that use it — `MelScale::Htk`, `MelScale::Slaney`
("`librosa` (`htk=False`) and ... Whisper"), `LogBase::Natural` ("Kaldi-flavoured
frontends"), `SpectrumKind::Power`. Whisper appears exactly as Kaldi and HTK do.
Stripping only the Whisper names would break a consistent pattern that tells a
caller which convention to pick.

**`RangeClamp` and `AffineCompress` cite Whisper as provenance.**
`maximum(log_spec, log_spec.max() - 8.0)` and `(log_spec + 4.0) / 4.0` are
Whisper's, not librosa's, and they are why those options exist. Remove the
attribution and the constants become unexplained. `[C-5]` covers this.

**`cross_test.rs`'s eight mentions are exempt outright** under `[C-5]`: it is a
parity test against `whisper.audio.log_mel_spectrogram` using a
`whisper_logmel.f32` fixture. Whisper is its subject.

**The module-level "defaults reproduce Whisper / librosa" is a checkable
claim**, pinned by `test_defaults_are_whisper` and by `cross_test`.

**Fixed:** `MelConverter::forward`'s rustdoc explained its return shape by
naming Whisper's audio encoder. Reordered so the real reason leads — frames sit
on the middle axis because that is the streaming concat axis — and the
transpose is offered to any channels-first consumer.

Unchanged by earlier decision: the `unfold` hazard note in `converter.rs`.

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

### [U-2] `AttnKv` vs `KVCache` — RESOLVED (documented, not converged)

Corrections to this entry as first written: the three mechanisms serve
**disjoint** stacks, and none is unused, so "retire one" was never available.

| mechanism | used by | provenance |
| --- | --- | --- |
| `KVCache` | nanochat kit + `CausalSelfAttention` | pre-existing on `main` |
| `AttnKv` | the Whisper decoder | new in this branch |
| burn's `MhaCache` | nothing; rejected in the module docs | upstream |

Convergence is *possible* — both trade in head-split `[B, H, T, D]`, so
`insert_kv` takes exactly what `project_kv` produces — but it was rejected.
`AttnKv` serves two roles and only one overlaps: self-attention growth
duplicates `KVCache`, while cross-attention uses it as an immutable per-layer
pair, which `KVCache` does not model at all. Converging would refactor decode
code that is currently verified against the ONNX reference, gain no
capability, and still leave the cross-attention half uncovered.

The real defect was documentary. The module already answered "Why not
`MhaCache`" but said nothing about `KVCache` — which lives in the same module
and is re-exported from the same `attention::*`, so it is the question a
bunsen reader hits first. That section now exists, stating the split:
`KVCache` for a preallocated multi-layer cache over bunsen's own attention
with the geometry known up front; `AttnKv` for per-layer k/v over burn's
`MultiHeadAttention`, cross-attention included.

`AttnKv::concat`'s cost is now recorded where it is paid: growing by
`Tensor::cat` copies the whole cache each step, so a decode of length T copies
O(T^2). Invisible at Whisper's 448-position window, wrong for long context,
and the doc points at `KVCache` for that case.

### [U-3] `MelConverterOptions::to_vec_*` — RESOLVED (no action)

Closed without changes. This entry was written on the assumption that the
three extractors were invented surface; they are not — `to_vec_*` /
`to_tensor_*` is an established convention in this very module tree.
`SamplingWindowBuilder` defines the pair (`window_builder.rs`), and both
`stft_window.rs` and `cosine_window.rs` implement it. The mel options apply the
same idea to the same module's other host-side artifact: materialize on the
host what the config describes.

`to_vec_filterbank` also has a demonstrated external use — `cross_test` calls
it exactly as an outside consumer would, and it saves unpacking seven config
fields to reach `mel_filterbank`.

**Option 1 from this entry does not survive contact.** Moving
`to_vec_dft_tables` to `ops::signal` as a generic builder fails twice: it is
not generic (its tables are `n_fft` rows rather than `fft_len`, folding the
`pad_to_pow2` widening back in — a fact about this converter's geometry, not
about DFTs), and there is no second caller, since `sliding_stft` goes through
burn's `stft`/`rfft`. Extracting it would create public generic API with one
consumer, which is the speculative generality `[U-5]` objects to.

Noted, not acted on: the convention pairs `to_vec_X` with `to_tensor_X` and
only the `to_vec_` half exists here — correct, since the tensor side belongs to
the `Module` built by `try_init`. And `to_vec_dft_tables` returns a bare
`(Vec, Vec)` where the caller must know cos-then-sin; a named struct would be
safer but is more ceremony than one call site earns.

### [U-4] `mel_windows` in the Whisper kit — RESOLVED (extracted)

Option 2 from this entry landed: there **was** an existing equivalent.
`mel_windows` hand-rolled a `while seek < frames` loop that reproduced
`Tensor::split` — burn's implementation is the same loop with `narrow` for
`slice_dim` — and added zero-padding of the short final chunk. Only the padding
was new, and the empty-input behaviour matched already.

That padding is now `ops::split::split_padded`. Placement deviates from this
entry's wording: the operation carries no signal or mel content, and `ops/mod`
already has a **Shape transforms** section with `repeat_interleave` as a
single-function module at the `ops` root. Putting a signal-agnostic op under
`ops::signal` would have been the same kind of misplacement the `[L-*]` items
were about. It follows `repeat_interleave`'s shape: tensor first, `dim` last
and negatively indexable via `AsIndex`.

`mel_windows` stays in the kit as a delegation. It still earns its keep — it
names the frame axis and states the padding policy as Whisper's 30 s context,
domain knowledge the generic function cannot carry.

The speculative-generality objection that closed `[U-3]` does not apply here:
`split_padded` is not new capability wrapped around one caller, it is the
subtraction of a reimplementation, and what remains is five lines over a burn
primitive.

### [U-5] Speculative / dead config surface — RESOLVED

Three sub-items that this entry wrongly lumped together; they resolve
differently.

**The "unused" enum variants were a non-issue — the entry measured the wrong
thing.** `PaddingMode`, `SpectrumKind`, `LogBase` and `RangeClamp` variants are
dispatched on in real code (`context.rs` `pad_len` and `t_stage_extend`) and
exercised by tests (`with_end_padding(PaddingMode::Zero)`,
`with_log_base(LogBase::E)`). Nothing dead. No action.

**`pre_emphasis` / `remove_dc` stay, and the reason is serde.** This entry
argued "a config field that always errors is worse than an absent one", which
holds in Rust — you cannot set a field that does not exist. But
`MelConverterOptions` derives `Config`, which is serde, with a round-trip test;
burn's derive does **not** emit `deny_unknown_fields`, so unknown keys are
silently ignored. Delete the fields and a JSON config carrying
`"pre_emphasis": 0.97` loads clean and does nothing. The field plus the
`validate` rejection is exactly what turns that silence into an error. They are
also documented as landing with `PLAN.md`'s Stage 8, which is live work.
`t_stage_preproc` staying an identity follows from the same decision.

**`spectrum_impl` was the real finding, and worse than recorded.**
`MelConverter::spectrum` never read it — the DFT-matmul path ran
unconditionally. So the field was inert *and* unguarded, where the other two
inert options at least reject loudly. Harmless at one variant; the moment
`Stft` was added, setting it would have been silently ignored.

Fixed by matching on it exhaustively in `spectrum`. Same body, no runtime cost,
no API change — but the field is read, and a new variant now fails to compile
until it has a path. That is stronger than a `validate` check because it fires
at build time. Verified by adding a probe variant:
`error[E0004]: non-exhaustive patterns`.

### [U-6] `t_stage_*` visibility — RESOLVED (attribute dropped, width kept)

`pub(crate)` is correct and stays. `PLAN.md`'s "Visibility" section records the
decision deliberately — private-to-the-crate keeps the stages off the public
API surface while letting a future `mels::testing` module drive them, the same
shape `ops::signal::testing` now has one level up after `[U-1]`. With more mel
work planned, that reservation is not idle.

What was wrong was the attribute beside it. `#[doc(hidden)]` on a `pub(crate)`
item is a no-op: rustdoc does not document non-public items without
`--document-private-items`, and under that flag the attribute *hid* them — the
one context where a maintainer reading the pipeline wants them, and where the
plan says their shape contracts are the whole point. Six attributes removed;
no visibility or behaviour change. The plan's note now says not to re-add it.

Narrowing to plain private `fn` was rejected: every caller is in-file today, so
it would compile, but it contradicts a documented decision and would need
reverting the moment `mels::testing` exists.

### [U-7] Cross-check crate helpers — CLOSED (the premise was wrong)

This entry claimed `dev_crates/whisper-onnx-crosscheck/` defines `read_f32`,
`read_i64` and `summarize`, duplicating the `[U-1]` helpers. On checking:

- `read_f32` and `read_i64` exist nowhere in the repository.
- `summarize` is not in that crate. It is a single definition in
  `examples/whisper-dev/src/main.rs`, a println-based min/mean/max debug
  printer in a binary, with no duplicate and no bunsen equivalent.
- The crate's only helpers are `load_bunsen` and `decoder_inputs`, both
  crate-specific with no counterpart anywhere.

Written from a stale reading during the original survey and carried into this
document unverified. There is nothing to consolidate; no action.

The reasoning attached to it survives its subject and is folded into `[C-2]`:
a cross-check should not share *test helpers* with what it validates. That is
about helpers, not dependencies — the crate necessarily depends on bunsen,
since comparing bunsen's Whisper to ONNX is its whole job.


---

## Suggested order

1. ~~`[U-1]` test-helper consolidation~~ — **done**.
2. ~~`[L-1]` `repair_strided_weights`~~ — **done**.
3. ~~`[L-2]`~~ (with `[N-4]`), ~~`[L-3]`~~, ~~`[L-4]`~~ — **done**.
4. ~~`[N-1]` plan-file disposition~~ — **done**. `[N-2]` is unblocked.
5. ~~`[N-2]`~~, ~~`[N-3]`~~, ~~`[N-5]`~~ — **done**.
6. ~~`[U-2]` KV-cache convergence~~ — **done** (documented, not converged).
7. ~~`[U-3]`~~ (no action), ~~`[U-4]`~~, ~~`[U-5]`~~, ~~`[U-6]`~~, ~~`[U-7]`~~ (premise wrong), ~~`[L-5]`~~ — **done**.

**All items are closed.** What remains is the Residual list above and the
last-call removals below.

---

## Residual — deferred, with a home needed before this file goes

Nothing below is a defect found by the burn-down; each is a decision taken to
defer, recorded here because `[TODO.md]` is meant to be deleted. Give them a
home — issues, `PLAN.md`, or rustdoc — rather than losing them with the file.

1. **Whisper-shaped defaults on `MelConverterOptions`.** `range_clamp` and
   `affine` default to `Some(..)`, and **every caller turns them off** — the
   Whisper pipeline itself included (`whisper-dev` disables both and applies
   them once after joining, because `PerCall` reduces per call while Whisper
   clamps against the whole clip). They are Whisper's input packaging, not a
   mel-spectrogram convention, and they cause the one surprising behaviour in
   the type: the chunking caveat exists because a non-chunk-invariant reduction
   is on by default. Proposal: default both to `None`, keep the librosa-shaped
   geometry defaults, and add a `whisper::mel_options(..)` factory in the kit.
   **This is free only until `mels/` ships** — the whole module is new on this
   branch, so `MelConverterOptions` has no compatibility surface yet.

2. **Promoting `to_whisper_mels` into the kit.** The Whisper packaging recipe
   (slice `[..-1]`, clamp, affine, transpose) exists only in
   `examples/whisper-dev`, so any real consumer must re-derive it. Pairs
   naturally with item 1. Raised at `[L-4]` and `[U-4]`.

3. **The store-adapter repair** — see `[C-4]`. Blocked on `PytorchStore`
   gaining `with_adapter` upstream, and on making an over-matching pattern
   loud rather than silent.

4. **No test fails when burn fixes the `unfold` defect** — see `[N-3]`. The
   defect did not reproduce under a direct probe, so the behaviour pin that
   would announce an upstream fix was never written.

5. **`Tensor -> Vec<E>` ergonomics** — see `[U-1]`. Deliberately not fixed
   locally; wait for the `burner::tensor` ext traits to be realigned with their
   upstreamed versions.

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
- ~~**`MEL_CONVERTER_PLAN.md`**~~ — resolved by `[N-1]`: **kept**, not
      removed. It is a live design document for planned mel-filter work, now
      at `crates/bunsen/src/ops/signal/mels/PLAN.md`. Do not delete it with
      this file.

Deferred removals with an *external* trigger belong to `[N-3]`, not here —
they are keyed to a burn version bump rather than to this branch finishing.
