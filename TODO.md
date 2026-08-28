# Normalization burn-down (round 2)

Scope: the full diff `main...crutcher/whisper_wip` — **56 commits, 53 files,
+8148/-105**, of which 30 commits are the Whisper feature work and 26 are the
first normalization pass.

This is a **re-survey**. Round 1 opened 17 items across narrative, layer
violations, and new utilities; all 17 are closed, and their reasoning lives in
the commit messages rather than being restated here. What follows is what the
re-run found: a short list of open items, the conventions that survived, and
the deferred decisions that still need a home.

Same rules as before: options, not fixes, until each is agreed.

Legend: `[N-n]` narrative, `[L-n]` layer violation, `[U-n]` new utility.

---

## Round 1: closed

| item | outcome |
| --- | --- |
| `[U-1]` duplicated test helpers | assertions promoted to `support::testing`; `to_f64` deleted; fixtures left local per `[C-2]` |
| `[L-1]` `MlpConfig::repair_strided_weights` | removed; the kit repairs its own MLP weights, symmetric with its MHA |
| `[L-2]` Whisper named inside `burner/` | back-reference cut; `## Scope` evidence kept per `[C-5]` |
| `[L-3]` Whisper rationale in `multihead_utils` | example deleted; regression-test rationale restated without the incident |
| `[L-4]` `# Producing Whisper input` recipe | deleted — it duplicated working code in `whisper-dev` |
| `[N-1]` `MEL_CONVERTER_PLAN.md` | kept, co-hosted as `mels/PLAN.md`; all three rustdoc citations were vestigial |
| `[N-2]` `Stage N:` ordinals | pipeline stages named; build stages keep numbers |
| `[N-3]` "delete on the burn bump" | `analyze` defended and directly tested; notes left by decision |
| `[N-4]` defect-history prose | recast as a live constraint |
| `[N-5]` cross-check defect counts | three counts dropped; the defect list kept |
| `[U-2]` three K/V caches | documented, not converged |
| `[U-3]` `to_vec_*` extractors | no action — they follow the house `to_vec_*` convention |
| `[U-4]` `mel_windows` | `ops::split_padded` extracted; it had reimplemented `Tensor::split` |
| `[U-5]` dead config surface | `spectrum_impl` turned into a compile-time guard; the rest kept, with reasons |
| `[U-6]` `t_stage_*` visibility | redundant `doc(hidden)` dropped; `pub(crate)` kept |
| `[U-7]` cross-check helpers | closed — the premise was wrong, the named helpers do not exist |
| `[L-5]` Whisper defaults and prose | one doc fix; the rest a false positive |

---

## 1. Narrative discussion

### [N-6] `PLAN.md` pointed at a personal branch and a commit SHA — RESOLVED

The reference named branch `crutcher/tv-dev`, commit `671b494`, and an IDE run
config that no longer exists — none of it resolvable by a reader. Rather than
delete it, the module it pointed at was **ported**: `burner::repro::unfold`,
its own file beside the stride repro, one file per defect. `PLAN.md` now cites
it as an ordinary in-repo reference and states the rule inline.

**It still reproduces.** On wgpu at burn 0.21.0 / cubecl 0.10.0 the sweep finds
42 of 315 configurations wrong — the count recorded when it was written — and
row 1 starts at the truncated offset. `Flex` is correct.

That corrects an earlier finding of mine recorded under `[N-3]`: a direct probe
had shown no corruption and I reported being unable to reproduce the defect.
The probe used float tensors at large geometries; every observed failure has an
inferred line width of 2 or 4. The defect is real; I generalised a negative
result past the range it covered.

A behaviour pin was added, closing Residual 4 — see that entry.

### [N-7] `PLAN.md` cites a kit source file by line number — RESOLVED

Dropped. The sentence already names `WhisperModel::forward_encoder`, which is
the durable locator; the path and line added nothing findable and were the only
part that could silently go wrong.

Correction to this entry as opened: the citation was **still accurate** — line
209 is exactly the claim it referenced. The case was prospective, not remedial.

The plan's four other source-line citations are **kept**. Three sit under the
Findings table, whose header pins them to `burn-tensor-0.21.0`, so they are
dated evidence of a resolved investigation rather than references that drift
with local edits; they can only go stale on a burn bump, which already has
`[N-8]` and Residual 7 attached to it. Two loose ends noted and judged not
worth the churn: `base.rs:2767` inherits that version anchor but not its path,
and `burn-core/.../primitive.rs:14` has no anchor at all.

### [N-8] One "delete this comment" instruction remains

`crates/bunsen/src/ops/signal/mels/converter.rs:749` still reads "Fixed
upstream in burn 0.22.0-dev. Keep the slice on the bump (it states the
contract), but delete this comment." Left by decision in round 1, recorded here
so it is a known state rather than an oversight. Nothing enforces it.

---

## 2. Layer-violating discussion and APIs

### [L-7] Model and kit names introduced by round 1 — RESOLVED

Both deleted. Neither loses information, which is the same test `[L-3]`
applied.

- "and the nanochat kit" was redundant: the sentence says `KVCache` serves
  bunsen's own attention stack, and the `CausalSelfAttention` link beside it
  already says exactly that. Considered keeping it as the index-of-consumers
  pattern `[L-5]` preserved in `mels` — it is not that. That pattern names
  which convention *each* variant follows, consistently, to help a caller
  choose; this was one parenthetical naming a downstream consumer.
- "Whisper's text window is 448" illustrated a sentence that had already made
  its point — structurally identical to the cross-attention example deleted at
  `[L-3]`.

`blocks/` is again free of model names except `blocks/mod.rs:16` and
`conv_seq_1d.rs:198`, both of which predate this branch.

### [L-6] `conv_seq_1d.rs:198` — still excluded

Pre-existing on `main`, untouched by this branch. Re-verified. Not this
branch's to fix; worth a separate cleanup if wanted.

### Settled, re-verified as **not** defects

`mels/converter.rs`'s Whisper mentions (variant-usage index, and provenance for
`RangeClamp`/`AffineCompress`), `mels/cross_test.rs` (parity test — `[C-5]`),
`burner/repro/pytorch_strided_weights.rs:39` (evidence — `[C-5]`), and
`blocks/mod.rs:16` / `support/audio.rs:70` (both pre-existing on `main`).

---

## 3. New utility methods

### [U-8] `mels::testing` reserved but absent — RESOLVED (note fixed, width kept)

No consumer exists. Nothing outside `mels/` touches `MelConverter` except the
`whisper-dev` example; inside it, `cross_test.rs`'s helpers are used only by
`cross_test.rs`, and the `t_stage_*` methods only by `context.rs`'s own tests.
So there is nothing to put in such a module, and building one to satisfy a plan
note would be an empty shell. `ops::signal::testing` earned its existence from a
helper used across two files — that is the standard to hold this to.

The width stays, per `[U-6]`. What was wrong was the note justifying it:

- it cited `ops::signal::window_builder::testing` as the model to follow, and
  **that module no longer exists** — `[U-1]` moved it to `ops::signal::testing`
  in `b693ce4` and the plan citing it was never updated. Same class of miss as
  `[N-6]` and `[L-7]`, and further evidence for `[C-6]`.
- it read as though `mels::testing` exists, rather than being a reservation.

Both corrected; the note now also says not to add the module until something
needs it. A sweep found no other references to the moved path.

### Re-verified as clean

The decode API is properly layered — `decode_window` delegates to
`decode_window_batched`, and `decode_chunked` maps `decode_window` over
`mel_windows` — so there is no duplication among the three. A scan for
duplicated helper bodies across the crate found none beyond trait-method and
accessor names.

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

**C-6. Re-survey after the pass, not just before it.** Round 1's `[U-2]`
commit put two caller names into `blocks/` **two commits after** `[L-3]`
removed the identical pattern from a neighbouring file. Writing new prose is
exactly when a violation gets reintroduced, and no amount of care during the
fix prevents it — only a second sweep found it. Treat the re-survey as a step
of the work, not an optional check: a normalization pass that is not re-run
against its own output has not been verified. The same sweep also caught a
stale scope header, a reference to a deleted run config, and a conclusion of
mine that was simply wrong (`[N-6]`).

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

4. ~~**No test fails when burn fixes the `unfold` defect.**~~ **Closed.**
   `burner::repro::unfold::test_performance_backend_is_currently_wrong` asserts
   the sweep still finds wrong configurations, so a fix breaks it and names the
   covered-span trims to re-check. Gated on an accelerator feature, because
   `PerformanceBackend` falls back to the (correct) `Flex` otherwise; verified
   absent on a CPU-only run and present with `--features wgpu`.

5. **`Tensor -> Vec<E>` ergonomics** — see `[U-1]`. Deliberately not fixed
   locally; wait for the `burner::tensor` ext traits to be realigned with their
   upstreamed versions.

6. ~~**The `unfold` repro on `crutcher/tv-dev`.**~~ **Closed** — ported; see
   `[N-6]`.

7. **The two covered-span comments describe geometries that do not trip the
   defect.** `converter.rs` says the Whisper geometry "trips it at every chunk
   size" and `sliding_stft` says the default 1024/256 geometry is corrupt for
   `batch > 1`. Probed directly with `Int` tensors, neither fires — every
   observed failure has line width 2 or 4, and the sweep covers only
   `size 2..=8`. The trims stay regardless: they are semantically free and make
   the hazard unreachable rather than merely unobserved, which is why Whisper
   no longer breaks. Left by decision; recorded so the wording is a known state.

---

## Last-call removals

Artifacts that exist only to carry this normalization work and must not outlive
it. Check these off before the branch lands.

- [ ] **`TODO.md`** (this file). Once the items above are resolved or moved
      somewhere durable, delete it. The **Conventions** section is the part
      most likely to deserve a real home — decide where it goes rather than
      keeping this file alive to hold it.
- ~~**`MEL_CONVERTER_PLAN.md`**~~ — resolved by `[N-1]`: **kept**, not removed.
      It is a live design document for planned mel-filter work, now at
      `crates/bunsen/src/ops/signal/mels/PLAN.md`. Do not delete it with this
      file.

Deferred removals with an *external* trigger — the burn-version-bump comments —
are `[N-8]` and Residual 4, not last-call items: they are keyed to an upstream
release rather than to this branch finishing.
