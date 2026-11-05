# `bunsen` Naming and Placement

Rules for naming and placing the public items of the `bunsen` crate. They extend the
repository's [style guide](../../../STYLE.md), whose "Module design" section covers how a
module, its config and its metadata fit together, and where `ops` ends and `blocks` begins.

## Names

Every module re-exports its children with `pub use m::*;`, so a public name spreads into every
namespace above it, and sits beside burn's wherever a kit glob-imports both.

* A type that is bunsen's own version of a burn type, in the same role, is `Bunsen{BurnName}`:
  `BunsenActivation`, `BunsenNormalization`. A glob import of burn's `nn` beside bunsen's
  `blocks` then has no ambiguous names.
  * The exception: a shadow that burn-store recognizes by its type's name keeps burn's name.
    `blocks::norm::LayerNorm` is one: burn-store's adapters map a checkpoint's `weight` and
    `bias` onto a `LayerNorm`'s `gamma` and `beta` by matching `Struct:LayerNorm`, and leave it
    out of the half-precision cast.
* A public name carries its family or its owner: `DecoderStackCache`, not `DecoderStackCache`;
  `TransformerMlp`, not `TransformerMlp`; `DwStridingFlattenOrder`, not `DwStridingFlattenOrder`.
* A name says what the value is, in the field's standard terms. The residual is the branch,
  `F(x)`; `x + F(x)` is an update to the residual stream, so a function returning it is
  `*_residual_update`. "Connection" names a link, not a value.

## Placement

* A free function that works on one type's internals is a method, or an associated function,
  of that type.
* A function with a natural home in `ops` (a tensor operation, a table) or in `burner` (dtype,
  tensor and module machinery) lives there, not in `blocks`.
* An operation specific to one kit lives in that kit's own `ops` submodule (`swin::ops`, `conway::ops`),
  not in the crate's `ops`.
* Two names for one concept become one when the concept is unambiguously the same. When the
  parts do not line up, they stay apart, each named for its owner.
* An enum seam's kinds are modules of their own, never bare values: weight loading relies on
  each variant holding a module.
* Math operations as modules live in `blocks::math`.
* A module whose natural name would collide with a function or an `ops` value object (`ClampOp`,
  a parameter set that embeds in configs) spells out `Module`: `PowOpModule`, with
  `PowOpModuleConfig` and `PowOpModuleMeta`.

## Kit layout

* `kits` groups the model kits by domain: `images`, `language`, `speech`. `kits::support` holds
  the machinery shared across domains. Simulations are not model kits: they are `bunsen::sims`.
* A domain whose kits serve different tasks groups them by task: `speech::{asr, diarization,
  vad}`, each with its own model-independent parts (`asr::driver`) and its models. A family
  layer that crosses tasks (`speech::nemo`) and the domain's `testing` sit beside the tasks.
* A kit's module is named for its model, as close to the model's own name as a Rust identifier
  allows, even when that repeats the group: `speech::asr::qwen3_asr`, `speech::vad::silero_vad`.

## Test fixtures

* A fixture may stay local to the tests that use it.
* A fixture that establishes a general mechanism, independent of any one test, goes to a
  `testing` module:
  * `support::testing` or `errors::testing`, for the crate-wide ones;
  * a module's own `testing` submodule, for one module's fixtures
    (`blocks::transformers::embedding::testing`).
* A `testing` module is `pub mod testing`, gated `#[cfg(any(test, feature = "testing"))]`, and
  is not glob-re-exported: callers name its path.
