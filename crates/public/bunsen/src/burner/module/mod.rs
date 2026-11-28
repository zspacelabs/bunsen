//! # Module Support
//!
//! Helpers for building `burn` modules from configs, and for working with
//! them once built.
//!
//! - [`ModuleInit`] &mdash; the Config → Module step. A bunsen config that
//!   builds its module from a device alone implements it, with a fallible
//!   `try_init` and a panicking `init`. Its docs are the reference for the
//!   config lifecycle: the Simple Config and Stacked Config shapes, each with a
//!   compiled example.
//! - [`ToStructureConfig`] &mdash; the config-side twin of `ModuleInit`: it
//!   lowers an upper policy config of a Stacked Config family to its structure
//!   config, and a blanket impl gives every implementor `ModuleInit` through
//!   that structure.
//! - [`InputDTypes`], [`CoercesInput`], [`ResultDType`] and [`DTypeWrapped`]
//!   &mdash; a block's float dtypes: the one it computes in and casts its
//!   inputs to, and the one its output is in (see [`InputDTypes`]).
//! - [`HasDType`] &mdash; reports the [`DType`](burn::tensor::DType) a value
//!   that is not a block works in: a simulation's state, a dynamic tensor.
//! - [`DTypeMapper`] &mdash; a [`ModuleMapper`](burn::module::ModuleMapper)
//!   that casts every float parameter of a built module to one `DType`
//!   (`model.map(&mut DTypeMapper::new(DType::F16))`).
//! - [`reflection`] &mdash; (feature `reflection`) a queryable XML view of a
//!   built module tree, for selecting parameters with `XPath`.
//! - [`Rekey`] &mdash; a checkpoint's tensor names to a module's paths, built
//!   from rules: a kit routes a checkpoint's prefixes onto its model's fields
//!   and mounts each block's own map for the checkpoint's format there.
//! - [`CanonicalKeys`] &mdash; a module's canonical weight names, and the map
//!   from them onto its record paths: a kit's map lands a checkpoint on the
//!   canonical names, and [`Rekey::then`] the module's map takes them on.
//!
//! - [`Forward1Module`] … [`Forward4Module`] (and [`ForwardModule`], the
//!   first's common name) &mdash; a module's forward by its number of
//!   arguments, for code generic over the modules it calls ([Forward
//!   modules](#forward-modules)).
//! - [`Gate`] &mdash; a view of two modules as a gate, `a(x) ⊙ b(gate)`, its
//!   product taken in a [`Widening`](crate::burner::tensor::Widening).
//!
//! `ModuleInit` and `ToStructureConfig` are also in [`crate::prelude`].
//!
//! ## Forward modules
//!
//! burn gives a module's forward no trait: each module has an inherent
//! `forward`, of its own arguments. [`Forward1Module`] … [`Forward4Module`]
//! name one by its number of arguments, `ForwardN<A1, …, AN>`, its result
//! the associated `Output`; `forward` takes `&self`. They stand in for the
//! `Fn` traits, which a type cannot implement on stable Rust; they are
//! restricted to modules (a [`Module`](burn::module::Module) supertrait),
//! and give way to the `Fn` traits when a module can implement those.
//!
//! A module implements them by delegating to its inherent `forward`, which
//! method calls on the concrete type still resolve to first. Code bounded
//! on two of them for one type names the trait it calls,
//! `Forward1Module::forward(module, x)`: the method's name is the same.
//!
//! The norm and activation seams implement [`Forward1Module`] at every rank.
//! A view such as [`Gate`] borrows modules, and is not one.
//!
//! ## Design Rules
//!
//! The config lifecycle's invariants, for every module family in the crate.
//! [`ModuleInit`] and [`ToStructureConfig`] define the two config shapes,
//! with a compiled example of each; these rules say what a family must do.
//!
//! - **A config lives in the same file as the module it builds.** A config and
//!   its module change together, and the build sits next to the struct it
//!   fills. [`BunsenMlpConfig`] sits beside [`BunsenMlp`].
//! - **A config whose module needs only a device implements [`ModuleInit`],
//!   never an inherent `init`.** One with no parameters ignores the device
//!   (`DropPathConfig`). A policy config implements [`ToStructureConfig`]
//!   instead, and gets `ModuleInit` from the blanket impl. Only a constructor
//!   that needs more than a device keeps an inherent `try_init` / `init` pair
//!   ([hand-written `init`](ModuleInit#hand-written-init)). Every config then
//!   builds through one call shape, and generic code bounded on `C:
//!   ModuleInit<M>` accepts it.
//! - **`try_init` reports a config whose fields disagree** as an
//!   [`Illegal`](crate::errors::BunsenErrorKind::Illegal) error whose cause is
//!   a [`ConstraintError`](crate::errors::ConstraintError) naming the config
//!   and the field, on the shaped [`Rule`](crate::errors::Rule) where one fits
//!   (zero or empty, out of range, not a multiple, a relation between two
//!   fields, a chain) and `custom` for a rule in words; a compound check is one
//!   check per rule. Free-text `illegal` is for a refusal no config or argument
//!   owns (a stream fed after it ended), and `unsupported` for a valid choice
//!   that isn't built. `init` panics only with that error's report, never
//!   through an `assert!` of its own.
//! - **A `try_` body builds its fallible children with `try_`** (`try_init`,
//!   `try_validate`, `try_x`), passing their errors on, and an inherent
//!   `try_init` has its panicking `init` twin. A panicking child inside a
//!   `try_` body is a panic the caller asked not to get.
//! - **A model's numeric choices are config fields**, with `#[config(default =
//!   ..)]` defaults, not constants or statics in its code: a width rounding, an
//!   epsilon, a kernel list. A prefab states a value by setting the field.
//!   Configs are data: deserialized, assembled by loaders, edited by users. A
//!   literal default is written as a literal (`#[config(default = 3)]`, `=
//!   1e-5`, `= false`); quotes mean an expression (`"VIT_NORM_EPSILON"`,
//!   `"None"`, `"Vec::new()"`) or a negative number, which the derive takes
//!   only as a string. The derive documents the default as written, so a quoted
//!   literal reads as a string in rustdoc.
//! - **A Simple family's config is `FooConfig`, and builds `Foo`. A Stacked
//!   family is a `FooStructureConfig`**, the unrolled tree that implements
//!   `ModuleInit`, plus policy configs named for their policy
//!   (`FooContractConfig`, `FooApiConfig`, ...). A bare `FooConfig` never
//!   coexists with a `FooStructureConfig`. A policy's logic, validation
//!   included, lives in `try_to_structure`, and a test checks that
//!   `policy.init(&d)` and `policy.to_structure().init(&d)` agree. Loaders and
//!   tooling work on the tree and users on the knobs, and the two build paths
//!   cannot drift apart.
//!   - A policy's `Structure` is the next config down; lowering may chain
//!     through another policy, as `LmDecoderVariantConfig` lowers to
//!     [`LmDecoderStructureConfig`].
//!   - A family's user-facing config over another family's structure is that
//!     family's own, and keeps the bare name: [`ConformerEncoderConfig`] lowers
//!     to [`SubsampledEncoderConfig`].
//!   - Promote a Simple family to Stacked when its user-facing knobs diverge
//!     from the implementation's parameters, when a second default policy
//!     appears, or when loaders and tooling need the unrolled tree.
//! - **Every family has a narrow `FooMeta` trait**, which its config(s) and its
//!   module implement: raw fields as required methods, derived values as
//!   provided ones. Everything else stays on the config, reached from the
//!   module through an accessor (`options()`, `config()`). A Simple family's
//!   `FooConfig` and `Foo` implement it; a Stacked family's
//!   `FooStructureConfig` and `Foo` must, and its policies may. A test asserts
//!   that a config and the module it builds answer every method alike, with a
//!   config that is non-default in every field. Tests and reflective code then
//!   read the geometry from either form ([`ExpertStackMeta`]).
//! - **A type that owns a `Tensor` derives `Module`**, as `#[derive(Module,
//!   Debug)]` without `Clone` (the derive provides it), even when nothing in it
//!   is learnable: `Module` is what lets `to_device` reach the tensors. A held
//!   config or other value object is `#[module(skip)]`.
//!   - A tensor rebuilt from the config (a table, a window, a state) is held
//!     bare, not as a `Param`. Bare tensors are not written to records, are
//!     skipped by `ModuleMapper` passes (dtype casts), and do not appear in
//!     reflection; the type says so. [`SlidingStft`]'s window is one.
//!   - A tensor that a checkpoint fills is a `Param`, even when nothing trains
//!     it, because only records reach it: [`EuclideanCodebook`]'s loaded
//!     buffers.
//! - **A constant tensor that meets a model's data is cast to that data's dtype
//!   where they meet**: a mask, a ramp, a lookup, a host-side embedding. A
//!   block states the float dtype it computes in ([`InputDTypes`],
//!   [`ResultDType`], [`CoercesInput`]), or its holder wraps it in
//!   [`DTypeWrapped`]. burn does not promote dtypes: on the cubecl backends a
//!   mixed op runs and returns garbage, and only a half-precision run shows it.
//! - **Precision is always chosen explicitly.** A load's dtype comes from its
//!   `LoadPolicy`, which only widens; an f32 island is a
//!   [`Widening`](crate::burner::tensor::Widening) config field; a table's
//!   dtype comes from a policy such as [`RotaryTableDTypePolicy`]. The device's
//!   default float stays f32, and nothing sets it to a half dtype: a half
//!   default hides loaders that skip the policy, and silently narrows tables.
//!
//! [`BunsenMlp`]: crate::blocks::mlp::BunsenMlp
//! [`BunsenMlpConfig`]: crate::blocks::mlp::BunsenMlpConfig
//! [`LmDecoderStructureConfig`]: crate::blocks::transformers::models::language_decoder::LmDecoderStructureConfig
//! [`ConformerEncoderConfig`]: crate::blocks::transformers::encoders::conformer::ConformerEncoderConfig
//! [`SubsampledEncoderConfig`]: crate::blocks::transformers::encoders::SubsampledEncoderConfig
//! [`ExpertStackMeta`]: crate::blocks::transformers::moe::experts::ExpertStackMeta
//! [`SlidingStft`]: crate::ops::signal::SlidingStft
//! [`EuclideanCodebook`]: crate::blocks::quantization::EuclideanCodebook
//! [`RotaryTableDTypePolicy`]: crate::ops::embedding::RotaryTableDTypePolicy

#[cfg(feature = "reflection")]
pub mod reflection;

mod canonical_keys;
mod dtype_traits;
mod forward_module;
mod gate;
mod has_dtype;
mod module_init;
mod rekey;
mod tensor_load_transforms;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod to_structure_config;
mod type_mapper;

pub use canonical_keys::*;
pub use dtype_traits::*;
pub use forward_module::*;
pub use gate::*;
pub use has_dtype::*;
pub use module_init::*;
pub use rekey::*;
pub use tensor_load_transforms::*;
pub use to_structure_config::*;
pub use type_mapper::*;
