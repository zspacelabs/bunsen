# Module design conventions

A bunsen module family has three parts: a config, the module it builds, and
a narrow trait that both of them answer. This chapter explains
why the parts are shaped as they are, and how to choose between the two
config shapes, Simple and Stacked. What the shapes are and how they work is
defined, with a compiled example of each, in
[`ModuleInit`](bunsen::burner::module::ModuleInit#two-config-shapes) and
[`ToStructureConfig`](bunsen::burner::module::ToStructureConfig). What the
author of a family must do (naming, promotion, the required tests) is set by
[STYLE.md, "Module design"](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#module-design).

## Why conventions at all

burn provides the basic parts: a `#[derive(Config)]` struct that builds a
`#[derive(Module)]` struct. That is enough for one self-contained module. It
starts to strain once modules nest inside larger ones, and once configs come
from files, loaders and users instead of from your own code. Nesting raises
three questions, and the conventions answer each of them the same way
everywhere:

- how a config reports that it cannot build its module;
- how a parent, a loader or a test reads a child's geometry without keeping
  a second copy of it;
- how a module can offer users a short list of knobs while its
  implementation needs a long list of parameters.

## Common to both shapes

**The config lives in the module's file.** The knobs sit next to the code
they drive, and a reader never has to search for the other half.

**A narrow `FooMeta` trait.** A parent that owns a child needs structural
facts about it: its width, its head count, its output resolution. If the
parent copies those numbers into its own fields, the same number lives in
two places, and the copies can disagree. Instead, the family defines a trait
that holds only the values a caller or a test needs to read back. Both the
config and the module implement it, and the module answers from its live
tensor shapes. Code that holds either form can ask the same question.

**[`ModuleInit`](bunsen::burner::module::ModuleInit), with a fallible
`try_init`.** A config is data. It is deserialized, assembled by loaders,
and edited by users. A config whose fields disagree is an input error to
report, not a bug, so `try_init` returns an error and `init` is the
panicking convenience
([why it is fallible](bunsen::burner::module::ModuleInit#why-it-is-fallible)).
The trait is in [`bunsen::prelude`]. The module type comes
from the binding, not from the device
([calling it](bunsen::burner::module::ModuleInit#calling-it)). A few configs
need more than a device to build their module, and keep an inherent `init`
instead
([hand-written `init`](bunsen::burner::module::ModuleInit#hand-written-init)).

**`try_x` and `x`.** The same pairing runs through the whole crate. `try_x`
returns a [`BunsenResult`](bunsen::errors::BunsenResult), and `x` (or
`expect_x`) panics through
[`WithOkOrPanic`](bunsen::errors::WithOkOrPanic). Input that can be wrong is
reported as an error of the [kind](bunsen::errors::BunsenErrorKind) it is,
not as a panic: a config that breaks its own documented rule is `Illegal`,
and the same config read from a file is `Policy`
([the convention](bunsen::errors#convention-try_x-and-x),
[which kind](bunsen::errors#which-kind); see also [Errors](./errors.md)).

**Modules over bare tensors.** A type that owns a tensor derives `Module`
even when nothing in it is learnable. `Module` is burn's traversal trait,
and deriving it is what lets `to_device` reach the tensors. Hold fixed
tables as bare tensors, not as `Param`s. A bare tensor is not written to a
record, `ModuleMapper` passes skip it, and reflection does not see it, so
say so on the type. A config that the module keeps is a `#[module(skip)]`
field with an accessor, as in
[`PerceptiveAudioConverter::options`](bunsen::ops::signal::perceptive_audio::PerceptiveAudioConverter::options).
The derive treats every field that is not a primitive as a sub-module, so
any other value object the module holds is skipped too.

**The agreement test.** A family's tests build a module from a config that
is non-default in every field, then assert that the config and the module
answer every `FooMeta` method alike. This test catches a module that
silently ignores one of its knobs.

**Variant behaviour lives on the enum.** When a value differs per variant,
it is a method on the enum, not a `match` at each use site. A new variant
then changes one place.

## Simple Config

`FooConfig` builds `Foo` and implements `ModuleInit` itself. Both implement
`FooMeta`. This is the shape for a module whose user-facing knobs are the
parameters its implementation needs.

In the crate, [`MlpConfig`](bunsen::blocks::transformers::mlp::MlpConfig)
builds [`Mlp`](bunsen::blocks::transformers::mlp::Mlp), and
[`SlidingStftConfig`](bunsen::ops::signal::SlidingStftConfig) builds
[`SlidingStft`](bunsen::ops::signal::SlidingStft). The compiled toy is
[`ModuleInit`'s Simple Config example](bunsen::burner::module::ModuleInit#simple-config).

## Stacked Config

A module's parameter list grows in two unrelated directions. The knobs that
describe what the module is for, such as "12 layers, 768 wide", are few and
stable. The parameters the implementation needs, a config for every
sub-module of every layer, are many, and they change whenever the
implementation does. A single config that holds both is tedious to fill in.
It also turns every implementation change into a breaking change for callers
who only wanted twelve layers.

The Stacked shape keeps the two apart:

- **`FooStructureConfig`** is the unrolled tree, with one field per
  sub-module config, and it implements `ModuleInit`. Loaders and tooling
  work with it, because the layers are spelled out there.
- **At least one upper policy config** computes that tree from the user's
  knobs. It is named for its policy (`FooContractConfig`, `FooApiConfig`,
  `FooSignalConfig`) and never bare `FooConfig`, which would mean the Simple
  shape. "Contract" here names the user-facing knobs; it has nothing to do
  with [shape contracts](./contracts.md). Several policies may build the
  same structure, or refine one another.
- **A policy implements
  [`ToStructureConfig`](bunsen::burner::module::ToStructureConfig)**, and
  gets `ModuleInit` from that trait's blanket impl. If a policy also
  implements `ModuleInit` itself, the compiler rejects it (E0119). That is
  deliberate: `policy.init(&device)` and `policy.to_structure().init(&device)`
  run the same code, so they cannot drift apart
  ([`init` for free](bunsen::burner::module::ToStructureConfig#init-for-free)).
  Per-policy logic, including validation, goes in `try_to_structure`.
- **`FooMeta`** is required on the structure config and the module, and
  optional on the policies.

Both pathways are real uses. A caller lowers first to inspect or edit the
tree before building it, as a loader overriding one layer would. A caller
that only has the knobs builds straight from the policy. Each Stacked family
has a test that the two pathways build modules that agree. The compiled toy
is [`ModuleInit`'s Stacked Config example](bunsen::burner::module::ModuleInit#stacked-config).

The crate's Stacked families show the variations:

- **ResNet** has one policy,
  [`ResNetContractConfig`](bunsen::kits::images::resnet::ResNetContractConfig),
  over [`ResNetStructureConfig`](bunsen::kits::images::resnet::ResNetStructureConfig).
  Inside it,
  [`ResidualBlockStructureConfig`](bunsen::kits::images::resnet::blocks::ResidualBlockStructureConfig)
  is an enum. That makes it one contract with two implementations (basic and
  bottleneck), and the enum, not its callers, dispatches each method to the
  variant it holds.
- **Silero VAD** has two chained policies.
  [`SileroVadSignalConfig`](bunsen::kits::speech::silero_vad::blocks::SileroVadSignalConfig)
  refines into
  [`SileroVadStftConfig`](bunsen::kits::speech::silero_vad::blocks::SileroVadStftConfig)
  through an inherent
  [`to_stft`](bunsen::kits::speech::silero_vad::blocks::SileroVadSignalConfig::to_stft),
  and both lower straight to
  [`SileroVadStructureConfig`](bunsen::kits::speech::silero_vad::blocks::SileroVadStructureConfig).
- **Whisper** has
  [`WhisperApiConfig`](bunsen::kits::speech::whisper::blocks::WhisperApiConfig)
  over
  [`WhisperStructureConfig`](bunsen::kits::speech::whisper::blocks::WhisperStructureConfig).
- **Swin Transformer V2** has a fallible lowering.
  [`SwinTransformerV2ContractConfig`](bunsen::kits::images::swin::v2::SwinTransformerV2ContractConfig)
  checks that its stages fit the input and the window, so `try_init` on a
  bad policy returns an error instead of panicking.

## Choosing, and promoting

Start with Simple. Promote a family to Stacked when one of these happens:

- the knobs users set diverge from the parameters the implementation needs;
- a second default policy appears;
- loaders or tooling need the unrolled tree.

When you promote, `FooConfig` takes a policy name and a `FooStructureConfig`
appears. A bare `FooConfig` never coexists with a `FooStructureConfig`. Call
sites that build with `.init(&device)` keep working, because the policy gets
`init` from the blanket impl; only the type name changes.

## Pitfalls

- **`init` needs the module type.** `ModuleInit` is generic over the module
  it builds, so where nothing else pins that type, `config.init(&device)`
  needs a binding such as `let m: Foo = ...`, or inference fails.
- **A bare tensor is not loaded.** A fixed table held as a bare tensor is not
  in the record, so `init` must recompute it. A load never restores it.
- **Don't override a policy's `init`.** The E0119 error you get when you try
  is the convention working as intended. Put the logic in
  `try_to_structure`.
