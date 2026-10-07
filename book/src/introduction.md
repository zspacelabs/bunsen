# Introduction

## bunsen - tackling tomorrow's burn problems today!

The [burn](https://burn.dev) team is working hard to give us a stable long term foundation
for developing high-performance and maintainable tensor programming in
Rust; The "in Rust" aspect requires a degree of type and runtime R&D research
and that requires a conservative posture to api extension.

`bunsen` provides a "batteries included" complementary community standard library
for research extensions to `burn`; ideas which are useful enough
to collect into one place, document, and build upon; but front-run
the `burn` interface frontier.
Every idea in `bunsen` seeks to either be refined and ported into `burn`,
or supplanted by the eventual `burn` solution to that problem.

## This book and the API docs

The [API docs](bunsen) answer *what is this type, and how do I call it?*
This book answers *how is bunsen put together, why is it built that way, and
how do we work on it?* The two are deliberately split:

- every interface, lifecycle and example lives in the API docs, where the
  compiler checks it;
- this book explains the organization, the bunsen-specific systems that cut
  across modules, and the development strategies behind the code, and links
  into the API docs for everything else.

Every link from this book into the API is checked when the book is built,
so a renamed or moved item breaks the build, not the reader.

## Why a "standard library"?

The burn ecosystem moves quickly, and single-purpose extension crates tend
to drift out of sync with each burn release. `bunsen` exists to:

1. track the `burn` release cycle, so dependent code doesn't have to;
2. give one dependency surface for common building blocks, instead of a
   tangle of single-purpose crates;
3. centralize testing, documentation and validation, so contributed
   components can be trusted across projects.

## How to read this book

- **New to bunsen?** Start with [the workspace](./organization/workspace.md)
  and [the bunsen crate](./organization/crate.md), then
  [installing and first use](./organization/install.md).
- **Building models on bunsen?** Read
  [module design conventions](./systems/conventions.md) and the systems
  chapters relevant to you: [shape contracts](./systems/contracts.md),
  [pretrained models](./systems/pretrained.md),
  [streaming contexts](./systems/streaming.md),
  [parameter groups](./systems/param-groups.md).
- **Contributing?** The [development strategies](./development/setup.md)
  part explains how bunsen is tested, validated and released;
  [`CONTRIBUTING.md`](https://github.com/zspacelabs/bunsen/blob/main/CONTRIBUTING.md)
  is the canonical contributor guide.
