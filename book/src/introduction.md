# Introduction

## bunsen - inventing tomorrow's burn problems today!

The [burn](https://burn.dev) community is building a stable, long-term
foundation for high-performance, maintainable tensor programming in Rust;
work which requires extensive incremental novel research into Rust type and
runtime extensions.

`bunsen` contributes to the community work as a "batteries included"
standard library for the frontier, by driving real models and applications
ahead of the `burn` foundation: exploring the novel problems they raise,
unifying their partial solutions incrementally, and upstreaming stabilized
components back into `burn`.

The frontier has two halves: the components model families are built from,
and the runtime around those models (loading, driving and testing them).
`bunsen` drives research through a development pipeline in which each stage
increases breadth:

- The frontier is **Expanded** by porting new model families and the
  components they need, widening what is possible.
- The model families are **Consolidated** by building and improving the
  runtime around those applications (loading, driving, testing), making them
  supportable.
- Bespoke solutions are incrementally **Unified** into shared parts, making
  the next family easier to develop.
- Stabilized foundations are **Upstreamed** into `burn`, making them
  standard.

Every idea in `bunsen` is headed for `burn`: refined and ported, or
supplanted by `burn`'s own solution.

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
  [pretrained models](./systems/data.md),
  [streaming contexts](./systems/streaming.md),
  [parameter groups](./systems/param-groups.md).
- **Contributing?** The [development strategies](./development/setup.md)
  part explains how bunsen is tested, validated and released;
  [`CONTRIBUTING.md`](https://github.com/zspacelabs/bunsen/blob/main/CONTRIBUTING.md)
  is the canonical contributor guide.
