# Writing documentation

bunsen has two kinds of documentation, and each fact belongs in exactly one
of them. This page is the contract between them, for anyone writing either:
what goes where, how the book links into the API, how code in the book stays
correct, and how to build the book. STYLE.md's
[rustdoc chapter](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#rustdoc)
has the rules for the rustdoc side.

## rustdoc is the reference

rustdoc documents every interface and every lifecycle: what a type is, how a
config becomes a module, how a name becomes a loaded model, what a method
returns and when it fails. The compiler checks its links, its examples run as
doctests, and it moves with the code.

The book explains what rustdoc can't hold: how the workspace and the crate
are organized, the systems that cut across modules and why they are built
the way they are, and how the project is developed. It never restates a
signature, a method or field list, a feature table or a module tree. It
names the concept and links to it.

When a book page needs a fact that rustdoc doesn't have, the fix is in
rustdoc: add the fact there, then link to it. A fact that lives only in the
book rots, because nothing checks it
([rustdoc is the reference](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#rustdoc-is-the-reference)).

## Module docs must render

A private module's `//!` never reaches the API docs. With bunsen's
`mod x; pub use x::*;` layout, rustdoc shows the re-exported items and
nothing of the file, so prose written there is invisible and goes stale
unchecked. Put what a type is on the type's `///`. Put why an area exists,
and its lifecycle across files, in the public parent's `//!`. Keep a private
file to a one-line title.
[`tools/check_hidden_module_docs.py`](https://github.com/zspacelabs/bunsen/blob/main/tools/check_hidden_module_docs.py)
lists the modules that break the rule, and CI runs it on every PR. STYLE.md's
[Module docs must render](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#module-docs-must-render)
has the details, including when `#[doc(inline)]` matters.

## Linking into the API

API links in the book are rustdoc intra-doc paths, not URLs:

| To link | Write |
|---|---|
| an item | ``[`ShapeContract`](bunsen::contracts::ShapeContract)`` |
| a module | ``[`bunsen::ops`]`` |
| a section of a page | ``[the ops/blocks rule](bunsen::ops#ops-and-blocks)`` |
| a method | ``[`try_init`](bunsen::burner::module::ModuleInit::try_init)`` |
| a macro | ``[`shape_contract!`](bunsen::contracts#macros-and-methods)``, the section of its module's page that lists it (see below) |
| another published crate | ``[`bunsen_firehose`]`` |

Macros are the one exception to linking the item itself. rustdoc can't
resolve a path to a `macro_rules!` macro that another crate re-exports
(`bunsen::contracts::shape_contract!` reports "no item named `bunsen` in
scope"), so a book link to a bunsen macro goes to the module section that
documents it.

`mdbook-rustdoc-links` resolves them at build time with rustdoc's own
resolver, so a link in the book is exactly as valid as the same link in a
doc comment. Renaming or moving an item, method or field leaves the link
unresolved, and that fails the book build in CI, in the same PR as the
rename.

What resolves is what docs.rs shows. `bunsen` is built with the feature
list in its `[package.metadata.docs.rs]`, and the other published crates
with their defaults, so an item behind a feature that docs.rs doesn't build
can't be linked. The `crates/dev` and `crates/validation` crates and the
examples are not in that build. Link them, and other repository files, as
GitHub URLs, and keep those few.

The base URL the links point at is chosen per build:

- **Local builds** link to `/api`, a symlink from `book/src/api` to
  `target/doc`, and include the API docs in the book's output, so a local
  book reads against the API at HEAD, offline.
- **Release builds**, selected when `CI` is set, link to docs.rs, pinned to
  each crate's version from `Cargo.lock`. That is why a book for readers is
  built only from a release ([Releasing](./release.md)).
- `MDBOOK_PREPROCESSOR__RUSTDOC_LINKS__BASE_URL` overrides both.

Links between chapters are relative `.md` paths. `mdbook-linkcheck2` checks
that the file exists, but not the `#anchor`.

Square brackets in prose are read as links, and an unresolved one fails the
build. Escape a stray bracket (`\[x\]`) or put it in a code span.

## Code in the book

Every Rust block in the book is compiled, or it is plainly a sketch:

- Runnable code that names bunsen items is pulled in from compiled code
  with an mdbook include, `\{{#include path:anchor}}`, from
  [`crates/dev/book-snippets`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/book-snippets)
  or an example. The included region sits between `// ANCHOR: name` and
  `// ANCHOR_END: name` in a test or a `main`, so `cargo test` runs it, and
  CI builds, lints and formats it like any other workspace member. Prefer
  including code an example already has over writing a new snippet.
- A sketch that names no bunsen items may use a `text` or `rust,ignore`
  fence, and the prose around it says it is a sketch.
- The book never copies a rustdoc example. It links to the doctest.

## Shapes, math and diagrams

- **Tensor shapes** are one code span in square brackets, with the
  dimension arithmetic inside, such as `[batch, h_wins*size, w_wins*size, channels]`.
  This is STYLE.md's
  [shape notation](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#tensor-shape-notation),
  the same in rustdoc and the book, and it reads like the
  [shape contract](../systems/contracts.md) that checks the shape. The code
  span also keeps the brackets away from the link resolver.
- **Math** is KaTeX, between dollar signs. Book-wide macros, such as `\RR`,
  `\NN` and `\ZZ` for the number sets, are defined in
  [`book/katex-macros.txt`](https://github.com/zspacelabs/bunsen/blob/main/book/katex-macros.txt).
  Add a macro there rather than repeating a definition on a page.
- **Diagrams** are mermaid, or `text` fences. Prefer a `text` diagram when
  the same picture may belong in rustdoc.

## Building the book

The book's tasks are in `Makefile.toml`. Each installs the pinned mdbook
tools on first use, at the versions CI uses:

| Task | What it does |
|---|---|
| `cargo make book` | builds the book against the local API docs; an unresolved API link is a warning |
| `cargo make book-check` | the same build CI runs: an unresolved API link is an error |
| `cargo make book-serve` | serves the book locally, rebuilding as you edit |
| `cargo make book-release` | builds the book for publishing, with API links pinned to this release on docs.rs |

Run `book-check` before pushing a change that touches the book or renames
anything public. The build runs `cargo doc` itself, so the first run is
slow. `book-release` is part of a release, not of everyday work
([Releasing](./release.md)).

## Writing a page

Open each page with a sentence or two on what it covers. Say what a thing is
for and when to reach for it, and stop when the *why* is told: one good
paragraph and a link beat a list of features. Every API noun is a link.
