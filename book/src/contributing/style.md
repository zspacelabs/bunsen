# Style and Conventions

## Rust

- Formatting is enforced by `rustfmt` (nightly toolchain, settings live in
  [`rustfmt.toml`](https://github.com/zspacelabs/bunsen/blob/main/rustfmt.toml)).
  Run `cargo make format` before committing.
- Lints are enforced by `cargo clippy --no-deps` with workspace-level
  `warnings = "deny"`. Run `cargo make clippy`.
- Public items need rustdoc. Module-level docs should explain *why* the
  module exists, not just *what* it contains.
- Errors use `bunsen::errors` types where appropriate; avoid leaking
  `anyhow::Error` across crate boundaries.

## Tests

- Prefer `bunsen::support` helpers for tensor assertions; bare `assert_eq!`
  on floats is almost always wrong.
- New public APIs ship with at least one test.

## Documentation

- The Book uses Markdown with the mermaid and KaTeX preprocessors.
- Prefer prose that explains the *motivation* and links out to API docs for
  the *signatures*. The book is not a substitute for `rustdoc`.
- Cross-reference within the book using relative links so
  `mdbook-linkcheck` can validate them.
