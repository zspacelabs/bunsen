# Development Setup

## Prerequisites

- The Rust toolchain pinned in
  [`rust-toolchain.toml`](https://github.com/zspacelabs/bunsen/blob/main/rust-toolchain.toml).
- [`cargo-make`](https://github.com/sagiegurari/cargo-make) for the project's
  task runner.
- For working on the book: `mdbook`, `mdbook-mermaid`, `mdbook-katex`,
  `mdbook-linkcheck`.

```bash
cargo install cargo-make
cargo install mdbook mdbook-mermaid mdbook-katex mdbook-linkcheck
```

## Common tasks

```bash
# Default: fix + ci (format, clippy, tests).
cargo make

# Targeted tasks:
cargo make test
cargo make clippy
cargo make format

# Book tasks (see "Add cargo-make tasks" in the project Makefile.toml):
cargo make book          # build the book
cargo make book-serve    # build + watch + serve on localhost
cargo make book-check    # run mdbook-linkcheck
```

## Working on the book

The book lives in [`book/`](https://github.com/zspacelabs/bunsen/tree/main/book).
Source files are Markdown under `book/src/`, and the output is written to
`book/book/` (ignored by git).

```bash
cd book
mdbook serve --open
```
