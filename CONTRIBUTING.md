# CONTRIBUTING.md — wordchipper

## Development Setup

You need stable Rust (MSRV 1.93.0) and nightly `rustfmt`:

```sh
rustup toolchain install stable nightly
rustup component add --toolchain nightly rustfmt miri
```

## Workflow

1. Format: `cargo +nightly fmt`
2. Lint: `cargo clippy --no-deps` (fix all warnings — they're treated as errors in CI)
3. Test: `cargo test --workspace`
4. Commit.

