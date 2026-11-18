# bunsen-app

Shared command-line plumbing for [bunsen](https://crates.io/crates/bunsen)'s binaries, built on
[clap](https://crates.io/crates/clap): flag groups a binary flattens into its commands, and the commands they open.

- `cache`: the weights cache's flags (`--cache-dir`, `--offline`), and a `cache` subcommand that shows where the
  cache is, lists what it holds by name, and clears it.
- `catalog`: the build catalogs' commands (`CatalogCommand`): import catalogs in order, then show them as a
  directory view (`tree`), as written (`dump`), an entry or a resolved build (`show`, `resolve`), and run
  selection pipelines with each stage explained (`select`). `bunsen-tool catalog` runs them.
- `device`: `--device` and `--device-index`, resolved into a configured burn `Device` (its default float f32); a
  backend is offered when the binary was built with its feature (`wgpu`, `vulkan`, `metal`, `cuda`); the CPU always is.
- `precision`: `--min-precision`, a widen-only floor under a loaded model's dtypes, and `--init-dtype`, the dtype a
  program creates its own tensors in.
- `logging`: `-v`/`-q`/`--ts` log-level flags.
- `shards`: dataset shard selection and fetch policy.

[`bunsen-tool`](https://crates.io/crates/bunsen-tool) is built on it.
