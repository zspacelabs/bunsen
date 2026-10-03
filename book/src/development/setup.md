# Development setup and CI

This page explains what a working checkout needs, what CI checks on every
pull request, and why it checks the same code in several configurations.
The steps themselves, what to install and which commands to run, are in
[`CONTRIBUTING.md`](https://github.com/zspacelabs/bunsen/blob/main/CONTRIBUTING.md),
the canonical contributor guide. This page doesn't repeat them.

## What a checkout needs

- **Stable Rust** builds and tests everything.
- **Nightly `rustfmt`** for formatting. The repository's `rustfmt.toml`
  uses unstable options; stable `rustfmt` silently ignores them and
  reformats the whole tree.
- **ALSA headers** on Linux. whisper-cli's `live` command captures audio
  through `cpal`, whose Linux backend links ALSA, and clippy and the tests
  build every workspace member.
- **cargo-make**, which runs the project's tasks, including the book's
  ([Writing documentation](./docs.md)).

These are the cost of building the whole workspace. A crate that depends on
bunsen needs none of them.

## What CI checks

The [workflow](https://github.com/zspacelabs/bunsen/blob/main/.github/workflows/ci.yml)
runs four jobs, and comments each step:

| Job | Checks |
|---|---|
| Format | nightly `rustfmt`, in check mode. |
| Clippy | the whole workspace, with warnings as errors. |
| Book | the book builds, and every API link in it resolves ([Writing documentation](./docs.md)). |
| CI | the API docs, the tests, and the feature configurations below. |

CI has no accelerator and passes no backend feature, so its tests run on
the CPU ([Testing and backends](./testing.md)).

## Why several configurations

Cargo unifies features. When one command builds several crates, each
dependency is built once, with every feature that any of them asked for. A
crate that forgot to enable a feature it needs still builds, because a
sibling enabled it, and the failure waits for a user who depends on that
crate alone. This happened: `bunsen-firehose` relied on `bunsen` turning on
`burn/std`, built cleanly in the workspace, and failed to publish at
v0.31.0.

So the test job builds the code several ways, each standing in for a real
consumer:

- **The workspace, with `download` and `indicatif`.** Everything, including
  the build-time asset fetches ([Assets](./assets.md)) and the run-time
  network code, with the tests.
- **`bunsen` with no default features.** A user who writes
  `default-features = false` and adds back only what they use.
- **`bunsen` with `cache` but not `fetch`.** The disk cache must compile
  without the network code.
- **Each published crate on its own.** One `cargo check` per manifest, which
  is what `cargo publish` does. Several `-p` flags in one command would
  unify again.

The API docs get the same treatment, because a doc build unifies features
too: an intra-doc link can resolve in the workspace build and break for
`bunsen` alone, which is how docs.rs builds it. CI builds the workspace
docs, then `bunsen` alone with the feature list docs.rs uses, then `bunsen`
with its default features. A broken link then fails the PR, not the docs.rs
build after a release.

## Running the checks locally

The IDE run configurations in `.idea/runConfigurations/` are the local form
of the merge bar: a chain of clippy-fix, format and doc lint, then the
workspace tests, with or without a backend. STYLE.md's
[IDE run configurations](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#ide-run-configurations)
chapter describes the tiers and the rules for adding one. Without the IDE,
`CONTRIBUTING.md` lists the commands.

Neither runs every configuration above. A change to a feature list, or to a
published crate's dependencies, is worth checking in isolation before you
push.

## Commits

Commit subjects, and the titles of squash-merged PRs, follow
[Conventional Commits](https://www.conventionalcommits.org). This is not
style for its own sake: release-plz reads the prefixes to write each
crate's changelog and to choose the version bump ([Releasing](./release.md)).
`CONTRIBUTING.md` has the
[prefixes and what each one does](https://github.com/zspacelabs/bunsen/blob/main/CONTRIBUTING.md#commit-messages).

## Where to talk

For a small fix, open a pull request. For anything larger, such as a new
component, a new kit or a breaking API change, open an
[issue](https://github.com/zspacelabs/bunsen/issues) first. The project's
Discord is linked from the
[README](https://github.com/zspacelabs/bunsen/blob/main/README.md).
