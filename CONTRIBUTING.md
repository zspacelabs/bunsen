# Contributing to bunsen

This is the canonical contributor guide: what a checkout needs, the checks a pull request must pass, how to write
commit messages, and how a release (including the book) is published. The book's
[Development strategies](https://zspacelabs.ai/bunsen/book/development/setup.html) part explains why the project works
this way; the steps are here.

For a small fix, open a pull request. For anything larger, such as a new component, a new kit or a breaking API change,
open an [issue](https://github.com/zspacelabs/bunsen/issues) first. Code and documentation follow
[STYLE.md](STYLE.md).

## Development setup

- **Stable Rust**, at least the workspace's `rust-version` (1.94.1, in the root [`Cargo.toml`](Cargo.toml)).
  [`rust-toolchain.toml`](rust-toolchain.toml) selects the stable channel, with `rustfmt` and `clippy`.
- **Nightly `rustfmt`.** [`rustfmt.toml`](rustfmt.toml) uses unstable options, so format with `cargo +nightly fmt`.
  Stable `rustfmt` ignores those options and reformats the whole tree.
- **ALSA headers, on Linux:** `libasound2-dev` on Debian and Ubuntu (`alsa-lib-devel` or `alsa-lib` elsewhere).
  `whisper-cli live` captures audio through `cpal`, which links ALSA, and clippy and the tests build every workspace
  member.
- **[cargo-make](https://github.com/sagiegurari/cargo-make)**, which runs the tasks in [`Makefile.toml`](Makefile.toml).
- **The mdbook tools**, for the book: `cargo make setup` installs mdbook and its plugins at the versions CI uses.
- `python3` runs the repository checks in [`tools/`](tools), and `jq` is used by `cargo make doc`.

```sh
rustup toolchain install stable
rustup toolchain install nightly --profile minimal --component rustfmt
sudo apt-get install libasound2-dev   # Debian and Ubuntu
cargo install cargo-make
cargo make setup
```

## Before you push

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) is the merge bar. Locally, the core of it is:

```sh
cargo +nightly fmt                                  # format
cargo clippy --no-deps -- -D warnings               # lint; warnings are errors
python3 tools/check_hidden_module_docs.py crates    # module docs must render
cargo test --workspace                              # the tests, on the CPU
```

A plain `cargo test` runs on the CPU: bunsen's tests compute on its `PerformanceBackend`, which needs a backend
feature to leave the CPU. Test a change that touches tensor code with one too, such as
`cargo test -p bunsen --features wgpu`; the book's
[Testing and backends](https://zspacelabs.ai/bunsen/book/development/testing.html) explains why. Run
`cargo make book-check` when you change the book or rename anything public ([Working on the book](#working-on-the-book)).

The IDE run configurations in `.idea/runConfigurations/` chain clippy-fix, format and doc lint, then the workspace
tests, with or without a backend; STYLE.md's
[IDE run configurations](STYLE.md#ide-run-configurations) chapter describes them.

### What CI runs

Four jobs, each step commented in the workflow:

- **Format:** `cargo +nightly fmt -- --check`.
- **Clippy:** the hidden-module-docs check, then `cargo clippy --no-deps -- -D warnings` over the whole workspace.
- **Book:** `mdbook build book` with `CI` set, so every API link in the book must resolve.
- **CI:** the API docs three ways (the workspace with `--all-features`; `bunsen` alone with the feature list docs.rs
  uses, which `cargo make doc-docsrs` reproduces; `bunsen` with its default features), then
  `cargo test --features download,indicatif` over the workspace, `cargo test -p bunsen --no-default-features`,
  `cargo check -p bunsen --no-default-features` with no features, then with `--features cache` and with
  `--features train`, and one `cargo check --manifest-path` per published crate.

The repeated builds are on purpose: Cargo unifies features across the crates in one command, so a crate that forgets
a feature still builds in the workspace and fails for its users. The book's
[Development setup and CI](https://zspacelabs.ai/bunsen/book/development/setup.html) explains each configuration. The
`download` feature lets build scripts fetch assets (about 435 MB for Whisper's), so a cold run of that test step takes
a while.

## Commit messages

Releases are automated by [release-plz](https://release-plz.dev), which builds changelogs from the commit history, so
commit subjects (and squash-merge PR titles, which become the commit subject) should follow
[Conventional Commits](https://www.conventionalcommits.org). A scope, naming the crate or area, is optional:

```
feat: add Whisper text decoder
fix(bunsen): correct DropBlock mask scaling
docs(book): document the firehose pipeline
```

The prefix decides both the changelog section and the version bump:

- `feat:` → **Added** section, **minor** bump.
- `fix:` → **Fixed** section, **patch** bump.
- `feat!:` (or a `BREAKING CHANGE:` footer) → **Breaking** section, **major** bump.
- `docs:`, `refactor:`, `chore:`, … → other/hidden, **patch** bump.

release-plz also runs `cargo-semver-checks`, so an API break is promoted to a major bump even if the prefix doesn't mark
it as breaking.

## Release

Releases are automated by **release-plz**: you never bump versions, edit crate changelogs, or run `cargo publish` by
hand. Configuration lives in [`release-plz.toml`](release-plz.toml); the automation runs from
[`.github/workflows/release-plz.yml`](.github/workflows/release-plz.yml). The one manual step is publishing the book.

### What gets released

The seven published crates, the ones in `crates/public/`, are versioned in **lockstep**: they share one version
(`version_group = "bunsen"` in `release-plz.toml`, matching `version.workspace = true` in the root `Cargo.toml`):

- `bunsen`
- `bunsen-contracts-macros`
- `bunsen-firehose`
- `bunsen-firehose-image`
- `bunsen-arrow-dataloaders`
- `bunsen-bundled-silero`
- `bunsen-bundled-whisper`

Nothing else is released: release-plz is opt-in (`release = false` for the workspace, `release = true` per crate), so
the examples and the `crates/dev` and `crates/validation` crates stay out.

### How a release happens

1. Merge your changes to `main` as usual (with Conventional Commit messages).
2. release-plz opens or updates a **release PR** titled like `chore: release vX.Y.Z`. It bumps the workspace version
   and regenerates each crate's `CHANGELOG.md` from the commits since the last release. This PR accumulates and
   rebases itself as more changes land on `main`.
3. When you're ready to cut a release, review and **merge the release PR**.
4. On merge, release-plz tags the release (`bunsen-vX.Y.Z`, and one tag per crate), publishes the crates to
   crates.io, and creates a GitHub release with the changelog notes.
5. **Publish the book**, by hand ([below](#publishing-the-book)).

To hold back a release, just don't merge the release PR yet; it keeps updating itself until you do.

### Publishing the book

The published book is built from the release, with every API link pinned to the released version on docs.rs, so
publish it only after the release PR has merged and [docs.rs](https://docs.rs/crate/bunsen/latest) has built the new
version (usually a few minutes). Before that, the pinned links are dead.

1. Check out the release commit (the merged release PR, tagged `bunsen-vX.Y.Z`).
2. Run `cargo make book-release`. It builds as CI does, so an unresolved API link is an error, points the API links at
   docs.rs at the versions in `Cargo.lock`, and sets the site URL to `/bunsen/book/`. The output is `book/book/html`.
3. In a checkout of [`zspacelabs/zspacelabs.github.io`](https://github.com/zspacelabs/zspacelabs.github.io), replace
   the contents of `bunsen/book/` with the contents of `book/book/html`, then commit and push to `main`. GitHub Pages
   serves it at <https://zspacelabs.ai/bunsen/book/>.

CI builds and checks the book on every pull request; only this step publishes it.

### One-time setup

These must be configured on the GitHub repository before the workflow can run:

- **`CARGO_REGISTRY_TOKEN`**: a [crates.io API token](https://crates.io/settings/tokens) with the `publish-new` and
  `publish-update` scopes for the `bunsen*` crates, added under *Settings → Secrets and variables → Actions*. Give it a
  deliberate expiry. An expired token fails the release with "403 Forbidden: authentication failed"; rotate it with
  `gh secret set CARGO_REGISTRY_TOKEN` and re-run the failed job with `gh run rerun <run-id> --failed`.
- **Allow Actions to open PRs**: *Settings → Actions → General → Workflow permissions* → enable "Allow GitHub Actions
  to create and approve pull requests".
- **`RELEASE_PLZ_WORKFLOW`** (optional): a personal access token or GitHub App token. PRs opened with the built-in
  `GITHUB_TOKEN` don't trigger other workflows, so without it `ci.yml` does not run on the release PR. The workflow
  already uses this secret when it is set and falls back to `GITHUB_TOKEN`; see the comments in `release-plz.yml` for
  the token's scopes.

### Recovering a partial release

If a release dies part-way, with some crates published and the rest not, an ordinary PR does not resume it. Merge any
PR from a branch named `release-plz-*` (squash merges are fine), and release-plz publishes whatever is still missing at
the manifest version. The `release_always` comment in `release-plz.toml` explains why.

### Previewing locally

With the [`release-plz` CLI](https://release-plz.dev/docs/usage/installation)
installed you can dry-run what the next release PR would contain, without touching the remote:

```sh
release-plz update          # show the version bumps + changelog edits it would make
```

> Note: the per-crate `CHANGELOG.md` files generated by release-plz supersede
> the hand-written root [`CHANGELOG.md`](CHANGELOG.md), which is kept as a
> historical record of releases up to the adoption of release-plz.

## Working on the book

The book's source is in [`book/src`](book/src). Its contributor's guide is
[Writing documentation](book/src/development/docs.md): what belongs in the book and what in rustdoc, how the book
links into the API, and how code in the book stays compiled. The tasks, from `Makefile.toml`:

| Task                      | What it does                                                                          |
|---------------------------|---------------------------------------------------------------------------------------|
| `cargo make book`         | builds the book against the local API docs; an unresolved API link is a warning       |
| `cargo make book-check`   | the build CI runs: an unresolved API link is an error                                 |
| `cargo make book-serve`   | serves the book locally, rebuilding as you edit                                       |
| `cargo make book-release` | the publishing build, run once per release ([Publishing the book](#publishing-the-book)) |

Each task installs the pinned mdbook tools on first use. The build runs `cargo doc` itself, so the first run is slow.
