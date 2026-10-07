# Releasing

Releases are automated by [release-plz](https://release-plz.dev). The
process, and what a maintainer does at each step, is in
[`CONTRIBUTING.md`](https://github.com/zspacelabs/bunsen/blob/main/CONTRIBUTING.md#release),
the canonical guide. This page only says how the pieces fit.

## Versions

Nobody bumps a version or runs `cargo publish` by hand. release-plz keeps a
release PR open against `main`, carrying the next version and the
changelogs it wrote from the commit history, which is why commits follow
Conventional Commits ([Development setup](./setup.md#commits)). Merging that
PR publishes the release. The published crates share one version and move
together.

bunsen's version is its own semantic version, and it is not tied to burn's.
A version bump says something about bunsen's API, and nothing about which
burn release it builds against. release-plz runs `cargo-semver-checks`, so
an API break gets a breaking bump even when its commit didn't say so.

## The book

The release workflow publishes the book after every release, to GitHub
Pages. The build points every API link at docs.rs, pinned to the version
just released, so it is built from the release, not from `main`, and the
pinned links are dead until docs.rs has built that version. CI builds and
checks the book on every pull request, but only a release publishes it.
`CONTRIBUTING.md`'s release section says how to republish by hand, and
[Writing documentation](./docs.md#building-the-book) covers the other book
tasks.
