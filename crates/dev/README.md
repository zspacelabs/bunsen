# Development Crates

This directory contains development and testing crates that are not published to crates.io.

| Crate | What it is |
|---|---|
| `book-snippets` | Code the bunsen book includes, compiled and run by `cargo test`, so the book's examples cannot drift from the API. |
| `bunsen-test-audio` | The audio the workspace's tests, validation suites and fixture scripts read, with its provenance and ground truth: the `jfk` clips, tones, Silero's test clip, and, with the `ami` feature, an AMI meeting excerpt and its RTTM, fetched and pinned. A leaf: it does not depend on bunsen. |
| `burn_bug_repro` | Reproductions of defects in burn and its ecosystem, and pins on the current behaviour. |
| `silero-bench` | A binary that times the Silero voice-activity detector over a WAV file, and checks its probabilities against an expected JSON array when given one. |
