# Development Crates

This directory contains development and testing crates that are not published to crates.io.

| Crate | What it is |
|---|---|
| `book-snippets` | Code the bunsen book includes, compiled and run by `cargo test`, so the book's examples cannot drift from the API. |
| `bunsen-chat` | Chat with bunsen's Qwen3: a binary (`bunsen-chat list`, `fetch`, `run`, `chat`) and the library under it: the checkpoint's tokenizer (on `wordchipper`) and chat template (on `minijinja`), and `ChatSession`, a conversation held on the KV cache, each held to `transformers` by `qwen3-model-validation`. |
| `bunsen-test-audio` | The audio the workspace's tests, validation suites and fixture scripts read, with its provenance and ground truth: the `jfk` clips, tones, Silero's test clip, and, with the `ami` feature, an AMI meeting excerpt and its RTTM, fetched and pinned. A leaf: it does not depend on bunsen. |
| `burn_bug_repro` | Reproductions of defects in burn and its ecosystem, and pins on the current behaviour. |
| `kv-cache-bench` | A binary that times decode-shaped KV cache growth under each `KVCacheStorage` (and `Auto`), interleaving repetitions and reporting medians and memory; how `KVCacheStorage::auto_for`'s table is measured and retuned. |
| `silero-bench` | A binary that times the Silero voice-activity detector over a WAV file, and checks its probabilities against an expected JSON array when given one. |
