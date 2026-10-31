# Speech fixtures

What `openai-whisper` decodes for the spoken clips this crate's `audio`
module transcribes. The clips themselves, their ground-truth transcripts and
their provenance are in
[`bunsen-test-audio`](../../../../crates/dev/bunsen-test-audio/testdata/README.md):
`jfk_moon.mp3` and `jfk_moon.txt`.

| file | |
|---|---|
| `jfk_moon.reference.json` | what `openai-whisper` decodes for `jfk_moon.mp3`, several ways |

The reference holds four decodes of the clip from the same model, and the
token layout they were made with (`vocab_size`, the `prompt`, `eot`, ...):

| key | decode | for |
|---|---|---|
| `windows` | greedy, no timestamps, fixed 30 s windows | the agreement gate |
| `with_timestamps` | the same, timestamp tokens kept | the stream driver's timestamp phase |
| `beam5` | beam 5, no timestamps | its beam-search phase |
| `transcribe` | `transcribe()` at temperature 0: seek loop, segments with times | its seek and segment phases |

Each is a per-window (or per-segment) `tokens` and `text`; `with_timestamps`
also carries `text_with_timestamps`, the `<|1.02|>` rendering, which pins the
special-token spellings. The fixture-integrity check decodes every one of them
and checks the segment times against the timestamp arithmetic.

Ids become text through the vocabulary `bind-whisper-onnx` fetches
(`multilingual.tiktoken`, behind its `vocab` feature) and bunsen's own
`.tiktoken` parser — the same path a user of the kit takes, so the
fixture-integrity check exercises it too.

## What is asserted

The transcript is the authority. Two gates, both on text, both tunable per
fixture in `src/audio.rs`:

- **`max_wer`** — word error rate of bunsen's transcription against
  `{name}.txt`. This is the accuracy knob. Raise it to accept a weaker model
  or a harder clip; lower it to hold a gain.
- **`max_reference_wer`** — word error rate against `{name}.reference.json`.
  A different question: not "is bunsen accurate" but "does bunsen agree with
  the implementation it was transliterated from".

Nothing asserts token ids. A greedy decode argmaxes over 51865 logits at every
step, so a backend differing in the last few digits can flip a token and
cascade — while the *text* barely moves. Text is also what a transcription is
actually judged on.

### Measured

On `wgpu`, against `whisper-base`:

| | WER |
|---|---|
| bunsen vs transcript | 0.0684 |
| `openai-whisper` vs transcript | 0.0684 |
| ONNX reference vs transcript | 0.0684 |
| bunsen vs `openai-whisper` | 0.0000 |
| bunsen vs ONNX reference | 0.0000 |

Three independent implementations — bunsen, OpenAI's own, and a graph
generated from the `onnx-community/whisper-base` export — decode this clip
token for token identically.

The other reference variants, against the transcript, for orientation: the
`transcribe()` decode scores 0.0513 (its prompt carry helps), the beam-5
windows 0.1282, and the timestamped windows 0.2051 — a fixed window decoded
with timestamps stops at its last timestamp and drops its tail, which is the
loss the seek loop exists to recover. Only the greedy and `transcribe()`
references are held to `max_wer`; the other two are reported.

Of the eight word errors against the transcript, three are the model and two
are the normalizer: `normalize_transcript` lowercases, strips punctuation and
collapses whitespace, but deliberately does **not** reconcile number words
with digits (`thirty-five` vs `35`) or expand contractions (`we are` vs
`we're`). Whisper's own `EnglishTextNormalizer` does; reimplementing it was
not worth it for a threshold set by measurement.

The model's real errors, for reference: `fly` → `why`, `in this decade` →
`and disdicate`, `not` → `that`. `whisper-base` is small.

## Regenerating

`validation/tools/gen_speech_fixtures.py` writes every `*.reference.json`. It needs
`openai-whisper`; the recipe and the pinned
versions are in the script's docstring. Run once, commit the outputs — as with
`tools/gen_mel_fixtures.py`, regenerating is not part of CI.

Running the gated tests needs a checkpoint, which is not in this repository.
This crate's `build.rs` fetches and caches one under the `download` feature:

```sh
cargo test --release -p whisper-model-validation \
  --features download,gpu-tests,wgpu
```

The fixture-integrity checks need no model and run in a plain `cargo test`.

## Provenance

The references are `openai-whisper`'s output, written by
`validation/tools/gen_speech_fixtures.py`; the clips' provenance is in
[`bunsen-test-audio`](../../../../crates/dev/bunsen-test-audio/testdata/README.md#provenance).
