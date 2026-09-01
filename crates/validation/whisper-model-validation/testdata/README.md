# Speech fixtures

Spoken audio with ground-truth transcripts, used by
this crate's `audio` module.

Clips are **16 kHz mono mp3**, which is what Whisper consumes and what
`support::audio::load_audio_mono_sr` requires — it decodes, it does not
resample or downmix. The source was not in that form; see *Provenance*.

| file | | |
|---|---|---|
| `jfk_moon.mp3` | 60.0 s | "We choose to go to the Moon", live, with hall noise and applause |
| `jfk_moon.txt` | | **ground truth** — what a person hears |
| `jfk_moon.reference.json` | | what `openai-whisper` decodes, several ways |

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

Ids become text through the vocabulary `bunsen-bundled-whisper` fetches
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

`tools/gen_speech_fixtures.py` writes every `*.reference.json`. It needs
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

Everything here must be redistributable under bunsen's own MIT/Apache-2.0
terms: `testdata/` is included in the published crate, so a fixture under a
copyleft or non-commercial licence would ship with it. A Creative Commons
BY-SA sample was considered for this directory and rejected on those grounds.

### `jfk_moon.mp3`

- Source: [Jfk rice university we choose to go to the moon.ogg](https://commons.wikimedia.org/wiki/File:Jfk_rice_university_we_choose_to_go_to_the_moon.ogg),
  retrieved from Wikimedia Commons on 2026-08-28
- Author: John F. Kennedy, address at Rice University, 1962-09-12
- Credit: John F. Kennedy Presidential Library & Museum
- Licence: public domain (work of the U.S. federal government)
- Transcode: `-ss 503 -t 60 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k`
  (from 44.1 kHz stereo Vorbis) — not bit-identical to the source

The source is the complete 17.7-minute address, ~17 MB. Only the 60 s from
08:23 to 09:23 is kept: the "why climb the highest mountain" series through
the end of the famous passage. That offset is also chosen so the 30 s window
boundary falls on "not because they are easy" — a greedy decode is sensitive
to where a window lands, and this cut decodes cleanly on both sides of it.
