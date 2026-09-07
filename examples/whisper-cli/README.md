# whisper-cli example

Transcribes an audio file with the bundled OpenAI Whisper `base` checkpoint and its vocabulary, through bunsen's Whisper
stream driver. The audio is pushed in chunks as a live loop would feed it, and segments are printed with their times as
they become final; under the responsive preset, drafts come first and are marked `~`.

Where [`whisper-dev`](../../crates/dev/whisper-dev) takes a checkpoint and a vocabulary by path and drives the mel and
decode ops by hand, this example takes only the audio: everything else comes from bunsen's features.

## Bunsen features exercised

- `whisper-weights` — `Whisper::load_pretrained` loads the bundled multilingual
  `base` checkpoint (fetched at build time, pinned to a SHA-256, cached), and
  `pretrained::bundled_vocabulary` picks the `.tiktoken` rank file that matches its token layout. From the vocabulary
  come the text, via a
  `wordchipper` detokenizer, and upstream's default suppress list.
- `silero-weights` — `SileroVad::load_16khz_pretrained`, the voice-activity model the `conservative` and `responsive`
  presets gate on.
- `bunsen::kits::speech::whisper::driver` — `WhisperStreamDriverConfig`,
  `WhisperStreamContext`: the stream driver, with its emission presets, the timestamp seek loop, beams, per-stream
  language detection, and the fallback ladder behind flags.

## The backend

The example computes on `bunsen::support::testing::PerformanceBackend`: the backend bunsen's own compute-heavy tests run
on. This crate does not choose it. **bunsen's backend feature** does, at build time, so the flag that picks the backend
for `cargo test` picks it here too, and what the example runs on is what the tests ran on.

| build with                           | backend                     |
|--------------------------------------|-----------------------------|
| `--features bunsen/cuda`             | `burn::backend::Cuda`       |
| `--features bunsen/metal`            | `burn::backend::Metal`      |
| `--features bunsen/wgpu`             | `burn::backend::Wgpu`       |
| `--features bunsen/flex`, or nothing | `burn::backend::Flex` (CPU) |

When several are on, the first of `cuda`, `metal`, `wgpu`, `flex` wins. bunsen's default `testing` feature enables
`flex`, which is why a bare build runs on the CPU. The `dependency/feature` form of `--features` works from any package
in the workspace, so `cargo run -p whisper-cli --features bunsen/wgpu`
from the root and `cargo run --features bunsen/wgpu` from this directory are the same build.

## Running the example

The first build fetches the checkpoint (145 MB) and the two vocabularies into the build cache.

```bash
$ cargo run --release -p whisper-cli --features bunsen/wgpu -- \
   transcribe --timestamps /path/to/clip.wav
```

Options:

- `--chunk-ms` — milliseconds of audio per push (default `1000`).
- `--language` — a Whisper language code; detected from the first window when omitted. `--task` — `transcribe` (default)
  or `translate`, to English.
- `--timestamps` — emit timestamp tokens and split segments on them, seeking to the last closed timestamp as upstream's
  `transcribe()` does.
- `--beam-size` — beams per window (default `1`, greedy).
- `--max-tokens` — cap per window (default `224`).
- `--no-prompt-carry` — do not prompt each window with the transcript so far.
- `--fallback` — climb upstream's temperature ladder when a window's decode fails its thresholds; without it,
  temperature zero alone.
- `--preset` — `offline` (default: whole windows, all final), `conservative`
  (speech regions as well, all final), `responsive` (drafts every 600 ms of speech besides). The last two load the
  bundled VAD.
- `--ids` — print each segment's ids beside its text.

## Benchmarks

### Setting up the SLR45 Dataset

See the [SLR45](https://www.openslr.org/45/) dataset.

```terminaloutput
cd $DATA_DIR
mkdir SLR45
cd SLR45
wget https://openslr.trmal.net/resources/45/ST-AEDS-20180100_1-OS.tgz
tar xzf ST-AEDS-20180100_1-OS.tgz
rm ST-AEDS-20180100_1-OS.tgz
```

### Running a small benchmark

```terminaloutput
$ cargo run --release -p whisper-cli --features bunsen/wgpu -- transcribe  --print-filename $DATA_DIR/SLR45/f0001_us_f0001_0000{1,2,3,4,5}.wav
...
INFO model: 80 n_mels, vocabulary 51865, d_model 512, 6 + 6 layers
INFO language: detected from the first window
INFO  [    0.00 -->     4.68]
f0001_us_f0001_00001.wav        The world needs opportunities for new leaders and new ideas.
INFO  [    0.00 -->     3.08]
f0001_us_f0001_00002.wav        along with all the other references that I had.
INFO  [    0.00 -->     2.64]
f0001_us_f0001_00003.wav        wouldn't have hesitated for a second.
INFO  [    0.00 -->     2.56]
f0001_us_f0001_00004.wav        I would always examine the patient.
INFO  [    0.00 -->     2.52]
f0001_us_f0001_00005.wav        Will we like to play with stuff?
INFO mean sample: 3.1s      
INFO mean decode: 1.5s      
INFO mean sample/decode: 3.77

$ head -5 $DATA_DIR/SLR45/text.txt 
f0001_us_f0001_00001.wav        the world needs opportunities for new leaders and new ideas.
f0001_us_f0001_00002.wav        along with all the other reference that I had
f0001_us_f0001_00003.wav        I wouldn't have hesitated for a second.
f0001_us_f0001_00004.wav        I would always examine the patient.
f0001_us_f0001_00005.wav        Well we like to play with stuff.
```
