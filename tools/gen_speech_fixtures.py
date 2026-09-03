#!/usr/bin/env python3
"""Generate the Whisper speech-fixture references.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/melvenv
    /tmp/melvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/melvenv/bin/pip install openai-whisper
    /tmp/melvenv/bin/python tools/gen_speech_fixtures.py

Pinned versions used to produce the committed fixtures:
    openai-whisper 20250625, torch 2.12.0 (a CUDA wheel, run on the CPU), numpy 2.4.6

Everything runs on the CPU, deliberately: the fixture only has to be correct,
and a CPU run is deterministic.

One output per clip, under `crates/validation/whisper-model-validation/testdata/`:

`{clip}.reference.json`
    What `openai-whisper` itself decodes for the clip, several ways, all from
    the same model. Every later phase of the stream driver finds its oracle
    here rather than inventing one:

    `windows`
        Greedy, temperature 0, without timestamps, at the same fixed 30 s
        windowing bunsen uses -- NOT `transcribe()`, whose seek logic picks
        its own boundaries. The agreement gate.
    `with_timestamps`
        The same decode with timestamp tokens on. `tokens` includes them;
        `text` is `Tokenizer.decode` (timestamps dropped) and
        `text_with_timestamps` is `decode_with_timestamps` (rendered as
        `<|1.02|>`), which pins the special-token spellings as well.
    `beam5`
        Beam 5, patience 1, no length penalty, without timestamps.
    `transcribe`
        `whisper.transcribe()`: the seek loop, prompt carry, and segments with
        times. At temperature 0 only -- the fallback ladder samples, and
        sampling is not reproducible across implementations, so it is not a
        reference for anything. Per-segment `temperature` records that no
        fallback happened.

    Plus the token layout the decodes were made with -- `vocab_size`, the
    `prompt` (the sot sequence), `no_timestamps`, `eot`, `timestamp_begin` --
    so that the vocabulary pairing is in the fixture and checkable.
"""

import json
import pathlib
import re

import numpy as np
import torch
import whisper
import whisper.tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/validation/whisper-model-validation/testdata"

SR = 16000
N_SAMPLES = 30 * SR
N_FRAMES = 3000
MODEL = "base"


def windows_of(audio):
    """`[n_windows, 80, 3000]` log-perceptive_audio, the geometry bunsen decodes over."""
    n_windows = max(1, -(-len(audio) // N_SAMPLES))
    padded = np.zeros(n_windows * N_SAMPLES, dtype=np.float32)
    padded[: len(audio)] = audio

    mel = whisper.audio.log_mel_spectrogram(torch.from_numpy(padded), n_mels=80)
    return [mel[:, w * N_FRAMES: (w + 1) * N_FRAMES] for w in range(n_windows)]


def decode_windows(model, tokenizer, windows, label, **options):
    """One `{tokens, text}` per window, `whisper.decode` with `options`."""
    out = []
    for w, mel in enumerate(windows):
        result = whisper.decode(
            model,
            mel,
            whisper.DecodingOptions(
                task="transcribe", language="en", fp16=False, temperature=0.0, **options
            ),
        )
        # `result.tokens` excludes the prompt and the stop token, which is the
        # same slice `Whisper::decode_window` returns.
        entry = {"tokens": list(result.tokens), "text": result.text}
        if not options.get("without_timestamps", False):
            entry["text_with_timestamps"] = tokenizer.decode_with_timestamps(result.tokens)
        out.append(entry)
        print(f"  {label} window {w}: {len(result.tokens)} tokens  {result.text[:60]!r}")
    return out


SEGMENT_KEYS = (
    "id",
    "seek",
    "start",
    "end",
    "text",
    "tokens",
    "temperature",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
)


def inline_int_arrays(text):
    """Puts every all-integer JSON array on one line; token lists are long."""
    return re.sub(
        r"\[\s+((?:-?\d+,\s+)*-?\d+)\s+\]",
        lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]",
        text,
    )


def write_reference(model, clip):
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )
    audio = whisper.load_audio(str(clip), sr=SR)
    windows = windows_of(audio)

    greedy = decode_windows(model, tokenizer, windows, "greedy", without_timestamps=True)
    with_timestamps = decode_windows(model, tokenizer, windows, "timestamps", without_timestamps=False)
    beam5 = decode_windows(model, tokenizer, windows, "beam5", without_timestamps=True, beam_size=5)

    transcribed = whisper.transcribe(
        model,
        audio,
        language="en",
        task="transcribe",
        fp16=False,
        temperature=0.0,
        beam_size=None,
        best_of=None,
        condition_on_previous_text=True,
        verbose=None,
    )
    segments = [{k: seg[k] for k in SEGMENT_KEYS} for seg in transcribed["segments"]]
    for seg in segments:
        print(f"  transcribe segment {seg['id']}: {seg['start']:.2f}-{seg['end']:.2f}  {seg['text'][:50]!r}")

    payload = {
        "source": f"openai-whisper {MODEL}",
        "vocab_size": model.dims.n_vocab,
        "language": "en",
        "task": "transcribe",
        "prompt": list(tokenizer.sot_sequence),
        "no_timestamps": tokenizer.no_timestamps,
        "eot": tokenizer.eot,
        "timestamp_begin": tokenizer.timestamp_begin,
        "decode": "greedy, temperature 0, without_timestamps, fixed 30 s windows",
        "windows": greedy,
        "with_timestamps": {
            "decode": "greedy, temperature 0, with timestamps, fixed 30 s windows",
            "windows": with_timestamps,
        },
        "beam5": {
            "decode": "beam 5, patience 1, no length penalty, temperature 0, without_timestamps, fixed 30 s windows",
            "windows": beam5,
        },
        "transcribe": {
            "decode": "transcribe(): greedy, temperature 0 only (no fallback ladder), with timestamps, condition_on_previous_text",
            "text": transcribed["text"],
            "segments": segments,
        },
    }

    path = OUT / f"{clip.stem}.reference.json"
    path.write_text(inline_int_arrays(json.dumps(payload, indent=2, ensure_ascii=False)) + "\n")
    print(f"{path.name}  {len(greedy)} window(s)  {len(segments)} segment(s)  {path.stat().st_size} B")


def main():
    model = whisper.load_model(MODEL, device="cpu")
    for clip in sorted(OUT.glob("*.mp3")):
        print(f"\n{clip.name}")
        write_reference(model, clip)


if __name__ == "__main__":
    main()
