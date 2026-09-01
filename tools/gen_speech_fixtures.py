#!/usr/bin/env python3
"""Generate the Whisper speech-fixture references.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/melvenv
    /tmp/melvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/melvenv/bin/pip install openai-whisper
    /tmp/melvenv/bin/python tools/gen_speech_fixtures.py

Pinned versions used to produce the committed fixtures:
    openai-whisper 20250625, torch 2.9.1+cpu, numpy 2.5.2

CPU torch is deliberate: the fixture only has to be correct, and a CPU run is
deterministic and a far smaller download than the CUDA wheels.

One output per clip, under `crates/validation/whisper-model-validation/testdata/`:

`{clip}.reference.json`
    What `openai-whisper` itself decodes for each 30 s window of a clip, at
    the same fixed windowing bunsen uses -- NOT `transcribe()`, whose seek
    logic picks its own boundaries and so is not comparable.
"""

import json
import pathlib

import numpy as np
import torch
import whisper

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/validation/whisper-model-validation/testdata"

SR = 16000
N_SAMPLES = 30 * SR
N_FRAMES = 3000
MODEL = "base"


def windows_of(clip):
    """`[n_windows, 80, 3000]` log-mels, the geometry bunsen decodes over."""
    audio = whisper.load_audio(str(clip), sr=SR)

    n_windows = max(1, -(-len(audio) // N_SAMPLES))
    padded = np.zeros(n_windows * N_SAMPLES, dtype=np.float32)
    padded[: len(audio)] = audio

    mel = whisper.audio.log_mel_spectrogram(torch.from_numpy(padded), n_mels=80)
    return [mel[:, w * N_FRAMES : (w + 1) * N_FRAMES] for w in range(n_windows)]


def write_reference(clip):
    model = whisper.load_model(MODEL)
    options = whisper.DecodingOptions(
        task="transcribe",
        language="en",
        without_timestamps=True,
        fp16=False,
        beam_size=None,      # greedy, matching `GreedyDecodeConfig`
        temperature=0.0,
    )

    windows = []
    for w, mel in enumerate(windows_of(clip)):
        result = whisper.decode(model, mel, options)
        # `result.tokens` excludes the prompt and the stop token, which is the
        # same slice `Whisper::decode_window` returns.
        windows.append({"tokens": list(result.tokens), "text": result.text})
        print(f"  window {w}: {len(result.tokens)} tokens  {result.text[:60]!r}")

    payload = {
        "source": f"openai-whisper {MODEL}",
        "decode": "greedy, temperature 0, without_timestamps, fixed 30 s windows",
        "windows": windows,
    }

    path = OUT / f"{clip.stem}.reference.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{path.name}  {len(windows)} window(s)  {path.stat().st_size} B")


def main():
    for clip in sorted(OUT.glob("*.mp3")):
        print(f"\n{clip.name}")
        write_reference(clip)


if __name__ == "__main__":
    main()
