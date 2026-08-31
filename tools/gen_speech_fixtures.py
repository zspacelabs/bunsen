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

Two outputs, both under `crates/validation/whisper-model-validation/testdata/`:

`whisper_vocab.bin`
    The decode half of Whisper's multilingual tokenizer: id -> raw bytes.
    Decoding needs no merge table, so this is the whole of what bunsen needs
    to turn token ids into text. Format:

        magic   8 bytes  b"BWVOCAB1"
        count   u32 little-endian
        count x (u8 length, `length` bytes)

    Byte-level BPE is already undone here, so an entry is literal UTF-8 (or
    a fragment of it); a special token is its `<|name|>` spelling.

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
import whisper.tokenizer as wt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/validation/whisper-model-validation/testdata"

SR = 16000
N_SAMPLES = 30 * SR
N_FRAMES = 3000
MODEL = "base"

MAGIC = b"BWVOCAB1"


def write_vocab():
    enc = wt.get_encoding("multilingual")

    blob = bytearray(MAGIC)
    blob += enc.n_vocab.to_bytes(4, "little")

    for i in range(enc.n_vocab):
        raw = enc.decode_single_token_bytes(i)
        if len(raw) > 255:
            raise SystemExit(f"token {i} is {len(raw)} bytes; the format assumes < 256")
        blob.append(len(raw))
        blob += raw

    path = OUT / "whisper_vocab.bin"
    path.write_bytes(bytes(blob))
    print(f"whisper_vocab.bin  {enc.n_vocab} tokens  {path.stat().st_size} B")


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
    write_vocab()
    for clip in sorted(OUT.glob("*.mp3")):
        print(f"\n{clip.name}")
        write_reference(clip)


if __name__ == "__main__":
    main()
