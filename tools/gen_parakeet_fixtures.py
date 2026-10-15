#!/usr/bin/env python3
"""Generate the Parakeet ONNX-reference fixture.

Run once; commit the output. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install onnxruntime soundfile numpy
    /tmp/pkvenv/bin/python tools/gen_parakeet_fixtures.py ONNX_DIR

`ONNX_DIR` holds `istupakov/parakeet-tdt-0.6b-v3-onnx` at revision
8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce (`nemo128.onnx`, `encoder-model.onnx`
and its `.data`, `decoder_joint-model.onnx`): the files
`bunsen-bundled-parakeet` fetches.

Pinned versions used to produce the committed fixture:
    onnxruntime 1.31.0, numpy 2.4.6, soundfile 0.13.1 (libsndfile 1.2.2)

Everything runs on the CPU through onnxruntime, the export's own runtime, so
the fixture says what the graphs compute independently of burn.

One output, `crates/validation/parakeet-model-validation/testdata/onnx_reference.json`:

`preprocessor`, `encoder`, `decoder_joint`
    Each graph run once on deterministic synthetic input that the Rust side
    rebuilds from the same formula (see that crate's `synthetic_*`). Each
    output tensor is recorded as its `shape`, `sum`, `abs_sum`, and a
    `stride`-subsampled `samples` of its row-major flattening.
`speech`
    A greedy TDT decode of `whisper-model-validation`'s `jfk_moon.mp3`
    (60 s; the speech fixtures are shared, not copied): token `ids` (blanks
    dropped) and the encoder frame each was emitted at.
"""

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/validation/parakeet-model-validation/testdata/onnx_reference.json"
CLIP = ROOT / "crates/validation/whisper-model-validation/testdata/jfk_moon.mp3"

N_MELS = 128
D_ENC = 1024
D_PRED = 640
BLANK = 8192
N_DURATIONS = 5
MAX_SYMBOLS = 10

# Synthetic input geometry; the Rust side uses the same constants.
N_SAMPLES = 32000
N_FRAMES = 400
TARGETS = [BLANK, 0, 506, 8191]

# About this many samples per recorded tensor.
N_RECORDED = 2048


def synthetic_waveform() -> np.ndarray:
    """`[1, N_SAMPLES]`: two tones and a deterministic ramp of noise."""
    n = np.arange(N_SAMPLES, dtype=np.float64)
    x = (
        0.30 * np.sin(2 * np.pi * 440.0 * n / 16000.0)
        + 0.15 * np.sin(2 * np.pi * 1234.5 * n / 16000.0)
        + 0.05 * (((n * 7919) % 1009) / 504.5 - 1.0)
    )
    return x.astype(np.float32)[None]


def synthetic_features() -> np.ndarray:
    """`[1, N_MELS, N_FRAMES]`, the formula `whisper-model-validation` uses."""
    m = np.arange(N_MELS)[:, None]
    f = np.arange(N_FRAMES)[None, :]
    return (((m * 7 + f * 13) % 211) / 105.0 - 1.0).astype(np.float32)[None]


def synthetic_encoder_frame() -> np.ndarray:
    """`[1, D_ENC, 1]`."""
    c = np.arange(D_ENC)
    return ((((c * 5) % 197) / 98.0 - 1.0) * 0.5).astype(np.float32)[None, :, None]


def synthetic_state(salt: int) -> np.ndarray:
    """`[2, 1, D_PRED]`, small and non-zero, so the state path is exercised."""
    k = np.arange(2 * D_PRED).reshape(2, 1, D_PRED)
    return ((((k * 3 + salt) % 101) / 50.0 - 1.0) * 0.25).astype(np.float32)


def record(x: np.ndarray) -> dict:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    stride = max(1, flat.size // N_RECORDED)
    return {
        "shape": list(x.shape),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "stride": stride,
        "samples": [float(v) for v in flat[::stride]],
    }


def main(onnx_dir: pathlib.Path) -> None:
    pre = ort.InferenceSession(str(onnx_dir / "nemo128.onnx"))
    enc = ort.InferenceSession(str(onnx_dir / "encoder-model.onnx"))
    dec = ort.InferenceSession(str(onnx_dir / "decoder_joint-model.onnx"))

    out = {
        "source": "istupakov/parakeet-tdt-0.6b-v3-onnx@8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        "runtime": f"onnxruntime {ort.__version__}",
    }

    wave = synthetic_waveform()
    feats, feats_lens = pre.run(
        None, {"waveforms": wave, "waveforms_lens": np.array([N_SAMPLES], np.int64)}
    )
    out["preprocessor"] = {
        "n_samples": N_SAMPLES,
        "features": record(feats),
        "features_lens": [int(v) for v in feats_lens],
    }

    x = synthetic_features()
    enc_out, enc_lens = enc.run(
        None, {"audio_signal": x, "length": np.array([N_FRAMES], np.int64)}
    )
    out["encoder"] = {
        "n_frames": N_FRAMES,
        "outputs": record(enc_out),
        "encoded_lengths": [int(v) for v in enc_lens],
    }

    frame = synthetic_encoder_frame()
    steps = []
    for i, target in enumerate(TARGETS):
        s1, s2 = synthetic_state(2 * i), synthetic_state(2 * i + 1)
        logits, _, o1, o2 = dec.run(
            None,
            {
                "encoder_outputs": frame,
                "targets": np.array([[target]], np.int32),
                "target_length": np.array([1], np.int32),
                "input_states_1": s1,
                "input_states_2": s2,
            },
        )
        steps.append(
            {
                "target": target,
                "logits": record(logits),
                "token_argmax": int(logits[0, 0, 0, : BLANK + 1].argmax()),
                "duration_argmax": int(logits[0, 0, 0, BLANK + 1 :].argmax()),
                "output_states_1": record(o1),
                "output_states_2": record(o2),
            }
        )
    out["decoder_joint"] = {"steps": steps}

    wave, sr = sf.read(str(CLIP), dtype="float32")
    assert sr == 16000 and wave.ndim == 1, (sr, wave.shape)
    feats, feats_lens = pre.run(
        None,
        {"waveforms": wave[None], "waveforms_lens": np.array([len(wave)], np.int64)},
    )
    enc_out, enc_lens = enc.run(None, {"audio_signal": feats, "length": feats_lens})
    ids, frames = greedy_tdt(dec, enc_out, int(enc_lens[0]))
    out["speech"] = {
        "clip": "jfk_moon.mp3",
        "n_samples": int(len(wave)),
        "encoded_length": int(enc_lens[0]),
        "ids": ids,
        "frames": frames,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}: {len(ids)} ids {ids}")


def greedy_tdt(dec, enc_out: np.ndarray, length: int) -> tuple[list, list]:
    """Greedy TDT, as NeMo and transformers decode it.

    At each step the joint scores (frame t, last emitted token). A token other
    than blank is emitted and advances the prediction network; the duration
    argmax advances t, and a blank with duration 0 advances it by 1. At most
    `MAX_SYMBOLS` tokens are emitted at one frame before t is forced on.
    """
    s1 = np.zeros((2, 1, D_PRED), np.float32)
    s2 = np.zeros((2, 1, D_PRED), np.float32)
    t, last, at_frame = 0, BLANK, 0
    ids, frames = [], []
    while t < length:
        logits, _, n1, n2 = dec.run(
            None,
            {
                "encoder_outputs": enc_out[:, :, t : t + 1],
                "targets": np.array([[last]], np.int32),
                "target_length": np.array([1], np.int32),
                "input_states_1": s1,
                "input_states_2": s2,
            },
        )
        row = logits[0, 0, 0]
        token = int(row[: BLANK + 1].argmax())
        duration = int(row[BLANK + 1 :].argmax())
        if token != BLANK:
            ids.append(token)
            frames.append(t)
            last, s1, s2 = token, n1, n2
            at_frame += 1
        if token == BLANK and duration == 0:
            duration = 1
        if duration == 0 and at_frame >= MAX_SYMBOLS:
            duration = 1
        if duration > 0:
            at_frame = 0
        t += duration
    return ids, frames


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]))
