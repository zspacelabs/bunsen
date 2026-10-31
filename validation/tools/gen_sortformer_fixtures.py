#!/usr/bin/env python3
"""Generate the Sortformer ONNX-reference fixture.

Run once; commit the output. Regenerating is not part of CI.

    python3 -m venv /tmp/sfvenv
    /tmp/sfvenv/bin/pip install onnxruntime numpy
    /tmp/sfvenv/bin/python validation/tools/gen_sortformer_fixtures.py NEMOTRON_ONNX V2_1_ONNX

`NEMOTRON_ONNX` is `altunenes/parakeet-rs`'s
`nemotron-3-diarization/nemotron3_diar_v3.onnx` at revision
4d2a8bc71f5c896ec40faa59732e6716295edaf2 (sha256 915e4fa2...): the file
`bind-sortformer-onnx` fetches, before its build rewrites it for
`burn-onnx`. `V2_1_ONNX` is `Scrybl/diar_streaming_sortformer_4spk-v2.1`'s
`diar_streaming_sortformer_4spk-v2.1.onnx` at revision
3be92087de17e4e9063c9cfabc104f9809b2d037 (sha256 0182539a...), which the
crate imports as published.

Pinned versions used to produce the committed fixture:
    onnxruntime 1.31.0, numpy 2.5.4

Everything runs on the CPU through onnxruntime, the export's own runtime,
with graph optimizations off, so the fixture says what the published graph
computes independently of burn and of the rewrite.

One output, `validation/models/sortformer-model-validation/testdata/onnx_reference.json`:

`nemotron`
    The streaming step run on deterministic synthetic input that the Rust side
    rebuilds from the same formulas (that crate's `synthetic_*`), at two
    geometries. The speaker cache and FIFO are partly valid (their lengths
    below their allocated frames), so the graph's concat-and-pad path runs.
    Each output tensor is recorded as its `shape`, `sum`, `abs_sum`, and a
    `stride`-subsampled `samples` of its row-major flattening.
`v2_1`
    The v2.1 step on the same inputs, at the same geometries and a third with
    an empty speaker cache and FIFO (a stream's first step): its 80 ms
    predictions and the chunk's pre-encode embeddings.
"""

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/sortformer-model-validation/testdata/onnx_reference.json"

N_MELS = 128
D_ENC = 512

# (mel frames, cache frames, cache valid, fifo frames, fifo valid); the Rust
# side uses the same constants.
GEOMETRIES = [(112, 24, 20, 16, 10), (64, 8, 8, 4, 4)]
V2_1_GEOMETRIES = GEOMETRIES + [(64, 0, 0, 0, 0)]

# About this many samples per recorded tensor.
N_RECORDED = 2048


def synthetic_chunk(frames: int) -> np.ndarray:
    """`[1, frames, N_MELS]`: a tonal pattern that moves through time, and a
    deterministic ramp of noise."""
    t = np.arange(frames, dtype=np.float64)[:, None]
    f = np.arange(N_MELS, dtype=np.float64)[None, :]
    noise = ((t * 31 + f * 17) % 97) / 48.5 - 1.0
    x = 3.0 * np.sin(2 * np.pi * (f / 16.0 + t / 30.0)) + 0.3 * noise
    return x.astype(np.float32)[None]


def synthetic_embeddings(frames: int, salt: int) -> np.ndarray:
    """`[1, frames, D_ENC]`, embedding-sized values; `salt` varies them."""
    r = np.arange(frames, dtype=np.float64)[:, None]
    c = np.arange(D_ENC, dtype=np.float64)[None, :]
    noise = ((r * 13 + c * 7 + salt * 29) % 89) / 44.5 - 1.0
    x = 2.0 * np.sin(2 * np.pi * (c / 37.0 + r / 11.0 + salt / 5.0)) + 0.5 * noise
    return x.astype(np.float32)[None]


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


def session(onnx: pathlib.Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])


def inputs(mels: int, cache: int, cache_valid: int, fifo: int, fifo_valid: int) -> dict:
    i64 = lambda v: np.array([v], np.int64)
    return {
        "chunk": synthetic_chunk(mels),
        "chunk_lengths": i64(mels),
        "spkcache": synthetic_embeddings(cache, 1),
        "spkcache_lengths": i64(cache_valid),
        "fifo": synthetic_embeddings(fifo, 2),
        "fifo_lengths": i64(fifo_valid),
    }


def geometry(mels: int, cache: int, cache_valid: int, fifo: int, fifo_valid: int) -> dict:
    return {
        "mel_frames": mels,
        "spkcache_frames": cache,
        "spkcache_valid": cache_valid,
        "fifo_frames": fifo,
        "fifo_valid": fifo_valid,
    }


def v2_1_steps(onnx: pathlib.Path) -> list:
    graph = session(onnx)
    steps = []
    for g in V2_1_GEOMETRIES:
        preds, embs, emb_lens = graph.run(None, inputs(*g))
        steps.append(geometry(*g) | {
            "preds": record(preds),
            "chunk_pre_encode_embs": record(embs),
            "chunk_pre_encode_lengths": [int(v) for v in emb_lens],
        })
        print(f"v2.1 mels {g[0]} cache {g[1]}: preds {preds.shape} max {preds.max():.3f}")
    return steps


def main(onnx: pathlib.Path, v2_1: pathlib.Path) -> None:
    session_ = session(onnx)

    steps = []
    for mels, cache, cache_valid, fifo, fifo_valid in GEOMETRIES:
        preds, hires, embs, emb_lens = session_.run(
            None, inputs(mels, cache, cache_valid, fifo, fifo_valid)
        )
        steps.append({
            "mel_frames": mels,
            "spkcache_frames": cache,
            "spkcache_valid": cache_valid,
            "fifo_frames": fifo,
            "fifo_valid": fifo_valid,
            "preds_diar": record(preds),
            "preds_hires": record(hires),
            "chunk_pre_encode_embs": record(embs),
            "chunk_pre_encode_lengths": [int(v) for v in emb_lens],
        })
        print(f"mels {mels}: preds {preds.shape} max {preds.max():.3f}, hires {hires.shape}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "altunenes/parakeet-rs@4d2a8bc71f5c896ec40faa59732e6716295edaf2"
                  "/nemotron-3-diarization/nemotron3_diar_v3.onnx",
        "runtime": f"onnxruntime {ort.__version__}",
        "nemotron": steps,
        "v2_1_source": "Scrybl/diar_streaming_sortformer_4spk-v2.1@3be92087de17e4e9063c9cfabc104f9809b2d037"
                       "/diar_streaming_sortformer_4spk-v2.1.onnx",
        "v2_1": v2_1_steps(v2_1),
    }) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
