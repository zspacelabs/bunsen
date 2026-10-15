#!/usr/bin/env python3
"""Generate the `.fermion` container fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install numpy safetensors torch
    /tmp/pkvenv/bin/python tools/gen_parakeet_container_fixture.py FERMION_CONTAINER_PY

`FERMION_CONTAINER_PY` is `fermion_container.py` from `FermionResearch/Phonon-2`
(Apache-2.0): the reference reader of the `fermion-five-value-parakeet-v1`
format. This script writes a small container of seeded random tensors in every
record kind (`five_value`, `int6`, `int8`, `fp16`, including a five-value row
width that is not a multiple of 5 and an int6 count that is not a multiple of
4), checks that the reference reader returns exactly what was encoded, and
stores those values.

Pinned versions used to produce the committed fixture:
    numpy 2.4.6, safetensors 0.8.0, torch 2.14.1+cpu

Outputs, under `crates/public/bunsen/testdata/parakeet/`:

`toy.fermion`
    The container.
`toy_fermion.safetensors`
    What the reference reader decodes it to, as float32, under the record
    names it reports (`<module>.weight` for five-value records).
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/parakeet"

FORMAT = "fermion-five-value-parakeet-v1"


def five_value(rng, o: int, i: int) -> tuple[bytes, np.ndarray]:
    """Encode a random five-value `[o, i]` matrix; return (blob, dense)."""
    codes = rng.integers(0, 3, size=(o, i))  # 0 -> -1, 1 -> 0, 2 -> +1
    nz = codes != 1
    is_hi = rng.integers(0, 2, size=(o, i)).astype(bool) & nz
    lo = (rng.random(o) * 0.1 + 0.01).astype(np.float16)
    hi = (rng.random(o) * 0.2 + 0.1).astype(np.float16)

    rb = (i + 4) // 5
    padded = np.full((o, rb * 5), 1, dtype=np.int64)
    padded[:, :i] = codes
    trits = padded.reshape(o, rb, 5)
    packed = sum(trits[:, :, k] * 3**k for k in range(5)).astype(np.uint8)
    bits = np.packbits(is_hi[nz].astype(np.uint8), bitorder="little")
    blob = packed.tobytes() + bits.tobytes() + lo.tobytes() + hi.tobytes()

    sign = codes.astype(np.float32) - 1.0
    mag = np.where(is_hi, hi[:, None].astype(np.float32), lo[:, None].astype(np.float32))
    return blob, sign * mag


def int6(rng, shape) -> tuple[bytes, np.ndarray]:
    o = shape[0]
    total = int(np.prod(shape))
    q = rng.integers(-32, 32, size=total)
    u = (q + 32).astype(np.uint32)
    pad = (-total) % 4
    u = np.concatenate([u, np.zeros(pad, np.uint32)]).reshape(-1, 4)
    packed = u[:, 0] | (u[:, 1] << 6) | (u[:, 2] << 12) | (u[:, 3] << 18)
    body = np.stack([(packed >> s) & 0xFF for s in (0, 8, 16)], axis=1).astype(np.uint8)
    scales = (rng.random(o) * 0.05 + 0.001).astype(np.float16)
    dense = q.reshape(o, -1).astype(np.float32) * scales.astype(np.float32)[:, None]
    return body.tobytes() + scales.tobytes(), dense.reshape(shape)


def int8(rng, shape) -> tuple[bytes, np.ndarray]:
    o = shape[0]
    total = int(np.prod(shape))
    q = rng.integers(-128, 128, size=total).astype(np.int8)
    scales = (rng.random(o) * 0.05 + 0.001).astype(np.float16)
    dense = q.reshape(o, -1).astype(np.float32) * scales.astype(np.float32)[:, None]
    return q.tobytes() + scales.tobytes(), dense.reshape(shape)


def fp16(rng, shape) -> tuple[bytes, np.ndarray]:
    x = rng.standard_normal(shape).astype(np.float16)
    return x.tobytes(), x.astype(np.float32)


def main(reference: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location("fermion_container", reference)
    fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc)

    rng = np.random.default_rng(20261015)
    records = [
        ("encoder.layers.0.feed_forward1.linear1", "five_value", [6, 13], five_value),
        ("encoder.layers.0.conv.pointwise_conv1", "five_value", [4, 10], five_value),
        ("decoder.lstm.weight_ih_l0", "int6", [8, 3], int6),
        ("joint.head.weight", "int6", [5, 3], int6),
        ("decoder.embedding.weight", "int8", [3, 4], int8),
        ("encoder.layers.0.norm_out.weight", "fp16", [7], fp16),
        ("encoder.layers.0.conv.norm.num_batches_tracked", "fp16", [], fp16),
    ]
    index, blobs, expected = [], [], {}
    for name, kind, shape, encode in records:
        if kind == "five_value":
            blob, dense = encode(rng, *shape)
            key = name + ".weight"
        else:
            blob, dense = encode(rng, shape)
            key = name
        index.append({"n": name, "k": kind, "shape": shape, "b": len(blob)})
        blobs.append(blob)
        expected[key] = dense

    header = json.dumps({"format": FORMAT, "index": index}).encode()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "toy.fermion"
    with open(path, "wb") as fh:
        fh.write(len(header).to_bytes(8, "little"))
        fh.write(header)
        for blob in blobs:
            fh.write(blob)

    decoded, _ = fc.read_container(str(path))
    assert decoded.keys() == expected.keys(), (decoded.keys(), expected.keys())
    for key, dense in expected.items():
        got = np.asarray(decoded[key], dtype=np.float32).reshape(dense.shape)
        assert np.array_equal(got, dense), key

    save_file(
        {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in expected.items()},
        OUT / "toy_fermion.safetensors",
    )
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]))
