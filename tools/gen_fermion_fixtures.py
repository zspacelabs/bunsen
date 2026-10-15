#!/usr/bin/env python3
"""Generate the `FermionResearch` format fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install numpy safetensors torch "mlx[cpu]"
    /tmp/pkvenv/bin/python tools/gen_fermion_fixtures.py \\
        FERMION_CONTAINER_PY ARTIFACT_PY PACKAGE_RELEASE_BPS_PY

The three arguments are Fermion's own readers and writer, each the reference
for one format (all Apache-2.0):

- `FERMION_CONTAINER_PY`: `fermion_container.py` from `FermionResearch/Phonon-2`,
  the reader of the `fermion-five-value-parakeet-v1` container.
- `ARTIFACT_PY`: `fermion/_speech/_engine_cpu/artifact.py` from the
  `fermion-research` 0.2.11 package, whose `_unpack_symbols` reads the
  `ten-base5-per-24bit-v1` (quint5) decoder codes of Phonon-1.
- `PACKAGE_RELEASE_BPS_PY`: `package_release_bps.py` from `FermionResearch/Phonon-1`,
  the writer of the `phonon-byteplane-tar-zstd-v1` archive.

MLX's own `mlx.core.dequantize` is the reference for the affine records.

Pinned versions used to produce the committed fixtures:
    numpy 2.4.6, safetensors 0.8.0, torch 2.14.1+cpu, mlx 0.32.4, zstd 1.5.7

Outputs, under `crates/public/bunsen/testdata/fermion/`:

`five_value.fermion`, `five_value.safetensors`
    A small container of seeded random tensors in every record kind
    (`five_value`, `int6`, `int8`, `fp16`, including a five-value row width that
    is not a multiple of 5 and an int6 count that is not a multiple of 4), and
    what the reference reader decodes it to, as float32, under the record names
    it reports (`<module>.weight` for five-value records).
`quint5.safetensors`
    Two quint5 records (`<m>.quint5_q`, `<m>.base_alpha`, `<m>.residual_scale`,
    one with a width that is not a multiple of 10) and `<m>.dense`, the f32
    weight their symbols select.
`mlx_affine.safetensors`
    MLX affine records at 4, 6 and 8 bits, groups of 64 and 128
    (`<m>.weight` u32, `<m>.scales` / `<m>.biases` bf16) and `<m>.dense`,
    `scales * q + biases` evaluated in f32.
`archive.bps.tar.zst`, `archive.json`
    A two-member archive written with the reference writer (one safetensors
    member stored byte-plane split, one JSON member stored as is), and each
    member's SHA-256.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/fermion"

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


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(container_py: pathlib.Path, artifact_py: pathlib.Path, packer_py: pathlib.Path) -> None:
    five_value_fixture(load_module("fermion_container", container_py))
    quint5_fixture(load_module("fermion_artifact", artifact_py))
    affine_fixture()
    archive_fixture(load_module("package_release_bps", packer_py))


def five_value_fixture(fc) -> None:

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
    path = OUT / "five_value.fermion"
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
        OUT / "five_value.safetensors",
    )
    print("wrote", path)


def quint5_fixture(artifact) -> None:
    """quint5 records, decoded by the reference's `_unpack_symbols`."""
    import torch as t

    rng = np.random.default_rng(5)
    tensors = {}
    for name, (rows, cols) in {"a": (3, 30), "b": (4, 23)}.items():
        symbols = rng.integers(0, 5, size=(rows, cols))
        groups = (cols + 9) // 10
        padded = np.zeros((rows, groups * 10), dtype=np.int64)
        padded[:, :cols] = symbols
        digits = padded.reshape(rows, groups, 10)
        payload = sum(digits[:, :, k] * 5**k for k in range(10))
        packed = np.stack([(payload >> s) & 0xFF for s in (0, 8, 16)], axis=2)
        q = t.from_numpy(packed.reshape(rows, groups * 3).astype(np.uint8))
        assert t.equal(artifact._unpack_symbols(q, cols), t.from_numpy(symbols)), name

        alpha = t.from_numpy((rng.random(rows) * 0.1 + 0.05).astype(np.float32)).to(t.bfloat16)
        residual = t.tensor([0.013], dtype=t.bfloat16)
        a, r = alpha.to(t.float32)[:, None], residual.to(t.float32)
        levels = t.cat([-a - r, -a + r, t.zeros_like(a), a - r, a + r], dim=1)
        dense = t.gather(levels, 1, t.from_numpy(symbols))
        tensors.update({
            f"{name}.quint5_q": q,
            f"{name}.base_alpha": alpha,
            f"{name}.residual_scale": residual,
            f"{name}.dense": dense.contiguous(),
        })
    save_file(tensors, OUT / "quint5.safetensors")


def affine_fixture() -> None:
    """MLX affine records, dequantized by MLX."""
    import mlx.core as mx
    import torch as t

    rng = np.random.default_rng(6)
    tensors = {}
    for bits, group in [(4, 64), (6, 128), (8, 64)]:
        w = mx.array(rng.standard_normal((3, 256)).astype(np.float32)).astype(mx.bfloat16)
        q, scales, biases = mx.quantize(w, group_size=group, bits=bits)
        dense = (
            mx.dequantize(q, scales.astype(mx.float32), biases.astype(mx.float32),
                          group_size=group, bits=bits)
        )
        name = f"b{bits}g{group}"
        tensors.update({
            f"{name}.weight": t.from_numpy(np.array(q).view(np.int32).copy()).view(t.uint32),
            f"{name}.scales": t.from_numpy(np.array(scales.astype(mx.float32))).to(t.bfloat16),
            f"{name}.biases": t.from_numpy(np.array(biases.astype(mx.float32))).to(t.bfloat16),
            f"{name}.dense": t.from_numpy(np.array(dense)),
        })
    save_file(tensors, OUT / "mlx_affine.safetensors")


def archive_fixture(packer) -> None:
    """A two-member archive, written as the reference writer writes one."""
    import hashlib
    import io
    import subprocess
    import tarfile
    import tempfile

    import torch as t

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        model = tmp / "model-00001.safetensors"
        save_file({
            "x": t.arange(10, dtype=t.float32).to(t.bfloat16).reshape(2, 5),
            "y": t.arange(6, dtype=t.uint8),
        }, model)
        config = b'{"labels": ["a", "b"]}\n'
        split, meta = packer.split_file(model)
        assert packer.join_file(split, meta) == model.read_bytes()
        blob = model.read_bytes()
        manifest = {
            "release_format": packer.FORMAT,
            "profile": "toy",
            "compression": {"codec": "zstd", "level": 19},
            "transform": "byte-plane-split-per-tensor-v1",
            "files": [
                {"path": "config.json", "original_bytes": len(config),
                 "original_sha256": hashlib.sha256(config).hexdigest(),
                 "stored_bytes": len(config)},
                {"path": model.name, "original_bytes": len(blob),
                 "original_sha256": hashlib.sha256(blob).hexdigest(),
                 "stored_bytes": len(split), "transform": meta},
            ],
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
            for name, data in [("bps_manifest.json", manifest_bytes),
                               ("config.json", config), (model.name + ".bps", split)]:
                tar.addfile(packer.tar_info(name, len(data)), io.BytesIO(data))
        archive = OUT / "archive.bps.tar.zst"
        subprocess.run(["zstd", "-q", "-19", "-f", "-o", str(archive), "-"],
                       input=tar_bytes.getvalue(), check=True)
        out = tmp / "unpacked"
        packer.unpack(archive, out)
        assert (out / model.name).read_bytes() == blob
        (OUT / "archive.json").write_text(json.dumps({
            "config.json": hashlib.sha256(config).hexdigest(),
            model.name: hashlib.sha256(blob).hexdigest(),
        }, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*(pathlib.Path(a) for a in sys.argv[1:]))
