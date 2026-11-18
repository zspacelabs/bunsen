#!/usr/bin/env python3
"""Generate `zonos3-model-validation`'s fixtures: upstream's reference over ZONOS3-Mini.

Run once; commit the outputs. Regenerating is not part of CI.

    git clone https://github.com/Zyphra/ZONOS3-Mini upstream
    git -C upstream checkout b171b19149d29b8bcc31e0342d5e3b0063dc479b
    # Zyphra/ZONOS3-Mini @ f68e6e470fd14d6d1bc50e8461ee4d4f64d6c513: the release's
    # lm/zonos3_mini/params.json and lm/zonos3_mini/tokenizer/superbpe8192.json
    hf download Zyphra/ZONOS3-Mini lm/zonos3_mini/params.json \\
        lm/zonos3_mini/tokenizer/superbpe8192.json \\
        --revision f68e6e470fd14d6d1bc50e8461ee4d4f64d6c513 --local-dir native
    # Zyphra/ZONOS3-Mini-hf @ 73500c5a1fc3e0c4da69908e3cffd9346cdd97c2
    hf download Zyphra/ZONOS3-Mini-hf model.safetensors voices.safetensors tokenizer.json \\
        modeling_zonos3.py configuration_zonos3.py \\
        --revision 73500c5a1fc3e0c4da69908e3cffd9346cdd97c2 --local-dir hf
    python3.12 -m venv venv
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
    venv/bin/pip install transformers==4.57.3 safetensors numpy tokenizers zyphra-textnorm==0.1.0
    venv/bin/pip install --no-deps -e upstream/model
    venv/bin/python -I validation/tools/gen_zonos3_fixtures.py \\
        --upstream upstream --native native --hf hf --work WORK_DIR

Pinned versions used to produce the committed fixtures:
    torch 2.9.1+cpu, transformers 4.57.3, tokenizers 0.22.1, zyphra-textnorm 0.1.0,
    Python 3.12; every input checked against its pin (`git rev-parse HEAD`, sha256).

Everything runs on the CPU. The reference language model is upstream's
`zonos3.reference.lm.Zonos3LM` ("the numerical oracle"), loaded strictly from a checkpoint
directory `WORK_DIR` assembles: the release's `params.json` and tokenizer, and a `model.pth`
of the `-hf` file's `lm.*` tensors, which are the release's own (same names, shapes, dtypes and
bytes). The prompts are upstream's (`zonos3.reference.synth.plan_prompt`), normalized by its
text normalizer. The codec oracle is the `-hf` port's `Zonos3MiniCodec` (op for op with
upstream's decoder, weight norm fused), in float32.

The outputs are safetensors files: integers as int32, values as float32, sums as float64,
and the strings (JSON) in the header's `__metadata__`. Every file's metadata carries the pins
(`upstream`, `hf_revision`, `torch`) and the `steps` and `sample_stride` used.

For each of `REQUESTS` (the `prompt_requests.json` requests that need no extra text rows, no
controls and no start or pause tokens), and for the language model in float32
(`reference.safetensors`) and in bfloat16 (`reference_bf16.safetensors`, upstream's
`cast(bfloat16)`, the routers' biases kept float32; the logits only, the `prefill.*` and
`steps.*` tensors, teacher-forced over the float32 file's `generated` rows), under `<name>.`:

- metadata `<name>`: a JSON object, the `request`, its `normalized_text`, the `speaker_row`
  (embedding mode, else null), the `condition_row` (the first, else null) and the
  `condition_buckets`, the `silence_rows` ending the prompt, and the `stats_stride`;
- `rows` (`[R, 11]`), the rows the language model reads; `text_ids` (`[T]`, `[BOS] + ids +
  [EOS]`); `speaker_embedding` (`[S]`, embedding mode only);
- `prefill.argmax` and `prefill.margin` (`[R, 10]`): each row's argmax per codebook and its
  top-1 minus top-2 margin, so a reader can tell a near-tie from a disagreement;
  `prefill.top8.ids` and `prefill.top8.values` (`[10, 8]`): the last row's 8 best per codebook;
  `prefill.sums` (`[2]`, sum and absolute sum) and `prefill.sample` (every `stats_stride`-th
  value, about `STATS_SAMPLES` of them) of the whole `[R, 10, 4098]` logits;
- `generated` (`[STEPS, 11]`): the rows upstream's `generate_rows` samples at the request's
  seed, fed back one at a time through the reference's cache; `steps.argmax`, `steps.margin`
  (`[STEPS, 10]`), `steps.top8.ids` and `steps.top8.values` (`[STEPS, 10, 8]`) for each.

`codec.safetensors` (float32 only): the audio `Zonos3MiniCodec` decodes from fixed codes, under
`<case>.`: the `codes` (`[F, 10]`); `audio.stats` (`[4]`: the sample count, sum, absolute sum
and RMS); `audio.sample` (every `sample_stride`-th sample); `audio.windows` (`[3,
AUDIO_WINDOW]`), runs of consecutive samples at the start, middle and end, starting at
`audio.window_starts` (`[3]`). The cases (metadata `cases`, a JSON
list): `request:<name>` for the first three requests, the prompt's silence rows and the
generated rows undelayed to their complete frames; `voice:<name>` for the first `VOICE_FRAMES`
frames of `VOICES`' stock codes.

`tokenizer.safetensors`: `<stem>.ids`, the ids the `-hf` `tokenizer.json` gives
(`tokenizers`, no special tokens added) for each `bunsen-test-audio` `udhr_<stem>.txt`.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/zonos3-model-validation/testdata"
UDHR = ROOT / "crates/dev/bunsen-test-audio/testdata"

UPSTREAM_COMMIT = "b171b19149d29b8bcc31e0342d5e3b0063dc479b"
NATIVE_FILES = {
    "lm/zonos3_mini/params.json": "28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5a5b",
    "lm/zonos3_mini/tokenizer/superbpe8192.json": "90322f910600027dee7fe63d400d1d9dc1035c167d560aadf0871e3661d08bf3",
}
HF_FILES = {
    "model.safetensors": "2d84a7b35c8e8bd4e962de84616285cf46ced398df016d7391f8cd2324985cb3",
    "voices.safetensors": "da2377fb4c530dda80ae7aba71be99f032fa33377b447cbac706e376ce972c0b",
    "tokenizer.json": "065373b77939d79e27d86300c7c9112da8dcd7f36bd949fa658a7550ffdbfed0",
    "modeling_zonos3.py": "975e2756f24d45aa95d49de45aae799e0009e0d3ff8747e6e9482609074c65e2",
    "configuration_zonos3.py": "16715663cdbb99dae70f1709853dfaf0243b1222fe5f19c5bcc613b86dd0018c",
}

REQUESTS = [
    "plain",
    "embedding",
    "reference_codes",
    "condition_values",
    "silence_prefill_5",
    "silence_prefill_max",
    "pass_through_language",
    "reference_codes_long",
]
STEPS = 12
VOICES = ["Savannah", "Arjun"]
VOICE_FRAMES = 20
SAMPLE_STRIDE = 97
STATS_SAMPLES = 256
AUDIO_WINDOW = 1024


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_pins(args) -> None:
    head = subprocess.run(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        sys.exit(f"{args.upstream} is at {head}, not {UPSTREAM_COMMIT}")
    for root, files in ((args.native, NATIVE_FILES), (args.hf, HF_FILES)):
        for name, digest in files.items():
            got = sha256(root / name)
            if got != digest:
                sys.exit(f"{root / name}: sha256 {got}, not the pinned {digest}")


def reference_dir(args) -> pathlib.Path:
    """The reference's checkpoint directory: the release's params and tokenizer, and the
    `-hf` file's `lm.*` tensors as `model.pth`."""
    from safetensors import safe_open

    out = args.work / "reference-lm"
    if (out / "model.pth").exists():
        return out
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.native / "lm/zonos3_mini/params.json", out / "params.json")
    shutil.copy(
        args.native / "lm/zonos3_mini/tokenizer/superbpe8192.json",
        out / "tokenizer/superbpe8192.json",
    )
    state = {}
    with safe_open(args.hf / "model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("lm."):
                state[k[3:]] = f.get_tensor(k)
    torch.save(state, out / "model.pth")
    return out


def load_hf_codec(hf: pathlib.Path):
    package = types.ModuleType("zonos3_hf")
    package.__path__ = [str(hf)]
    sys.modules["zonos3_hf"] = package
    for name in ("configuration_zonos3", "modeling_zonos3"):
        spec = importlib.util.spec_from_file_location(f"zonos3_hf.{name}", hf / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"zonos3_hf.{name}"] = module
        spec.loader.exec_module(module)
    from safetensors import safe_open

    modeling = sys.modules["zonos3_hf.modeling_zonos3"]
    configuration = sys.modules["zonos3_hf.configuration_zonos3"]
    codec = modeling.Zonos3MiniCodec(dict(configuration.RELEASE_CODEC))
    state = {}
    with safe_open(hf / "model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("codec."):
                state[k[len("codec.") :]] = f.get_tensor(k).float()
    codec.load_state_dict(state, strict=True)
    return codec.float().eval()


def i32(x) -> torch.Tensor:
    return torch.as_tensor(x).to(torch.int32).contiguous()


def f32(x) -> torch.Tensor:
    return torch.as_tensor(x).to(torch.float32).contiguous()


def top8(logits: torch.Tensor, prefix: str) -> dict:
    """`logits` `[..., V]`: each codebook's 8 best ids and values."""
    values, ids = logits.float().topk(8, dim=-1)
    return {f"{prefix}.ids": i32(ids), f"{prefix}.values": f32(values)}


def argmax_margin(logits: torch.Tensor, prefix: str) -> dict:
    """`logits` `[..., V]`: the argmax and the top-1 minus top-2 margin."""
    top = logits.float().topk(2, dim=-1)
    return {
        f"{prefix}.argmax": i32(top.indices[..., 0]),
        f"{prefix}.margin": f32(top.values[..., 0] - top.values[..., 1]),
    }


def undelay(rows: torch.Tensor, n: int) -> torch.Tensor:
    """Delayed `rows` `[R, >=n]` to their complete frames `[R - n + 1, n]`."""
    frames = rows.shape[0] - n + 1
    return torch.stack([rows[j : j + frames, j] for j in range(n)], dim=1)


def lm_fixture(lm, reqs, assets, dtype_name, forced=None):
    """Each request's tensors and metadata; `forced`, the rows to teacher-force by request
    (else the ones `generate_rows` samples)."""
    from zonos3.reference.lm import LMCache
    from zonos3.reference.synth import Sampling, generate_rows, plan_prompt

    c = lm.config
    tensors, metadata = {}, {}
    for name in REQUESTS:
        request = reqs[name]["request"]
        plan = plan_prompt(lm, request, assets=assets, device="cpu")
        cond = plan.cond
        ids = cond.input_ids.long()
        speaker = cond.speaker_rows()
        conditions = cond.condition_rows()
        with torch.inference_mode():
            cache = LMCache()
            logits = lm(ids, cache, speaker_rows=speaker, condition_rows=conditions).logits
            if forced is None:
                attempt = generate_rows(
                    lm,
                    cond,
                    sampling=Sampling.from_request(plan.request),
                    seed=plan.seed,
                    max_tokens=STEPS,
                    stop_id=c.eoa_id,
                )
                generated = torch.stack([r.long() for r in attempt.rows])[:STEPS]
            else:
                generated = forced[name]
            steps = torch.stack([lm(row[None], cache).logits[0] for row in generated])
        flat = logits.float().flatten()
        stride = max(1, flat.numel() // STATS_SAMPLES)
        speaker_row = next(iter(speaker)) if speaker else None
        t = {
            "rows": i32(ids),
            "text_ids": i32([int(x) for x in ids[:, c.n_codebooks].tolist() if x != c.text_vocab]),
            **argmax_margin(logits, "prefill"),
            **top8(logits[-1], "prefill.top8"),
            "prefill.sums": torch.tensor(
                [flat.double().sum().item(), flat.double().abs().sum().item()], dtype=torch.float64
            ),
            "prefill.sample": f32(flat[::stride]),
            "generated": i32(generated),
            **argmax_margin(steps, "steps"),
            **top8(steps, "steps.top8"),
        }
        if speaker_row is not None:
            t["speaker_embedding"] = f32(speaker[speaker_row].float())
        tensors.update({f"{name}.{k}": v for k, v in t.items()})
        metadata[name] = json.dumps(
            {
                "request": request,
                "normalized_text": plan.text,
                "speaker_row": speaker_row,
                "condition_row": min(conditions) if conditions else None,
                "condition_buckets": [conditions[r] for r in sorted(conditions)],
                "silence_rows": 0 if plan.prefill_rows is None else len(plan.prefill_rows),
                "stats_stride": stride,
            }
        )
        print(f"{dtype_name} {name}: {len(ids)} rows, {len(generated)} steps")
    return tensors, metadata


def codec_fixture(codec, reference, metadata, hf):
    from safetensors import safe_open

    cases = {}
    for name in REQUESTS[:3]:
        rows = reference[f"{name}.rows"].long()
        silence = json.loads(metadata[name])["silence_rows"]
        rows = torch.cat([rows[len(rows) - silence :], reference[f"{name}.generated"].long()])
        cases[f"request:{name}"] = undelay(rows[:, :10], 10)
    with safe_open(hf / "voices.safetensors", "pt") as f:
        for voice in VOICES:
            cases[f"voice:{voice}"] = f.get_tensor(f"{voice}.codes")[:VOICE_FRAMES].long()
    tensors = {}
    with torch.inference_mode():
        for name, frames in cases.items():
            audio = codec(frames.T[None]).double().flatten()
            tensors[f"{name}.codes"] = i32(frames)
            tensors[f"{name}.audio.stats"] = torch.tensor(
                [
                    audio.numel(),
                    audio.sum().item(),
                    audio.abs().sum().item(),
                    audio.square().mean().sqrt().item(),
                ],
                dtype=torch.float64,
            )
            tensors[f"{name}.audio.sample"] = f32(audio[::SAMPLE_STRIDE])
            n = audio.numel()
            starts = [0, (n - AUDIO_WINDOW) // 2, n - AUDIO_WINDOW]
            tensors[f"{name}.audio.window_starts"] = i32(starts)
            tensors[f"{name}.audio.windows"] = f32(
                torch.stack([audio[i : i + AUDIO_WINDOW] for i in starts])
            )
            print(f"codec {name}: {len(frames)} frames, {audio.numel()} samples")
    return tensors, {"cases": json.dumps(list(cases))}


def tokenizer_fixture(hf):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(hf / "tokenizer.json"))
    return {
        f"{p.stem.removeprefix('udhr_')}.ids": i32(
            tok.encode(p.read_text(), add_special_tokens=False).ids
        )
        for p in sorted(UDHR.glob("udhr_*.txt"))
    }


DTYPES = {torch.int32: "I32", torch.float32: "F32", torch.float64: "F64"}


def write(name: str, tensors: dict, metadata: dict) -> None:
    """A safetensors file, its header's keys sorted, so a regeneration is byte-identical
    (`safetensors.torch.save_file` orders the metadata as a hash map does)."""
    header, blobs, offset = {"__metadata__": dict(sorted(metadata.items()))}, [], 0
    for key in sorted(tensors):
        t = tensors[key].contiguous()
        blob = t.numpy().tobytes()
        header[key] = {
            "dtype": DTYPES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("upstream", "native", "hf", "work"):
        parser.add_argument(f"--{flag}", type=pathlib.Path, required=True)
    args = parser.parse_args()
    check_pins(args)
    torch.set_num_threads(32)
    sys.path.insert(0, str(args.upstream / "model/src"))
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets

    reqs = {
        r["name"]: r
        for r in json.loads((args.upstream / "model/tests/fixtures/prompt_requests.json").read_text())
    }
    lm_dir = reference_dir(args)
    assets = ReferenceAssets.load()
    meta = {
        "upstream": UPSTREAM_COMMIT,
        "hf_revision": "73500c5a1fc3e0c4da69908e3cffd9346cdd97c2",
        "torch": torch.__version__,
        "steps": str(STEPS),
        "sample_stride": str(SAMPLE_STRIDE),
    }
    lm = Zonos3LM.load(lm_dir, dtype=torch.float32)
    reference, metadata = lm_fixture(lm, reqs, assets, "f32")
    write("reference.safetensors", reference, {**meta, "dtype": "f32", **metadata})
    del lm
    forced = {name: reference[f"{name}.generated"].long() for name in REQUESTS}
    lm = Zonos3LM.load(lm_dir, dtype=torch.bfloat16)
    bf16, _ = lm_fixture(lm, reqs, assets, "bf16", forced)
    bf16 = {k: v for k, v in bf16.items() if ".prefill." in k or ".steps." in k}
    write("reference_bf16.safetensors", bf16, {**meta, "dtype": "bf16"})
    del lm
    codec = load_hf_codec(args.hf)
    tensors, cases = codec_fixture(codec, reference, metadata, args.hf)
    write("codec.safetensors", tensors, {**meta, **cases})
    write("tokenizer.safetensors", tokenizer_fixture(args.hf), meta)

if __name__ == "__main__":
    main()
