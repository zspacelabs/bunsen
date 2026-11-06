#!/usr/bin/env python3
"""Generate the dense Qwen3.5 reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors tokenizers jinja2 numpy
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_5_fixtures.py ROW=HF_DIR...

- `ROW=HF_DIR`: a bunsen row name and a directory holding that row's pinned files under their
  repo names (`config.json`, the safetensors, `tokenizer.json`, ...), for the rows of `ROWS`
  and `BF16_ROWS` (revisions as bunsen's rows pin them). Each output is written when every row
  it needs is given.

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, numpy 2.5.4, tokenizers 0.23.3

Every model is `transformers`' `Qwen3_5ForConditionalGeneration` (the checkpoints' own
class) on the CPU with eager attention, run on text ids, as a text prompt runs through it.
The delta rule runs in its reference forms (`flash-linear-attention` is not installed): the
chunked one for a prefill, the recurrent one for each cached step. No Qwen3.5 file is written
here: only ids, texts and logits.

Outputs, under `validation/models/qwen3-5-model-validation/testdata/`:

`transformers.json`
    For each of `ROWS`: the row's tokenizer encoding `PROMPT` (no special tokens: a Qwen
    tokenizer adds none) to `prompt_ids`, then `NEW_TOKENS` greedy tokens in float32 through
    `transformers`' own cache: the generated `ids`, their `text`, each step's `margins`
    (top-1 minus top-2 logit), and the prefill's last-position logits, `last_logits`
    (recorded as `shape`, `sum`, `abs_sum` and a strided sample).
`transformers_chunks.json`
    For each of `ROWS`: a prompt of at least `CHUNKS_MIN_POSITIONS` ids, not a multiple of the
    delta rule's chunk of 64 (a generated ledger, no third-party text), then
    `CHUNKS_NEW_TOKENS` greedy tokens in float32, recorded as above: a prefill whose linear
    layers carry their state across several chunks.
`transformers_bf16.json`
    For each of `BF16_ROWS` (`qwen3.5-9b`, which a 24 GB device holds only in bf16): the
    greedy decode of `PROMPT` in bfloat16, its `last_logits` as float32, and
    `f32_last_logits_max_abs`, how far the float32 model's prefill logits are from those: a
    bf16 run's own distance from float32, which bunsen's bf16 run is measured against.
"""

import json
import pathlib
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/qwen3-5-model-validation/testdata"

N_RECORDED = 2048
PROMPT = "The lighthouse keeper climbed the stairs every evening, and"
NEW_TOKENS = 32
ROWS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"]
BF16_ROWS = ["qwen3.5-9b"]
CHUNKS_MIN_POSITIONS = 200
CHUNKS_NEW_TOKENS = 16
CHUNKS_SEED = 20261106
CHUNK = 64


def record(x) -> dict:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    stride = max(1, flat.size // N_RECORDED)
    return {
        "shape": list(np.shape(x)),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "stride": stride,
        "samples": [float(v) for v in flat[::stride]],
    }


def load(hf_dir: pathlib.Path, dtype):
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        hf_dir, dtype=dtype, attn_implementation="eager"
    ).eval()


def decode(model, tok, prompt_ids: list[int], new_tokens: int) -> tuple[dict, np.ndarray]:
    """A greedy decode of `new_tokens` from `prompt_ids` through `model`'s own cache, and the
    prefill's whole last-position logits, as float32."""
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last = out.logits[0, -1].float()
        last_logits = last.numpy()[None]
        past = out.past_key_values
        for _ in range(new_tokens):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past,
                        use_cache=True)
            last, past = out.logits[0, -1].float(), out.past_key_values
    return {"ids": ids, "text": tok.decode(ids), "margins": margins,
            "last_logits": record(last_logits)}, last_logits


def ledger_text(items: int) -> str:
    """A generated warehouse ledger, one line per item: no third-party text."""
    rng = random.Random(CHUNKS_SEED)
    goods = ["bolts", "hinges", "lanterns", "ropes", "barrels", "anchors", "sails", "nets"]
    lines = ["Harbour warehouse ledger, inventory by aisle:"]
    for i in range(items):
        lines.append(f"Aisle {i + 1}: {rng.randint(2, 90)} {rng.choice(goods)}, "
                     f"checked by clerk {rng.randint(1, 9)}.")
    return "\n".join(lines)


def chunks_prompt(tok) -> list[int]:
    """A ledger of at least `CHUNKS_MIN_POSITIONS` ids, cut so its length is not a multiple
    of the delta rule's chunk."""
    items = 8
    while True:
        ids = tok(ledger_text(items))["input_ids"]
        if len(ids) >= CHUNKS_MIN_POSITIONS and len(ids) % CHUNK:
            return ids
        items += 1


def main() -> None:
    dirs = {row: pathlib.Path(d) for row, d in (arg.split("=", 1) for arg in sys.argv[1:])}
    unknown = set(dirs) - set(ROWS + BF16_ROWS)
    assert not unknown, f"not a fixture row: {sorted(unknown)}"
    OUT.mkdir(parents=True, exist_ok=True)
    if all(row in dirs for row in ROWS):
        write_f32(dirs)
    if all(row in dirs for row in BF16_ROWS):
        write_bf16(dirs)


def write_f32(dirs: dict) -> None:
    greedy, chunks = {}, {}
    for row in ROWS:
        tok = AutoTokenizer.from_pretrained(dirs[row])
        model = load(dirs[row], torch.float32)
        prompt_ids = tok(PROMPT)["input_ids"]
        entry, _ = decode(model, tok, prompt_ids, NEW_TOKENS)
        greedy[row] = {"prompt": PROMPT, "prompt_ids": prompt_ids, **entry}
        print(row, entry["text"][:80].replace("\n", " "), "| min margin", min(entry["margins"]),
              flush=True)
        prompt_ids = chunks_prompt(tok)
        entry, _ = decode(model, tok, prompt_ids, CHUNKS_NEW_TOKENS)
        chunks[row] = {"prompt_ids": prompt_ids, **entry}
        print(row, f"{len(prompt_ids)} positions:", entry["text"][:80].replace("\n", " "),
              "| min margin", min(entry["margins"]), flush=True)
        del model
    (OUT / "transformers.json").write_text(json.dumps(greedy) + "\n")
    (OUT / "transformers_chunks.json").write_text(json.dumps(chunks) + "\n")


def write_bf16(dirs: dict) -> None:
    bf16 = {}
    for row in BF16_ROWS:
        tok = AutoTokenizer.from_pretrained(dirs[row])
        prompt_ids = tok(PROMPT)["input_ids"]
        model = load(dirs[row], torch.bfloat16)
        entry, b16 = decode(model, tok, prompt_ids, NEW_TOKENS)
        del model
        model = load(dirs[row], torch.float32)
        with torch.no_grad():
            f32 = model(input_ids=torch.tensor([prompt_ids])).logits[0, -1].float().numpy()
        del model
        entry["f32_last_logits_max_abs"] = float(np.abs(f32 - b16[0]).max())
        bf16[row] = {"prompt": PROMPT, "prompt_ids": prompt_ids, **entry}
        print(row, "bf16:", entry["text"][:80].replace("\n", " "),
              "| bf16 vs f32 prefill logits, max abs", entry["f32_last_logits_max_abs"],
              flush=True)
    (OUT / "transformers_bf16.json").write_text(json.dumps(bf16) + "\n")


if __name__ == "__main__":
    main()
