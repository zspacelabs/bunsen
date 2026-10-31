#!/usr/bin/env python3
"""Generate the Qwen3 reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers onnxruntime safetensors tokenizers numpy
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_fixtures.py ONNX_DIR ROW=HF_DIR...

- `ONNX_DIR`: `onnx-community/Qwen3-0.6B-ONNX` at 58ca03a5e7b354b480b1a82b34bde8dc7b854cc0
  (`model.onnx` + `model.onnx_data` from its `onnx/`): the published graph, not the one
  `bind-qwen3-onnx` rewrites.
- `ROW=HF_DIR`: a bunsen row name and a directory holding that row's pinned files under their
  repo names (`config.json`, the safetensors, `tokenizer.json`, ...), for each of
  `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-4b-instruct-2507` (revisions as bunsen's rows
  pin them).

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, onnxruntime 1.31.0, numpy 2.5.4, tokenizers 0.23.3

Outputs, under `validation/models/qwen3-model-validation/testdata/`:

`onnx_reference.json`
    The published graph run once by onnxruntime (CPU, graph optimizations off) on fixed ids:
    a prefill of `ids` from an empty cache, then one step on the prefill's greedy next token
    `step_id`. Each output is recorded as its `shape`, `sum`, `abs_sum` and a strided sample.
`transformers.json`
    For each row: `transformers`' `Qwen3ForCausalLM` on the CPU in float32, the row's
    tokenizer encoding `PROMPT` (no chat template) to `prompt_ids`, then `NEW_TOKENS` greedy
    tokens through its own KV cache: the generated `ids`, their `text`, each step's `margins`
    (top-1 minus top-2 logit, so a reader can tell a near-tie from a disagreement), and the
    prefill's last-position logits, `last_logits`. For `qwen3-0.6b`, also how far the
    `lm_head.weight` its checkpoint carries is from `embed_tokens` (`transformers` ties it).
"""

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/qwen3-model-validation/testdata"

N_RECORDED = 2048
N_LAYERS = 28
ONNX_IDS = [9707, 11, 1879, 0, 151643, 13, 576, 6722, 315, 9625, 374]
PROMPT = "The lighthouse keeper climbed the stairs every evening, and"
NEW_TOKENS = 32
ROWS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-4b-instruct-2507"]


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


def onnx_reference(onnx_dir: pathlib.Path) -> dict:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    def run(ids, past, past_len):
        s = len(ids)
        feed = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, past_len + s), np.int64),
            "position_ids": np.arange(past_len, past_len + s, dtype=np.int64)[None],
        }
        for i in range(N_LAYERS):
            feed[f"past_key_values.{i}.key"], feed[f"past_key_values.{i}.value"] = past[i]
        d = dict(zip(names, sess.run(None, feed)))
        return d["logits"], [(d[f"present.{i}.key"], d[f"present.{i}.value"])
                             for i in range(N_LAYERS)]

    empty = [(np.zeros((1, 8, 0, 128), np.float32),) * 2 for _ in range(N_LAYERS)]
    logits, cache = run(ONNX_IDS, empty, 0)
    step_id = int(logits[0, -1].argmax())
    step_logits, step_cache = run([step_id], cache, len(ONNX_IDS))
    return {
        "source": "onnx-community/Qwen3-0.6B-ONNX@58ca03a5e7b354b480b1a82b34bde8dc7b854cc0 onnx/model.onnx",
        "ids": ONNX_IDS,
        "prefill_logits": record(logits),
        "prefill_key_0": record(cache[0][0]),
        "prefill_value_27": record(cache[27][1]),
        "step_id": step_id,
        "step_logits": record(step_logits),
        "step_key_27": record(step_cache[27][0]),
    }


def head_distance(hf_dir: pathlib.Path) -> float:
    with safe_open(str(hf_dir / "model.safetensors"), "pt") as f:
        if "lm_head.weight" not in f.keys():
            return 0.0
        head = f.get_tensor("lm_head.weight").float()
        embed = f.get_tensor("model.embed_tokens.weight").float()
    return float((head - embed).abs().max())


def greedy(row: str, hf_dir: pathlib.Path) -> dict:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last = out.logits[0, -1]
        last_logits = last.numpy()[None]
        past = out.past_key_values
        for _ in range(NEW_TOKENS):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past, use_cache=True)
            last, past = out.logits[0, -1], out.past_key_values
    entry = {
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "ids": ids,
        "text": tok.decode(ids),
        "margins": margins,
        "last_logits": record(last_logits),
    }
    print(row, entry["text"][:80].replace("\n", " "), "min margin", min(margins))
    return entry


def main() -> None:
    onnx_dir = pathlib.Path(sys.argv[1])
    dirs = dict(arg.split("=", 1) for arg in sys.argv[2:])
    assert sorted(dirs) == sorted(ROWS), f"need {ROWS}, got {sorted(dirs)}"
    OUT.mkdir(parents=True, exist_ok=True)

    reference = onnx_reference(onnx_dir)
    (OUT / "onnx_reference.json").write_text(json.dumps(reference) + "\n")
    print("onnx step id", reference["step_id"])

    fixture = {row: greedy(row, pathlib.Path(dirs[row])) for row in ROWS}
    fixture["qwen3-0.6b"]["lm_head_vs_embed_max_abs"] = head_distance(
        pathlib.Path(dirs["qwen3-0.6b"]))
    (OUT / "transformers.json").write_text(json.dumps(fixture) + "\n")


if __name__ == "__main__":
    main()
