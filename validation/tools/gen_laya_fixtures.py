#!/usr/bin/env python3
"""Generate the Laya reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors tokenizers numpy
    /tmp/pkvenv/bin/python -I validation/tools/gen_laya_fixtures.py ROW=DIR...

- `ROW=DIR`: a bunsen row name and the directory of its checkpoint's files under their
  names in the repo (`model.safetensors`, `encoder/config.json`, `rl_agent_config.json`,
  `tokenizer/`): the repo's root for `laya`, its `multilingual/` and `typed-decisions/`
  folders for the others.

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, numpy 2.5.4, tokenizers 0.23.3

The reference is upstream's code (`rl_common.py` and `rl_agent_api.py` of
`convaiinnovations/laya` at 7b928d82, Apache-2.0): the parts inference runs are ported below
as written, so nothing the repo ships is executed; its files are read as data (weights,
configs, `tokenizer.json`).

Each model runs on the CPU in float32, as upstream's `RLAgent` runs off a GPU (no autocast),
its encoder through `transformers`' sdpa attention as upstream builds it. Every question is
its own batch of one, so no padding enters the reference. No Laya file is written here:
only texts, ids and recorded tensors.

Output, `validation/models/laya-model-validation/testdata/upstream.json`: for each row, its
`cases`, each a `state` and its `questions` (in bunsen's schema: `type`, `question`,
`options`), and for each question what upstream computes:

- `ids` and `markers`: `build_sequence`'s prompt and its options' `[MASK]` positions;
- `encoder`: the encoder's last hidden state at the first token and each marker, as a
  record (`shape`, `sum`, `abs_sum` and a strided sample);
- `logits`: the scorer's logits at the markers; `act_logits`: the act head's;
- `probabilities`: the logits calibrated as `RLAgent.system_one` calibrates them, its
  temperature by the question's type and option count; `act_probability`, the act head's
  softmax at its first act.
"""

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

N_RECORDED = 256

OUT = pathlib.Path(__file__).resolve().parent.parent / "models/laya-model-validation/testdata"

REFUND = (
    "Customer: I ordered a kettle three weeks ago and it arrived with a cracked lid. "
    "I asked for a refund twice and nobody answered. Agent: I am sorry about that. "
    "I can see the order and the photos you sent."
)
LEDGER = " ".join(
    f"Line {i}: pallet {100 + i} moved from bay {i % 7} to bay {(i * 3) % 11}, checked by shift {i % 3}."
    for i in range(120)
)
CASES = [
    {
        "state": REFUND,
        "questions": [
            {"type": "choice", "question": "What should the agent do next?",
             "options": ["refund", "replace", {"escalate": "hand the case to a supervisor"}, "close"]},
            {"type": "noul", "question": "Is the customer satisfied?"},
            {"type": "score", "question": "How urgent is this case?",
             "options": ["not urgent", "a little urgent", "urgent", "very urgent", "critical"]},
        ],
    },
    {
        "state": "The build failed twice on the same test, which passes locally.",
        "questions": [
            {"type": "choice", "question": "What is the most likely cause?",
             "options": ["a flaky test", "an environment difference", "a real regression"]},
            {"type": "noul", "question": "Should the release wait?",
             "options": [{"false": "ship it now"}, {"true": "hold the release"}]},
        ],
    },
    {
        "state": LEDGER + " Question from the floor: where did pallet 100 go?",
        "questions": [
            {"type": "choice", "question": "Which bay did pallet 100 move to?",
             "options": ["bay 0", "bay 3", "bay 5"]},
        ],
    },
]
CASES_MULTILINGUAL = CASES + [
    {
        "state": "El cliente pidió un reembolso dos veces y nadie le respondió.",
        "questions": [
            {"type": "noul", "question": "¿Está satisfecho el cliente?"},
            {"type": "choice", "question": "¿Qué debe hacer el agente?",
             "options": ["reembolsar", "escalar", "cerrar"]},
        ],
    },
]


# ----------------------------------------------------------------------------- upstream
# Ported from `rl_common.py` and `rl_agent_api.py` (7b928d82), as written; the training-only
# paths (checkpointing, a detached encoder, the episode and target fields) are left out.

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def render_options(q):
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q, max_len, head_max_len, option_order=None, truncate_left=False):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id] + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(state.replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_model(cfg, encoder_dir):
    from transformers import AutoConfig, AutoModel
    ecfg = AutoConfig.from_pretrained(encoder_dir)
    enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    return DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)


def collate_one(ids, markers, qtype, pad_id):
    """`collate_items` for one item: no padding."""
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        "marker_pos": torch.tensor([markers], dtype=torch.long),
        "marker_mask": torch.ones((1, len(markers)), dtype=torch.bool),
        "qtype": torch.tensor([qtype]),
    }


def temp_bucket(qtype, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


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


def upstream_question(q: dict) -> dict:
    """bunsen's question as upstream's internal one: `t`, `ins`, and `crit` (a choice's
    {name: description}, a score's level descriptions, a yes/no question's {false, true})."""
    def entry(o):
        return (o, None) if isinstance(o, str) else next(iter(o.items()))
    opts = [entry(o) for o in q.get("options", [])]
    if q["type"] == "choice":
        crit = {k: v for k, v in opts}
    elif q["type"] == "score":
        crit = [v if v is not None else k for k, v in opts]
    else:
        crit = {k: v for k, v in opts} or None
    return {"t": q["type"], "ins": q["question"], "crit": crit}


def run_row(d: pathlib.Path, cases: list) -> dict:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    cfg = json.loads((d / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(d / "tokenizer")
    model = build_model(cfg, encoder_dir=str(d / "encoder"))
    model.load_state_dict(load_file(d / "model.safetensors"), strict=True)
    model = model.float().eval()
    temps, by_options = cfg.get("temperature", [1.0, 1.0, 1.0]), cfg.get("temperature_by_options", {})

    hidden = {}
    model.encoder.register_forward_hook(lambda _m, _i, out: hidden.__setitem__("h", out.last_hidden_state))

    out_cases = []
    for case in cases:
        prompts = []
        for q in case["questions"]:
            uq = upstream_question(q)
            ids, markers = build_sequence(tok, case["state"], uq, cfg["max_len"], cfg["head_max_len"])
            k = len(render_options(uq))
            assert len(markers) == k, (q, markers)
            qt = QTYPES[uq["t"]]
            b = collate_one(ids, markers, qt, tok.pad_token_id)
            with torch.no_grad():
                logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                    b["marker_mask"], b["qtype"])
            z = logits[0, :k].double() / by_options.get(temp_bucket(qt, k), temps[qt])
            p = torch.softmax(z, -1)
            h = hidden["h"][0, [0] + markers].double().numpy()
            prompts.append({
                "ids": [int(i) for i in ids],
                "markers": [int(m) for m in markers],
                "encoder": record(h),
                "logits": [float(v) for v in logits[0, :k]],
                "act_logits": [float(v) for v in act[0]],
                "probabilities": [float(v) for v in p],
                "act_probability": float(torch.softmax(act[0].double(), -1)[0]),
            })
        out_cases.append({**case, "prompts": prompts})
    return {"cases": out_cases}


def main(argv):
    rows = {}
    for arg in argv:
        row, d = arg.split("=", 1)
        cases = CASES_MULTILINGUAL if "multilingual" in row else CASES
        rows[row] = run_row(pathlib.Path(d), cases)
        print(row, sum(len(c["prompts"]) for c in rows[row]["cases"]), "prompts")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "upstream.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
