#!/usr/bin/env python3
"""Generate the Qwen3.5 tokenizer and chat-template fixtures.

Run once; commit the output. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers tokenizers jinja2
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_5_chat_fixtures.py ROW=HF_DIR...

- `ROW=HF_DIR`: a bunsen row name and a directory holding that row's pinned files under their
  repo names, for each of `ROWS` (revisions as bunsen's rows pin them). The rows cover each
  tokenizer and each template the dense rows carry: `qwen3.5-0.8b`'s template (the 2B's too)
  answers at once unless `enable_thinking` is true, `qwen3.5-4b`'s (the 9B's and 27B's)
  thinks unless it is false; the chat rows share one `tokenizer.json`, the `-base` rows
  another. A `-base` row needs its tokenizer files alone.

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, tokenizers 0.23.3, jinja2 3.1.6, torch 2.14.1+cpu

Outputs, under `validation/models/qwen3-5-model-validation/testdata/`, by row:

`chat.json`
    - `encode`: `TEXTS`, each with the ids `transformers`' fast tokenizer gives it, no
      special tokens added (added tokens in the text are matched as `transformers` matches
      them). The texts include combining marks, for the `\\p{M}` classes of Qwen3.5's split.
    - `templates`, for the chat rows: `CONVERSATIONS` under each setting of
      `add_generation_prompt` and `enable_thinking` (`None` is not passed), each with the
      text `apply_chat_template` renders and its ids.
`chat_replies.json`
    For each row of `SESSIONS`: conversations of user turns, each answered by `transformers`'
    `Qwen3_5ForConditionalGeneration` (CPU, float32) greedily through its own cache from the
    chat-templated prompt (generation prompt on, `enable_thinking` as the session says), until
    a stop token (`<|im_end|>` or `<|endoftext|>`: the dense repos have no
    `generation_config.json`) or `budget` tokens. Each reply: its `ids` (the stop token
    included), its `text` (without it), and each step's `margins` (top-1 minus top-2 logit).
    The reply joins the conversation as an assistant turn of its text.
"""

import json
import pathlib
import sys

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/qwen3-5-model-validation/testdata/chat.json"
REPLIES = ROOT / "validation/models/qwen3-5-model-validation/testdata/chat_replies.json"

ROWS = ["qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-0.8b-base"]
CHAT_ROWS = ["qwen3.5-0.8b", "qwen3.5-4b"]
STOPS = ["<|im_end|>", "<|endoftext|>"]

TEXTS = [
    "",
    "Hello, world!",
    "The lighthouse keeper climbed the stairs every evening, and",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\n\n\nand more",
    "numbers: 12345 3.14159 -42 1e10 0x7fff",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "Ünïcödé façade naïve résumé",
    "é versus é versus Ȩ́",
    "हिन्दी पाठ और संयुक्ताक्षर",
    "ภาษาไทย ผสม วรรณยุกต์",
    "áb̂c̃ d⃝ e̶",
    "中文测试：你好，世界！",
    "日本語のテキストです。カタカナとひらがな。",
    "emoji 🎉🚀👩‍💻 and flags 🇺🇸🇯🇵",
    "<|im_start|>user\nhi<|im_end|>\n",
    "<think>\nreasoning\n</think>\n\nanswer",
    "glued<|im_end|>text<|endoftext|>more",
    "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
    "don't won't I'm you're they've we'll he'd",
    "     ",
    "a" * 300,
]

CONVERSATIONS = {
    "one_turn": [{"role": "user", "content": "What is 2 + 2?"}],
    "system": [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Name a prime number."},
    ],
    "multi_turn_thinking": [
        {"role": "user", "content": "Pick a colour."},
        {"role": "assistant", "content": "<think>\nBlue is common.\n</think>\n\nBlue."},
        {"role": "user", "content": "Another?"},
    ],
    "reasoning_content": [
        {"role": "user", "content": "Pick a number."},
        {"role": "assistant", "content": "Seven.", "reasoning_content": "Seven is lucky."},
        {"role": "user", "content": "Why?"},
    ],
    "ends_with_assistant": [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "Hi!"},
    ],
}

SETTINGS = [
    {"add_generation_prompt": True, "enable_thinking": None},
    {"add_generation_prompt": True, "enable_thinking": True},
    {"add_generation_prompt": True, "enable_thinking": False},
    {"add_generation_prompt": False, "enable_thinking": None},
]

SESSIONS = {
    "qwen3.5-0.8b": [
        {"name": "no_thinking", "enable_thinking": None, "budget": 48,
         "turns": ["Name three colours of the rainbow.", "Which of them is warmest?",
                   "And the coolest?"]},
        {"name": "thinking", "enable_thinking": True, "budget": 40,
         "turns": ["What is 17 + 25?"]},
    ],
}


def greedy_reply(model, tok, prompt_ids, stops, budget) -> dict:
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last, past = out.logits[0, -1], out.past_key_values
        for _ in range(budget):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            if ids[-1] in stops:
                break
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past, use_cache=True)
            last, past = out.logits[0, -1], out.past_key_values
    body = ids[:-1] if ids and ids[-1] in stops else ids
    return {"ids": ids, "text": tok.decode(body), "margins": margins}


def replies_fixture(row: str, hf_dir: pathlib.Path) -> list:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        hf_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    stops = [tok.convert_tokens_to_ids(t) for t in STOPS]
    sessions = []
    for session in SESSIONS[row]:
        messages, replies = [], []
        for user in session["turns"]:
            messages.append({"role": "user", "content": user})
            kwargs = {"add_generation_prompt": True}
            if session["enable_thinking"] is not None:
                kwargs["enable_thinking"] = session["enable_thinking"]
            text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
            prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
            reply = greedy_reply(model, tok, prompt_ids, stops, session["budget"])
            replies.append({"user": user, "prompt_ids": prompt_ids, **reply})
            messages.append({"role": "assistant", "content": reply["text"]})
            print(row, session["name"], repr(reply["text"][:60]), "min margin",
                  min(reply["margins"]), flush=True)
        sessions.append({**session, "replies": replies})
    return sessions


def row_fixture(row: str, hf_dir: pathlib.Path) -> dict:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    encode = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TEXTS]
    templates = []
    if row in CHAT_ROWS:
        for name, messages in CONVERSATIONS.items():
            for setting in SETTINGS:
                kwargs = {"add_generation_prompt": setting["add_generation_prompt"]}
                if setting["enable_thinking"] is not None:
                    kwargs["enable_thinking"] = setting["enable_thinking"]
                text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
                templates.append({
                    "name": name,
                    "messages": messages,
                    **setting,
                    "text": text,
                    "ids": tok(text, add_special_tokens=False)["input_ids"],
                })
    return {"encode": encode, "templates": templates}


def main() -> None:
    dirs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    assert sorted(dirs) == sorted(ROWS), f"need {ROWS}, got {sorted(dirs)}"
    fixture = {row: row_fixture(row, pathlib.Path(dirs[row])) for row in ROWS}
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUT, {r: (len(f["encode"]), len(f["templates"])) for r, f in fixture.items()})
    replies = {row: replies_fixture(row, pathlib.Path(dirs[row])) for row in SESSIONS}
    REPLIES.write_text(json.dumps(replies, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
