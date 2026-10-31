#!/usr/bin/env python3
"""Generate `nanochat-model-validation`'s chat fixture: upstream's tokenizers and conversation layout.

Run once; commit the output. Regenerating is not part of CI.

    /tmp/ncvenv/bin/python validation/tools/gen_nanochat_chat_fixtures.py \\
        /tmp/nanochat-2025 /tmp/nanochat-2026 CACHE_ROOT

The checkouts and `CACHE_ROOT` are as for `gen_nanochat_fixtures.py`, whose docstring builds the
venv. Pinned versions used to produce the committed fixture:
    torch 2.15.0+cpu, tiktoken 0.14.0, rustbpe 0.1.0

For each of the two tokenizers the rows use (the 2025 one, `nanochat-students`', and the 2026
one, `nitishpandey04`'s), upstream's own `RustBPETokenizer` at its generation's pinned commit
(`a088b7a6`, `a3ca42a6`), from the row's `tokenizer.pkl`:

- `encode`: each of `TEXTS` as upstream's `encode` gives it (no special tokens: their spellings
  are text), and `decode` of those ids back;
- `specials`: the special tokens' ids, by name;
- `conversations`: each of `CONVERSATIONS` as upstream's `render_for_completion` renders it for
  the assistant's next turn (an empty assistant turn appended, which it pops).

Output: `validation/models/nanochat-model-validation/testdata/chat.json`, by tokenizer.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/nanochat-model-validation/testdata/chat.json"

# Each tokenizer: its generation, and (namespace, sha256, path) of its `tokenizer.pkl`.
TOKENIZERS = {
    "2025": ("nanochat-students", "8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b",
             "tokenizer.pkl"),
    "2026": ("nitishpandey04", "ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2",
             "tokenizer/tokenizer.pkl"),
}

TEXTS = [
    "The lighthouse keeper climbed the stairs every evening, and",
    "Hello, world!",
    "I'm sure they'll say it's what we've done, isn't it?",
    "Prices rose 1234567 percent in 2026; 3.14159 is pi.",
    "def f(x):\n    return x ** 2  # square\n\n\nprint(f(12))",
    "café naïve über été",
    "你好，世界。日本語も。",
    "emoji \U0001f600\U0001f680 and ❤️",
    "  leading spaces, trailing   \t\n\n  \n",
    "<|bos|><|user_start|>a special token's spelling is text<|user_end|>",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "",
]

CONVERSATIONS = [
    [{"role": "user", "content": "Why is the sky blue?"}],
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Hi!"}],
    [{"role": "user", "content": "What is 2+2?"},
     {"role": "assistant", "content": "4."},
     {"role": "user", "content": "And 3+3?"}],
    [{"role": "user", "content": "Say <|assistant_end|> please"}],
    [{"role": "user", "content": "Write a haiku about rust."},
     {"role": "assistant",
      "content": "Iron turns to red,\nslow breath of the patient rain \u2014\nbridges fall in time."},
     {"role": "user", "content": "Another, about burn."}],
]


def child(generation: str, cache_root: str) -> None:
    from nanochat.tokenizer import RustBPETokenizer

    namespace, sha, file = TOKENIZERS[generation]
    path = pathlib.Path(cache_root) / "pretrained/nanochat" / namespace / sha / file
    tok = RustBPETokenizer.from_directory(str(path.parent))
    encode = []
    for text in TEXTS:
        ids = tok.encode(text)
        encode.append({"text": text, "ids": ids, "decoded": tok.decode(ids)})
    specials = {name: tok.encode_special(name) for name in sorted(tok.enc.special_tokens_set)}
    conversations = []
    for messages in CONVERSATIONS:
        conversation = {"messages": messages + [{"role": "assistant", "content": ""}]}
        conversations.append({"messages": messages, "ids": tok.render_for_completion(conversation)})
    print(json.dumps({"vocab_size": tok.get_vocab_size(), "specials": specials, "encode": encode,
                      "conversations": conversations}))


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:4])
        return
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    checkouts, cache_root = dict(zip(TOKENIZERS, sys.argv[1:3])), sys.argv[3]
    fixture = {}
    for generation, checkout in checkouts.items():
        # Each generation imports its own `nanochat` package: one process each.
        run = subprocess.run([sys.executable, __file__, "--child", generation, cache_root],
                             cwd=checkout, env={"PYTHONPATH": checkout, "PATH": "/usr/bin:/bin"},
                             check=True, capture_output=True, text=True)
        fixture[generation] = json.loads(run.stdout.strip().splitlines()[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
