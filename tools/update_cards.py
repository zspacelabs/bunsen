#!/usr/bin/env python3
"""Rebuild the models card in the READMEs: `bunsen-tool card --section kits`, as YAML.

    python3 tools/update_cards.py           # rewrite every README's card
    python3 tools/update_cards.py --check   # exit 1, naming each stale README, rewriting none

Each README holds the card between two markers:

    <!-- card: bunsen-tool card --section kits -->
    ...
    <!-- /card -->

and this replaces what lies between them with the card in a `yaml` code block. The card is
the models bunsen knows by name (`kits`: by kind and family, a name and a description
each), from the factories' rows, so it changes when a row does. `cargo make cards` runs
this; `cargo make cards-check` runs `--check`. Stdlib only; run from the repository root.
"""

import pathlib
import subprocess
import sys

READMES = [
    "README.md",
    "crates/public/bunsen/README.md",
    "crates/public/bunsen-tool/README.md",
]
BEGIN = "<!-- card: bunsen-tool card --section kits -->"
END = "<!-- /card -->"


def card() -> str:
    """The kits card, as YAML: from a build of `bunsen-tool` without its optional features."""
    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "bunsen-tool", "--no-default-features", "--",
         "card", "--section", "kits", "--format", "yml"],
        check=True, capture_output=True, text=True)
    return out.stdout


def spliced(text: str, block: str, path: str) -> str:
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < start:
        sys.exit(f"{path}: no `{BEGIN}` ... `{END}` markers")
    return text[: start + len(BEGIN)] + "\n" + block + text[end:]


def main() -> None:
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(__doc__)
    block = "\n```yaml\n" + card() + "```\n\n"
    stale = []
    for path in READMES:
        p = pathlib.Path(path)
        text = p.read_text()
        new = spliced(text, block, path)
        if new != text:
            stale.append(path)
            if not check:
                p.write_text(new)
    if check and stale:
        sys.exit("stale cards (run `cargo make cards`): " + ", ".join(stale))
    print(("stale: " if check else "updated: ") + (", ".join(stale) or "none"))


if __name__ == "__main__":
    main()
