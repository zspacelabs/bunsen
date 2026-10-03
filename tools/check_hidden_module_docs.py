#!/usr/bin/env python3
"""Fail if a non-public module carries `//!` prose that rustdoc will never render.

A private module's inner docs (`//!`) do not appear in the API docs: with the
crate's `mod x; pub use x::*;` layout, rustdoc shows the re-exported items and
nothing of the file. Prose written there is invisible and rots unchecked.
STYLE.md ("Module docs must render") puts item prose on the item's `///` and
lifecycle prose in the public parent's `//!`. A private file may keep a single
`//!` title line.

Usage: tools/check_hidden_module_docs.py [ROOT ...]   (default: crates)

Exits 1 and lists offenders when any non-public, non-test module has more
than one non-blank `//!` line. Stdlib only; run from the repository root.
"""

import os
import re
import sys

# `mod name;` with an optional visibility and optional attributes before it.
MOD_DECL = re.compile(
    r"^(?P<attrs>(?:[ \t]*#\[[^\n]*\][ \t]*\n)*)"
    r"[ \t]*(?P<vis>pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>\w+)\s*;",
    re.M,
)
MAX_TITLE_LINES = 1


def module_file(parent: str, name: str):
    """Return the file holding module `name` declared in `parent`, if any."""
    directory, filename = os.path.split(parent)
    if filename not in ("mod.rs", "lib.rs", "main.rs"):
        directory = os.path.join(directory, filename[:-3])
    for candidate in (
        os.path.join(directory, name + ".rs"),
        os.path.join(directory, name, "mod.rs"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def inner_doc_lines(path: str) -> int:
    """Count non-blank `//!` lines in the file's leading inner-doc block."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("//!"):
                if stripped[3:].strip():
                    count += 1
            elif stripped == "" or stripped.startswith("#!"):
                continue
            else:
                break
    return count


def is_test_module(name: str, attrs: str) -> bool:
    return "test" in attrs or name in ("tests", "test") or name.endswith(("_test", "_tests"))


def main(roots):
    offenders = []
    for root in roots:
        for directory, _, files in os.walk(root):
            if "/target" in directory or "/." in directory:
                continue
            for filename in files:
                if not filename.endswith(".rs"):
                    continue
                parent = os.path.join(directory, filename)
                with open(parent, encoding="utf-8") as f:
                    source = f.read()
                for decl in MOD_DECL.finditer(source):
                    vis = (decl.group("vis") or "").strip()
                    if vis == "pub":
                        continue
                    name = decl.group("name")
                    if is_test_module(name, decl.group("attrs")):
                        continue
                    child = module_file(parent, name)
                    if child is None:
                        continue
                    lines = inner_doc_lines(child)
                    if lines > MAX_TITLE_LINES:
                        offenders.append((child, lines))
    for path, lines in sorted(offenders):
        print(f"{path}: {lines} lines of `//!` in a non-public module (rustdoc never renders them)")
    if offenders:
        print(
            f"\n{len(offenders)} non-public module(s) carry hidden prose. Move it to the "
            "item's `///` or the public parent's `//!` (STYLE.md, 'Module docs must render').",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["crates"]))
