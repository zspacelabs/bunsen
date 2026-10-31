#!/usr/bin/env python3
"""Check that `validation/Cargo.lock` builds bunsen's crates against the root lock's versions.

The validation workspace depends on the bunsen workspace's crates by path, and has a lockfile of its
own. Every package those crates reach in `validation/Cargo.lock` must be one `Cargo.lock` has, at
the same version, or the validation suites test a dependency set the bunsen workspace never
builds. Packages only the validation crates reach (`burn-onnx`, the protobuf crates) are theirs
to choose.

Stdlib only; run from the repository root.
"""

import sys
import tomllib
from pathlib import Path

REFRESH = """\
Refresh the validation lock from the root's:

    cp Cargo.lock validation/Cargo.lock && cargo metadata --manifest-path validation/Cargo.toml >/dev/null"""


def packages(lock: Path) -> list[dict]:
    with lock.open("rb") as f:
        return tomllib.load(f)["package"]


def main() -> int:
    root = {(p["name"], p["version"]) for p in packages(Path("Cargo.lock"))}
    validation = packages(Path("validation/Cargo.lock"))
    members = {p.parent.name for p in Path("validation").glob("*/*/Cargo.toml")}

    by_name: dict[str, list[dict]] = {}
    for p in validation:
        by_name.setdefault(p["name"], []).append(p)

    def resolve(dep: str) -> dict:
        # A lock dependency is `name`, or `name version` when several versions are locked.
        name, _, version = dep.partition(" ")
        candidates = by_name[name]
        return next(p for p in candidates if not version or p["version"] == version)

    # bunsen's crates: the path packages that are not validation members.
    roots = [p for p in validation if "source" not in p and p["name"] not in members]
    seen: set[tuple[str, str]] = set()
    stack = list(roots)
    while stack:
        p = stack.pop()
        key = (p["name"], p["version"])
        if key in seen:
            continue
        seen.add(key)
        stack.extend(resolve(d) for d in p.get("dependencies", []))

    missing = sorted(seen - root)
    for name, version in missing:
        print(f"validation/Cargo.lock: {name} {version} is not in Cargo.lock")
    if missing:
        print(REFRESH)
        return 1
    print(f"validation/Cargo.lock: {len(seen)} packages reached from bunsen's crates, all as Cargo.lock has them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
