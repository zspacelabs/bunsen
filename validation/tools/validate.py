#!/usr/bin/env python3
"""Run the validation crates under a profile: the models it chooses, their tests alone.

    python3 validation/tools/validate.py list   [--profile P] [--crate C]...
    python3 validation/tools/validate.py filter [--profile P] [--crate C]... [--backend B] [--dtypes D,...]
    python3 validation/tools/validate.py run    [--profile P] [--crate C]... [--backend B] [--dtypes D,...] [--dry-run] [-- NEXTEST_ARGS]
    python3 validation/tools/validate.py check  [--crate C]...

`validation/models.yml` is the catalog: every model a crate checks, with its size, whether
it is a representative (`core`), what it varies, and the dtypes it is checked in. A profile
(`validation/profiles/<name>.yml`, or a path) chooses models from it:

    extends: medium-core          # a profile, or a list of them, whose choice this adds to
    include: [{size: medium, core: true}, {crate: qwen3, model: qwen3-1.7b}]
    deny:    [{crate: voxtral}]   # removed from everything included, extended ones too
    dtypes:  [f32, bf16]          # `bf16` checks run on cuda alone; default [f32]
    backend: cuda                 # the default for --backend

A selector matches a model when every key it has does: `crate`, `model`, `size` (one or a
list) and `core`. The profile is `--profile`, else `BUNSEN_VALIDATE_PROFILE`, else
`small-core`.

`run` runs `cargo nextest` in the validation workspace, under its `validate` profile
(`validation/.config/nextest.toml`), over the converted crates (`runner: nextest`), with
the crate's `features`, its `reference` feature when a chosen model needs it, and the
backend's; and a filter that drops the tests of every model not chosen (a test of a model
has the model's id in its path: `qwen3-0.6b` is the segment `qwen3_0_6b`) and, without
`bf16`, the `cuda_bf16` test binaries. Tests of no model (the fixture checks) always run.
The checkpoints are fetched by the tests into the weights cache (`BUNSEN_CACHE_DIR`, or the
user's cache directory); `BUNSEN_VALIDATE_OFFLINE=1` forbids it.

`check` lists the converted crates' tests (it builds them) and checks the catalog against
them: every model has a test, and every test of a model is of a catalog model.

Needs PyYAML (`pip install pyyaml`).
"""

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("validate.py needs PyYAML: pip install pyyaml")

VALIDATION = pathlib.Path(__file__).resolve().parents[1]
CATALOG = VALIDATION / "models.yml"
PROFILES = VALIDATION / "profiles"
SIZES = ("small", "medium", "large")
SELECTOR_KEYS = {"crate", "model", "size", "core"}
BACKENDS = ("wgpu", "vulkan", "cuda", "metal")


def ident(model: str) -> str:
    """A model id's segment in a test path: its last `/` part, `-` and `.` as `_`."""
    return re.sub(r"[-.]", "_", model.rsplit("/", 1)[-1])


def load_catalog() -> dict:
    catalog = yaml.safe_load(CATALOG.read_text())["crates"]
    for name, crate in catalog.items():
        for model, entry in crate["models"].items():
            if entry.get("size") not in SIZES:
                sys.exit(f"models.yml: {name}/{model}: size must be one of {SIZES}")
            for key in ("gb", "core", "dtypes", "varies"):
                if key not in entry:
                    sys.exit(f"models.yml: {name}/{model}: no `{key}`")
            unknown = set(entry) - {"size", "gb", "core", "reference", "dtypes", "varies"}
            if unknown:
                sys.exit(f"models.yml: {name}/{model}: unknown keys {sorted(unknown)}"
                         " (a `varies` with a comma, unquoted in a {...} entry?)")
    return catalog


def profile_path(name: str) -> pathlib.Path:
    path = pathlib.Path(name)
    if path.suffix in (".yml", ".yaml"):
        if path.exists():
            return path
        path = PROFILES / path.name
    else:
        path = PROFILES / f"{name}.yml"
    if not path.exists():
        known = sorted(p.stem for p in PROFILES.glob("*.yml"))
        sys.exit(f"no profile {name!r}: a path to a .yml, or one of {known}")
    return path


def load_profile(name: str, seen: tuple = ()) -> dict:
    """The profile, its `extends` folded in: include and deny as lists of selectors."""
    if name in seen:
        sys.exit(f"profiles extend in a cycle: {' -> '.join(seen + (name,))}")
    raw = yaml.safe_load(profile_path(name).read_text()) or {}
    unknown = set(raw) - {"extends", "include", "deny", "dtypes", "backend"}
    if unknown:
        sys.exit(f"profile {name}: unknown keys {sorted(unknown)}")
    extends = raw.get("extends") or []
    if isinstance(extends, str):
        extends = [extends]
    profile = {"include": [], "deny": [], "dtypes": None, "backend": None}
    for parent in extends:
        base = load_profile(parent, seen + (name,))
        profile["include"] += base["include"]
        profile["deny"] += base["deny"]
        profile["dtypes"] = profile["dtypes"] or base["dtypes"]
        profile["backend"] = profile["backend"] or base["backend"]
    for key in ("include", "deny"):
        for selector in raw.get(key) or []:
            bad = set(selector) - SELECTOR_KEYS
            if bad:
                sys.exit(f"profile {name}: {key}: unknown selector keys {sorted(bad)}")
            profile[key].append(selector)
    profile["dtypes"] = raw.get("dtypes", profile["dtypes"])
    profile["backend"] = raw.get("backend", profile["backend"])
    return profile


def matches(selector: dict, crate: str, model: str, entry: dict) -> bool:
    for key, want in selector.items():
        have = {"crate": crate, "model": model, "size": entry["size"], "core": entry["core"]}[key]
        wants = want if isinstance(want, list) else [want]
        if have not in wants:
            return False
    return True


def choose(catalog: dict, profile: dict, crates: list[str]) -> dict:
    """Per crate, its models: chosen or not."""
    chosen = {}
    for crate, spec in catalog.items():
        if crates and crate not in crates:
            continue
        chosen[crate] = {
            model: any(matches(s, crate, model, e) for s in profile["include"])
            and not any(matches(s, crate, model, e) for s in profile["deny"])
            for model, e in spec["models"].items()
        }
    return chosen


def segment_regex(models: list[str]) -> str:
    return "(^|::)(" + "|".join(ident(m) for m in models) + ")(::|$)"


def nextest_filter(catalog: dict, chosen: dict, dtypes: list[str]) -> tuple[str, list[str]]:
    """The filterset over the converted crates, and the features they run with."""
    parts, features = [], []
    for crate, models in chosen.items():
        spec = catalog[crate]
        if spec["runner"] != "nextest":
            continue
        package = spec["package"]
        expr = f"package({package})"
        dropped = [m for m, keep in models.items() if not keep]
        if dropped:
            expr += f" - test(/{segment_regex(dropped)}/)"
        if "bf16" not in dtypes:
            expr += " - binary(cuda_bf16)"
        parts.append(f"({expr})")
        features += [f"{package}/{f}" for f in spec.get("features", [])]
        if spec.get("reference") and any(
            keep and spec["models"][m].get("reference") for m, keep in models.items()
        ):
            features.append(f"{package}/{spec['reference']}")
    return " | ".join(parts), features


def cmd_list(args) -> None:
    catalog, profile = load_catalog(), load_profile(args.profile)
    chosen = choose(catalog, profile, args.crate)
    total = 0.0
    print(f"profile {args.profile}; dtypes {profile['dtypes'] or ['f32']}")
    for crate, models in chosen.items():
        spec = catalog[crate]
        note = "" if spec["runner"] == "nextest" else "  (not converted: cargo make validate-" + crate + ")"
        print(f"\n{crate}{note}")
        for model, keep in models.items():
            e = spec["models"][model]
            total += e["gb"] if keep else 0.0
            mark = "+" if keep else "-"
            core = "core" if e["core"] else "    "
            print(f"  {mark} {model:42} {e['size']:6} {core} {e['gb']:5.1f} GB  {e['varies']}")
    print(f"\nchosen: {total:.1f} GB on a cold cache")


def nextest_command(args) -> list[str]:
    catalog, profile = load_catalog(), load_profile(args.profile)
    backend = args.backend or profile["backend"] or "wgpu"
    if backend not in BACKENDS:
        sys.exit(f"backend {backend!r}: one of {BACKENDS}")
    dtypes = args.dtypes.split(",") if args.dtypes else (profile["dtypes"] or ["f32"])
    if "bf16" in dtypes and backend != "cuda":
        print(f"bf16 checks run on cuda alone; dropping them on {backend}", file=sys.stderr)
        dtypes = [d for d in dtypes if d != "bf16"]
    chosen = choose(catalog, profile, args.crate)
    expr, features = nextest_filter(catalog, chosen, dtypes)
    if not expr:
        sys.exit("no converted crate in the selection")
    packages = [catalog[c]["package"] for c in chosen if catalog[c]["runner"] == "nextest"]
    features += [f"{p}/{backend}" for p in packages]
    command = ["cargo", "nextest", "run", "--release", "--profile", "validate", "--no-tests", "warn"]
    for p in packages:
        command += ["-p", p]
    command += ["--features", ",".join(features), "-E", expr]
    return command + args.nextest_args


def cmd_filter(args) -> None:
    print(shlex.join(nextest_command(args)))


def cmd_run(args) -> None:
    command = nextest_command(args)
    print("+ (cd validation &&", shlex.join(command) + ")", file=sys.stderr)
    if args.dry_run:
        return
    sys.exit(subprocess.run(command, cwd=VALIDATION).returncode)


def cmd_check(args) -> None:
    catalog = load_catalog()
    failed = False
    for crate, spec in catalog.items():
        if spec["runner"] != "nextest" or (args.crate and crate not in args.crate):
            continue
        package = spec["package"]
        features = list(spec.get("features", []))
        if spec.get("reference"):
            features.append(spec["reference"])
        out = subprocess.run(
            ["cargo", "nextest", "list", "-p", package, "--features", ",".join(features),
             "--message-format", "oneline"],
            cwd=VALIDATION, check=True, capture_output=True, text=True).stdout
        tests = [line.split(" ", 1)[1] for line in out.splitlines() if line.startswith(package + " ")]
        models = {ident(m): m for m in spec["models"]}
        free = []
        for model, seg in ((m, ident(m)) for m in spec["models"]):
            if not any(re.search(segment_regex([model]), t) for t in tests):
                print(f"{crate}: {model} has no test (no `{seg}` in a test path)")
                failed = True
        for t in tests:
            if not any(s in models for s in t.split("::")):
                free.append(t)
        print(f"{crate}: {len(tests)} tests; of no model, always run: {free}")
    sys.exit(1 if failed else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("list", cmd_list), ("filter", cmd_filter), ("run", cmd_run), ("check", cmd_check)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--crate", action="append", default=[], help="only this crate (repeatable)")
        if name != "check":
            p.add_argument("--profile", default=os.environ.get("BUNSEN_VALIDATE_PROFILE", "small-core"))
        if name in ("filter", "run"):
            p.add_argument("--backend", choices=BACKENDS)
            p.add_argument("--dtypes", help="over the profile's: f32, bf16 (cuda alone), comma-separated")
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
        if name in ("filter", "run"):
            p.add_argument("nextest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if getattr(args, "nextest_args", None) and args.nextest_args[0] == "--":
        args.nextest_args = args.nextest_args[1:]
    args.fn(args)


if __name__ == "__main__":
    main()
