#!/usr/bin/env python3
"""Exercise `bunsen-asr` on a local GPU: every kit with its default model, or every model.

Builds `bunsen-asr` in release mode for one backend, runs each case of
`tools/exercise_asr.toml` over a real clip, and reports what each took and how
far its transcript is from the reference. Not a test suite and not for CI: it
fetches every model it runs into the weights cache, many GB for `all`, and is
worth running only on an accelerator.

Usage: tools/exercise_asr.py {defaults,all} BACKEND [--only TEXT] [--dry-run]

  defaults  each family's default model, each profile, Silero's VAD gate, and
            the diarizer (tools/exercise_asr.toml)
  all       every model `bunsen-asr family <family> list` and `diarize list`
            name, at the time of the run
  BACKEND   wgpu, vulkan, cuda or metal: the feature `bunsen-asr` is built
            with. Required; there is no CPU default.

`--only` keeps the cases whose name contains TEXT; `--dry-run` prints the
command lines without building or running anything. `cargo make
exercise-defaults BACKEND` and `cargo make exercise-all BACKEND` call this.

A report lands in `target/exercise/<set>-<backend>.json`. Exits 1 when a case
fails: an error, no output, a transcript over `max_wer`, or no speaker turn.
Stdlib only; run from the repository root.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

BACKENDS = ("wgpu", "vulkan", "cuda", "metal")
CONFIG = Path("tools/exercise_asr.toml")
BINARY = Path("target/release/bunsen-asr")

# A model row in `list`: `provider:ref`, optionally `(default)`, at column 0.
# `hf:org/repo` is a template, not a model, and is skipped.
LIST_ROW = re.compile(r"^(?P<id>[a-z-]+:\S+)(?: \(default\))?$")

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    """`n`, below a million, as words: 35 is `thirty five`."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    rest = n % 1000
    return number_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + number_words(rest))


def words(text: str) -> list[str]:
    """Lowercase words, without punctuation; hyphens split; numbers as words."""
    out = []
    for token in re.findall(r"[\w']+", text.lower().replace("-", " ")):
        if token.isdigit() and int(token) < 1_000_000:
            out.extend(number_words(int(token)).split())
        else:
            out.append(token.strip("'"))
    return [w for w in out if w]


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Word error rate: the edit distance over the reference's length."""
    row = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hypothesis, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(len(reference), 1)


def listed_models(command: list[str]) -> list[str]:
    """The model ids a `list` command prints."""
    out = subprocess.run([str(BINARY), *command, "list"], check=True, capture_output=True, text=True).stdout
    return [m["id"] for line in out.splitlines() if (m := LIST_ROW.match(line))]


def all_cases(config: dict) -> list[dict]:
    """Every listed model of every family, transcribing, and of the diarizer."""
    cases = []
    for family in config["families"]:
        for model in listed_models(["family", family]):
            cases.append({
                "name": f"{family} {model}",
                "kind": "text",
                "args": ["family", family, "transcribe", "--model", model],
            })
    for model in listed_models(["diarize"]):
        cases.append({"name": f"diarize {model}", "kind": "turns", "args": ["diarize", "run", "--model", model]})
    return cases


def run_case(case: dict, clip: str, reference: list[str], max_wer: float) -> dict:
    """Runs one case; its outcome, timing and score."""
    command = [str(BINARY), *case["args"], "--json", "--quiet", clip]
    start = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True)
    seconds = time.monotonic() - start
    result = {"name": case["name"], "command": command[1:], "seconds": round(seconds, 1)}
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        return {**result, "ok": False, "error": " | ".join(tail) or f"exit {proc.returncode}"}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {**result, "ok": False, "error": "no JSON output"}
    output = json.loads(lines[-1])
    if case["kind"] == "turns":
        turns = output.get("turns", [])
        return {**result, "ok": bool(turns), "turns": len(turns), "speakers": len(output.get("speakers", []))}
    text = output.get("text", "")
    score = wer(reference, words(text))
    return {
        **result,
        "ok": bool(text.strip()) and score <= max_wer,
        "wer": round(score, 3),
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", choices=("defaults", "all"))
    parser.add_argument("backend", choices=BACKENDS, help="the accelerator `bunsen-asr` is built for")
    parser.add_argument("--only", help="keep the cases whose name contains this")
    parser.add_argument("--dry-run", action="store_true", help="print the command lines, build and run nothing")
    args = parser.parse_args()

    config = tomllib.loads(CONFIG.read_text())
    build = ["cargo", "build", "--release", "-p", "bunsen-asr", "--features", args.backend]
    if args.dry_run and args.set == "all" and not BINARY.is_file():
        print(" ".join(build))
        print(f"then every model `{BINARY} family <family> list` and `diarize list` name")
        return 0
    if not args.dry_run:
        subprocess.run(build, check=True)

    cases = config["defaults"] if args.set == "defaults" else all_cases(config)
    if args.only:
        cases = [c for c in cases if args.only in c["name"]]
    if args.dry_run:
        print(" ".join(build))
        for case in cases:
            print(" ".join([str(BINARY), *case["args"], "--json", "--quiet", config["clip"]]))
        return 0

    reference = words(Path(config["transcript"]).read_text())
    results = []
    for case in cases:
        print(f"-- {case['name']}", flush=True)
        result = run_case(case, config["clip"], reference, config["max_wer"])
        results.append(result)
        detail = result.get("error") or (
            f"wer {result['wer']:.3f}" if "wer" in result else f"{result['turns']} turns, {result['speakers']} speakers"
        )
        print(f"   {'ok  ' if result['ok'] else 'FAIL'} {result['seconds']:7.1f} s  {detail}", flush=True)

    report = Path("target/exercise") / f"{args.set}-{args.backend}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"set": args.set, "backend": args.backend, "results": results}, indent=2) + "\n")

    failed = [r["name"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)} of {len(results)} ok, on {args.backend}; report: {report}")
    for name in failed:
        print(f"  FAIL {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
