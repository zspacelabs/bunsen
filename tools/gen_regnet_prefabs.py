"""Generate the regnet kit's prefabs from timm: every `regnet.py` architecture with hub weights.

Run with a Python that has timm (and torch):

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_regnet_prefabs.py
    cargo +nightly fmt --all

Each architecture's `RegNetCfg` (timm's `model_cfgs`) becomes a `RegNetContractConfig` prefab; an
architecture with a knob the contract has no field for is skipped, and the script names it. The
same architectures, built on the meta device, give the stage table the prefab tests check the
kit's width generator against: each stage's depth, output width, bottleneck width and group
count, read from timm's modules, so the check does not depend on the script re-deriving timm's
float32 width arithmetic.

The prefab functions, the prefab table and the stage table are written between the
`gen_regnet_prefabs` markers in
`crates/public/bunsen/src/kits/images/classification/regnet/pretrained/prefabs.rs`; everything
else in the file is hand-written. The output is rustfmt's to lay out, hence the `fmt` after it.
"""

import argparse
import dataclasses
import pathlib

import timm
import torch
from timm.models import _registry
from timm.models import regnet

PREFABS = pathlib.Path(__file__).resolve().parent.parent.joinpath(
    "crates/public/bunsen/src/kits/images/classification/regnet/pretrained/prefabs.rs"
)

DEFAULTS = regnet.RegNetCfg()

ACTS = {"relu": "Relu", "silu": "Silu"}


def archs():
    """The `regnet.py` architectures with a pretrained hub tag."""
    return sorted(
        {
            name.split(".")[0]
            for name in timm.list_pretrained()
            if _registry._model_to_module.get(name.split(".")[0]) == "regnet"
            and getattr(timm.get_pretrained_cfg(name), "hf_hub_id", None)
        }
    )


def lower(arch, cfg):
    """`(expression, notes)`: the Rust prefab of `cfg`; or a `str`, the knob the contract lacks."""
    expr = f"Self::regnet({cfg.depth}, {cfg.w0}, {cfg.wa!r}, {cfg.wm!r}, {cfg.group_size})"
    notes = []
    known = {"depth", "w0", "wa", "wm", "group_size"}
    for field in dataclasses.fields(cfg):
        name, value = field.name, getattr(cfg, field.name)
        if name in known or value == getattr(DEFAULTS, name):
            continue
        if name == "se_ratio":
            expr += f".with_se_ratio({value!r})"
            notes.append(f"squeeze-excite of {value!r}")
        elif name == "group_min_ratio":
            expr += f".with_group_min_ratio({value!r})"
            notes.append(f"widths rounded to the groups with a floor of {value!r}")
        elif name == "act_layer" and value in ACTS:
            expr += f".with_act_layer(BunsenActivationConfig::{ACTS[value]})"
            notes.append(f"`{value}`")
        else:
            return f"{name}={value!r}"
    return expr, notes


def stages(arch):
    """Each stage of timm's `arch`: `(depth, width, bottleneck width, groups)`."""
    with torch.device("meta"):
        model = timm.create_model(arch, pretrained=False)
    out = []
    for i in range(1, 5):
        stage = getattr(model, f"s{i}")
        blocks = list(stage.children())
        conv2 = blocks[0].conv2.conv
        out.append((len(blocks), blocks[0].conv3.conv.out_channels, conv2.out_channels, conv2.groups))
    return out


def render(built):
    fns, rows, table = [], [], []
    for arch, (expr, notes), stage_rows in built:
        kind = {"x": "`RegNetX`", "y": "`RegNetY`"}[arch[len("regnet")]]
        flops = arch.split("_")[1]
        doc = f"The `{arch}` prefab: {kind} of {int(flops) / 10:g} GF"
        if notes:
            doc += "; " + ", ".join(notes)
        fns.append(f"    /// {doc}.\n    pub fn {arch}() -> Self {{\n        {expr}\n    }}\n")
        rows.append(f'        ("{arch}", Self::{arch}),\n')
        cells = ", ".join(f"({d}, {w}, {m}, {g})" for d, w, m, g in stage_rows)
        table.append(f'        ("{arch}", [{cells}]),\n')
    return "\n".join(fns), "".join(rows), "".join(table)


def splice(text, marker, body):
    begin, end = f"// gen_regnet_prefabs: begin {marker}\n", f"// gen_regnet_prefabs: end {marker}"
    head, found, tail = text.partition(begin)
    assert found, f"no `{begin.strip()}` marker"
    indent = head[head.rfind("\n") + 1 :]
    _, found, tail = tail.partition(end)
    assert found, f"no `{end}` marker"
    return f"{head}{begin}{body}{indent}{end}{tail}"


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    built, skipped = [], []
    for arch in archs():
        lowered = lower(arch, regnet.model_cfgs[arch])
        if isinstance(lowered, str):
            skipped.append((arch, lowered))
            continue
        built.append((arch, lowered, stages(arch)))
    # RegNetX, then RegNetY, each by its FLOPs (the name's number), then its name.
    built.sort(key=lambda b: (b[0][len("regnet")], int(b[0].split("_")[1]), b[0]))
    fns, rows, table = render(built)
    text = PREFABS.read_text()
    text = splice(splice(splice(text, "fns", fns), "table", rows), "stages", table)
    PREFABS.write_text(text)
    for arch, reason in skipped:
        print(f"skipped  {arch}: {reason}")
    print(f"{len(built)} prefabs; {len(skipped)} skipped")


if __name__ == "__main__":
    main()
