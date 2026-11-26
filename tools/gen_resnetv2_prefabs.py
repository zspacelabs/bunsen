"""Generate the resnetv2 kit's prefabs from timm: every `resnetv2.py` architecture with hub weights.

Run with a Python that has timm (and torch):

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_resnetv2_prefabs.py
    cargo +nightly fmt --all

For each architecture `resnetv2.py` registers that has a pretrained hub tag, timm builds it on
the meta device while the script records the constructor kwargs its entrypoint passes
(`build_model_with_cfg`); each maps onto a `ResNetV2ContractConfig` knob, and an architecture
with a kwarg the contract has no knob for is skipped, and named. The same models give the stage
table the prefab tests check the kit's widths against: the stem's convs' widths, and each
stage's depth, output width and bottleneck width, read from timm's modules.

The prefab functions, the prefab table and the stage table are written between the
`gen_resnetv2_prefabs` markers in
`crates/public/bunsen/src/kits/images/classification/resnetv2/pretrained/prefabs.rs`; everything
else in the file is hand-written. The output is rustfmt's to lay out, hence the `fmt` after it.
"""

import argparse
import functools
import pathlib

import timm
import torch
from timm.models import _registry
from timm.models import resnetv2

PREFABS = pathlib.Path(__file__).resolve().parent.parent.joinpath(
    "crates/public/bunsen/src/kits/images/classification/resnetv2/pretrained/prefabs.rs"
)

NORMS = {"GroupNormAct": "GroupNorm", "BatchNormAct2d": "BatchNorm", "EvoNorm2dS0": "EvoNormS0"}
STEMS = {"fixed": "Fixed", "deep": "Deep", "": "Plain"}


def norm(value):
    if isinstance(value, functools.partial):
        return {"partial": value.func.__name__, **{k: norm(v) for k, v in value.keywords.items()}}
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return [norm(v) for v in value]
    return getattr(value, "__name__", type(value).__name__)


def record():
    """Each pretrained `resnetv2.py` architecture: its kwargs and its model."""
    archs = sorted(
        {
            name.split(".")[0]
            for name in timm.list_pretrained()
            if _registry._model_to_module.get(name.split(".")[0]) == "resnetv2"
            and getattr(timm.get_pretrained_cfg(name), "hf_hub_id", None)
        }
    )
    seen = {}
    build = resnetv2.build_model_with_cfg

    def spy(cls, variant, pretrained, **kwargs):
        skip = ("pretrained_cfg", "pretrained_cfg_overlay", "feature_cfg", "pretrained_strict", "cache_dir")
        seen[variant] = {k: norm(v) for k, v in kwargs.items() if k not in skip}
        return build(cls, variant, pretrained, **kwargs)

    resnetv2.build_model_with_cfg = spy
    out = {}
    for arch in archs:
        with torch.device("meta"):
            model = timm.create_model(arch, pretrained=False)
        out[arch] = (seen[arch], model)
    resnetv2.build_model_with_cfg = build
    return out


def lower(kwargs):
    """`(expression, notes)`: the Rust prefab; or a `str`, the kwarg the contract lacks."""
    kw = dict(kwargs)
    layers = kw.pop("layers")
    expr = f"Self::resnetv2(vec!{list(layers)})"
    notes = []
    channels = kw.pop("channels", None)
    if channels is not None and list(channels) != [256, 512, 1024, 2048]:
        expr += f".with_channels(vec!{list(channels)})"
    wf = kw.pop("width_factor", 1)
    if wf != 1:
        expr += f".with_width_factor({wf})"
        notes.append(f"{wf}x wide")
    stem = kw.pop("stem_type", "")
    if stem not in STEMS:
        return f"stem_type={stem!r}"
    if stem:
        expr += f".with_stem_type(ResNetV2Stem::{STEMS[stem]})"
        notes.append(f"the `{stem}` stem")
    conv = kw.pop("conv_layer", "create_conv2d")
    if isinstance(conv, dict) and conv.get("partial") == "StdConv2d" and conv.get("eps") == 1e-8:
        expr += ".with_std_conv(Some(BIT_STD_CONV_EPSILON))"
        notes.append("weight-standardized convs")
    elif conv != "create_conv2d":
        return f"conv_layer={conv}"
    nl = kw.pop("norm_layer", "GroupNormAct")
    if nl not in NORMS:
        return f"norm_layer={nl}"
    if nl != "GroupNormAct":
        expr += f".with_norm_layer(ResNetV2Norm::{NORMS[nl]})"
    notes.append({"GroupNormAct": "group norms", "BatchNormAct2d": "batch norms", "EvoNorm2dS0": "`EvoNorm-S0`"}[nl])
    if kw.pop("basic", False):
        expr += ".with_basic(true)"
        notes.append("basic units")
    ratio = kw.pop("bottle_ratio", 0.25)
    if ratio != 0.25:
        expr += f".with_bottle_ratio({float(ratio)!r})"
    if kw.pop("avg_down", False):
        expr += ".with_avg_down(true)"
        notes.append("average-pool shortcuts")
    if kw:
        return ", ".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return expr, notes


def stages(model):
    """The stem's conv widths, and each stage's `(depth, width, bottleneck width)`."""
    stem = [m.out_channels for n, m in model.stem.named_children() if n.startswith("conv")]
    out = []
    for stage in model.stages:
        blocks = list(stage.blocks)
        first = blocks[0]
        # A bottleneck's last conv is `conv3`, a basic unit's `conv2`.
        last = first.conv3 if hasattr(first, "conv3") else first.conv2
        out.append((len(blocks), last.out_channels, first.conv1.out_channels))
    return stem, out


def render(built):
    fns, rows, table = [], [], []
    for arch, (expr, notes), (stem, stage_rows) in built:
        label = arch.replace("resnetv2_", "").replace("_bit", "")
        kind = "`BiT`" if arch.endswith("_bit") else "`ResNetV2`"
        doc = f"The `{arch}` prefab: {kind} `{label}`"
        if notes:
            doc += "; " + ", ".join(notes)
        fns.append(f"    /// {doc}.\n    pub fn {arch}() -> Self {{\n        {expr}\n    }}\n")
        rows.append(f'        ("{arch}", Self::{arch}),\n')
        cells = ", ".join(f"({d}, {w}, {m})" for d, w, m in stage_rows)
        table.append(f'        ("{arch}", &{stem}, [{cells}]),\n')
    return "\n".join(fns), "".join(rows), "".join(table)


def splice(text, marker, body):
    begin, end = f"// gen_resnetv2_prefabs: begin {marker}\n", f"// gen_resnetv2_prefabs: end {marker}"
    head, found, tail = text.partition(begin)
    assert found, f"no `{begin.strip()}` marker"
    indent = head[head.rfind("\n") + 1 :]
    _, found, tail = tail.partition(end)
    assert found, f"no `{end}` marker"
    return f"{head}{begin}{body}{indent}{end}{tail}"


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    built, skipped = [], []
    for arch, (kwargs, model) in record().items():
        lowered = lower(kwargs)
        if isinstance(lowered, str):
            skipped.append((arch, lowered))
            continue
        built.append((arch, lowered, stages(model)))
    # timm's own, then `BiT`, each by depth, then width, then name.
    def key(b):
        arch = b[0]
        bit = arch.endswith("_bit")
        depth = int(arch.split("_")[1].split("x")[0].rstrip("d"))
        return (bit, depth, arch)

    built.sort(key=key)
    fns, rows, table = render(built)
    text = PREFABS.read_text()
    text = splice(splice(splice(text, "fns", fns), "table", rows), "stages", table)
    PREFABS.write_text(text)
    for arch, reason in skipped:
        print(f"skipped  {arch}: {reason}")
    print(f"{len(built)} prefabs; {len(skipped)} skipped")


if __name__ == "__main__":
    main()
