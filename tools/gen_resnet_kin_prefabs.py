"""Generate the ResNet kin kits' prefabs from timm: every architecture with hub weights of
`res2net.py` and `resnest.py` (and the other `ResNet`-with-its-own-block files, as their kits
land).

Run with a Python that has timm (and torch):

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_resnet_kin_prefabs.py
    cargo +nightly fmt --all

Each of these files builds timm's `ResNet` with a block of its own (`block=`), its knobs in
`block_args`. For each architecture with a pretrained hub tag, timm builds it on the meta device
while the script records the kwargs its entrypoint passes to `build_model_with_cfg`; the `ResNet`
kwargs map onto `ResNetContractConfig` knobs, the block's onto the family's, and an architecture
with a kwarg the contract has no knob for is skipped, and named. The same models give the stage
table the prefab tests check the kit's widths against: the stem's convs' widths, and each stage's
depth, output width and first `1x1`'s width, read from timm's modules.

The prefab functions, the prefab table and the stage table are written between the
`gen_resnet_kin_prefabs` markers in each kit's `pretrained/prefabs.rs`; everything else in those
files is hand-written. The output is rustfmt's to lay out, hence the `fmt` after it.
"""

import argparse
import functools
import pathlib

import timm
import torch
from timm.models import _registry, res2net, resnest

KITS = pathlib.Path(__file__).resolve().parent.parent.joinpath(
    "crates/public/bunsen/src/kits/images/classification"
)


def norm(value):
    if isinstance(value, functools.partial):
        return {"partial": value.func.__name__, **{k: norm(v) for k, v in value.keywords.items()}}
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return [norm(v) for v in value]
    return getattr(value, "__name__", type(value).__name__)


def record(module):
    """Each pretrained architecture of timm's `module`: its kwargs and its model."""
    name = module.__name__.rsplit(".", 1)[-1]
    archs = sorted(
        {
            tag.split(".")[0]
            for tag in timm.list_pretrained()
            if _registry._model_to_module.get(tag.split(".")[0]) == name
            and getattr(timm.get_pretrained_cfg(tag), "hf_hub_id", None)
        }
    )
    seen = {}
    build = module.build_model_with_cfg

    def spy(cls, variant, pretrained, **kwargs):
        skip = ("pretrained_cfg", "pretrained_cfg_overlay", "feature_cfg", "pretrained_strict", "cache_dir")
        seen[variant] = {k: norm(v) for k, v in kwargs.items() if k not in skip}
        return build(cls, variant, pretrained, **kwargs)

    module.build_model_with_cfg = spy
    out = {}
    for arch in archs:
        with torch.device("meta"):
            model = timm.create_model(arch, pretrained=False)
        out[arch] = (seen[arch], model)
    module.build_model_with_cfg = build
    return out


def lower_resnet(kw, notes):
    """The `ResNet` kwargs of `kw` (popped) as a chain after the family's constructor: `(suffix,
    base_width, cardinality)`; or a `str`, a kwarg the contract lacks."""
    base_width = kw.pop("base_width", 64)
    cardinality = kw.pop("cardinality", 1)
    suffix = ""
    stem = kw.pop("stem_type", "")
    stem_width = kw.pop("stem_width", 64)
    avg_down = kw.pop("avg_down", False)
    if stem == "deep":
        suffix += f".with_deep_stem({stem_width}, {str(avg_down).lower()})"
        notes.append(f"the `deep` stem of {stem_width}")
        if avg_down:
            notes.append("average-pool shortcuts")
    elif stem or avg_down or stem_width != 64:
        return f"stem_type={stem!r}, stem_width={stem_width}, avg_down={avg_down}"
    return suffix, base_width, cardinality


def lower_res2net(kwargs):
    """`(expression, notes)`: a `Res2Net` prefab; or a `str`, the kwarg the contract lacks."""
    kw = dict(kwargs)
    if kw.pop("block") != "Bottle2neck":
        return "block (not Bottle2neck)"
    layers = kw.pop("layers")
    scale = kw.pop("block_args", {}).pop("scale", 4)
    notes = []
    lowered = lower_resnet(kw, notes)
    if isinstance(lowered, str):
        return lowered
    suffix, base_width, cardinality = lowered
    notes.insert(0, f"{scale} chunks of {base_width} per 64 planes" + (f", {cardinality} groups" if cardinality != 1 else ""))
    if kw:
        return ", ".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return f"Self::res2net(vec!{list(layers)}, {base_width}, {cardinality}, {scale}){suffix}", notes


def lower_resnest(kwargs):
    """`(expression, notes)`: a `ResNeSt` prefab; or a `str`, the kwarg the contract lacks."""
    kw = dict(kwargs)
    if kw.pop("block") != "ResNestBottleneck":
        return "block (not ResNestBottleneck)"
    layers = kw.pop("layers")
    block = dict(kw.pop("block_args", {}))
    radix = block.pop("radix", 1)
    if not block.pop("avd", False):
        return "block_args.avd=False"
    avd_first = block.pop("avd_first", False)
    if block:
        return ", ".join(f"block_args.{k}={v}" for k, v in sorted(block.items()))
    notes = []
    lowered = lower_resnet(kw, notes)
    if isinstance(lowered, str):
        return lowered
    suffix, base_width, cardinality = lowered
    head = f"radix {radix}"
    if (cardinality, base_width) != (1, 64):
        head += f", {cardinality} groups of {base_width} per 64 planes"
    if avd_first:
        head += ", the `avd` pool before the mixer"
    notes.insert(0, head)
    if kw:
        return ", ".join(f"{k}={v}" for k, v in sorted(kw.items()))
    args = f"vec!{list(layers)}, {base_width}, {cardinality}, {radix}, {str(avd_first).lower()}"
    return f"Self::resnest({args}){suffix}", notes


def stages(model):
    """The stem's conv widths, and each stage's `(depth, width, first 1x1's width)`."""
    if isinstance(model.conv1, torch.nn.Sequential):
        stem = [m.out_channels for m in model.conv1 if isinstance(m, torch.nn.Conv2d)]
    else:
        stem = [model.conv1.out_channels]
    out = []
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        blocks = list(layer)
        first = blocks[0]
        last = first.conv3 if hasattr(first, "conv3") else first.conv2
        width = last.out_channels if isinstance(last, torch.nn.Conv2d) else last.conv.out_channels
        conv1 = first.conv1 if isinstance(first.conv1, torch.nn.Conv2d) else first.conv1.conv
        out.append((len(blocks), width, conv1.out_channels))
    return stem, out


# Each kit: its timm module, its lowering, and its label for an architecture's doc.
FAMILIES = {
    "res2net": (res2net, lower_res2net, "`Res2Net`"),
    "resnest": (resnest, lower_resnest, "`ResNeSt`"),
}


def render(built, label):
    fns, rows, table = [], [], []
    for arch, (expr, notes), (stem, stage_rows) in built:
        doc = f"The `{arch}` prefab: {label} `{arch}`"
        if notes:
            doc += "; " + ", ".join(notes)
        fns.append(f"    /// {doc}.\n    pub fn {arch}() -> Self {{\n        {expr}\n    }}\n")
        rows.append(f'        ("{arch}", Self::{arch}),\n')
        cells = ", ".join(f"({d}, {w}, {m})" for d, w, m in stage_rows)
        table.append(f'        ("{arch}", &{stem}, [{cells}]),\n')
    return "\n".join(fns), "".join(rows), "".join(table)


def splice(text, marker, body):
    begin, end = f"// gen_resnet_kin_prefabs: begin {marker}\n", f"// gen_resnet_kin_prefabs: end {marker}"
    head, found, tail = text.partition(begin)
    assert found, f"no `{begin.strip()}` marker"
    indent = head[head.rfind("\n") + 1 :]
    _, found, tail = tail.partition(end)
    assert found, f"no `{end}` marker"
    return f"{head}{begin}{body}{indent}{end}{tail}"


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    for kit, (module, lower, label) in FAMILIES.items():
        built, skipped = [], []
        for arch, (kwargs, model) in record(module).items():
            lowered = lower(kwargs)
            if isinstance(lowered, str):
                skipped.append((arch, lowered))
                continue
            built.append((arch, lowered, stages(model)))
        # By depth (the units in all), then by name.
        built.sort(key=lambda b: (sum(d for d, _, _ in b[2][1]), b[0]))
        fns, rows, table = render(built, label)
        path = KITS / kit / "pretrained" / "prefabs.rs"
        text = path.read_text()
        path.write_text(splice(splice(splice(text, "fns", fns), "table", rows), "stages", table))
        for arch, reason in skipped:
            print(f"{kit}: skipped  {arch}: {reason}")
        print(f"{kit}: {len(built)} prefabs; {len(skipped)} skipped")


if __name__ == "__main__":
    main()
