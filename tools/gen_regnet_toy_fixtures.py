"""Generate the regnet kit's toy fixtures: tiny seeded timm `RegNet`s and their answers.

Nine toys, each timm's own `timm.models.regnet.RegNet` of a small `RegNetCfg`: 7 units,
`w0=8`, `wa=6.0`, `wm=1.5`, which timm's width rule makes four stages of `(1, 2, 2, 2)` units,
`(8, 16, 24, 40)` wide, on its 32-wide `3x3` stem; 10 classes:

- `toy_x`: a `RegNetX`: grouped mixers of 4 channels a group, no squeeze-excite;
- `toy_y`: a `RegNetY`: the same with a squeeze-excite of `se_ratio=0.25` after the mixer, its
  reduction a quarter of the unit's input;
- `toy_tv`: a `_tv` arch's rounding: groups of 12 under `group_min_ratio=0.9`, which moves the
  widths (`make_divisible(16, 12)` rounds up to 24), with the squeeze-excite;
- `toy_headless`: `toy_y` with no classifier (`num_classes=0`), as a `SEER` trunk: the
  output is the pooled features;
- `toy_odd`: `toy_y` on 36x36 images, so the stages see odd sizes (18, 9, 5, 3);
- `toy_v`: a `RegNetV`: pre-activation units (`preact=True`: a bare stem conv, each unit's input
  normalized and activated before its path and its shortcut, a shortcut conv with no norm, no
  activation after the add, an activation after the last stage), `SiLU`, the squeeze-excite;
- `toy_v_avg`: `toy_v` with `avg` shortcuts (a `2x2` pool, then the conv), as `regnetv_064`;
- `toy_z`: a `RegNetZ`: a bottleneck 4x its unit's width (`bottle_ratio=4`), no shortcut convs
  (`downsample=None`: a unit that changes shape has no residual), linear unit outputs and an
  activation after the last stage (`linear_out=True`), `SiLU`, the squeeze-excite;
- `toy_z_h`: `toy_z` with a final `1x1` conv, norm and activation to 48 features
  (`num_features`), as `regnetz_040_h`.

The toy test builds each as bunsen's `RegNetContractConfig` of the same knobs.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
batch norms' running statistics random, so a misplaced norm shows. A two-image batch runs
through `forward_intermediates` (the stem, then each stage), the pooled features and the
logits. The fixture is the weights under timm's own state-dict names (what its hub safetensors
hold) and `io.*`: `images`, `stage_0` to `stage_4`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_regnet_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.regnet import RegNet, RegNetCfg  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_running_stats,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "regnet"

TOY = dict(depth=7, w0=8, wa=6.0, wm=1.5, group_size=4)

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_x": (dict(TOY), {}),
    "toy_y": (dict(TOY, se_ratio=0.25), {}),
    "toy_tv": (dict(TOY, group_size=12, group_min_ratio=0.9, se_ratio=0.25), {}),
    "toy_headless": (dict(TOY, se_ratio=0.25), dict(num_classes=0)),
    "toy_odd": (dict(TOY, se_ratio=0.25), {}),
    "toy_v": (dict(TOY, preact=True, act_layer="silu", se_ratio=0.25), {}),
    "toy_v_avg": (dict(TOY, preact=True, act_layer="silu", se_ratio=0.25, downsample="avg"), {}),
    "toy_z": (
        dict(TOY, bottle_ratio=4.0, downsample=None, linear_out=True, act_layer="silu", se_ratio=0.25),
        {},
    ),
    "toy_z_h": (
        dict(
            TOY, bottle_ratio=4.0, downsample=None, linear_out=True, act_layer="silu", se_ratio=0.25,
            num_features=48,
        ),
        {},
    ),
}

# The image side of each toy: odd stage sizes for `toy_odd`.
SIZES = {"toy_odd": 36}


def norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's norms."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            names.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
    return names


def main() -> None:
    for i, (name, (cfg, args)) in enumerate(TOYS.items()):
        args = dict(dict(num_classes=10), **args)
        model = build(lambda: RegNet(RegNetCfg(**cfg), **args), seed=2345 + i)
        norms = norm_names(model)
        init_weights(
            model,
            norm_bias=0.1,
            is_norm=lambda n: n in norms and n.endswith(".weight"),
        )
        init_running_stats(model)
        torch.manual_seed(55 + i)
        side = SIZES.get(name, 32)
        images = torch.randn(2, 3, side, side, dtype=torch.float64)
        with torch.no_grad():
            stages = model.forward_intermediates(images, intermediates_only=True)
            pooled = model.forward_head(model.forward_features(images), pre_logits=True)
            logits = model(images)
        assert len(stages) == 5, len(stages)
        io = {"images": images, "pooled": pooled, "logits": logits}
        io.update({f"stage_{k}": x for k, x in enumerate(stages)})
        write(OUT / f"{name}.safetensors", weights(model), io_tensors(io))
        print(name, {k: tuple(v.shape) for k, v in io.items()})


if __name__ == "__main__":
    main()
