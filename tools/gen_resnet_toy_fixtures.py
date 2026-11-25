"""Generate the resnet kit's toy fixtures: tiny seeded timm `ResNet`s and their answers.

Twelve toys, each timm's own `timm.models.resnet.ResNet` with four stages of one block and
narrow stage widths (its `channels`), on timm's fixed 64-wide 7x7 stem, 10 classes (timm's
forward runs exactly four stages):

- `toy_basic`: `BasicBlock`s, each stage `(8, 16, 32, 64)` planes wide, the first after a
  downsample from the stem's 64;
- `toy_bottleneck`: `Bottleneck`s of `(8, 16, 32, 64)` planes, `(32, 64, 128, 256)` out (4x
  the planes), each with a downsample;
- `toy_gn`: `Bottleneck`s of 32 planes in every stage, 128 out, with timm's `groupnorm`
  (32 groups) for every norm, as `resnet50_gn` has;
- `toy_resnext`: `Bottleneck`s of `(8, 16, 32, 64)` planes whose mixers are 4 groups of
  `base_width=16` per 64 planes (`(8, 16, 32, 64)` wide), as a `resnext50_32x4d`'s are;
- `toy_deep`: `BasicBlock`s on a deep stem (`stem_width=16`: 16, 16, 32) with `avg_down`
  shortcuts, as a `resnet18d`'s;
- `toy_tiered`: `Bottleneck`s on a tiered stem (`stem_width=16`: 12, 16, 32) with
  `avg_down`, as a `resnet26t`'s;
- `toy_dilated`: `toy_tiered`'s geometry at `output_stride=8`: its last two stages dilate,
  and their `avg_down` pools are timm's `AvgPool2dSame`;
- `toy_se`: `Bottleneck`s of `(8, 16, 32, 64)` planes with timm's `'se'` attention
  (reductions of 1/16 in 8s: 8, 8, 8, 16), as a `seresnet50`'s;
- `toy_eca`: the same with `'eca'` (kernels 3, 3, 5, 5), as an `ecaresnet50d`'s blocks;
- `toy_blur`: `BasicBlock`s anti-aliased by `BlurPool2d` (the stem's max pool at stride 1,
  then a blur), as a `resnetblur18`'s;
- `toy_aa`: `Bottleneck`s on a deep stem with `avg_down`, anti-aliased by `nn.AvgPool2d` (the
  stem's pool a `2x2` average), as a `resnetaa50d`'s;
- `toy_blur_dilated`: `Bottleneck`s anti-aliased by `BlurPool2d` at `output_stride=16`: the
  last stage's first unit blurs at stride 1 where it would have strided.

The stem and blur toys run on 36x36 images, so the stages see odd sizes (9, 5, 3) and the shortcuts'
ceil-mode pools a partial window; the others on 32x32. (`toy_aa` can't be odd: timm's own
`AvgPool2d(2)` floors an odd side that its `avg_down` shortcut's ceil-mode pool rounds up.)

The toy test builds each as bunsen's `ResNetContractConfig`, with the same `channels`.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
batch norms' running statistics random, so a misplaced norm shows. A two-image
`[2, 3, 32, 32]` batch runs through `forward_intermediates` (the stem's activation, then
each stage), the pooled features and the logits. The fixture is the weights under timm's
own state-dict names (what its hub safetensors hold) and `io.*`: `images`, `stage_0` to
`stage_4`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_resnet_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.layers import BlurPool2d  # noqa: E402
from timm.models.resnet import BasicBlock, Bottleneck, ResNet  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_running_stats,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "resnet"

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_basic": dict(block=BasicBlock, channels=(8, 16, 32, 64)),
    "toy_bottleneck": dict(block=Bottleneck, channels=(8, 16, 32, 64)),
    "toy_gn": dict(block=Bottleneck, channels=(32, 32, 32, 32), norm_layer="groupnorm"),
    "toy_resnext": dict(block=Bottleneck, channels=(8, 16, 32, 64), cardinality=4, base_width=16),
    "toy_deep": dict(
        block=BasicBlock, channels=(8, 16, 32, 64), stem_type="deep", stem_width=16,
        avg_down=True,
    ),
    "toy_tiered": dict(
        block=Bottleneck, channels=(8, 16, 32, 64), stem_type="deep_tiered", stem_width=16,
        avg_down=True,
    ),
    "toy_dilated": dict(
        block=Bottleneck, channels=(8, 16, 32, 64), stem_type="deep_tiered", stem_width=16,
        avg_down=True, output_stride=8,
    ),
    "toy_se": dict(block=Bottleneck, channels=(8, 16, 32, 64), block_args=dict(attn_layer="se")),
    "toy_eca": dict(
        block=Bottleneck, channels=(8, 16, 32, 64), block_args=dict(attn_layer="eca"),
    ),
    "toy_blur": dict(block=BasicBlock, channels=(8, 16, 32, 64), aa_layer=BlurPool2d),
    "toy_aa": dict(
        block=Bottleneck, channels=(8, 16, 32, 64), stem_type="deep", stem_width=16,
        avg_down=True, aa_layer=torch.nn.AvgPool2d,
    ),
    "toy_blur_dilated": dict(
        block=Bottleneck, channels=(8, 16, 32, 64), aa_layer=BlurPool2d, output_stride=16,
    ),
}

# The image side of each toy: odd stage sizes for the stem toys.
SIZES = {
    name: 36
    for name in ("toy_deep", "toy_tiered", "toy_dilated", "toy_blur", "toy_blur_dilated")
}


def norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's norms (`bn1`, `downsample.1`, ...)."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            names.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
    return names


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        model = build(
            lambda: ResNet(layers=(1, 1, 1, 1), num_classes=10, **args),
            seed=1234 + i,
        )
        norms = norm_names(model)
        init_weights(
            model,
            norm_bias=0.1,
            is_norm=lambda n: n in norms and n.endswith(".weight"),
        )
        init_running_stats(model)
        torch.manual_seed(99 + i)
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
