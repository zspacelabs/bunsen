"""Generate the resnetv2 kit's toy fixtures: tiny seeded timm `ResNetV2`s and their answers.

Eight toys, each timm's own `timm.models.resnetv2.ResNetV2`, four stages of one unit, 10 classes:

- `toy_bit`: a `BiT`'s knobs: weight-standardized convs (`StdConv2d`, eps 1e-8), 32-group norms
  (`GroupNormAct`), the `fixed` stem (a bare `7x7` conv, then a zero pad and an unpadded max
  pool), pre-activation bottlenecks `(128, 128, 256, 256)` wide on a 32-wide stem;
- `toy_bit_odd`: `toy_bit` on 36x36 images, so the zero-padded pool and the stages see odd sizes;
- `toy_bn`: batch norms (`BatchNormAct2d`), plain convs, the plain stem (the `7x7` conv and a
  padded max pool), bottlenecks `(64, 128, 128, 256)` wide;
- `toy_basic_deep`: basic units (`PreActBasic`), `(16, 32, 64, 128)` wide, a deep stem (three
  `3x3` convs, the last bare) and average-pool shortcuts (`avg_down`), batch norms, as
  `resnetv2_18d`;
- `toy_gn_deep`: group norms on the deep stem with average-pool shortcuts, plain convs, as
  `resnetv2_50d_gn`;
- `toy_evos`: `EvoNorm2dS0` (its activation its own) on the deep stem with average-pool shortcuts,
  as `resnetv2_50d_evos`, on 64x64 images;
- `toy_hybrid`: the hybrid ViT's backbone (`vision_transformer_hybrid._resnetv2`): `preact=False`
  (`Bottleneck`s, a normed stem and shortcuts, no final norm), the `same` stem (a SAME-padded max
  pool) and `StdConv2dSame` (eps 1e-8), 32-group norms, on 32x32 images: every stride-2 layer's
  SAME pad is uneven there, the odd pixel bottom-right;
- `toy_hybrid_odd`: `toy_hybrid` on 35x35 images, where the pads are even.

The toy test builds each as bunsen's `ResNetV2ContractConfig` of the same knobs.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
batch norms' running statistics random, so a misplaced norm shows. A two-image batch runs
through `forward_intermediates` (the stem's conv, then each stage), the pooled features and the
logits. The fixture is the weights under timm's own state-dict names (what its hub safetensors
hold) and `io.*`: `images`, `stage_0` to `stage_4`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_resnetv2_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from functools import partial  # noqa: E402

import torch  # noqa: E402
from timm.layers import (  # noqa: E402
    BatchNormAct2d,
    EvoNorm2dS0,
    GroupNormAct,
    StdConv2d,
    StdConv2dSame,
    create_conv2d,
)
from timm.models.resnetv2 import ResNetV2  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_running_stats,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "resnetv2"

BIT = dict(
    channels=(128, 128, 256, 256),
    stem_chs=32,
    stem_type="fixed",
    conv_layer=partial(StdConv2d, eps=1e-8),
    norm_layer=partial(GroupNormAct, num_groups=32),
)
DEEP = dict(stem_type="deep", avg_down=True, conv_layer=create_conv2d)
HYBRID = dict(
    channels=(128, 128, 256, 256),
    stem_chs=32,
    stem_type="same",
    preact=False,
    conv_layer=partial(StdConv2dSame, eps=1e-8),
    norm_layer=partial(GroupNormAct, num_groups=32),
)

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_bit": dict(BIT),
    "toy_bit_odd": dict(BIT),
    "toy_bn": dict(
        channels=(64, 128, 128, 256), stem_chs=32, conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
    ),
    "toy_basic_deep": dict(
        DEEP, channels=(16, 32, 64, 128), stem_chs=32, basic=True, bottle_ratio=1.0,
        norm_layer=BatchNormAct2d,
    ),
    "toy_gn_deep": dict(
        DEEP, channels=(128, 128, 256, 256), stem_chs=64, norm_layer=partial(GroupNormAct, num_groups=32),
    ),
    "toy_evos": dict(DEEP, channels=(128, 128, 256, 256), stem_chs=64, norm_layer=EvoNorm2dS0),
    "toy_hybrid": dict(HYBRID),
    "toy_hybrid_odd": dict(HYBRID),
}

# The image side of each toy: odd sizes for `toy_bit_odd`; for `toy_evos`, 64, so its last
# stage is 2x2: on a 1x1 map each 8-channel group's deviation is over 8 values, so ill-conditioned
# that timm's own float32 is 7.5e-4 from its float64 there.
SIZES = {"toy_bit_odd": 36, "toy_evos": 64, "toy_hybrid_odd": 35}


def norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's norms' gains and shifts (EvoNorm's `v` among the
    others)."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, EvoNorm2dS0)):
            names.update(
                f"{prefix}.{name}" for name, _ in module.named_parameters() if name in ("weight", "bias")
            )
    return names


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        model = build(lambda: ResNetV2(layers=(1, 1, 1, 1), num_classes=10, **args), seed=3456 + i)
        norms = norm_names(model)
        init_weights(
            model,
            norm_bias=0.1,
            is_norm=lambda n: n in norms and n.endswith(".weight"),
        )
        init_running_stats(model)
        torch.manual_seed(66 + i)
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
