"""Generate the resnest kit's toy fixtures: tiny seeded timm `ResNeSt`s and their answers.

Four toys, each timm's own `timm.models.resnet.ResNet` of `ResNestBottleneck`s (`resnest.py`'s
block) on a deep stem (`stem_width=16`) with `avg_down` shortcuts, as every `ResNeSt` is, with
narrow stage widths (its `channels`), the first stage of two blocks, 10 classes:

- `toy_resnest`: `resnest50d`'s block knobs (`radix=2`, `avd`, the pool after the mixer) at
  `(8, 16, 32, 64)` planes; its attention widths are timm's floor of 32;
- `toy_resnest_odd`: `toy_resnest` on 36x36 images, so the `avd` pools and the shortcuts' see odd
  sizes (9, 5, 3);
- `toy_resnest_4s2x`: `resnest50d_4s2x40d`'s (`radix=4`, `cardinality=2`, `base_width=40`, the
  pool before the mixer) at `(32, 32, 64, 64)` planes;
- `toy_resnest_1s4x`: `resnest50d_1s4x24d`'s (`radix=1`, a sigmoid gate, `cardinality=4`,
  `base_width=24`, the pool before the mixer) at `(16, 32, 32, 64)` planes.

The toy test builds each as bunsen's `ResNeStContractConfig` of the same knobs.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
batch norms' running statistics random, so a misplaced norm shows. A two-image batch runs
through `forward_intermediates` (the stem's activation, then each stage), the pooled features
and the logits. The fixture is the weights under timm's own state-dict names (what its hub
safetensors hold) and `io.*`: `images`, `stage_0` to `stage_4`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_resnest_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.resnest import ResNestBottleneck  # noqa: E402
from timm.models.resnet import ResNet  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_running_stats,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "resnest"

DEEP = dict(layers=(2, 1, 1, 1), stem_type="deep", stem_width=16, avg_down=True)

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_resnest": dict(DEEP, channels=(8, 16, 32, 64), block_args=dict(radix=2, avd=True)),
    "toy_resnest_odd": dict(DEEP, channels=(8, 16, 32, 64), block_args=dict(radix=2, avd=True)),
    "toy_resnest_4s2x": dict(
        DEEP, channels=(32, 32, 64, 64), cardinality=2, base_width=40,
        block_args=dict(radix=4, avd=True, avd_first=True),
    ),
    "toy_resnest_1s4x": dict(
        DEEP, channels=(16, 32, 32, 64), cardinality=4, base_width=24,
        block_args=dict(radix=1, avd=True, avd_first=True),
    ),
}

# The image side of each toy: odd stage sizes for `toy_resnest_odd`.
SIZES = {"toy_resnest_odd": 36}


def norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's norms (`bn1`, `conv2.bn0`, `downsample.2`, ...)."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            names.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
    return names


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        model = build(
            lambda: ResNet(block=ResNestBottleneck, num_classes=10, **args),
            seed=1357 + i,
        )
        norms = norm_names(model)
        init_weights(
            model,
            norm_bias=0.1,
            is_norm=lambda n: n in norms and n.endswith(".weight"),
        )
        init_running_stats(model)
        torch.manual_seed(44 + i)
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
