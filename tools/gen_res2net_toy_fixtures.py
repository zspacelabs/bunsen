"""Generate the res2net kit's toy fixtures: tiny seeded timm `Res2Net`s and their answers.

Four toys, each timm's own `timm.models.resnet.ResNet` of `Bottle2neck`s (`res2net.py`'s block)
with narrow stage widths (its `channels`), 10 classes; a stage of two blocks takes the second
one's hierarchical path, each branch after the first adding the previous branch's output:

- `toy_res2net`: `res2net50_26w_4s`'s block knobs (`base_width=26`, `scale=4`) at
  `(8, 16, 32, 64)` planes, the first two stages of two blocks, on timm's 64-wide 7x7 stem;
- `toy_res2next`: `Res2NeXt`'s grouped branches (`cardinality=2`, `base_width=8`) at `scale=3`;
- `toy_res2net_d`: `res2net50d`'s deep stem (`stem_width=16`) and `avg_down` shortcuts at
  `scale=2`, `base_width=48`, on 36x36 images, so the stages see odd sizes (9, 5, 3) and the
  last chunk's pool a partial window;
- `toy_res2net_8s`: eight chunks (`scale=8`, `base_width=14`), of one channel each in the first
  stage.

The toy test builds each as bunsen's `Res2NetContractConfig` of the same knobs.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
batch norms' running statistics random, so a misplaced norm shows. A two-image batch runs
through `forward_intermediates` (the stem's activation, then each stage), the pooled features
and the logits. The fixture is the weights under timm's own state-dict names (what its hub
safetensors hold) and `io.*`: `images`, `stage_0` to `stage_4`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_res2net_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.res2net import Bottle2neck  # noqa: E402
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

OUT = TESTDATA / "res2net"

CHANNELS = (8, 16, 32, 64)

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_res2net": dict(layers=(2, 2, 1, 1), base_width=26, block_args=dict(scale=4)),
    "toy_res2next": dict(
        layers=(2, 1, 1, 1), cardinality=2, base_width=8, block_args=dict(scale=3),
    ),
    "toy_res2net_d": dict(
        layers=(1, 2, 1, 1), base_width=48, stem_type="deep", stem_width=16, avg_down=True,
        block_args=dict(scale=2),
    ),
    "toy_res2net_8s": dict(layers=(2, 1, 1, 1), base_width=14, block_args=dict(scale=8)),
}

# The image side of each toy: odd stage sizes for the deep-stem toy.
SIZES = {"toy_res2net_d": 36}


def norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's norms (`bn1`, `bns.0`, `downsample.1`, ...)."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            names.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
    return names


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        model = build(
            lambda: ResNet(block=Bottle2neck, channels=CHANNELS, num_classes=10, **args),
            seed=2468 + i,
        )
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
