"""Generate the sknet kit's toy fixtures: tiny seeded timm `SKNet`s and their answers.

Three toys, each timm's own `timm.models.resnet.ResNet` of `sknet.py`'s blocks with narrow stage
widths (its `channels`), on timm's 64-wide 7x7 stem, the first stage of two blocks, 10 classes:

- `toy_skresnet`: `skresnet18`'s knobs: `SelectiveKernelBasic`s (the selective kernel their first
  conv), an attention of 1/8 of the channels in 16s, the input split across the two paths, at
  `(16, 32, 32, 64)` planes;
- `toy_skresnet_odd`: `toy_skresnet` on 36x36 images, so the stages see odd sizes (9, 5, 3);
- `toy_skresnext`: `skresnext50_32x4d`'s: `SelectiveKernelBottleneck`s of 4 groups of 16 per 64
  planes, an attention of 1/16 in 32s, every path reading the whole input, at `(8, 16, 32, 64)`
  planes.

The toy test builds each as bunsen's `SKNetContractConfig` of the same knobs.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
batch norms' running statistics random, so a misplaced norm shows. A two-image batch runs
through `forward_intermediates` (the stem's activation, then each stage), the pooled features
and the logits. The fixture is the weights under timm's own state-dict names (what its hub
safetensors hold) and `io.*`: `images`, `stage_0` to `stage_4`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_sknet_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.resnet import ResNet  # noqa: E402
from timm.models.sknet import SelectiveKernelBasic, SelectiveKernelBottleneck  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_running_stats,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "sknet"

BASIC = dict(
    block=SelectiveKernelBasic, layers=(2, 1, 1, 1), channels=(16, 32, 32, 64),
    block_args=dict(sk_kwargs=dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)),
)

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_skresnet": dict(BASIC),
    "toy_skresnet_odd": dict(BASIC),
    "toy_skresnext": dict(
        block=SelectiveKernelBottleneck, layers=(2, 1, 1, 1), channels=(8, 16, 32, 64),
        cardinality=4, base_width=16,
        block_args=dict(sk_kwargs=dict(rd_ratio=1 / 16, rd_divisor=32, split_input=False)),
    ),
}

# The image side of each toy: odd stage sizes for `toy_skresnet_odd`.
SIZES = {"toy_skresnet_odd": 36}


def norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's norms (`bn1`, `conv1.paths.0.bn`, ...)."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            names.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
    return names


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        model = build(
            lambda: ResNet(num_classes=10, zero_init_last=False, **args),
            seed=9753 + i,
        )
        norms = norm_names(model)
        init_weights(
            model,
            norm_bias=0.1,
            is_norm=lambda n: n in norms and n.endswith(".weight"),
        )
        init_running_stats(model)
        torch.manual_seed(33 + i)
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
