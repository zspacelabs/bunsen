"""Generate the resnet kit's toy fixtures: tiny seeded timm `ResNet`s and their answers.

Two toys, each timm's own `timm.models.resnet.ResNet` with four stages of one block and
narrow stage widths, `channels=(8, 16, 32, 64)`, on timm's fixed 64-wide 7x7 stem, 10
classes (timm's forward runs exactly four stages):

- `toy_basic`: `BasicBlock`s, each stage `(8, 16, 32, 64)` planes wide, the first after a
  downsample from the stem's 64;
- `toy_bottleneck`: `Bottleneck`s, `(32, 64, 128, 256)` out (4x the planes), each with a
  downsample.

bunsen's `ResNetContractConfig` derives its stage widths from the stem's, so the toy test
builds this geometry as a `ResNetStructureConfig`, block by block.

Each is built in float64 on the CPU with seeded weights: the batch norms' gains and shifts
and their running statistics random, so a misplaced norm shows. A two-image
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

TOYS = {
    "toy_basic": BasicBlock,
    "toy_bottleneck": Bottleneck,
}


def batch_norm_names(model: torch.nn.Module) -> set:
    """The parameter names of the model's batch norms (`bn1`, `downsample.1`, ...)."""
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            names.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
    return names


def main() -> None:
    for i, (name, block) in enumerate(TOYS.items()):
        model = build(
            lambda: ResNet(block=block, layers=(1, 1, 1, 1), channels=(8, 16, 32, 64), num_classes=10),
            seed=1234 + i,
        )
        norms = batch_norm_names(model)
        init_weights(
            model,
            norm_bias=0.1,
            is_norm=lambda n: n in norms and n.endswith(".weight"),
        )
        init_running_stats(model)
        torch.manual_seed(99 + i)
        images = torch.randn(2, 3, 32, 32, dtype=torch.float64)
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
