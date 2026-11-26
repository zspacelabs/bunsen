"""Generate the Swin V1 kit's toy fixtures: tiny seeded timm `SwinTransformer`s and their answers.

Five toys, each timm's own `timm.models.swin_transformer.SwinTransformer` in `4x4` patches, 16
wide, two stages of two blocks (2 and 4 heads, the second stage after a patch merge), 10 classes:

- `toy_v1`: 32x32 images (an `8x8` grid, then `4x4`) in `4x4` windows;
- `toy_v1_pad`: 28x28 images: a `7x7` grid that pads to whole windows, and an odd side the merge
  pads;
- `toy_v1_stage_windows`: a window per stage, `4x4` then `2x2` (as Swin-S3's);
- `toy_v1_no_qkv_bias`: `qkv_bias=False`;
- `toy_v1_headless`: `toy_v1` with no classifier (`num_classes=0`).

The toy test builds each as bunsen's `SwinV1ContractConfig`.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts and the
relative position bias tables random. A two-image batch runs through `forward_intermediates`
(each stage's output as a map), `forward_features`, the pooled features and the logits. The
fixture is the weights under timm's own state-dict names (what its hub safetensors hold) and
`io.*`: `images`, `stage_0`, `stage_1`, `features`, `pooled`, `logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_swin_v1_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.swin_transformer import SwinTransformer  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "swin"

BASE = dict(
    img_size=32, patch_size=4, embed_dim=16, depths=(2, 2), num_heads=(2, 4), window_size=4,
    num_classes=10,
)

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_v1": dict(),
    "toy_v1_pad": dict(img_size=28),
    "toy_v1_stage_windows": dict(window_size=((4, 4), (2, 2))),
    "toy_v1_no_qkv_bias": dict(qkv_bias=False),
    "toy_v1_headless": dict(num_classes=0),
}


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        args = dict(BASE, **args)
        model = build(lambda: SwinTransformer(**args), seed=9753 + i)
        init_weights(model, norm_bias=0.1)
        torch.manual_seed(44 + i)
        side = args["img_size"]
        images = torch.randn(2, 3, side, side, dtype=torch.float64)
        with torch.no_grad():
            stages = model.forward_intermediates(images, intermediates_only=True)
            features = model.forward_features(images)
            pooled = model.forward_head(features, pre_logits=True)
            logits = model(images)
        assert len(stages) == 2, len(stages)
        b, h, w, c = features.shape
        io = {
            "images": images,
            "features": features.reshape(b, h * w, c),
            "pooled": pooled,
            "logits": logits,
        }
        io.update({f"stage_{k}": x for k, x in enumerate(stages)})
        drop = {k for k in model.state_dict() if k.endswith(
            ("relative_position_index", "attn_mask"))}
        write(OUT / f"{name}.safetensors", weights(model, drop=drop), io_tensors(io))
        print(name, {k: tuple(v.shape) for k, v in io.items()})


if __name__ == "__main__":
    main()
