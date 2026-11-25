"""Generate the deit kit's toy fixtures: tiny seeded timm `DeiT`s and their answers.

Three toys on 16x16 images in 4x4 patches (a 4x4 grid), 16 wide, two layers of two heads,
10 classes:

- `toy_distilled`: timm's `deit.VisionTransformerDistilled`: a distillation token after the
  class token, and a classifier on it (`head_dist`), the two classifiers' logits averaged; the
  pooled features the two tokens' mean;
- `toy_deit3`: timm's `VisionTransformer` as `DeiT` III builds it: the position embedding over
  the patches alone (`no_embed_class`), and layer scale (`init_values`);
- `toy_distilled_headless`: `toy_distilled` with no classifiers (`num_classes=0`): the output
  is the pooled features.

The toy test builds each as bunsen's `DeiTContractConfig`.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts random, so
a misplaced norm shows. A two-image `[2, 3, 16, 16]` batch runs through
`forward_intermediates` (each layer's patch tokens as a map), the pooled features and the
logits. The fixture is the weights under timm's own state-dict names (what its hub safetensors
hold) and `io.*`: `images`, `layer_0`, `layer_1`, `features` (`forward_features`), `pooled`,
`logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_deit_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.deit import VisionTransformerDistilled  # noqa: E402
from timm.models.vision_transformer import VisionTransformer  # noqa: E402

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "deit"

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_distilled": (VisionTransformerDistilled, dict()),
    "toy_deit3": (VisionTransformer, dict(no_embed_class=True, init_values=1e-6)),
    "toy_distilled_headless": (VisionTransformerDistilled, dict(num_classes=0)),
}


def main() -> None:
    for i, (name, (cls, args)) in enumerate(TOYS.items()):
        args = dict(dict(num_classes=10), **args)
        model = build(
            lambda: cls(img_size=16, patch_size=4, embed_dim=16, depth=2, num_heads=2, **args),
            seed=5678 + i,
        )
        init_weights(model, norm_bias=0.1)
        torch.manual_seed(55 + i)
        images = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        with torch.no_grad():
            layers = model.forward_intermediates(images, intermediates_only=True)
            features = model.forward_features(images)
            pooled = model.forward_head(features, pre_logits=True)
            logits = model(images)
        assert len(layers) == 2, len(layers)
        io = {"images": images, "features": features, "pooled": pooled, "logits": logits}
        io.update({f"layer_{k}": x for k, x in enumerate(layers)})
        write(OUT / f"{name}.safetensors", weights(model), io_tensors(io))
        print(name, {k: tuple(v.shape) for k, v in io.items()})


if __name__ == "__main__":
    main()
