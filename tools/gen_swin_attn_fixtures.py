"""Generate the window attention's fixtures: timm's own Swin blocks' attention on seeded weights.

`blocks::transformers::attention::window::WindowAttention` is checked against timm's block
attention, `SwinTransformerV2Block._attn` and `SwinTransformerBlock._attn` (the roll, the padding,
the partition, the windows' attention with its relative position bias and shifted-window mask,
and back), in `testdata/swin/window_attn.safetensors`. Each case, 16 wide in 2 heads, is:

- `{case}.attn.*`: the block's `attn` weights under timm's names, seeded at random (V2's
  `logit_scale` too);
- `{case}.x`, `[2, h, w, 16]`, and `{case}.y`: the block's `_attn` of it.

The cases: V2 shifted on an `8x8` grid in `4x4` windows; V2 shifted on a `7x10` grid that pads;
V2 on a `6x6` grid a `8x8` window shrinks to (no shift); V2 with a `pretrained_window_size`;
V1 shifted on `8x8`; V1 shifted on `7x10`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_swin_attn_fixtures.py

Pinned versions used to produce the committed fixture: timm 1.0.31, torch 2.15.0, safetensors 0.8.0.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.swin_transformer import SwinTransformerBlock  # noqa: E402
from timm.models.swin_transformer_v2 import SwinTransformerV2Block  # noqa: E402

from toy_fixture import TESTDATA, init_weights, write  # noqa: E402

OUT = TESTDATA / "swin"

# name: (version, grid, window, shift, pretrained window)
CASES = {
    "v2_shift": (2, (8, 8), 4, 2, 0),
    "v2_pad": (2, (7, 10), 4, 2, 0),
    "v2_clamp": (2, (6, 6), 8, 4, 0),
    "v2_pretrained": (2, (8, 8), 4, 2, 3),
    "v1_shift": (1, (8, 8), 4, 2, 0),
    "v1_pad": (1, (7, 10), 4, 2, 0),
}


def main() -> None:
    tensors = {}
    for i, (name, (version, grid, window, shift, pretrained)) in enumerate(CASES.items()):
        torch.manual_seed(1357 + i)
        if version == 2:
            block = SwinTransformerV2Block(
                dim=16, input_resolution=grid, num_heads=2, window_size=window,
                shift_size=shift, pretrained_window_size=pretrained,
            )
        else:
            block = SwinTransformerBlock(
                dim=16, input_resolution=grid, num_heads=2, window_size=window,
                shift_size=shift,
            )
        block = block.to(torch.float64).eval()
        init_weights(block)
        if version == 2:
            with torch.no_grad():
                block.attn.logit_scale.copy_(1.5 + torch.rand_like(block.attn.logit_scale))
        x = torch.randn(2, *grid, 16, dtype=torch.float64)
        with torch.no_grad():
            y = block._attn(x)
        for k, v in block.attn.state_dict().items():
            if k in ("relative_position_index", "relative_coords_table", "k_bias"):
                continue
            tensors[f"{name}.attn.{k}"] = v
        tensors[f"{name}.x"] = x
        tensors[f"{name}.y"] = y
        print(name, tuple(x.shape), block.window_size, block.shift_size)
    write(
        OUT / "window_attn.safetensors",
        {k: v.float().contiguous() for k, v in tensors.items()},
    )


if __name__ == "__main__":
    main()
