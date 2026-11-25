"""Generate the window ops' fixtures: timm's own Swin functions on seeded inputs.

`ops::images::windows` is written against timm's `swin_transformer_v2.py` (and its V1, where they
agree); this records timm's answers for its toy tests, in `testdata/swin/window_ops.safetensors`:

- `partition.x`, `[2, 8, 12, 3]`, and `partition.windows`, timm's `window_partition` of it into
  `4x6` windows; `window_reverse` of those is `partition.x` again;
- `mask.{h}x{w}.w{wh}x{ww}.s{sh}x{sw}`, a block's shifted-window `attn_mask` (timm's
  `SwinTransformerV2Block.get_attn_mask`), for grids that divide into the window and ones that pad;
- `shift.{h}x{w}.w{w}`: a block's window and shift, as `[wh, ww, sh, sw]`, after its
  `_calc_window_shift` clamps them to the grid;
- `index.w{wh}x{ww}`: `WindowAttention`'s `relative_position_index`;
- `coords.w{wh}x{ww}.p{ph}x{pw}`: V2's `relative_coords_table` (log-spaced, normalized by the
  pretrained window when there is one);
- `merge.x`, `[2, 5, 7, 3]` (odd sides, so it pads), and `merge.collated`: timm's
  `PatchMerging` collation of it, before the reduction and norm;
- `roll.x`, `[2, 5, 6, 3]`, and `roll.back`: a block's cyclic shift of it, `torch.roll` by `(-2, -3)`
  over the grid.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_swin_op_fixtures.py

Pinned versions used to produce the committed fixture: timm 1.0.31, torch 2.15.0, safetensors 0.8.0.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from timm.models.swin_transformer_v2 import (  # noqa: E402
    SwinTransformerV2Block,
    WindowAttention,
    window_partition,
)

from toy_fixture import TESTDATA, write  # noqa: E402

OUT = TESTDATA / "swin"

# (grid, window, shift): grids that divide, that pad, and that a window covers.
MASKS = [
    ((8, 12), (4, 4), (2, 2)),
    ((7, 10), (4, 4), (2, 2)),
    ((8, 8), (4, 6), (2, 3)),
]

# (grid, window): a window larger than the grid shrinks to it, with no shift.
SHIFTS = [((16, 16), 8), ((8, 8), 8), ((6, 8), 8), ((4, 4), 12)]

INDICES = [(3, 4), (7, 7)]

COORDS = [((8, 8), (0, 0)), ((16, 16), (12, 12)), ((3, 5), (0, 0))]


def main() -> None:
    torch.manual_seed(2468)
    tensors = {}
    x = torch.randn(2, 8, 12, 3)
    tensors["partition.x"] = x
    tensors["partition.windows"] = window_partition(x, (4, 6))
    for grid, window, shift in MASKS:
        block = SwinTransformerV2Block(
            dim=4, input_resolution=grid, num_heads=1, window_size=window, shift_size=shift
        )
        assert block.window_size == window and block.shift_size == shift, (block.window_size, grid)
        name = f"mask.{grid[0]}x{grid[1]}.w{window[0]}x{window[1]}.s{shift[0]}x{shift[1]}"
        tensors[name] = block.attn_mask
    for grid, window in SHIFTS:
        block = SwinTransformerV2Block(
            dim=4, input_resolution=grid, num_heads=1, window_size=window, shift_size=window // 2
        )
        tensors[f"shift.{grid[0]}x{grid[1]}.w{window}"] = torch.tensor(
            [*block.window_size, *block.shift_size], dtype=torch.float32
        )
    for window in INDICES:
        attn = WindowAttention(dim=4, window_size=window, num_heads=1)
        tensors[f"index.w{window[0]}x{window[1]}"] = attn.relative_position_index.float()
    for window, pretrained in COORDS:
        attn = WindowAttention(
            dim=4, window_size=window, num_heads=1, pretrained_window_size=pretrained
        )
        tensors[f"coords.w{window[0]}x{window[1]}.p{pretrained[0]}x{pretrained[1]}"] = (
            attn.relative_coords_table[0]
        )
    x = torch.randn(2, 5, 7, 3)
    tensors["merge.x"] = x
    # timm's `PatchMerging.forward`, before its reduction and norm.
    b, h, w, c = x.shape
    padded = torch.nn.functional.pad(x, (0, 0, 0, w % 2, 0, h % 2))
    _, h, w, _ = padded.shape
    tensors["merge.collated"] = (
        padded.reshape(b, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 4, 2, 5).flatten(3)
    )
    x = torch.randn(2, 5, 6, 3)
    tensors["roll.x"] = x
    tensors["roll.back"] = torch.roll(x, shifts=(-2, -3), dims=(1, 2))
    write(OUT / "window_ops.safetensors", {k: v.float().contiguous() for k, v in tensors.items()})
    for k, v in tensors.items():
        print(k, tuple(v.shape))


if __name__ == "__main__":
    main()
