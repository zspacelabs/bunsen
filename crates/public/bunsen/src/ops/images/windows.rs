//! Window ops: what a windowed vision transformer (Swin) does around its
//! attention, on `[batch, height, width, channels]` grids of tokens, as
//! timm's `swin_transformer_v2.py` (and its V1, where they agree) computes
//! them.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    Tensor,
    tensor::{
        Device,
        Int,
        TensorData,
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::contracts::unpack_shape_contract;

/// A stage's window and shift, after timm's `_calc_window_shift`: an axis
/// whose grid is no larger than the window takes the grid's size as its
/// window, and no shift.
///
/// # Arguments
///
/// * `grid` - the stage's `[height, width]` in tokens.
/// * `window` - the target window, `[height, width]`.
/// * `shift` - the target shift (half the window on a shifted block, 0 on the
///   other).
///
/// # Returns
///
/// The `(window, shift)` the block uses.
pub fn window_and_shift(
    grid: [usize; 2],
    window: [usize; 2],
    shift: [usize; 2],
) -> ([usize; 2], [usize; 2]) {
    let window = [grid[0].min(window[0]), grid[1].min(window[1])];
    let shift = [0, 1].map(|i| if grid[i] <= window[i] { 0 } else { shift[i] });
    (window, shift)
}

/// The padding that makes `grid` a whole number of windows: the rows and the
/// columns added at the bottom and the right.
pub fn window_padding(
    grid: [usize; 2],
    window: [usize; 2],
) -> [usize; 2] {
    [0, 1].map(|i| (window[i] - grid[i] % window[i]) % window[i])
}

/// `x` with `pad` zero rows and columns added at the bottom and the right of
/// its grid, as timm pads a grid to whole windows (and a merge to even
/// sides).
///
/// # Arguments
///
/// * `x` - `[batch, height, width, channels]`.
/// * `pad` - the `[rows, columns]` to add.
///
/// # Returns
///
/// `[batch, height + rows, width + columns, channels]`.
pub fn pad_bottom_right(
    x: Tensor<4>,
    pad: [usize; 2],
) -> Tensor<4> {
    let [b, h, w, c] = unpack_shape_contract!(["batch", "height", "width", "channels"], &x.dims());
    let device = x.device();
    let dtype = x.dtype();
    let x = match pad[0] {
        0 => x,
        rows => Tensor::cat(vec![x, Tensor::zeros([b, rows, w, c], (&device, dtype))], 1),
    };
    let x = match pad[1] {
        0 => x,
        cols => Tensor::cat(
            vec![x, Tensor::zeros([b, h + pad[0], cols, c], (&device, dtype))],
            2,
        ),
    };
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["batch", "height" + "rows", "width" + "columns", "channels"],
        &x.dims(),
        &[
            ("batch", b),
            ("height", h),
            ("width", w),
            ("channels", c),
            ("rows", pad[0]),
            ("columns", pad[1]),
        ]
    );
    x
}

/// The grid cut into windows, timm's `window_partition`: each window's
/// tokens row by row, the windows row by row, batch by batch.
///
/// # Arguments
///
/// * `x` - `[batch, height, width, channels]`, its sides multiples of the
///   window's.
/// * `window` - `[height, width]`.
///
/// # Returns
///
/// `[batch * windows, window_height, window_width, channels]`.
///
/// # Panics
///
/// If the window does not divide the grid.
pub fn window_partition(
    x: Tensor<4>,
    window: [usize; 2],
) -> Tensor<4> {
    let [wh, ww] = window;
    let [b, nh, nw, c] = unpack_shape_contract!(
        ["batch", "h_wins" * "wh", "w_wins" * "ww", "channels"],
        &x.dims(),
        &["batch", "h_wins", "w_wins", "channels"],
        &[("wh", wh), ("ww", ww)]
    );
    x.reshape([b, nh, wh, nw, ww, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b * nh * nw, wh, ww, c])
}

/// Windows put back into their grid, timm's `window_reverse`: the inverse
/// of [`window_partition`].
///
/// # Arguments
///
/// * `windows` - `[batch * windows, window_height, window_width, channels]`.
/// * `grid` - the `[height, width]` they were cut from.
///
/// # Returns
///
/// `[batch, height, width, channels]`.
pub fn window_reverse(
    windows: Tensor<4>,
    grid: [usize; 2],
) -> Tensor<4> {
    let [n, wh, ww, c] =
        unpack_shape_contract!(["windows", "wh", "ww", "channels"], &windows.dims());
    // The grid is a whole number of windows, and the windows a whole number
    // of grids.
    let [h, w] = grid;
    let [nh, nw, b] = unpack_shape_contract!(
        [
            "h_wins" * "wh",
            "w_wins" * "ww",
            "batch" * "h_wins" * "w_wins"
        ],
        &[h, w, n],
        &["h_wins", "w_wins", "batch"],
        &[("wh", wh), ("ww", ww)]
    );
    windows
        .reshape([b, nh, nw, wh, ww, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, h, w, c])
}

/// The shifted-window attention mask of a block over `grid`, timm's
/// `get_attn_mask`: after the cyclic shift, a window's tokens from different
/// regions of the grid may not attend to each other. The grid is padded to
/// whole windows first. `None` when the block does not shift.
///
/// # Arguments
///
/// * `grid` - the stage's `[height, width]`.
/// * `window`, `shift` - the block's ([`window_and_shift`]).
///
/// # Returns
///
/// `[windows, window_area, window_area]`: 0 where a query attends a key,
/// -100 where it may not, the windows in [`window_partition`]'s order.
pub fn shifted_window_mask(
    grid: [usize; 2],
    window: [usize; 2],
    shift: [usize; 2],
    device: &Device,
) -> Option<Tensor<3>> {
    if shift == [0, 0] {
        return None;
    }
    let pad = window_padding(grid, window);
    let [h, w] = [grid[0] + pad[0], grid[1] + pad[1]];
    // Each axis's three regions: before the last window, the last window
    // before the shift, the shifted-in strip.
    let region = |i: usize, n: usize, axis: usize| {
        if i < n - window[axis] {
            0
        } else if i < n - shift[axis] {
            1
        } else {
            2
        }
    };
    let label = |r: usize, c: usize| 3 * region(r, h, 0) + region(c, w, 1);
    let [wh, ww] = window;
    let area = wh * ww;
    let mut values = Vec::with_capacity((h / wh) * (w / ww) * area * area);
    for wr in 0..h / wh {
        for wc in 0..w / ww {
            let labels: Vec<usize> = (0..area)
                .map(|t| label(wr * wh + t / ww, wc * ww + t % ww))
                .collect();
            for q in &labels {
                for k in &labels {
                    values.push(if q == k { 0.0f32 } else { -100.0 });
                }
            }
        }
    }
    let mask = Tensor::from_data(
        TensorData::new(values, [(h / wh) * (w / ww), area, area]),
        device,
    );
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["h_wins" * "w_wins", "wh" * "ww", "wh" * "ww"],
        &mask.dims(),
        &[
            ("h_wins", h / wh),
            ("w_wins", w / ww),
            ("wh", wh),
            ("ww", ww)
        ]
    );
    Some(mask)
}

/// The relative position index of a window's token pairs, timm's
/// `relative_position_index`: pair `(q, k)`'s offset `(dy, dx)` as a row of
/// a `(2 wh - 1) x (2 ww - 1)` table, `(dy + wh - 1) * (2 ww - 1) + dx + ww -
/// 1`.
///
/// # Returns
///
/// `[window_area, window_area]`.
pub fn relative_position_index(
    window: [usize; 2],
    device: &Device,
) -> Tensor<2, Int> {
    let [wh, ww] = window;
    let area = wh * ww;
    let mut values = Vec::with_capacity(area * area);
    for q in 0..area {
        for k in 0..area {
            let dy = (q / ww) as i64 - (k / ww) as i64 + wh as i64 - 1;
            let dx = (q % ww) as i64 - (k % ww) as i64 + ww as i64 - 1;
            values.push(dy * (2 * ww as i64 - 1) + dx);
        }
    }
    let index = Tensor::from_data(TensorData::new(values, [area, area]), device);
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["wh" * "ww", "wh" * "ww"],
        &index.dims(),
        &[("wh", wh), ("ww", ww)]
    );
    index
}

/// Swin V2's log-spaced relative coordinate table, timm's
/// `relative_coords_table`: each offset `(dy, dx)` of a window, scaled to
/// `[-8, 8]` by the pretraining window (or the window, without one), then
/// `sign(t) * log2(|t| + 1) / 3`.
///
/// # Arguments
///
/// * `window` - `[height, width]`.
/// * `pretrained` - the window the weights were trained at, timm's
///   `pretrained_window_size`; `None` for the window itself.
///
/// # Returns
///
/// `[2 wh - 1, 2 ww - 1, 2]`: `(dy, dx)` per offset, row-major.
pub fn log_relative_coords_table(
    window: [usize; 2],
    pretrained: Option<[usize; 2]>,
    device: &Device,
) -> Tensor<3> {
    let [wh, ww] = window;
    let scale = pretrained.unwrap_or(window);
    let norm = |d: i64, axis: usize| {
        let t = d as f32 / (scale[axis].max(2) - 1) as f32 * 8.0;
        t.signum() * (t.abs() + 1.0).log2() / 3.0
    };
    let mut values = Vec::with_capacity((2 * wh - 1) * (2 * ww - 1) * 2);
    for dy in -(wh as i64 - 1)..wh as i64 {
        for dx in -(ww as i64 - 1)..ww as i64 {
            values.push(norm(dy, 0));
            values.push(norm(dx, 1));
        }
    }
    let table = Tensor::from_data(TensorData::new(values, [2 * wh - 1, 2 * ww - 1, 2]), device);
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        [2 * "wh" - 1, 2 * "ww" - 1, 2],
        &table.dims(),
        &[("wh", wh), ("ww", ww)]
    );
    table
}

/// The grid cyclically shifted by `shift` rows and columns, as `torch.roll`
/// over its two axes: a positive shift moves tokens down and right, the ones
/// that fall off wrapping to the top and left. A shifted Swin block rolls by
/// minus its shift before its attention, and back after.
///
/// # Arguments
///
/// * `x` - `[batch, height, width, channels]`.
/// * `shift` - `[rows, columns]`.
pub fn roll_grid(
    x: Tensor<4>,
    shift: [i64; 2],
) -> Tensor<4> {
    x.roll(&shift, &[1, 2])
}

/// Each `2x2` block of the grid as one token, its four tokens' channels
/// side by side in timm's `PatchMerging` order, `[x(0,0), x(1,0), x(0,1),
/// x(1,1)]` (the row offset fastest); an odd side is padded with zeros first.
///
/// # Arguments
///
/// * `x` - `[batch, height, width, channels]`.
///
/// # Returns
///
/// `[batch, ceil(height / 2), ceil(width / 2), 4 * channels]`.
pub fn merge_2x2(x: Tensor<4>) -> Tensor<4> {
    let [b, h, w, c] = unpack_shape_contract!(["batch", "height", "width", "channels"], &x.dims());
    let [mh, mw] = [h.div_ceil(2), w.div_ceil(2)];
    pad_bottom_right(x, [h % 2, w % 2])
        .reshape([b, mh, 2, mw, 2, c])
        .permute([0, 1, 3, 4, 2, 5])
        .reshape([b, mh, mw, 4 * c])
}

// The tests read seeded timm fixtures, which need `store`.
#[cfg(all(test, feature = "store"))]
mod tests {

    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        tensor_to_vec_as,
        testdata,
    };

    /// timm's answers (`tools/gen_swin_op_fixtures.py`).
    fn fixture(device: &Device) -> crate::support::testing::ToyFixture {
        crate::support::testing::ToyFixture::load_io(
            testdata!("swin/window_ops.safetensors"),
            device,
        )
    }

    fn exact() -> Tolerance<f32> {
        Tolerance::rel_abs(0.0, 0.0)
    }

    /// The partition is timm's, and reversing it is the identity.
    #[test]
    fn test_partition_and_reverse() {
        let device = cpu_device();
        let fixture = fixture(&device);
        let x: Tensor<4> = fixture.io("partition.x");
        let windows = window_partition(x.clone(), [4, 6]);
        assert_tensors_close(&windows, &fixture.io::<4>("partition.windows"), exact());
        assert_tensors_close(&window_reverse(windows, [8, 12]), &x, exact());
    }

    /// The window shrinks to a smaller grid, and loses its shift.
    #[test]
    fn test_window_and_shift() {
        let device = cpu_device();
        let fixture = fixture(&device);
        for (grid, window) in [([16, 16], 8), ([8, 8], 8), ([6, 8], 8), ([4, 4], 12)] {
            let (w, s) = window_and_shift(grid, [window; 2], [window / 2; 2]);
            let want: Tensor<1> = fixture.io(&format!("shift.{}x{}.w{window}", grid[0], grid[1]));
            let want: Vec<usize> = tensor_to_vec_as::<f32, _>(&want)
                .into_iter()
                .map(|v| v as usize)
                .collect();
            assert_eq!([w[0], w[1], s[0], s[1]].to_vec(), want, "{grid:?} {window}");
        }
    }

    /// The mask is timm's, on grids that divide and grids that pad.
    #[test]
    fn test_shifted_window_mask() {
        let device = cpu_device();
        let fixture = fixture(&device);
        for (grid, window, shift) in [
            ([8, 12], [4, 4], [2, 2]),
            ([7, 10], [4, 4], [2, 2]),
            ([8, 8], [4, 6], [2, 3]),
        ] {
            let name = format!(
                "mask.{}x{}.w{}x{}.s{}x{}",
                grid[0], grid[1], window[0], window[1], shift[0], shift[1]
            );
            let mask = shifted_window_mask(grid, window, shift, &device).unwrap();
            assert_tensors_close(&mask, &fixture.io::<3>(&name), exact());
        }
        assert!(shifted_window_mask([8, 8], [4, 4], [0, 0], &device).is_none());
    }

    #[test]
    fn test_relative_position_index() {
        let device = cpu_device();
        let fixture = fixture(&device);
        for window in [[3, 4], [7, 7]] {
            let index = relative_position_index(window, &device).float();
            let name = format!("index.w{}x{}", window[0], window[1]);
            assert_tensors_close(&index, &fixture.io::<2>(&name), exact());
        }
    }

    /// The log-spaced table is timm's, scaled by the window or the
    /// pretraining window.
    #[test]
    fn test_log_relative_coords_table() {
        let device = cpu_device();
        let fixture = fixture(&device);
        for (window, pretrained) in [([8, 8], None), ([16, 16], Some([12, 12])), ([3, 5], None)] {
            let p = pretrained.unwrap_or([0, 0]);
            let name = format!("coords.w{}x{}.p{}x{}", window[0], window[1], p[0], p[1]);
            assert_tensors_close(
                &log_relative_coords_table(window, pretrained, &device),
                &fixture.io::<3>(&name),
                Tolerance::rel_abs(1e-6, 1e-6),
            );
        }
    }

    /// The roll is `torch.roll`'s, and rolling back is the identity.
    #[test]
    fn test_roll_grid() {
        let device = cpu_device();
        let fixture = fixture(&device);
        let x: Tensor<4> = fixture.io("roll.x");
        let back = roll_grid(x.clone(), [-2, -3]);
        assert_tensors_close(&back, &fixture.io::<4>("roll.back"), exact());
        assert_tensors_close(&roll_grid(back, [2, 3]), &x, exact());
    }

    /// The merge's collation is timm's, an odd grid padded.
    #[test]
    fn test_merge_2x2() {
        let device = cpu_device();
        let fixture = fixture(&device);
        let merged = merge_2x2(fixture.io("merge.x"));
        assert_tensors_close(&merged, &fixture.io::<4>("merge.collated"), exact());
    }
}
