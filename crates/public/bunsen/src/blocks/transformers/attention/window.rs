//! [`WindowAttention`]: a Swin block's shifted-window self-attention, a token
//! mixer of a stage's fixed grid.

use alloc::format;

use burn::{
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Dropout,
        DropoutConfig,
        Linear,
        LinearConfig,
    },
    prelude::Tensor,
    tensor::{
        Device,
        Distribution,
        FloatDType,
        Int,
        activation::{
            relu,
            sigmoid,
            softmax,
        },
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    burner::module::InputDTypes,
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
        WithOkOrPanic,
    },
    ops::images::windows::{
        log_relative_coords_table,
        pad_bottom_right,
        relative_position_index,
        roll_grid,
        shifted_window_mask,
        window_and_shift,
        window_padding,
        window_partition,
        window_reverse,
    },
};

/// The per-head relative position bias a window's attention adds to its
/// scores (`[pc_rel_bias]`): the config of a [`RelativePositionBias`].
///
/// A seam: each kind is a module of its own.
#[derive(Config, Debug)]
pub enum RelativePositionBiasConfig {
    /// A learnt table, one row per offset of the window, one column per head
    /// (Swin V1's `relative_position_bias_table`).
    Table,

    /// Swin V2's continuous position bias: an MLP (a `hidden`-wide `ReLU`
    /// layer, then one bias-free output per head) over the window's
    /// log-spaced offsets, `16 * sigmoid` of its output. The offsets are
    /// scaled by the `pretrained` window when the weights were trained at
    /// another (timm's `pretrained_window_size`).
    Cpb {
        /// The MLP's hidden width: 512 in Swin V2.
        hidden: usize,

        /// The window the weights were trained at, if not this one.
        pretrained: Option<[usize; 2]>,
    },
}

impl RelativePositionBiasConfig {
    /// Swin V2's: a 512-wide MLP, scaled by `pretrained` if given.
    pub fn cpb(pretrained: Option<[usize; 2]>) -> Self {
        Self::Cpb {
            hidden: 512,
            pretrained,
        }
    }
}

/// A learnt relative position bias table; built by
/// [`RelativePositionBiasConfig::Table`].
#[derive(Module, Debug)]
pub struct RelativePositionBiasTable {
    /// `[(2 wh - 1) * (2 ww - 1), heads]`.
    pub table: Param<Tensor<2>>,
}

/// Swin V2's continuous position bias MLP; built by
/// [`RelativePositionBiasConfig::Cpb`].
#[derive(Module, Debug)]
pub struct ContinuousPositionBias {
    /// Offsets to the hidden layer (timm's `cpb_mlp.0`).
    pub linear1: Linear,

    /// The hidden layer to one bias per head, no bias of its own (timm's
    /// `cpb_mlp.2`).
    pub linear2: Linear,

    /// The window's log-spaced offsets, `[2 wh - 1, 2 ww - 1, 2]`: a
    /// constant.
    #[module(skip)]
    pub coords: Tensor<3>,
}

/// A window's per-head relative position bias; built by
/// [`RelativePositionBiasConfig`].
// The CPB's two layers outweigh the table's one parameter; a block holds one
// bias, so the size gap costs nothing worth a box.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum RelativePositionBias {
    /// A learnt table.
    Table(RelativePositionBiasTable),

    /// Swin V2's continuous position bias.
    Cpb(ContinuousPositionBias),
}

impl RelativePositionBias {
    /// Each offset's bias per head, `[offsets, heads]`, in `dtype`.
    fn offset_table(
        &self,
        dtype: FloatDType,
    ) -> Tensor<2> {
        match self {
            Self::Table(t) => t.table.val().cast(dtype),
            Self::Cpb(c) => {
                let [rows, cols] = unpack_shape_contract!(
                    ["rows", "cols", 2],
                    &c.coords.dims(),
                    &["rows", "cols"]
                );
                let coords = c.coords.clone().cast(dtype).reshape([rows * cols, 2]);
                let hidden = relu(c.linear1.forward(coords));
                let bias = sigmoid(c.linear2.forward(hidden)) * 16.0;
                #[cfg(any(debug_assertions, test))]
                crate::contracts::assert_shape_contract_periodically!(
                    ["rows" * "cols", "heads"],
                    &bias.dims(),
                    &[
                        ("rows", rows),
                        ("cols", cols),
                        ("heads", c.linear2.weight.dims()[1])
                    ]
                );
                bias
            }
        }
    }

    /// The bias of each query-key pair of a window, per head: `[heads, area,
    /// area]`, in `dtype`.
    fn pair_bias(
        &self,
        index: &Tensor<2, Int>,
        dtype: FloatDType,
    ) -> Tensor<3> {
        let [area] = unpack_shape_contract!(["area", "area"], &index.dims(), &["area"]);
        let table = self.offset_table(dtype);
        let [heads] = unpack_shape_contract!(["offsets", "heads"], &table.dims(), &["heads"]);
        table
            .select(0, index.clone().reshape([area * area]))
            .reshape([area, area, heads])
            .permute([2, 0, 1])
    }
}

/// [`WindowAttention`]'s config: the stage's grid, the target window and
/// shift, the heads, the biases, the attention kind.
#[derive(Config, Debug)]
pub struct WindowAttentionConfig {
    /// The token width.
    pub d_model: usize,

    /// The attention heads; they divide `d_model`.
    pub n_heads: usize,

    /// The stage's `[height, width]` grid, in tokens.
    pub grid: [usize; 2],

    /// The target window; a side larger than the grid's shrinks to it
    /// ([`window_and_shift`]).
    pub window: [usize; 2],

    /// Whether the block shifts its windows by half a window (every second
    /// block of a stage).
    #[config(default = false)]
    pub shifted: bool,

    /// Whether the query and value projections have a bias.
    #[config(default = true)]
    pub qkv_bias: bool,

    /// Whether the key projection has one too: Swin V1's does, Swin V2's has
    /// none (its `k_bias` a zero buffer).
    #[config(default = true)]
    pub key_bias: bool,

    /// Swin V2's cosine attention: the queries and keys L2-normalized, their
    /// dot product times a learnt per-head scale (`exp` of a logit clamped at
    /// `ln 100`); else scaled dot-product attention.
    #[config(default = false)]
    pub cosine: bool,

    /// The per-head relative position bias.
    #[config(default = "RelativePositionBiasConfig::Table")]
    pub bias: RelativePositionBiasConfig,

    /// The dropout on the attention weights, after the softmax (training
    /// only).
    #[config(default = 0.0)]
    pub attn_drop: f64,
}

impl WindowAttentionConfig {
    /// The `(window, shift)` the block uses, after the grid clamps them.
    pub fn window_and_shift(&self) -> ([usize; 2], [usize; 2]) {
        let shift = match self.shifted {
            true => [self.window[0] / 2, self.window[1] / 2],
            false => [0, 0],
        };
        window_and_shift(self.grid, self.window, shift)
    }

    /// Builds the attention for layer `slot`.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for heads that
    /// don't divide the width, or a zero grid or window side.
    pub fn try_init(
        &self,
        slot: usize,
        device: &Device,
    ) -> BunsenResult<WindowAttention> {
        let (d, heads) = (self.d_model, self.n_heads);
        if heads == 0 {
            return Err(ConstraintError::zero_or_empty("WindowAttentionConfig", "n_heads").into());
        }
        if !d.is_multiple_of(heads) {
            return Err(ConstraintError::new(
                "WindowAttentionConfig",
                "d_model",
                Rule::NotMultiple {
                    value: d,
                    of: format!("n_heads ({heads})"),
                },
            )
            .into());
        }
        for (field, zero) in [
            ("grid", self.grid.contains(&0)),
            ("window", self.window.contains(&0)),
        ] {
            if zero {
                return Err(ConstraintError::zero_or_empty("WindowAttentionConfig", field).into());
            }
        }
        let (window, shift) = self.window_and_shift();
        let offsets = (2 * window[0] - 1) * (2 * window[1] - 1);
        let linear = |bias: bool| LinearConfig::new(d, d).with_bias(bias).init(device);
        let bias = match &self.bias {
            RelativePositionBiasConfig::Table => {
                RelativePositionBias::Table(RelativePositionBiasTable {
                    table: Param::from_tensor(Tensor::random(
                        [offsets, heads],
                        Distribution::Normal(0.0, 0.02),
                        device,
                    )),
                })
            }
            RelativePositionBiasConfig::Cpb { hidden, pretrained } => {
                RelativePositionBias::Cpb(ContinuousPositionBias {
                    linear1: LinearConfig::new(2, *hidden).init(device),
                    linear2: LinearConfig::new(*hidden, heads)
                        .with_bias(false)
                        .init(device),
                    coords: log_relative_coords_table(window, *pretrained, device),
                })
            }
        };
        Ok(WindowAttention {
            layer_index: slot,
            n_heads: heads,
            grid: self.grid,
            window,
            shift,
            c_q: linear(self.qkv_bias),
            c_k: linear(self.qkv_bias && self.key_bias),
            c_v: linear(self.qkv_bias),
            c_proj: linear(true),
            logit_scale: self
                .cosine
                .then(|| Param::from_tensor(Tensor::full([heads, 1, 1], 10f64.ln(), device))),
            bias,
            attn_drop: (self.attn_drop > 0.0).then(|| DropoutConfig::new(self.attn_drop).init()),
            index: relative_position_index(window, device),
            mask: shifted_window_mask(self.grid, window, shift, device),
        })
    }

    /// [`try_init`](Self::try_init), panicking with its error's report.
    ///
    /// # Panics
    ///
    /// When `try_init` fails.
    pub fn init(
        &self,
        slot: usize,
        device: &Device,
    ) -> WindowAttention {
        self.try_init(slot, device).ok_or_panic()
    }
}

/// A Swin block's shifted-window self-attention over a stage's fixed grid,
/// as timm's `SwinTransformer[V2]Block._attn` and its `WindowAttention`: the
/// tokens rolled back by the shift, padded to whole windows, cut into
/// windows; multi-head attention within each window, with a per-head
/// relative position bias and, on a shifted block, the mask that keeps the
/// rolled-in regions apart; the windows put back, unpadded, rolled forward.
///
/// Its projections are `c_q`, `c_k`, `c_v` (timm's fused `qkv`, split on
/// load, with V2's `q_bias` / `v_bias`) and `c_proj` (timm's `proj`).
///
/// Built by [`WindowAttentionConfig`].
#[derive(Module, Debug)]
pub struct WindowAttention {
    /// The layer's index in its stack.
    pub layer_index: usize,

    /// The attention heads.
    pub n_heads: usize,

    /// The stage's grid.
    pub grid: [usize; 2],

    /// The window, after the grid clamps it.
    pub window: [usize; 2],

    /// The shift, after the grid clamps it; `[0, 0]` on an unshifted block.
    pub shift: [usize; 2],

    /// The query projection.
    pub c_q: Linear,

    /// The key projection.
    pub c_k: Linear,

    /// The value projection.
    pub c_v: Linear,

    /// The output projection.
    pub c_proj: Linear,

    /// Swin V2's per-head logit scale, `[heads, 1, 1]`, under cosine
    /// attention.
    pub logit_scale: Option<Param<Tensor<3>>>,

    /// The per-head relative position bias.
    pub bias: RelativePositionBias,

    /// The dropout on the attention weights, when its rate is not 0.
    pub attn_drop: Option<Dropout>,

    /// The window's relative position index: a constant.
    #[module(skip)]
    pub index: Tensor<2, Int>,

    /// The shifted-window mask: a constant; `None` on an unshifted block.
    #[module(skip)]
    pub mask: Option<Tensor<3>>,
}

impl InputDTypes for WindowAttention {
    /// Its projections'.
    fn preferred_dtype(&self) -> FloatDType {
        FloatDType::from(self.c_q.weight.dtype())
    }
}

impl WindowAttention {
    /// The token width.
    pub fn n_embed(&self) -> usize {
        self.c_q.weight.dims()[0]
    }

    /// Attention within each window, timm's `WindowAttention.forward`.
    ///
    /// # Arguments
    ///
    /// * `x` - `[windows * batch, area, d]`.
    ///
    /// # Returns
    ///
    /// `[windows * batch, area, d]`.
    fn attend(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let heads = self.n_heads;
        let d = self.n_embed();
        let [bw, n, hd] = unpack_shape_contract!(
            ["batch_windows", "area", "d_model" = "heads" * "head_dim"],
            &x.dims(),
            &["batch_windows", "area", "head_dim"],
            &[("heads", heads), ("d_model", d)]
        );
        let split = |t: Tensor<3>| t.reshape([bw, n, heads, hd]).swap_dims(1, 2);
        let (q, k, v) = (
            split(self.c_q.forward(x.clone())),
            split(self.c_k.forward(x.clone())),
            split(self.c_v.forward(x)),
        );
        let dtype = FloatDType::from(q.dtype());
        let scores = match &self.logit_scale {
            Some(scale) => {
                let unit = |t: Tensor<4>| {
                    let norm = t.clone().powi_scalar(2).sum_dim(3).sqrt().clamp_min(1e-12);
                    t / norm
                };
                let scale = scale.val().cast(dtype).clamp_max(100f64.ln()).exp();
                unit(q).matmul(unit(k).swap_dims(2, 3)) * scale.unsqueeze::<4>()
            }
            None => (q * (hd as f64).powf(-0.5)).matmul(k.swap_dims(2, 3)),
        };
        let scores = scores + self.bias.pair_bias(&self.index, dtype).unsqueeze::<4>();
        let scores = match &self.mask {
            Some(mask) => {
                // The windows of each batch item, each its own mask.
                let [windows] = unpack_shape_contract!(
                    ["windows", "area", "area"],
                    &mask.dims(),
                    &["windows"],
                    &[("area", n)]
                );
                let [batch] = unpack_shape_contract!(
                    ["batch" * "windows"],
                    &[bw],
                    &["batch"],
                    &[("windows", windows)]
                );
                let mask = mask.clone().cast(dtype).reshape([1, windows, 1, n, n]);
                (scores.reshape([batch, windows, heads, n, n]) + mask).reshape([bw, heads, n, n])
            }
            None => scores,
        };
        let attn = softmax(scores, 3);
        let attn = match &self.attn_drop {
            Some(drop) => drop.forward(attn),
            None => attn,
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch_windows", "heads", "area", "area"],
            &attn.dims(),
            &[("batch_windows", bw), ("heads", heads), ("area", n)]
        );
        let y = attn.matmul(v).swap_dims(1, 2).reshape([bw, n, d]);
        let y = self.c_proj.forward(y);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch_windows", "area", "d_model"],
            &y.dims(),
            &[("batch_windows", bw), ("area", n), ("d_model", d)]
        );
        y
    }

    /// The shifted-window attention over the stage's grid.
    ///
    /// # Arguments
    ///
    /// * `x` - `[batch, height * width, d]`, the grid row by row.
    ///
    /// # Returns
    ///
    /// `[batch, height * width, d]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [h, w] = self.grid;
        let d = self.n_embed();
        let [b] = unpack_shape_contract!(
            ["batch", "height" * "width", "d_model"],
            &x.dims(),
            &["batch"],
            &[("height", h), ("width", w), ("d_model", d)]
        );
        let l = h * w;
        let [sh, sw] = self.shift.map(|s| s as i64);
        let x = x.reshape([b, h, w, d]);
        let x = match self.shift {
            [0, 0] => x,
            _ => roll_grid(x, [-sh, -sw]),
        };
        let pad = window_padding(self.grid, self.window);
        let [hp, wp] = [h + pad[0], w + pad[1]];
        let x = pad_bottom_right(x, pad);
        let [wh, ww] = self.window;
        let windows =
            window_partition(x, self.window).reshape([b * (hp / wh) * (wp / ww), wh * ww, d]);
        let windows = self
            .attend(windows)
            .reshape([b * (hp / wh) * (wp / ww), wh, ww, d]);
        let x = window_reverse(windows, [hp, wp]).slice([0..b, 0..h, 0..w, 0..d]);
        let x = match self.shift {
            [0, 0] => x,
            _ => roll_grid(x, [sh, sw]),
        };
        x.reshape([b, l, d])
    }
}

// The tests read seeded timm fixtures, which need `store`.
#[cfg(all(test, feature = "store"))]
mod tests {

    use burn::tensor::s;

    use super::*;
    use crate::support::testing::{
        ToyFixture,
        assert_io_close,
        cpu_device,
        testdata,
    };

    fn fixture(device: &Device) -> ToyFixture {
        ToyFixture::load_io(testdata!("swin/window_attn.safetensors"), device)
    }

    /// A linear layer of torch's `[out, in]` weight and its bias.
    fn linear(
        weight: Tensor<2>,
        bias: Option<Tensor<1>>,
    ) -> Linear {
        let [out, input] = weight.dims();
        let mut fc = LinearConfig::new(input, out)
            .with_bias(bias.is_some())
            .init(&weight.device());
        fc.weight = Param::from_tensor(weight.transpose());
        fc.bias = bias.map(Param::from_tensor);
        fc
    }

    /// The case's attention, its weights timm's.
    fn load(
        fixture: &ToyFixture,
        case: &str,
        config: WindowAttentionConfig,
    ) -> WindowAttention {
        let device = fixture.device().clone();
        let mut attn = config.try_init(0, &device).unwrap();
        let t2 = |k: &str| fixture.io::<2>(&format!("{case}.attn.{k}"));
        let t1 = |k: &str| fixture.io::<1>(&format!("{case}.attn.{k}"));
        let has = |k: &str| fixture.has(&format!("{case}.attn.{k}"));
        let d = config.d_model;
        let qkv = t2("qkv.weight");
        let rows = |i: usize| qkv.clone().slice(s![i * d..(i + 1) * d, ..]);
        let (qb, kb, vb) = if has("qkv.bias") {
            let b = t1("qkv.bias");
            (
                Some(b.clone().slice(s![0..d])),
                Some(b.clone().slice(s![d..2 * d])),
                Some(b.slice(s![2 * d..3 * d])),
            )
        } else {
            (Some(t1("q_bias")), None, Some(t1("v_bias")))
        };
        attn.c_q = linear(rows(0), qb);
        attn.c_k = linear(rows(1), kb);
        attn.c_v = linear(rows(2), vb);
        attn.c_proj = linear(t2("proj.weight"), Some(t1("proj.bias")));
        match &mut attn.bias {
            RelativePositionBias::Table(t) => {
                t.table = Param::from_tensor(t2("relative_position_bias_table"));
            }
            RelativePositionBias::Cpb(c) => {
                c.linear1 = linear(t2("cpb_mlp.0.weight"), Some(t1("cpb_mlp.0.bias")));
                c.linear2 = linear(t2("cpb_mlp.2.weight"), None);
            }
        }
        if has("logit_scale") {
            attn.logit_scale = Some(Param::from_tensor(
                fixture.io::<3>(&format!("{case}.attn.logit_scale")),
            ));
        }
        attn
    }

    /// Each case's attention is timm's block attention.
    #[test]
    fn test_window_attention_is_timms() {
        let device = cpu_device();
        let fixture = fixture(&device);
        let v2 = |grid, window, pretrained: Option<[usize; 2]>| {
            WindowAttentionConfig::new(16, 2, grid, window)
                .with_shifted(true)
                .with_key_bias(false)
                .with_cosine(true)
                .with_bias(RelativePositionBiasConfig::cpb(pretrained))
        };
        let v1 = |grid, window| WindowAttentionConfig::new(16, 2, grid, window).with_shifted(true);
        for (case, config) in [
            ("v2_shift", v2([8, 8], [4, 4], None)),
            ("v2_pad", v2([7, 10], [4, 4], None)),
            ("v2_clamp", v2([6, 6], [8, 8], None)),
            ("v2_pretrained", v2([8, 8], [4, 4], Some([3, 3]))),
            ("v1_shift", v1([8, 8], [4, 4])),
            ("v1_pad", v1([7, 10], [4, 4])),
        ] {
            let grid = config.grid;
            let attn = load(&fixture, case, config);
            let x: Tensor<4> = fixture.io(&format!("{case}.x"));
            let [b, h, w, d] = x.dims();
            let y = attn.forward(x.reshape([b, h * w, d]));
            assert_eq!([h, w], grid);
            let want: Tensor<4> = fixture.io(&format!("{case}.y"));
            assert_io_close(case, y.reshape([b, h, w, d]), want);
        }
    }

    /// A window larger than the grid shrinks to it and drops the shift.
    #[test]
    fn test_the_grid_clamps_the_window() {
        let config = WindowAttentionConfig::new(16, 2, [6, 6], [8, 8]).with_shifted(true);
        assert_eq!(config.window_and_shift(), ([6, 6], [0, 0]));
        let attn = config.try_init(0, &cpu_device()).unwrap();
        assert!(attn.mask.is_none());
    }
}
