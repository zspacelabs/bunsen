//! # Patch merging
//!
//! [`PatchMerge`]: a staged vision transformer's downsample between stages,
//! timm's Swin `PatchMerging`: each `2x2` block of the token grid as one
//! token of four times the channels, reduced to the next stage's width, and
//! normed (before the reduction in Swin V1, after it in V2).

use alloc::format;

use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::norm::{
        BunsenNormalization,
        BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::images::windows::merge_2x2,
};

/// [`PatchMerge`]'s config: the grid it merges, the widths, the norm and its
/// place.
#[derive(Config, Debug)]
pub struct PatchMergeConfig {
    /// The `[height, width]` grid it merges, in tokens.
    pub grid: [usize; 2],

    /// The width of the tokens it merges.
    pub d_input: usize,

    /// The width of the merged tokens: twice the input's in Swin.
    pub d_output: usize,

    /// The norm, sized at init: over the four tokens' channels before the
    /// reduction (Swin V1), or over the output after it (Swin V2).
    pub norm: BunsenNormalizationConfig,

    /// Whether the norm comes before the reduction (Swin V1), not after
    /// (Swin V2).
    #[config(default = "false")]
    pub norm_first: bool,
}

impl PatchMergeConfig {
    /// The merged grid: each side halved, rounded up (an odd side is padded).
    pub fn output_grid(&self) -> [usize; 2] {
        self.grid.map(|n| n.div_ceil(2))
    }
}

impl ModuleInit<PatchMerge> for PatchMergeConfig {
    /// Builds the merge.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a zero width
    /// or grid side; as the norm's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<PatchMerge> {
        if self.d_input == 0 || self.d_output == 0 || self.grid.contains(&0) {
            return Err(BunsenError::illegal(format!(
                "PatchMergeConfig: {} to {} wide over a {:?} grid",
                self.d_input, self.d_output, self.grid
            )));
        }
        let norm_features = match self.norm_first {
            true => 4 * self.d_input,
            false => self.d_output,
        };
        Ok(PatchMerge {
            grid: self.grid,
            norm_first: self.norm_first,
            reduction: LinearConfig::new(4 * self.d_input, self.d_output)
                .with_bias(false)
                .init(device),
            norm: self
                .norm
                .clone()
                .with_num_features(norm_features)
                .try_init(device)?,
        })
    }
}

/// A staged vision transformer's patch merge, timm's Swin `PatchMerging`.
/// Its parameters are `reduction` (no bias) and `norm`.
///
/// Built by [`PatchMergeConfig`].
#[derive(Module, Debug)]
pub struct PatchMerge {
    /// The grid it merges.
    pub grid: [usize; 2],

    /// Whether the norm comes before the reduction.
    pub norm_first: bool,

    /// The four tokens' channels to the output width.
    pub reduction: Linear,

    /// The norm.
    pub norm: BunsenNormalization,
}

impl PatchMerge {
    /// The merged grid: each side halved, rounded up.
    pub fn output_grid(&self) -> [usize; 2] {
        self.grid.map(|n| n.div_ceil(2))
    }

    /// The width of the merged tokens.
    pub fn d_output(&self) -> usize {
        self.reduction.weight.dims()[1]
    }

    /// The merge of a grid of tokens.
    ///
    /// # Arguments
    ///
    /// * `x` - `[batch, height * width, d_input]`, the grid row by row.
    ///
    /// # Returns
    ///
    /// `[batch, ceil(height / 2) * ceil(width / 2), d_output]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [h, w] = self.grid;
        // The reduction reads the four merged tokens' channels.
        let c = self.reduction.weight.dims()[0] / 4;
        let [b] = unpack_shape_contract!(
            ["batch", "height" * "width", "d_input"],
            &x.dims(),
            &["batch"],
            &[("height", h), ("width", w), ("d_input", c)]
        );
        let [mh, mw] = self.output_grid();
        let x = merge_2x2(x.reshape([b, h, w, c])).reshape([b, mh * mw, 4 * c]);
        let y = match self.norm_first {
            true => self.reduction.forward(self.norm.forward(x)),
            false => self.norm.forward(self.reduction.forward(x)),
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "half_h" * "half_w", "d_output"],
            &y.dims(),
            &[
                ("batch", b),
                ("half_h", mh),
                ("half_w", mw),
                ("d_output", self.d_output())
            ]
        );
        y
    }
}

// The tests read seeded timm fixtures, which need `store`.
#[cfg(all(test, feature = "store"))]
mod tests {

    use burn::module::Param;

    use super::*;
    use crate::{
        blocks::norm::BunsenLayerNormConfig,
        support::testing::{
            ToyFixture,
            assert_io_close,
            cpu_device,
            expect_variant,
            testdata,
        },
    };

    /// V1's and V2's merges are timm's `PatchMerging`, an odd side padded.
    #[test]
    fn test_patch_merge_is_timms() {
        let device = cpu_device();
        let fixture = ToyFixture::load_io(testdata!("swin/window_attn.safetensors"), &device);
        for (case, norm_first) in [("merge_v1", true), ("merge_v2", false)] {
            let config = PatchMergeConfig::new(
                [5, 6],
                8,
                16,
                BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(0)),
            )
            .with_norm_first(norm_first);
            let mut merge: PatchMerge = config.init(&device);
            merge.reduction.weight = Param::from_tensor(
                fixture
                    .io::<2>(&format!("{case}.reduction.weight"))
                    .transpose(),
            );
            expect_variant!(let BunsenNormalization::Layer(norm) = &mut merge.norm, "a layer norm");
            norm.gamma = Param::from_tensor(fixture.io(&format!("{case}.norm.weight")));
            norm.beta = Some(Param::from_tensor(fixture.io(&format!("{case}.norm.bias"))));
            let x: Tensor<4> = fixture.io(&format!("{case}.x"));
            let y = merge.forward(x.reshape([2, 30, 8]));
            assert_eq!(merge.output_grid(), [3, 3]);
            assert_io_close(
                case,
                y.reshape([2, 3, 3, 16]),
                fixture.io(&format!("{case}.y")),
            );
        }
    }
}
