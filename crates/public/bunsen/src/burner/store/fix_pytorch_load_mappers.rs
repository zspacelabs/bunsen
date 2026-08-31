//! Attaching the `PyTorch` store-boundary repairs to a built module tree.

use burn::{
    nn::{
        Linear,
        attention::MultiHeadAttention,
    },
    prelude::Backend,
};

use crate::burner::store::repair_pytorch_strided_weight;

/// Attaches the `PyTorch` load/save repairs a module tree needs.
///
/// [`repair_pytorch_strided_weight`] fixes one parameter; this walks a whole
/// module and applies it wherever a `PyTorch` checkpoint holds the weight as a
/// column-major view. A module handles the parameters it declares itself and
/// defers to its children for theirs, so the knowledge of *which* parameters
/// are affected stays next to the module that owns them.
///
/// This is deliberately **not** part of module initialization. The repair is
/// only correct for weights arriving from a `PyTorch` checkpoint; on a
/// parameter that did not need it, the mapper is a silent transpose. Apply it
/// on the way to a `PytorchStore` load, and nowhere else:
///
/// ```rust,ignore
/// let mut module = cfg.try_init(device)?.fix_pytorch_load_mappers();
/// module.load_from(&mut store)?;
/// ```
///
/// The mappers only run at the store boundary, so this leaves every live
/// parameter value untouched.
pub trait FixPytorchLoadMappers {
    /// Attaches the repair mappers, returning the module.
    fn fix_pytorch_load_mappers(self) -> Self;
}

impl<T: FixPytorchLoadMappers> FixPytorchLoadMappers for Vec<T> {
    fn fix_pytorch_load_mappers(self) -> Self {
        self.into_iter().map(T::fix_pytorch_load_mappers).collect()
    }
}

impl<B: Backend> FixPytorchLoadMappers for Linear<B> {
    /// The projection weight is the affected parameter; a bias is rank-1, and
    /// a rank-1 view cannot be strided this way.
    fn fix_pytorch_load_mappers(mut self) -> Self {
        self.weight = repair_pytorch_strided_weight(self.weight);
        self
    }
}

impl<B: Backend> FixPytorchLoadMappers for MultiHeadAttention<B> {
    /// All four projections are `Linear`, and `PyTorch` stores each of them
    /// transposed.
    fn fix_pytorch_load_mappers(mut self) -> Self {
        self.query = self.query.fix_pytorch_load_mappers();
        self.key = self.key.fix_pytorch_load_mappers();
        self.value = self.value.fix_pytorch_load_mappers();
        self.output = self.output.fix_pytorch_load_mappers();
        self
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        module::Param,
        nn::{
            LinearConfig,
            attention::MultiHeadAttentionConfig,
        },
        prelude::{
            Device,
            TensorData,
        },
        tensor::DType,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        param_load_mapping,
    };

    type B = CpuBackend;

    /// A `[2, 3]` probe the repair visibly reorders.
    fn probe(device: &Device<B>) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new((0..6).map(|v| v as f64).collect::<Vec<_>>(), [2, 3]),
            device,
        )
    }

    /// Runs `param`'s load mapping over [`probe`], as a flat `f32` row.
    fn load_mapped_probe(
        param: &Param<Tensor<B, 2>>,
        device: &Device<B>,
    ) -> Vec<f32> {
        param_load_mapping(param, probe(device))
            .cast(DType::F32)
            .to_data()
            .to_vec()
            .unwrap()
    }

    /// The probe as it arrives, when nothing is attached.
    const UNMAPPED: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    /// The probe transposed and reshaped back to `[2, 3]` — the repair.
    const REPAIRED: [f32; 6] = [0.0, 3.0, 1.0, 4.0, 2.0, 5.0];

    /// A `Linear` gets the repair on its weight, and its value is untouched.
    #[test]
    fn test_linear_weight_is_repaired() {
        let device = Default::default();

        let linear = LinearConfig::new(3, 5).init::<B>(&device);
        let before = linear.weight.val().to_data();

        let linear = linear.fix_pytorch_load_mappers();

        assert_eq!(load_mapped_probe(&linear.weight, &device), REPAIRED);
        assert_eq!(linear.weight.val().to_data(), before);
    }

    /// Every attention projection is reached — a miss on any one of them is a
    /// silently wrong model rather than a load error.
    #[test]
    fn test_attention_repairs_every_projection() {
        let device: Device<B> = Default::default();

        let attn = MultiHeadAttentionConfig::new(4, 2).init::<B>(&device);
        assert_eq!(load_mapped_probe(&attn.query.weight, &device), UNMAPPED);

        let attn = attn.fix_pytorch_load_mappers();

        for weight in [
            &attn.query.weight,
            &attn.key.weight,
            &attn.value.weight,
            &attn.output.weight,
        ] {
            assert_eq!(load_mapped_probe(weight, &device), REPAIRED);
        }
    }

    /// The `Vec` impl carries the walk into a stack of blocks.
    #[test]
    fn test_vec_repairs_each_element() {
        let device: Device<B> = Default::default();

        let linears: Vec<Linear<B>> = (0..3)
            .map(|_| LinearConfig::new(3, 5).init::<B>(&device))
            .collect();

        for linear in linears.fix_pytorch_load_mappers() {
            assert_eq!(load_mapped_probe(&linear.weight, &device), REPAIRED);
        }
    }
}
