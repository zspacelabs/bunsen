//! burn's `PyTorch`-to-burn adaptation, with bunsen's transpose.
use alloc::{
    boxed::Box,
    string::{
        String,
        ToString,
    },
};

use burn::{
    store::{
        ModuleAdapter,
        ModuleContext,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::{
        DType,
        Shape,
    },
};

use crate::burner::tensor::transpose_data;

/// The container type burn's module derive gives a `Linear`.
const LINEAR: &str = "Struct:Linear";

/// The container types of the normalizations whose `PyTorch` `weight` and
/// `bias` are burn's `gamma` and `beta`.
const NORMALIZATIONS: [&str; 4] = [
    "Struct:BatchNorm",
    "Struct:LayerNorm",
    "Struct:GroupNorm",
    "Struct:RmsNorm",
];

/// burn's [`PyTorchToBurnAdapter`](burn::store::PyTorchToBurnAdapter), rule
/// for rule: a `Linear`'s 2-D `weight` transposed (`PyTorch`'s `[out, in]` to
/// burn's `[in, out]`), and a normalization's `weight` / `bias` read as its
/// `gamma` / `beta`; quantized tensors pass through.
///
/// The transpose is bunsen's [`transpose_data`]: burn's moves each element
/// with a call sized at run time, which made it most of a large model's load
/// (Qwen3-4B on CUDA: 34 s, 25 s with this one).
#[derive(Debug, Clone, Default)]
pub struct BunsenPyTorchToBurnAdapter;

impl BunsenPyTorchToBurnAdapter {
    /// What becomes of a tensor at a parameter named `param` in a module of
    /// `module_type`: whether it is transposed, and the name it is read as.
    pub fn rule(
        module_type: &str,
        param: &str,
        rank: usize,
        dtype: DType,
    ) -> (bool, Option<&'static str>) {
        if module_type == LINEAR
            && param == "weight"
            && rank == 2
            && !matches!(dtype, DType::QFloat(_))
        {
            return (true, None);
        }
        if NORMALIZATIONS.contains(&module_type) {
            return (false, burn_norm_name(param));
        }
        (false, None)
    }
}

/// A normalization's burn name for its `PyTorch` parameter name.
fn burn_norm_name(param: &str) -> Option<&'static str> {
    match param {
        "weight" => Some("gamma"),
        "bias" => Some("beta"),
        _ => None,
    }
}

/// The last segment of a tensor's name: its parameter's own name.
fn param_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or("")
}

impl ModuleAdapter for BunsenPyTorchToBurnAdapter {
    fn adapt(
        &self,
        mut tensor: PackTensor,
        ctx: ModuleContext<'_>,
    ) -> PackTensor {
        let Some(module_type) = ctx.module_type() else {
            return tensor;
        };
        let param = param_name(&tensor.name).to_string();
        match Self::rule(module_type, &param, tensor.shape.len(), tensor.dtype) {
            (true, _) => {
                let (name, dtype) = (tensor.name.clone(), tensor.dtype);
                let shape = Shape::from([tensor.shape[1], tensor.shape[0]]);
                bridge::map_data(tensor, name, dtype, shape, transpose_data)
            }
            (false, Some(burn_name)) => {
                let keep = tensor.name.len() - param.len();
                tensor.name.truncate(keep);
                tensor.name.push_str(burn_name);
                tensor
            }
            (false, None) => tensor,
        }
    }

    fn get_alternative_param_name(
        &self,
        param_name: &str,
        container_type: &str,
    ) -> Option<String> {
        if !NORMALIZATIONS.contains(&container_type) {
            return None;
        }
        match param_name {
            "gamma" => Some("weight".to_string()),
            "beta" => Some("bias".to_string()),
            _ => None,
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        module::Module,
        nn::{
            LayerNorm,
            LayerNormConfig,
            Linear,
            LinearConfig,
        },
        store::{
            ModuleSnapshot,
            PyTorchToBurnAdapter,
        },
        tensor::TensorData,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    #[derive(Module, Debug)]
    struct Net {
        proj: Linear,
        norm: LayerNorm,
    }

    fn net() -> Net {
        let device = cpu_device();
        Net {
            proj: LinearConfig::new(5, 3).init(&device),
            norm: LayerNormConfig::new(3).init(&device),
        }
    }

    /// A state dict as `PyTorch` names and lays it out.
    fn state_dict() -> alloc::vec::Vec<PackTensor> {
        let tensor = |name: &str, shape: alloc::vec::Vec<usize>| {
            let n: usize = shape.iter().product();
            let values: alloc::vec::Vec<f32> = (0..n).map(|v| v as f32 * 0.5 - 1.0).collect();
            bridge::from_data(TensorData::new(values, shape), name.to_string(), None)
        };
        alloc::vec![
            tensor("proj.weight", alloc::vec![3, 5]),
            tensor("proj.bias", alloc::vec![3]),
            tensor("norm.weight", alloc::vec![3]),
            tensor("norm.bias", alloc::vec![3]),
        ]
    }

    /// Every parameter, loaded through either adapter, is the same.
    #[test]
    fn test_loads_what_burns_adapter_loads() {
        let (mut theirs, mut ours) = (net(), net());
        let a = theirs.apply(
            state_dict(),
            None,
            Some(Box::new(PyTorchToBurnAdapter)),
            true,
        );
        let b = ours.apply(
            state_dict(),
            None,
            Some(Box::new(BunsenPyTorchToBurnAdapter)),
            true,
        );
        assert!(a.errors.is_empty() && b.errors.is_empty());
        assert_eq!(a.applied.len(), 4);
        assert_eq!(b.applied, a.applied);
        let data = |t: Tensor<2>| t.into_data();
        data(ours.proj.weight.val()).assert_eq(&data(theirs.proj.weight.val()), true);
        for (o, t) in [
            (
                ours.proj.bias.clone().unwrap().val(),
                theirs.proj.bias.clone().unwrap().val(),
            ),
            (ours.norm.gamma.val(), theirs.norm.gamma.val()),
            (
                ours.norm.beta.clone().unwrap().val(),
                theirs.norm.beta.clone().unwrap().val(),
            ),
        ] {
            o.into_data().assert_eq(&t.into_data(), true);
        }
    }

    #[test]
    fn test_rule() {
        assert_eq!(
            BunsenPyTorchToBurnAdapter::rule(LINEAR, "weight", 2, DType::BF16),
            (true, None)
        );
        assert_eq!(
            BunsenPyTorchToBurnAdapter::rule(LINEAR, "bias", 1, DType::BF16),
            (false, None)
        );
        assert_eq!(
            BunsenPyTorchToBurnAdapter::rule("Struct:RmsNorm", "weight", 1, DType::F32),
            (false, Some("gamma"))
        );
        assert_eq!(
            BunsenPyTorchToBurnAdapter::rule("Struct:Embedding", "weight", 2, DType::F32),
            (false, None)
        );
    }
}
