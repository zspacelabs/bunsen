//! Type mapper for changing the data type of tensors in a module.
use burn::{
    Tensor,
    module::{
        ModuleMapper,
        Param,
    },
    tensor::DType,
};

/// Type mapper for changing the data type of tensors in a module.
pub struct DTypeMapper {
    /// Target data type for tensor conversion.
    pub dt: DType,
}

impl DTypeMapper {
    /// Creates a new type mapper with the specified target data type.
    pub fn new(dt: DType) -> Self {
        Self { dt }
    }
}

impl ModuleMapper for DTypeMapper {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<D>>,
    ) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(self.dt))
    }
}
