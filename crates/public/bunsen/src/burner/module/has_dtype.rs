use burn::tensor::DType;

/// This target/semantic [`DType`].
///
/// This is not a guarantee that all internal params are this dtype;
/// only that callers should assume this dtype is appropriate for inputs.
pub trait HasDType {
    /// The semantic [`DType`] of the module.
    fn dtype(&self) -> DType;
}
