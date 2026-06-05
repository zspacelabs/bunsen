use burn::{
    module::Module,
    prelude::Backend,
};

use crate::errors::{
    BunsenResult,
    WithOkOrPanic,
};

/// Trait for initializing a Module.
pub trait ModuleInit<B: Backend, M: Module<B>> {
    /// Initializes a Module.
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<M>;

    /// Initializes a Module, panicking on error.
    fn init(
        &self,
        device: &B::Device,
    ) -> M {
        self.try_init(device).ok_or_panic()
    }
}
