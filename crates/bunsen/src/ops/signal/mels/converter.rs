use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Options for [`MelConverter`].
#[derive(Config, Debug, Default)]
pub struct MelConverterOptions {}

impl<B: Backend> ModuleInit<B, MelConverter<B>> for MelConverterOptions {
    /// Initializes a [`MelConverter`] on `device`.
    fn try_init(
        &self,
        _device: &B::Device,
    ) -> BunsenResult<MelConverter<B>> {
        Ok(MelConverter {
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Waveform to Mels conversion module.
#[derive(Module, Debug)]
pub struct MelConverter<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::WithOkOrPanic,
        support::testing::PerformanceBackend,
    };

    #[test]
    fn test_converter() {
        type B = PerformanceBackend;
        let device = Default::default();

        let options = MelConverterOptions::default();
        let _conv: MelConverter<B> = options.try_init(&device).ok_or_panic();
    }
}
