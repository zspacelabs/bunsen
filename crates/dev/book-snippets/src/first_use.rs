//! The first program in "Installing and first use".

/// Builds a block from its config, runs it, and checks its output shape.
pub fn first_use() {
    // ANCHOR: first_use
    use bunsen::{
        blocks::transformers::mlp::{
            Mlp,
            MlpConfig,
        },
        prelude::*,
        support::testing::cpu_device,
    };
    use burn::tensor::Tensor;

    let device = cpu_device();

    // `init` comes from `ModuleInit`; the binding's type names the module.
    let mlp: Mlp = MlpConfig::new(16).init(&device);

    let x = Tensor::<3>::zeros([2, 5, 16], &device);
    let y = mlp.forward(x);

    // Check the output's shape, and name its dimensions while doing it.
    let [batch, time, embed] = unpack_shape_contract!(["batch", "time", "embed"], &y.dims());
    assert_eq!((batch, time, embed), (2, 5, 16));
    // ANCHOR_END: first_use
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_use() {
        first_use();
    }
}
