//! [`Forward1Module`] … [`Forward4Module`], a module's forward by arity.

use burn::module::Module;

/// One forward-module trait: its name, its doc line, and its arguments.
macro_rules! forward_module {
    ($(#[$doc:meta])* $name:ident: $($arg:ident: $ty:ident),+) => {
        $(#[$doc])*
        ///
        /// See [Forward modules](crate::burner::module#forward-modules).
        pub trait $name<$($ty),+>: Module {
            /// What the forward returns.
            type Output;

            /// The module's forward.
            fn forward(
                &self,
                $($arg: $ty),+
            ) -> Self::Output;
        }
    };
}

forward_module!(
    /// A module whose forward takes one argument.
    Forward1Module: a1: A1
);
forward_module!(
    /// A module whose forward takes two arguments.
    Forward2Module: a1: A1, a2: A2
);
forward_module!(
    /// A module whose forward takes three arguments.
    Forward3Module: a1: A1, a2: A2, a3: A3
);
forward_module!(
    /// A module whose forward takes four arguments.
    Forward4Module: a1: A1, a2: A2, a3: A3, a4: A4
);

/// A module whose forward takes one argument: [`Forward1Module`], by its
/// common name.
pub use self::Forward1Module as ForwardModule;

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        prelude::Device,
        tensor::Distribution,
    };

    use super::*;
    use crate::support::testing::{
        cpu_device,
        seeded_tensor,
    };

    /// A module with an inherent forward and forwards of two arities.
    #[derive(Module, Debug)]
    struct Offset {
        by: f64,
    }

    impl Offset {
        fn forward<const R: usize>(
            &self,
            x: Tensor<R>,
        ) -> Tensor<R> {
            x.add_scalar(self.by)
        }
    }

    impl<const R: usize> Forward1Module<Tensor<R>> for Offset {
        type Output = Tensor<R>;

        fn forward(
            &self,
            x: Tensor<R>,
        ) -> Tensor<R> {
            Offset::forward(self, x)
        }
    }

    impl<const R: usize> Forward2Module<Tensor<R>, Tensor<R>> for Offset {
        type Output = Tensor<R>;

        fn forward(
            &self,
            x: Tensor<R>,
            y: Tensor<R>,
        ) -> Tensor<R> {
            Offset::forward(self, x + y)
        }
    }

    fn twice<const R: usize, M: ForwardModule<Tensor<R>, Output = Tensor<R>>>(
        module: &M,
        x: Tensor<R>,
    ) -> Tensor<R> {
        module.forward(module.forward(x))
    }

    /// Bound on two arities, code names the trait it calls.
    fn both<const R: usize, M>(
        module: &M,
        x: Tensor<R>,
    ) -> Tensor<R>
    where
        M: Forward1Module<Tensor<R>, Output = Tensor<R>>
            + Forward2Module<Tensor<R>, Tensor<R>, Output = Tensor<R>>,
    {
        let once = Forward1Module::forward(module, x.clone());
        Forward2Module::forward(module, once, x)
    }

    fn input(device: &Device) -> Tensor<2> {
        seeded_tensor::<2>(1, [2, 3], Distribution::Default, device)
    }

    #[test]
    fn test_forward_modules() {
        let device = cpu_device();
        let offset = Offset { by: 0.5 };
        let x = input(&device);
        // The inherent forward resolves first.
        offset
            .forward(x.clone())
            .into_data()
            .assert_eq(&x.clone().add_scalar(0.5).into_data(), true);
        twice(&offset, x.clone())
            .into_data()
            .assert_eq(&x.clone().add_scalar(0.5).add_scalar(0.5).into_data(), true);
        both(&offset, x.clone()).into_data().assert_eq(
            &(x.clone().add_scalar(0.5) + x).add_scalar(0.5).into_data(),
            true,
        );
    }
}
