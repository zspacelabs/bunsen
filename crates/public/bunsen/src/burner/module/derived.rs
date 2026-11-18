//! [`Derived`]: a parameter forward reads through a fixed transform.

use core::fmt;

use burn::{
    Tensor,
    module::{
        Content,
        Devices,
        Module,
        ModuleDisplay,
        ModuleDisplayDefault,
        ModuleMapper,
        ModuleVisitor,
        Param,
    },
    tensor::Device,
};

/// Which form a [`Derived`] parameter holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParamForm {
    /// The form a checkpoint stores and training updates: the transform is
    /// applied on every read. Every parameter starts here, as built or
    /// loaded.
    #[default]
    Stored,

    /// The transform already applied ([`Module::materialize`], or a source
    /// that stores the effective form): a read returns it as it is.
    Effective,
}

/// A named transform from a parameter's stored form to the form forward
/// uses: `abs`, `-exp`, a widening cast.
#[derive(Clone, Copy)]
pub struct ParamFn<const D: usize> {
    /// What the transform is called, for reports and errors.
    pub name: &'static str,

    /// The transform.
    pub apply: fn(Tensor<D>) -> Tensor<D>,
}

impl<const D: usize> ParamFn<D> {
    /// The tensor widened to at least f32 ([`Widening::F32`]): a width-only
    /// transform, its effective form the parameter held wide.
    ///
    /// [`Widening::F32`]: crate::burner::tensor::Widening::F32
    pub const WIDEN_F32: Self = Self {
        name: "widen_f32",
        apply: widen_f32::<D>,
    };
}

/// `t` widened to at least f32.
fn widen_f32<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    crate::burner::tensor::Widening::F32
        .scope(t.dtype())
        .widen(t)
}

impl<const D: usize> fmt::Debug for ParamFn<D> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "ParamFn({})", self.name)
    }
}

/// A parameter forward reads through a fixed transform, as
/// [`effective`](Self::effective): computed on every read in its
/// [`Stored`](ParamForm::Stored) form, or folded once into the parameter by
/// [`Module::materialize`], after which a read costs nothing.
///
/// The parameter keeps its own name in a record: the module visits and maps
/// it as it would a bare [`Param`], so checkpoint keys are a bare param's,
/// and a load writes the stored form (into a parameter built
/// [`Stored`](ParamForm::Stored)). Training reads the stored form through
/// the transform, so gradients reach the stored parameter; a materialized
/// parameter is for inference, and refuses [`Module::train`].
#[derive(Clone, Debug)]
pub struct Derived<const D: usize> {
    param: Param<Tensor<D>>,
    transform: ParamFn<D>,
    form: ParamForm,
}

impl<const D: usize> Derived<D> {
    /// `param`, in its stored form, read through `transform`.
    pub fn new(
        param: Param<Tensor<D>>,
        transform: ParamFn<D>,
    ) -> Self {
        Self {
            param,
            transform,
            form: ParamForm::Stored,
        }
    }

    /// The same, declared as holding `form`: a loader whose source stores
    /// the effective form says so.
    pub fn with_form(
        mut self,
        form: ParamForm,
    ) -> Self {
        self.form = form;
        self
    }

    /// The value forward uses: the transform of the stored form, or the
    /// effective form as held.
    pub fn effective(&self) -> Tensor<D> {
        match self.form {
            ParamForm::Stored => (self.transform.apply)(self.param.val()),
            ParamForm::Effective => self.param.val(),
        }
    }

    /// The value held, in whichever form [`form`](Self::form) says.
    pub fn held(&self) -> Tensor<D> {
        self.param.val()
    }

    /// The parameter.
    pub fn param(&self) -> &Param<Tensor<D>> {
        &self.param
    }

    /// The form held.
    pub fn form(&self) -> ParamForm {
        self.form
    }

    /// The transform's name.
    pub fn transform_name(&self) -> &'static str {
        self.transform.name
    }

    /// The effective form folded into the parameter, once; as it is when
    /// already [`Effective`](ParamForm::Effective).
    pub fn bake(self) -> Self {
        match self.form {
            ParamForm::Effective => self,
            ParamForm::Stored => {
                let value = self.effective().detach();
                let require_grad = self.param.val().is_require_grad();
                Self {
                    param: Param::initialized(self.param.id, value.set_require_grad(require_grad)),
                    transform: self.transform,
                    form: ParamForm::Effective,
                }
            }
        }
    }

    fn with_param(
        self,
        param: Param<Tensor<D>>,
    ) -> Self {
        Self { param, ..self }
    }
}

impl<const D: usize> Module for Derived<D> {
    fn collect_devices(
        &self,
        devices: Devices,
    ) -> Devices {
        Module::collect_devices(&self.param, devices)
    }

    fn fork(
        self,
        device: &Device,
    ) -> Self {
        let param = Module::fork(self.param.clone(), device);
        self.with_param(param)
    }

    fn to_device(
        self,
        device: &Device,
    ) -> Self {
        let param = Module::to_device(self.param.clone(), device);
        self.with_param(param)
    }

    /// # Panics
    /// For an [`Effective`](ParamForm::Effective) parameter: training it
    /// would update the transformed value, not the stored one.
    fn train(self) -> Self {
        assert_eq!(
            self.form,
            ParamForm::Stored,
            "a materialized `{}` parameter is for inference; train the stored form",
            self.transform.name
        );
        let param = Module::train(self.param.clone());
        self.with_param(param)
    }

    fn valid(&self) -> Self {
        self.clone().with_param(Module::valid(&self.param))
    }

    /// The parameter's own reparameterizations folded first (a LoRA adapter),
    /// then this transform ([`bake`](Self::bake)).
    fn materialize(self) -> Self {
        let param = Module::materialize(self.param.clone());
        self.with_param(param).bake()
    }

    fn visit<Visitor: ModuleVisitor>(
        &self,
        visitor: &mut Visitor,
    ) {
        Module::visit(&self.param, visitor);
    }

    fn map<Mapper: ModuleMapper>(
        self,
        mapper: &mut Mapper,
    ) -> Self {
        let param = Module::map(self.param.clone(), mapper);
        self.with_param(param)
    }
}

impl<const D: usize> ModuleDisplayDefault for Derived<D> {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        content
            .add_formatted(&format!(
                "Derived {{rank: {D}, shape: {:?}, transform: {}, form: {:?}}}",
                self.param.shape().as_slice(),
                self.transform.name,
                self.form
            ))
            .optional()
    }
}

impl<const D: usize> ModuleDisplay for Derived<D> {}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::support::testing::cpu_device;

    const NEG_EXP: ParamFn<1> = ParamFn {
        name: "neg_exp",
        apply: |t| t.exp().neg(),
    };

    fn values(t: Tensor<1>) -> Vec<f32> {
        t.into_data().try_into_vec_as::<f32>().unwrap()
    }

    fn derived() -> Derived<1> {
        Derived::new(
            Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.0f32, 1.0], [2]),
                &cpu_device(),
            )),
            NEG_EXP,
        )
    }

    /// Stored, the transform on every read; materialized, the same value
    /// once, held; a second materialize applies nothing more.
    #[test]
    fn test_effective_stored_and_baked() {
        let stored = derived();
        assert_eq!(stored.form(), ParamForm::Stored);
        let expected = values(stored.effective());
        assert_eq!(expected, [-1.0, -core::f32::consts::E]);

        let baked = stored.clone().materialize();
        assert_eq!(baked.form(), ParamForm::Effective);
        assert_eq!(values(baked.effective()), expected);
        assert_eq!(values(baked.held()), expected);
        assert_eq!(
            values(baked.clone().materialize().effective()),
            expected,
            "once"
        );
        assert_eq!(baked.param().id, stored.param().id, "the same parameter");
    }

    /// A source that stores the effective form is read as it is.
    #[test]
    fn test_declared_effective() {
        let declared = derived().with_form(ParamForm::Effective);
        assert_eq!(values(declared.effective()), [0.0, 1.0]);
    }

    /// The module visits the parameter as a bare one: one float, no levels
    /// of its own.
    #[test]
    fn test_visits_as_a_bare_param() {
        #[derive(Default)]
        struct Count {
            floats: usize,
            modules: usize,
        }
        impl ModuleVisitor for Count {
            fn visit_float<const D: usize>(
                &mut self,
                _: &Param<Tensor<D>>,
            ) {
                self.floats += 1;
            }

            fn enter_module(
                &mut self,
                _: &str,
                _: &str,
            ) {
                self.modules += 1;
            }
        }
        let mut count = Count::default();
        derived().visit(&mut count);
        assert_eq!((count.floats, count.modules), (1, 0));
    }

    /// A materialized parameter refuses training.
    #[test]
    #[should_panic(expected = "materialized `neg_exp`")]
    fn test_train_refuses_effective() {
        let _ = derived().materialize().train();
    }
}
