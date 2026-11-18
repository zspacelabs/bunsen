//! [`Materialized`], and the inventory of a module's derived params.

use alloc::{
    string::String,
    vec::Vec,
};
use core::cell::RefCell;
use std::collections::HashMap;

use burn::{
    Tensor,
    module::{
        Module,
        ModuleVisitor,
        Param,
        ParamId,
    },
};

use super::ParamForm;
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A [`Derived`](super::Derived) param of a module: where it is, its transform,
/// and the form it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedParamInfo {
    /// Its path in the module, as its record key: `stack.layers.0.attn.a_log`.
    pub path: String,

    /// Its transform's name.
    pub transform: &'static str,

    /// The form it holds.
    pub form: ParamForm,
}

/// A [`Derived`](super::Derived) param as its `visit` notes it: its id,
/// transform and form.
type Noted = (ParamId, &'static str, ParamForm);

thread_local! {
    /// While [`derived_params`] walks a module, the params noted so far.
    static NOTED: RefCell<Option<Vec<Noted>>> = const { RefCell::new(None) };
}

/// Notes a [`Derived`](super::Derived) param, while an inventory is being
/// taken.
pub(super) fn note_derived(
    id: ParamId,
    transform: &'static str,
    form: ParamForm,
) {
    NOTED.with(|noted| {
        if let Some(noted) = noted.borrow_mut().as_mut() {
            noted.push((id, transform, form));
        }
    });
}

/// Every float param's path, by id.
#[derive(Default)]
struct Paths {
    at: Vec<String>,
    by_id: HashMap<ParamId, String>,
}

impl ModuleVisitor for Paths {
    fn enter_module(
        &mut self,
        name: &str,
        _container_type: &str,
    ) {
        self.at.push(name.into());
    }

    fn exit_module(
        &mut self,
        _name: &str,
        _container_type: &str,
    ) {
        self.at.pop();
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<D>>,
    ) {
        self.by_id.insert(param.id, self.at.join("."));
    }
}

/// `module`'s [`Derived`](super::Derived) params, in the order it visits them.
///
/// [`Derived`](super::Derived) visits as a bare param, so its record key is a
/// bare param's; this reads, beside the walk, what each one notes of itself.
pub fn derived_params<M: Module>(module: &M) -> Vec<DerivedParamInfo> {
    NOTED.with(|noted| *noted.borrow_mut() = Some(Vec::new()));
    let mut paths = Paths::default();
    module.visit(&mut paths);
    let noted = NOTED
        .with(|noted| noted.borrow_mut().take())
        .unwrap_or_default();
    noted
        .into_iter()
        .map(|(id, transform, form)| DerivedParamInfo {
            path: paths.by_id.get(&id).cloned().unwrap_or_default(),
            transform,
            form,
        })
        .collect()
}

/// Checks that every [`Derived`](super::Derived) param of `module` holds its
/// stored form, as a checkpoint load writes: loading into a materialized module
/// would leave raw values marked effective.
///
/// # Errors
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), naming the params
/// that hold their effective form.
pub fn ensure_derived_stored<M: Module>(module: &M) -> BunsenResult<()> {
    let effective: Vec<String> = derived_params(module)
        .into_iter()
        .filter(|p| p.form == ParamForm::Effective)
        .map(|p| format!("{} ({})", p.path, p.transform))
        .collect();
    if effective.is_empty() {
        Ok(())
    } else {
        Err(BunsenError::illegal(format!(
            "a checkpoint loads into stored forms; materialized: {}",
            effective.join(", ")
        )))
    }
}

/// A module materialized for inference ([`Module::valid`], then
/// [`Module::materialize`]): its [`Derived`](super::Derived) params hold their
/// effective forms, and [`baked`](Self::baked) says which it folded.
///
/// It has no load or save path: a checkpoint loads into, and is written
/// from, the stored forms, which a materialized module no longer holds.
#[derive(Clone, Debug)]
pub struct Materialized<M: Module> {
    module: M,
    baked: Vec<DerivedParamInfo>,
}

impl<M: Module> Materialized<M> {
    /// `module`, materialized; `module` as it was is left to the caller.
    pub fn new(module: &M) -> Self {
        let baked = derived_params(module)
            .into_iter()
            .filter(|p| p.form == ParamForm::Stored)
            .collect();
        Self {
            module: module.valid().materialize(),
            baked,
        }
    }

    /// The materialized module.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// The [`Derived`](super::Derived) params materializing folded: the ones
    /// that held their stored forms.
    pub fn baked(&self) -> &[DerivedParamInfo] {
        &self.baked
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::*;
    use crate::{
        burner::module::{
            Derived,
            ParamFn,
        },
        errors::BunsenErrorKind,
        support::testing::cpu_device,
    };

    #[derive(Module, Debug)]
    struct Holder {
        plain: Param<Tensor<1>>,
        gain: Derived<1>,
        inner: Inner,
    }

    #[derive(Module, Debug)]
    struct Inner {
        bias: Derived<2>,
    }

    fn holder() -> Holder {
        let device = cpu_device();
        Holder {
            plain: Param::from_tensor(Tensor::ones([3], &device)),
            gain: Derived::new(
                Param::from_tensor(Tensor::from_data([-1.0f32, 2.0], &device)),
                ParamFn::ABS,
            ),
            inner: Inner {
                bias: Derived::new(
                    Param::from_tensor(Tensor::zeros([2, 2], &device)),
                    ParamFn::WIDEN_F32,
                ),
            },
        }
    }

    /// Each `Derived` param, at its record path, with its transform and form.
    #[test]
    fn test_derived_params() {
        let found = derived_params(&holder());
        assert_eq!(
            found,
            [
                DerivedParamInfo {
                    path: "gain".into(),
                    transform: "abs",
                    form: ParamForm::Stored,
                },
                DerivedParamInfo {
                    path: "inner.bias".into(),
                    transform: "widen_f32",
                    form: ParamForm::Stored,
                },
            ]
        );
    }

    /// A built module may be loaded into; a materialized one may not.
    #[test]
    fn test_ensure_derived_stored() {
        let module = holder();
        ensure_derived_stored(&module).unwrap();
        let baked = Materialized::new(&module);
        assert_eq!(
            baked
                .baked()
                .iter()
                .map(|p| p.path.as_str())
                .collect::<Vec<_>>(),
            ["gain", "inner.bias"]
        );
        let error = ensure_derived_stored(baked.module()).unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Illegal);
        assert!(error.to_string().contains("gain (abs)"), "{error}");
        ensure_derived_stored(&module).unwrap();
    }
}
