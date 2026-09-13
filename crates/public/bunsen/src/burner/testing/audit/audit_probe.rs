use core::fmt::Debug;
use std::{
    collections::HashMap,
    panic::Location,
};

use burn::{
    Tensor,
    prelude::{
        Backend,
        TensorData,
    },
    tensor::{
        BasicOps,
        Element,
    },
};

use crate::{
    burner::testing::audit::{
        AuditProbeEventParams,
        AuditProbeEventPrefix,
        AuditProbeEventStub,
        handlers::AuditProbeEventHandler,
    },
    errors::BunsenResult,
    prelude::TensorElemOpExt,
};

/// Probe for auditing purposes.
#[derive(Debug)]
pub struct AuditProbe {
    handlers: Vec<Box<dyn AuditProbeEventHandler>>,
}

impl AuditProbe {
    /// Construct a new audit probe.
    pub fn new(handlers: Vec<Box<dyn AuditProbeEventHandler>>) -> Self {
        Self { handlers }
    }

    /// Dispatch event to handlers.
    ///
    /// # Arguments
    /// * `event` - an event stub to handle.
    ///
    /// # Panics and/or Err Returns
    /// If the event does not match the expected event under verification.
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        for handler in &mut self.handlers {
            handler.on_event(stub)?;
        }
        Ok(())
    }

    /// Location forwarding impl of `assert_eq`.
    fn loc_assert_eq(
        &mut self,
        _label: &str,
        _location: &Location,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        let data: HashMap<String, Vec<&TensorData>> =
            HashMap::from([("data".to_string(), vec![data])]);

        self.on_event(&AuditProbeEventStub {
            prefix: AuditProbeEventPrefix::new(None),
            params: AuditProbeEventParams::AssertEq { strict },
            data,
        })
    }

    /// [`TensorData`] equality.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    pub fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        self.loc_assert_eq(label, Location::caller(), data, strict)
    }

    /// [`Tensor`] equality.
    ///
    /// Data is used as `tensor.to_data_as::<E>()`.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    pub fn assert_tensor_eq_as<B, const R: usize, K, E>(
        &mut self,
        label: &str,
        tensor: &Tensor<B, R, K>,
        strict: bool,
    ) -> BunsenResult<()>
    where
        E: Element,
        B: Backend,
        K: BasicOps<B>,
    {
        let data = tensor.to_data_as::<E>();
        self.loc_assert_eq(label, Location::caller(), &data, strict)
    }
}
