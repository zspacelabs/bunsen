//! # Operator registry
//!
//! Operators can be registered *globally* with the
//! [`register_firehose_operator_factory!
//! `](crate::register_firehose_operator_factory) / [`define_firehose_operator!
//! `](crate::define_firehose_operator) macros (built on [`inventory`]).
//! [`init_default_operator_environment`](crate::ops::init_default_operator_environment)
//! then collects every such registration — across all linked crates — into a
//! fresh [`MapOpEnvironment`] ready to validate and run build plans.
//!
//! This is how downstream crates publish reusable operators: e.g.
//! `bunsen-firehose-image` registers `IMAGE_TO_TENSOR_DATA`, `AUGMENT_IMAGE`,
//! and friends, which become available simply by linking the crate.
//!
//! # Example
//!
//! ```
//! use bunsen_firehose::{
//!     core::operations::environment::FirehoseOperatorEnvironment,
//!     ops::init_default_operator_environment,
//! };
//!
//! // A fresh, owned environment pre-populated with all registered operators.
//! let env = init_default_operator_environment();
//!
//! // Operators are looked up by their fully-qualified string id.
//! assert!(
//!     env.lookup_operator_factory("fh:op://not::a::Real::Op")
//!         .is_err()
//! );
//! ```
//!
//! [`MapOpEnvironment`]: crate::core::operations::environment::MapOpEnvironment

use crate::core::operations::{
    environment::MapOpEnvironment,
    registration::FirehoseOperatorFactoryRegistration,
};

/// Build the default environment.
///
/// This constructs a `MapOpEnvironment` and adds all operator builders
/// registered with `bunsen_firehose::register_default_operator_builder!`.
///
/// Each call `build_default_environment` will create a new mutable environment.
pub fn init_default_operator_environment() -> MapOpEnvironment {
    let mut env = MapOpEnvironment::default();

    for reg in FirehoseOperatorFactoryRegistration::list_default_registrations() {
        env.add_operator(reg.get_builder()).unwrap();
    }

    env
}
