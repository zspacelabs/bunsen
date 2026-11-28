//! The default operator environment.

use crate::core::operations::{
    environment::MapOpEnvironment,
    registration::FirehoseOperatorFactoryRegistration,
};

/// Build the default environment.
///
/// This constructs a [`MapOpEnvironment`] and adds every operator factory
/// registered with
/// [`define_firehose_operator`](crate::define_firehose_operator) or
/// [`register_firehose_operator_factory`](crate::register_firehose_operator_factory)
/// in any linked crate.
///
/// Each call creates a new environment, which the caller may extend.
///
/// # Panics
///
/// If two registrations share an operator id, or a registered factory's
/// signature has a different id from its registration.
pub fn init_default_operator_environment() -> MapOpEnvironment {
    let mut env = MapOpEnvironment::default();

    for reg in FirehoseOperatorFactoryRegistration::list_default_registrations() {
        env.add_operator(reg.get_builder()).unwrap();
    }

    env
}
