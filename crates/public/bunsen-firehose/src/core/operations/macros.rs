//! The operator definition and registration macros.

/// Combined macro to define and register a firehose operator.
///
/// # Arguments
///
/// * `$name`: The name of the operator ID to define; will create a
///   self-referential static string constant.
/// * `$constructor`: An expression that builds the operator's factory, a
///   `FirehoseOperatorFactory` (or an `Arc` of one). It is evaluated each time
///   the registration is read.
///
/// This macro combines the functionality of `define_firehose_operator_id`
/// and `register_firehose_operator_factory`.
#[macro_export]
macro_rules! define_firehose_operator {
    ($name:ident, $constructor:expr) => {
        $crate::define_firehose_operator_id!($name);
        $crate::register_firehose_operator_factory!($name, $constructor);
    };
}

/// Define a self-referential operator ID.
///
/// The id will be defined as a static string constant that refers to its own
/// namespace path.
///
/// # Arguments
///
/// * `$name`: The name of the operator ID to define.
#[macro_export]
macro_rules! define_firehose_operator_id {
    ($name:ident) => {
        $crate::define_self_referential_id!("fh:op", $name);
    };
}

/// Macro to register a default operator factory.
///
/// Builders which do not require runtime configuration can be registered
/// using this macro; and collected globally using
/// [`FirehoseOperatorFactoryRegistration::list_default_registrations`](crate::core::operations::registration::FirehoseOperatorFactoryRegistration::list_default_registrations).
///
/// You can also collect a default environment with all registered builders
/// using [`init_default_operator_environment`](crate::ops::init_default_operator_environment).
#[macro_export]
macro_rules! register_firehose_operator_factory {
    ($name:ident, $constructor:expr) => {
        inventory::submit! {
            $crate::core::operations::registration::FirehoseOperatorFactoryRegistration {
                operator_id: $name,
                supplier: || {
                    let v = ($constructor);
                    std::sync::Arc::from(v) as
                    std::sync::Arc<dyn $crate::core::operations::factory::FirehoseOperatorFactory>
                },
            }
        }
    };
}
