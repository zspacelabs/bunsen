//! Exact-unpack machinery for [`AuditProbeEventView`] data maps.
//!
//! See [`unpack_event_data`] for the one-shot pattern syntax.
//!
//! [`AuditProbeEventView`]: crate::burner::testing::audit::AuditProbeEventView

use std::collections::HashMap;

use burn::prelude::TensorData;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Borrowed view over an event's data map.
///
/// This is the shape returned by
/// [`AuditProbeEventView::data_map_view`](`crate::burner::testing::audit::AuditProbeEventView::data_map_view`).
pub type EventDataView<'a> = HashMap<&'a str, Vec<&'a TensorData>>;

/// Assert that `view` contains every key in `keys`, and no others.
///
/// Keys absent from `view` are reported by the `take_*` functions; this
/// reports keys present in `view` but not declared by the pattern. Together
/// they make an unpack pattern an exact description of the data map.
///
/// # Arguments
/// * `target` - the source expression, for diagnostics.
/// * `view` - the data map being unpacked.
/// * `keys` - the complete set of keys declared by the pattern.
///
/// # Errors
/// [`BunsenError::InvalidArgument`] if `view` holds any undeclared key.
pub fn assert_exact_data_keys(
    target: &str,
    view: &EventDataView<'_>,
    keys: &[&str],
) -> BunsenResult<()> {
    let mut unexpected: Vec<&str> = view
        .keys()
        .copied()
        .filter(|key| !keys.contains(key))
        .collect();

    if unexpected.is_empty() {
        return Ok(());
    }
    unexpected.sort_unstable();

    Err(BunsenError::InvalidArgument {
        msg: format!(
            "in `{target}`: unexpected event data keys {unexpected:?}; \
             pattern declares {keys:?}"
        ),
    })
}

/// Build the "wrong arity" error for a key.
fn arity_error(
    target: &str,
    key: &str,
    expected: &str,
    actual: usize,
) -> BunsenError {
    BunsenError::InvalidArgument {
        msg: format!(
            "in `{target}`: event data key `{key}` has {actual} values, expected {expected}"
        ),
    }
}

/// Look up `key`, or report it missing.
fn get_values<'v, 'x>(
    target: &str,
    view: &'v EventDataView<'x>,
    key: &str,
) -> BunsenResult<&'v [&'x TensorData]> {
    match view.get(key) {
        Some(values) => Ok(values),
        None => {
            let mut present: Vec<&str> = view.keys().copied().collect();
            present.sort_unstable();
            Err(BunsenError::InvalidArgument {
                msg: format!(
                    "in `{target}`: missing event data key `{key}`; present keys {present:?}"
                ),
            })
        }
    }
}

/// Take the single value bound to `key`.
///
/// # Arguments
/// * `target` - the source expression, for diagnostics.
/// * `view` - the data map being unpacked.
/// * `key` - the key to take.
///
/// # Errors
/// [`BunsenError::InvalidArgument`] if `key` is absent, or is not bound to
/// exactly one value.
pub fn take_one<'x>(
    target: &str,
    view: &EventDataView<'x>,
    key: &str,
) -> BunsenResult<&'x TensorData> {
    let values = get_values(target, view, key)?;
    match values {
        [value] => Ok(value),
        _ => Err(arity_error(target, key, "1", values.len())),
    }
}

/// Take exactly `K` values bound to `key`.
///
/// # Arguments
/// * `target` - the source expression, for diagnostics.
/// * `view` - the data map being unpacked.
/// * `key` - the key to take.
///
/// # Errors
/// [`BunsenError::InvalidArgument`] if `key` is absent, or is not bound to
/// exactly `K` values.
pub fn take_fixed<'x, const K: usize>(
    target: &str,
    view: &EventDataView<'x>,
    key: &str,
) -> BunsenResult<[&'x TensorData; K]> {
    let values = get_values(target, view, key)?;
    <[&'x TensorData; K]>::try_from(values)
        .map_err(|_| arity_error(target, key, &K.to_string(), values.len()))
}

/// Take every value bound to `key`.
///
/// The key must be present; the pattern asserts presence, not arity.
///
/// # Arguments
/// * `target` - the source expression, for diagnostics.
/// * `view` - the data map being unpacked.
/// * `key` - the key to take.
///
/// # Errors
/// [`BunsenError::InvalidArgument`] if `key` is absent.
pub fn take_any<'x>(
    target: &str,
    view: &EventDataView<'x>,
    key: &str,
) -> BunsenResult<Vec<&'x TensorData>> {
    Ok(get_values(target, view, key)?.to_vec())
}

/// Exact-unpack one or more event data maps against a single pattern.
///
/// Every target is unpacked against the same pattern into a nonce-defined
/// struct with one field per declared key, and the targets are returned as an
/// array in declaration order. Because all targets share the generated type,
/// unpacking them identically is a property of the type rather than a
/// convention.
///
/// The pattern is *exact*: a declared key must be present with the declared
/// arity, and any key not declared is an error. There is no escape hatch, so
/// reading the pattern tells you the complete data map.
///
/// # Arity Forms
/// * `name` - exactly one value; binds `&TensorData`.
/// * `name: [K]` - exactly `K` values; binds `[&TensorData; K]`.
/// * `name: [..]` - present, any number of values; binds `Vec<&TensorData>`.
///
/// Keys are written as identifiers and matched by name.
///
/// # Arguments
/// * targets - a bracketed list of `&impl AuditProbeEventView` expressions.
/// * pattern - a braced list of arity forms.
///
/// # Returns
/// [`BunsenResult`] of an array of unpacked structs, one per target.
///
/// # Errors
/// [`BunsenError::InvalidArgument`] if any target's data map does not match
/// the pattern exactly. The error names the target expression.
pub use crate::__unpack_event_data as unpack_event_data;

#[doc(hidden)]
#[macro_export]
macro_rules! __unpack_event_data {
    ([$($target:expr),+ $(,)?], { $($name:ident $(: $arity:tt)?),+ $(,)? } $(,)?) => {{
        #[allow(dead_code)]
        #[derive(Debug)]
        struct Unpacked<'x> {
            $($name: $crate::__unpack_event_data!(@ty 'x, $($arity)?),)+
        }

        fn unpack_one<'x, V>(
            target: &str,
            event: &'x V,
        ) -> $crate::errors::BunsenResult<Unpacked<'x>>
        where
            V: $crate::burner::testing::audit::AuditProbeEventView,
        {
            let view = $crate::burner::testing::audit::AuditProbeEventView::data_map_view(event);
            $crate::burner::testing::audit::assert_exact_data_keys(
                target,
                &view,
                &[$(::core::stringify!($name)),+],
            )?;
            Ok(Unpacked {
                $($name: $crate::__unpack_event_data!(
                    @take target, &view, ::core::stringify!($name), $($arity)?
                )?,)+
            })
        }

        (|| ::core::result::Result::Ok::<_, $crate::errors::BunsenError>([
            $(unpack_one(::core::stringify!($target), $target)?,)+
        ]))()
    }};

    (@ty $lt:lifetime,) => { &$lt ::burn::prelude::TensorData };
    (@ty $lt:lifetime, [..]) => { ::std::vec::Vec<&$lt ::burn::prelude::TensorData> };
    (@ty $lt:lifetime, [$n:literal]) => { [&$lt ::burn::prelude::TensorData; $n] };

    (@take $target:expr, $view:expr, $key:expr,) => {
        $crate::burner::testing::audit::take_one($target, $view, $key)
    };
    (@take $target:expr, $view:expr, $key:expr, [..]) => {
        $crate::burner::testing::audit::take_any($target, $view, $key)
    };
    (@take $target:expr, $view:expr, $key:expr, [$n:literal]) => {
        $crate::burner::testing::audit::take_fixed::<$n>($target, $view, $key)
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::burner::testing::audit::{
        AuditProbeEvent,
        AuditProbeEventHeader,
        AuditProbeEventStub,
        audit_probe::AuditProbeEventParams,
    };

    fn event(data: HashMap<String, Vec<TensorData>>) -> AuditProbeEvent {
        AuditProbeEvent {
            header: AuditProbeEventHeader::new(None, None, None),
            params: AuditProbeEventParams::AssertTensorEq {
                strict: true,
                tolerance: None,
            },
            data,
        }
    }

    #[test]
    fn test_arity_forms() -> BunsenResult<()> {
        let a = TensorData::from([1, 2, 3]);
        let b = TensorData::from([[2.0, 3.0], [4.0, 5.0]]);

        let e = event(HashMap::from([
            ("solo".to_string(), vec![a.clone()]),
            ("pair".to_string(), vec![a.clone(), b.clone()]),
            ("rest".to_string(), vec![b.clone(), a.clone(), b.clone()]),
        ]));

        let [unpacked] = unpack_event_data!([&e], {
            solo,
            pair: [2],
            rest: [..],
        })?;

        assert_eq!(unpacked.solo, &a);
        assert_eq!(unpacked.pair, [&a, &b]);
        assert_eq!(unpacked.rest, vec![&b, &a, &b]);

        Ok(())
    }

    #[test]
    fn test_multiple_targets() -> BunsenResult<()> {
        let a = TensorData::from([1, 2, 3]);
        let b = TensorData::from([4, 5, 6]);

        let lhs = event(HashMap::from([("data".to_string(), vec![a.clone()])]));
        let rhs = event(HashMap::from([("data".to_string(), vec![b.clone()])]));

        let [l, r] = unpack_event_data!([&lhs, &rhs], { data })?;

        assert_eq!(l.data, &a);
        assert_eq!(r.data, &b);

        Ok(())
    }

    /// Targets may be different view types; the unpacked type is the same.
    #[test]
    fn test_heterogeneous_targets() -> BunsenResult<()> {
        let a = TensorData::from([1, 2, 3]);

        let owned = event(HashMap::from([("data".to_string(), vec![a.clone()])]));
        let stub = AuditProbeEventStub {
            header: AuditProbeEventHeader::new(None, None, None),
            params: AuditProbeEventParams::AssertTensorEq {
                strict: true,
                tolerance: None,
            },
            data: HashMap::from([("data".to_string(), vec![&a])]),
        };

        let [from_owned, from_stub] = unpack_event_data!([&owned, &stub], { data })?;

        assert_eq!(from_owned.data, from_stub.data);

        Ok(())
    }

    #[test]
    fn test_rejects_undeclared_key() {
        let a = TensorData::from([1, 2, 3]);
        let e = event(HashMap::from([
            ("data".to_string(), vec![a.clone()]),
            ("grad".to_string(), vec![a.clone()]),
        ]));

        let err = unpack_event_data!([&e], { data }).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("in `&e`"), "{msg}");
        assert!(
            msg.contains("unexpected event data keys [\"grad\"]"),
            "{msg}"
        );
        assert!(msg.contains("pattern declares [\"data\"]"), "{msg}");
    }

    #[test]
    fn test_rejects_missing_key() {
        let a = TensorData::from([1, 2, 3]);
        let e = event(HashMap::from([("data".to_string(), vec![a.clone()])]));

        let err = unpack_event_data!([&e], { data, grad }).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("missing event data key `grad`"), "{msg}");
        assert!(msg.contains("present keys [\"data\"]"), "{msg}");
    }

    #[test]
    fn test_rejects_wrong_arity() {
        let a = TensorData::from([1, 2, 3]);
        let e = event(HashMap::from([(
            "data".to_string(),
            vec![a.clone(), a.clone()],
        )]));

        let solo = unpack_event_data!([&e], { data }).unwrap_err();
        assert!(
            solo.to_string()
                .contains("key `data` has 2 values, expected 1"),
            "{solo}"
        );

        let fixed = unpack_event_data!([&e], { data: [3] }).unwrap_err();
        assert!(
            fixed
                .to_string()
                .contains("key `data` has 2 values, expected 3"),
            "{fixed}"
        );
    }

    /// The failing target is named, not just the key.
    #[test]
    fn test_names_the_failing_target() {
        let a = TensorData::from([1, 2, 3]);
        let good = event(HashMap::from([("data".to_string(), vec![a.clone()])]));
        let bad = event(HashMap::from([("other".to_string(), vec![a.clone()])]));

        let err = unpack_event_data!([&good, &bad], { data }).unwrap_err();

        assert!(err.to_string().contains("in `&bad`"), "{err}");
    }
}
