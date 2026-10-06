//! Exact unpacking of audit event data maps.

use std::collections::HashMap;

use burn::prelude::TensorData;

use crate::errors::{
    BunsenError,
    BunsenResult,
    ValueMismatch,
};

/// Borrowed view over an event's data map.
///
/// This is the shape returned by
/// [`AuditProbeEventView::data_map_view`](`crate::audit::AuditProbeEventView::data_map_view`).
///
/// Read one with [`unpack_audit_probe_event_data!`], which states the whole
/// map as a pattern. The functions beside this type,
/// [`assert_exact_data_keys`], [`take_one`], [`take_fixed`] and [`take_any`],
/// are what the macro expands to. They are public because the expansion
/// calls them from the caller's crate; use the macro rather than them.
///
/// [`unpack_audit_probe_event_data!`]: crate::audit::unpack_audit_probe_event_data
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
/// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
/// [`ValueMismatch`] cause, if `view` holds any undeclared key. The error has
/// a frame naming `target`.
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

    Err(mismatch(
        target,
        format!("unexpected event data keys {unexpected:?}; pattern declares {keys:?}"),
    ))
}

/// An event data map that does not match the pattern: a [`ValueMismatch`],
/// under a frame naming the `target`.
#[track_caller]
fn mismatch(
    target: &str,
    summary: String,
) -> BunsenError {
    BunsenError::from(ValueMismatch::Other {
        summary,
        details: None,
    })
    .context(format!("unpacking `{target}`"))
}

/// Build the "wrong arity" error for a key.
#[track_caller]
fn arity_error(
    target: &str,
    key: &str,
    expected: &str,
    actual: usize,
) -> BunsenError {
    mismatch(
        target,
        format!("event data key `{key}` has {actual} values, expected {expected}"),
    )
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
            Err(mismatch(
                target,
                format!("missing event data key `{key}`; present keys {present:?}"),
            ))
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
/// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
/// [`ValueMismatch`] cause, if `key` is absent, or is not bound to exactly one
/// value. The error has a frame naming `target`.
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
/// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
/// [`ValueMismatch`] cause, if `key` is absent, or is not bound to exactly `K`
/// values. The error has a frame naming `target`.
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
/// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
/// [`ValueMismatch`] cause, if `key` is absent. The error has a frame naming
/// `target`.
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
/// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
/// [`ValueMismatch`](crate::errors::ValueMismatch) cause, if any target's data
/// map does not match the pattern exactly. The error has a frame naming the
/// target expression: ``unpacking `&event` ``.
pub use crate::__unpack_audit_probe_event_data as unpack_audit_probe_event_data;

#[doc(hidden)]
#[macro_export]
macro_rules! __unpack_audit_probe_event_data {
    ([$($target:expr),+ $(,)?], { $($name:ident $(: $arity:tt)?),+ $(,)? } $(,)?) => {{
        #[allow(dead_code)]
        #[derive(Debug)]
        struct Unpacked<'x> {
            $($name: $crate::__unpack_audit_probe_event_data!(@ty 'x, $($arity)?),)+
        }

        fn unpack_one<'x, V>(
            target: &str,
            event: &'x V,
        ) -> $crate::errors::BunsenResult<Unpacked<'x>>
        where
            V: $crate::audit::AuditProbeEventView,
        {
            let view = $crate::audit::AuditProbeEventView::data_map_view(event);
            $crate::audit::assert_exact_data_keys(
                target,
                &view,
                &[$(::core::stringify!($name)),+],
            )?;
            Ok(Unpacked {
                $($name: $crate::__unpack_audit_probe_event_data!(
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
        $crate::audit::take_one($target, $view, $key)
    };
    (@take $target:expr, $view:expr, $key:expr, [..]) => {
        $crate::audit::take_any($target, $view, $key)
    };
    (@take $target:expr, $view:expr, $key:expr, [$n:literal]) => {
        $crate::audit::take_fixed::<$n>($target, $view, $key)
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        audit::{
            AuditProbeEvent,
            AuditProbeEventHeader,
            AuditProbeEventStub,
            audit_probe::AuditProbeEventParams,
        },
        burner::descriptors::ToleranceDesc,
        errors::{
            BunsenErrorKind,
            testing::{
                ErrorMatcher,
                text,
            },
        },
    };

    fn event(data: HashMap<String, Vec<TensorData>>) -> AuditProbeEvent {
        AuditProbeEvent {
            header: AuditProbeEventHeader::new(None, None, None),
            params: AuditProbeEventParams::AssertTensorApproxEx {
                tolerance: ToleranceDesc::of::<f32>(Default::default()).unwrap(),
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

        let [unpacked] = unpack_audit_probe_event_data!([&e], {
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

        let [l, r] = unpack_audit_probe_event_data!([&lhs, &rhs], { data })?;

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
            params: AuditProbeEventParams::AssertTensorApproxEx {
                tolerance: ToleranceDesc::of::<f32>(Default::default())?,
            },
            data: HashMap::from([("data".to_string(), vec![&a])]),
        };

        let [from_owned, from_stub] = unpack_audit_probe_event_data!([&owned, &stub], { data })?;

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

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame(text::eq("unpacking `&e`"))
            .message_contains("unexpected event data keys [\"grad\"]")
            .message_contains("pattern declares [\"data\"]")
            .has_cause::<ValueMismatch>()
            .assert_err(&unpack_audit_probe_event_data!([&e], { data }));
    }

    #[test]
    fn test_rejects_missing_key() {
        let a = TensorData::from([1, 2, 3]);
        let e = event(HashMap::from([("data".to_string(), vec![a.clone()])]));

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame(text::eq("unpacking `&e`"))
            .message_contains("missing event data key `grad`")
            .message_contains("present keys [\"data\"]")
            .assert_err(&unpack_audit_probe_event_data!([&e], { data, grad }));
    }

    #[test]
    fn test_rejects_wrong_arity() {
        let a = TensorData::from([1, 2, 3]);
        let e = event(HashMap::from([(
            "data".to_string(),
            vec![a.clone(), a.clone()],
        )]));

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("key `data` has 2 values, expected 1")
            .assert_err(&unpack_audit_probe_event_data!([&e], { data }));

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("key `data` has 2 values, expected 3")
            .assert_err(&unpack_audit_probe_event_data!([&e], { data: [3] }));
    }

    /// The failing target is named, not just the key.
    #[test]
    fn test_names_the_failing_target() {
        let a = TensorData::from([1, 2, 3]);
        let good = event(HashMap::from([("data".to_string(), vec![a.clone()])]));
        let bad = event(HashMap::from([("other".to_string(), vec![a.clone()])]));

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame(text::eq("unpacking `&bad`"))
            .display_contains("unpacking `&bad`: ")
            .assert_err(&unpack_audit_probe_event_data!([&good, &bad], { data }));
    }
}
