//! Configs compared as data: `assert_config_eq`, `config_eq`.

use alloc::{
    format,
    string::String,
    vec::Vec,
};

use serde::Serialize;
use serde_json::Value;

/// The config as JSON, for comparison.
fn as_value<C: Serialize>(config: &C) -> Value {
    serde_json::to_value(config).expect("a config that serializes")
}

/// The JSON pointers where `a` and `b` differ, each with both values.
fn differences(
    pointer: &str,
    a: &Value,
    b: &Value,
    out: &mut Vec<String>,
) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: alloc::collections::BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            for key in keys {
                let null = Value::Null;
                let path = format!("{pointer}/{key}");
                differences(
                    &path,
                    a.get(key).unwrap_or(&null),
                    b.get(key).unwrap_or(&null),
                    out,
                );
            }
        }
        (Value::Array(xs), Value::Array(ys)) if xs.len() == ys.len() => {
            for (k, (x, y)) in xs.iter().zip(ys).enumerate() {
                differences(&format!("{pointer}/{k}"), x, y, out);
            }
        }
        _ if a != b => out.push(format!(
            "  {}: {a} != {b}",
            if pointer.is_empty() { "/" } else { pointer }
        )),
        _ => {}
    }
}

/// Whether two configs are the same config: equal as [`serde_json::Value`]s.
///
/// burn's `#[derive(Config)]` types have no `PartialEq`; their serialized
/// form is what a config file holds, so it is what two configs agree on.
///
/// # Panics
///
/// If either fails to serialize.
pub fn config_eq<C: Serialize>(
    a: &C,
    b: &C,
) -> bool {
    as_value(a) == as_value(b)
}

/// Asserts that two configs are the same config, as [`config_eq`].
///
/// ```
/// use bunsen::support::testing::assert_config_eq;
///
/// #[derive(serde::Serialize)]
/// struct Stem {
///     width: usize,
///     deep: bool,
/// }
///
/// assert_config_eq(
///     "a stem",
///     &Stem {
///         width: 64,
///         deep: true,
///     },
///     &Stem {
///         width: 64,
///         deep: true,
///     },
/// );
/// ```
///
/// # Panics
///
/// When they differ; the message names `what` and lists each differing JSON
/// pointer with both values (`/resnet/stem_width: 32 != 64`, actual first).
#[track_caller]
pub fn assert_config_eq<C: Serialize>(
    what: &str,
    actual: &C,
    expected: &C,
) {
    let mut out = Vec::new();
    differences("", &as_value(actual), &as_value(expected), &mut out);
    assert!(
        out.is_empty(),
        "{what}: the configs differ at {} places (actual != expected):\n{}",
        out.len(),
        out.join("\n")
    );
}

#[cfg(test)]
mod tests {
    use alloc::{
        vec,
        vec::Vec,
    };

    use super::*;

    #[derive(Serialize)]
    struct Stem {
        width: usize,
        layers: Vec<usize>,
        norm: Option<&'static str>,
    }

    #[test]
    fn test_equal_configs() {
        let a = Stem {
            width: 64,
            layers: vec![2, 2],
            norm: None,
        };
        assert!(config_eq(&a, &a));
        assert_config_eq("a stem", &a, &a);
    }

    /// Each differing leaf is listed by its pointer; a list of another length
    /// is one difference.
    #[test]
    #[should_panic(
        expected = "a stem: the configs differ at 3 places (actual != expected):\n  \
                               /layers: [2,2] != [2,2,2]\n  /norm: \"batch\" != null\n  \
                               /width: 32 != 64"
    )]
    fn test_differences_are_listed() {
        let a = Stem {
            width: 32,
            layers: vec![2, 2],
            norm: Some("batch"),
        };
        let b = Stem {
            width: 64,
            layers: vec![2, 2, 2],
            norm: None,
        };
        assert!(!config_eq(&a, &b));
        assert_config_eq("a stem", &a, &b);
    }
}
