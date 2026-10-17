//! Reading a `transformers` `config.json`.
use serde_json::Value;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A `transformers` `config.json`, or one section of it, read field by
/// field: what a kit maps onto its own config when an `hf:` repo carries
/// one.
///
/// Every accessor names the field's dotted path in its error, so a repo
/// whose config a kit cannot read says which field.
#[derive(Clone, Debug)]
pub struct HfConfigJson {
    path: String,
    value: Value,
}

impl HfConfigJson {
    /// Parses a `config.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it is not a JSON object.
    pub fn parse(json: &str) -> BunsenResult<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a config.json").with_cause(e))?;
        if !value.is_object() {
            return Err(BunsenError::invalid_resource(
                "a config.json that is not an object",
            ));
        }
        Ok(Self {
            path: String::new(),
            value,
        })
    }

    fn field(
        &self,
        name: &str,
    ) -> String {
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.path)
        }
    }

    fn get(
        &self,
        name: &str,
    ) -> Option<&Value> {
        self.value.get(name).filter(|v| !v.is_null())
    }

    fn missing(
        &self,
        name: &str,
        what: &str,
    ) -> BunsenError {
        BunsenError::invalid_resource(format!(
            "config.json: `{}` is missing or not {what}",
            self.field(name)
        ))
    }

    /// The object at `name`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// naming the field when it is missing or not an object.
    pub fn section(
        &self,
        name: &str,
    ) -> BunsenResult<Self> {
        match self.get(name) {
            Some(v) if v.is_object() => Ok(Self {
                path: self.field(name),
                value: v.clone(),
            }),
            _ => Err(self.missing(name, "an object")),
        }
    }

    /// The non-negative integer at `name`.
    ///
    /// # Errors
    /// As [`section`](Self::section), for a missing field or one that is not
    /// a non-negative integer.
    pub fn usize(
        &self,
        name: &str,
    ) -> BunsenResult<usize> {
        self.get(name)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| self.missing(name, "a non-negative integer"))
    }

    /// The number at `name`.
    ///
    /// # Errors
    /// As [`section`](Self::section), for a missing field or one that is not
    /// a number.
    pub fn f64(
        &self,
        name: &str,
    ) -> BunsenResult<f64> {
        self.get(name)
            .and_then(Value::as_f64)
            .ok_or_else(|| self.missing(name, "a number"))
    }

    /// The string at `name`.
    ///
    /// # Errors
    /// As [`section`](Self::section), for a missing field or one that is not
    /// a string.
    pub fn str(
        &self,
        name: &str,
    ) -> BunsenResult<&str> {
        self.get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| self.missing(name, "a string"))
    }

    /// The boolean at `name`, or `default` when it is absent.
    ///
    /// # Errors
    /// As [`section`](Self::section), for a field that is present and not a
    /// boolean.
    pub fn bool_or(
        &self,
        name: &str,
        default: bool,
    ) -> BunsenResult<bool> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v.as_bool().ok_or_else(|| self.missing(name, "a boolean")),
        }
    }

    /// The array of non-negative integers at `name`.
    ///
    /// # Errors
    /// As [`section`](Self::section), for a missing field or one that is not
    /// such an array.
    pub fn usizes(
        &self,
        name: &str,
    ) -> BunsenResult<Vec<usize>> {
        self.get(name)
            .and_then(Value::as_array)
            .and_then(|a| {
                a.iter()
                    .map(|v| v.as_u64().map(|v| v as usize))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or_else(|| self.missing(name, "an array of non-negative integers"))
    }

    /// Requires the string at `name` to be `expected`.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and both values, when it is another; as
    /// [`str`](Self::str) when it is not a string.
    pub fn expect_str(
        &self,
        name: &str,
        expected: &str,
    ) -> BunsenResult<()> {
        let found = self.str(name)?;
        if found == expected {
            Ok(())
        } else {
            Err(BunsenError::unsupported(format!(
                "config.json: `{}` is {found:?}, and only {expected:?} is supported",
                self.field(name)
            )))
        }
    }

    /// Requires `found`, read from `name`, to be `expected`: a field whose
    /// value the kit's model fixes.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and both values, when they differ.
    pub fn expect_eq<T: PartialEq + core::fmt::Debug>(
        &self,
        name: &str,
        found: T,
        expected: T,
    ) -> BunsenResult<()> {
        if found == expected {
            Ok(())
        } else {
            Err(BunsenError::unsupported(format!(
                "config.json: `{}` is {found:?}, and only {expected:?} is supported",
                self.field(name)
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    #[test]
    fn test_fields_and_errors() {
        let c = HfConfigJson::parse(
            r#"{"model_type": "x", "n": 3, "eps": 1e-6, "flag": true,
                "durations": [0, 1, 2], "sub": {"d": 8, "missing": null}}"#,
        )
        .unwrap();
        assert_eq!(c.usize("n").unwrap(), 3);
        assert_eq!(c.f64("eps").unwrap(), 1e-6);
        assert!(c.bool_or("flag", false).unwrap());
        assert!(c.bool_or("absent", true).unwrap());
        assert_eq!(c.usizes("durations").unwrap(), [0, 1, 2]);
        c.expect_str("model_type", "x").unwrap();

        let sub = c.section("sub").unwrap();
        assert_eq!(sub.usize("d").unwrap(), 8);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("`sub.missing`")
            .assert_err(&sub.usize("missing"));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("`model_type` is \"x\"")
            .assert_err(&c.expect_str("model_type", "y"));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("`sub.d` is 8")
            .assert_err(&sub.expect_eq("d", 8, 16));
        assert!(HfConfigJson::parse("[1]").is_err());
    }
}
