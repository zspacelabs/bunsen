use std::{
    any::{
        Any,
        TypeId,
    },
    fmt::Debug,
};

use bunsen::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    WithOkOrPanic,
};
use serde::{
    Serialize,
    de::DeserializeOwned,
};

/// A wrapper type that can hold either a JSON value or a boxed value of any
/// type.
pub enum FirehoseValue {
    /// Holds a JSON value.
    Value(serde_json::Value),

    /// Holds a boxed value of any type that implements `Any` and `Send`.
    Boxed(Box<dyn Any + 'static + Send>),
}

/// Checks if a type is boxable in a `ValueBox::Boxed`.
///
/// # Type Parameters
///
/// - `Outer`: The outer display type that will contain the boxed value.
/// - `Inner`: The inner type that is being checked.
///
/// # Returns
///
/// A `BunsenResult<()>` indicating whether the type can be boxed: an
/// [`Illegal`](BunsenErrorKind::Illegal) error for a type that must be
/// serialized instead.
pub fn try_boxable<T: 'static>() -> BunsenResult<()> {
    let type_id = TypeId::of::<T>();
    let forbidden_types = [
        TypeId::of::<serde_json::Value>(),
        TypeId::of::<&'static str>(),
        TypeId::of::<&str>(),
        TypeId::of::<String>(),
        TypeId::of::<i32>(),
        TypeId::of::<i64>(),
        TypeId::of::<f32>(),
        TypeId::of::<f64>(),
        TypeId::of::<usize>(),
        TypeId::of::<isize>(),
        TypeId::of::<bool>(),
    ];
    if forbidden_types.contains(&type_id) {
        let type_name = std::any::type_name::<T>();
        return Err(BunsenError::illegal(format!(
            "Type `{type_name}` must be serialized, not boxed; \
             use a ValueBox::Value instead."
        )));
    }

    Ok(())
}

/// Checks if a type is boxable in a `ValueBox::Boxed`.
///
/// `panic!` wrapper for `try_boxable`.
pub fn check_boxable<T: 'static>() {
    try_boxable::<T>().ok_or_panic()
}

impl Debug for FirehoseValue {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            FirehoseValue::Value(value) => write!(f, "{{{value}}}"),
            FirehoseValue::Boxed(boxed) => write!(f, "[{boxed:?}]"),
        }
    }
}

impl FirehoseValue {
    /// Creates a new `ValueBox::Value` by serializing an object.
    pub fn serialized<T>(obj: T) -> BunsenResult<Self>
    where
        T: 'static + Serialize,
    {
        serde_json::to_value(obj)
            .map_err(|e| {
                BunsenError::from_cause(BunsenErrorKind::Illegal, e)
                    .context(format!("serializing a `{}`", std::any::type_name::<T>()))
            })
            .map(FirehoseValue::Value)
    }

    /// Creates a new `ValueBox::Value` from a `serde_json::Value`.
    pub fn from_json_value(value: serde_json::Value) -> Self {
        FirehoseValue::Value(value)
    }

    /// Creates a new `ValueBox::Boxed` by boxing an object that implements
    /// `Any` and `Send`.
    pub fn boxing<T>(obj: T) -> Self
    where
        T: Any + 'static + Send,
    {
        Self::from_box(Box::new(obj))
    }

    /// Creates a new `ValueBox::Boxed` from a boxed object that implements
    /// `Any` and `Send`.
    pub fn from_box<T>(boxed: Box<T>) -> Self
    where
        T: Any + 'static + Send,
    {
        check_boxable::<T>();
        FirehoseValue::Boxed(boxed)
    }

    /// Returns true if the `ValueBox` contains a JSON value.
    pub fn is_value(&self) -> bool {
        matches!(self, FirehoseValue::Value(_))
    }

    /// Returns true if the `ValueBox` contains a boxed value.
    pub fn is_boxed(&self) -> bool {
        matches!(self, FirehoseValue::Boxed(_))
    }

    /// Unwraps the `ValueBox` and returns a reference to the contained JSON
    /// value.
    pub fn unwrap_value(&self) -> &serde_json::Value {
        if let FirehoseValue::Value(value) = self {
            value
        } else {
            panic!("ValueBox::unwrap_value() called on {self:?}");
        }
    }

    /// Parses the value as the target `DeserializeOwned` type.
    ///
    /// # Type Parameters
    ///
    /// `T`: The type to deserialize the JSON value into. It must implement
    /// `DeserializeOwned`.
    ///
    /// # Panics
    ///
    /// If the `ValueBox` does not contain a JSON value.
    ///
    /// # Returns
    ///
    /// A `BunsenResult<T>` where `T` is the deserialized type;
    /// if deserialization fails, an [`Illegal`](BunsenErrorKind::Illegal)
    /// error whose cause is the `serde_json::Error`.
    pub fn parse_as<T>(&self) -> BunsenResult<T>
    where
        T: DeserializeOwned + 'static,
    {
        let value = self.unwrap_value();
        serde_json::from_value(value.clone()).map_err(|e| {
            BunsenError::from_cause(BunsenErrorKind::Illegal, e).context(format!(
                "parsing {value} as a `{}`",
                std::any::type_name::<T>()
            ))
        })
    }

    /// Parses the value as the target `DeserializeOwned` type.
    ///
    /// # Type Parameters
    ///
    /// `T`: The type to deserialize the JSON value into. It must implement
    /// `DeserializeOwned`.
    ///
    /// # Panics
    ///
    /// If the `ValueBox` does not contain a JSON value;
    /// or the value fails to deserialize into the specified type.
    ///
    /// # Returns
    ///
    /// A `T`.
    pub fn expect_parse_as<T>(&self) -> T
    where
        T: DeserializeOwned + 'static,
    {
        self.parse_as::<T>().ok_or_panic()
    }

    /// Unwraps the `ValueBox` and returns a reference to the contained boxed
    /// value.
    ///
    /// # Type Parameters
    ///
    /// `T`: The type to downcast the boxed value to.
    ///
    /// # Panics
    ///
    /// If the `ValueBox` does not contain a boxed value.
    ///
    /// # Returns
    ///
    /// A `BunsenResult<&T>` where `T` is the down-cast type;
    /// if the downcast fails, an [`Illegal`](BunsenErrorKind::Illegal) error.
    pub fn as_ref<T>(&self) -> BunsenResult<&T>
    where
        T: 'static,
    {
        if let FirehoseValue::Boxed(boxed) = self {
            boxed.downcast_ref::<T>().ok_or_else(|| {
                BunsenError::illegal(format!(
                    "Failed to downcast boxed value to `{}`",
                    std::any::type_name::<T>()
                ))
            })
        } else {
            panic!(
                "ValueBox::unwrap_boxed::<{}>() called on {self:?}",
                std::any::type_name::<T>()
            );
        }
    }

    /// Unwraps the `ValueBox` and returns a reference to the contained boxed
    /// value.
    ///
    /// # Type Parameters
    ///
    /// `T`: The type to downcast the boxed value to.
    ///
    /// # Panics
    ///
    /// If the `ValueBox` does not contain a boxed value or if the downcast
    /// fails.
    ///
    /// # Returns
    ///
    /// A `&T`.
    pub fn expect_as_ref<T>(&self) -> &T
    where
        T: 'static,
    {
        self.as_ref::<T>().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::testing::ErrorMatcher;

    use super::*;

    #[test]
    fn test_try_boxable() {
        let not_boxable = ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("must be serialized, not boxed");
        not_boxable.assert_err(&try_boxable::<serde_json::Value>());
        not_boxable.assert_err(&try_boxable::<&'static str>());
        not_boxable.assert_err(&try_boxable::<&str>());
        not_boxable.assert_err(&try_boxable::<String>());
        not_boxable.assert_err(&try_boxable::<i32>());
        not_boxable.assert_err(&try_boxable::<i64>());
        not_boxable.assert_err(&try_boxable::<f32>());
        not_boxable.assert_err(&try_boxable::<f64>());
        not_boxable.assert_err(&try_boxable::<usize>());
        not_boxable.assert_err(&try_boxable::<isize>());
        not_boxable.assert_err(&try_boxable::<bool>());

        assert!(try_boxable::<MyStruct>().is_ok());
    }

    #[should_panic(
        expected = "Type `i32` must be serialized, not boxed; use a ValueBox::Value instead."
    )]
    #[test]
    fn test_check_boxable() {
        check_boxable::<i32>();
    }

    #[test]
    fn test_from_json_value() -> BunsenResult<()> {
        let json_value = serde_json::json!(["foo", "bar"]);
        let vb = FirehoseValue::from_json_value(json_value.clone());
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.parse_as::<Vec<String>>()?, vec!["foo", "bar"]);

        assert_eq!(format!("{vb:?}"), "{[\"foo\",\"bar\"]}");

        Ok(())
    }

    #[test]
    fn test_string_value() -> BunsenResult<()> {
        let vb = FirehoseValue::serialized("abc")?;
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.parse_as::<String>()?, "abc");

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .frame_contains("parsing \"abc\" as a `i32`")
            .message_eq("invalid type: string \"abc\", expected i32")
            .has_cause::<serde_json::Error>()
            .assert_err(&vb.parse_as::<i32>());

        assert_eq!(vb.unwrap_value().as_str().unwrap(), "abc");

        Ok(())
    }

    #[test]
    #[should_panic(expected = r"called on {")]
    fn test_unwrap_value_as_boxed() {
        let vb = FirehoseValue::serialized("abc").unwrap();
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        vb.as_ref::<MyStruct>().unwrap();
    }

    #[test]
    #[should_panic(expected = r"unwrap_value() called")]
    fn test_unwrap_boxed_as_value() {
        let my_struct = MyStruct {
            field1: "test".to_string(),
            field2: 123,
        };

        let vb = FirehoseValue::boxing(my_struct.clone());
        assert!(!vb.is_value());
        assert!(vb.is_boxed());

        vb.parse_as::<String>().unwrap();
    }

    #[test]
    fn test_int_value() -> BunsenResult<()> {
        let vb = FirehoseValue::serialized(42_i32)?;
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.parse_as::<i32>()?, 42_i32);
        assert_eq!(vb.parse_as::<i64>()?, 42_i64);
        assert_eq!(vb.parse_as::<usize>()?, 42_usize);

        assert_eq!(vb.unwrap_value().as_i64().unwrap(), 42_i64);

        Ok(())
    }

    #[test]
    fn test_int_array() -> BunsenResult<()> {
        let vb = FirehoseValue::serialized(vec![42_i32, 0_i32])?;
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.parse_as::<Vec<i32>>()?, vec![42_i32, 0_i32]);
        assert_eq!(vb.parse_as::<Vec<i64>>()?, vec![42_i64, 0_i64]);
        assert_eq!(vb.parse_as::<Vec<usize>>()?, vec![42_usize, 0_usize]);

        Ok(())
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
    pub struct MyStruct {
        pub field1: String,
        pub field2: i32,
    }

    #[test]
    fn test_boxed_value() -> BunsenResult<()> {
        let my_struct = MyStruct {
            field1: "test".to_string(),
            field2: 123,
        };

        let vb = FirehoseValue::boxing(my_struct.clone());
        assert!(!vb.is_value());
        assert!(vb.is_boxed());

        assert_eq!(vb.as_ref::<MyStruct>()?, &my_struct);

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_eq(
                "Failed to downcast boxed value to `alloc::vec::Vec<alloc::string::String>`",
            )
            .assert_err(&vb.as_ref::<Vec<String>>());

        Ok(())
    }
}
