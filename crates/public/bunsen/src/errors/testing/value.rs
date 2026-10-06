//! # Value matchers
//!
//! [`Matcher`]s on any `T: Debug + PartialEq`: [`eq`] and [`one_of`]. A
//! mismatch prints the actual value's `Debug`.

use alloc::vec::Vec;
use core::fmt::{
    Debug,
    Write,
};

use crate::errors::testing::{
    Description,
    Matcher,
};

/// Matches a value equal to its own. See [`eq`].
#[derive(Clone, Debug)]
pub struct ValueEq<T> {
    expected: T,
}

/// Matches a value equal to `expected`.
pub fn eq<T>(expected: T) -> ValueEq<T>
where
    T: Debug + PartialEq + Send + Sync,
{
    ValueEq { expected }
}

impl<T> Matcher<T> for ValueEq<T>
where
    T: Debug + PartialEq + Send + Sync,
{
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        *actual == self.expected
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        let _ = write!(d, "{:?}", self.expected);
    }

    fn describe_mismatch(
        &self,
        actual: &T,
        d: &mut Description,
    ) {
        let _ = write!(d, "was {actual:?}");
    }
}

/// Matches a value equal to one of its own. See [`one_of`].
#[derive(Clone, Debug)]
pub struct OneOf<T> {
    expected: Vec<T>,
}

/// Matches a value equal to one of `expected`.
pub fn one_of<T>(expected: impl IntoIterator<Item = T>) -> OneOf<T>
where
    T: Debug + PartialEq + Send + Sync,
{
    OneOf {
        expected: expected.into_iter().collect(),
    }
}

impl<T> Matcher<T> for OneOf<T>
where
    T: Debug + PartialEq + Send + Sync,
{
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        self.expected.contains(actual)
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        let _ = write!(d, "one of {:?}", self.expected);
    }

    fn describe_mismatch(
        &self,
        actual: &T,
        d: &mut Description,
    ) {
        let _ = write!(d, "was {actual:?}");
    }
}
