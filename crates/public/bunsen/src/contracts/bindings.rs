//! Name/value bindings.

use alloc::vec::Vec;

/// Name lookup over a list of bindings.
///
/// This trait, [`MutableStackMap`] and [`MutableStackEnvironment`] are
/// general helpers over [`StackEnvironment`]. The contract checks don't use
/// them: a check resolves names through the contract's
/// [`index`](crate::contracts::ShapeContract::index). Contract users normally
/// don't need them.
pub trait StackMap<'a, V>
where
    V: Default + Copy,
{
    /// Looks up a value associated with the given key in the stack.
    ///
    /// # Arguments
    ///
    /// * `key`: The key to look up.
    ///
    /// # Returns
    ///
    /// An `Option<V>` containing the value if found, or `None` if not found.
    #[must_use]
    fn lookup(
        &self,
        key: &'a str,
    ) -> Option<V>;

    /// Exports the values for the given keys from the local bindings.
    ///
    /// # Arguments
    ///
    /// * `keys`: An array of keys to look up in the local bindings.
    ///
    /// # Returns
    ///
    /// The values of the keys, in key order.
    ///
    /// # Panics
    ///
    /// Panics if any key is not found in the local bindings.
    #[must_use]
    fn export_key_values<const K: usize>(
        &self,
        keys: &'a [&str; K],
    ) -> [V; K] {
        let mut values: [V; K] = [Default::default(); K];
        for i in 0..K {
            let key = keys[i];
            values[i] = match self.lookup(key) {
                Some(v) => v,
                None => panic!("No value for key \"{key}\""),
            };
        }
        values
    }
}

/// Bindings: the values a caller already knows, as `(name, value)` pairs.
///
/// This is the `env` (or `bindings`) argument of every
/// [`ShapeContract`](crate::contracts::ShapeContract) check. For example,
/// `&[("window", 8)]` binds `"window"` to 8. Every name must be in the
/// contract's pattern.
pub type StackEnvironment<'a> = &'a [(&'a str, usize)];

impl<'a> StackMap<'a, usize> for StackEnvironment<'a> {
    #[inline]
    fn lookup(
        &self,
        key: &'a str,
    ) -> Option<usize> {
        for &(k, v) in self.iter() {
            if k == key {
                return Some(v);
            }
        }
        None
    }
}

/// A [`StackMap`] that can bind new names.
///
/// Provides no support for removing keys. Not used by the contract checks;
/// see [`StackMap`].
pub trait MutableStackMap<'a, V>: StackMap<'a, V>
where
    V: Default + Copy,
{
    /// Inserts a key-value pair into the stack.
    ///
    /// It is an error to bind a key which is already present;
    /// but this is only checked when `debug_assertions` are on.
    ///
    /// # Arguments
    ///
    /// * `key`: The key to insert.
    /// * `value`: The value to associate with the key.
    ///
    /// # Returns
    ///
    /// This method does not return a value,
    /// but modifies the stack by inserting the key-value pair.
    fn bind(
        &mut self,
        key: &'a str,
        value: V,
    );
}

/// A growable set of bindings: a [`StackEnvironment`] plus a `Vec` of new
/// bindings.
///
/// Not used by the contract checks; see [`StackMap`].
pub struct MutableStackEnvironment<'a> {
    /// The backing stack environment that contains the original bindings.
    pub backing: StackEnvironment<'a>,

    /// The updates made to the stack environment.
    pub updates: Vec<(&'a str, usize)>,
}

impl<'a> MutableStackEnvironment<'a> {
    /// Looks up a value associated with the given key in the updates first.
    ///
    /// This is useful when exporting values, as the newly bound values
    /// tend to be the keys being exported.
    ///
    /// # Arguments
    ///
    /// * `key`: The key to look up.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` containing the value if found in updates,
    /// or `None` if not found.
    #[inline]
    #[must_use]
    fn updates_first_lookup(
        &self,
        key: &'a str,
    ) -> Option<usize> {
        self.updates
            .as_slice()
            .lookup(key)
            .or_else(|| self.backing.lookup(key))
    }
}

impl<'a> StackMap<'a, usize> for MutableStackEnvironment<'a> {
    #[inline]
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize> {
        self.backing
            .lookup(key)
            .or_else(|| self.updates.as_slice().lookup(key))
    }

    #[inline]
    fn export_key_values<const K: usize>(
        &self,
        keys: &'a [&str; K],
    ) -> [usize; K] {
        let mut values: [usize; K] = [Default::default(); K];
        for i in 0..K {
            let key = keys[i];
            values[i] = match self.updates_first_lookup(key) {
                Some(v) => v,
                None => panic!("No value for key \"{key}\""),
            };
        }
        values
    }
}

impl<'a> MutableStackMap<'a, usize> for MutableStackEnvironment<'a> {
    #[inline]
    fn bind(
        &mut self,
        key: &'a str,
        value: usize,
    ) {
        #[cfg(debug_assertions)]
        assert!(self.lookup(key).is_none(), "double-bind: {key}");

        self.updates.push((key, value))
    }
}

impl<'a> MutableStackEnvironment<'a> {
    /// Creates a new `MutableStackEnvironment` with the given backing bindings.
    #[inline(always)]
    pub fn new(bindings: StackEnvironment<'a>) -> Self {
        MutableStackEnvironment {
            backing: bindings,
            updates: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_bindings() {
        {
            // stack bindings.
            let env: StackEnvironment = &[("a", 1), ("b", 2)];

            assert_eq!(env.lookup("a"), Some(1));
            assert_eq!(env.lookup("b"), Some(2));
            assert_eq!(env.lookup("c"), None);
        }
        {
            // static bindings.
            static ENV: StackEnvironment = &[("a", 1), ("b", 2)];

            assert_eq!(ENV.lookup("a"), Some(1));
            assert_eq!(ENV.lookup("b"), Some(2));
            assert_eq!(ENV.lookup("c"), None);

            // Exporting values
            let keys = ["b", "a"];
            assert_eq!(ENV.export_key_values(&keys), [2, 1]);
        }
    }

    #[should_panic(expected = "No value for key \"d\"")]
    #[test]
    fn test_stack_env_export_key_values_panic() {
        let env: StackEnvironment = &[("a", 1), ("b", 2)];
        let keys = ["a", "b", "d"]; // 'd' does not exist
        let _ = env.export_key_values(&keys); // This should panic
    }

    #[test]
    fn test_mutable_stack_bindings() {
        let mut env = MutableStackEnvironment::new(&[("a", 1), ("b", 2)]);

        assert_eq!(env.lookup("a"), Some(1));
        assert_eq!(env.lookup("b"), Some(2));
        assert_eq!(env.lookup("c"), None);

        env.bind("c", 3);
        assert_eq!(env.lookup("c"), Some(3));

        // Exporting values
        let keys = ["a", "b", "c"];
        let values = env.export_key_values(&keys);
        assert_eq!(values, [1, 2, 3]);
    }

    #[should_panic(expected = "No value for key \"d\"")]
    #[test]
    fn test_muttable_stack_env_export_key_values_panic() {
        let env = MutableStackEnvironment::new(&[("a", 1), ("b", 2)]);
        let keys = ["a", "b", "d"]; // 'd' does not exist
        let _ = env.export_key_values(&keys); // This should panic
    }

    #[should_panic(expected = "double-bind: a")]
    #[test]
    #[cfg(debug_assertions)]
    fn test_double_bind_panic() {
        let mut env = MutableStackEnvironment::new(&[("a", 1), ("b", 2)]);
        env.bind("a", 3);
    }
}
