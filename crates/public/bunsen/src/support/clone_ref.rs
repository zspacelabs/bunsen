/// Utility for holding either a reference or a clone.
pub enum CloneRef<'a, T> {
    /// A reference to a value.
    Ref(&'a T),

    /// A clone of a value.
    Clone(T),
}

impl<T> AsRef<T> for CloneRef<'_, T> {
    fn as_ref(&self) -> &T {
        match self {
            CloneRef::Ref(r) => r,
            CloneRef::Clone(c) => c,
        }
    }
}
