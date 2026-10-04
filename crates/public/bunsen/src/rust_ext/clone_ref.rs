/// Either a borrowed `&T` or an owned `T`, read through [`AsRef<T>`].
///
/// Like [`Cow`](std::borrow::Cow) without the `ToOwned` bound: a function
/// returns the borrow when it has the value already, and an owned value when
/// it had to build one. The audit probe's data argument uses it, returning a
/// borrowed `TensorData` as is and a converted or read-back one owned.
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
