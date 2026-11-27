//! `testdata!`: a path under the calling crate's `testdata/`.

/// A path under the calling crate's `testdata/` directory, as a
/// [`PathBuf`](std::path::PathBuf): `testdata!("swin/window_attn.safetensors")`,
/// or `testdata!()` for the directory itself.
///
/// A macro, not a function, so `CARGO_MANIFEST_DIR` is the crate that calls
/// it: a test of a crate built on bunsen finds its own fixtures.
///
/// ```
/// use bunsen::support::testing::testdata;
///
/// let path = testdata!("swin/window_attn.safetensors");
/// assert!(path.ends_with("testdata/swin/window_attn.safetensors"));
/// assert!(testdata!().ends_with("testdata"));
/// ```
pub use crate::__testdata as testdata;

#[doc(hidden)]
#[macro_export]
macro_rules! __testdata {
    () => {
        ::std::path::Path::new(::core::env!("CARGO_MANIFEST_DIR")).join("testdata")
    };
    ($path:expr $(,)?) => {
        ::std::path::Path::new(::core::env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join($path)
    };
}
