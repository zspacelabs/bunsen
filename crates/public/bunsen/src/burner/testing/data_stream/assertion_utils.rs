use burn::prelude::TensorData;

use crate::errors::BunsenResult;

/// [`TensorData::assert_eq`] api, returning a [`BunsenResult`].
pub fn tensor_data_assert_eq(
    actual: &TensorData,
    expected: &TensorData,
    strict: bool,
) -> BunsenResult<()> {
    // TODO: Result-generating version of this.
    // * Expand `burn` api.
    // * Clone `burn` api, generate Results.
    // * `panic::catch_unwind` version of this.
    actual.assert_eq(&expected, strict);

    Ok(())
}
