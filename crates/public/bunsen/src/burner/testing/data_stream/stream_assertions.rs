use std::panic::Location;

use burn::{
    Tensor,
    prelude::{
        Backend,
        TensorData,
    },
    tensor::{
        BasicOps,
        Element,
    },
};

use crate::{
    burner::testing::data_stream::{
        EventMeta,
        OnStreamEvent,
        StreamEventFrameMeta,
        StreamEventFrameStub,
        StreamEventParams,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    prelude::TensorElemOpExt,
};

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
    actual.assert_eq(expected, strict);

    Ok(())
}

/// Compare two stream events.
pub fn try_match_stream_events<A, B>(
    actual: &A,
    expected: &B,
) -> BunsenResult<()>
where
    A: StreamEventFrameMeta + ?Sized,
    B: StreamEventFrameMeta + ?Sized,
{
    expected.meta().compare(actual.meta())?;

    if expected.params() != actual.params() {
        return Err(BunsenError::InvalidArgument {
            msg: format!(
                "StreamEvent params {:?} != expected {:?}",
                expected.params(),
                actual.params()
            ),
        });
    }
    if expected.data().len() != actual.data().len() {
        return Err(BunsenError::InvalidArgument {
            msg: format!(
                "StreamEvent data count ({:?}) != expected ({:?})",
                expected.data().len(),
                actual.data().len()
            ),
        });
    }

    if !actual
        .data()
        .iter()
        .zip(expected.data().iter())
        .all(|(a, e)| a.shape == e.shape)
    {
        return Err(BunsenError::InvalidArgument {
            msg: format!(
                "event data shapes do not match:\nactual: {:?}\nexpect: {:?}",
                actual
                    .data()
                    .iter()
                    .map(|d| d.shape.clone())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|d| d.shape.clone())
                    .collect::<Vec<_>>()
            ),
        });
    }

    match expected.params() {
        StreamEventParams::AssertEq { strict, .. } => {
            assert_eq!(actual.data().len(), 1);
            tensor_data_assert_eq(&actual.data()[0], &expected.data()[0], *strict)
        }
    }
}

impl<T: OnStreamEvent> TensorDataTestStream for T {}

/// `TensorData` Test Stream.
pub trait TensorDataTestStream: OnStreamEvent {
    /// Handle a stream event.
    ///
    /// # Arguments
    /// * `params` - the [`StreamEventParams`] to handle.
    /// * `data` - the [`TensorData`] to compare.
    ///
    /// # Panics and/or Err Returns
    /// If the event does not match the expected event under verification.
    fn handle_event(
        &mut self,
        meta: &EventMeta,
        params: &StreamEventParams,
        data: &[TensorData],
        _location: &Location,
    ) -> BunsenResult<()> {
        let event = StreamEventFrameStub { meta, params, data };
        <Self as OnStreamEvent>::on_stream_event(self, &event)
    }
}

impl<T: ?Sized + TensorDataTestStream> TensorDataTestStreamLocExt for T {}

/// [`Location`]-aware base extension.
pub trait TensorDataTestStreamLocExt: TensorDataTestStream {
    /// Location forwarding impl of [`assert_eq`](`Self::assert_eq`).
    fn loc_assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
        location: &Location,
    ) -> BunsenResult<()> {
        self.handle_event(
            &EventMeta::new(label.to_string(), None),
            &StreamEventParams::AssertEq { strict },
            std::slice::from_ref(data),
            location,
        )
    }
}

impl<T: ?Sized + TensorDataTestStream> TensorDataTestStreamExt for T {}

/// `TensorData` Test Stream Extension
pub trait TensorDataTestStreamExt: TensorDataTestStream + TensorDataTestStreamLocExt {
    /// [`TensorData`] equality; run over
    /// [`handle_event`](`TensorDataTestStream::handle_event`).
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        self.loc_assert_eq(label, data, strict, Location::caller())
    }

    /// [`Tensor`] equality; run over
    /// [`assert_eq`](`TensorDataTestStreamExt::assert_eq`).
    ///
    /// Data is used as `tensor.to_data_as::<E>()`.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    fn assert_tensor_eq_as<B, const R: usize, K, E>(
        &mut self,
        label: &str,
        tensor: &Tensor<B, R, K>,
        strict: bool,
    ) -> BunsenResult<()>
    where
        E: Element,
        B: Backend,
        K: BasicOps<B>,
    {
        let data = tensor.to_data_as::<E>();
        self.loc_assert_eq(label, &data, strict, Location::caller())
    }
}
