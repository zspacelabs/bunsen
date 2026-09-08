use crate::{
    burner::testing::data_stream::{
        OnStreamEvent,
        StreamEventFrame,
        StreamEventMeta,
    },
    prelude::*,
};

/// [`TensorDataTestStream`] recorder for [`Vec<StreamEventFrame>`].
#[derive(Default, Debug, Clone)]
pub struct TensorDataVecStreamRecorder {
    vec: Vec<StreamEventFrame>,
}

impl TensorDataVecStreamRecorder {
    ///  [`TensorDataTestStream`] verifier for [`Vec<StreamEventFrame>`].
    pub fn verifier(&self) -> TensorDataVecStreamVerifier {
        TensorDataVecStreamVerifier::new(self.vec.clone())
    }
}

impl From<TensorDataVecStreamRecorder> for TensorDataVecStreamVerifier {
    fn from(stream: TensorDataVecStreamRecorder) -> Self {
        Self::new(stream.vec)
    }
}

impl OnStreamEvent for TensorDataVecStreamRecorder {
    fn on_stream_event(
        &mut self,
        event: &impl StreamEventMeta,
    ) -> BunsenResult<()> {
        self.vec.push(event.to_owned());
        Ok(())
    }
}

///  [`TensorDataTestStream`] verifier for [`Vec<StreamEventFrame>`].
pub struct TensorDataVecStreamVerifier {
    vec: Vec<StreamEventFrame>,

    next: Option<usize>,
}

impl TensorDataVecStreamVerifier {
    /// Create a new empty verifier.
    pub fn new(vec: Vec<StreamEventFrame>) -> Self {
        Self { vec, next: Some(0) }
    }

    /// Pop the next event from the stream.
    fn read(&mut self) -> BunsenResult<&StreamEventFrame> {
        match self.next {
            Some(i) => {
                let event = &self.vec[i];
                if i + 1 < self.vec.len() {
                    self.next = Some(i + 1);
                } else {
                    self.next = None;
                }
                Ok(event)
            }
            None => Err(BunsenError::Invalid("No more events in stream".to_string())),
        }
    }
}

impl OnStreamEvent for TensorDataVecStreamVerifier {
    fn on_stream_event(
        &mut self,
        event: &impl StreamEventMeta,
    ) -> BunsenResult<()> {
        event.try_match(self.read()?)
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;

    use super::*;
    use crate::{
        burner::testing::data_stream::*,
        support::testing::{
            CpuBackend,
            PerformanceBackend,
        },
    };

    #[test]
    #[serial_test::serial]
    fn test_stream() -> BunsenResult<()> {
        fn example<B: Backend>(stream: &mut impl TensorDataTestStream) -> BunsenResult<()> {
            let device = Default::default();

            let a = TensorData::from([1, 2]);
            stream.assert_eq("a", &a, false)?;

            let b: Tensor<B, 1, Int> = Tensor::arange(0..4, &device);
            stream.assert_tensor_eq_as::<B, _, _, f32>("b", &b, false)?;

            Ok(())
        }

        let mut stream = TensorDataVecStreamRecorder::default();
        example::<PerformanceBackend>(&mut stream)?;

        let mut verifier = stream.verifier();
        example::<CpuBackend>(&mut verifier)?;

        Ok(())
    }
}
