use burn::prelude::TensorData;

use crate::{
    burner::testing::data_stream::{
        StreamEvent,
        TensorDataTestStream,
        TensorDataTestStreamVerifier,
    },
    prelude::*,
};

/// [`TensorDataTestStream`] writer.
#[derive(Default, Debug, Clone)]
pub struct TensorDataVecStreamRecorder {
    vec: Vec<StreamEvent>,
}

impl TensorDataVecStreamRecorder {
    ///  [`TensorDataTestStream`] verifier for [`Vec<StreamEvent>`].
    pub fn verifier(&self) -> TensorDataVecStreamVerifier {
        TensorDataVecStreamVerifier::new(self.vec.clone())
    }
}

impl From<TensorDataVecStreamRecorder> for TensorDataVecStreamVerifier {
    fn from(stream: TensorDataVecStreamRecorder) -> Self {
        Self::new(stream.vec)
    }
}

impl TensorDataTestStream for TensorDataVecStreamRecorder {
    fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        let label = label.to_string();
        self.vec.push(StreamEvent::AssertEq {
            label,
            data: data.clone(),
            strict,
        });

        Ok(())
    }
}

///  [`TensorDataTestStream`] verifier for [`Vec<StreamEvent>`].
pub struct TensorDataVecStreamVerifier {
    vec: Vec<StreamEvent>,

    next: Option<usize>,
}

impl TensorDataVecStreamVerifier {
    /// Create a new empty verifier.
    pub fn new(vec: Vec<StreamEvent>) -> Self {
        Self { vec, next: Some(0) }
    }
}

impl TensorDataTestStreamVerifier for TensorDataVecStreamVerifier {
    /// Pop the next event from the stream.
    fn pop(&mut self) -> BunsenResult<&StreamEvent> {
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
        fn example<B: Backend>(stream: &mut dyn TensorDataTestStream) -> BunsenResult<()> {
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
