//! Testing utilities.
use std::fmt::Debug;

use num_traits::float::Float;

cfg_select! {
    feature = "cuda" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Cuda;
    }
    feature = "metal" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Metal;
    }
    feature = "wgpu" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Wgpu;
    }
    _ => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Flex;
    }
}
/// Selected burn backend for fast-setup tests.
pub type CpuBackend = ::burn::backend::Flex;

/// Asserts that two vectors of floating-point numbers are close to each other
/// within a given tolerance.
pub fn assert_close_to_vec<T>(
    actual: &[T],
    expected: &[T],
    tolerance: T,
) where
    T: Float + std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy + Debug,
{
    let mut pass = actual.len() == expected.len();
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if !pass {
            break;
        }
        if (a - e).abs() > tolerance {
            pass = false;
            break;
        }
    }
    if !pass {
        panic!("Expected (+/- {tolerance:?}):\n{expected:?}\nActual:\n{actual:?}");
    }
}

#[cfg(test)]
mod tests {
    use crate::support::testing::assert_close_to_vec;

    #[test]
    fn test_assert_close_to_vec() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);

        let actual = vec![1.0, 2.0, 3.1];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.2);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_bad_values() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.5];
        assert_close_to_vec(&actual, &expected, 0.01);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_different_lengths() {
        let actual = vec![1.0, 2.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);
    }
}

/// Audio fixture loading, for tests that drive models from real waveforms.
#[cfg(test)]
pub mod audio {
    use std::path::Path;

    use hound::{
        SampleFormat,
        WavReader,
        WavSpec,
    };

    use crate::errors::{
        BunsenError,
        BunsenResult,
    };

    /// Loads a mono audio file as `[-1, 1]` samples.
    ///
    /// Integer formats are scaled by `1 << (bits - 1)`, so a 16-bit fixture
    /// round-trips exactly through a `* 32768` rescale.
    ///
    /// # Arguments
    /// * `filename` - path to an audio file.
    /// * `sample_rate` - the sample rate the file is required to be at.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the file is not single-channel or is not at
    /// `sample_rate`; [`BunsenError::External`] if it cannot be read.
    pub fn load_audio_mono_sr<P: AsRef<Path>>(
        filename: P,
        sample_rate: usize,
    ) -> BunsenResult<(WavSpec, Vec<f32>)> {
        let filename = filename.as_ref();

        let mut reader = WavReader::open(filename).map_err(BunsenError::external)?;
        let spec = reader.spec();

        if spec.channels != 1 {
            return Err(BunsenError::Invalid(
                "The audio must be single-channel".to_string(),
            ));
        }
        if spec.sample_rate as usize != sample_rate {
            return Err(BunsenError::Invalid(format!(
                "Expected sample_rate = {}, but found {}",
                sample_rate, spec.sample_rate
            )));
        }

        let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .map(|s| s.unwrap())
                .collect::<Vec<f32>>(),
            (SampleFormat::Int, bits) => {
                let scale = (1i64 << (bits - 1)) as f32;
                reader
                    .samples::<i32>()
                    .collect::<Result<Vec<i32>, _>>()
                    .map_err(BunsenError::external)?
                    .into_iter()
                    .map(|s| s as f32 / scale)
                    .collect()
            }
            _ => unreachable!("hound rejects other formats at open"),
        };

        Ok((spec, samples))
    }
}
