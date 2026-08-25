//! # Audio Support

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

/// Loads a mono audio file.
///
/// # Arguments
/// * `filename` - path to an audio file.
/// * `sample_rate` - sample rate of the audio file.
pub fn load_audio_mono_sr(
    filename: impl AsRef<Path>,
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

    let spec = reader.spec();
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_load_audio_mono_sr() {
        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.wav");
        let sample_rate = 16000;

        let (spec, samples) = super::load_audio_mono_sr(wav_path, sample_rate).unwrap();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, sample_rate as u32);
        assert_eq!(samples.len(), 960000);
    }
}
