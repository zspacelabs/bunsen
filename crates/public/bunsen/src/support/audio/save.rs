//! Writing audio files: [`save_wav_mono`].

use std::path::Path;

use hound::{
    SampleFormat,
    WavSpec,
    WavWriter,
};

use super::load::hound_error;
use crate::errors::BunsenResult;

/// Writes mono `samples`, in `[-1, 1]` (clipped there), to a 16-bit PCM WAV
/// file at `sample_rate`.
///
/// # Errors
/// Naming the file: as [`sys_at`](crate::errors::sys_at) sorts an io failure;
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// any other failure to write the WAV.
pub fn save_wav_mono(
    filename: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> BunsenResult<()> {
    let filename = filename.as_ref();
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(filename, spec).map_err(hound_error("create", filename))?;
    for &sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        writer
            .write_sample(pcm)
            .map_err(hound_error("write", filename))?;
    }
    writer.finalize().map_err(hound_error("write", filename))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{
        audio::load_audio_mono_sr,
        testing::assert_scalar_close,
    };

    /// Saved samples load back within the 16-bit step, clipped to `[-1, 1]`.
    #[test]
    fn test_save_wav_mono() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("saved.wav");
        let samples = [0.0, 0.5, -0.25, 1.5, -2.0];
        save_wav_mono(&path, &samples, 8000).unwrap();
        let loaded = load_audio_mono_sr(&path, 8000).unwrap();
        let expected = [0.0, 0.5, -0.25, 1.0, -1.0];
        assert_eq!(loaded.len(), expected.len());
        for (got, want) in loaded.iter().zip(expected) {
            assert_scalar_close(*got, want, 1e-4);
        }
    }
}
