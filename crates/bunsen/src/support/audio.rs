//! # Audio Support

use std::path::Path;

use hound::{
    SampleFormat,
    WavReader,
};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{
        CODEC_TYPE_NULL,
        DecoderOptions,
    },
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Loads a mono audio file, as `f32` samples in `[-1, 1]`.
///
/// `.wav` is read with `hound`; every other extension is handed to
/// `symphonia`, which covers the compressed formats (mp3).
///
/// The file must *already* be mono at `sample_rate`: this decodes, it does not
/// resample or downmix. A file that disagrees is an error rather than a silent
/// conversion, because a resample is a signal-processing decision the caller
/// should make deliberately — the mel front end's output depends on it.
///
/// # Arguments
/// * `filename` - path to an audio file.
/// * `sample_rate` - the sample rate the file is required to have.
pub fn load_audio_mono_sr(
    filename: impl AsRef<Path>,
    sample_rate: usize,
) -> BunsenResult<Vec<f32>> {
    let filename = filename.as_ref();

    let ext = filename
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match ext.as_str() {
        "wav" | "wave" => load_wav_mono_sr(filename, sample_rate),
        _ => load_compressed_mono_sr(filename, &ext, sample_rate),
    }
}

/// Rejects anything that is not single-channel audio at `expected`.
fn check_mono_sr(
    channels: usize,
    rate: usize,
    expected: usize,
) -> BunsenResult<()> {
    if channels != 1 {
        return Err(BunsenError::Invalid(
            "The audio must be single-channel".to_string(),
        ));
    }
    if rate != expected {
        return Err(BunsenError::Invalid(format!(
            "Expected sample_rate = {expected}, but found {rate}"
        )));
    }
    Ok(())
}

/// Reads a WAV file through `hound`.
fn load_wav_mono_sr(
    filename: &Path,
    sample_rate: usize,
) -> BunsenResult<Vec<f32>> {
    let mut reader = WavReader::open(filename).map_err(BunsenError::external)?;
    let spec = reader.spec();

    check_mono_sr(
        spec.channels as usize,
        spec.sample_rate as usize,
        sample_rate,
    )?;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(BunsenError::external)?,
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

    Ok(samples)
}

/// Decodes a compressed file through `symphonia`.
///
/// Gapless playback is enabled, so an mp3's encoder delay and padding are
/// trimmed rather than returned as leading and trailing silence — which would
/// otherwise shift every frame of a spectrogram computed from it.
fn load_compressed_mono_sr(
    filename: &Path,
    ext: &str,
    sample_rate: usize,
) -> BunsenResult<Vec<f32>> {
    let file = std::fs::File::open(filename).map_err(BunsenError::external)?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if !ext.is_empty() {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };

    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &format_opts, &MetadataOptions::default())
        .map_err(BunsenError::external)?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| BunsenError::Invalid("The file has no decodable audio track".to_string()))?;

    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(BunsenError::external)?;

    let mut samples: Vec<f32> = Vec::new();
    let mut buffer: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // Symphonia signals a clean end of stream as an EOF io error.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(BunsenError::external(e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).map_err(BunsenError::external)?;
        let spec = *decoded.spec();

        check_mono_sr(spec.channels.count(), spec.rate as usize, sample_rate)?;

        // The buffer is reused across packets, but `copy_interleaved_ref`
        // panics rather than growing, so a packet wider than the one that
        // sized it must force a reallocation.
        let needed = decoded.capacity() as u64;
        let buf = match buffer.take() {
            Some(buf) if buf.capacity() as u64 >= needed => buf,
            _ => SampleBuffer::new(needed, spec),
        };
        let buf = buffer.insert(buf);

        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short synthetic tone, in both formats. See `testdata/audio/README.md`.
    const WAV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/audio/tone.wav");
    const MP3: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/audio/tone.mp3");

    /// 0.5 s at 16 kHz.
    const EXPECTED: usize = 8_000;

    /// The rate a WAV declares is checked, not assumed.
    #[test]
    fn test_load_audio_mono_sr_wav() {
        let samples = load_audio_mono_sr(WAV, 16000).unwrap();
        assert_eq!(samples.len(), EXPECTED);

        assert!(
            load_audio_mono_sr(WAV, 8000).is_err(),
            "a sample-rate mismatch must be an error, not a silent resample",
        );
    }

    /// mp3 goes through a different decoder than WAV, so it gets its own case.
    #[test]
    fn test_load_audio_mono_sr_mp3() {
        let samples = load_audio_mono_sr(MP3, 16000).unwrap();

        // Gapless trims the encoder delay, but mp3 still frames in blocks of
        // 576 samples, so the tail is padded out to a whole frame.
        let slack = samples.len().abs_diff(EXPECTED);
        assert!(
            slack <= 1152,
            "decoded {} samples, expected about {EXPECTED} (off by {slack})",
            samples.len(),
        );

        assert!(
            samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
            "samples must be finite and within [-1, 1]",
        );
        assert!(
            samples.iter().any(|&s| s.abs() > 0.01),
            "decoded to silence",
        );

        assert!(
            load_audio_mono_sr(MP3, 44100).is_err(),
            "a sample-rate mismatch must be an error, not a silent resample",
        );
    }

    /// The two formats must agree on the signal they carry.
    ///
    /// mp3 is lossy, so this is a coarse check — but a decoder that dropped a
    /// channel, mis-scaled, or returned garbage would fail it.
    #[test]
    fn test_wav_and_mp3_carry_the_same_tone() {
        let wav = load_audio_mono_sr(WAV, 16000).unwrap();
        let mp3 = load_audio_mono_sr(MP3, 16000).unwrap();

        let rms =
            |x: &[f32]| (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64).sqrt();

        let (a, b) = (rms(&wav[..EXPECTED / 2]), rms(&mp3[..EXPECTED / 2]));
        assert!(
            (a - b).abs() / a < 0.05,
            "rms differs by more than 5%: wav {a:.4} vs mp3 {b:.4}",
        );
    }
}
