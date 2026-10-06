//! # Audio Support

use std::path::Path;

use hound::{
    SampleFormat,
    WavReader,
};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    formats::{
        FormatOptions,
        TrackType,
        probe::Hint,
    },
    io::MediaSourceStream,
    meta::MetadataOptions,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
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
///
/// # Errors
/// Every error names the file.
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a missing or
///   forbidden file; [`Sys`](crate::errors::BunsenErrorKind::Sys) (or another
///   kind [`sys_at`] sorts it to) for any other io failure;
/// - [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for a
///   file that does not decode, has no audio track, is not mono, or is not at
///   `sample_rate`; the decoder's error, if any, is the cause;
/// - [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
///   container, codec or WAV format the decoders do not support.
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
    filename: &Path,
    channels: usize,
    rate: usize,
    expected: usize,
) -> BunsenResult<()> {
    if channels != 1 {
        return Err(BunsenError::invalid_resource(format!(
            "{}: the audio has {channels} channels; it must be single-channel",
            filename.display()
        )));
    }
    if rate != expected {
        return Err(BunsenError::invalid_resource(format!(
            "{}: the sample rate is {rate}; it must be {expected}",
            filename.display()
        )));
    }
    Ok(())
}

/// Sorts a `hound` error about `filename`, met while doing `op`.
///
/// An io error is sorted by [`sys_at`], except an unexpected end of file:
/// the file is truncated.
fn hound_error(
    op: &'static str,
    filename: &Path,
) -> impl FnOnce(hound::Error) -> BunsenError {
    let filename = filename.to_path_buf();
    move |e| match e {
        hound::Error::IoError(e) if e.kind() != std::io::ErrorKind::UnexpectedEof => {
            sys_at(op, &filename)(e)
        }
        hound::Error::Unsupported => BunsenError::unsupported(format!(
            "{}: {op}: the WAV format is not supported",
            filename.display()
        ))
        .with_cause(e),
        e => BunsenError::invalid_resource(format!(
            "{}: {op}: not a readable WAV file",
            filename.display()
        ))
        .with_cause(e),
    }
}

/// Sorts a `symphonia` error about `filename`, met while doing `op`.
///
/// An io error is sorted by [`sys_at`], except an unexpected end of file:
/// the stream is truncated.
fn symphonia_error(
    op: &'static str,
    filename: &Path,
) -> impl FnOnce(symphonia::core::errors::Error) -> BunsenError {
    use symphonia::core::errors::Error;

    let filename = filename.to_path_buf();
    move |e| match e {
        Error::IoError(e) if e.kind() != std::io::ErrorKind::UnexpectedEof => {
            sys_at(op, &filename)(e)
        }
        Error::Unsupported(_) => BunsenError::unsupported(format!(
            "{}: {op}: the audio format is not supported",
            filename.display()
        ))
        .with_cause(e),
        e => BunsenError::invalid_resource(format!(
            "{}: {op}: the audio does not decode",
            filename.display()
        ))
        .with_cause(e),
    }
}

/// Reads a WAV file through `hound`.
fn load_wav_mono_sr(
    filename: &Path,
    sample_rate: usize,
) -> BunsenResult<Vec<f32>> {
    let mut reader = WavReader::open(filename).map_err(hound_error("open", filename))?;
    let spec = reader.spec();

    check_mono_sr(
        filename,
        spec.channels as usize,
        spec.sample_rate as usize,
        sample_rate,
    )?;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(hound_error("read samples", filename))?,
        (SampleFormat::Int, bits) => {
            let scale = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .map_err(hound_error("read samples", filename))?
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
/// Gapless playback is the `AudioDecoderOptions` default, so an mp3's encoder
/// delay and padding are trimmed rather than returned as leading and trailing
/// silence — which would otherwise shift every frame of a spectrogram computed
/// from it.
fn load_compressed_mono_sr(
    filename: &Path,
    ext: &str,
    sample_rate: usize,
) -> BunsenResult<Vec<f32>> {
    let file = std::fs::File::open(filename).map_err(sys_at("open", filename))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if !ext.is_empty() {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(symphonia_error("probe", filename))?;

    let track = format.default_track(TrackType::Audio).ok_or_else(|| {
        BunsenError::invalid_resource(format!(
            "{}: the file has no decodable audio track",
            filename.display()
        ))
    })?;

    let track_id = track.id;

    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| {
            BunsenError::invalid_resource(format!(
                "{}: the audio track has no audio codec parameters",
                filename.display()
            ))
        })?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .map_err(symphonia_error("make a decoder", filename))?;

    let mut samples: Vec<f32> = Vec::new();

    // Reused across packets: `copy_to_vec_interleaved` resizes to the exact
    // sample count and keeps whatever capacity the widest packet so far needed.
    let mut packet_samples: Vec<f32> = Vec::new();

    // A clean end of stream is `Ok(None)`.
    while let Some(packet) = format
        .next_packet()
        .map_err(symphonia_error("read a packet", filename))?
    {
        if packet.track_id != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(symphonia_error("decode a packet", filename))?;
        let spec = decoded.spec();

        check_mono_sr(
            filename,
            spec.channels().count(),
            spec.rate() as usize,
            sample_rate,
        )?;

        decoded.copy_to_vec_interleaved(&mut packet_samples);
        samples.extend_from_slice(&packet_samples);
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        LookupError,
        testing::ErrorMatcher,
    };

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

        // A sample-rate mismatch is an error, not a silent resample.
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("tone.wav: the sample rate is 16000; it must be 8000")
            .assert_err(&load_audio_mono_sr(WAV, 8000));
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

        // A sample-rate mismatch is an error, not a silent resample.
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("tone.mp3: the sample rate is 16000; it must be 44100")
            .assert_err(&load_audio_mono_sr(MP3, 44100));
    }

    /// A missing file is a `Lookup` naming the path, in both decoders.
    #[test]
    fn test_missing_file_is_a_lookup() {
        for name in ["no-such-file.wav", "no-such-file.mp3"] {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/audio/").to_string() + name;
            ErrorMatcher::kind(BunsenErrorKind::Lookup)
                .message_contains(name)
                .has_cause::<LookupError>()
                .assert_err(&load_audio_mono_sr(&path, 16000));
        }
    }

    /// A file that is not audio is an `InvalidResource` or `Unsupported`
    /// naming the path.
    #[test]
    fn test_non_audio_file_names_the_file() {
        let wav = load_audio_mono_sr(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"), 16000);
        ErrorMatcher::new()
            .with_kind_matching(crate::errors::testing::value::one_of([
                BunsenErrorKind::InvalidResource,
                BunsenErrorKind::Unsupported,
            ]))
            .message_contains("Cargo.toml")
            .assert_err(&wav);
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
