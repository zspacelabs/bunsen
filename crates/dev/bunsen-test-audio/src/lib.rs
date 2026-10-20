//! # The workspace's test audio
//!
//! Every audio clip the workspace's tests, validation suites and fixture
//! scripts read, in one place: the clip, what it is, where it came from, and
//! its ground truth when it has one. `testdata/README.md` has each clip's
//! provenance and how it was cut.
//!
//! A test takes this crate as a dev-dependency and reads a clip by its
//! [`AudioFixture`]:
//!
//! ```
//! let clip = bunsen_test_audio::JFK_MOON;
//! assert!(clip.path().exists());
//! assert!(
//!     clip.transcript()
//!         .unwrap()
//!         .contains("We choose to go to the Moon")
//! );
//! ```
//!
//! The crate is a leaf: it depends on nothing in the workspace, so `bunsen`
//! itself can take it as a dev-dependency without its tests seeing two copies
//! of bunsen. It hands out paths, not samples: decoding is
//! `bunsen::support::audio`'s.
//!
//! Fixtures that are a model's output on a clip (an upstream decode, a
//! reference's probabilities) are not audio sources, and live with the
//! validation crate of the model that made them.

use std::path::{
    Path,
    PathBuf,
};

/// One committed audio clip, under `testdata/`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioFixture {
    /// The clip's file name.
    pub file: &'static str,

    /// The sample rate, in Hz.
    pub sample_rate: usize,

    /// The channel count.
    pub channels: usize,

    /// The duration, in seconds, as encoded.
    pub seconds: f64,

    /// What the clip holds.
    pub description: &'static str,

    /// The clip's licence.
    pub licence: &'static str,

    /// The SHA-256 of the file.
    pub sha256: &'static str,

    /// The file name of the ground-truth transcript (what a person hears), if
    /// the clip has one.
    pub transcript_file: Option<&'static str>,
}

impl AudioFixture {
    /// The clip's path.
    pub fn path(&self) -> PathBuf {
        testdata_dir().join(self.file)
    }

    /// The ground-truth transcript's path, if the clip has one.
    pub fn transcript_path(&self) -> Option<PathBuf> {
        self.transcript_file.map(|f| testdata_dir().join(f))
    }

    /// The ground-truth transcript, if the clip has one.
    ///
    /// # Panics
    /// If the transcript is named but cannot be read.
    pub fn transcript(&self) -> Option<String> {
        self.transcript_path().map(|p| {
            std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("read the transcript {}: {e}", p.display()))
        })
    }
}

/// The directory the committed clips are in.
pub fn testdata_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata"))
}

/// 60 s of John F. Kennedy's 1962 Rice University address ("We choose to go
/// to the Moon"), live, with hall noise and applause, and its transcript.
pub const JFK_MOON: AudioFixture = AudioFixture {
    file: "jfk_moon.mp3",
    sample_rate: 16_000,
    channels: 1,
    seconds: 60.0,
    description: "\"We choose to go to the Moon\", live, with hall noise and applause",
    licence: "public domain (work of the U.S. federal government)",
    sha256: "22c8682f71304713c08f5c7fe7a9228fc806ea822abc16eb5c178986506a8c0a",
    transcript_file: Some("jfk_moon.txt"),
};

/// Seconds 7.0 to 11.0 of [`JFK_MOON`]: two phrases around a one-second
/// pause, so a voice-activity gate with a short silence threshold finds two
/// regions and a patient one finds one.
pub const JFK_MOON_4S: AudioFixture = AudioFixture {
    file: "jfk_moon_4s.mp3",
    sample_rate: 16_000,
    channels: 1,
    seconds: 4.0,
    description: "speech with a one-second pause, cut from jfk_moon.mp3",
    licence: "public domain (work of the U.S. federal government)",
    sha256: "948ab6418b8e38c991acaee9f152fef0fe3798d9c816ef91f34420d10fb923fd",
    transcript_file: None,
};

/// A synthetic 440 Hz tone, 0.5 s: 8000 samples exactly, 16-bit PCM.
pub const TONE_WAV: AudioFixture = AudioFixture {
    file: "tone.wav",
    sample_rate: 16_000,
    channels: 1,
    seconds: 0.5,
    description: "a synthetic 440 Hz tone, 16-bit PCM",
    licence: "none needed (synthetic)",
    sha256: "273f00149c8a92008d90b30865e76775a31162e5a0006591cc1d128a73e8528b",
    transcript_file: None,
};

/// [`TONE_WAV`] as an mp3, LAME 48 kbps.
///
/// It decodes to slightly more than 8000 samples: mp3 frames in blocks of 576
/// samples, so the tail rounds up.
pub const TONE_MP3: AudioFixture = AudioFixture {
    file: "tone.mp3",
    sample_rate: 16_000,
    channels: 1,
    seconds: 0.5,
    description: "the synthetic tone, LAME 48 kbps",
    licence: "none needed (synthetic)",
    sha256: "8e3ed9c49530690819023425f1e3249ebf60bd9cb8b1882e2ab4242bd8b60900",
    transcript_file: None,
};

/// silero-vad's own test recording: 60 s of speech, 16-bit PCM.
pub const SILERO_VAD_TEST: AudioFixture = AudioFixture {
    file: "silero_vad_test.wav",
    sample_rate: 16_000,
    channels: 1,
    seconds: 60.0,
    description: "silero-vad's tests/data/test.wav",
    licence: "MIT (silero-vad)",
    sha256: "89f17d9c94c4b31eb320f424628bcbc920abaddbee6e2760fd868bfb1d9a2e47",
    transcript_file: None,
};

/// Every committed clip.
pub const ALL: &[AudioFixture] = &[JFK_MOON, JFK_MOON_4S, TONE_WAV, TONE_MP3, SILERO_VAD_TEST];

/// The clip whose file is `file`, as a fixture that records the clip it was
/// made from names it.
///
/// # Panics
/// If no clip has that file name.
pub fn by_file(file: &str) -> AudioFixture {
    *ALL.iter()
        .find(|c| c.file == file)
        .unwrap_or_else(|| panic!("no test clip named {file}"))
}

#[cfg(test)]
mod tests {
    use sha2::{
        Digest,
        Sha256,
    };

    use super::*;

    fn sha256_hex(path: &Path) -> String {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Every clip is the file its pin names, and every named transcript
    /// exists.
    #[test]
    fn test_every_clip_is_pinned() {
        for clip in ALL {
            assert_eq!(sha256_hex(&clip.path()), clip.sha256, "{}", clip.file);
            if let Some(path) = clip.transcript_path() {
                assert!(path.is_file(), "{}", path.display());
            }
        }
    }

    /// The file names are distinct, and each finds its clip.
    #[test]
    fn test_file_names_are_distinct() {
        for (i, a) in ALL.iter().enumerate() {
            for b in &ALL[i + 1..] {
                assert_ne!(a.file, b.file);
            }
            assert_eq!(by_file(a.file), *a);
        }
    }

    /// A PCM clip's header agrees with its fixture: rate, channels, and
    /// duration from the data chunk's length.
    #[test]
    fn test_wav_headers_agree() {
        for clip in ALL.iter().filter(|c| c.file.ends_with(".wav")) {
            let bytes = std::fs::read(clip.path()).unwrap();
            assert_eq!(&bytes[0..4], b"RIFF", "{}", clip.file);
            let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;
            let u32_at = |i: usize| {
                u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize
            };

            // Walk the chunks for `fmt ` and `data`.
            let (mut fmt, mut data) = (None, None);
            let mut at = 12;
            while at + 8 <= bytes.len() {
                let len = u32_at(at + 4);
                match &bytes[at..at + 4] {
                    b"fmt " => fmt = Some(at + 8),
                    b"data" => data = Some(len),
                    _ => {}
                }
                at += 8 + len + (len & 1);
            }
            let fmt = fmt.expect("a fmt chunk");
            let data = data.expect("a data chunk");

            let channels = u16_at(fmt + 2);
            let rate = u32_at(fmt + 4);
            let bits = u16_at(fmt + 14);
            assert_eq!(channels, clip.channels, "{}", clip.file);
            assert_eq!(rate, clip.sample_rate, "{}", clip.file);
            let seconds = data as f64 / (rate * channels * bits / 8) as f64;
            assert!(
                (seconds - clip.seconds).abs() < 1e-3,
                "{}: {seconds}",
                clip.file
            );
        }
    }
}
