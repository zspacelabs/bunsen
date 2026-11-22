//! A [`TtsVoiceReference`]'s directory: its clips, and `reference.json`.

use std::{
    fs,
    path::Path,
};

use sha2::{
    Digest,
    Sha256,
};

use super::{
    TTS_VOICE_REFERENCE_VERSION,
    TtsClipEdit,
    TtsClipRecord,
    TtsReferenceClip,
    TtsVoiceReference,
    TtsVoiceReferenceRecord,
};
use crate::{
    data::cache::sha256_of,
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    support::audio::load_audio,
};

/// The record's file in a reference directory.
pub const TTS_VOICE_REFERENCE_FILE: &str = "reference.json";

/// What a person gives of a voice, the annotate stage's input: each field
/// given replaces the reference's; one not given keeps it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TtsVoiceAnnotations {
    /// The voice's name.
    pub name: Option<String>,

    /// What the clips say.
    pub transcript: Option<String>,

    /// The clips' language: a language tag or an engine's key.
    pub language: Option<String>,

    /// The speaker's gender.
    pub gender: Option<String>,

    /// The speaker's style.
    pub style: Option<String>,
}

impl TtsVoiceAnnotations {
    fn apply(
        &self,
        record: &mut TtsVoiceReferenceRecord,
    ) {
        let set = |to: &mut Option<String>, from: &Option<String>| {
            if from.is_some() {
                to.clone_from(from);
            }
        };
        set(&mut record.name, &self.name);
        set(&mut record.transcript, &self.transcript);
        set(&mut record.language, &self.language);
        set(&mut record.gender, &self.gender);
        set(&mut record.style, &self.style);
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn write_record(
    dir: &Path,
    record: &TtsVoiceReferenceRecord,
) -> BunsenResult<()> {
    let path = dir.join(TTS_VOICE_REFERENCE_FILE);
    let json = serde_json::to_string_pretty(record)
        .map_err(|e| BunsenError::invalid_resource(format!("a voice reference record: {e}")))?;
    fs::write(&path, json + "\n").map_err(sys_at("write", &path))
}

impl TtsVoiceReference {
    /// Collects a voice reference into `dir`, which must not exist or be empty:
    /// the ingest stage (each of `clips` copied in byte for byte, decoded at
    /// its own rate and channels, its hash, rate, channels and length recorded)
    /// and the annotate stage (`annotations`); then reads it back
    /// ([`read_dir`](Self::read_dir)).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no clips,
    /// two of one file name, a clip of no samples, or a `dir` that has
    /// anything in it (a reference is never written over);
    /// [`load_audio`]'s for a clip that does not decode; as [`sys_at`]
    /// sorts an io failure.
    pub fn collect(
        dir: impl AsRef<Path>,
        clips: &[impl AsRef<Path>],
        annotations: &TtsVoiceAnnotations,
    ) -> BunsenResult<Self> {
        let dir = dir.as_ref();
        if clips.is_empty() {
            return Err(BunsenError::illegal("a voice reference of no clips"));
        }
        if dir.exists() {
            let mut entries = fs::read_dir(dir).map_err(sys_at("read", dir))?;
            if entries.next().is_some() {
                return Err(BunsenError::illegal(format!(
                    "{}: a voice reference is collected into a new or empty directory",
                    dir.display()
                )));
            }
        }
        let mut records = Vec::with_capacity(clips.len());
        for clip in clips {
            let clip = clip.as_ref();
            let file = clip
                .file_name()
                .and_then(|f| f.to_str())
                .ok_or_else(|| {
                    BunsenError::illegal(format!("{}: a clip with no file name", clip.display()))
                })?
                .to_string();
            if records.iter().any(|r: &TtsClipRecord| r.file == file) {
                return Err(BunsenError::illegal(format!(
                    "two clips named {file}: a reference keeps each under its own name"
                )));
            }
            let audio = load_audio(clip)?;
            let samples = audio.channels.first().map_or(0, Vec::len);
            if samples == 0 {
                return Err(BunsenError::illegal(format!(
                    "{}: a clip of no samples",
                    clip.display()
                )));
            }
            records.push(TtsClipRecord {
                file,
                sha256: sha256_of(clip)?,
                sample_rate: audio.sample_rate,
                channels: audio.channels.len(),
                samples,
                edits: Vec::new(),
            });
        }
        fs::create_dir_all(dir).map_err(sys_at("create", dir))?;
        for (clip, record) in clips.iter().zip(&records) {
            let to = dir.join(&record.file);
            fs::copy(clip.as_ref(), &to).map_err(sys_at("copy", &to))?;
        }
        let created_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        let mut record = TtsVoiceReferenceRecord {
            version: TTS_VOICE_REFERENCE_VERSION,
            name: None,
            transcript: None,
            language: None,
            gender: None,
            style: None,
            created_unix,
            clips: records,
        };
        annotations.apply(&mut record);
        write_record(dir, &record)?;
        Self::read_dir(dir)
    }

    /// Updates the reference in `dir` with `annotations` (the annotate stage
    /// again: a corrected transcript, a name), and reads it back. Its digest
    /// changes with its record.
    ///
    /// # Errors
    /// As [`read_dir`](Self::read_dir), and as [`sys_at`] sorts an io
    /// failure.
    pub fn annotate(
        dir: impl AsRef<Path>,
        annotations: &TtsVoiceAnnotations,
    ) -> BunsenResult<Self> {
        let dir = dir.as_ref();
        let mut record = Self::read_dir(dir)?.record;
        annotations.apply(&mut record);
        write_record(dir, &record)?;
        Self::read_dir(dir)
    }

    /// Reads the voice reference in `dir`: its record, each clip decoded (its
    /// hash, rate, channels and length checked against the record, its edits
    /// applied), and its digest, `reference.json`'s sha256.
    ///
    /// # Errors
    /// As [`sys_at`] sorts an io failure (a `dir` with no `reference.json` is a
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup));
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for a record that does not parse or is of another version, a clip
    /// whose file changed since it was recorded or decodes otherwise, or an
    /// edit outside its clip; each names the directory.
    pub fn read_dir(dir: impl AsRef<Path>) -> BunsenResult<Self> {
        let dir = dir.as_ref();
        let at = |e: BunsenError| e.context(dir.display());
        let path = dir.join(TTS_VOICE_REFERENCE_FILE);
        let bytes = fs::read(&path).map_err(sys_at("read", &path))?;
        let record: TtsVoiceReferenceRecord = serde_json::from_slice(&bytes)
            .map_err(|e| BunsenError::invalid_resource(format!("{}: {e}", path.display())))?;
        if record.version != TTS_VOICE_REFERENCE_VERSION {
            return Err(at(BunsenError::invalid_resource(format!(
                "a voice reference of version {}; this reads version {TTS_VOICE_REFERENCE_VERSION}",
                record.version
            ))));
        }
        let clips = record
            .clips
            .iter()
            .map(|clip| {
                let file = dir.join(&clip.file);
                if sha256_of(&file)? != clip.sha256 {
                    return Err(BunsenError::invalid_resource(format!(
                        "{}: the clip changed since the reference recorded it",
                        file.display()
                    )));
                }
                let audio = load_audio(&file)?;
                let samples = audio.channels.first().map_or(0, Vec::len);
                if (audio.sample_rate, audio.channels.len(), samples)
                    != (clip.sample_rate, clip.channels, clip.samples)
                {
                    return Err(BunsenError::invalid_resource(format!(
                        "{}: decodes as {} Hz, {} channels, {samples} samples; the reference \
                         recorded {} Hz, {} channels, {} samples",
                        file.display(),
                        audio.sample_rate,
                        audio.channels.len(),
                        clip.sample_rate,
                        clip.channels,
                        clip.samples
                    )));
                }
                let mut channels = audio.channels;
                for edit in &clip.edits {
                    match *edit {
                        TtsClipEdit::Trim { start, end } => {
                            let len = channels[0].len();
                            if start >= end || end > len {
                                return Err(BunsenError::invalid_resource(format!(
                                    "{}: a trim to {start}..{end} of {len} samples",
                                    file.display()
                                )));
                            }
                            for c in &mut channels {
                                *c = c[start..end].to_vec();
                            }
                        }
                    }
                }
                Ok(TtsReferenceClip {
                    record: clip.clone(),
                    channels,
                })
            })
            .collect::<BunsenResult<Vec<_>>>()
            .map_err(at)?;
        Ok(Self {
            record,
            clips,
            digest: hex_sha256(&bytes),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::BunsenErrorKind,
        support::audio::save_wav_mono,
    };

    fn clip(
        dir: &Path,
        name: &str,
        n: usize,
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let samples: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        save_wav_mono(&path, &samples, 16_000).unwrap();
        path
    }

    /// A reference made from clips reads back as made: the clips copied byte
    /// for byte, recorded, annotated; its digest is its record's.
    #[test]
    fn test_create_and_read() {
        let work = tempfile::tempdir().unwrap();
        let a = clip(work.path(), "a.wav", 1600);
        let b = clip(work.path(), "b.wav", 800);
        let dir = work.path().join("ada.voice-ref");
        let annotations = TtsVoiceAnnotations {
            name: Some("Ada".into()),
            transcript: Some("hello there".into()),
            ..TtsVoiceAnnotations::default()
        };
        let reference = TtsVoiceReference::collect(&dir, &[&a, &b], &annotations).unwrap();
        assert_eq!(reference.record.name.as_deref(), Some("Ada"));
        assert_eq!(reference.transcript(), Some("hello there"));
        assert_eq!(reference.clips.len(), 2);
        assert_eq!(reference.clips[0].record.samples, 1600);
        assert_eq!(reference.clips[1].sample_rate(), 16_000);
        assert!((reference.seconds() - 0.15).abs() < 1e-9);
        assert_eq!(fs::read(&a).unwrap(), fs::read(dir.join("a.wav")).unwrap());
        let again = TtsVoiceReference::read_dir(&dir).unwrap();
        assert_eq!(again, reference);
        assert_eq!(
            reference.digest,
            hex_sha256(&fs::read(dir.join(TTS_VOICE_REFERENCE_FILE)).unwrap())
        );

        // Annotating again changes the record, and so the digest.
        let renamed = TtsVoiceReference::annotate(
            &dir,
            &TtsVoiceAnnotations {
                language: Some("en".into()),
                ..TtsVoiceAnnotations::default()
            },
        )
        .unwrap();
        assert_eq!(renamed.record.language.as_deref(), Some("en"));
        assert_eq!(renamed.record.name.as_deref(), Some("Ada"), "kept");
        assert_ne!(renamed.digest, reference.digest);
    }

    /// A reference is never written over; a changed clip, a bad trim and a
    /// missing record are refused.
    #[test]
    fn test_refusals() {
        let work = tempfile::tempdir().unwrap();
        let a = clip(work.path(), "a.wav", 1600);
        let none: &[&Path] = &[];
        let dir = work.path().join("src");
        let err =
            TtsVoiceReference::collect(&dir, none, &TtsVoiceAnnotations::default()).unwrap_err();
        assert_eq!(err.kind(), BunsenErrorKind::Illegal);
        TtsVoiceReference::collect(&dir, &[&a], &TtsVoiceAnnotations::default()).unwrap();
        let err =
            TtsVoiceReference::collect(&dir, &[&a], &TtsVoiceAnnotations::default()).unwrap_err();
        assert_eq!(err.kind(), BunsenErrorKind::Illegal, "not written over");
        let twice = work.path().join("twice");
        let err = TtsVoiceReference::collect(&twice, &[&a, &a], &TtsVoiceAnnotations::default())
            .unwrap_err();
        assert_eq!(err.kind(), BunsenErrorKind::Illegal);

        // A trim outside the clip.
        let path = dir.join(TTS_VOICE_REFERENCE_FILE);
        let mut record: TtsVoiceReferenceRecord =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record.clips[0].edits = vec![TtsClipEdit::Trim {
            start: 10,
            end: 5000,
        }];
        write_record(&dir, &record).unwrap();
        let err = TtsVoiceReference::read_dir(&dir).unwrap_err();
        assert_eq!(err.kind(), BunsenErrorKind::InvalidResource);
        record.clips[0].edits = vec![TtsClipEdit::Trim {
            start: 100,
            end: 600,
        }];
        write_record(&dir, &record).unwrap();
        let trimmed = TtsVoiceReference::read_dir(&dir).unwrap();
        assert_eq!(trimmed.clips[0].channels[0].len(), 500);

        // A changed clip.
        save_wav_mono(dir.join("a.wav"), &[0.0; 1600], 16_000).unwrap();
        let err = TtsVoiceReference::read_dir(&dir).unwrap_err();
        assert!(format!("{err}").contains("changed"), "{err}");

        let empty = work.path().join("empty");
        fs::create_dir(&empty).unwrap();
        assert_eq!(
            TtsVoiceReference::read_dir(&empty).unwrap_err().kind(),
            BunsenErrorKind::Lookup
        );
    }
}
