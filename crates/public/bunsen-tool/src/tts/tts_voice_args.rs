//! [`TtsModelArgs`], [`TtsAnnotationArgs`] and [`TtsVoiceOutArgs`].

use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
        sys_at,
    },
    kits::speech::tts::driver::{
        TtsDriverConfig,
        TtsSynthesizer,
        TtsVoiceAnnotations,
        TtsVoicePrint,
        TtsVoiceReference,
    },
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    precision::MinPrecisionArgs,
};
use clap::ArgMatches;

use crate::{
    support::{
        ChooseArgs,
        with_patch,
        with_presets,
    },
    tts::{
        TtsFamily,
        TtsFamilyArgs,
        choose_tts_model,
    },
};

/// A voice file's extension in a voice directory: `NAME.voice.safetensors`.
pub const TTS_VOICE_FILE_SUFFIX: &str = ".voice.safetensors";

/// The flags that choose a TTS checkpoint and load it.
#[derive(clap::Args, Debug)]
pub struct TtsModelArgs {
    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    /// The device. A model loads at the precision it ships in where the
    /// device accelerates it, else in f32 (`LoadPolicy::performance`);
    /// `--min-precision` puts a floor under that.
    #[clap(flatten)]
    pub device: BackendArgs,

    /// The narrowest float the checkpoint's weights are placed in.
    #[clap(flatten)]
    pub precision: MinPrecisionArgs,

    /// The checkpoint, chosen among every TTS family's builds by `--model`
    /// and the stage flags; the default checkpoint when nothing is named.
    #[clap(flatten)]
    pub choose: ChooseArgs,

    /// The family the checkpoint is chosen among (`zonos3`, `qwen3-tts`).
    ///
    /// Every family's when omitted. Needed for a provider ref
    /// (`hf:org/repo`), which names no family.
    #[arg(long, value_enum)]
    pub family: Option<TtsFamily>,

    /// The flags of one family's engine.
    #[clap(flatten)]
    pub family_flags: TtsFamilyArgs,
}

/// A loaded checkpoint: the synthesizer, its name and its family.
pub struct TtsLoaded {
    /// The synthesizer.
    pub synthesizer: Arc<dyn TtsSynthesizer>,

    /// The checkpoint, as chosen.
    pub model: String,

    /// Its family.
    pub family: TtsFamily,
}

impl TtsModelArgs {
    /// The chosen checkpoint, loaded, its driver configured by `driver` for
    /// its family; `matches` are the command's own, for the stage flags'
    /// order.
    ///
    /// # Errors
    /// The choice's, the cache's, the device's and the family's load.
    pub fn load(
        &self,
        matches: &ArgMatches,
        driver: impl FnOnce(TtsFamily) -> BunsenResult<TtsDriverConfig>,
    ) -> BunsenResult<TtsLoaded> {
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        let flags = self.choose.flags(matches)?;
        let (mut model, family) = choose_tts_model(&self.choose.model, &flags, self.family)?;
        if !flags.presets.presets.is_empty() {
            let presets: Vec<&str> = flags.presets.presets.iter().map(String::as_str).collect();
            model = with_presets(&model, &presets)?;
        }
        if let Some(patch) = self.choose.patch()? {
            model = with_patch(&model, &patch)?;
        }
        let synthesizer = family.load(
            &model,
            &cache,
            &device,
            self.precision.load_policy(),
            driver(family)?,
            &self.family_flags,
        )?;
        Ok(TtsLoaded {
            synthesizer,
            model,
            family,
        })
    }
}

/// What is given of a voice when its reference is collected: the annotate
/// stage.
#[derive(clap::Args, Debug, Default)]
pub struct TtsAnnotationArgs {
    /// The voice's name; its voice file is named so in a voice directory.
    ///
    /// The reference directory's name when omitted.
    #[arg(long)]
    pub name: Option<String>,

    /// What the clips say.
    ///
    /// A mode that reads the reference's words (Qwen3-TTS's `icl`) needs it.
    #[arg(long)]
    pub transcript: Option<String>,

    /// The clips' language: a language tag, or a model's key.
    #[arg(long)]
    pub language: Option<String>,

    /// The speaker's gender, as you would say it.
    #[arg(long)]
    pub gender: Option<String>,

    /// The speaker's style, as you would say it.
    #[arg(long)]
    pub style: Option<String>,
}

impl TtsAnnotationArgs {
    /// The annotations, for a reference.
    pub fn annotations(&self) -> TtsVoiceAnnotations {
        TtsVoiceAnnotations {
            name: self.name.clone(),
            transcript: self.transcript.clone(),
            language: self.language.clone(),
            gender: self.gender.clone(),
            style: self.style.clone(),
        }
    }
}

/// Where a built voice is written: a file, or a voice directory.
#[derive(clap::Args, Debug, Default)]
#[group(required = true, multiple = false)]
pub struct TtsVoiceOutArgs {
    /// The voice file to write.
    #[arg(long, short)]
    pub out: Option<PathBuf>,

    /// The voice directory to write the voice into, as
    /// `NAME.voice.safetensors`.
    #[arg(long)]
    pub voice_dir: Option<PathBuf>,
}

impl TtsVoiceOutArgs {
    /// Where `reference`'s voice goes: `--out`, or `NAME.voice.safetensors` in
    /// `--voice-dir`, `NAME` the reference's name (else its directory's).
    pub fn path(
        &self,
        reference: &TtsVoiceReference,
        reference_dir: &Path,
    ) -> PathBuf {
        match (&self.out, &self.voice_dir) {
            (Some(out), _) => out.clone(),
            (None, Some(dir)) => dir.join(format!(
                "{}{TTS_VOICE_FILE_SUFFIX}",
                voice_name(reference, reference_dir)
            )),
            (None, None) => unreachable!("clap requires one"),
        }
    }

    /// Writes `voice` for `reference`, never over a file.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) for a file
    /// that is there already; the write's, as `sys_at` sorts it.
    pub fn write(
        &self,
        voice: &TtsVoicePrint,
        reference: &TtsVoiceReference,
        reference_dir: &Path,
    ) -> BunsenResult<PathBuf> {
        let path = self.path(reference, reference_dir);
        if path.exists() {
            return Err(BunsenError::illegal(format!(
                "{}: a voice file is never written over; remove it, or name another",
                path.display()
            )));
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(sys_at("create", dir))?;
        }
        voice.write(&path)?;
        Ok(path)
    }
}

/// A reference's voice name: its record's, else its directory's file name
/// (less a `.voice-ref` suffix).
pub fn voice_name(
    reference: &TtsVoiceReference,
    reference_dir: &Path,
) -> String {
    reference.record.name.clone().unwrap_or_else(|| {
        let file = reference_dir
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("voice");
        file.strip_suffix(".voice-ref").unwrap_or(file).to_string()
    })
}

/// The voice files in `dir`, by name, sorted.
///
/// # Errors
/// The directory's read, as `sys_at` sorts it.
pub fn voice_files(dir: &Path) -> BunsenResult<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(sys_at("read", dir))? {
        let path = entry.map_err(sys_at("read", dir))?.path();
        let name = path
            .file_name()
            .and_then(|f| f.to_str())
            .and_then(|f| f.strip_suffix(TTS_VOICE_FILE_SUFFIX))
            .map(str::to_string);
        if let Some(name) = name {
            found.push((name, path));
        }
    }
    found.sort();
    Ok(found)
}

/// The voice file `name` in `dir`.
///
/// # Errors
/// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) for a name with no
/// file there, naming the voices there are; the file's read.
pub fn read_voice_file(
    dir: &Path,
    name: &str,
) -> BunsenResult<TtsVoicePrint> {
    let path = dir.join(format!("{name}{TTS_VOICE_FILE_SUFFIX}"));
    if !path.exists() {
        let there: Vec<String> = voice_files(dir)?.into_iter().map(|(n, _)| n).collect();
        return Err(BunsenError::lookup(
            LookupError::missing("voice", name).with_candidates(there),
        ));
    }
    TtsVoicePrint::read(&path)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bunsen::kits::speech::tts::driver::{
        TTS_VOICE_REFERENCE_VERSION,
        TtsVoiceIdentity,
        TtsVoiceReferenceRecord,
        TtsVoiceTensor,
    };

    use super::*;

    fn reference(name: Option<&str>) -> TtsVoiceReference {
        TtsVoiceReference {
            record: TtsVoiceReferenceRecord {
                version: TTS_VOICE_REFERENCE_VERSION,
                name: name.map(Into::into),
                transcript: None,
                language: None,
                gender: None,
                style: None,
                created_unix: None,
                clips: Vec::new(),
            },
            clips: Vec::new(),
            digest: "ab".into(),
        }
    }

    fn voice() -> TtsVoicePrint {
        TtsVoicePrint {
            identity: TtsVoiceIdentity {
                family: "zonos3".into(),
                encoder: "e".into(),
                mode: "embedding".into(),
                reference: "ab".into(),
            },
            tensors: BTreeMap::from([(
                "embedding".to_string(),
                TtsVoiceTensor::F32 {
                    shape: vec![1],
                    values: vec![0.5],
                },
            )]),
        }
    }

    /// A voice is named for its reference (else its directory), written once
    /// into a voice directory, and found there by name.
    #[test]
    fn test_voice_dir() {
        let work = tempfile::tempdir().unwrap();
        let dir = work.path().join("voices");
        let src = Path::new("/x/ada.voice-ref");
        assert_eq!(voice_name(&reference(None), src), "ada");
        assert_eq!(voice_name(&reference(Some("Bea")), src), "Bea");
        let out = TtsVoiceOutArgs {
            out: None,
            voice_dir: Some(dir.clone()),
        };
        let path = out.write(&voice(), &reference(None), src).unwrap();
        assert_eq!(path, dir.join("ada.voice.safetensors"));
        assert!(
            out.write(&voice(), &reference(None), src).is_err(),
            "not over"
        );
        assert_eq!(voice_files(&dir).unwrap(), vec![("ada".into(), path)]);
        assert_eq!(read_voice_file(&dir, "ada").unwrap(), voice());
        let err = read_voice_file(&dir, "bea").unwrap_err();
        assert_eq!(err.kind(), bunsen::errors::BunsenErrorKind::Lookup);
    }
}
