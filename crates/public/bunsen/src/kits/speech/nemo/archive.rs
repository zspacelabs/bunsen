//! `.nemo` checkpoints: `NeMo`'s tar of a config and a state dict.
use std::{
    fs::File,
    io::{
        BufReader,
        Read,
    },
    path::{
        Path,
        PathBuf,
    },
};

use burn::store::{
    bridge::from_pytorch,
    burn_pack::Tensor as PackTensor,
    pytorch::PytorchReader,
};
use serde::de::DeserializeOwned;

use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// The member holding the model's config: `NeMo`'s `model_config.yaml`.
pub const NEMO_CONFIG_MEMBER: &str = "model_config.yaml";

/// The member holding the model's state dict, a `torch.save` ZIP:
/// `NeMo`'s `model_weights.ckpt`.
pub const NEMO_WEIGHTS_MEMBER: &str = "model_weights.ckpt";

/// A `.nemo` archive: `NeMo`'s `save_to`, a tar (plain, or gzipped by older
/// `NeMo`s) of [`NEMO_CONFIG_MEMBER`] and [`NEMO_WEIGHTS_MEMBER`].
///
/// Opening reads the config; the weights stay in the archive until
/// [`extract_weights`](Self::extract_weights) copies them out. The burn-store's
/// `PyTorch` reader reads only a file of its own, so the state dict is read
/// from that copy ([`read_nemo_weights`]).
#[derive(Clone, Debug)]
pub struct NemoArchive {
    path: PathBuf,
    config: String,
}

impl NemoArchive {
    /// Opens the archive at `path` and reads its config.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when there is no
    /// file; [`Sys`](crate::errors::BunsenErrorKind::Sys) when it does not
    /// read;
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it is not a tar, or holds no [`NEMO_CONFIG_MEMBER`] or no
    /// [`NEMO_WEIGHTS_MEMBER`], or the config is not UTF-8.
    pub fn open(path: &Path) -> BunsenResult<Self> {
        let mut config = None;
        let mut weights = false;
        for_each_member(path, |name, entry| {
            match name {
                NEMO_CONFIG_MEMBER => {
                    let mut text = String::new();
                    entry.read_to_string(&mut text).map_err(|e| {
                        BunsenError::invalid_resource(format!(
                            "{NEMO_CONFIG_MEMBER} does not read as UTF-8"
                        ))
                        .with_cause(e)
                    })?;
                    config = Some(text);
                }
                NEMO_WEIGHTS_MEMBER => weights = true,
                _ => {}
            }
            Ok(config.is_some() && weights)
        })?;
        let missing = |member: &str| {
            BunsenError::invalid_resource(format!(
                "{}: a .nemo archive without {member}",
                path.display()
            ))
        };
        let config = config.ok_or_else(|| missing(NEMO_CONFIG_MEMBER))?;
        if !weights {
            return Err(missing(NEMO_WEIGHTS_MEMBER));
        }
        Ok(Self {
            path: path.to_path_buf(),
            config,
        })
    }

    /// The archive's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The config, as the archive holds it: YAML.
    pub fn config_yaml(&self) -> &str {
        &self.config
    }

    /// The config, deserialized: [`parse_nemo_yaml`] of
    /// [`config_yaml`](Self::config_yaml).
    ///
    /// # Errors
    /// As [`parse_nemo_yaml`].
    pub fn config<T: DeserializeOwned>(&self) -> BunsenResult<T> {
        parse_nemo_yaml(&self.config).map_err(|e| e.context(self.path.display()))
    }

    /// Copies the state dict out of the archive to `dest`, written whole or
    /// not at all: to a sibling first, then renamed into place.
    ///
    /// # Errors
    /// [`Sys`](crate::errors::BunsenErrorKind::Sys) when the archive does not
    /// read or `dest` does not write;
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when the archive no longer holds the member.
    pub fn extract_weights(
        &self,
        dest: &Path,
    ) -> BunsenResult<()> {
        let partial = dest.with_extension("partial");
        let mut found = false;
        for_each_member(&self.path, |name, entry| {
            if name != NEMO_WEIGHTS_MEMBER {
                return Ok(false);
            }
            let mut out = File::create(&partial).map_err(sys_at("create", &partial))?;
            std::io::copy(entry, &mut out).map_err(sys_at("write", &partial))?;
            out.sync_all().map_err(sys_at("write", &partial))?;
            found = true;
            Ok(true)
        })?;
        if !found {
            return Err(BunsenError::invalid_resource(format!(
                "{}: a .nemo archive without {NEMO_WEIGHTS_MEMBER}",
                self.path.display()
            )));
        }
        std::fs::rename(&partial, dest).map_err(sys_at("rename", dest))
    }

    /// Calls `f` with the state dict, read ([`read_nemo_weights`]) from a
    /// copy extracted to a fresh temporary directory, which is removed once
    /// `f` returns: the tensors read their bytes from the copy, so `f` is
    /// where they are applied.
    ///
    /// # Errors
    /// [`Sys`](crate::errors::BunsenErrorKind::Sys) when the temporary
    /// directory cannot be made; as
    /// [`extract_weights`](Self::extract_weights), [`read_nemo_weights`],
    /// and `f`.
    pub fn with_weights<R>(
        &self,
        f: impl FnOnce(Vec<PackTensor>) -> BunsenResult<R>,
    ) -> BunsenResult<R> {
        let dir = tempfile::Builder::new()
            .prefix("bunsen-nemo-")
            .tempdir()
            .map_err(|e| BunsenError::sys("create a temporary directory").with_cause(e))?;
        let ckpt = dir.path().join(NEMO_WEIGHTS_MEMBER);
        self.extract_weights(&ckpt)?;
        f(read_nemo_weights(&ckpt)?)
    }

    /// The config of a plain (not gzipped) archive from a prefix of its
    /// bytes: what a range request for the start of a `.nemo` returns, so a
    /// geometry can be read before the weights download. `NeMo` writes the
    /// config before the weights; directory entries, the PAX headers
    /// Python's `tarfile` writes before each member, and other members (a
    /// tokenizer's files) may come before it.
    ///
    /// # Returns
    /// `None` when the prefix ends before the config does, or the weights
    /// come first, or the archive is gzipped: the whole archive is then
    /// needed.
    pub fn config_from_prefix(prefix: &[u8]) -> Option<String> {
        if is_gzip(prefix) {
            return None;
        }
        let mut archive = tar::Archive::new(prefix);
        for entry in archive.entries().ok()? {
            let mut entry = entry.ok()?;
            if entry.header().entry_type().is_dir() {
                continue;
            }
            let name = member_name(&entry.path().ok()?).map(str::to_string);
            match name.as_deref() {
                Some(NEMO_CONFIG_MEMBER) => {
                    let size = usize::try_from(entry.size()).ok()?;
                    let mut text = String::with_capacity(size);
                    entry.read_to_string(&mut text).ok()?;
                    return (text.len() == size).then_some(text);
                }
                Some(NEMO_WEIGHTS_MEMBER) => return None,
                _ => continue,
            }
        }
        None
    }
}

/// Reads a `NeMo` state dict, as [`NemoArchive::extract_weights`] writes it,
/// through the burn-store's `PyTorch` reader: every entry, in name order, each
/// in its own dtype and named as `NeMo` names it.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// the file is not a `torch.save` checkpoint the reader reads.
pub fn read_nemo_weights(path: &Path) -> BunsenResult<Vec<PackTensor>> {
    let reader = PytorchReader::new(path).map_err(|e| {
        BunsenError::invalid_resource(format!(
            "{}: not a PyTorch checkpoint the reader reads",
            path.display()
        ))
        .with_cause(e)
    })?;
    let mut tensors: Vec<PackTensor> = reader
        .into_tensors()
        .into_values()
        .map(from_pytorch)
        .collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

/// Deserializes a `NeMo` YAML config (an archive's
/// [`NEMO_CONFIG_MEMBER`]), or a section of one, into `T`. Fields `T` does
/// not name are ignored, so a type names only what it reads.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource), with
/// the parser's message (its line and column), when the text is not YAML or
/// does not fit `T`.
pub fn parse_nemo_yaml<T: DeserializeOwned>(yaml: &str) -> BunsenResult<T> {
    serde_saphyr::from_str(yaml).map_err(|e| {
        BunsenError::invalid_resource(format!("a NeMo config that does not parse: {e}"))
    })
}

/// Whether `bytes` start as gzip does.
fn is_gzip(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x1f, 0x8b])
}

/// A member's name without the `./` `NeMo` writes before it; `None` for one
/// below the archive's root.
fn member_name(path: &Path) -> Option<&str> {
    let name = path.to_str()?;
    let name = name.strip_prefix("./").unwrap_or(name);
    (!name.contains('/')).then_some(name)
}

/// Calls `f` with each file member of the archive at `path` in order, by
/// name, until it returns `true`.
fn for_each_member(
    path: &Path,
    mut f: impl FnMut(&str, &mut dyn Read) -> BunsenResult<bool>,
) -> BunsenResult<()> {
    let mut file = BufReader::new(File::open(path).map_err(sys_at("open", path))?);
    let mut magic = [0u8; 2];
    let n = file.read(&mut magic).map_err(sys_at("read", path))?;
    let file = BufReader::new(File::open(path).map_err(sys_at("open", path))?);
    let reader: Box<dyn Read> = if is_gzip(&magic[..n]) {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let not_tar = |e: std::io::Error| {
        BunsenError::invalid_resource(format!("{}: not a .nemo archive (a tar)", path.display()))
            .with_cause(e)
    };
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries().map_err(not_tar)? {
        let mut entry = entry.map_err(not_tar)?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let entry_path = entry.path().map_err(not_tar)?.into_owned();
        let Some(name) = member_name(&entry_path) else {
            continue;
        };
        let name = name.to_string();
        if f(&name, &mut entry)? {
            break;
        }
    }
    Ok(())
}
