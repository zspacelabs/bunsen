//! Safetensors checkpoints: one file, or the shards of one.

use std::{
    collections::{
        BTreeMap,
        BTreeSet,
    },
    io::Read,
    path::{
        Path,
        PathBuf,
    },
};

use burn::prelude::Backend;
use burn_store::{
    ModuleSnapshot,
    SafetensorsStore,
    SafetensorsStoreError,
};
use serde::Deserialize;

use super::LoadedResources;
use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    ParseError,
    sys_at,
};

/// The longest header accepted, `safetensors`' own bound: 100 MB.
const MAX_HEADER_LEN: u64 = 100_000_000;

/// One tensor as a safetensors header describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsEntry {
    /// The element type, as the file spells it: `F16`, `F32`, `BF16`.
    pub dtype: String,

    /// The shape.
    pub shape: Vec<usize>,
}

/// The header of a safetensors file: every tensor's name, element type
/// and shape, read without touching the data.
///
/// This is how a kit learns a checkpoint's geometry before loading it;
/// [`SafetensorsCheckpoint::headers`] merges it across shards.
///
/// # Errors
/// An `io::Error` sorted by [`sys_at`], naming the path, for a file that
/// cannot be read: a [`Lookup`](BunsenErrorKind::Lookup) for one that is not
/// there. [`InvalidResource`](BunsenErrorKind::InvalidResource), with a
/// [`ParseError`], for a file that is not safetensors: one shorter than the
/// length prefix, a header that is not JSON or whose entries lack a dtype
/// or a shape; a header length past the file or past 100 MB is refused
/// before anything is allocated for it.
pub fn safetensors_header(path: &Path) -> BunsenResult<BTreeMap<String, SafetensorsEntry>> {
    let mut file = std::fs::File::open(path).map_err(sys_at("open", path))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            not_safetensors(
                path,
                "safetensors file",
                "shorter than its 8-byte length prefix",
            )
        } else {
            sys_at("read", path)(e)
        }
    })?;
    let len = u64::from_le_bytes(len);
    let file_len = file.metadata().map_err(sys_at("stat", path))?.len();
    if len > MAX_HEADER_LEN || len.saturating_add(8) > file_len {
        return Err(not_safetensors(
            path,
            "safetensors file",
            format!("a header of {len} bytes, in a file of {file_len}"),
        ));
    }
    let len = usize::try_from(len).map_err(|e| {
        BunsenError::internal(format!(
            "a header of {len} bytes, checked, does not fit usize"
        ))
        .with_cause(e)
    })?;
    let mut header = vec![0u8; len];
    file.read_exact(&mut header).map_err(sys_at("read", path))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|e| {
            BunsenError::from_cause(
                BunsenErrorKind::InvalidResource,
                ParseError::new("safetensors header")
                    .at(path.display())
                    .with_source(e),
            )
        })?;

    let mut entries = BTreeMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| {
                not_safetensors(path, "safetensors header", format!("{name}: no dtype"))
            })?
            .to_string();
        let shape = value
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| {
                not_safetensors(path, "safetensors header", format!("{name}: no shape"))
            })?
            .iter()
            .map(|d| {
                d.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| {
                        not_safetensors(
                            path,
                            "safetensors header",
                            format!("{name}: a shape that is not usize"),
                        )
                    })
            })
            .collect::<BunsenResult<Vec<usize>>>()?;
        entries.insert(name, SafetensorsEntry { dtype, shape });
    }
    Ok(entries)
}

/// An [`InvalidResource`](BunsenErrorKind::InvalidResource) error: the file
/// at `path` does not parse as `what`, for `reason`.
#[track_caller]
fn not_safetensors(
    path: &Path,
    what: &'static str,
    reason: impl core::fmt::Display,
) -> BunsenError {
    BunsenError::from_cause(
        BunsenErrorKind::InvalidResource,
        ParseError::new(what).at(path.display()).because(reason),
    )
}

/// `model.safetensors.index.json`: the shard each tensor is in.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct SafetensorsIndex {
    /// What the writer recorded: `total_size`, say.
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,

    /// Tensor name to shard file name.
    pub weight_map: BTreeMap<String, String>,
}

impl SafetensorsIndex {
    /// Reads an index file.
    ///
    /// # Errors
    /// An `io::Error` sorted by [`sys_at`], naming the path, for a file that
    /// cannot be opened;
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), with a
    /// [`ParseError`], for one that is not an index.
    pub fn read(path: &Path) -> BunsenResult<Self> {
        let file = std::fs::File::open(path).map_err(sys_at("open", path))?;
        serde_json::from_reader(std::io::BufReader::new(file)).map_err(|e| {
            BunsenError::from_cause(
                BunsenErrorKind::InvalidResource,
                ParseError::new("safetensors index")
                    .at(path.display())
                    .with_source(e),
            )
        })
    }

    /// The shard files the index names, each once, in name order.
    pub fn shards(&self) -> Vec<&str> {
        let set: BTreeSet<&str> = self.weight_map.values().map(String::as_str).collect();
        set.into_iter().collect()
    }
}

/// A checkpoint in safetensors: one file, or the shards of one.
///
/// `transformers` saves a model as `model.safetensors` until it passes a
/// shard-size limit, then as `model-00001-of-0000N.safetensors` and so on,
/// with `model.safetensors.index.json`, whose `weight_map`
/// ([`SafetensorsIndex`]) names the shard each tensor is in. A
/// `SafetensorsCheckpoint` is either: [`from_loaded`](Self::from_loaded)
/// gathers it from a [family](super::LoadedResources::family) of
/// resources under the kit's checkpoint key, the one file or the index and
/// the shards it names. That is the layout an
/// [`HfProvider`](super::HfProvider) row has.
///
/// A kit's [`Construct`](super::Construct) hook makes one from the
/// [`LoadedResources`] the cache brought local. [`headers`](Self::headers)
/// reads every tensor's name, element type and shape without touching the
/// data. [`load_into`](Self::load_into) loads every shard into a module
/// through `burn-store`'s [`SafetensorsStore`], configured by the kit (its
/// name remaps, its adapter), and checks that every parameter of the
/// module was found in some shard. A safetensors file is contiguous and
/// row-major, so the `PyTorch` adapter's `Linear` transposition is right
/// as it stands; the strided-view repair `OpenAI`'s `.pt` files need does
/// not apply. Whisper's hook reads `hf:` rows this way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsCheckpoint {
    /// The files, in shard order; one for a checkpoint that is one file.
    pub shards: Vec<PathBuf>,
}

/// What a load found: the tensors applied, and the file tensors no
/// parameter took.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SafetensorsApplied {
    /// The parameters applied, by the module's names, in name order.
    pub applied: Vec<String>,

    /// The tensors of the files no parameter took, by their file names.
    pub unused: Vec<String>,
}

impl SafetensorsCheckpoint {
    /// A checkpoint that is one file.
    pub fn single(path: impl Into<PathBuf>) -> Self {
        Self {
            shards: vec![path.into()],
        }
    }

    /// The checkpoint under `key` in `loaded`: the one file keyed `key`,
    /// or the index keyed `key.index` and the shards keyed `key.<n>`,
    /// which must be exactly the files the index names.
    ///
    /// # Errors
    /// As [`LoadedResources::expect`] when `loaded` has nothing under `key`,
    /// naming what it has; as [`SafetensorsIndex::read`] for the index.
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), under a frame
    /// naming the map, for a family with no index, or shards that are not
    /// the index's.
    pub fn from_loaded(
        loaded: &LoadedResources,
        key: &str,
    ) -> BunsenResult<Self> {
        if let Some(single) = loaded.get(key) {
            return Ok(Self::single(&single.path));
        }
        let family = loaded.family(key);
        if family.is_empty() {
            loaded.expect(key)?;
        }
        let index_key = format!("{key}.index");
        let Some((_, index)) = family.iter().find(|(k, _)| *k == index_key) else {
            return Err(BunsenError::invalid_resource(format!(
                "{} shards under {key:?} and no {index_key:?}",
                family.len()
            ))
            .context(&loaded.map.name));
        };
        let index = SafetensorsIndex::read(&index.path)?;
        let named: Vec<&str> = index.shards();
        let mut by_file: BTreeMap<&str, &Path> = BTreeMap::new();
        for (k, part) in &family {
            if *k == index_key {
                continue;
            }
            let file = part
                .path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or_default();
            by_file.insert(file, &part.path);
        }
        let have: Vec<&str> = by_file.keys().copied().collect();
        if have != named {
            let list = |files: &[&str]| {
                if files.is_empty() {
                    "none".to_string()
                } else {
                    files.join("\n  ")
                }
            };
            return Err(BunsenError::invalid_resource(
                "the map's shards are not the ones the index names",
            )
            .with_details(format!(
                "the index names:\n  {}\nthe map has:\n  {}",
                list(&named),
                list(&have)
            ))
            .context(&loaded.map.name));
        }
        Ok(Self {
            shards: named.iter().map(|f| by_file[f].to_path_buf()).collect(),
        })
    }

    /// The headers of every shard, merged: every tensor's name, element
    /// type and shape.
    ///
    /// # Errors
    /// As [`safetensors_header`];
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource) for a tensor two
    /// shards both hold.
    pub fn headers(&self) -> BunsenResult<BTreeMap<String, SafetensorsEntry>> {
        let mut merged = BTreeMap::new();
        for shard in &self.shards {
            for (name, entry) in safetensors_header(shard)? {
                if merged.insert(name.clone(), entry).is_some() {
                    return Err(BunsenError::invalid_resource(format!(
                        "{name} is in more than one shard"
                    ))
                    .context(shard.display()));
                }
            }
        }
        Ok(merged)
    }

    /// Loads every shard into `module`, each through a store `configure`
    /// sets up from the bare one (the kit's name remaps and adapter), and
    /// checks that every parameter of the module was found in some shard.
    ///
    /// # Errors
    /// Under a frame naming the shard, the store's error as the cause: an
    /// `io::Error` sorted by [`sys_at`] for a file that cannot be read;
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), with the
    /// [`SafetensorsStoreError`], for anything else the store refuses, a
    /// tensor whose shape does not fit its parameter say.
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource) naming the
    /// parameters no shard held, listed in the details.
    pub fn load_into<B: Backend, M: ModuleSnapshot<B>>(
        &self,
        module: &mut M,
        configure: impl Fn(SafetensorsStore) -> SafetensorsStore,
    ) -> BunsenResult<SafetensorsApplied> {
        let mut applied = BTreeSet::new();
        let mut unused = BTreeSet::new();
        let mut all: Option<BTreeSet<String>> = None;
        for shard in &self.shards {
            let mut store =
                configure(SafetensorsStore::from_file(shard.clone()).allow_partial(true));
            let result = module
                .load_from(&mut store)
                .map_err(|e| store_error(shard, e))?;
            if all.is_none() {
                let mut every: BTreeSet<String> = result.applied.iter().cloned().collect();
                every.extend(result.missing.iter().map(|(path, _)| path.clone()));
                every.extend(result.skipped.iter().cloned());
                all = Some(every);
            }
            applied.extend(result.applied.iter().cloned());
            unused.extend(result.unused.iter().cloned());
        }
        let never: Vec<String> = all
            .unwrap_or_default()
            .into_iter()
            .filter(|p| !applied.contains(p))
            .collect();
        if !never.is_empty() {
            let mut error = BunsenError::invalid_resource(format!(
                "the checkpoint lacks {} of the model's parameters: {}{}",
                never.len(),
                never
                    .iter()
                    .take(8)
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                if never.len() > 8 { ", ..." } else { "" }
            ))
            .with_details(never.join("\n"));
            if let Some(first) = self.shards.first() {
                error = error.context(first.display());
            }
            return Err(error);
        }
        Ok(SafetensorsApplied {
            applied: applied.into_iter().collect(),
            unused: unused.into_iter().collect(),
        })
    }
}

/// The store's error for `shard`, sorted: an `io::Error` by [`sys_at`];
/// the rest, a format the store refuses or a tensor that does not fit,
/// [`InvalidResource`](BunsenErrorKind::InvalidResource) with the store's
/// error as the cause.
#[track_caller]
fn store_error(
    shard: &Path,
    error: SafetensorsStoreError,
) -> BunsenError {
    match error {
        SafetensorsStoreError::Io(e) => sys_at("load", shard)(e),
        other => BunsenError::invalid_resource("the store refused the shard")
            .with_cause(other)
            .context(shard.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        LookupError,
        testing::ErrorMatcher,
    };

    /// Writes a safetensors file with `header` and `data`.
    fn write_safetensors(
        path: &Path,
        header: &str,
        data: &[u8],
    ) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes).unwrap();
    }

    /// The header is read without the data; a file that is not
    /// safetensors is refused, not allocated for.
    #[test]
    fn test_the_header_is_read_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        write_safetensors(
            &path,
            r#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"bias":{"dtype":"F16","shape":[2],"data_offsets":[24,28]}}"#,
            &[0u8; 28],
        );
        let header = safetensors_header(&path).unwrap();
        assert_eq!(header.keys().collect::<Vec<_>>(), ["bias", "weight"]);
        assert_eq!(
            header["weight"],
            SafetensorsEntry {
                dtype: "F32".to_string(),
                shape: vec![2, 3]
            }
        );
        assert_eq!(header["bias"].dtype, "F16");

        // "not a sa" as a little-endian length is enormous: refused, not
        // allocated for.
        let junk = dir.path().join("junk.safetensors");
        std::fs::write(&junk, b"not a safetensors file at all").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("cannot parse safetensors file")
            .has_cause::<ParseError>()
            .assert_err(&safetensors_header(&junk));

        // Shorter than the length prefix: not safetensors either.
        let short = dir.path().join("short.safetensors");
        std::fs::write(&short, b"abc").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("shorter than its 8-byte length prefix")
            .assert_err(&safetensors_header(&short));

        // Not there at all: a lookup of the path.
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .has_cause::<LookupError>()
            .assert_err(&safetensors_header(&dir.path().join("absent.safetensors")));
    }

    /// An index names its shards once each; a checkpoint from loaded
    /// resources is the one file, or the index's shards in its order, and
    /// shards that are not the index's are refused.
    #[test]
    fn test_a_checkpoint_from_loaded_resources() {
        use crate::data::pretrained::{
            Provenance,
            ResolvedResource,
            ResourceMap,
        };

        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            r#"{"metadata":{"total_size":48},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let index = SafetensorsIndex::read(&index_path).unwrap();
        assert_eq!(
            index.shards(),
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        assert_eq!(index.metadata["total_size"], 48);

        let shard = |name: &str, header: &str| {
            let path = dir.path().join(name);
            write_safetensors(&path, header, &[0u8; 24]);
            path
        };
        let s1 = shard(
            "model-00001-of-00002.safetensors",
            r#"{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
        );
        let s2 = shard(
            "model-00002-of-00002.safetensors",
            r#"{"b":{"dtype":"F32","shape":[6],"data_offsets":[0,24]}}"#,
        );
        let part = |path: &Path| ResolvedResource {
            path: path.to_path_buf(),
            provenance: Provenance::LocalDir,
        };
        let loaded = |parts: Vec<(&str, &Path)>| LoadedResources {
            map: ResourceMap::new("sharded"),
            parts: parts
                .into_iter()
                .map(|(k, p)| (k.to_string(), part(p)))
                .collect(),
        };

        let sharded = loaded(vec![
            ("checkpoint.index", &index_path),
            ("checkpoint.00002", &s2),
            ("checkpoint.00001", &s1),
            ("config", &index_path),
        ]);
        let checkpoint = SafetensorsCheckpoint::from_loaded(&sharded, "checkpoint").unwrap();
        assert_eq!(checkpoint.shards, [s1.clone(), s2.clone()]);
        let headers = checkpoint.headers().unwrap();
        assert_eq!(headers.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(headers["b"].shape, [6]);

        let single = loaded(vec![("checkpoint", &s1)]);
        assert_eq!(
            SafetensorsCheckpoint::from_loaded(&single, "checkpoint").unwrap(),
            SafetensorsCheckpoint::single(&s1)
        );

        let short = loaded(vec![
            ("checkpoint.index", &index_path),
            ("checkpoint.00001", &s1),
        ]);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_eq("the map's shards are not the ones the index names")
            .details_contains("model-00002-of-00002.safetensors")
            .frame_contains("sharded")
            .assert_err(&SafetensorsCheckpoint::from_loaded(&short, "checkpoint"));
        let no_index = loaded(vec![("checkpoint.00001", &s1)]);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("no \"checkpoint.index\"")
            .assert_err(&SafetensorsCheckpoint::from_loaded(&no_index, "checkpoint"));
        let nothing = loaded(vec![("config", &index_path)]);
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .has_cause::<LookupError>()
            .assert_err(&SafetensorsCheckpoint::from_loaded(&nothing, "checkpoint"));
    }
}
