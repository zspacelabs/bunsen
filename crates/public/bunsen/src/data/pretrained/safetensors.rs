//! # Safetensors checkpoints
//!
//! A checkpoint in safetensors is one file, or several: `transformers`
//! saves a model as `model.safetensors` until it passes a shard-size
//! limit, then as `model-00001-of-0000N.safetensors` and so on with
//! `model.safetensors.index.json`, whose `weight_map` names the shard each
//! tensor is in. [`SafetensorsCheckpoint`] is either, from a map's
//! [family](super::LoadedResources::family) of resources under the kit's
//! checkpoint key: the one file, or the index and the shards it names.
//!
//! [`safetensors_header`] reads a file's header alone: every tensor's
//! name, element type and shape, without touching the data, which is how
//! a kit learns a checkpoint's geometry before loading it.
//! [`SafetensorsCheckpoint::load_into`] loads every shard into a module
//! through `burn-store`'s [`SafetensorsStore`], configured by the kit (its
//! name remaps, its adapter), and checks that every parameter of the
//! module was found in some shard. A safetensors file is contiguous and
//! row-major, so the `PyTorch` adapter's `Linear` transposition is right as
//! it stands; the strided-view repair `OpenAI`'s `.pt` files need does not
//! apply.

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
};
use serde::Deserialize;

use super::LoadedResources;
use crate::errors::{
    BunsenError,
    BunsenResult,
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
/// # Errors
/// [`BunsenError::External`] for a file that cannot be read;
/// [`BunsenError::Invalid`] for a file that is not safetensors: a header
/// length past the file or past 100 MB is refused before anything is
/// allocated for it.
pub fn safetensors_header(path: &Path) -> BunsenResult<BTreeMap<String, SafetensorsEntry>> {
    let mut file = std::fs::File::open(path).map_err(BunsenError::external)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(BunsenError::external)?;
    let len = u64::from_le_bytes(len);
    let file_len = file.metadata().map_err(BunsenError::external)?.len();
    if len > MAX_HEADER_LEN || len.saturating_add(8) > file_len {
        return Err(BunsenError::Invalid(format!(
            "{}: not a safetensors file: a header of {len} bytes",
            path.display()
        )));
    }
    let len = usize::try_from(len).map_err(BunsenError::external)?;
    let mut header = vec![0u8; len];
    file.read_exact(&mut header)
        .map_err(BunsenError::external)?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|e| {
            BunsenError::Invalid(format!("{}: not a safetensors header: {e}", path.display()))
        })?;

    let mut entries = BTreeMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| BunsenError::Invalid(format!("{}: {name}: no dtype", path.display())))?
            .to_string();
        let shape = value
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| BunsenError::Invalid(format!("{}: {name}: no shape", path.display())))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| {
                        BunsenError::Invalid(format!(
                            "{}: {name}: a shape that is not usize",
                            path.display()
                        ))
                    })
            })
            .collect::<BunsenResult<Vec<usize>>>()?;
        entries.insert(name, SafetensorsEntry { dtype, shape });
    }
    Ok(entries)
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
    /// [`BunsenError::External`] for a file that cannot be read;
    /// [`BunsenError::Invalid`] for one that is not an index.
    pub fn read(path: &Path) -> BunsenResult<Self> {
        let file = std::fs::File::open(path).map_err(BunsenError::external)?;
        serde_json::from_reader(std::io::BufReader::new(file)).map_err(|e| {
            BunsenError::Invalid(format!("{}: not a safetensors index: {e}", path.display()))
        })
    }

    /// The shard files the index names, each once, in name order.
    pub fn shards(&self) -> Vec<&str> {
        let set: BTreeSet<&str> = self.weight_map.values().map(String::as_str).collect();
        set.into_iter().collect()
    }
}

/// A checkpoint in safetensors: one file, or the shards of one.
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
    /// [`BunsenError::ResourceNotFound`] when `loaded` has nothing under
    /// `key`, naming what it has; [`BunsenError::Invalid`] for a family
    /// with no index, or shards that are not the index's.
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
            return Err(BunsenError::Invalid(format!(
                "{}: {} shards under {key:?} and no {index_key:?}",
                loaded.map.name,
                family.len()
            )));
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
            return Err(BunsenError::Invalid(format!(
                "{}: the index names shards {} but the map has {}",
                loaded.map.name,
                named.join(", "),
                if have.is_empty() {
                    "none".to_string()
                } else {
                    have.join(", ")
                }
            )));
        }
        Ok(Self {
            shards: named.iter().map(|f| by_file[f].to_path_buf()).collect(),
        })
    }

    /// The headers of every shard, merged: every tensor's name, element
    /// type and shape.
    ///
    /// # Errors
    /// As [`safetensors_header`]; [`BunsenError::Invalid`] for a tensor
    /// two shards both hold.
    pub fn headers(&self) -> BunsenResult<BTreeMap<String, SafetensorsEntry>> {
        let mut merged = BTreeMap::new();
        for shard in &self.shards {
            for (name, entry) in safetensors_header(shard)? {
                if merged.insert(name.clone(), entry).is_some() {
                    return Err(BunsenError::Invalid(format!(
                        "{}: {name} is in more than one shard",
                        shard.display()
                    )));
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
    /// [`BunsenError::External`] from the store: a file that cannot be
    /// read, a tensor whose shape does not fit its parameter;
    /// [`BunsenError::Invalid`] naming the parameters no shard held.
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
                .map_err(|e| BunsenError::External(format!("{}: {e}", shard.display())))?;
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
            return Err(BunsenError::Invalid(format!(
                "{}: the checkpoint lacks {} of the model's parameters: {}",
                self.shards
                    .first()
                    .map(|s| s.display().to_string())
                    .unwrap_or_default(),
                never.len(),
                never
                    .iter()
                    .take(8)
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(SafetensorsApplied {
            applied: applied.into_iter().collect(),
            unused: unused.into_iter().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let err = safetensors_header(&junk).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("not a safetensors file")),
            "{err}"
        );
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
        let err = SafetensorsCheckpoint::from_loaded(&short, "checkpoint").unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("the index names shards")),
            "{err}"
        );
        let no_index = loaded(vec![("checkpoint.00001", &s1)]);
        let err = SafetensorsCheckpoint::from_loaded(&no_index, "checkpoint").unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("no \"checkpoint.index\"")),
            "{err}"
        );
        let nothing = loaded(vec![("config", &index_path)]);
        assert!(matches!(
            SafetensorsCheckpoint::from_loaded(&nothing, "checkpoint"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }
}
