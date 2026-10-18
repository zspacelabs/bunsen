//! The `phonon-byteplane-tar-zstd-v1` archive.
use alloc::collections::BTreeMap;
use std::{
    io::Read,
    path::Path,
};

use serde::Deserialize;
use sha2::{
    Digest,
    Sha256,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// The archive format this reads.
pub const PHONON_ARCHIVE_FORMAT: &str = "phonon-byteplane-tar-zstd-v1";

/// Every name a manifest gives [`PHONON_ARCHIVE_FORMAT`]: its own, and
/// `sttg1a-byteplane-tar-zstd-v1`, which the Phonon-1 archives carry.
pub const PHONON_ARCHIVE_FORMATS: [&str; 2] =
    [PHONON_ARCHIVE_FORMAT, "sttg1a-byteplane-tar-zstd-v1"];

/// The manifest, the archive's first member.
pub const PHONON_ARCHIVE_MANIFEST: &str = "bps_manifest.json";

/// The suffix of a member stored byte-plane split.
const SPLIT_SUFFIX: &str = ".bps";

/// One manifest entry.
#[derive(Debug, Deserialize)]
struct ManifestFile {
    path: String,
    original_bytes: usize,
    original_sha256: String,
    #[serde(default)]
    transform: Option<SplitPlan>,
}

/// How a split member is put back together.
#[derive(Debug, Deserialize)]
struct SplitPlan {
    /// The bytes before the first tensor (the safetensors header), stored
    /// verbatim.
    base: usize,
    /// `[start, end, itemsize]` per split tensor, offsets past `base`.
    plan: Vec<[usize; 3]>,
    /// The bytes past `base`.
    payload_bytes: usize,
    original_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    release_format: String,
    files: Vec<ManifestFile>,
}

/// A `FermionResearch` Phonon archive (`*.bps.tar.zst`), unpacked in memory.
///
/// The release format of every Phonon model: a zstd-compressed tar whose
/// first member, [`PHONON_ARCHIVE_MANIFEST`], lists every other member with
/// its original length and SHA-256. A safetensors member may be stored
/// **byte-plane split** (named `<path>.bps`): each tensor's bytes regrouped by
/// position within an element, the header and padding verbatim, which
/// compresses better and is a transport transform only. Reading inverts the
/// split and checks every member against the manifest, so what a caller sees
/// is the original file, byte for byte.
#[derive(Debug)]
pub struct PhononArchive {
    members: BTreeMap<String, Vec<u8>>,
}

impl PhononArchive {
    /// Reads an archive from a file.
    ///
    /// # Errors
    /// An `io::Error` sorted by [`sys_at`] for a file that cannot be read;
    /// otherwise as [`from_compressed`](Self::from_compressed), under a frame
    /// naming the path.
    pub fn read(path: &Path) -> BunsenResult<Self> {
        let bytes = std::fs::read(path).map_err(sys_at("read", path))?;
        Self::from_compressed(&bytes).map_err(|e| e.context(path.display()))
    }

    /// Reads an archive from its compressed bytes.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for bytes that are not zstd or not a tar, a missing or malformed
    /// manifest, another format, a member the manifest does not list, a split
    /// member whose plan does not fit it, or a member whose length or digest
    /// is not the manifest's.
    pub fn from_compressed(bytes: &[u8]) -> BunsenResult<Self> {
        let bad = |why: String| BunsenError::invalid_resource(format!("a Phonon archive: {why}"));
        let mut tar_bytes = Vec::new();
        ruzstd::decoding::StreamingDecoder::new(bytes)
            .map_err(|e| BunsenError::invalid_resource("not a zstd stream").with_cause(e))?
            .read_to_end(&mut tar_bytes)
            .map_err(|e| BunsenError::invalid_resource("a corrupt zstd stream").with_cause(e))?;

        let mut stored: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut archive = tar::Archive::new(tar_bytes.as_slice());
        let entries = archive
            .entries()
            .map_err(|e| BunsenError::invalid_resource("not a tar archive").with_cause(e))?;
        for entry in entries {
            let mut entry = entry
                .map_err(|e| BunsenError::invalid_resource("a corrupt tar entry").with_cause(e))?;
            let name = entry
                .path()
                .map_err(|e| BunsenError::invalid_resource("a tar entry's path").with_cause(e))?
                .to_string_lossy()
                .into_owned();
            let mut data = Vec::new();
            entry
                .read_to_end(&mut data)
                .map_err(|e| BunsenError::invalid_resource("a corrupt tar entry").with_cause(e))?;
            stored.insert(name, data);
        }

        let manifest = stored
            .remove(PHONON_ARCHIVE_MANIFEST)
            .ok_or_else(|| bad(format!("no `{PHONON_ARCHIVE_MANIFEST}`")))?;
        let manifest: Manifest = serde_json::from_slice(&manifest).map_err(|e| {
            BunsenError::invalid_resource("a Phonon archive manifest").with_cause(e)
        })?;
        if !PHONON_ARCHIVE_FORMATS.contains(&manifest.release_format.as_str()) {
            return Err(bad(format!(
                "format `{}`, not `{PHONON_ARCHIVE_FORMAT}`",
                manifest.release_format
            )));
        }

        let mut members = BTreeMap::new();
        for file in manifest.files {
            let data = match &file.transform {
                Some(plan) => {
                    let split = stored
                        .remove(&format!("{}{SPLIT_SUFFIX}", file.path))
                        .ok_or_else(|| bad(format!("no `{}{SPLIT_SUFFIX}`", file.path)))?;
                    join_planes(&split, plan)
                        .map_err(|why| bad(format!("`{}`: {why}", file.path)))?
                }
                None => stored
                    .remove(&file.path)
                    .ok_or_else(|| bad(format!("no `{}`", file.path)))?,
            };
            if data.len() != file.original_bytes {
                return Err(bad(format!(
                    "`{}` is {} bytes, not {}",
                    file.path,
                    data.len(),
                    file.original_bytes
                )));
            }
            let digest: String = Sha256::digest(&data)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if digest != file.original_sha256 {
                return Err(bad(format!(
                    "`{}` has digest {digest}, not {}",
                    file.path, file.original_sha256
                )));
            }
            members.insert(file.path, data);
        }
        if let Some(extra) = stored.keys().next() {
            return Err(bad(format!("`{extra}` is not in the manifest")));
        }
        Ok(Self { members })
    }

    /// The members' paths, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    /// A member's bytes, if the archive has it.
    pub fn get(
        &self,
        path: &str,
    ) -> Option<&[u8]> {
        self.members.get(path).map(Vec::as_slice)
    }

    /// A member's bytes.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// naming the member, when the archive does not have it.
    pub fn expect(
        &self,
        path: &str,
    ) -> BunsenResult<&[u8]> {
        self.get(path).ok_or_else(|| {
            BunsenError::invalid_resource(format!("a Phonon archive without `{path}`"))
        })
    }

    /// Takes a member's bytes out of the archive.
    pub fn take(
        &mut self,
        path: &str,
    ) -> Option<Vec<u8>> {
        self.members.remove(path)
    }
}

/// Inverts the byte-plane split of one member.
fn join_planes(
    split: &[u8],
    plan: &SplitPlan,
) -> Result<Vec<u8>, String> {
    let base = plan.base;
    if split.len() < base || plan.original_bytes < base {
        return Err("shorter than its header".into());
    }
    let mut out = vec![0u8; plan.original_bytes];
    out[..base].copy_from_slice(&split[..base]);
    let (mut src, mut cursor) = (base, 0usize);
    let copy = |out: &mut Vec<u8>, at: usize, src: &mut usize, n: usize| -> Result<(), String> {
        let from = split
            .get(*src..*src + n)
            .ok_or("a plan past the stored bytes")?;
        out.get_mut(base + at..base + at + n)
            .ok_or("a plan past the original bytes")?
            .copy_from_slice(from);
        *src += n;
        Ok(())
    };
    for &[start, end, itemsize] in &plan.plan {
        if start < cursor || end < start || itemsize == 0 || !(end - start).is_multiple_of(itemsize)
        {
            return Err(format!(
                "a malformed plan entry [{start}, {end}, {itemsize}]"
            ));
        }
        if start > cursor {
            copy(&mut out, cursor, &mut src, start - cursor)?;
        }
        let n = end - start;
        let per = n / itemsize;
        let planes = split
            .get(src..src + n)
            .ok_or("a plan past the stored bytes")?;
        let dest = out
            .get_mut(base + start..base + end)
            .ok_or("a plan past the original bytes")?;
        for (i, plane) in planes.chunks_exact(per.max(1)).take(itemsize).enumerate() {
            for (j, &byte) in plane.iter().enumerate() {
                dest[j * itemsize + i] = byte;
            }
        }
        src += n;
        cursor = end;
    }
    if plan.payload_bytes < cursor {
        return Err("a plan past the payload".into());
    }
    copy(&mut out, cursor, &mut src, plan.payload_bytes - cursor)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn testdata(file: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/fermion")
            .join(file)
    }

    /// The fixture, written by the format's reference writer, reads back to
    /// the members the writer was given.
    #[test]
    fn test_reads_the_reference_writers_archive() {
        let archive = PhononArchive::read(&testdata("archive.bps.tar.zst")).unwrap();
        let expected: BTreeMap<String, String> =
            serde_json::from_str(&std::fs::read_to_string(testdata("archive.json")).unwrap())
                .unwrap();
        assert_eq!(
            archive.paths().collect::<Vec<_>>(),
            expected.keys().map(String::as_str).collect::<Vec<_>>()
        );
        for (path, digest) in &expected {
            let found: String = Sha256::digest(archive.expect(path).unwrap())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(&found, digest, "{path}");
        }
        assert!(archive.expect("missing.json").is_err());
    }

    #[test]
    fn test_refuses_what_is_not_an_archive() {
        assert!(PhononArchive::from_compressed(b"not zstd").is_err());
        let good = std::fs::read(testdata("archive.bps.tar.zst")).unwrap();
        assert!(PhononArchive::from_compressed(&good[..good.len() / 2]).is_err());
    }

    #[test]
    fn test_join_planes() {
        // A 2-byte header, then two u16 elements split into planes, then a
        // trailing byte.
        let split = [9, 9, 0x01, 0x03, 0x02, 0x04, 7];
        let plan = SplitPlan {
            base: 2,
            plan: vec![[0, 4, 2]],
            payload_bytes: 5,
            original_bytes: 7,
        };
        assert_eq!(
            join_planes(&split, &plan).unwrap(),
            [9, 9, 0x01, 0x02, 0x03, 0x04, 7]
        );
    }
}
