//! Reading a webdataset shard: a tar file of samples, each its own files.

use alloc::collections::BTreeMap;
use std::{
    fs::File,
    io::Read,
    path::Path,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// One sample of a webdataset shard: its key and its files, by field (a
/// member `n01440764_10026.jpg` is the field `jpg` of the sample
/// `n01440764_10026`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebDatasetSample {
    /// The sample's key: its members' names up to the first `.` of the last
    /// path segment.
    pub key: String,

    /// The sample's files, by field: what follows that `.` (`jpg`, `cls`).
    pub fields: BTreeMap<String, Vec<u8>>,
}

impl WebDatasetSample {
    /// The field `name`'s bytes, when the sample has it.
    pub fn field(
        &self,
        name: &str,
    ) -> Option<&[u8]> {
        self.fields.get(name).map(Vec::as_slice)
    }

    /// The field `name` as UTF-8 text, trimmed: a `.cls` class index, a
    /// `.txt` caption.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a sample without
    /// the field; [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for one that is not UTF-8.
    pub fn text(
        &self,
        name: &str,
    ) -> BunsenResult<&str> {
        let bytes = self.field(name).ok_or_else(|| {
            BunsenError::lookup(crate::errors::LookupError::missing("field", name))
                .context(format!("sample {}", self.key))
        })?;
        core::str::from_utf8(bytes)
            .map(str::trim)
            .map_err(|e| BunsenError::invalid_resource(format!("sample {}: {name}: {e}", self.key)))
    }
}

/// The samples of a webdataset shard, as webdataset groups them: consecutive
/// tar members of one key are one sample. Members whose field is not in
/// `fields` are skipped unread, so a reader of images and labels does not
/// hold a shard's other files; `None` keeps every field.
///
/// The shard is read once, in order, `limit` samples at most.
///
/// See the [errors convention](crate::errors#convention-try_x-and-x).
///
/// # Errors
/// The `io::Error` of a shard that cannot be opened or read;
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// one that is not a tar, or a member name without a field.
pub fn try_read_webdataset(
    path: &Path,
    fields: Option<&[&str]>,
    limit: Option<usize>,
) -> BunsenResult<Vec<WebDatasetSample>> {
    let file = File::open(path).map_err(sys_at("open webdataset shard", path))?;
    let mut archive = tar::Archive::new(file);
    let invalid = |e: std::io::Error| {
        BunsenError::invalid_resource(format!("{}: not a tar: {e}", path.display()))
    };
    let mut samples: Vec<WebDatasetSample> = Vec::new();
    for entry in archive.entries().map_err(invalid)? {
        let mut entry = entry.map_err(invalid)?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry
            .path()
            .map_err(invalid)?
            .to_string_lossy()
            .into_owned();
        let (dir, base) = name.rsplit_once('/').unwrap_or(("", &name));
        let (stem, field) = base.split_once('.').ok_or_else(|| {
            BunsenError::invalid_resource(format!(
                "{}: a member without a field: {name}",
                path.display()
            ))
        })?;
        let key = if dir.is_empty() {
            stem.to_string()
        } else {
            format!("{dir}/{stem}")
        };
        if samples.last().is_none_or(|s| s.key != key) {
            if limit.is_some_and(|n| samples.len() == n) {
                break;
            }
            samples.push(WebDatasetSample {
                key,
                fields: BTreeMap::new(),
            });
        }
        if fields.is_some_and(|keep| !keep.contains(&field)) {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(invalid)?;
        samples
            .last_mut()
            .expect("pushed above")
            .fields
            .insert(field.to_string(), bytes);
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    /// A tar of `members`, in order, at `path`.
    fn write_tar(
        path: &Path,
        members: &[(&str, &[u8])],
    ) {
        let mut builder = tar::Builder::new(File::create(path).unwrap());
        for (name, bytes) in members {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *bytes).unwrap();
        }
        builder.finish().unwrap();
    }

    /// Members group by key, in order; a field not kept is skipped; the
    /// limit stops the read.
    #[test]
    fn test_samples_group_by_key() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("shard.tar");
        write_tar(
            &shard,
            &[
                ("a.jpg", b"A"),
                ("a.cls", b"7\n"),
                ("a.json", b"{}"),
                ("sub/b.jpg", b"B"),
                ("sub/b.cls", b"8"),
                ("c.jpg", b"C"),
            ],
        );
        let all = try_read_webdataset(&shard, None, None).unwrap();
        assert_eq!(
            all.iter().map(|s| s.key.as_str()).collect::<Vec<_>>(),
            ["a", "sub/b", "c"]
        );
        assert_eq!(all[0].fields.len(), 3);

        let kept = try_read_webdataset(&shard, Some(&["jpg", "cls"]), Some(2)).unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].field("jpg"), Some(&b"A"[..]));
        assert_eq!(kept[0].field("json"), None);
        assert_eq!(kept[0].text("cls").unwrap(), "7");
        assert_eq!(kept[1].text("cls").unwrap(), "8");
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .frame_contains("sample a")
            .assert_err(&kept[0].text("txt"));
    }

    #[test]
    fn test_a_bad_shard_is_an_invalid_resource() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("shard.tar");
        write_tar(&shard, &[("noext", b"x")]);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("a member without a field: noext")
            .assert_err(&try_read_webdataset(&shard, None, None));
    }
}
