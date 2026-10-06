//! Digests: SHA-256 over files and streams.

use std::{
    fs,
    io::{
        self,
        Read,
    },
    path::Path,
};

use sha2::{
    Digest,
    Sha256,
};

use crate::errors::{
    BunsenResult,
    DigestMismatch,
    DigestOrigin,
    sys_at,
};

/// The lowercase hex SHA-256 of a file.
///
/// # Errors
/// If the file cannot be read, sorted by its `io::Error` and naming the path
/// (see [`sys_at`]): [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a
/// missing or forbidden file, usually
/// [`Sys`](crate::errors::BunsenErrorKind::Sys) otherwise.
pub fn sha256_of(path: &Path) -> BunsenResult<String> {
    let file = fs::File::open(path).map_err(sys_at("open", path))?;
    let mut reader = HashingReader::new(file);
    io::copy(&mut reader, &mut io::sink()).map_err(sys_at("read", path))?;
    Ok(reader.hex_digest())
}

/// Checks a file at rest against a lowercase hex SHA-256.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource), with
/// a [`DigestMismatch`] cause whose origin is [`DigestOrigin::AtRest`], on a
/// mismatch; as [`sha256_of`] if the file cannot be read.
pub fn verify_sha256(
    path: &Path,
    sha256: &str,
) -> BunsenResult<()> {
    let found = sha256_of(path)?;
    if found == sha256 {
        Ok(())
    } else {
        Err(DigestMismatch::new(
            path.display().to_string(),
            DigestOrigin::AtRest,
            sha256,
            found,
        )
        .into())
    }
}

/// A reader that digests what passes through it, so a transfer is hashed as
/// it lands rather than read back afterwards.
pub struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> HashingReader<R> {
    /// Hashes everything read through `inner`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    /// The lowercase hex SHA-256 of everything read so far.
    pub fn hex_digest(self) -> String {
        self.hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(
        &mut self,
        buf: &mut [u8],
    ) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{
            Cursor,
            Read,
        },
    };

    use super::*;
    use crate::{
        data::cache::testing::ABC_SHA256,
        errors::{
            BunsenErrorKind,
            LookupError,
            LookupProblem,
            testing::{
                ErrorMatcher,
                predicate,
            },
        },
    };

    #[test]
    fn test_sha256_of_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc");
        fs::write(&path, b"abc").unwrap();

        assert_eq!(sha256_of(&path).unwrap(), ABC_SHA256);
        verify_sha256(&path, ABC_SHA256).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .cause(predicate("an at-rest mismatch", |m: &DigestMismatch| {
                m.origin == DigestOrigin::AtRest && m.found == ABC_SHA256
            }))
            .assert_err(&verify_sha256(&path, &"0".repeat(64)));
    }

    /// A missing file is a `Lookup` of its path.
    #[test]
    fn test_a_missing_file_is_a_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing");
        let missing = ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .cause(predicate("a missing path", |l: &LookupError| {
                l.problem == LookupProblem::Missing
            }));
        missing.assert_err(&sha256_of(&path));
        missing.assert_err(&verify_sha256(&path, ABC_SHA256));
    }

    /// The digest covers every byte however the reads are split, and the
    /// bytes come through untouched.
    #[test]
    fn test_hashing_reader_digests_what_passes_through() {
        let mut reader = HashingReader::new(Cursor::new(b"abc".to_vec()));
        let mut out = Vec::new();
        let mut one = [0u8; 1];
        loop {
            let n = reader.read(&mut one).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&one[..n]);
        }
        assert_eq!(out, b"abc");
        assert_eq!(reader.hex_digest(), ABC_SHA256);
    }
}
