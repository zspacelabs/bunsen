//! # Digests
//!
//! SHA-256 over files and streams, for digest-pinned cache entries: a file
//! whose path carries its digest was verified when it was written, and is
//! trusted on later runs without re-hashing it.

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
    BunsenError,
    BunsenResult,
};

/// The lowercase hex SHA-256 of a file.
///
/// # Errors
/// [`BunsenError::External`] if the file cannot be read.
pub fn sha256_of(path: &Path) -> BunsenResult<String> {
    let file = fs::File::open(path).map_err(BunsenError::external)?;
    let mut reader = HashingReader::new(file);
    io::copy(&mut reader, &mut io::sink()).map_err(BunsenError::external)?;
    Ok(reader.hex_digest())
}

/// Checks a file against a lowercase hex SHA-256.
///
/// # Errors
/// [`BunsenError::Invalid`] on a mismatch, [`BunsenError::External`] if the
/// file cannot be read.
pub fn verify_sha256(
    path: &Path,
    sha256: &str,
) -> BunsenResult<()> {
    let found = sha256_of(path)?;
    if found == sha256 {
        Ok(())
    } else {
        Err(BunsenError::Invalid(format!(
            "{}: sha256 is {found}, expected {sha256}",
            path.display()
        )))
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
    use crate::data::cache::testing::ABC_SHA256;

    #[test]
    fn test_sha256_of_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc");
        fs::write(&path, b"abc").unwrap();

        assert_eq!(sha256_of(&path).unwrap(), ABC_SHA256);
        verify_sha256(&path, ABC_SHA256).unwrap();
        assert!(matches!(
            verify_sha256(&path, &"0".repeat(64)),
            Err(BunsenError::Invalid(_))
        ));
    }

    #[test]
    fn test_a_missing_file_is_external() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing");
        assert!(matches!(sha256_of(&path), Err(BunsenError::External(_))));
        assert!(matches!(
            verify_sha256(&path, ABC_SHA256),
            Err(BunsenError::External(_))
        ));
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
