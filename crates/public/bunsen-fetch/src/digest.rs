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

/// A reader that digests what passes through it, so a transfer is hashed as
/// it lands rather than read back afterwards.
pub struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R> HashingReader<R> {
    /// Wraps `inner`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    /// The lowercase hex SHA-256 of everything read so far.
    pub fn hex_digest(&self) -> String {
        self.hasher
            .clone()
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

/// The lowercase hex SHA-256 of everything `reader` yields.
///
/// # Errors
/// As `reader`'s reads.
pub fn sha256_of_reader(reader: impl Read) -> io::Result<String> {
    let mut reader = HashingReader::new(reader);
    io::copy(&mut reader, &mut io::sink())?;
    Ok(reader.hex_digest())
}

/// The lowercase hex SHA-256 of a file, read in chunks so a multi-gigabyte
/// file stays off the heap.
///
/// # Errors
/// When the file cannot be opened or read.
pub fn sha256_of(path: &Path) -> io::Result<String> {
    sha256_of_reader(fs::File::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `abc`'s digest, through a file and through the reader.
    #[test]
    fn test_digests() {
        const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_of_reader(&b"abc"[..]).unwrap(), ABC);
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        fs::write(&file, b"abc").unwrap();
        assert_eq!(sha256_of(&file).unwrap(), ABC);
        assert!(sha256_of(&dir.path().join("missing")).is_err());
    }
}
