//! The pretrained cache layout.
use std::path::{
    Path,
    PathBuf,
};

/// The root segment of the pretrained cache layout.
pub const PRETRAINED_DIR: &str = "pretrained";

/// `pretrained/<kit>/<namespace>/<pin>/<file>` under `root`: where bunsen's
/// `PretrainedCache` keeps a resource, `pin` its SHA-256 when it is pinned,
/// so a cache rooted at `root` finds a file laid out here without knowing
/// where it came from.
pub fn pretrained_cache_path(
    root: &Path,
    kit: &str,
    namespace: &str,
    pin: &str,
    file: &str,
) -> PathBuf {
    root.join(PRETRAINED_DIR)
        .join(kit)
        .join(namespace)
        .join(pin)
        .join(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout() {
        assert_eq!(
            pretrained_cache_path(Path::new("/r"), "k", "ns", "ff", "f.bin"),
            Path::new("/r/pretrained/k/ns/ff/f.bin")
        );
    }
}
