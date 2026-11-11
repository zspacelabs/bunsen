//! The pretrained cache layout.
use std::path::{
    Path,
    PathBuf,
};

/// The root segment of the pretrained cache layout.
pub const PRETRAINED_DIR: &str = "pretrained";

/// `pretrained/<namespace>/<pin>/<file>` under `root`: where bunsen's
/// `PretrainedCache` keeps a resource, `pin` its SHA-256 when it is pinned,
/// so a cache rooted at `root` finds a file laid out here without knowing
/// where it came from. A catalog resource's namespace is its full name
/// (`openai/whisper/base/checkpoint`), each `/` a directory.
pub fn pretrained_cache_path(
    root: &Path,
    namespace: &str,
    pin: &str,
    file: &str,
) -> PathBuf {
    let mut path = root.join(PRETRAINED_DIR);
    for segment in namespace.split('/') {
        path.push(segment);
    }
    path.join(pin).join(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout() {
        assert_eq!(
            pretrained_cache_path(Path::new("/r"), "pub/fam/build/checkpoint", "ff", "f.bin"),
            Path::new("/r/pretrained/pub/fam/build/checkpoint/ff/f.bin")
        );
    }
}
