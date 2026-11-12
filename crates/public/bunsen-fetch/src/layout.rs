//! The resource cache layout.
use std::path::{
    Path,
    PathBuf,
};

/// The root segment of the resource cache layout.
pub const RESOURCES_DIR: &str = "resources";

/// `resources/<name>/<pin>/<file>` under `root`: where bunsen's
/// `ResourceCache` keeps a resource, `pin` its SHA-256 when it is pinned,
/// so a cache rooted at `root` finds a file laid out here without knowing
/// where it came from. A catalog resource's name is its full name
/// (`openai/whisper/base/checkpoint`), each `/` a directory.
pub fn resource_cache_path(
    root: &Path,
    name: &str,
    pin: &str,
    file: &str,
) -> PathBuf {
    let mut path = root.join(RESOURCES_DIR);
    for segment in name.split('/') {
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
            resource_cache_path(Path::new("/r"), "pub/fam/build/checkpoint", "ff", "f.bin"),
            Path::new("/r/resources/pub/fam/build/checkpoint/ff/f.bin")
        );
    }
}
