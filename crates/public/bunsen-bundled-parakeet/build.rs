//! Fetches the Parakeet TDT assets.
//!
//! Three sets, each behind its own feature: `checkpoint` is
//! `FermionResearch`'s Phonon-2 archive, `base_checkpoint` is NVIDIA's
//! `parakeet-tdt-0.6b-v3` as `transformers` ships it, both laid out as a
//! bunsen pretrained cache, and `onnx_gen` is the `istupakov` ONNX export of
//! NVIDIA's `parakeet-tdt-0.6b-v3` that `parakeet-model-validation` compares
//! bunsen against: its encoder graph and its prediction-and-joint graph. The
//! export's feature-extractor graph (`nemo128.onnx`) is not generated:
//! `burn-onnx` 0.22 cannot infer its types (`ArgType::Shape has no DType`).
//!
//! As in `bunsen-bundled-whisper`, a build that silently downloads is a bad
//! neighbour, so:
//!
//! * A fetch only happens under the feature that needs the asset.
//! * Assets land in `OUT_DIR`, the one directory a build script may write to.
//! * Every asset is pinned to a SHA-256 and re-verified on each build. A cached
//!   copy that fails is deleted and re-fetched once.
//! * `PHONON_2_ARCHIVE`, `PARAKEET_SAFETENSORS`, `PARAKEET_TOKENIZER`,
//!   `PARAKEET_ONNX_ENCODER`, `PARAKEET_ONNX_ENCODER_DATA` and
//!   `PARAKEET_ONNX_DECODER_JOINT` point the build at local files instead, for
//!   working offline or against a different export. An override that is the
//!   pinned file is laid out where the download would have gone; one that is
//!   not is used where it is.

// Each feature uses its own constants and helpers; the rest are dead in that
// build. Enumerating `cfg` on every item would be noisier than this.
#![allow(unused)]

use std::{
    env,
    fs,
    io,
    io::Read,
    path::{
        Path,
        PathBuf,
    },
};

use sha2::{
    Digest,
    Sha256,
};

/// The export's repository, pinned to one revision so every URL names one
/// file forever.
const ONNX_BASE_URL: &str = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce";

/// One fetched asset: where it comes from, its pin, and its local name.
struct Asset {
    /// The environment variable naming a local file to use instead.
    override_var: &'static str,

    /// The file's name in the export's repository.
    remote: &'static str,

    /// SHA-256 of the file.
    sha256: &'static str,

    /// Local name under `OUT_DIR`. `ModelGen` derives the generated module's
    /// name from a graph's file name, so it is also the name `src/lib.rs`
    /// includes.
    local: &'static str,
}

/// The `FastConformer` encoder graph. Its weights are external, in
/// [`ENCODER_DATA`].
const ENCODER: Asset = Asset {
    override_var: "PARAKEET_ONNX_ENCODER",
    remote: "encoder-model.onnx",
    sha256: "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1",
    local: "parakeet_encoder.onnx",
};

/// The encoder's external weights. The graph names this file by its remote
/// name, relative to the graph's directory, so it keeps that name locally.
const ENCODER_DATA: Asset = Asset {
    override_var: "PARAKEET_ONNX_ENCODER_DATA",
    remote: "encoder-model.onnx.data",
    sha256: "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
    local: "encoder-model.onnx.data",
};

/// The TDT prediction network and joint, one step at a time.
const DECODER_JOINT: Asset = Asset {
    override_var: "PARAKEET_ONNX_DECODER_JOINT",
    remote: "decoder_joint-model.onnx",
    sha256: "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1",
    local: "parakeet_decoder_joint.onnx",
};

/// The root segment of the pretrained cache layout,
/// `pretrained/<kit>/<namespace>/<sha256>/<file>`.
const PRETRAINED_DIR: &str = "pretrained";

/// The Parakeet kit's segment of the layout: `PARAKEET_KIT` in bunsen.
const PARAKEET_KIT: &str = "parakeet";

/// A pretrained asset: an [`Asset`] fetched from its own URL into the cache
/// layout under `namespace`, and the compile-time variable naming its path.
struct Pretrained {
    asset: Asset,
    url: &'static str,
    namespace: &'static str,
    path_var: &'static str,
}

/// `FermionResearch/Phonon-2`'s archive, at the revision bunsen's
/// `fermion/phonon-2` row pins.
const PHONON_2: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PHONON_2_ARCHIVE",
        remote: "phonon-2.bps.tar.zst",
        sha256: "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e",
        local: "phonon-2.bps.tar.zst",
    },
    url: "https://huggingface.co/FermionResearch/Phonon-2/resolve/7e153e4054c0a10db6d47d6fff5c062d4c77c154/phonon-2.bps.tar.zst",
    namespace: "fermion",
    path_var: "PHONON_2_ARCHIVE_PATH",
};

/// `nvidia/parakeet-tdt-0.6b-v3`'s checkpoint, at the revision bunsen's
/// `nvidia/parakeet-tdt-0.6b-v3` row pins.
const BASE_SAFETENSORS: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PARAKEET_SAFETENSORS",
        remote: "model.safetensors",
        sha256: "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
        local: "model.safetensors",
    },
    url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/model.safetensors",
    namespace: "nvidia",
    path_var: "PARAKEET_SAFETENSORS_PATH",
};

/// `nvidia/parakeet-tdt-0.6b-v3`'s vocabulary.
const BASE_TOKENIZER: Pretrained = Pretrained {
    asset: Asset {
        override_var: "PARAKEET_TOKENIZER",
        remote: "tokenizer.json",
        sha256: "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
        local: "tokenizer.json",
    },
    url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/tokenizer.json",
    namespace: "nvidia",
    path_var: "PARAKEET_TOKENIZER_PATH",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for asset in [&ENCODER, &ENCODER_DATA, &DECODER_JOINT] {
        println!("cargo:rerun-if-env-changed={}", asset.override_var);
    }
    for p in [&PHONON_2, &BASE_SAFETENSORS, &BASE_TOKENIZER] {
        println!("cargo:rerun-if-env-changed={}", p.asset.override_var);
    }
    println!(
        "cargo:rustc-env=PARAKEET_CACHE_DIR={}",
        cache_dir().display()
    );

    #[cfg(feature = "checkpoint")]
    fetch_pretrained(&PHONON_2);

    #[cfg(feature = "base_checkpoint")]
    for p in [&BASE_SAFETENSORS, &BASE_TOKENIZER] {
        fetch_pretrained(p);
    }

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches a pretrained asset into the cache layout,
/// `pretrained/parakeet/<namespace>/<sha256>/<file>`, and names its path as
/// a compile-time variable.
fn fetch_pretrained(p: &Pretrained) {
    let dest = cache_dir()
        .join(PRETRAINED_DIR)
        .join(PARAKEET_KIT)
        .join(p.namespace)
        .join(p.asset.sha256)
        .join(p.asset.local);
    let path = resolve_to(&p.asset, p.url, &dest);
    println!("cargo:rustc-env={}={}", p.path_var, path.display());
}

/// Fetches the ONNX export and generates Rust models from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded: the encoder's alone are 2.4 GB.
    println!(
        "cargo:rustc-env=PARAKEET_ONNX_OUT_DIR={}",
        env::var("OUT_DIR").unwrap()
    );

    // The encoder graph reads its weights from beside itself, so they are
    // resolved first, into the same directory.
    // Both are real files there, never symlinks: `onnx-ir` canonicalizes the
    // data path and refuses one that resolves outside the graph's directory.
    let encoder_dir = cache_dir();
    let data = resolve_asset(&ENCODER_DATA);
    place_file(&data, &encoder_dir.join(ENCODER_DATA.local));

    for asset in [&ENCODER, &DECODER_JOINT] {
        let mut onnx = resolve_asset(asset);
        if asset.local == ENCODER.local {
            let dest = encoder_dir.join(ENCODER.local);
            place_file(&onnx, &dest);
            onnx = dest;
        }
        println!("cargo:rerun-if-changed={}", onnx.display());

        burn_onnx::ModelGen::new()
            .input(onnx.to_str().expect("asset path is UTF-8"))
            .out_dir("./")
            .run_from_script();
    }
}

/// Where fetched assets land: `OUT_DIR`.
fn cache_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"))
}

/// Returns a path to an asset, honouring its override or fetching it.
///
/// A caller-supplied file is deliberately **not** digest-pinned — the point of
/// the override is to try something else. One that happens to be the pinned
/// asset is laid out where the download would have gone; one that is not
/// stays where it is.
fn resolve_asset(asset: &Asset) -> PathBuf {
    let url = format!("{ONNX_BASE_URL}/{}", asset.remote);
    resolve_to(asset, &url, &cache_dir().join(asset.local))
}

/// [`resolve_asset`], from `url` to `dest`.
fn resolve_to(
    asset: &Asset,
    url: &str,
    dest: &Path,
) -> PathBuf {
    let dest = dest.to_path_buf();
    let override_var = asset.override_var;
    if let Some(local) = env::var_os(override_var) {
        let path = PathBuf::from(local);
        assert!(
            path.is_file(),
            "{override_var} is set but not a file: {}",
            path.display(),
        );
        return match digest_of(&path) {
            Ok(found) if found == asset.sha256 => {
                place(&path, &dest);
                dest
            }
            Ok(found) => {
                println!(
                    "cargo:warning={override_var}: {} has digest {found}, not {}; used in place",
                    path.display(),
                    asset.sha256,
                );
                path
            }
            Err(e) => {
                println!(
                    "cargo:warning={override_var}: cannot read {}: {e}; used in place",
                    path.display()
                );
                path
            }
        };
    }

    fetch_verified(url, asset.sha256, &dest)
}

/// Puts `src` at `dest`: a symlink where the platform has them, else a copy.
fn place(
    src: &Path,
    dest: &Path,
) {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("create the asset directory");
    }
    if dest.exists() || dest.symlink_metadata().is_ok() {
        fs::remove_file(dest).expect("replace the laid-out asset");
    }
    #[cfg(unix)]
    if std::os::unix::fs::symlink(src, dest).is_ok() {
        return;
    }
    fs::copy(src, dest)
        .unwrap_or_else(|e| panic!("placing {} at {}: {e}", src.display(), dest.display()));
}

/// Makes `dest` a real file with `src`'s contents: a hard link where the two
/// share a filesystem, else a copy. A no-op when `dest` already is `src`.
fn place_file(
    src: &Path,
    dest: &Path,
) {
    if src == dest && !dest.is_symlink() {
        return;
    }
    let src = fs::canonicalize(src).unwrap_or_else(|e| panic!("resolving {}: {e}", src.display()));
    let tmp = dest.with_extension("placing");
    let _ = fs::remove_file(&tmp);
    if fs::hard_link(&src, &tmp).is_err() {
        fs::copy(&src, &tmp)
            .unwrap_or_else(|e| panic!("copying {} to {}: {e}", src.display(), tmp.display()));
    }
    fs::rename(&tmp, dest)
        .unwrap_or_else(|e| panic!("placing {} at {}: {e}", src.display(), dest.display()));
}

/// Ensures `dest` holds the asset at `url` with the given digest.
///
/// A cache entry whose digest does not match is treated as corrupt: it is
/// removed and fetched once more, and a second failure is fatal.
fn fetch_verified(
    url: &str,
    sha256: &str,
    dest: &Path,
) -> PathBuf {
    if dest.is_file() {
        match digest_of(dest) {
            Ok(found) if found == sha256 => return dest.to_path_buf(),
            Ok(found) => {
                println!(
                    "cargo:warning=cached {} has digest {found}, refetching",
                    dest.display()
                );
                let _ = fs::remove_file(dest);
            }
            Err(e) => {
                println!(
                    "cargo:warning=cannot read cached {}: {e}, refetching",
                    dest.display()
                );
                let _ = fs::remove_file(dest);
            }
        }
    }

    println!("cargo:warning=fetching {url}");
    download(url, dest).unwrap_or_else(|e| panic!("fetching {url}: {e}"));

    let found = digest_of(dest).expect("digest the freshly fetched asset");
    assert_eq!(
        found, sha256,
        "{url} has digest {found}, expected {sha256}. The upstream asset \
         changed, or the transfer was corrupt.",
    );

    dest.to_path_buf()
}

/// Streams `url` to `dest`, via a temporary file so an interrupted transfer
/// cannot leave a truncated asset in the cache.
fn download(
    url: &str,
    dest: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("partial");

    let mut response = ureq::get(url).call()?;
    let mut reader = response.body_mut().as_reader();
    let mut file = fs::File::create(&tmp)?;
    io::copy(&mut reader, &mut file)?;
    drop(file);

    fs::rename(&tmp, dest)?;
    Ok(())
}

/// The lowercase hex SHA-256 of a file, read in chunks so a 2.4 GB asset stays
/// off the heap.
fn digest_of(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];

    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }

    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
