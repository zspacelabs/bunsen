//! A build's checkpoint into a model: read, built, loaded, placed.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    module::Module,
    store::burn_pack::Tensor as PackTensor,
    tensor::{
        Device,
        FloatDType,
    },
};

use crate::{
    burner::module::ModuleInit,
    data::{
        formats::{
            LoadPolicy,
            NamedTensorsApplied,
            SafetensorsCheckpoint,
            read_pytorch_tensors,
            stored_float_dtype,
        },
        resources::{
            LoadedResourceSet,
            SAFETENSORS,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenResult,
        ResultContext,
    },
};

/// Whether `loaded`'s checkpoint under `key` is safetensors: one file of the
/// [`SAFETENSORS`] format (or, with no format, a `.safetensors` file), or the
/// shards of a sharded one (an index under `{key}.index`).
pub fn checkpoint_is_safetensors(
    loaded: &LoadedResourceSet,
    key: &str,
) -> bool {
    match loaded.get(key) {
        Some(part) => match loaded.map.get(key).and_then(|r| r.format.as_deref()) {
            Some(format) => format == SAFETENSORS || format == SAFETENSORS_INDEX,
            None => part.path.extension().is_some_and(|e| e == "safetensors"),
        },
        None => loaded.get(&format!("{key}.index")).is_some(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::data::resources::{
        Provenance,
        ResolvedResource,
        ResourceSet,
    };

    /// A safetensors file of one `F32` tensor of 2 under `name`.
    fn write_one(
        path: &Path,
        name: &str,
    ) {
        let header = format!(r#"{{"{name}":{{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}}}"#);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(path, bytes).unwrap();
    }

    fn loaded(parts: &[(&str, &Path)]) -> LoadedResourceSet {
        LoadedResourceSet {
            map: ResourceSet::new("toy"),
            parts: parts
                .iter()
                .map(|(k, p)| {
                    (
                        k.to_string(),
                        ResolvedResource {
                            path: p.to_path_buf(),
                            provenance: Provenance::LocalDir,
                        },
                    )
                })
                .collect(),
        }
    }

    /// One file, or the shards of an index, read as safetensors; a file of
    /// another name is a `PyTorch` state dict.
    #[test]
    fn test_one_file_or_shards() {
        let dir = tempfile::tempdir().unwrap();
        let one = dir.path().join("model.safetensors");
        write_one(&one, "w");
        let single = loaded(&[("checkpoint", &one)]);
        assert!(checkpoint_is_safetensors(&single, "checkpoint"));
        assert_eq!(checkpoint_tensors(&single, "checkpoint").unwrap().len(), 1);

        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &index,
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let (s1, s2) = (
            dir.path().join("model-00001-of-00002.safetensors"),
            dir.path().join("model-00002-of-00002.safetensors"),
        );
        write_one(&s1, "a");
        write_one(&s2, "b");
        let sharded = loaded(&[
            ("checkpoint.index", &index),
            ("checkpoint.00001", &s1),
            ("checkpoint.00002", &s2),
        ]);
        assert!(checkpoint_is_safetensors(&sharded, "checkpoint"));
        assert_eq!(checkpoint_tensors(&sharded, "checkpoint").unwrap().len(), 2);

        let pth = dir.path().join("model.pth");
        std::fs::write(&pth, b"not a pickle").unwrap();
        let state_dict = loaded(&[("checkpoint", &pth)]);
        assert!(!checkpoint_is_safetensors(&state_dict, "checkpoint"));
        crate::errors::testing::ErrorMatcher::new()
            .frame_contains("reading toy's checkpoint")
            .assert_err(&checkpoint_tensors(&state_dict, "checkpoint").map(|_| ()));
    }
}

/// The tensors of `loaded`'s checkpoint under `key`, as stored: safetensors
/// (one file, or shards under an index: [`SafetensorsCheckpoint::from_loaded`])
/// when [`checkpoint_is_safetensors`], else a `PyTorch` state dict.
///
/// # Errors
/// As the reader's, under a frame naming the checkpoint.
pub fn checkpoint_tensors(
    loaded: &LoadedResourceSet,
    key: &str,
) -> BunsenResult<Vec<PackTensor>> {
    let what = || format!("reading {}'s {key}", loaded.map.name);
    if checkpoint_is_safetensors(loaded, key) {
        SafetensorsCheckpoint::from_loaded(loaded, key)
            .and_then(|checkpoint| checkpoint.tensors())
            .with_context(what)
    } else {
        read_pytorch_tensors(loaded.expect(key)?).with_context(what)
    }
}

/// The model of `loaded`'s checkpoint under `key`, built from `config`,
/// loaded by `apply` (the kit's `load_{target}_tensors`) at the precision
/// the file stores, then placed by `policy`.
///
/// The shape for a model a kit builds in its own float and casts after: the
/// image kits, the speech front ends. [`load_checkpoint_as`] is the other
/// shape.
///
/// # Errors
/// As [`checkpoint_tensors`], the config's
/// [`try_init`](ModuleInit::try_init), `apply` (under a frame naming the
/// checkpoint), and [`LoadPolicy::place`].
pub fn load_checkpoint_placed<C, M>(
    loaded: &LoadedResourceSet,
    key: &str,
    config: &C,
    device: &Device,
    policy: LoadPolicy,
    apply: impl FnOnce(&mut M, Vec<PackTensor>) -> BunsenResult<NamedTensorsApplied>,
) -> BunsenResult<M>
where
    C: ModuleInit<M>,
    M: Module,
{
    let tensors = checkpoint_tensors(loaded, key)?;
    let mut model: M = config.try_init(device)?;
    apply(&mut model, tensors).with_context(|| format!("loading {}'s {key}", loaded.map.name))?;
    policy.place(model, device, |_| true)
}

/// The model of `loaded`'s checkpoint under `key`, built from `config` and
/// loaded by `apply` (the kit's `load_{target}_tensors`) directly in the
/// float `policy` resolves from the stored one, among the dtypes `accepts`
/// allows.
///
/// The shape for a large model, read once in its target dtype: the language
/// models, and the speech models with one inside. [`load_checkpoint_placed`]
/// is the other shape.
///
/// # Errors
/// As [`checkpoint_tensors`], [`LoadPolicy::resolve`], the config's
/// [`try_init`](ModuleInit::try_init), and `apply` (under a frame naming the
/// checkpoint).
pub fn load_checkpoint_as<C, M>(
    loaded: &LoadedResourceSet,
    key: &str,
    config: &C,
    device: &Device,
    policy: LoadPolicy,
    accepts: impl Fn(FloatDType) -> bool,
    apply: impl FnOnce(&mut M, Vec<PackTensor>, FloatDType) -> BunsenResult<NamedTensorsApplied>,
) -> BunsenResult<M>
where
    C: ModuleInit<M>,
    M: Module,
{
    let tensors = checkpoint_tensors(loaded, key)?;
    let stored = stored_float_dtype(&tensors).unwrap_or(FloatDType::F32);
    let dtype = policy.resolve(stored, device, accepts)?;
    let mut model: M = config.try_init(device)?;
    apply(&mut model, tensors, dtype)
        .with_context(|| format!("loading {}'s {key}", loaded.map.name))?;
    Ok(model)
}
