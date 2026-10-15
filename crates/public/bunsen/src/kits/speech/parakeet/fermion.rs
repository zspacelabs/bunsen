//! The `fermion-five-value-parakeet-v1` container.
use std::path::Path;

use burn::{
    prelude::TensorData,
    store::{
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::f16,
};
use serde::Deserialize;

use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// The container format this reads.
pub const FERMION_FORMAT: &str = "fermion-five-value-parakeet-v1";

/// How one record is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FermionRecordKind {
    /// A matrix over a five-value alphabet per row, `{0, ±lo, ±hi}`: base-3
    /// sign codes packed five to a byte, one bit per non-zero choosing `hi`
    /// over `lo`, then the rows' `lo` and `hi` as fp16.
    FiveValue,

    /// Symmetric per-row integers of this many bits (6 or 8) times an fp16
    /// row scale. Six-bit values are packed four to three bytes, little-endian,
    /// offset by 32.
    Int(u8),

    /// fp16, as stored.
    Fp16,
}

impl FermionRecordKind {
    /// The kind a header spells `k`.
    fn parse(k: &str) -> Option<Self> {
        match k {
            "five_value" => Some(Self::FiveValue),
            "int6" => Some(Self::Int(6)),
            "int8" => Some(Self::Int(8)),
            "fp16" => Some(Self::Fp16),
            _ => None,
        }
    }

    /// Expands `blob`, a record of `shape`, to its f32 values in row-major
    /// order.
    fn decode(
        self,
        blob: &[u8],
        shape: &[usize],
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::FiveValue => decode_five_value(blob, shape),
            Self::Int(bits) => decode_int(blob, shape, bits),
            Self::Fp16 => decode_fp16(blob, shape),
        }
    }
}

/// One header entry.
#[derive(Debug, Deserialize)]
struct IndexEntry {
    /// The record's name.
    n: String,
    /// Its kind.
    k: String,
    /// Its shape.
    shape: Vec<usize>,
    /// Its byte length.
    b: usize,
}

/// The header.
#[derive(Debug, Deserialize)]
struct Header {
    format: String,
    index: Vec<IndexEntry>,
}

/// Reads a `fermion-five-value-parakeet-v1` container
/// ([`read_fermion_bytes`]) from a file.
///
/// # Errors
/// An `io::Error` sorted by [`sys_at`], naming the path, for a file that
/// cannot be read; otherwise as [`read_fermion_bytes`], under a frame naming
/// the path.
pub fn read_fermion_container(path: &Path) -> BunsenResult<Vec<PackTensor>> {
    let bytes = std::fs::read(path).map_err(sys_at("read", path))?;
    read_fermion_bytes(&bytes).map_err(|e| e.context(path.display()))
}

/// Reads a `fermion-five-value-parakeet-v1` container: the format
/// `FermionResearch`'s Phonon-2 ships its Parakeet TDT weights in.
///
/// The layout is an 8-byte little-endian header length, a JSON header
/// `{format, index: [{n, k, shape, b}]}`, and each record's `b` bytes in
/// index order. Every record is expanded to f32 exactly, as the format's
/// reference reader (`fermion_container.py`) expands it, and named as
/// `transformers`' `ParakeetForTDT` state dict names it: a five-value record
/// `m` is the weight `m.weight`, and the kernel-1 pointwise convolutions,
/// stored squeezed to `[out, in]`, get their `[out, in, 1]` back.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// bytes that are not such a container: a short or malformed header, another
/// format, an unknown record kind, a record whose length does not fit its
/// shape, or trailing bytes.
pub fn read_fermion_bytes(bytes: &[u8]) -> BunsenResult<Vec<PackTensor>> {
    let bad = |why: String| BunsenError::invalid_resource(format!("a fermion container: {why}"));
    let len = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
        .ok_or_else(|| bad("shorter than its length prefix".into()))?;
    let header = bytes
        .get(8..8usize.saturating_add(len))
        .ok_or_else(|| bad(format!("a header of {len} bytes past the end")))?;
    let header: Header = serde_json::from_slice(header)
        .map_err(|e| BunsenError::invalid_resource("a fermion container header").with_cause(e))?;
    if header.format != FERMION_FORMAT {
        return Err(bad(format!(
            "format `{}`, not `{FERMION_FORMAT}`",
            header.format
        )));
    }

    let mut offset = 8 + len;
    let mut tensors = Vec::with_capacity(header.index.len());
    for entry in header.index {
        let kind = FermionRecordKind::parse(&entry.k)
            .ok_or_else(|| bad(format!("`{}` has unknown kind `{}`", entry.n, entry.k)))?;
        let blob = bytes
            .get(offset..offset.saturating_add(entry.b))
            .ok_or_else(|| bad(format!("`{}` runs past the end", entry.n)))?;
        offset += entry.b;

        let values = kind
            .decode(blob, &entry.shape)
            .map_err(|why| bad(format!("`{}`: {why}", entry.n)))?;
        let (name, shape) = hf_name_and_shape(&entry.n, kind, entry.shape);
        tensors.push(bridge::from_data(
            TensorData::new(values, shape),
            name,
            None,
        ));
    }
    if offset != bytes.len() {
        return Err(bad(format!("{} trailing bytes", bytes.len() - offset)));
    }
    Ok(tensors)
}

/// The state-dict name and shape of a record.
fn hf_name_and_shape(
    name: &str,
    kind: FermionRecordKind,
    mut shape: Vec<usize>,
) -> (String, Vec<usize>) {
    let name = match kind {
        FermionRecordKind::FiveValue => format!("{name}.weight"),
        _ => name.to_string(),
    };
    if shape.len() == 2
        && (name.ends_with(".conv.pointwise_conv1.weight")
            || name.ends_with(".conv.pointwise_conv2.weight"))
    {
        shape.push(1);
    }
    (name, shape)
}

fn decode_fp16(
    blob: &[u8],
    shape: &[usize],
) -> Result<Vec<f32>, String> {
    let n: usize = shape.iter().product();
    if blob.len() != 2 * n {
        return Err(format!("{} bytes for {n} fp16 values", blob.len()));
    }
    Ok(fp16s(blob))
}

/// Little-endian fp16 values, widened.
fn fp16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f16::from_le_bytes(b).to_f32())
        .collect()
}

fn decode_int(
    blob: &[u8],
    shape: &[usize],
    bits: u8,
) -> Result<Vec<f32>, String> {
    let rows = *shape.first().ok_or("an integer record of rank 0")?;
    let total: usize = shape.iter().product();
    let cols = total / rows.max(1);
    let body_len = blob
        .len()
        .checked_sub(2 * rows)
        .ok_or("shorter than its row scales")?;
    let (body, scales) = blob.split_at(body_len);
    let scales = fp16s(scales);

    let q: Vec<i32> = match bits {
        8 => {
            if body.len() < total {
                return Err(format!("{} bytes for {total} int8 values", body.len()));
            }
            body[..total].iter().map(|&b| b as i8 as i32).collect()
        }
        6 => {
            if body.len() % 3 != 0 || body.len() / 3 * 4 < total {
                return Err(format!("{} bytes for {total} int6 values", body.len()));
            }
            body.as_chunks::<3>()
                .0
                .iter()
                .flat_map(|b| {
                    let packed = b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16;
                    [0, 6, 12, 18].map(|s| ((packed >> s) & 0x3f) as i32 - 32)
                })
                .take(total)
                .collect()
        }
        _ => return Err(format!("int{bits} is not a stored width")),
    };
    Ok(q.iter()
        .enumerate()
        .map(|(k, &q)| q as f32 * scales[k / cols])
        .collect())
}

fn decode_five_value(
    blob: &[u8],
    shape: &[usize],
) -> Result<Vec<f32>, String> {
    let &[rows, cols] = shape else {
        return Err(format!("a five-value record of shape {shape:?}"));
    };
    let row_bytes = cols.div_ceil(5);
    let codes_len = rows * row_bytes;
    let codes = blob.get(..codes_len).ok_or("shorter than its codes")?;

    // Sign codes: 0 is -1, 1 is 0, 2 is +1.
    let mut signs = Vec::with_capacity(rows * cols);
    for row in codes.chunks_exact(row_bytes) {
        let mut row_signs = Vec::with_capacity(row_bytes * 5);
        for &byte in row {
            let mut x = byte;
            for _ in 0..5 {
                row_signs.push((x % 3) as i8 - 1);
                x /= 3;
            }
        }
        signs.extend_from_slice(&row_signs[..cols]);
    }
    let nonzero = signs.iter().filter(|&&s| s != 0).count();
    let bits_len = nonzero.div_ceil(8);
    let expected = codes_len + bits_len + 4 * rows;
    if blob.len() != expected {
        return Err(format!("{} bytes, not {expected}", blob.len()));
    }
    let bits = &blob[codes_len..codes_len + bits_len];
    let lo = fp16s(&blob[codes_len + bits_len..codes_len + bits_len + 2 * rows]);
    let hi = fp16s(&blob[codes_len + bits_len + 2 * rows..]);

    let mut nz = 0;
    Ok(signs
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            if s == 0 {
                return 0.0;
            }
            let is_hi = bits[nz / 8] >> (nz % 8) & 1 == 1;
            nz += 1;
            let row = k / cols;
            s as f32 * if is_hi { hi[row] } else { lo[row] }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn testdata(file: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/parakeet")
            .join(file)
    }

    /// Every record kind decodes to what the format's reference reader
    /// (`fermion_container.py`) decodes it to, bit for bit.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_matches_the_reference_reader() {
        use burn::store::{
            ModuleStore,
            SafetensorsStore,
        };

        let tensors = read_fermion_container(&testdata("toy.fermion")).unwrap();
        let mut store = SafetensorsStore::from_file(testdata("toy_fermion.safetensors"));
        let expected = store.get_all_tensors().unwrap().clone();
        assert_eq!(tensors.len(), expected.len());

        for tensor in &tensors {
            let ours = bridge::to_data(tensor).unwrap();
            let theirs = bridge::to_data(&expected[&tensor.name]).unwrap();
            let mut theirs_shape = theirs.shape().to_vec();
            if tensor.name.contains("pointwise_conv") {
                theirs_shape.push(1);
            }
            // The store reads a rank-0 tensor back as `[1]`.
            if ours.shape().is_empty() {
                theirs_shape.retain(|&d| d != 1);
            }
            assert_eq!(ours.shape().to_vec(), theirs_shape, "{}", tensor.name);
            assert_eq!(
                ours.try_to_vec::<f32>().unwrap(),
                theirs.try_to_vec::<f32>().unwrap(),
                "{}",
                tensor.name
            );
        }
    }

    #[test]
    fn test_refuses_malformed_containers() {
        let good = std::fs::read(testdata("toy.fermion")).unwrap();
        assert!(read_fermion_bytes(&good).is_ok());
        assert!(read_fermion_bytes(&good[..4]).is_err());
        assert!(read_fermion_bytes(&good[..good.len() - 1]).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(read_fermion_bytes(&trailing).is_err());
        let mut other = good.clone();
        let at = good
            .windows(FERMION_FORMAT.len())
            .position(|w| w == FERMION_FORMAT.as_bytes())
            .unwrap();
        other[at + FERMION_FORMAT.len() - 1] = b'9';
        assert!(read_fermion_bytes(&other).is_err());
    }
}
