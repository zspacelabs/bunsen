//! Chunked reductions of `[rows, n]` along `n`.

use alloc::vec::Vec;

use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
    },
    tensor::DType,
};

use crate::contracts::unpack_shape_contract;

/// The width of the chunks a row is reduced by: each chunk reduces in
/// parallel with the others, then the chunks' results reduce.
pub const ROW_CHUNK: usize = 256;

/// `[rows, n]` as `[rows, chunks, ROW_CHUNK]` in f32, padded with `-inf` to
/// whole chunks.
fn chunked(x: Tensor<2>) -> Tensor<3> {
    let [rows, n] = unpack_shape_contract!(["rows", "n"], &x.dims());
    let x = x.cast(DType::F32);
    let chunks = n.div_ceil(ROW_CHUNK);
    let x = pad_neg_inf(x, chunks * ROW_CHUNK);
    x.reshape([rows, chunks, ROW_CHUNK])
}

/// `[rows, n]` f32 padded with `-inf` to `[rows, width]`.
fn pad_neg_inf(
    x: Tensor<2>,
    width: usize,
) -> Tensor<2> {
    let [rows, n] = unpack_shape_contract!(["rows", "n"], &x.dims());
    if width == n {
        return x;
    }
    let pad = Tensor::<2>::full(
        [rows, width - n],
        f32::NEG_INFINITY,
        (&x.device(), DType::F32),
    );
    let x = Tensor::cat(vec![x, pad], 1);
    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["rows", "width"],
        &x.dims(),
        &[("rows", rows), ("width", width)]
    );
    x
}

/// Each row's argmax, `[rows, 1]`: as burn's `argmax(1)`, the first index
/// among equal maxima.
pub fn argmax_rows(x: Tensor<2>) -> Tensor<2, Int> {
    let [rows] = unpack_shape_contract!(["rows", "n"], &x.dims(), &["rows"]);
    // The first max within each chunk, then the first chunk among equal
    // chunk maxima: the row's first max.
    let (maxes, within) = chunked(x).max_dim_with_indices(2); // [rows, chunks, 1]
    let best = maxes.argmax(1); // [rows, 1, 1]
    let offset = within.gather(1, best.clone()).reshape([rows, 1]);
    best.reshape([rows, 1]).mul_scalar(ROW_CHUNK as i64) + offset
}

/// Each row's max, `[rows, 1]`, in f32.
pub fn max_rows(x: Tensor<2>) -> Tensor<2> {
    let [rows] = unpack_shape_contract!(["rows", "n"], &x.dims(), &["rows"]);
    chunked(x).max_dim(2).max_dim(1).reshape([rows, 1])
}

/// Each row's softmax normalizer as a softmax computes it: the max `m`, and
/// `sum(exp(x - m))`, each `[rows, 1]`, in f32. `exp(x - m) / sum` is then
/// any entry's probability as the whole row's softmax gives it, to the
/// rounding of the sum's order (by chunks, then across them).
pub fn softmax_normalizer_rows(x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let [rows] = unpack_shape_contract!(["rows", "n"], &x.dims(), &["rows"]);
    let c = chunked(x);
    let max = c.clone().max_dim(2).max_dim(1); // [rows, 1, 1]
    let sum = (c - max.clone()).exp().sum_dim(2).sum_dim(1);
    (max.reshape([rows, 1]), sum.reshape([rows, 1]))
}

/// Each row's `log(sum(exp(x)))`, `[rows, 1]`, in f32: the normalizer a log
/// probability subtracts, `log_softmax(x) = x - logsumexp(x)`.
pub fn logsumexp_rows(x: Tensor<2>) -> Tensor<2> {
    let (max, sum) = softmax_normalizer_rows(x);
    sum.log() + max
}

/// Each row's top `k` values (f32) and their indices, `[rows, k]`, best
/// first, as burn's `topk_with_indices(k, 1)`: [`topk_rows_host`]'s, on the
/// device.
///
/// # Panics
/// Unless `1 <= k <= n`.
pub fn topk_rows(
    x: Tensor<2>,
    k: usize,
) -> (Tensor<2>, Tensor<2, Int>) {
    let [rows] = unpack_shape_contract!(["rows", "n"], &x.dims(), &["rows"]);
    let device = x.device();
    let top = topk_rows_host(x, k);
    (
        Tensor::from_data(
            TensorData::new(top.values, [rows, k]),
            (&device, DType::F32),
        ),
        Tensor::from_data(TensorData::new(top.indices, [rows, k]), &device),
    )
}

/// A row-major `[rows, k]` top-k, on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTopK {
    /// Each row's top `k` values, best first, in f32.
    pub values: Vec<f32>,

    /// Their indices in the row.
    pub indices: Vec<i64>,
}

/// Each row's top `k` values and their indices, best first, read to the
/// host: as burn's `topk_with_indices(k, 1)`, ties to the first index.
///
/// The top `k` of a row lie in the `k` chunks (of [`ROW_CHUNK`]) with the
/// largest maxima, ties to the first chunk: a chunk outside them has `k`
/// chunks' maxima at or above its own, each a distinct value at least as
/// good. So the chunks' maxima are reduced on the device and read; those `k`
/// chunks are gathered on the device and read, `k * ROW_CHUNK` values; and
/// their top `k` selected on the host. Two reads, a few small kernels that
/// do not depend on `k`; burn's top-k reduction compiles a kernel for each
/// `k` and shape, seconds each on CUDA.
///
/// A row with fewer than `k` finite values fills the rest with its `-inf`s,
/// first index first. NaN, in the total order, ranks above every number.
///
/// # Panics
/// Unless `1 <= k <= n`.
pub fn topk_rows_host(
    x: Tensor<2>,
    k: usize,
) -> HostTopK {
    let [rows, n] = unpack_shape_contract!(["rows", "n"], &x.dims());
    assert!((1..=n).contains(&k), "the top {k} of {n}");
    let device = x.device();
    let c = chunked(x);
    let [chunks] = unpack_shape_contract!(
        ["rows", "chunks", "chunk"],
        &c.dims(),
        &["chunks"],
        &[("rows", rows), ("chunk", ROW_CHUNK)]
    );
    let maxima = read_f32(c.clone().max_dim(2));
    let s = k.min(chunks);
    let mut picked: Vec<i64> = Vec::with_capacity(rows * s);
    for row in maxima.chunks(chunks) {
        let mut best = select_top(row, s, |i| i);
        // In row order, so the gathered candidates keep the row's order.
        best.sort_unstable();
        picked.extend(best.into_iter().map(|i| i as i64));
    }
    let ids = Tensor::<2, Int>::from_data(TensorData::new(picked.clone(), [rows, s]), &device);
    let width = s * ROW_CHUNK;
    let candidates = read_f32(c.gather(1, ids.reshape([rows, s, 1]).expand([rows, s, ROW_CHUNK])));
    let mut top = HostTopK {
        values: Vec::with_capacity(rows * k),
        indices: Vec::with_capacity(rows * k),
    };
    for (row, chunk_ids) in candidates.chunks(width).zip(picked.chunks(s)) {
        // Candidate `j` is position `j % ROW_CHUNK` of the `j / ROW_CHUNK`th
        // picked chunk; padding sits past `n`, after every real `-inf`.
        let at = |j: usize| chunk_ids[j / ROW_CHUNK] as usize * ROW_CHUNK + j % ROW_CHUNK;
        for j in select_top(row, k, at) {
            top.values.push(row[j]);
            top.indices.push(at(j) as i64);
        }
    }
    top
}

/// The positions of `values`' top `k`, best first: by value in the total
/// order, then by `index(position)`, the lower first.
fn select_top(
    values: &[f32],
    k: usize,
    index: impl Fn(usize) -> usize,
) -> Vec<usize> {
    let order = |a: &usize, b: &usize| {
        values[*b]
            .total_cmp(&values[*a])
            .then(index(*a).cmp(&index(*b)))
    };
    let mut positions: Vec<usize> = (0..values.len()).collect();
    if k < positions.len() {
        positions.select_nth_unstable_by(k - 1, order);
        positions.truncate(k);
    }
    positions.sort_unstable_by(order);
    positions
}

/// A tensor's values, read to the host as f32.
fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .expect("f32 values read back")
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        TensorData,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    fn ids(t: Tensor<2, Int>) -> Vec<i64> {
        t.into_data().convert::<i64>().try_to_vec::<i64>().unwrap()
    }

    /// The chunked reductions are burn's whole-row ones, on rows that are
    /// not a whole number of chunks.
    #[test]
    fn test_reductions_are_the_whole_rows() {
        let device = cpu_device();
        let x: Tensor<2> = seeded_tensor(3, [3, 1000], Distribution::Default, &device);
        assert_eq!(ids(argmax_rows(x.clone())), ids(x.clone().argmax(1)));
        assert_tensors_close(
            &max_rows(x.clone()),
            &x.clone().max_dim(1),
            Tolerance::default(),
        );
        let max = x.clone().max_dim(1);
        let lse = (x.clone() - max.clone()).exp().sum_dim(1).log() + max;
        assert_tensors_close(&logsumexp_rows(x.clone()), &lse, Tolerance::default());
        let (v, i) = topk_rows(x.clone(), 6);
        let (rv, ri) = x.topk_with_indices(6, 1);
        assert_eq!(ids(i), ids(ri));
        assert_tensors_close(&v, &rv, Tolerance::default());
    }

    /// A row many chunks wide: its top 20 and top 64 are burn's, on the
    /// device and on the host.
    #[test]
    fn test_topk_of_a_vocabulary() {
        let device = cpu_device();
        let x: Tensor<2> = seeded_tensor(5, [2, 151_936], Distribution::Default, &device);
        for k in [20, 64] {
            let (v, i) = topk_rows(x.clone(), k);
            let (rv, ri) = x.clone().topk_with_indices(k, 1);
            assert_eq!(ids(i), ids(ri.clone()));
            assert_tensors_close(&v, &rv, Tolerance::default());
            let host = topk_rows_host(x.clone(), k);
            assert_eq!(host.indices, ids(ri));
        }
    }

    /// A `k` past the chunks takes every chunk: the whole row's top-k.
    #[test]
    fn test_topk_wider_than_the_chunks() {
        let device = cpu_device();
        let x: Tensor<2> = seeded_tensor(11, [2, 600], Distribution::Default, &device);
        let (_, ri) = x.clone().topk_with_indices(500, 1);
        assert_eq!(ids(topk_rows(x, 500).1), ids(ri));
    }

    /// Ties go to the first index, as burn's do, chunks' maxima included:
    /// of three chunks tied at their max, the first two are the top two.
    #[test]
    fn test_ties_take_the_first_index() {
        let device = cpu_device();
        let mut row = vec![0.0f32; 700];
        row[300] = 1.0;
        row[600] = 1.0;
        let x = Tensor::<2>::from_data(TensorData::new(row, [1, 700]), &device);
        assert_eq!(ids(argmax_rows(x.clone())), [300]);
        assert_eq!(ids(topk_rows(x, 2).1), [300, 600]);

        let mut row = vec![0.0f32; 768];
        row[5] = 1.0;
        row[260] = 1.0;
        row[700] = 1.0;
        let x = Tensor::<2>::from_data(TensorData::new(row, [1, 768]), &device);
        assert_eq!(ids(topk_rows(x.clone(), 2).1), [5, 260]);
        assert_eq!(ids(topk_rows(x, 4).1), [5, 260, 700, 0]);
    }

    /// A row with fewer finite values than `k` fills the rest with its
    /// `-inf`s, first index first, never a padding position past the row.
    #[test]
    fn test_topk_past_the_finite_values() {
        let device = cpu_device();
        let mut row = vec![f32::NEG_INFINITY; 1000];
        row[10] = 2.0;
        row[900] = 1.0;
        let x = Tensor::<2>::from_data(TensorData::new(row, [1, 1000]), &device);
        let (v, i) = topk_rows(x, 5);
        let v: Vec<f32> = v.into_data().try_into_vec_as::<f32>().unwrap();
        let i = ids(i);
        assert_eq!(&i[..2], [10, 900]);
        assert_eq!(&v[..2], [2.0, 1.0]);
        assert!(v[2..].iter().all(|v| *v == f32::NEG_INFINITY));
        assert_eq!(&i[2..], [0, 1, 2]);
    }

    /// A softmax entry from the normalizer is the whole row's softmax's.
    #[test]
    fn test_softmax_normalizer_is_the_softmax() {
        let device = cpu_device();
        let x: Tensor<2> = seeded_tensor(7, [2, 777], Distribution::Normal(0.0, 3.0), &device);
        let (max, sum) = softmax_normalizer_rows(x.clone());
        let probs = (x.clone() - max).exp() / sum;
        let rp = burn::tensor::activation::softmax(x, 1);
        assert_tensors_close(&probs, &rp, Tolerance::default());
    }
}
