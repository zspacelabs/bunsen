//! # Slice Utils

use burn::tensor::Slice;

/// Resolve a slice to a list of indices.
pub fn slice_to_indices(
    slice: Slice,
    collection_size: usize,
) -> anyhow::Result<Vec<usize>> {
    let isize_size = collection_size as isize;

    let start: usize = {
        let mut start = slice.start;
        if start < 0 {
            start += isize_size;
        }
        assert!(
            start > 0 && start < isize_size,
            "Invalid slice: index out of range for size={collection_size}: {slice:?}"
        );
        start as usize
    };

    let end: usize = {
        let mut end = slice.end.unwrap_or(isize_size);
        if end < 0 {
            end += isize_size;
        }
        assert!(
            end > 0 && end < isize_size,
            "Invalid slice: index out of range for size={collection_size}: {slice:?}"
        );
        end as usize
    };

    let step = if (slice.step > 0) != (start > end) {
        -slice.step
    } else {
        slice.step
    };

    let mut indices = Vec::new();
    let mut idx = start;
    while idx < end {
        indices.push(idx);
        idx = (idx as isize + step) as usize;
    }

    Ok(indices)
}

#[cfg(test)]
mod tests {}
