//! Advance Conway's Game of Life.

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        Int,
        s,
    },
    tensor::Slice,
};

use crate::{
    kits::sims::conway::{
        ops::project_wrapped_toroidal_boarders,
        util::ConwayRules,
    },
    prelude::{
        TensorBoolOpExt,
        TensorOrderedOpExt,
    },
    support::range_util::{
        range_into,
        shift_range,
    },
};

static INNER_SLICE: Slice = Slice {
    start: 1,
    end: Some(-1),
    step: 1,
};

fn update_and_wrap<B, const R: usize, F>(
    state: Tensor<B, R, Bool>,
    f: F,
) -> Tensor<B, R, Bool>
where
    B: Backend,
    F: Fn(Tensor<B, R, Bool>) -> Tensor<B, R, Bool>,
{
    let inner_update = f(state.clone());

    let dirty_boarders = state.slice_assign([INNER_SLICE; R], inner_update);

    project_wrapped_toroidal_boarders(dirty_boarders)
}

/// Returns the next board.
///
/// # Arguments
///
/// - `state`: a `[H, W]` game state.
///
/// # Returns
/// - the `[H, W]` evolved interior state, with wrapped edges.
pub fn next_state_wrapped_2d<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    update_and_wrap(state, next_interior_2d)
}

/// Returns the interior board next-state.
///
/// # Arguments
/// - `state`: a `[H, W]` game state.
///
/// # Returns
/// - the `[H-2, W-2]` evolved interior state.
pub fn next_interior_2d<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    #[cfg(any(test, debug_assertions))]
    let [h, w] = crate::contracts::unpack_shape_contract!(["h", "w"], &state.dims());

    // [H-2, W-2]
    let is_live = state.clone().slice(s![1..-1, 1..-1]);

    // [H-2, W-2]
    let window_count = state
        .unfold::<3, _>(0, 3, 1)
        .unfold::<4, _>(1, 3, 1)
        .count_dims(&[2, 3])
        .squeeze_dims::<2>(&[2, 3]);

    let n_is_3 = window_count.clone().equal_elem(3);
    let n_is_4 = window_count.equal_elem(4);

    let inner = n_is_3.bool_or(n_is_4.bool_and(is_live));

    #[cfg(any(test, debug_assertions))]
    crate::contracts::assert_shape_contract_periodically!(
        ["h" - 2, "w" - 2],
        &inner.dims(),
        &[("h", h), ("w", w)],
    );

    // [H-2, W-2]
    inner
}

/// Returns the next board.
///
/// # Arguments
///
/// - `state`: a `[H, W, Z]` game state.
/// - `rules`: a ruleset.
///
/// # Returns
/// - the `[H, W, Z]` evolved interior state, with wrapped edges.
pub fn next_state_wrapped_3d<B: Backend>(
    state: Tensor<B, 3, Bool>,
    rules: &ConwayRules,
) -> Tensor<B, 3, Bool> {
    update_and_wrap(state, |state| next_interior_3d(state, rules))
}

/// Returns the interior board next-state.
///
/// # Arguments
///
/// - `state`: a `[H, W, Z]` game state.
/// - `rules`: the ruleset to use.
///
/// # Returns
/// - the `[H-2, W-2, Z-2]` evolved interior state.
pub fn next_interior_3d<B: Backend>(
    state: Tensor<B, 3, Bool>,
    rules: &ConwayRules,
) -> Tensor<B, 3, Bool> {
    #[cfg(debug_assertions)]
    let [h, w, z] = crate::contracts::unpack_shape_contract!(["h", "w", "z"], &state.dims());

    // [H-2, W-2, Z-2]
    let is_live = state.clone().slice(s![1..-1, 1..-1, 1..-1]);

    // [H-2, W-2, Z-2]
    let win_counts: Tensor<B, 3, Int> = state
        .clone()
        .unfold::<4, _>(0, 3, 1)
        .unfold::<5, _>(1, 3, 1)
        .unfold::<6, _>(2, 3, 1)
        .count_dims(&[3, 4, 5])
        .squeeze_dims::<3>(&[3, 4, 5]);

    let spawn_range = range_into(&rules.spawn);
    let keep_range = shift_range(range_into(&rules.keep), 1);

    let spawn_points = win_counts.clone().in_range_scalar(spawn_range);
    let keep_points = win_counts.in_range_scalar(keep_range);

    let update = spawn_points.mask_where(is_live, keep_points);

    #[cfg(debug_assertions)]
    crate::contracts::assert_shape_contract_periodically!(
        ["h" - 2, "w" - 2, "z" - 2],
        &update.dims(),
        &[("h", h), ("w", w), ("z", z)],
    );

    // [H-2, W-2, Z-2]
    update
}
