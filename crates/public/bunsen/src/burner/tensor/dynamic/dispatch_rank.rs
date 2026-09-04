use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Dynamic to static rank dispatch handler.
pub trait RankHandler: Sized {
    /// The output type of the static-rank handler.
    type Output;

    /// Call the static-rank handler.
    fn call<const R: usize>(self) -> BunsenResult<Self::Output>;

    /// Dynamic rank dispatch.
    ///
    /// Handles up to rank=12.
    fn dyn_call(
        self,
        rank: usize,
    ) -> BunsenResult<Self::Output> {
        dispatch_rank::<Self>(rank, self)
    }
}

/// Dynamic rank dispatch.
///
/// Handles up to rank=12.
fn dispatch_rank<H: RankHandler>(
    rank: usize,
    handler: H,
) -> BunsenResult<H::Output> {
    match rank {
        1 => handler.call::<1>(),
        2 => handler.call::<2>(),
        3 => handler.call::<3>(),
        4 => handler.call::<4>(),
        5 => handler.call::<5>(),
        6 => handler.call::<6>(),
        7 => handler.call::<7>(),
        8 => handler.call::<8>(),
        9 => handler.call::<9>(),
        10 => handler.call::<10>(),
        11 => handler.call::<11>(),
        12 => handler.call::<12>(),
        _ => Err(BunsenError::UnsupportedRank {
            msg: "unsupported rank".to_string(),
            rank,
        }),
    }
}
