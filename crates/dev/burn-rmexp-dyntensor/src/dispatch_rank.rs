use bunsen::errors::BunsenError;

/// Dynamic to static rank dispatch handler.
pub trait RankHandler {
    type Output;

    /// Call the static-rank handler.
    fn call<const R: usize>(self) -> Result<Self::Output, BunsenError>;
}

/// Dynamic rank dispatch.
///
/// Handles up to rank=12.
pub fn dispatch_rank<H: RankHandler>(
    rank: usize,
    handler: H,
) -> Result<H::Output, BunsenError> {
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
