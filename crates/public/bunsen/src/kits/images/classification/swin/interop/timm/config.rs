//! The config conversion.

use alloc::format;

use crate::{
    blocks::transformers::models::vision_encoder::TokenPoolConfig,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::support::interop::timm::TimmModelConfigMirror,
};

/// What a Swin kit takes from a timm `config.json` beside the architecture:
/// the side of its square input, and its pool, if it states one.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a pool
/// the head doesn't build (`token`), or an input that isn't square or has
/// other than `in_channels` channels; `kit` names the kit in the message.
pub(crate) fn swin_timm_input(
    config: &TimmModelConfigMirror,
    kit: &str,
    in_channels: usize,
) -> BunsenResult<(usize, Option<TokenPoolConfig>)> {
    let architecture = config.architecture.as_str();
    let pool = match config.global_pool.as_deref() {
        Some(pool) => Some(
            TokenPoolConfig::from_timm(pool)
                .filter(|p| *p != TokenPoolConfig::Token)
                .ok_or_else(|| {
                    BunsenError::unsupported(format!(
                        "timm's {architecture} with a {pool:?} pool: the {kit} kit pools by avg, \
                         max or avgmax"
                    ))
                })?,
        ),
        None => None,
    };
    let [channels, height, width] = config.pretrained_cfg.input_size;
    if height != width || channels != in_channels {
        return Err(BunsenError::unsupported(format!(
            "timm's {architecture} at {channels}x{height}x{width}: the {kit} kit takes square \
             images of {in_channels} channels"
        )));
    }
    Ok((height, pool))
}
