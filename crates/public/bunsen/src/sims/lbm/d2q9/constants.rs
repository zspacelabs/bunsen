//! The lattice's constants.

use crate::support::math::FRAC_1_SQRT_3;

/// The speed of sound.
pub const SPEED_OF_SOUND: f64 = FRAC_1_SQRT_3;

/// The speed of sound squared.
pub const C2: f64 = SPEED_OF_SOUND * SPEED_OF_SOUND;

/// The speed of sound cubed.
pub const C4: f64 = C2 * C2;
