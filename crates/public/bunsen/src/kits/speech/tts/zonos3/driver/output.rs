//! The release's output stage: decoder audio to the served samples.

/// Decoder audio to the served samples, in place, as the release serves
/// them: a gain of `-headroom_db`, then a soft limiter above the knee at
/// `limit_dbfs`, `sign(x) (k + (1 - k) tanh((|x| - k) / (1 - k)))`, which
/// keeps every sample under 1. In f32, as the release computes it.
pub fn zonos3_output_stage(
    samples: &mut [f32],
    headroom_db: f64,
    limit_dbfs: f64,
) {
    let gain = 10f64.powf(-headroom_db / 20.0) as f32;
    let knee = 10f64.powf(limit_dbfs / 20.0) as f32;
    let width = 1.0 - knee;
    for x in samples {
        let y = *x * gain;
        *x = if y.abs() > knee {
            (knee + width * ((y.abs() - knee) / width).tanh()).copysign(y)
        } else {
            y
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::assert_scalar_close;

    #[test]
    fn test_output_stage() {
        // Expected: the release's formula in f64 (python3), at its settings.
        let mut samples = [0.5, 1.5, -2.0, 1.25];
        zonos3_output_stage(&mut samples, 3.0, -1.0);
        let expected = [0.353_973, 0.990_966, -0.999_986, 0.884_932];
        for (got, want) in samples.iter().zip(expected) {
            assert_scalar_close(*got, want, 1e-6);
        }
    }
}
