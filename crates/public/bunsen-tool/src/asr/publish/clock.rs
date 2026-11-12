use std::time::{
    Duration,
    SystemTime,
};

/// How a stream's clock reads as times: from the start of the audio, or,
/// for the microphone, as the wall clock.
#[derive(Clone, Copy, Debug)]
pub enum Clock {
    /// Seconds from the start of the audio: a file's.
    Offset,

    /// The wall clock, the stream's 0 at `origin`: the microphone's.
    #[cfg_attr(not(feature = "live"), allow(dead_code))]
    Wall { origin: SystemTime },
}

#[cfg_attr(not(feature = "live"), allow(dead_code))]
impl Clock {
    /// The wall clock, the stream's 0 now.
    pub fn wall_now() -> Self {
        Clock::Wall {
            origin: SystemTime::now(),
        }
    }

    /// The wall clock, `seconds` of the stream's clock having passed now.
    pub fn wall_since(seconds: f64) -> Self {
        let now = SystemTime::now();
        Clock::Wall {
            origin: now
                .checked_sub(Duration::from_secs_f64(seconds.max(0.0)))
                .unwrap_or(now),
        }
    }
}
