//! bunsen's driver on the checkpoint: a take of a stock voice, end to end.
//!
//! A sampled take matches the release's in distribution, never code for
//! code (the draws are not torch's), so this holds the take to what any
//! take of the sentence must be: it ends on the end of audio, within a
//! plausible length, its samples bounded. Its decode speed is reported.

use std::time::Instant;

use bunsen::{
    kits::{
        speech::tts::zonos3::driver::{
            Zonos3Driver,
            Zonos3DriverConfig,
            Zonos3Request,
            Zonos3Stop,
        },
        support::tokens::BunsenTokenizer,
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;

/// The stock voice the take speaks in.
pub const VOICE: &str = "Savannah";

/// The take's text, normalized.
pub const TEXT: &str = "Hello, this is a short test of the speech synthesizer.";

mod zonos3_mini {
    use super::*;

    /// A take of [`TEXT`] in [`VOICE`]'s voice ends on the end of audio,
    /// between one and ten seconds long, every sample in `(-1, 1)`.
    #[test]
    #[serial_test::serial]
    fn speaks() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let tokenizer = BunsenTokenizer::from_file(&construct.tokenizer).unwrap();
        let config = Zonos3DriverConfig::from_generation(&construct.generation);
        let driver = Zonos3Driver::new(construct, tokenizer, config).unwrap();
        let request = Zonos3Request::new(TEXT, driver.voice_reference(VOICE).unwrap()).with_seed(1);

        let started = Instant::now();
        let codes = driver.generate_codes(&request).unwrap();
        let generating = started.elapsed();
        let started = Instant::now();
        let samples = driver.render(&codes).unwrap();
        let rendering = started.elapsed();

        let seconds = samples.len() as f64 / f64::from(driver.sample_rate());
        eprintln!(
            "{VOICE}: {} rows in {:.2} s ({:.1} rows/s), {} frames, {seconds:.2} s of audio \
             rendered in {:.2} s, stop {:?}",
            codes.rows,
            generating.as_secs_f64(),
            codes.rows as f64 / generating.as_secs_f64(),
            codes.frames.len(),
            rendering.as_secs_f64(),
            codes.stop,
        );
        assert_eq!(codes.stop, Zonos3Stop::EndOfAudio);
        assert!((1.0..10.0).contains(&seconds), "{seconds} s");
        assert!(samples.iter().all(|x| x.abs() < 1.0));
    }
}
