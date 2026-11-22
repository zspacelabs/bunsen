//! # Mel filterbanks
//!
//! [`MelScale`]: the frequency-to-mel curves (Slaney, HTK, Kaldi), each with
//! its inverse. The banks over an FFT's bins, built as host tables:
//! [`MelFilterbankConfig`], `librosa`'s (triangles linear in Hz between mel
//! points, optionally area-normalized, [`FilterNorm`]), and
//! [`KaldiMelBanksConfig`], Kaldi's (triangles linear in mels).

mod filterbank;
mod kaldi;

pub use filterbank::*;
pub use kaldi::*;
