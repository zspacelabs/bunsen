//! Fetches the test audio too large to commit, each set behind its own
//! feature: `ami` is a 60 s excerpt of the AMI Meeting Corpus with its
//! speaker reference and transcript, from a pinned commit of
//! `NVIDIA/NeMo-Speech.cpp`'s test files.
//!
//! Fetching, pinning and the `BUNSEN_TEST_AUDIO_AMI_*` overrides follow
//! `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

/// The revision of `NVIDIA/NeMo-Speech.cpp` the AMI excerpt is fetched at.
const NEMO_SPEECH_CPP: &str = "https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/2bdfb03b38d5e3b0fa55aeb91a079a9f57385373/test_files/diar";

/// The AMI excerpt's files: (override variable, file, sha256, path variable).
const AMI: &[(&str, &str, &str, &str)] = &[
    (
        "BUNSEN_TEST_AUDIO_AMI_WAV",
        "ami_en2002d_2132.wav",
        "f00f92e53115a4a6724aec0ddaf9c675df6d5fabd3a489b43fffe56ba52a3ae9",
        "AMI_EN2002D_2132_WAV",
    ),
    (
        "BUNSEN_TEST_AUDIO_AMI_RTTM",
        "ami_en2002d_2132.rttm",
        "40751c9fc0bb934f741789ce3fafccd2c1f9d7c112585b8b40ac177a9de13b2d",
        "AMI_EN2002D_2132_RTTM",
    ),
    (
        "BUNSEN_TEST_AUDIO_AMI_JSON",
        "ami_en2002d_2132.json",
        "8a77aab95e65e301f3cb23cfd0a0641b3295a9da5686ff30f2ab12d0f282e229",
        "AMI_EN2002D_2132_JSON",
    ),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "ami")]
    for &(override_var, file, sha256, path_var) in AMI {
        let url = format!("{NEMO_SPEECH_CPP}/{file}");
        let asset = bunsen_fetch::Asset {
            override_var,
            url: &url,
            sha256,
        };
        let path = asset.resolve(&bunsen_fetch::out_dir().join(file));
        println!("cargo:rustc-env={path_var}={}", path.display());
    }
}
