//! The Whisper stream driver: audio pushed in chunks, timed transcript out.
//!
//! What a caller touches lives here: the driver and its config, the stream
//! context, the emission policy and what it emits, the clock, the clamp
//! policies, the voice-activity filter and its regions, the bundle, and the
//! token layout. [`support`] holds the internals underneath, and nothing in
//! it is part of the driver's API.
//!
//! # Driver and context
//!
//! A transcription is split in two. The [`WhisperStreamDriver`] is the
//! shared, immutable half: the model and its token layout (a
//! [`WhisperBundle`] behind an `Arc`), the mel front end, the logit
//! filters, the decode settings, and the voice-activity model when one is
//! attached. A [`WhisperStreamContext`] is one stream in progress, and the
//! only stateful type here: the front end's carry, the frames not yet
//! decoded, the seek pointer, the committed transcript, the stream's clock,
//! its clamp policy and its voice-activity state.
//!
//! Stream state is injected, never owned by the model. The context holds
//! it, and the driver and the model are only read, so one driver serves any
//! number of streams in one process, each in a context of its own, and
//! [`advance_ready`] batches the decodes of many contexts through the one
//! model. The same split runs underneath:
//! [`SileroVad`](crate::kits::speech::silero_vad::SileroVad) and its
//! [`SileroVadContext`](crate::kits::speech::silero_vad::SileroVadContext),
//! the mel converter and its
//! [`PerceptiveAudioConversionContext`](crate::ops::signal::perceptive_audio::PerceptiveAudioConversionContext),
//! and the text decoder and the
//! [`TextDecoderCache`](crate::kits::speech::whisper::blocks::TextDecoderCache)
//! each decode opens.
//!
//! One driver serves three deployments, configured rather than forked:
//! server-batch offline inference; conservative real time, for a
//! programmatic consumer; and best-effort real time, for a human reading
//! during the utterance. The three [`EmissionPolicy`] presets are those
//! three.
//!
//! # Lifecycle
//!
//! 1. A [`WhisperBundle`]: from a name, through
//!    [`default_whisper_factory`](crate::kits::speech::whisper::pretrained::default_whisper_factory)
//!    and a [`PretrainedCache`](crate::data::pretrained::PretrainedCache), or
//!    over a model already in hand.
//! 2. A [`WhisperStreamDriverConfig`] (language, task, timestamps, search,
//!    [`EmissionPolicy`], fallback ladder), built into a driver over the bundle
//!    by [`init_from_bundle`](WhisperStreamDriverConfig::init_from_bundle).
//!    [`with_vad`](WhisperStreamDriver::with_vad) attaches a voice-activity
//!    model, which the real-time presets need.
//! 3. One context per stream, from
//!    [`new_context`](WhisperStreamDriver::new_context), given the stream's
//!    [`StreamClock`] and a [`StreamClampPolicy`].
//! 4. Samples in through [`write_read`](WhisperStreamContext::write_read),
//!    [`TranscriptEvent`]s out; [`end_read`](WhisperStreamContext::end_read)
//!    ends the stream and decodes what is left.
//!
//! # The `write_read` fold
//!
//! [`write_read`](WhisperStreamContext::write_read) is a fold over the
//! samples it is handed, in two halves.
//! [`write`](WhisperStreamContext::write) appends them to staging, moves
//! whole hops through the mel front end, offers the new frames to the clamp
//! policy and appends them to the frame ring, and runs the voice-activity
//! model over whole chunks, stepping its gate.
//! [`read`](WhisperStreamContext::read) runs every decode the emission
//! policy says is due, commits each, and then drafts if a draft is due. The
//! halves are separate so that a batch of streams can be written one at a
//! time and read together by [`advance_ready`].
//!
//! The hop alignment never reaches the caller. Staging holds whatever is
//! not yet a whole hop, so a sound card's 480- or 1024-sample buffers are
//! written as they come.
//!
//! No decode state crosses a window boundary: each decode opens its own
//! text-decoder cache. What carries from one window to the next is
//! host-side: the seek pointer, the prompt carry (the tail of the committed
//! transcript), the language once detected, and the clock. The ring holds
//! every frame from the seek pointer onward and nothing before it.
//!
//! # Emission: commit and draft
//!
//! The policy has two axes, [`DecodeTriggers`] (when a decode runs) and a
//! [`CommitRule`] (when its output is final), and the output has two
//! variants, [`Committed`](TranscriptEvent::Committed) and
//! [`Draft`](TranscriptEvent::Draft). **A `window_full` or `endpoint`
//! decode commits, per the commit rule; an `interval` decode drafts.** What
//! the three deployments differ by falls out of those two sentences.
//!
//! A draft covers the audio after the last commit and replaces the previous
//! draft whole, so there is no retraction protocol, no sequence numbers,
//! and never two drafts to reconcile.
//!
//! | Preset | Triggers | Commit rule | Drafts |
//! |---|---|---|---|
//! | [`offline`](EmissionPolicy::offline) | `window_full` | `Complete` | never |
//! | [`conservative`](EmissionPolicy::conservative) | `window_full`, `endpoint` | `LastTimestamp` | with timestamps on, a decode's unfinished tail |
//! | [`responsive`](EmissionPolicy::responsive) | `window_full`, `endpoint`, `interval` (600 ms) | `LastTimestamp` | as conservative, plus one per interval of speech |
//!
//! ## What a decode may touch
//!
//! A decode reads the context through `&self`: cutting its frames from the
//! ring, packaging them against the clamp policy's
//! [`reference`](StreamClampPolicy::reference), the model's forward, and
//! the prompt. Only a commit moves the stream (the seek pointer, the
//! transcript, the prompt carry); a draft moves nothing but its own
//! pacing. So a draft is the same decode as a commit, not a second code
//! path, and adding the `interval` trigger cannot change what is
//! committed: `responsive` commits exactly what `conservative` commits on
//! the same audio, which the context's tests pin. That holds on a driver
//! that detects the language too: only a commit fixes a stream's language,
//! and a draft before the first commit detects one for its own decode and
//! keeps nothing.
//!
//! # The stream clock
//!
//! Every context carries a [`StreamClock`]: sorted `(sample, time)` anchors
//! and a sample rate, mapping a sample index to media time. A bare stream
//! gets [`StreamClock::uniform`], one anchor at `(0, 0.0)`, which
//! reproduces exactly the arithmetic upstream does from its seek pointer.
//! Everything richer (a capture callback's timestamp, a container's
//! presentation time, a dropped buffer becoming a new anchor rather than a
//! permanent shift) is an addition to that, not a departure:
//! [`anchor_write_read`](WhisperStreamContext::anchor_write_read) anchors
//! the next sample before writing. Making the general case the only case
//! costs nothing, because the bare case *is* the general case with one
//! anchor.
//!
//! A segment's times are its frames put on the clock: timestamp token `i`
//! of a window that starts at sample `s` is `clock.time_at(s + i * grid)`,
//! where `grid` is the [`encoder_grid`](WhisperStreamDriver::encoder_grid),
//! 320 samples at 16 kHz.
//!
//! # The clamp policy
//!
//! Whisper floors its log-mels 8 dB below a reference maximum, and upstream
//! takes that maximum over the **whole clip** before cutting windows. A
//! stream cannot see the whole clip, so where the reference comes from is a
//! policy, and the policy is an injected object behind [`StreamClampPolicy`]
//! rather than a variant the driver understands. The context knows only
//! two points: every arriving frame is offered to the policy once
//! ([`observe`](StreamClampPolicy::observe), `&mut self`), and a window's
//! reference is asked for just before packaging
//! ([`reference`](StreamClampPolicy::reference), `&self`). That split is
//! forced, not chosen: a decode reads the context through `&self`, so the
//! call it makes on the way cannot mutate either, and packaging a window
//! for a draft leaves the policy as the commit will find it.
//!
//! Two implementations cover four choices of reference.
//! [`RunningMaxClamp`] in a context written the whole clip before its first
//! read is upstream's global reference; written as audio arrives, it is the
//! running one; in a context opened per region, it is the per-region one.
//! [`PerWindow`] floors each window against its own maximum, as
//! [`package_mels`](crate::kits::speech::whisper::blocks::WhisperFrontEndConfig::package_mels)
//! does. Packaging itself, and the clamp range, belong to the model's front
//! end
//! ([`package_window`](crate::kits::speech::whisper::blocks::WhisperFrontEndConfig::package_window));
//! a policy supplies only the reference.
//!
//! # Voice activity: regions as sub-streams
//!
//! Under a policy with the `endpoint` trigger, the context runs the
//! attached voice-activity model over every chunk, and a
//! [`VoiceActivityFilter`] (from the driver's [`VoiceActivityFilterConfig`])
//! turns its speech probabilities into [`SpeechRegion`]s. A region the
//! filter closes is padded, snapped outward onto the encoder grid so that
//! its edges fall on timestamp positions, and becomes a decode unit of its
//! own: its frames are cut from the ring, decoded a window at a time, and
//! committed with times off the parent stream's clock. That is
//! region-as-stream without a second stream object. Under `window_full` a
//! full window is then decoded only while speech is in progress inside it;
//! a full window of silence is skipped, not decoded, which is the gating
//! half of voice activity. A policy without `endpoint` ignores an attached
//! model.
//!
//! A caller that decodes regions as streams of their own, say from the
//! whole-clip [`speech_regions`](VoiceActivityFilterConfig::speech_regions),
//! opens a context per region with the region's
//! [`clock`](SpeechRegion::clock): the parent's, sliced, so the region's
//! times stay absolute.

pub mod support;

mod emission_policy;
mod speech_region;
mod stream_clamp_policy;
mod stream_clock;
mod transcript;
mod voice_activity_filter;
mod whisper_bundle;
mod whisper_stream_context;
mod whisper_stream_driver;
mod whisper_token_layout;

pub use emission_policy::*;
pub use speech_region::*;
pub use stream_clamp_policy::*;
pub use stream_clock::*;
pub use transcript::*;
pub use voice_activity_filter::*;
pub use whisper_bundle::*;
pub use whisper_stream_context::*;
pub use whisper_stream_driver::*;
pub use whisper_token_layout::*;
