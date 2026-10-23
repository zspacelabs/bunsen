# Stateless modules and streaming contexts

This chapter covers how bunsen keeps state that changes as data arrives: a
decoder's key/value cache, an audio front end's leftover samples, a
recurrent model's hidden state, a transcription in progress. It opens with
the principle behind all of them, then the pattern that follows from it, and
ends with the Whisper stream driver, the largest system built on it.

## The principle: state is injected

**Cache and stream state is injected, never owned by the model.** The rule
is defined in [`bunsen::ops`](bunsen::ops#ops-and-blocks): the model is
immutable once it is built or loaded, and the state of one stream or one
decode lives in a value the caller holds and passes in.

The payoff is that one model serves several streams, or several caches, in
the same process. A server transcribes many streams with one copy of the
weights. Several decodes run at once over one model, each with a cache of
its own. A batch of decodes from different streams runs through the model
together. None of that works if the model holds a stream's state in its own
fields, because then the model *is* the stream.

The rule also decides where code lives: a cache or a stream context is an
op, not a block, even when blocks use it
([Ops, blocks and burn extensions](./ops-and-blocks.md)).

## The pattern

Each stateful system splits three ways, as
[`bunsen::ops::signal`](bunsen::ops::signal#config-coefficients-stream-context)
spells out for the audio front ends:

1. A **config**, which builds the module.
2. An **immutable module**: the weights, or fixed tables such as an analysis
   window or a mel filterbank. Fixed tables are bare tensors, so they move
   with the module but are not recorded
   ([module design conventions](./conventions.md)).
3. A **context** or **cache** per stream or per decode, made by the caller
   and passed back in on every call.

The pairs in the tree:

- [`SlidingStft`](bunsen::ops::signal::SlidingStft) and
  [`SlidingStftContext`](bunsen::ops::signal::SlidingStftContext): an STFT
  over a stream of samples.
- [`PerceptiveAudioConverter`](bunsen::ops::signal::perceptive_audio::PerceptiveAudioConverter)
  and
  [`PerceptiveAudioConversionContext`](bunsen::ops::signal::perceptive_audio::PerceptiveAudioConversionContext):
  waveform to log-mel, Whisper's front end.
- [`CausalSelfAttention`](bunsen::blocks::transformers::attention::csa::CausalSelfAttention)
  and [`KVCache`](bunsen::ops::transformers::attention::KVCache): one
  multi-layer cache per decode, shared by every layer of the model
  ([decode modes](bunsen::blocks::transformers::attention#kv-cache-decode-modes)).
- [`SileroVad`](bunsen::kits::speech::silero_vad::SileroVad) and
  [`SileroVadContext`](bunsen::kits::speech::silero_vad::SileroVadContext):
  the recurrent state and the tail of the last chunk
  ([streams](bunsen::kits::speech::silero_vad::SileroVad#streams)).
- [`WhisperStreamDriver`](bunsen::kits::speech::whisper::driver::WhisperStreamDriver)
  and
  [`WhisperStreamContext`](bunsen::kits::speech::whisper::driver::WhisperStreamContext):
  a whole transcription, including the front-end and voice-activity
  contexts of one stream.

The speech kits'
[table of pairs](bunsen::kits::speech#stream-state-is-injected) adds
Whisper's text-decoder cache.

### Chunking must not be observable

A context's job is to carry what a stream has not finished with, so that
feeding audio in chunks gives the same result as one call over the whole
signal. With the log-mel context, for example, hop-aligned chunks produce
exactly the frames one call would. This guarantee shapes the APIs. Whisper's "keep 8 dB below the maximum"
clamp is a step the caller applies to a finished spectrogram, not an option
on the converter: a clamp relative to a maximum reduces over whatever it is
given, so inside a streaming converter it would make the chunking visible
([why](bunsen::ops::signal::perceptive_audio#dynamic-range-packaging-is-the-callers-step)).

### Constructor names vary

The pattern is consistent, but the names of the methods that open a context
are not, yet. A context comes from
[`SlidingStft::init_state`](bunsen::ops::signal::SlidingStft::init_state),
[`PerceptiveAudioConverter::new_context`](bunsen::ops::signal::perceptive_audio::PerceptiveAudioConverter::new_context),
[`WhisperStreamDriver::new_context`](bunsen::kits::speech::whisper::driver::WhisperStreamDriver::new_context),
[`KVCacheModel::new_kv_cache`](bunsen::ops::transformers::attention::KVCacheModel::new_kv_cache) (a model's),
[`KVCacheConfig::init`](bunsen::ops::transformers::attention::KVCacheConfig::init)
(which takes no device), or
[`SileroVadContextConfig::init`](bunsen::kits::speech::silero_vad::SileroVadContextConfig::init)
(which takes the model it serves). Look on the module first, then on a
context config.

## The Whisper stream driver

The Whisper driver turns audio written in chunks into timed transcript
events. It is the principle applied at the scale of a whole pipeline, and
its design is documented in [`whisper::driver`](bunsen::kits::speech::whisper::driver).

Most of it does not depend on Whisper. The buffering, the voice-activity
gate, the decision of when to decode and what to commit, the clock and the
batching live in [`asr_driver`](bunsen::kits::speech::asr_driver), and a
model plugs in as an [`AsrEngine`](bunsen::kits::speech::asr_driver::AsrEngine):
its streaming features, the decode of one window, and its text. Whisper is one
engine;
[`Qwen3AsrEngine`](bunsen::kits::speech::qwen3_asr::Qwen3AsrEngine) is
another, driven by a
[`Qwen3AsrStreamDriver`](bunsen::kits::speech::qwen3_asr::Qwen3AsrStreamDriver)
under the same policies. Qwen3-ASR has no timestamps, so it commits each
window whole.
[`ParakeetEngine`](bunsen::kits::speech::parakeet::ParakeetEngine) is a
third: its transducer times every token, so a window cut at its full length
commits up to a pause between words, and the rest is decoded again with the
audio after it.

### Driver and context

A transcription splits in two. The
[`WhisperStreamDriver`](bunsen::kits::speech::whisper::driver::WhisperStreamDriver)
is the shared, immutable half: the model and its token layout, the mel
front end, the logit filters, the decode settings, and the voice-activity
model if one is attached. A
[`WhisperStreamContext`](bunsen::kits::speech::whisper::driver::WhisperStreamContext)
is one stream in progress, and it is the only stateful type: the front
end's carry, the frames not yet decoded, the committed transcript, the
stream's clock, its level and its voice-activity state. One driver
serves any number of contexts, and
[`advance_ready`](bunsen::kits::speech::asr_driver::advance_ready)
batches the decodes of many contexts through the one model
([driver and context](bunsen::kits::speech::whisper::driver#driver-and-context)).

### One driver, three deployments

The driver serves three deployments, and it is configured for each, never
forked: server-batch offline inference; conservative real time, for a
program that consumes the transcript; and best-effort real time, for a
person reading along during the utterance. The three
[`EmissionPolicy`](bunsen::kits::speech::asr_driver::EmissionPolicy)
presets are those three deployments.

### The `write_read` fold

[`write_read`](bunsen::kits::speech::whisper::driver::WhisperStreamContext::write_read)
has two halves. `write` takes samples in any buffer size, moves whole hops
through the mel front end, and runs the voice-activity model. `read` runs
every decode the emission policy says is due. They are separate so that a
batch of streams can be written one at a time and then read together.

No decode state crosses a window boundary: each decode opens its own
text-decoder cache. What carries from one window to the next is host-side:
the seek pointer, the prompt carry, the detected language and the clock
([the fold](bunsen::kits::speech::whisper::driver#the-write_read-fold)).

### Emission: commit and draft

The policy has two axes: [`DecodeTriggers`](bunsen::kits::speech::asr_driver::DecodeTriggers),
which says when a decode runs, and a
[`CommitRule`](bunsen::kits::speech::asr_driver::CommitRule), which
says when its output is final. A decode triggered by a full window or by
the end of a speech region commits. A decode triggered by the interval
timer drafts.

Drafts are opt-in, per policy. `offline` never drafts. `conservative`
drafts only the unfinished tail of a timestamped decode, before that tail
is decoded again with more audio. `responsive` adds a draft every 600 ms of
speech. A consumer that wants final text only keeps the
[`Committed`](bunsen::kits::speech::asr_driver::TranscriptEvent::Committed)
events. A draft covers the audio after the last commit and replaces the
previous draft whole, so there is no retraction protocol.

A draft is the same decode as a commit, not a second code path. A decode
reads the context through `&self`, and only a commit moves the stream, so
`responsive` commits exactly what `conservative` commits on the same audio.
That holds with language detection too: only a commit fixes a stream's
language
([what a decode may touch](bunsen::kits::speech::whisper::driver#what-a-decode-may-touch)).
The [preset table](bunsen::kits::speech::whisper::driver#emission-commit-and-draft)
lists each preset's triggers, commit rule and drafts.

### The stream clock

Every context carries a
[`StreamClock`](bunsen::kits::speech::asr_driver::StreamClock): anchors
that map a sample index to media time. A bare stream gets one anchor at
zero, which reproduces upstream's arithmetic exactly. A capture timestamp, a
container's presentation time, or a dropped buffer is one more anchor. The
bare case is the general case with one anchor, so the general case costs
nothing extra
([the stream clock](bunsen::kits::speech::whisper::driver#the-stream-clock)).

### The level is the stream's policy

Whisper floors its log-mels a fixed distance below a reference maximum, and
upstream takes that maximum over the whole clip. A stream cannot see the
whole clip, so where the reference comes from is the stream's
[`LevelPolicy`](bunsen::kits::speech::asr_driver::LevelPolicy), given to
each context when it opens and shared by every model on the driver core.
The context offers every new frame to its level, and asks it for a reference
just before a window is packaged.
[`Running`](bunsen::kits::speech::asr_driver::LevelPolicy::Running)
gives upstream's whole-clip behaviour, a running maximum, or a per-region
one, depending on how the context is fed.
[`PerWindow`](bunsen::kits::speech::asr_driver::LevelPolicy::PerWindow)
floors each window on its own. What the level summarizes is the model's:
Whisper's is a maximum
([`MaxLevel`](bunsen::kits::speech::asr_driver::MaxLevel))
([the level](bunsen::kits::speech::whisper::driver#the-level-where-the-clamps-reference-comes-from)).

### Voice-activity regions as sub-streams

The real-time presets decode at the end of each speech region, so they need
a voice-activity model, attached with
[`with_vad`](bunsen::kits::speech::whisper::driver::WhisperStreamDriver::with_vad).
Each stream then runs Silero in a Silero context of its own. A
[`VoiceActivityFilterConfig`](bunsen::kits::speech::asr_driver::VoiceActivityFilterConfig)
turns Silero's probabilities into
[`SpeechRegion`](bunsen::kits::speech::asr_driver::SpeechRegion)s.
A closed region is decoded as a unit of its own, with times taken off the
parent stream's clock, and a full window of silence is skipped instead of
decoded. That gives a region the behaviour of a stream without a second
stream object
([regions as sub-streams](bunsen::kits::speech::whisper::driver#voice-activity-regions-as-sub-streams)).

### Speakers

A driver can carry a diarizer as well as a voice-activity model, attached
with
[`with_speakers`](bunsen::kits::speech::asr_driver::AsrStreamDriver::with_speakers).
The driver knows it only as a
[`SpeakerActivityModel`](bunsen::kits::speech::diarization::SpeakerActivityModel),
per-frame speaker probabilities, so any diarizer serves any engine. Each
stream runs it in a stream of its own, and every segment the driver emits is
attributed to a speaker by a
[`SpeakerAttribution`](bunsen::kits::speech::asr_driver::SpeakerAttribution)
rule, such as `NeMo`'s
[`MeanActivity`](bunsen::kits::speech::asr_driver::MeanActivity).

A diarizer finalizes a frame only after some look-ahead, so a unit is
decoded only once the diarizer has finalized all of its audio: its commits
are attributed from final activity, and an engine that reads the activity
sees all of it, however the audio was chunked. A draft is attributed from
what the diarizer has finalized so far. The engine sees the window's activity
in its decode request, and one that decodes windows whole cuts them at a
change of speaker first
([`speaker_cut`](bunsen::kits::speech::asr_driver::speaker_cut)), so a
window is one speaker's as far as the turns go. Turns on their own, without transcription, come from a
[`TurnFilter`](bunsen::kits::speech::diarization::TurnFilter) over the
diarizer's activity.

### Known gaps

- [`CommitRule::Agreement`](bunsen::kits::speech::asr_driver::CommitRule::Agreement)
  is not implemented yet: building a driver under it fails with
  an [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) error.
- The `interval` trigger drafts only while the voice-activity gate says
  speech is in progress, and that gate runs only under `endpoint`
  ([`DecodeTriggers`](bunsen::kits::speech::asr_driver::DecodeTriggers)),
  so a driver with `interval` and without `endpoint` fails to build.
