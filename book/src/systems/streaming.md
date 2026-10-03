# Stateless modules and streaming contexts

This chapter covers how bunsen keeps state that changes as data arrives: a
decoder's key/value cache, an audio front end's leftover samples, a
recurrent model's hidden state, a transcription in progress. It opens with
the principle behind all of them, then the pattern that follows from it, and
ends with the Whisper stream driver, the largest system built on it.

## The principle: state is injected

**Cache and stream state is injected, never owned by the model.** A model
is immutable once it is built or loaded. The state of one stream or one
decode lives in a separate value that the caller creates, passes in on
every call, and drops or resets when it is done.

The payoff is that one model serves several streams, or several caches, in
the same process. A server transcribes many streams with one copy of the
weights. Several decodes run at once over one model, each with a cache of
its own. A batch of decodes from different streams runs through the model
together. None of that works if the model holds a stream's state in its own
fields, because then the model *is* the stream.

The rule is stated in
[STYLE.md](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#injected-state).
It also decides where code lives: a cache or a stream context is an op,
not a block, even when blocks use it
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
[`NanoChatGpt::new_kv_cache`](bunsen::kits::gpts::nanochat::NanoChatGpt::new_kv_cache),
[`KVCacheConfig::init`](bunsen::ops::transformers::attention::KVCacheConfig::init)
(which takes no device), or
[`SileroVadContextConfig::init`](bunsen::kits::speech::silero_vad::SileroVadContextConfig::init)
(which takes the model it serves). Look on the module first, then on a
context config.

## The Whisper stream driver

The Whisper driver turns audio written in chunks into timed transcript
events. It is the principle applied at the scale of a whole pipeline, and
its design is documented in [`whisper::driver`](bunsen::kits::speech::whisper::driver).

### Driver and context

A transcription splits in two. The
[`WhisperStreamDriver`](bunsen::kits::speech::whisper::driver::WhisperStreamDriver)
is the shared, immutable half: the model and its token layout, the mel
front end, the logit filters, the decode settings, and the voice-activity
model if one is attached. A
[`WhisperStreamContext`](bunsen::kits::speech::whisper::driver::WhisperStreamContext)
is one stream in progress, and it is the only stateful type: the front
end's carry, the frames not yet decoded, the committed transcript, the
stream's clock, its clamp policy and its voice-activity state. One driver
serves any number of contexts, and
[`advance_ready`](bunsen::kits::speech::whisper::driver::advance_ready)
batches the decodes of many contexts through the one model
([driver and context](bunsen::kits::speech::whisper::driver#driver-and-context)).

### One driver, three deployments

The driver serves three deployments, and it is configured for each, never
forked: server-batch offline inference; conservative real time, for a
program that consumes the transcript; and best-effort real time, for a
person reading along during the utterance. The three
[`EmissionPolicy`](bunsen::kits::speech::whisper::driver::EmissionPolicy)
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

The policy has two axes: [`DecodeTriggers`](bunsen::kits::speech::whisper::driver::DecodeTriggers),
which says when a decode runs, and a
[`CommitRule`](bunsen::kits::speech::whisper::driver::CommitRule), which
says when its output is final. A decode triggered by a full window or by
the end of a speech region commits. A decode triggered by the interval
timer drafts.

Drafts are opt-in, per policy. `offline` never drafts. `conservative`
drafts only the unfinished tail of a timestamped decode, before that tail
is decoded again with more audio. `responsive` adds a draft every 600 ms of
speech. A consumer that wants final text only keeps the
[`Committed`](bunsen::kits::speech::whisper::driver::TranscriptEvent::Committed)
events. A draft covers the audio after the last commit and replaces the
previous draft whole, so there is no retraction protocol.

A draft is the same decode as a commit, not a second code path. A decode
reads the context through `&self`, and only a commit moves the stream, so
`responsive` commits exactly what `conservative` commits on the same audio.
There is one exception today, involving language detection
([what a decode may touch](bunsen::kits::speech::whisper::driver#what-a-decode-may-touch)).
The [preset table](bunsen::kits::speech::whisper::driver#emission-commit-and-draft)
lists each preset's triggers, commit rule and drafts.

### The stream clock

Every context carries a
[`StreamClock`](bunsen::kits::speech::whisper::driver::StreamClock): anchors
that map a sample index to media time. A bare stream gets one anchor at
zero, which reproduces upstream's arithmetic exactly. A capture timestamp, a
container's presentation time, or a dropped buffer is one more anchor. The
bare case is the general case with one anchor, so the general case costs
nothing extra
([the stream clock](bunsen::kits::speech::whisper::driver#the-stream-clock)).

### The clamp policy is an injected object

Whisper floors its log-mels a fixed distance below a reference maximum, and
upstream takes that maximum over the whole clip. A stream cannot see the
whole clip, so where the reference comes from is a policy, and the policy
is an object the caller hands to each context
([`StreamClampPolicy`](bunsen::kits::speech::whisper::driver::StreamClampPolicy)),
not a variant the driver knows about. The context offers every new frame to
the policy, and asks it for a reference just before a window is packaged.
[`RunningMaxClamp`](bunsen::kits::speech::whisper::driver::RunningMaxClamp)
gives upstream's whole-clip behaviour, a running maximum, or a per-region
one, depending on how the context is fed.
[`PerWindow`](bunsen::kits::speech::whisper::driver::PerWindow) floors each
window on its own
([the clamp policy](bunsen::kits::speech::whisper::driver#the-clamp-policy)).

### Voice-activity regions as sub-streams

The real-time presets decode at the end of each speech region, so they need
a voice-activity model, attached with
[`with_vad`](bunsen::kits::speech::whisper::driver::WhisperStreamDriver::with_vad).
Each stream then runs Silero in a Silero context of its own. A
[`VoiceActivityFilterConfig`](bunsen::kits::speech::whisper::driver::VoiceActivityFilterConfig)
turns Silero's probabilities into
[`SpeechRegion`](bunsen::kits::speech::whisper::driver::SpeechRegion)s.
A closed region is decoded as a unit of its own, with times taken off the
parent stream's clock, and a full window of silence is skipped instead of
decoded. That gives a region the behaviour of a stream without a second
stream object
([regions as sub-streams](bunsen::kits::speech::whisper::driver#voice-activity-regions-as-sub-streams)).

### Known gaps

- [`CommitRule::Agreement`](bunsen::kits::speech::whisper::driver::CommitRule::Agreement)
  is not implemented yet: building a driver under it fails with
  `BunsenError::Invalid`.
- The `interval` trigger drafts only while the voice-activity gate says
  speech is in progress, and that gate runs only under `endpoint`
  ([`DecodeTriggers`](bunsen::kits::speech::whisper::driver::DecodeTriggers)).
