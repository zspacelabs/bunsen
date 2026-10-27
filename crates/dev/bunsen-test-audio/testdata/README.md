# Test audio

The audio clips the workspace's tests, validation suites and fixture scripts read. `src/lib.rs` names each one as an
`AudioFixture` (its rate, duration, licence and SHA-256), and its tests hold every file to its pin.

Every clip here is redistributable under bunsen's own MIT/Apache-2.0 terms, or needs no licence at all. A clip under
a copyleft or non-commercial licence does not belong here. A Creative Commons BY-SA speech sample was considered and
rejected on those grounds.

| file | | |
|---|---|---|
| `jfk_moon.mp3` | 60.0 s, 16 kHz mono, LAME 48 kbps | "We choose to go to the Moon", live, with hall noise and applause |
| `jfk_moon.txt` | | **ground truth** for `jfk_moon.mp3`: what a person hears |
| `jfk_moon_4s.mp3` | 4.0 s, 16 kHz mono, LAME 48 kbps | speech with a one-second pause |
| `tone.wav` | 0.5 s, 16 kHz mono, 16-bit PCM | a synthetic tone, 8000 samples exactly |
| `tone.mp3` | 0.5 s, 16 kHz mono, LAME 48 kbps | the same tone |
| `silero_vad_test.wav` | 60.0 s, 16 kHz mono, 16-bit PCM | silero-vad's own test recording |
| `udhr_de.mp3` | 22.2 s, 16 kHz mono, LAME 48 kbps | the Universal Declaration of Human Rights, read in German, with a steady hiss |
| `udhr_fr.mp3` | 20.0 s, 16 kHz mono, LAME 48 kbps | the same passage, read in French |
| `udhr_es.mp3` | 27.6 s, 16 kHz mono, LAME 48 kbps | the same passage, read in Spanish |
| `udhr_{de,fr,es}.txt` | | **ground truth** for each: what a person hears, in the language spoken |
| `udhr_{de,fr,es}.en.txt` | | **ground truth** for a translation of each to English |

Fixtures that are a model's output on these clips live with the validation crate of the model that made them:
`openai-whisper`'s decodes of `jfk_moon.mp3` with `whisper-model-validation`, the `transformers` and onnxruntime
decodes with the Parakeet and Qwen3-ASR validation crates, and Silero's reference probabilities with
`silero-model-validation`.

## Fetched

Recordings too large to commit are fetched by a feature, digest-pinned, into the build directory:

| feature | fixture | | licence |
|---|---|---|---|
| `ami` | `AMI_EN2002D_2132` | 60 s of AMI meeting EN2002d, 16 kHz mono PCM, with its speaker RTTM and transcript | CC BY 4.0; the RTTM Apache-2.0 |

### `AMI_EN2002D_2132`

- Source: [`NVIDIA/NeMo-Speech.cpp`](https://github.com/NVIDIA/NeMo-Speech.cpp) `test_files/diar/` at commit
  `2bdfb03b38d5e3b0fa55aeb91a079a9f57385373` (Apache-2.0 repository): `ami_en2002d_2132.{wav,rttm,json}`
- Audio: meeting EN2002d of the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), Mix-Headset, from
  2132.5 s, unmodified 16 kHz mono PCM. Licence:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution: the AMI Meeting Corpus.
- RTTM: cut from the word-aligned `only_words` reference of
  [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) (Apache-2.0), times
  relative to the excerpt.
- Transcript: the AMI manual word annotations for the excerpt (CC BY 4.0).

## Provenance

### `jfk_moon.mp3`

- Source: [Jfk rice university we choose to go to the moon.ogg](https://commons.wikimedia.org/wiki/File:Jfk_rice_university_we_choose_to_go_to_the_moon.ogg),
  retrieved from Wikimedia Commons on 2026-08-28
- Author: John F. Kennedy, address at Rice University, 1962-09-12
- Credit: John F. Kennedy Presidential Library & Museum
- Licence: public domain (work of the U.S. federal government)
- Transcode: `-ss 503 -t 60 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k` (from 44.1 kHz stereo Vorbis), not
  bit-identical to the source

The source is the complete 17.7-minute address, ~17 MB. Only the 60 s from 08:23 to 09:23 is kept: the "why climb the
highest mountain" series through the end of the famous passage. That offset is also chosen so the 30 s window boundary
falls on "not because they are easy": a greedy decode is sensitive to where a window lands, and this cut decodes
cleanly on both sides of it.

`jfk_moon.txt` is the transcript of those 60 s, as a person hears them.

### `jfk_moon_4s.mp3`

Seconds 7.0 to 11.0 of `jfk_moon.mp3`. The cut holds the end of "why climb the highest mountain?", about a second of
pause, and "Why, thirty-five years ago, fly the Atlantic?", so a voice-activity gate with a short silence threshold
finds two regions and a patient one finds one.

```sh
ffmpeg -i jfk_moon.mp3 -ss 7.0 -t 4.0 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k jfk_moon_4s.mp3
```

### `tone.wav`, `tone.mp3`

Synthetic on purpose: a format test does not need real audio, and a tone carries no licence or provenance question.

```sh
ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=0.5" \
  -af "volume=0.6" -ac 1 -ar 16000 -c:a pcm_s16le tone.wav
ffmpeg -i tone.wav -c:a libmp3lame -b:a 48k -ac 1 -ar 16000 tone.mp3
```

The mp3 decodes to slightly more than 8000 samples: gapless playback trims the encoder delay, but mp3 frames in blocks
of 576 samples, so the tail rounds up.

### `silero_vad_test.wav`

- Source: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) `tests/data/test.wav`, at commit
  `fbddc91a5d51f5f448f9c158b327ddf75b9355ad` (git blob `0b1b64ecd9b316445e06cfd5b6b709b0e5fcf18f`, byte-identical)
- Licence: MIT (silero-vad)

### `udhr_de.mp3`, `udhr_fr.mp3`, `udhr_es.mp3`

The same passage in three languages: the title, the preamble's heading and its first clause, "Whereas recognition of
the inherent dignity ...". One text read in several languages checks a model's choice of language and of task, to
transcribe or to translate, each against a reference.

- Source: LibriVox's [Universal Declaration of Human Rights, Volume
  01](https://librivox.org/the-universal-declaration-of-human-rights-by-the-united-nations) (archive.org item
  `universal_declaration_librivox`), retrieved on 2026-10-27, 64 kbps mp3 at 22.05 kHz:
  - `udhr_de.mp3`: `human_rights_un_ger_robu_64kb.mp3`, read by Roman Buettner (stereo; a steady hiss under the voice)
  - `udhr_fr.mp3`: `human_rights_un_frn_ezwa_64kb.mp3`, read by Ezwa
  - `udhr_es.mp3`: `human_rights_un_spn_krs_64kb.mp3`, read by Karen Savage (stereo)
- Licence: the recordings are public domain (LibriVox; it records only texts that are public domain in the USA). The
  Declaration is the United Nations' (General Assembly resolution 217 A (III), 1948), which may be reproduced
  without prior permission when distributed at no cost.
- The ground-truth files quote the Declaration under that condition: this crate is not published (`publish = false`),
  so they are distributed only with bunsen's repository, at no cost. A crate that is published must not carry them.
- Transcode: each reading's LibriVox disclaimer is dropped; the cut starts at the title and ends in the pause after the
  first clause. Not bit-identical to the source.

```sh
ffmpeg -i human_rights_un_ger_robu_64kb.mp3 -ss 23.0 -t 22.2 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k \
  -map_metadata -1 udhr_de.mp3
ffmpeg -i human_rights_un_frn_ezwa_64kb.mp3 -ss 17.3 -t 20.0 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k \
  -map_metadata -1 udhr_fr.mp3
ffmpeg -i human_rights_un_spn_krs_64kb.mp3 -ss 14.2 -t 27.6 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k \
  -map_metadata -1 udhr_es.mp3
```

`udhr_{de,fr,es}.txt` are what a person hears: each reader's own title and resolution line, then the clause as the
UN's text in that language has it. `udhr_{de,fr,es}.en.txt` are the same in English: the clause as the UN's English
text has it, the title and resolution line translated word for word. The German reading's clause ends mid-sentence,
at a comma, which its transcripts mark with `...`.
