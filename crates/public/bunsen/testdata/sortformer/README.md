# Sortformer fixtures

- `nemotron_toy.safetensors`, `nemotron_toy.json`: a tiny seeded `transformers`
  `Nemotron3DiarizationForAudioFrameClassification` (random weights, no checkpoint), and its answers in float64: each
  stage on whole inputs, and a long input through its offline mode (the speaker cache, compressed several times).
  Written once by `tools/gen_nemotron_diar_toy_fixtures.py`, whose docstring pins the versions. Synthetic: no licence
  question.
- `nemotron_config.json`: NVIDIA's `Nemotron-3-Diarization` `config.json` at revision `f667ed73` (sha256
  `dd795ec0...`), unchanged. OpenMDW-1.1: its terms allow redistribution with the licence and notices retained. The
  config parser's test reads it.
- `v2_toy.safetensors`, `v2_toy.json`: a tiny seeded `NeMo` `SortformerEncLabelModel` in Sortformer v2's layout
  (random weights, no checkpoint) under `NeMo`'s state-dict names, and its answers in float64, stage by stage: at a
  length the subsampling fills, one it pads, and a padded batch of two; and a long input through `NeMo`'s sync
  streaming (left context, the speaker cache compressed several times, a running-mean silence fill). Written once by
  `tools/gen_sortformer_v2_toy_fixtures.py`, whose docstring pins the versions. Synthetic: no licence question.
