# NeMo front-end fixture

`front_end.safetensors` is written once by `tools/gen_nemo_front_end_fixtures.py` (its docstring pins the versions):

- `waveforms`, `lengths`, `features`, `mask`: `transformers`' `NemotronAsrStreamingFeatureExtractor` (128 mels, no
  normalization) on two synthetic waveforms of different lengths, padded as one batch. Synthetic: no licence
  question.
- `window`, `filterbank`: the `preprocessor.featurizer.window` and `preprocessor.featurizer.fb` tables stored in
  NVIDIA's `Nemotron-3-Diarization` checkpoint (`Nemotron-3-Diarization.nemo` at revision `f667ed73`, OpenMDW-1.1,
  whose terms allow redistribution with the licence and notices retained). They are bf16, widened to float32 exactly:
  the tables `NeMo` computes with.

`kits::speech::nemo`'s tests hold `NemoMelFrontEndConfig::nemotron_diarization()` to both.

## `.nemo` fixture

`toy_sortformer_v2.nemo` and `toy_sortformer_v2.json` are written once by `tools/gen_nemo_fixtures.py` (its docstring
pins the versions): a seeded toy `SortformerEncLabelModel` in Sortformer v2's layout (a 2-layer FastConformer of width
16, a 2-layer post-LN Transformer of width 8, 3 speakers, 24 mels), saved by `NeMo`'s own `save_to`, and every entry of
its state dict (dtype, shape, the sum of its values). Synthetic weights: no licence question.

`kits::speech::nemo`'s archive tests (feature `nemo`) read its config into typed sections, extract and read its state
dict through the burn-store's `PyTorch` reader against the manifest, and load its encoder and transformer into bunsen's
blocks under their `NeMo` maps (`ConformerEncoder::nemo_rekey`, `ResidualEncoderTransformer::nemo_rekey`).
