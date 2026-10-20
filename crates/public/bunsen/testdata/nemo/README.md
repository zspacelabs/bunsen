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
