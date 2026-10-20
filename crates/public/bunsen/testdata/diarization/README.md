# Diarization fixtures

`nemo.json` is what NeMo itself computes, written once by `tools/gen_turn_fixtures.py` (its docstring pins the NeMo
commit and the tool versions):

- `turns`: seeded per-speaker probability tracks (float32, 80 ms frames, three speakers) and the speaker turns NeMo's
  post-processing gives them (`binarization_vectorized`, then `filtering`) under each `TurnFilterConfig` preset and
  two more configs. `kits::speech::diarization::turns`'s tests hold `TurnFilterConfig::turns` to them.
- `der`: seeded reference and hypothesis turns, and NeMo's md-eval scores of them (DER, confusion, false alarm, miss),
  with and without a collar and with and without overlap scored. `support::testing::diarization`'s tests hold
  `diarization_error` to them.

Every value is synthetic: no recording, no licence question.
