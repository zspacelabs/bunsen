# ONNX binding crates

One crate per validated model, each binding an upstream ONNX export into Rust for the model's
validation crate, and fetching the checkpoints it compares. None is published, and none fetches
anything unless a feature asks.

| Crate | What it is |
|---|---|
| `rewrite-genai-onnx` | A build-dependency of the `bind-{model}-onnx` crates: rewrites an onnxruntime-genai ONNX export's `com.microsoft` attention, rotary and RMS-norm ops to standard ONNX that `burn-onnx` imports. |
| `bind-{model}-onnx` | One per validated model (`whisper`, `silero`, `parakeet`, `qwen3`, `qwen3-asr`, `sortformer`, `voxtral`): fetches, at build time and only under its features, the checkpoints and ONNX exports the model's validation crate compares, digest-pinned and laid out as a pretrained cache, and generates the reference model from the export with `burn-onnx`. Hosts no files. |
