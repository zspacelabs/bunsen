# rewrite-genai-onnx

Rewrites an onnxruntime-genai ONNX export to standard ONNX that `burn-onnx` imports, for the build scripts of
the crates that generate bunsen's validation references (`bind-qwen3-onnx`, `bind-voxtral-onnx`). Not
published.

`onnx-community`'s exports of transformers models are onnxruntime-genai builds: their attention is
`com.microsoft::GroupQueryAttention`, their rotary embedding `com.microsoft::RotaryEmbedding`, their norms
`SimplifiedLayerNormalization` and `com.microsoft::SkipSimplifiedLayerNormalization`. `burn-onnx` 0.22 imports none
of them. `rewrite_for_burn_onnx` replaces each with standard opset-23 ONNX that computes the same thing; the API docs
say how.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
