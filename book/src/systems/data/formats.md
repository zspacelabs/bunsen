# Formats

A kit's hook reads its checkpoint's own format with the readers in
[`bunsen::data::formats`]: the safetensors checkpoint and header
([`SafetensorsCheckpoint`](bunsen::data::formats::SafetensorsCheckpoint)),
the `PyTorch` reader, MLX's affine-quantized tensors, `FermionResearch`'s
Phonon formats, and named-tensor application, which loads any of them into
a module under a kit's names, or under a
[`Rekey`](bunsen::burner::module::Rekey) whose targets may also change a
tensor's data on the way. The readers need the `store` feature
([features](../../organization/features.md)).

A [`LoadPolicy`](bunsen::data::formats::LoadPolicy) is the load call's
choice of the float dtype a model's parameters are placed in. A load only
ever widens: a tensor goes in the dtype it was stored in, or f32, the first
the policy, the device and the hook allow. `performance()` keeps a half
dtype only where the device accelerates it, `as_stored()` wherever the
device computes in it, and `at_least(f32)` holds a model to an f32
reference.

## PyTorch checkpoints

An OpenAI `.pt` checkpoint is read through burn-store's PyTorch reader, and
a Hugging Face safetensors repo through its safetensors reader. The PyTorch
reader honours each tensor's strides, so a weight that PyTorch saved as a
transposed view of another tensor loads as the weight it is.
