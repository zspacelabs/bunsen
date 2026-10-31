# nanochat-model-validation

Validates [`bunsen`](https://crates.io/crates/bunsen)'s nanochat (`kits::gpts::nanochat`) against upstream nanochat's
own code, on the community checkpoints of both of its models: `nanochat-students`' d20 base, mid and RL (upstream's
2025 model) and `nitishpandey04`'s d18 base and SFT (its 2026 model). Not published.

There is no independent port to hold bunsen to: upstream's model is one file of `PyTorch`, run as it is. The fixtures
are upstream's code at each model's pinned commit (`karpathy/nanochat` at `a088b7a6`, `a3ca42a6`) on the rows' own
files (`validation/tools/gen_nanochat_fixtures.py`):

- `testdata/upstream.json`: on the CPU in float32, a prompt as upstream's tokenizer encodes it, the prefill's last
  logits, and 32 greedy tokens through upstream's KV cache, with each step's margin;
- `testdata/upstream_bf16.json`: the same on CUDA in bf16, as upstream serves there (the 2025 model under autocast,
  the 2026 one in its bf16 compute dtype).

## Running it

```sh
cargo make validate-nanochat                          # wgpu, float32
VALIDATION_BACKEND=cuda cargo make validate-nanochat  # cuda: float32, and bf16
```

`download` fetches the checkpoints through `fetch-nanochat` (~10 GB on a cold cache, verified against pinned
digests), `gpu-tests` compiles the tests that load a model, and a backend feature says what they run on. Without them
the suite is the fixture-integrity checks.

On `cuda`, `tests/cuda_bf16.rs` (a test binary of its own, since it sets the device's float dtype to bf16) loads each
row into a bf16 model, checks that the checkpoint's bf16 tensors are the model's bit for bit, and holds its prefill
logits and greedy decode to upstream's bf16 run.

## Tolerance

Float32: the prefill logits within 1e-3 of upstream's (bunsen measures about 4e-5), and every greedy token agreeing
but a near-tie (a top-1 minus top-2 margin under 0.02). bf16: the logits within 0.5 + 5% (bunsen measures 0.11 to
0.22, as far as upstream's own bf16 and float32 logits are apart), and every token but a bf16 near-tie (a margin
under 0.5).
