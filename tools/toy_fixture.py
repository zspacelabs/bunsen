"""The shared steps of the toy-model fixture generators.

A toy fixture is a tiny model, `transformers`' own implementation, with seeded random
weights, run on the CPU in float64 and stored as float32: its weights under their
state-dict names, and the inputs and outputs a kit's toy test checks it against as `io.*`.
A generator states its model's config, which modules' outputs to keep, and its inputs;
this module builds, initializes, captures and writes.

    from toy_fixture import Capture, build, init_weights, io_tensors, weights, write

    model = build(lambda: Qwen3ForCausalLM(Qwen3Config(**CONFIG)), seed=1234)
    capture = Capture()
    capture.output("layer0", model.model.layers[0])
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    write(OUT / "toy.safetensors", weights(model), io_tensors({**capture.values, "logits": logits}))

The steps keep each generator's random stream: `build` seeds before it constructs, and
`init_weights` draws once per parameter it sets, in `named_parameters` order. A generator
moved onto them writes the fixture it wrote before, byte for byte.

Run the generators with `python -I` from the repository root; each puts `tools/` on the
path itself.
"""

import json
import pathlib
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
"""The repository's root."""

TESTDATA = ROOT / "crates/public/bunsen/testdata"
"""Where `bunsen`'s fixtures live: a kit's under `TESTDATA / kit`."""


def build(make: Callable[[], torch.nn.Module], seed: int) -> torch.nn.Module:
    """`make()`'s model, seeded with `seed` before it is constructed (its own initializer
    draws from the stream), in float64 and in eval mode."""
    torch.manual_seed(seed)
    return make().to(torch.float64).eval()


def is_norm_weight(name: str) -> bool:
    """A normalization's gain: a parameter named `*norm*` ending `.weight`."""
    return "norm" in name and name.endswith(".weight")


def is_norm_bias(name: str) -> bool:
    """A normalization's shift: a parameter named `*norm*` ending `.bias`."""
    return "norm" in name and name.endswith(".bias")


def fan_in(p: torch.Tensor) -> int:
    """A parameter's fan-in: its second axis times its kernel axes; 4 for a vector."""
    if p.ndim < 2:
        return 4
    n = p.shape[1]
    for d in p.shape[2:]:
        n *= d
    return n


@torch.no_grad()
def init_weights(
    model: torch.nn.Module,
    *,
    norm_gain: float = 0.2,
    norm_center: float = 1.0,
    norm_bias: Optional[float] = None,
    is_norm: Callable[[str], bool] = is_norm_weight,
    others: bool = True,
) -> None:
    """Sets the parameters to seeded random values, in `named_parameters` order, one draw
    each: what initializes to constants (norm gains, which `_init_weights` sets to one) as
    random, so a misplaced one is caught.

    - A norm gain (`is_norm`) is `norm_center + norm_gain * N(0, 1)`: about one for a
      gain applied as stored, about zero for one applied as `1 + w` (Gemma's).
    - A norm shift, when `norm_bias` is given, is `norm_bias * N(0, 1)`; otherwise it is
      one of the others.
    - The others, when `others`, are `N(0, 1) / sqrt(fan_in)`; otherwise they keep the
      model's own initialization.
    """
    for name, p in model.named_parameters():
        if is_norm(name):
            p.copy_(norm_center + norm_gain * torch.randn_like(p))
        elif norm_bias is not None and is_norm_bias(name):
            p.copy_(norm_bias * torch.randn_like(p))
        elif others:
            p.copy_(torch.randn_like(p) / np.sqrt(fan_in(p)))


@torch.no_grad()
def init_running_stats(model: torch.nn.Module) -> None:
    """Sets the batch norms' running statistics to seeded random values (means
    `0.3 * N(0, 1)`, variances in `[0.5, 1.5)`), so that eval mode's normalization is not
    the identity."""
    for name, b in model.named_buffers():
        if name.endswith("running_mean"):
            b.copy_(0.3 * torch.randn_like(b))
        elif name.endswith("running_var"):
            b.copy_(0.5 + torch.rand_like(b))


def first(output):
    """A module's output, or the first of a tuple of them (a decoder layer's)."""
    return output[0] if isinstance(output, tuple) else output


class Capture:
    """Keeps what modules see as a model runs, under names, in `values`: a module's
    output (`output`) or its first positional input (`input`). A module run more than
    once keeps its last."""

    def __init__(self):
        self.values: dict = {}

    def output(self, name: str, module: torch.nn.Module) -> None:
        """Keeps `module`'s output (the first of a tuple) as `name`."""
        module.register_forward_hook(
            lambda _m, _i, out: self.values.__setitem__(name, first(out))
        )

    def input(self, name: str, module: torch.nn.Module) -> None:
        """Keeps `module`'s first positional input as `name`."""
        module.register_forward_pre_hook(
            lambda _m, args: self.values.__setitem__(name, args[0])
        )


def weights(
    model: torch.nn.Module,
    *,
    drop: Iterable[str] = (),
    rename: Callable[[str], str] = lambda name: name,
) -> dict:
    """The model's state dict as float32, without the names in `drop` (a tied head),
    under `rename`'s names (the checkpoint's own layout)."""
    drop = set(drop)
    return {
        rename(k): v.float().contiguous()
        for k, v in model.state_dict().items()
        if k not in drop
    }


def io_tensors(io: Mapping[str, torch.Tensor], prefix: str = "io.") -> dict:
    """`io` as float32, each name under `prefix`."""
    return {f"{prefix}{k}": v.float().contiguous() for k, v in io.items()}


def write(path: pathlib.Path, *parts: Mapping[str, torch.Tensor]) -> None:
    """Writes the parts, merged in order, as one safetensors file; makes its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    for part in parts:
        tensors.update(part)
    save_file(tensors, path)


def write_json(path: pathlib.Path, value, *, indent: int = 1, sort_keys: bool = False) -> None:
    """Writes `value` as JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=indent, sort_keys=sort_keys) + "\n")


def write_config_json(path: pathlib.Path, config, architectures: list) -> None:
    """Writes a `transformers` config as `save_pretrained` would, without the fields that
    name the writer (`transformers_version`, `dtype`), and with its `architectures`."""
    saved = config.to_dict()
    for k in ("transformers_version", "dtype", "torch_dtype"):
        saved.pop(k, None)
    saved["architectures"] = architectures
    write_json(path, saved, indent=2, sort_keys=True)


def shapes(tensors: Mapping[str, torch.Tensor]) -> dict:
    """Each tensor's shape, for a generator's report."""
    return {k: tuple(v.shape) for k, v in tensors.items()}
