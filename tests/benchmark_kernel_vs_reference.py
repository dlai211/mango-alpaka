"""Before/after: PyTorch marginal spline versus the Alpaka kernel.

Four things this script is careful about, each after getting one wrong first:

1. **Same device.** An early revision left the torch model on the CPU while
   timing the kernel on the GPU. Both are now placed on the comparison device
   and the script asserts it.

2. **Same scope.** `ConditionalCubicFlow1D.forward` calls its conditioning MLP
   through `_params(cond)`; the kernel takes that MLP's output. The
   `torch spline` column shadows `_params` with cached tensors, giving a
   like-for-like comparison.

3. **Same dtype.** The pipeline runs `float32`; the kernel's parity reference is
   `float64`. Both are measured, because `float64` on a consumer GPU runs at
   1/32 the rate and would otherwise masquerade as a kernel weakness.

4. **The path a pipeline would take.** `alpaka realistic` runs the MLP in
   torch, copies the result to host, and calls the binding.

Usage
-----
    python tests/benchmark_kernel_vs_reference.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

UPSTREAM = Path(os.environ.get("MANGO_UPSTREAM", "C:/Users/laiji/test/conditional-nf-integration"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
if str(PROJECT / "build" / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT / "build" / "python"))

from nf.conditional_cubic_1d import ConditionalCubicFlow1D  # noqa: E402

import mango_alpaka_ext as mango  # noqa: E402

COND_DIM = 3
HIDDEN = 128
N_BINS = 32
SIZES = (1, 100, 1_000, 10_000, 50_000)
DTYPES = ((torch.float32, np.float32, "float32"), (torch.float64, np.float64, "float64"))


def bench(fn, warmup: int = 5, repeats: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - started)
    return best * 1e3


def make_model(dtype: torch.dtype, device: torch.device) -> ConditionalCubicFlow1D:
    torch.manual_seed(0)
    model = ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS).to(device)
    model = model.to(dtype).eval()
    return model


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: make_model(torch_dtype, device) for torch_dtype, _, name in DTYPES}
    for model in models.values():
        assert next(model.parameters()).device.type == device.type, "model is not on the comparison device"

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    print(f"Alpaka precisions: {mango.precisions()}")
    print()

    for direction in ("forward", "inverse"):
        op = direction
        print(f"=== {direction} ===")
        for torch_dtype, np_dtype, name in DTYPES:
            model = models[name]
            print(f"  {name}")
            header = (
                f"  {'N':>8} {'torch full':>11} {'torch spline':>13} {'alpaka':>9} "
                f"{'like-for-like':>14} {'realistic':>10} {'realistic x':>12}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))

            for n in SIZES:
                cond = torch.rand((n, COND_DIM), device=device, dtype=torch_dtype)
                u = torch.rand(n, device=device, dtype=torch_dtype)
                with torch.no_grad():
                    cached = model._params(cond)
                    point = u if direction == "forward" else model.forward(u, cond)[0]
                    raw_np = model.net(cond).detach().cpu().numpy().astype(np_dtype, copy=False)
                    point_np = point.detach().cpu().numpy().astype(np_dtype, copy=False)

                def torch_full():
                    getattr(model, op)(point, cond)

                orig_params = model._params
                model._params = lambda _cond, _c=cached: _c

                def torch_spline():
                    getattr(model, op)(point, cond)

                t_full = bench(torch_full)
                t_spline = bench(torch_spline)
                model._params = orig_params

                t_alpaka = bench(lambda: getattr(mango, op)(raw_np, point_np, backend="cuda"))

                def alpaka_realistic():
                    fresh = model.net(cond).detach().cpu().numpy().astype(np_dtype, copy=False)
                    getattr(mango, op)(fresh, point_np, backend="cuda")

                t_real = bench(alpaka_realistic)

                print(
                    f"  {n:>8,} {t_full:>11.2f} {t_spline:>13.2f} {t_alpaka:>9.2f} "
                    f"{t_spline / t_alpaka:>13.1f}x {t_real:>10.2f} {t_full / t_real:>11.1f}x"
                )
            print()

    print("torch full       = MLP + spline, as the pipeline calls it")
    print("torch spline     = spline only, like-for-like with the kernel")
    print("alpaka           = kernel with the MLP output precomputed")
    print("realistic        = MLP in torch + host copy + kernel (a real integration)")


if __name__ == "__main__":
    main()
