"""Before/after: PyTorch marginal spline versus the Alpaka kernel.

Three things this script is careful about, each after getting one wrong first:

1. **Same device.** An earlier revision left the torch model on the CPU while
   timing the kernel on the GPU. Both models are now placed on the comparison
   device and the script asserts it.

2. **Same scope.** `ConditionalCubicFlow1D.forward` calls its conditioning MLP
   through `_params(cond)`; the kernel takes that MLP's *output*. Comparing them
   as-is compares "MLP + spline" against "spline". This script reports both:
   `torch spline` shadows `_params` with cached tensors, giving a like-for-like
   comparison with the kernel.

3. **The path a pipeline would actually take.** `alpaka realistic` runs the MLP
   in torch, copies the result to host, and calls the binding — which is what an
   integration would do today. It is the number that matters for a deployment
   decision, and it is worse than the spline-only number.

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


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS).double().to(device).eval()
    assert next(model.parameters()).device.type == device.type, "torch model is not on the comparison device"

    # The trained model's logits live in about [-1.6, 1.3]; keep the benchmark
    # inside that range so the timing is representative (see RESULTS.md §3).
    with torch.no_grad():
        probe = model.net(torch.rand((4096, COND_DIM), dtype=torch.float64, device=device))
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    print(f"logit range probed: [{probe.min():.2f}, {probe.max():.2f}]")
    print()

    for direction in ("forward", "inverse"):
        print(direction)
        header = (
            f"{'N':>8} {'torch full':>11} {'torch spline':>13} "
            f"{'alpaka spline':>14} {'spline x':>9} {'alpaka realistic':>17} {'real x':>8}"
        )
        print(header)
        print("-" * len(header))

        for n in SIZES:
            cond = torch.rand((n, COND_DIM), device=device, dtype=torch.float64)
            u = torch.rand(n, device=device, dtype=torch.float64)
            with torch.no_grad():
                cached = model._params(cond)
                point = u if direction == "forward" else model.forward(u, cond)[0]
                raw_np = model.net(cond).detach().cpu().numpy()
                point_np = point.detach().cpu().numpy()

            op = "forward" if direction == "forward" else "inverse"

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
                fresh = model.net(cond).double().detach().cpu().numpy()
                getattr(mango, op)(fresh, point_np, backend="cuda")

            t_real = bench(alpaka_realistic)

            print(
                f"{n:>8,} {t_full:>11.2f} {t_spline:>13.2f} {t_alpaka:>14.2f} "
                f"{t_spline / t_alpaka:>8.1f}x {t_real:>17.2f} {t_full / t_real:>7.1f}x"
            )
        print()

    print("torch full       = MLP + spline, as the pipeline calls it")
    print("torch spline     = spline only, like-for-like with the kernel")
    print("alpaka spline    = kernel with the MLP output precomputed")
    print("alpaka realistic = MLP in torch + host copy + kernel (a real integration)")


if __name__ == "__main__":
    main()
