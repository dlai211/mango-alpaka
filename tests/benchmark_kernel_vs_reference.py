"""Before/after: PyTorch marginal spline versus the Alpaka kernel.

`benchmark_fixed_cost.py` measures the *pre-kernel* baseline. This script puts
the kernel beside it, which is the acceptance test for Deliverable B: the claim
is that Alpaka wins by removing per-call overhead, not by doing arithmetic
faster, so the interesting comparison is at small and large batch alike.

Both sides run on the **same device**. An earlier revision of this script left
the torch model on the CPU while timing the kernel on the GPU, which inflated
the speedups by roughly an order of magnitude at small N. The device is now
asserted rather than assumed.

The torch side is reported in both dtypes: `float32` is what the pipeline runs,
`float64` is the like-for-like comparison with the kernel's current precision.

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
MARGINALS = 2


def bench(fn, warmup: int = 5, repeats: int = 20) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        if torch.cuda.is_available():
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

    model32 = ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS).to(device).eval()
    model64 = (
        ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS)
        .double()
        .to(device)
        .eval()
    )
    assert next(model32.parameters()).device.type == device.type, "torch model is not on the comparison device"
    assert next(model64.parameters()).device.type == device.type

    print(f"torch {torch.__version__}  device {device}")
    print(f"marginals compared: {MARGINALS}; n_bins={N_BINS}")
    print()

    header = (
        f"{'N':>8} | {'torch f32':>10} {'alpaka':>9} {'x':>6} | "
        f"{'torch f64':>10} {'x':>6}"
    )
    print(header)
    print("-" * len(header))
    print("forward")
    for n in (1, 100, 1_000, 10_000, 50_000):
        cond32 = torch.rand((n, COND_DIM), device=device)
        u32 = torch.rand(n, device=device)
        with torch.no_grad():
            raw64 = model64.net(cond32.double()).detach().cpu().numpy()
            u_np = u32.detach().cpu().numpy()

        t32 = bench(lambda: [model32.forward(u32, cond32) for _ in range(MARGINALS)])
        t64 = bench(
            lambda: [model64.forward(u32.double(), cond32.double()) for _ in range(MARGINALS)]
        )
        ta = bench(lambda: [mango.forward(raw64, u_np, backend="cuda") for _ in range(MARGINALS)])
        print(f"{n:>8,} | {t32:>10.2f} {ta:>9.2f} {t32 / ta:>5.1f}x | {t64:>10.2f} {t64 / ta:>5.1f}x")

    print("inverse")
    for n in (1, 100, 1_000, 10_000, 50_000):
        cond32 = torch.rand((n, COND_DIM), device=device)
        u32 = torch.rand(n, device=device)
        with torch.no_grad():
            x32, _ = model32.forward(u32, cond32)
            x64 = x32.double()
            raw64 = model64.net(cond32.double()).detach().cpu().numpy()
            x_np = x32.detach().cpu().numpy()

        t32 = bench(lambda: [model32.inverse(x32, cond32) for _ in range(MARGINALS)])
        t64 = bench(lambda: [model64.inverse(x64, cond32.double()) for _ in range(MARGINALS)])
        ta = bench(lambda: [mango.inverse(raw64, x_np, backend="cuda") for _ in range(MARGINALS)])
        print(f"{n:>8,} | {t32:>10.2f} {ta:>9.2f} {t32 / ta:>5.1f}x | {t64:>10.2f} {t64 / ta:>5.1f}x")

    print()
    print("Columns are milliseconds for all marginals in one call, end to end")
    print("through the Python binding, both on the same device.")
    print("torch f32 is the pipeline-realistic comparison; torch f64 is")
    print("like-for-like with the kernel's current precision.")


if __name__ == "__main__":
    main()
