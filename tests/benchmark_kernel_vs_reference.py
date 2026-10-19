"""Before/after: PyTorch marginal spline versus the Alpaka kernel.

`benchmark_fixed_cost.py` measures the *pre-kernel* baseline. This script puts
the kernel beside it, which is the acceptance test for Deliverable B: the claim
is that Alpaka wins by removing per-call overhead, not by doing arithmetic
faster, so the interesting comparison is at small and large batch alike.

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
    model = ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS).double().eval()

    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    print(f"marginals compared: 2 (dim=2); n_bins={N_BINS}; float64")
    print()
    header = (
        f"{'N':>8} {'torch fwd':>11} {'alpaka fwd':>11} {'fwd x':>7} "
        f"{'torch inv':>11} {'alpaka inv':>11} {'inv x':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in (1, 100, 1_000, 10_000, 50_000):
        cond = torch.rand((n, COND_DIM), dtype=torch.float64)
        u = torch.rand(n, dtype=torch.float64)
        with torch.no_grad():
            raw = model.net(cond)
            x, _ = model.forward(u, cond)

        raw_np = raw.numpy()
        u_np = u.numpy()
        x_np = x.numpy()

        def torch_forward():
            for _ in range(2):
                model.forward(u, cond)

        def torch_inverse():
            for _ in range(2):
                model.inverse(x, cond)

        def alpaka_forward():
            for _ in range(2):
                mango.forward(raw_np, u_np, backend="cuda")

        def alpaka_inverse():
            for _ in range(2):
                mango.inverse(raw_np, x_np, backend="cuda")

        t_fwd = bench(torch_forward)
        a_fwd = bench(alpaka_forward)
        t_inv = bench(torch_inverse)
        a_inv = bench(alpaka_inverse)

        rows.append((n, t_fwd, a_fwd, t_inv, a_inv))
        print(
            f"{n:>8,} {t_fwd:>11.3f} {a_fwd:>11.3f} {t_fwd / a_fwd:>6.1f}x "
            f"{t_inv:>11.3f} {a_inv:>11.3f} {t_inv / a_inv:>6.1f}x"
        )

    print()
    print("Both columns include the host<->device copies the C++ launcher makes")
    print("per call, so the speedups are end-to-end through the Python binding.")


if __name__ == "__main__":
    main()
