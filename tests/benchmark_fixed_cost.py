"""Pre-kernel fixed-cost baseline for the PyTorch marginal spline.

Deliverable B claims the Alpaka kernel wins by removing per-call overhead, not
by doing arithmetic faster. Before that claim can be checked, the overhead has
to be measured. This script measures it on the *existing* implementation: a bank
of `ConditionalCubicFlow1D` marginals, evaluated exactly as the pipeline does.

For every (dimension, backend, batch size) it records:

  * ``sample``  — forward map u -> x, i.e. drawing from the marginal
  * ``log_density`` — inverse map x -> u, which is what `log_prob` costs

and derives the per-point cost. A flat curve at small N against a steep one at
large N is the signature of launch-bound execution.

Outputs
-------
results/fixed_cost_baseline.csv
results/fixed_cost_baseline.md

Usage
-----
    python tests/benchmark_fixed_cost.py
    python tests/benchmark_fixed_cost.py --dims 2 5 8 --sizes 1 100 512 10000
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

UPSTREAM = Path(os.environ.get("MANGO_UPSTREAM", "C:/Users/laiji/test/conditional-nf-integration"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from nf.conditional_cubic_1d import ConditionalCubicFlow1D  # noqa: E402

# These are the pipeline's actual values, from
# run_bayesian_selected_copula_benchmark.py via run_adaptive_...
COND_DIM = 3
HIDDEN = 128
N_BINS = 32


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_bank(dim: int, device: torch.device, seed: int = 0) -> torch.nn.ModuleList:
    torch.manual_seed(seed)
    bank = torch.nn.ModuleList(
        [ConditionalCubicFlow1D(cond_dim=COND_DIM, hidden=HIDDEN, n_bins=N_BINS) for _ in range(dim)]
    )
    bank = bank.to(device)
    bank.eval()
    return bank


def time_callable(fn, device: torch.device, warmup: int, repeats: int) -> float:
    """Return the best wall-clock time in seconds, with warmup and sync."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync(device)
        best = float("inf")
        for _ in range(repeats):
            sync(device)
            started = time.perf_counter()
            fn()
            sync(device)
            best = min(best, time.perf_counter() - started)
    return best


def measure(dim: int, n: int, device: torch.device, warmup: int, repeats: int) -> dict:
    bank = build_bank(dim, device)

    generator = torch.Generator(device="cpu").manual_seed(n + dim)
    cond = torch.rand((n, COND_DIM), generator=generator, dtype=torch.float32).to(device)
    u = torch.rand(n, generator=torch.Generator().manual_seed(n), dtype=torch.float32).to(device)

    with torch.no_grad():
        # A fixed x to invert, derived from the forward map so it lies in range.
        x, _ = bank[0].forward(u, cond)

        def do_sample():
            for marginal in bank:
                marginal.forward(u, cond)

        def do_log_density():
            for marginal in bank:
                marginal.inverse(x, cond)

        sample_s = time_callable(do_sample, device, warmup, repeats)
        logdensity_s = time_callable(do_log_density, device, warmup, repeats)

    return {
        "dim": dim,
        "n": n,
        "device": device.type,
        "sample_ms": sample_s * 1e3,
        "log_density_ms": logdensity_s * 1e3,
        # Per point, per marginal: d marginals are evaluated per call.
        "sample_us_per_point": sample_s * 1e6 / (n * dim),
        "log_density_us_per_point": logdensity_s * 1e6 / (n * dim),
    }


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| dim | device | N | sample (ms) | sample (µs/point) | log_density (ms) | log_density (µs/point) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["device"], r["dim"], r["n"])):
        lines.append(
            f"| {row['dim']} | {row['device']} | {row['n']:,} | "
            f"{row['sample_ms']:.3f} | {row['sample_us_per_point']:.2f} | "
            f"{row['log_density_ms']:.3f} | {row['log_density_us_per_point']:.2f} |"
        )
    return "\n".join(lines)


def fixed_cost_estimate(rows: list[dict], key: str) -> str:
    """Report ms at N=1, which is almost entirely fixed overhead."""
    lines = []
    for device in sorted({row["device"] for row in rows}):
        for dim in sorted({row["dim"] for row in rows}):
            subset = [r for r in rows if r["device"] == device and r["dim"] == dim]
            if not subset:
                continue
            smallest = min(subset, key=lambda r: r["n"])
            largest = max(subset, key=lambda r: r["n"])
            if smallest["n"] == largest["n"]:
                continue
            ratio = largest[key] / smallest[key] if smallest[key] else float("nan")
            lines.append(
                f"  {device:>4} dim={dim}  {smallest[key]:.2f} ms at N={smallest['n']:,}"
                f"  ->  {largest[key]:.1f} ms at N={largest['n']:,}"
                f"   ({ratio:.1f}x for {largest['n'] // smallest['n']}x the points)"
            )
    return "\n".join(lines)


def main() -> None:
    # The table uses µ; Windows consoles default to a legacy code page.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 8])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 512, 1000, 10_000, 50_000])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    args = parser.parse_args()

    devices = [torch.device(name) for name in args.devices]
    devices = [d for d in devices if d.type != "cuda" or torch.cuda.is_available()]

    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    print(f"marginal: ConditionalCubicFlow1D(hidden={HIDDEN}, n_bins={N_BINS})")
    print()

    rows = []
    for device in devices:
        for dim in args.dims:
            for n in args.sizes:
                repeats = args.repeats if n <= 10_000 else max(3, args.repeats // 4)
                row = measure(dim, n, device, args.warmup, repeats)
                rows.append(row)
                print(
                    f"  {device.type:>4} dim={dim} N={n:>6,}  "
                    f"sample {row['sample_ms']:9.3f} ms  "
                    f"log_density {row['log_density_ms']:9.3f} ms"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fixed_cost_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = markdown_table(rows)
    md_path = args.output_dir / "fixed_cost_baseline.md"
    md_path.write_text(
        "# Fixed-cost baseline — PyTorch marginal spline\n\n"
        "Measured by `tests/benchmark_fixed_cost.py`. Times are the best of "
        f"{args.repeats} repeats after {args.warmup} warmup calls, with "
        "`torch.cuda.synchronize()` around every CUDA measurement.\n\n"
        "A bank of `dim` independent `ConditionalCubicFlow1D` marginals is "
        "evaluated per call, matching the pipeline's `MarginalBank`.\n\n"
        + table
        + "\n\n## Fixed cost\n\n"
        "Milliseconds at the smallest batch, where arithmetic is negligible:\n\n"
        "```\n"
        + fixed_cost_estimate(rows, "sample_ms")
        + "\n"
        + fixed_cost_estimate(rows, "log_density_ms")
        + "\n```\n",
        encoding="utf-8",
    )

    print()
    print(table)
    print()
    print("Fixed cost (sample):")
    print(fixed_cost_estimate(rows, "sample_ms"))
    print()
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
