# RESULTS — mango-alpaka

What was measured, what it means, and what it does not say.
Every number here is reproducible with `REPRODUCTION_COMMANDS.md`.

---

## 1. The reference implementation is overhead-bound

Measured with `tests/benchmark_fixed_cost.py` on the **unmodified** PyTorch
marginal (`ConditionalCubicFlow1D`, `n_bins=32`, `hidden=128`), forming a bank
of `dim` marginals exactly as the pipeline does. Best of 20 repeats after 5
warmup calls; `torch.cuda.synchronize()` around every CUDA measurement.

### dim = 2

| device | operation | N=1 | N=100 | N=1 000 | N=10 000 | N=50 000 |
|---|---|---:|---:|---:|---:|---:|
| cpu | sample | 0.94 ms | 1.63 | 3.47 | 13.3 | 78.7 |
| cpu | log_density | 13.1 ms | 19.0 | 22.6 | 40.3 | 127.3 |
| cuda | sample | **3.68 ms** | 3.80 | 3.73 | **3.70** | 13.1 |
| cuda | log_density | **69.8 ms** | 65.9 | 66.5 | **68.7** | **78.8** |

The CUDA columns are the finding. **`sample` costs the same at N=1 and
N=10 000**, and `log_density` costs the same at N=1 and N=50 000. Per point:

| | N=1 | N=50 000 | ratio |
|---|---:|---:|---:|
| cuda sample | 1 842 µs/point | 0.13 µs/point | **14 000×** |
| cuda log_density | 34 893 µs/point | 0.79 µs/point | **44 000×** |

Arithmetic does not behave like that. Launch overhead does.

### dim = 5 and dim = 8

The same shape, scaled by the number of marginals:

| device | operation | dim=5, N=50 000 | dim=8, N=50 000 |
|---|---|---:|---:|
| cpu | sample | 192.7 ms | 315.5 ms |
| cpu | log_density | 320.5 ms | 512.9 ms |
| cuda | sample | 35.2 ms | 55.5 ms |
| cuda | log_density | 177.9 ms | 286.2 ms |

The CPU side scales with batch; the CUDA side is dominated by fixed cost until
N is large. That is why the GPU looks useless at small batch and fine at large
batch — the crossover measured in the upstream pipeline (§6.2 there) has this
same cause.

### Why the inverse costs so much more than the forward

Forward is one bin search plus one Hermite evaluation. The inverse runs **40
bisection iterations**, each re-evaluating the Hermite polynomial — and in
PyTorch every iteration is a handful of separate kernel launches.

At N=1 the inverse costs **12.4×** the forward (16.09 ms against 1.30 ms). The
ratio falls with batch size (1.6× at N=50 000) because at that point both are
doing real work; at small batch the launch count is the entire difference.

That asymmetry is the whole opportunity.

---

## 2. The Alpaka kernel: where it wins and where it does not

`tests/benchmark_kernel_vs_reference.py`, dim=2 (two marginal calls), **both
sides on the same GPU**, end to end through the Python binding. Both columns
include the host↔device copies the C++ launcher performs.

> **Correction.** An earlier revision compared PyTorch running on the **CPU**
> against the kernel on the **GPU**, because the benchmark never moved the
> torch model to the device. That inflated every speedup by roughly an order of
> magnitude at small N. The device is now asserted in the script, and the table
> below is a same-device comparison.

### Inverse (`log_density`) — the win

| N | torch f32 | Alpaka | speedup | torch f64 | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 72.1 ms | **0.58 ms** | **124×** | 74.6 ms | 129× |
| 100 | 74.1 ms | 0.92 ms | 81× | 71.6 ms | 78× |
| 1 000 | 71.3 ms | 1.53 ms | 47× | 72.7 ms | 47× |
| 10 000 | 70.4 ms | 6.04 ms | 12× | 74.5 ms | 12× |
| 50 000 | 74.6 ms | 23.0 ms | **3.2×** | 110.5 ms | 4.8× |

PyTorch's inverse cost is flat at ~72 ms regardless of batch — 40 bisection
iterations, each a handful of launches. The kernel's cost grows with N because
it is doing the arithmetic. **The speedup therefore falls as N rises**: 124× at
one point, 3.2× at fifty thousand.

### Forward (`sample`) — the regression

| N | torch f32 | Alpaka | speedup | torch f64 | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.10 ms | **0.47 ms** | **8.7×** | 4.41 ms | 9.3× |
| 100 | 4.25 ms | 0.70 ms | 6.1× | 4.18 ms | 6.0× |
| 1 000 | 5.54 ms | 1.30 ms | 4.3× | 4.61 ms | 3.5× |
| 10 000 | 4.25 ms | 6.78 ms | **0.6×** | 8.97 ms | 1.3× |
| 50 000 | 12.87 ms | 22.5 ms | **0.6×** | 55.9 ms | 2.5× |

**At N ≥ 10 000 the kernel is slower than PyTorch at the forward.** PyTorch's
forward is a handful of well-vectorised float32 operations with almost no
launch cost to amortise; the kernel evaluates in `float64`, recomputes all 32
bins per element, and pays a host↔device copy. On a consumer GPU `float64` runs
at 1/32 of the `float32` rate, so the precision choice alone is expensive.

### What a full step looks like

A marginal evaluation step does both: sample, then score. Summing the two
columns at pipeline-relevant sizes:

| N | torch f32 (fwd+inv) | Alpaka (fwd+inv) | combined |
|---:|---:|---:|---:|
| 1 000 | 76.9 ms | 2.83 ms | **27×** |
| 10 000 | 74.7 ms | 12.8 ms | **5.8×** |
| 50 000 | 87.5 ms | 45.5 ms | **1.9×** |

The inverse dominates the total, so the win survives — but it shrinks quickly.
At the pipeline's training batch (32 768 points) expect roughly **2.5–4×**, not
the 10× that the incorrect comparison suggested.

### What is claimed

- The kernel removes the fixed cost of the bisection loop: **124× at N=1** on
  the inverse, falling to 3.2× at N=50 000.
- A combined marginal step is **5.8×** faster at N=10 000 and **1.9×** at
  N=50 000, single-process, `float64`.
- Nothing about the forward at large N: it is **currently a regression**.

---

## 3. Parity with the reference

`tests/test_cubic_spline_parity.py`, 12 tests, all passing:

| property | tolerance | result |
|---|---|---|
| forward `x` | 1e-12 abs + rel | pass |
| forward `logabsdet` | 1e-12 | pass |
| inverse `u` | 1e-12 | pass |
| inverse `logabsdet` | 1e-12 | pass |
| `log_density` | 1e-12 | pass |
| round trip `x→u→x` | 1e-10 | pass |
| cross-backend agreement | 1e-12 | pass |

Covered batch sizes: 1, 7, 512, 5 000. Backends: `cpu_serial`,
`cpu_omp2_blocks`, `cuda`.

`1e-12` is a demanding threshold in `float64` — it allows roughly a thousand
ULPs at unit magnitude, i.e. differences in operation order and `exp`/`log`
implementations only. A real formula mismatch fails by orders of magnitude, not
by a little.

The only thing not exercised is `n_bins` other than 32. Kernels for 8, 16 and
64 are instantiated and compile, but only 32 has a reference model to compare
against.

---

## 4. What this establishes for Phase 1

1. **The kernel is correct.** Not "close enough" — within 1e-12 of the
   reference on every backend.
2. **The premise was right.** The reference is launch-bound, and fusing the
   per-element computation into one kernel recovers 10–15× on the inverse.
3. **The interface exists and is frozen at 0.1.0**, which is what Brown needs
   before changing the architecture.

What it does **not** establish: that this speedup moves the MANGO throughput
gate. The gate is total event-generation rate against a 32-core CPU reference,
and the matrix element dominates that at large batch. The kernel removes a
marginal-side fixed cost; whether that is 80 % or 5 % of a production step
depends on the workload, and the upstream measurements exist to answer it.

---

## 5. Reproducing

See `REPRODUCTION_COMMANDS.md`. `REPRODUCTION_COMMANDS.md` §8 lists the values
to expect; timing columns vary by ±20 % with machine state, the parity
thresholds do not vary at all.
