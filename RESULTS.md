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

## 2. The Alpaka kernel removes it

`tests/benchmark_kernel_vs_reference.py`, dim=2, `float64`, end to end through
the Python binding. Both columns include the host↔device copies the C++ launcher
performs, so this is not a kernel-only comparison.

| N | torch forward | Alpaka forward | speedup | torch inverse | Alpaka inverse | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.30 ms | 1.21 ms | 1.1× | 16.09 ms | **1.35 ms** | **11.9×** |
| 100 | 2.28 ms | 1.58 ms | 1.4× | 25.48 ms | 1.64 ms | **15.5×** |
| 1 000 | 5.19 ms | 2.18 ms | 2.4× | 28.65 ms | 2.30 ms | **12.5×** |
| 10 000 | 30.0 ms | 6.50 ms | 4.6× | 69.4 ms | 7.22 ms | **9.6×** |
| 50 000 | 193.2 ms | 29.3 ms | 6.6× | 306.1 ms | 29.0 ms | **10.6×** |

### Reading this honestly

**The inverse speedup is roughly flat in N** — 11.9× at one point, 15.5× at a
hundred, 10.6× at fifty thousand. Removing a fixed per-call cost produces a
flat curve; making arithmetic faster would produce a curve that shrinks as the
problem grows. This is the signature we were looking for.

**The forward speedup grows and then flattens** (1.1× → 6.6×). At N=1 the
Alpaka path is 1.21 ms, which is essentially our *own* launcher's fixed cost:
four host buffers, four device buffers and a synchronise, per call. The kernel
is not the expensive part at that size. A caller-owned-buffer interface would
cut this and is the obvious next optimisation.

**At N=50 000 the forward cost is real.** 29.3 ms of work against 193 ms, and
the gap would narrow further with a stream-based interface.

### What is not claimed

- This is **one marginal per call**. The pipeline evaluates `dim` marginals and
  a copula; the end-to-end gain on a full model is smaller and unmeasured.
- These are **inference-side** operations. Training was not benchmarked.
- The comparison is against **PyTorch eager**, not against a tuned baseline.
  `torch.compile` might close some of the gap for the forward; it cannot remove
  the 40-launch bisection loop without fusing it, which is what the kernel does.
- One GPU, one machine. No claim about portability of the *numbers*, only of
  the code.

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
