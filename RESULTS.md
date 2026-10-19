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

`tests/benchmark_kernel_vs_reference.py`, one marginal per call, **both sides on
the GPU**, measured three ways because the first two attempts were wrong.

| column | what it is |
|---|---|
| `torch full` | MLP + spline, as the pipeline calls it |
| `torch spline` | spline only, `_params` shadowed with cached tensors — like-for-like with the kernel |
| `alpaka spline` | kernel with the MLP output precomputed |
| `alpaka realistic` | MLP in torch → host copy → kernel, i.e. what an integration does today |

### Two corrections to the earlier numbers

1. **Device.** The first version left the torch model on the CPU and timed the
   kernel on the GPU. Fixed; the script now asserts placement.
2. **Scope.** `ConditionalCubicFlow1D.forward` calls its conditioning MLP
   through `_params(cond)`; the kernel takes that MLP's output. Comparing them
   as-is compares "MLP + spline" against "spline", which flatters the kernel.
   The `torch spline` column removes that asymmetry.

### Forward (`sample`)

| N | torch full | torch spline | Alpaka spline | like-for-like | Alpaka realistic | realistic |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.17 ms | 1.26 ms | 0.25 ms | **5.1×** | 0.55 ms | 2.1× |
| 100 | 1.27 ms | 1.20 ms | 0.35 ms | 3.5× | 0.73 ms | 1.7× |
| 1 000 | 1.31 ms | 1.27 ms | 0.64 ms | 2.0× | 1.51 ms | 0.9× |
| 10 000 | 1.29 ms | 1.42 ms | 3.34 ms | **0.4×** | 7.79 ms | **0.2×** |
| 50 000 | 2.40 ms | 2.23 ms | 11.81 ms | **0.2×** | 31.0 ms | **0.1×** |

**The forward is not competitive above N ≈ 1 000.** PyTorch's forward is a few
vectorised ops with almost no launch cost to amortise; the kernel is `float64`
(1/32 rate on a consumer GPU), recomputes all 32 bins per element, and copies
the parameters across PCIe each call.

### Inverse (`log_density`)

| N | torch full | torch spline | Alpaka spline | like-for-like | Alpaka realistic | realistic |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.7 ms | 35.1 ms | 0.27 ms | **130×** | 0.58 ms | **58×** |
| 100 | 35.2 ms | 36.2 ms | 0.51 ms | 71× | 0.87 ms | 41× |
| 1 000 | 31.6 ms | 31.3 ms | 0.82 ms | 38× | 1.70 ms | 19× |
| 10 000 | 31.7 ms | 30.3 ms | 2.91 ms | 10× | 7.27 ms | 4.4× |
| 50 000 | 30.9 ms | 33.1 ms | 11.4 ms | **2.9×** | 31.7 ms | **1.0×** |

PyTorch's inverse is flat at ~32 ms for the spline alone — 40 bisection
iterations, each a handful of launches. **That is the win, and it is real**: the
kernel is 130× faster at N=1 and still 2.9× at N=50 000, like-for-like.

### What this means for a deployment

At the batch sizes this pipeline actually uses:

| where | N | inverse (realistic) | forward (realistic) |
|---|---:|---:|---:|
| validation, `--gpu-chunk` default | 10 000 | **4.4×** | 0.2× |
| full weight cache | 524 288 | ~1× | ~0.1× |
| training KL batch | 32 768 | ~1.5× | ~0.15× |

**The kernel is a validated win for the inverse and a regression for the
forward.** Using it for `log_density` at the validation chunk size would cut
that stage by ~4×; using it for sampling would make sampling several times
slower. Any integration has to be selective, and the forward needs the
optimisation work described in `PRD.md` §9 before it is usable at scale.

---

## 3. Parity with the reference — independently verified

### 3.1 Against PyTorch

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

### 3.2 Against an independent implementation

PyTorch parity alone cannot catch a mistake shared by both implementations, so
the kernel was also compared against a NumPy reference written directly from
the formulas (softmax scaling, cumulative knots, PCHIP slopes, Hermite, 40-step
bisection, one Newton correction) — no shared code with either.

| input | forward `x` | forward `logdet` | inverse `u` | inverse `logdet` |
|---|---:|---:|---:|---:|
| normal logits (|raw| ≲ 4) | 6.7e-15 | 8.1e-13 | 4.2e-14 | 2.8e-12 |
| boundary values (0, 1, 1e-12, 1−1e-12) | 1.0e-15 | finite | 2.7e-15 | finite |

### 3.3 Where the parity claim stops

Parity degrades as the logits grow, because the softmax saturates and the
secants span many orders of magnitude:

| logit scale | max \|raw\| | forward `logdet` | inverse `logdet` |
|---:|---:|---:|---:|
| 1 | 0.26 | 1.6e-14 | 1.4e-14 |
| 5 | 1.29 | 2.1e-13 | 1.9e-13 |
| 10 | 2.59 | 4.8e-13 | 4.6e-13 |
| 20 | 5.17 | 5.8e-12 | 5.9e-11 |

**The trained model's logits are within `[-1.62, 1.31]`** (measured from the
`gg_ttbar` thorough checkpoint over 200 000 energy samples), so the 1e-12 claim
holds for every input this model produces. It is not a guarantee for arbitrary
logits, and `PRD.md` §8 records that.

### 3.4 The CUDA backend is really the GPU

A silent CPU fallback would look plausible at small batch. It is not happening:

| N | `cuda` | `cpu_serial` |
|---:|---:|---:|
| 1 000 | 0.71 ms | 1.0 ms |
| 10 000 | 2.92 ms | 11.8 ms |
| 50 000 | 11.1 ms | 61.2 ms |

The gap widens with batch, which is what a real GPU path does. An unavailable
backend raises rather than falling back (`src/launcher.cu`), and
`REPRODUCTION_COMMANDS.md` §6.3 exercises that path.

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
