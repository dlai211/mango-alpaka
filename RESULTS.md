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
the GPU**. Four columns because two earlier attempts were wrong:

| column | what it is |
|---|---|
| `torch full` | MLP + spline, as the pipeline calls it |
| `torch spline` | spline only, `_params` shadowed with cached tensors — like-for-like with the kernel |
| `alpaka` | kernel with the MLP output precomputed |
| `realistic` | MLP in torch → host copy → kernel, what an integration does today |

### 2.1 Two corrections, then two optimisations

The first measurement ran torch on the **CPU** while timing the kernel on the
GPU. The second compared torch "MLP + spline" against kernel "spline only".
Both inflated the kernel and are now impossible to reproduce accidentally — the
script asserts device placement and reports both scopes.

Then the kernel itself was improved (version 0.2.0):

* **`float32` instantiation.** `float64` runs at 1/32 the rate of `float32` on a
  consumer GPU, and the pipeline is `float32` anyway.
* **Caller-memory views instead of host buffers.** Each call previously
  allocated eight buffers and performed six memory operations; the host side is
  now a view over the caller's arrays.

Effect, like-for-like at N = 10 000: forward **0.4× → 1.3×**, inverse
**10× → 33×**. At N = 50 000 the inverse went **2.9× → 12.7×**.

### 2.2 Forward (`sample`), float32

| N | torch spline | Alpaka | like-for-like | realistic | realistic × |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.26 ms | 0.12 ms | **10.4×** | 0.42 ms | 2.9× |
| 100 | 1.31 ms | 0.13 ms | 9.8× | 0.44 ms | 2.7× |
| 1 000 | 1.32 ms | 0.19 ms | 7.1× | 0.55 ms | 2.4× |
| 10 000 | 1.30 ms | 1.00 ms | **1.3×** | 1.65 ms | 0.8× |
| 50 000 | 1.39 ms | 2.48 ms | **0.6×** | 5.47 ms | 0.3× |

Faster up to N ≈ 1 000, roughly at parity at 10 000, still behind at 50 000.
The residual gap is the 12.8 MB host→device copy of `raw` at N=50 000 plus the
MLP in the realistic path. Closing it needs device pointers (zero-copy), which
is an interface change, not a kernel change.

### 2.3 Inverse (`log_density`), float32

| N | torch spline | Alpaka | like-for-like | realistic | realistic × |
|---:|---:|---:|---:|---:|---:|
| 1 | 31.4 ms | 0.12 ms | **254×** | 0.41 ms | **77×** |
| 100 | 32.0 ms | 0.14 ms | 233× | 0.43 ms | 73× |
| 1 000 | 32.6 ms | 0.19 ms | 176× | 0.54 ms | 60× |
| 10 000 | 32.2 ms | 0.97 ms | 33× | 1.69 ms | 20× |
| 50 000 | 31.3 ms | 2.48 ms | **12.7×** | 5.42 ms | 5.9× |

PyTorch's inverse is flat at ~32 ms at every batch size — 40 bisection
iterations, each a handful of launches. **This is a win at every size**, and
still ~6× through a realistic integration at N = 50 000.

### 2.4 `float64` for reference

`float64` remains the parity reference but is not the deployment dtype. Same
script, `float64` model and buffers:

| N | forward like-for-like | inverse like-for-like |
|---:|---:|---:|
| 1 | 6.2× | 152× |
| 1 000 | 3.4× | 43× |
| 10 000 | 0.7× | 16× |
| 50 000 | 0.3× | 4.8× |

The forward in `float64` is a regression above N ≈ 1 000. That is the precision
cost, not the kernel: `float32` is 2.5–3.9× faster in the same code.

### 2.5 What this means for a deployment

At the batch sizes this pipeline uses, `float32`:

| where | N | inverse (realistic) | forward (realistic) |
|---|---:|---:|---:|
| validation, `--gpu-chunk` default | 10 000 | **20×** | 0.8× |
| training KL batch | 32 768 | ~9× | ~0.5× |
| full weight cache | 524 288 | ~3× | ~0.2× |

**The kernel is a validated win for the inverse at every size, and for the
forward below ~10 000.** For sampling at training scale it remains slower;
that is the open item in `PRD.md` §9.

## 3. Parity with the reference — independently verified

### 3.1 Against PyTorch

`tests/test_cubic_spline_parity.py`, **26 tests, all passing** (13 per
precision):

| property | float64 tolerance | float32 tolerance | result |
|---|---|---|---|
| forward `x` | 1e-12 | 2e-5 | pass |
| forward `logabsdet` | 1e-12 | 2e-5 | pass |
| inverse `u` | 1e-12 | 2e-5 | pass |
| inverse `logabsdet` | 1e-12 | 2e-5 | pass |
| `log_density` | 1e-12 | 2e-5 | pass |
| round trip `x→u→x` | 1e-10 | 1e-5 | pass |
| cross-backend agreement | 1e-12 | 2e-5 | pass |

Covered batch sizes: 1, 7, 512, 5 000. Backends: `cpu_serial`,
`cpu_omp2_blocks`, `cuda`.

`float32` is covered at a **separate, measured tolerance of 2e-5** rather than
the same 1e-12. The measured worst case over the model's logit range is 4.8e-7
for `x`, 5.4e-7 for `u` and 9.5e-6 for the log-determinant; 2e-5 is two orders
of margin. Claiming 1e-12 in `float32` would be meaningless.

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

1. **The kernel is correct.** Within 1e-12 of the reference in `float64` on
   every backend, and within a measured 2e-5 in `float32`. Independently
   confirmed against a NumPy implementation (§3.2).
2. **The premise was right.** The reference is launch-bound, and fusing the
   per-element computation into one kernel recovers **12–254× on the inverse**,
   depending on batch size and whether the MLP and copies are included.
3. **The interface is stable at 0.2.0**, which is what Brown needs before
   changing the architecture. 0.2.0 added the precision field; the semantics in
   `docs/INTERFACE_CONTRACT.md` §2.3 are unchanged.
4. **The forward is competitive below ~10 000 points and not above.** Stated
   plainly because it is the half a pipeline uses for sampling.

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
