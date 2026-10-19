# REPRODUCTION_COMMANDS — rebuild and re-measure everything

Every number in `RESULTS.md` is produced by a command in this file. Run them in
order; §8 lists the values to expect.

Paths assume the repository at `C:\Users\laiji\test\mango-alpaka` and the conda
environment at `C:\Users\laiji\.conda\envs\workspace`. Substitute as needed.

---

## 0. Provenance

| item | value |
|---|---|
| repository | `C:/Users/laiji/test/mango-alpaka` |
| upstream reference | `C:/Users/laiji/test/conditional-nf-integration` |
| upstream commit at time of measurement | `e349d15` |
| CMake | 3.28.1 |
| MSVC | 19.41.34120 |
| CUDA | 12.4.131 |
| Alpaka | 1.3.0 |
| Boost headers | 1.74.0 (conda-forge `boost-cpp`) |
| pybind11 | 3.1.0 |
| PyTorch | 2.3.0+cu121 |
| GPU | RTX 3060 Laptop, 6 GB, sm_86 |

---

## 1. One-time setup

```bash
# Alpaka 1.3.0 (1.0.0 does not compile under nvcc 12.4 — see RUNBOOK.md §5.1)
bash scripts/fetch_dependencies.sh

# Boost headers with a CMake package
conda install -n workspace -c conda-forge boost-cpp -y

# pybind11
/c/Users/laiji/.conda/envs/workspace/python.exe -m pip install pybind11
```

---

## 2. Build

```cmd
cd C:\Users\laiji\test\mango-alpaka
build.bat
```

Artifacts:

```
build\python\mango_alpaka_ext.cp310-win_amd64.pyd
build\python\cudart64_12.dll
```

Verify the three backends are present:

```bash
PYTHONPATH="build/python" /c/Users/laiji/.conda/envs/workspace/python.exe -c \
  "import mango_alpaka_ext as m; print(m.backends())"
```

Expected: `['cpu_serial', 'cpu_omp2_blocks', 'cuda']`

---

## 3. Parity against the reference (RESULTS.md §3)

```bash
cd C:/Users/laiji/test/mango-alpaka
PYTHONPATH="build/python" \
  /c/Users/laiji/.conda/envs/workspace/python.exe -m pytest tests/test_cubic_spline_parity.py -q
```

Expected: `12 passed`.

The test needs the upstream repo. It defaults to
`C:/Users/laiji/test/conditional-nf-integration`; override with:

```bash
MANGO_UPSTREAM=/path/to/conditional-nf-integration ...
```

What it covers: forward, inverse, log-density, round trip, contract
conformance, cross-backend agreement — at batch sizes 1, 7, 512 and 5 000, with
`float64`, tolerance `1e-12` absolute and relative.

---

## 4. Pre-kernel fixed-cost baseline (RESULTS.md §1)

This is the measurement of the *existing* PyTorch implementation. It does not
use the compiled module.

```bash
cd C:/Users/laiji/test/mango-alpaka
/c/Users/laiji/.conda/envs/workspace/python.exe tests/benchmark_fixed_cost.py
```

Writes:

```
results/fixed_cost_baseline.csv
results/fixed_cost_baseline.md
```

Defaults: dimensions `2 5 8`, sizes `1 100 512 1000 10000 50000`, devices
`cpu cuda`, 5 warmup calls, best of 20 (fewer for the largest size).

Options:

```bash
python tests/benchmark_fixed_cost.py --dims 2 --sizes 1 10000 --repeats 50
```

---

## 5. Before/after against the kernel (RESULTS.md §2)

```bash
cd C:/Users/laiji/test/mango-alpaka
PYTHONPATH="build/python" \
  /c/Users/laiji/.conda/envs/workspace/python.exe tests/benchmark_kernel_vs_reference.py
```

Compares two `ConditionalCubicFlow1D` marginals (dim=2) in `float64` at
N = 1, 100, 1 000, 10 000, 50 000, for forward and inverse, torch on CUDA
versus the Alpaka kernel on CUDA. Both columns include the host↔device copies
the launcher performs, so the numbers are end to end through the binding.

---

## 6. Individual checks

### 6.1 One forward call, all backends

```bash
PYTHONPATH="build/python" /c/Users/laiji/.conda/envs/workspace/python.exe - <<'EOF'
import numpy as np, mango_alpaka_ext as m
rng = np.random.default_rng(0)
raw = rng.normal(size=(5, 64)); u = rng.random(5)
for be in m.backends():
    x, ld = m.forward(raw, u, backend=be)
    print(be, np.round(x[:3], 8), np.round(ld[:3], 8))
EOF
```

All three lines must be identical.

### 6.2 Contract semantics

```bash
PYTHONPATH="build/python" /c/Users/laiji/.conda/envs/workspace/python.exe - <<'EOF'
import numpy as np, mango_alpaka_ext as m
rng = np.random.default_rng(1)
raw = rng.normal(size=(256, 64)); u = rng.random(256)
x, ld  = m.sample_and_log_prob(raw, u, backend="cuda")
x2     = m.sample(raw, u, backend="cuda")
logq   = m.log_density(raw, x, backend="cuda")
assert np.array_equal(x, x2)
print("max |log_density + logdet| =", np.max(np.abs(logq + ld)))
EOF
```

Expected: the two are negatives of each other to ~1e-12 (uniform base), since
`log_density` evaluates the Jacobian through the inverse.

### 6.3 Error paths

```bash
python -c "
import numpy as np, mango_alpaka_ext as m
try: m.forward(np.zeros((2,64)), np.zeros(2), backend='hip')
except ValueError as e: print('ok:', e)
"
```

An unavailable backend must raise, never fall back.

---

## 7. Where outputs land

| command | output |
|---|---|
| parity tests | stdout only |
| fixed-cost benchmark | `results/fixed_cost_baseline.{csv,md}` |
| before/after | stdout only |

---

## 8. Expected values

Exact times vary with machine state. These are the figures recorded in
`RESULTS.md`; treat ±20 % as normal for the timing columns and **1e-12 as a
hard bound** for parity.

### Parity

| item | expected |
|---|---|
| tests | `12 passed` |
| max |x − x_torch| | ≲ 1e-14 |
| max |logdet − logdet_torch| | ≲ 1e-13 |
| round trip |x→u→x| | < 1e-10 |

### Fixed cost (dim=2)

| device | operation | N=1 | N=10 000 | N=50 000 |
|---|---|---:|---:|---:|
| cpu | sample | 0.94 ms | 13.3 ms | 78.7 ms |
| cpu | log_density | 13.1 ms | 40.3 ms | 127.3 ms |
| cuda | sample | 3.68 ms | 3.70 ms | 13.1 ms |
| cuda | log_density | 69.8 ms | 68.7 ms | 78.8 ms |

### Before/after (both sides on the same GPU, 2 marginal calls)

| N | torch f32 inv | Alpaka inv | inv speedup | torch f32 fwd | Alpaka fwd | fwd speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 72.1 ms | 0.58 ms | **124×** | 4.10 ms | 0.47 ms | 8.7× |
| 100 | 74.1 ms | 0.92 ms | 81× | 4.25 ms | 0.70 ms | 6.1× |
| 1 000 | 71.3 ms | 1.53 ms | 47× | 5.54 ms | 1.30 ms | 4.3× |
| 10 000 | 70.4 ms | 6.04 ms | 12× | 4.25 ms | 6.78 ms | 0.6× |
| 50 000 | 74.6 ms | 23.0 ms | **3.2×** | 12.87 ms | 22.5 ms | 0.6× |

The forward is **slower** in the kernel above N≈10 000. This is a known
regression, not a measurement error: the kernel is `float64` and copies per
call. See `RESULTS.md` §2.

> An earlier revision of this table compared PyTorch on the **CPU** against the
> kernel on the **GPU**, inflating the small-N speedups by about an order of
> magnitude. The script now asserts that both run on the same device.
