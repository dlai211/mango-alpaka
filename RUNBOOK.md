# RUNBOOK — building mango-alpaka

Companion to `REPRODUCTION_COMMANDS.md`, which lists the commands to regenerate
results. This file explains the environment and, more importantly, the five
toolchain traps that cost time on the first build.

---

## 1. Prerequisites

Verified versions from the development machine (Windows 11):

| component | version | how to check |
|---|---|---|
| CMake | 3.28.1 | `cmake --version` |
| MSVC | 19.41.34120 | `vcvars64.bat` then `cl` |
| Windows SDK | 10.0.22621.0 | `rc.exe` must resolve after vcvars |
| CUDA Toolkit | 12.4.131 | `nvcc --version` |
| GPU | RTX 3060 Laptop, 6 GB, sm_86 | `nvidia-smi` |
| Python | 3.10.14 | `python --version` |
| Alpaka | **1.3.0** | `third_party/alpaka-1.3.0` |
| Boost headers | **1.74.0** | `Library/include/boost/version.hpp` |
| pybind11 | 3.1.0 | `python -m pybind11 --cmakedir` |

Python packages used by the tests and benchmarks: `numpy`, `torch` (CUDA
build), `pytest`. The kernel itself does not depend on torch.

### 1.1 Interpreter and torch build

Every build, test and benchmark command in this repo uses the `workspace`
conda environment:

```
C:\Users\laiji\.conda\envs\workspace\python.exe
```

It carries **`torch 2.3.0+cu121`**. Do **not** use the PATH default
(`C:\ProgramData\anaconda3\python.exe`): its user site-packages holds
`torch 2.3.0+cpu`, which is CPU-only and fails to import (`shm.dll`). A `+cpu`
wheel reports `torch.cuda.is_available() == False`, so any timing taken with it
is a CPU timing that looks like a GPU result.

Check before trusting a benchmark:

```cmd
"%PY%" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.3.0+cu121 True NVIDIA GeForce RTX 3060 Laptop GPU`.

The upstream repo carries the stronger check, `scripts/verify_torch_gpu.py`,
which also runs the trained conditional marginal on both devices and requires
float64 parity. Last run there: 18.1× on a 4096² float32 matmul, 3.6× on the
marginal at batch 65,536, `max |x_cuda - x_cpu| = 2.0e-15`.

---

## 2. Dependencies

### 2.1 Alpaka

```bash
bash scripts/fetch_dependencies.sh
```

This clones `alpaka-group/alpaka` at tag `1.3.0` into
`third_party/alpaka-1.3.0/`. It is not committed.

**Do not substitute another version** — see §5.1.

### 2.2 Boost headers

Alpaka's CMake calls `find_package(Boost 1.74 REQUIRED)`, so it needs a proper
CMake config, not just an include directory. If a Boost CMake package is not
already on the machine, conda-forge supplies one:

```bash
conda install -n <env> -c conda-forge boost-cpp -y
```

Check it will not disturb an existing environment before committing:

```bash
conda install -n <env> -c conda-forge boost-cpp --dry-run
```

On this machine that added `boost-cpp 1.74.0`, `lz4-c` and `zstd`, and updated
only `ca-certificates` — no Python, NumPy or torch changes.

Its CMake config lands in `<env>/Library/lib/cmake/Boost-1.74.0/`, which is why
`build.bat` passes `-DCMAKE_PREFIX_PATH=<env>/Library`.

### 2.3 pybind11

```bash
python -m pip install pybind11
```

The build discovers it either from `CMAKE_PREFIX_PATH` or by running
`python -m pybind11 --cmakedir`. Override the interpreter with `-DMANGO_PYTHON=`
if the build should use a different one.

---

## 3. Build

```cmd
build.bat
```

The script sources `vcvars64.bat`, configures with Ninja, and builds. It
hard-codes the conda environment path; edit the two `-DCMAKE_PREFIX_PATH` /
`-DBoost_ROOT` arguments for a different machine.

Equivalent manual invocation:

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=nvcc ^
      -DCMAKE_CUDA_ARCHITECTURES=86 ^
      -DCMAKE_PREFIX_PATH="<env>/Library" -DBoost_ROOT="<env>/Library" ^
      -DMANGO_PYTHON="<env>/python.exe"
cmake --build build --parallel
```

CMake options:

| option | default | effect |
|---|---|---|
| `MANGO_ENABLE_CUDA` | ON | build `AccGpuCudaRt` backend |
| `MANGO_ENABLE_OMP2` | ON | build `AccCpuOmp2Blocks` backend |
| `MANGO_ENABLE_SERIAL` | ON | build `AccCpuSerial` backend |

Turning off CUDA also removes the CUDA language requirement in principle, but
the project declaration still enables it — build CPU-only by setting
`-DMANGO_ENABLE_CUDA=OFF` and removing `CUDA` from `project(... LANGUAGES ...)`
if a machine has no toolkit.

### Artifacts

```
build/python/mango_alpaka_ext.cp310-win_amd64.pyd
build/python/cudart64_12.dll
```

The DLL is copied next to the module on purpose — see §5.4.

---

## 4. Verify

```cmd
set PY=C:\Users\laiji\.conda\envs\workspace\python.exe
set PYTHONPATH=build\python
"%PY%" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
"%PY%" -c "import mango_alpaka_ext as m; print(m.backends())"
"%PY%" -m pytest tests\test_cubic_spline_parity.py -q
```

Expected: `2.3.0+cu121 True`, all three backends listed, and **26 passed**
(13 tests per precision). Use the `workspace` interpreter, not whatever
`python` resolves to — see §1.1.

A quick numeric check that does not need the reference repo:

```python
import numpy as np, mango_alpaka_ext as m
rng = np.random.default_rng(0)
raw = rng.normal(size=(5, 64)); u = rng.random(5)
for be in m.backends():
    print(be, m.forward(raw, u, backend=be)[0][:3])
```

All three backends must print the same numbers.

---

## 5. Toolchain traps

Each of these was hit during the first build. They are recorded because the
error messages point away from the cause.

### 5.1 `Idx is not a template` — Alpaka version

```
alpaka/mem/buf/BufUniformCudaHipRt.hpp(81): error: Idx is not a template
```

**Cause:** Alpaka 1.0.0 uses an unqualified `Idx<TExtent>` in a dependent
`static_assert`. Under nvcc 12.4 that resolves to whatever `Idx` is visible,
not to `alpaka::Idx`, and fails inside Alpaka's own header.

**Fix:** use **1.3.0**, which writes `alpaka::Idx<TExtent>`. This is why
`scripts/fetch_dependencies.sh` pins the version.

The same message also appears if *your* code declares a global `using Idx = …`
in a translation unit that includes Alpaka — it shadows the trait during
Alpaka's header parse. Name the alias something else (`IdxT`).

### 5.2 "No CUDA toolset found"

```
CMake Error: No CUDA toolset found.
```

**Cause:** the Visual Studio generator requires the NVIDIA integration
component for VS, which is not installed here.

**Fix:** use Ninja with `-DCMAKE_CUDA_COMPILER=nvcc`.

### 5.3 `rc.exe` / `CMAKE_MT-NOTFOUND`

```
RC Pass 1: command "rc /fo ..." failed
```

**Cause:** `cl.exe` is on PATH because the VC bin directory is, but `rc.exe`
and `mt.exe` come from the **Windows SDK** and are only added by `vcvars64.bat`.
CMake's compiler check needs them even for a header-only project.

**Fix:** source `vcvars64.bat` before configuring — `build.bat` does.

### 5.4 `DLL load failed` on import

```
ImportError: DLL load failed while importing mango_alpaka_ext
```

**Cause:** CPython 3.8+ stopped searching `PATH` for the dependencies of an
extension module. `cudart64_12.dll` is on PATH and still not found.

**Fix:** the build copies `cudart64_12.dll` beside the `.pyd`. Linking the CUDA
runtime statically also fixes it but breaks Alpaka's CMake, which expects the
`CUDA::*_static` targets to exist.

### 5.5 "compiler has to support CUDA"

```
alpaka/atomic/AtomicUniformCudaHip.hpp: #error: If ALPAKA_ACC_GPU_CUDA_ENABLED
is set, the compiler has to support CUDA!
```

**Cause:** a `.cpp` translation unit compiled by MSVC while Alpaka's CUDA
backend macros are defined.

**Fix:** files that include Alpaka **and** the CUDA headers must be `.cu`.
`src/bindings.cpp` deliberately includes neither, which is also why pybind11 and
nvcc never meet in one translation unit.

---

## 6. Clean rebuild

```cmd
rmdir /s /q build
build.bat
```

The Alpaka clone is reusable; only `build/` is removed. A clean configure takes
about 15 s and a full build about 2 min on this machine (CUDA compilation of
`launcher.cu` dominates).

---

## 7. What is not covered

- **Linux / Talapas.** The kernel is portable C++17 but `build.bat` is
  Windows-only. A CMake invocation with GCC/Clang and `-DCMAKE_CUDA_COMPILER=`
  should work; it has not been tried.
- **AMD / Intel backends.** Alpaka supports them; this project does not enable
  them. Adding one is a `alpaka_ACC_*` cache variable plus a dispatch case.
- **The reference training loop.** It stays in the upstream repository. The
  kernel itself supports both `float32` and `float64` (`Precision` in the
  interface contract; 13 tests per dtype).
- **Packaging.** `build/python` is the install location; there is no wheel.
