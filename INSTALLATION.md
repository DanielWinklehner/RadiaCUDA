# INSTALLATION.md — RadiaCUDA Windows Build Guide

## Prerequisites

### 1. Visual Studio 2022

Download and install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/) (free).

During installation, select the **"Desktop development with C++"** workload. No other workloads are needed.

### 2. Conda (Anaconda or Miniconda)

If you don't already have it, install [Miniconda](https://docs.anaconda.com/miniconda/install/).

---

## Step 1: Create the Conda Environment

Open a plain **Command Prompt** (not PowerShell, not VS Developer Prompt).

```cmd
cd C:\path\to\RadiaCUDA
conda env create -f environment-win.yml
```

This creates an environment called `radiacuda` with Python 3.12, FFTW, Intel MPI, and all other dependencies.

---

## Step 2: Activate Visual Studio Compiler

**Important:** MSVC must be activated *before* conda. In the same Command Prompt:

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

You should see:
```
**********************************************************************
** Visual Studio 2022 Developer Command Prompt
**********************************************************************
```

Verify the compiler is available:
```cmd
cl
```

Expected output (version may differ):
```
Microsoft (R) C/C++ Optimizing Compiler Version 19.44.xxxxx for x64
```

> **Note:** If you have Visual Studio Professional or Enterprise instead of Community,
> replace `Community` with `Professional` or `Enterprise` in the path above.

---

## Step 3: Activate the Conda Environment

```cmd
conda activate radiacuda
```

Verify both the compiler and conda tools are visible:
```cmd
cl
where cmake
where ninja
where python
```

All four commands should produce output. `cmake`, `ninja`, and `python` should point to
paths inside `anaconda3\envs\radiacuda\`.

---

## Step 4: Set Build Type

Force a Release build (required — Debug builds produce incorrect numerical results):

```cmd
set CMAKE_BUILD_TYPE=Release
```

---

## Step 5: Build and Install

```cmd
cd C:\path\to\RadiaCUDA
pip install . -v --no-build-isolation
```

The build takes approximately 30-60 seconds. You should see:
- `[43/43]` compilation steps completing
- `Successfully built radia`
- `Successfully installed radia-4.200`

---

## Step 6: Verify the Installation

### Basic test

```cmd
python -c "import radia as rad; print('Radia version:', rad.UtiVer())"
```

### Run Example 1

```cmd
python examples\RADIA_Example01.py
```

Expected output:
```
RADIA Library Version: X.XXX

Values close to [0.12737, 0.028644, 0.077505] are expected.

[0.1273..., 0.0286..., 0.0775...]
```

### Run all examples

```cmd
python examples\RADIA_Example01.py
python examples\RADIA_Example02.py
python examples\RADIA_Example03.py
python examples\RADIA_Example04.py
python examples\RADIA_Example05.py
python examples\RADIA_Example06.py
```

Examples 2-6 produce matplotlib plots. Close each plot window to continue execution.

---

## Step 7: Verify MPI (Optional)

### Without mpi4py

```cmd
mpiexec -n 4 python -c "import radia as rad; r = rad.UtiMPI('on'); print(f'Rank: {r}', flush=True); rad.UtiMPI('off')"
```

Expected: four lines showing Ranks 0, 1, 2, 3.

### With mpi4py

```python
from mpi4py import MPI
import radia as rad

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Use 'in' (not 'on') when MPI is already initialized by mpi4py
result = rad.UtiMPI('in')
print(f"Rank {rank} of {size}, Radia rank: {result}", flush=True)
rad.UtiMPI('off')
```

```cmd
mpiexec -n 4 python test_radia_mpi.py
```

---

## Troubleshooting

### "No CMAKE_C_COMPILER could be found"

You forgot to activate MSVC. Run `vcvars64.bat` first (Step 2), then activate conda (Step 3).

### "Unsupported CMAKE_ARGS ignored: -DCMAKE_BUILD_TYPE=Release"

This warning is harmless. The `set CMAKE_BUILD_TYPE=Release` environment variable ensures
the correct build type is used.

### Solver does not converge / produces garbage results

Check that you are building in Release mode. In the build output, look for `/O2` (optimized).
If you see `/Od` (no optimization) and `-MDd` (debug runtime), the build is in Debug mode.
Re-run Step 4 and rebuild.

### "This function is not implemented on that platform" from ObjDrwOpenGL

This is expected — the OpenGL 3D viewer is not available on Windows. The examples handle
this gracefully and skip 3D visualization.

### MPI returns all zeros

- Verify `mpiexec` points to conda's Intel MPI: `where mpiexec`
- If using mpi4py, use `rad.UtiMPI('in')` instead of `rad.UtiMPI('on')`
- Do not mix system-installed MS-MPI with conda's Intel MPI

### LNK4006 warnings during build

If you see `LNK4006` warnings about duplicate symbols, ensure `radplnr2_old.cpp` has been
deleted from `cpp/src/core/`.

---

## Quick Reference: Complete Build Sequence

For returning users, the complete build sequence in a single block:

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
conda activate radiacuda
set CMAKE_BUILD_TYPE=Release
cd C:\path\to\RadiaCUDA
pip install . -v --no-build-isolation
python -c "import radia as rad; print('Radia version:', rad.UtiVer())"
```