# Assembly -> solve hand-off: state and measurements

Working note for picking this up in a fresh session. Everything below was
measured on an **RTX 4070 Ti SUPER (16376 MiB, CC 8.9, PCIe 4.0 x16)** with
**127.8 GiB host RAM** unless stated otherwise. Numbers carry run-to-run noise
of roughly +/-30% on the streaming figures; treat trends as solid and single
values as indicative.

---

## 1. Why this work exists

The dense interaction matrix is `36*N^2` bytes (9 floats per element pair). It
is the only O(N^2) structure anywhere in the pipeline; everything else is O(N).

Two separate limits follow:

* **VRAM** capped model size at ~17.5k elements with a desktop running
  (13.4 GiB free of 16). Addressed by `UtiGpuFallback('gpu_streaming')`, which
  keeps rows resident and streams only the overflow. **Done.**
* **Host RAM** bound next, because *two* full host copies of the matrix existed
  at once (`InteractMatrix` as `TMatrix3df**`, plus the assembly's flat
  buffer), so peak was `2 * 36*N^2`. **Fixed by phase 3.**

Concretely: the 44k all-tet 60 MeV model has a 69.7 GB matrix and needed
**~139 GB** of host RAM against ~94 GB free — it did not fit. At one
representation it needs 69.7 GB and fits.

Measured, 19683 elements (13.95 GB matrix), assembly + solve, peak process
commit above baseline, `UtiGpuFallback('gpu_streaming')`:

| | before | after |
|---|---|---|
| after `RlxPre` | 2.88 x matrix | 1.88 x |
| after `RlxAuto` | 3.03 x | 2.04 x |

Exactly one `36*N^2` host copy removed. The ~1 x that remains beyond the matrix
itself is **not** a host copy: on Windows/WDDM a device allocation carries a
same-size system-memory backing store in the process's commit charge, so it is
bounded by VRAM (~13-15 GB here) and does **not** grow with N. For the 44k
model that puts the requirement at ~70 GB + ~13 GB rather than ~139 GB + ~13 GB.

---

## 2. What is already committed

| commit | what |
|---|---|
| `581a3ce` | streaming matvec: page-lock the matrix in place, resident row block |
| `2004b80` | `RADIA_RLX_DEBUG` matrix-pass counter |
| `85228e9` | opt-in Galerkin assembly (+ `AmOfWarnings` 15->16 fix) |
| `cecf9b2` | **phase 1** — assembly emits the solver's layout directly |
| `00e123d` | **phase 2** — device hand-off; cold solves ~8x faster |
| (this) | **phase 3** — one host representation at a time; peak `36*N^2` |

### Phase 1 (done)
The assembly kernel writes **scalar row-major** (`[N3 x N3]`, `N3 = 3N`) instead
of block-major `[N*N*9]`, and took over the two pieces of semantics that used to
live in the host unpack:

* `MainTransPtrArray[i]->TrMatrix_inv`, i.e. left-multiply by `s*M_inv`
  (`gmtrans.h:79`). Extracted through the public virtual by transforming the
  identity — `TrMatrix_inv(I) == s*M_inv` exactly — so composite transforms stay
  correct and `radIdentTrans` rows come back as the identity.
* the non-finite ("sliver tet") backstop, now counted on the device and reported
  once with the offending element pair.

`radGPU_UnpackMatrix` is now a pure layout read.

### Phase 2 (done)
When the whole matrix fits in VRAM the assembly **stages** its device buffer and
`radGPU_PublishAssembledMatrix(stamp)` adopts it into the solver's resident
cache. This happens in `radintrc.cpp` right after `mGpuMatrixStamp` is issued,
because the stamp is assigned *after* `SetupInteractMatrix()` returns — the
assembly cannot publish under it directly.

The first solve then finds the cache warm and skips both the O(N^2) host flatten
and the H2D upload:

```
 elems   matrix     cold before   cold after    warm
  1000     34MB        0.77 ms      0.16 ms    0.15 ms
  2197    166MB        3.29 ms      0.54 ms    0.52 ms
  4096    576MB       11.24 ms      1.42 ms    1.45 ms
  5832   1168MB       21.08 ms      2.79 ms    2.78 ms
```

Matters most for **shim optimization**, where every iterate changes the geometry
so the cache never hit and the full cost was paid every solve.

Not published when assembly ran on the CPU or was tiled (only one tile is
resident then); the staged buffer is released on every error path.

---

## 3. Phase 3 (done)

**Goal:** stop materializing `InteractMatrix` when the GPU assembled the matrix,
so peak host memory drops from `2*36N^2` to `36N^2`.

### 3.1 What it does

The interaction now keeps the assembly's own output —
`radTInteraction::mAsmMatrix`, scalar row-major `float[N3 x N3]`, the layout the
matvec wants — instead of unpacking it into `InteractMatrix`. The invariant is
**one host representation at a time**:

* GPU assembled → `mAsmMatrix` holds it, `InteractMatrix` is null.
* CPU assembled → `InteractMatrix` holds it, `mAsmMatrix` is null.
* `EnsureInteractMatrix()` converts the first into the second on demand and
  **frees `mAsmMatrix`**, so the CPU-solve path never holds both (that would put
  peak straight back at `2*36N^2`).

A GPU solve with `skipMatrix == 0` **borrows** `mAsmMatrix` as `h_matrix`
(`RadGPURelaxData::h_matrixOwned == 0`) — no allocation and no O(N^2) flatten,
which also removes the second copy that used to appear during the solve.

Deliberately *not* done: filling from the device cache, as §3.5 of the original
plan suggested. The device cache is volatile — solving a different interaction
`cudaFree`s it (`radgpurlx.cu`, cache-miss branch) — so an interaction whose only
copy lived there would silently lose its matrix. The host buffer is the durable
representation; the device cache stays a pure accelerator. No getter needed.

### 3.2 The chokepoints

**The original claim in this section was wrong** — it said every relaxation
method is constructed in `MakeManualRelax` / `MakeAutoRelax`, so `radrlmet.cpp`
need not be touched. Re-grepping found two more construction sites *inside*
`radrlmet.cpp`: `radTRelaxationMethNo_6::AutoRelax` (`:1648`) and
`radTRelaxationMethNo_7::FillInSubMatrixArrays` (`:1933`), both on interactions
those methods build themselves (reachable from Python as `Solve(..., meth=6|7)`),
and `MethNo_7` also reads `InteractMatrix` directly in four of its members.

So the guard went into the **common base constructor**,
`radTIterativeRelaxMeth(radTInteraction*)` in `radrlmet.h` — one line that covers
every method, present and future, including the internally-constructed ones.
Plus, so nothing depends on statement order inside those two methods,
`EnsureInteractMatrix()` right where they create their interactions.

`radapl2.cpp` still calls it at the user-facing entry points, for the error
*path* rather than for coverage: a failure there returns `Radia::Error118`
cleanly instead of crashing inside a constructor. Methods 9/10/11 call it only
in the branches that actually leave the GPU.

Also guarded: `ShowInteractMatrix` (`radintrc.h`) and `DumpBin`
(`radintrc.cpp`) — without the latter a dump would silently come out without a
matrix.

### 3.3 The trap: self-blocks run on every solve (fixed first)

`radGPU_PackInteractionData` reads `InteractMatrix[i][i]` for the
self-interaction diagonal **outside** the `skipMatrix` guard, so it runs on every
GPU solve; had it triggered materialization, phase 3 would have bought nothing.
It now reads the 9 floats per element from whichever representation exists —
`mAsmMatrix` when the GPU assembled. Still O(N), and no device round trip.

### 3.4 Deferred allocation

`AllocateMemory` no longer allocates `InteractMatrix`; that moved to
`AllocateInteractMatrix()`, called from `SetupInteractMatrix` immediately before
the CPU assembly (master rank only, mirroring `IntrctMatrMemAllocShouldBeDone`)
and from `EnsureInteractMatrix()`.

Two latent bugs in the moved code, fixed in passing: the row-by-row failure path
freed `InteractMatrix[i]` in a loop over `k` (should be `[k]`), and the
contiguous allocation computed `AmOfMainElem*AmOfMainElem` in `int` — overflow
above ~46k elements, inside the range this work is aimed at. Separately, the
binary-parse constructor never initialized `InteractMatrix`, so a stream
carrying no matrix left it holding garbage for `DeallocateMemory` to delete.

---

## 4. Verification

**Non-negotiable:** bit-identity against the previous build. The harness is no
longer scratch — it is `studies/fingerprint.py`:

* four geometry cases — plain, **rotated**, mirrored, rotated+mirrored
* each solved four ways — `m4` (CPU), `m9` (GPU Jacobi), `m11` (GPU NK),
  `man` (`RlxMan`) — so the assembly's output is checked through every route
  it reaches a solver by
* full precision (`%.17e`) B at three points, plus misfit and iteration count
* build reference, run, `git stash` the change, rebuild, run, diff — must be
  empty. Self-reproducible run to run (`FldLenRndSw('off')`, CPU field eval).

Phase 3 is bit-identical across all 16 cases.

**The rotated case is the one that matters.** It is the only one with a
non-identity `MainTransPtrArray`, so it is the only one that exercises the row
transform at all. With a plain block that code path never runs.

**Gap closed:** `tests/test_gpu_asm_cpu_solve.py` covers **GPU-assembled ->
CPU-solved**, the path phase 3 could break (null `InteractMatrix`), written
before the phase-3 edits and passing on both builds. Routes used:

* anisotropic linear material (`KsiPar != KsiPerp`, Warning022) — the sharp one:
  the GPU solver hands off to CPU method 10, so methods 9 and 11 must reproduce
  an explicit method-10 call **bit for bit**;
* `RlxMan` on a GPU-assembled matrix; CPU solve then GPU solve on one
  interaction; a repeated GPU solve (warm device cache — the self-blocks path);
* the assembly declining: `RlxPre(use_gpu=False)`, and an extruded polygon
  (Warning020, no GPU kernel — the same fall-through an oversized model takes,
  without its O(N^2) CPU assembly);
* a genuinely VRAM-oversized model under `UtiGpuFallback('cpu')` — the literal
  case, opt-in via `RADIA_TEST_OVERSIZED=1` (~12 GB host RAM, ~4 min).

Staged relaxation (Warning023) is *not* reachable from Python — `FldCmpMeth` is
only set by `FieldCompMethForSubdividedRecMag`, which has no binding.

**Known flake, unrelated:** `tests/test_sliver_tet_asm.py` crashes ~50% of runs
with heap corruption (`0xC0000374`) *after* all its checks have passed and
printed. Pre-existing — measured 5/10 on `fc4cdb6` and reproducible with no
interaction matrix at all, from `ObjPolyhdr`'s non-convex rejection path alone.

### Lesson worth not relearning
Phase 1's first attempt differed from the host in the last ~2 digits (~1e-15
relative) on the rotated case only. Cause: **nvcc contracts `x*y + z` into FMA**,
which rounds once where the host rounds twice. Fixed with `__dmul_rn` /
`__dadd_rn`, which are non-contractible and reproduce
`operator*(TMatrix3d,TMatrix3d)` (`gmvect.h:413`) exactly. The existing suites
all passed *with* the discrepancy present, because they compare GPU vs CPU at
~1e-11; only a bit-exact diff against the previous build could see it.

---

## 5. Measured facts worth not re-deriving

* **The matvec kernel is at roofline** — ~575 GB/s, ~95% of this card's VRAM
  bandwidth. No headroom. Block size is irrelevant (64..1024 all land within
  2%; an earlier "tpb=256 is 2.5x slower" reading was measurement noise). Do not
  spend time optimizing it.
* **Method 11 does 1.04 matrix passes per reported iteration** — the reported
  count *is* the pass count (`RADIA_RLX_DEBUG=1`).
* PCIe: 23.9 GB/s pinned H2D at x16. (It was found at x8 = 12.7 GB/s and fixed
  in BIOS — worth re-checking with `nvidia-smi --query-gpu=pcie.link.width.current`
  if streaming ever looks half-speed.)
* Streaming cost scales with the **overflow**, not model size:
  `0% resident -> 5.0x` in-core, `50% -> 2.2x`, `90% -> 1.3x`.
* Oversized end-to-end, nothing forced: 21952 elems (16.2 GiB) = 21.6 s total;
  32768 elems (36.0 GiB, 2.5x the card) = 66.6 s total.
* Galerkin is opt-in, default off, and contains **no general fixes** — it moves
  60 MeV `mean_f` the wrong way. Keep it off.

---

## 6. Build and test recipe

Fast dev build (sm_89 only; `pip install .` builds 8 architectures):

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CONDA_PREFIX=C:\Users\Daniel\anaconda3\envs\radiacuda2"
cmake -S . -B build_dev -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_ARCHITECTURES=89 ^
      -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%\Library" ^
      -DPython_EXECUTABLE="%CONDA_PREFIX%\python.exe"
cmake --build build_dev --parallel
```

`CMAKE_PREFIX_PATH` is required or `find_library(FFTW3F_LIB)` fails. Load the
result with `sys.path.insert(0, "<repo>/build_dev/cpp")`; the installed
`site-packages/radia.pyd` stays untouched.

Suites (`tests/`): `test_gpu_fallback.py`, `test_gpu_fallback_policy.py`,
`test_sliver_tet_asm.py`, `test_galerkin_asm.py`.
Run them from **PowerShell** — `test_sliver_tet_asm.py` reports exit 127 with no
output under Git Bash (shell artifact, not a failure).

Env hooks: `RADIA_RLX_DEBUG`, `RADGPU_FORCE_STREAM`,
`RADGPU_STREAM_RESIDENT_FRAC`, `RADIA_GPU_FLD_MAX_OBS_CHUNK`, `RADIA_GALERKIN*`.

---

## 7. Downstream (cyclotron_optimizer)

`simulation.gpu_fallback: cpu | break | gpu_streaming` is wired in and validated
at config load (commit `4388988` on `geometry-isochronism-rework`). Documented,
commented out, in all six ymls. `_apply_gpu_fallback` must run **before**
`RlxPre` — the policy is read at assembly and again at solve — and it is a
per-process global, so anything spawning subprocesses must set it inside them
(`scripts/perturb_study/ladder_radia.py` does; note `scripts/` is gitignored).
