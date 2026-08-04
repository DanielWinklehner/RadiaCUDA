# Assembly -> solve hand-off: state, measurements, and the remaining phase

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
* **Host RAM** now binds instead, because *two* full host copies of the matrix
  exist at once (`InteractMatrix` as `TMatrix3df**`, plus the assembly's flat
  buffer). Peak is therefore `2 * 36*N^2`. **This is what phase 3 fixes.**

Concretely: the 44k all-tet 60 MeV model has a 69.7 GB matrix and currently
needs **~139 GB** of host RAM against ~94 GB free. **It does not fit today.**
At one representation it needs 69.7 GB and fits comfortably.

---

## 2. What is already committed

| commit | what |
|---|---|
| `581a3ce` | streaming matvec: page-lock the matrix in place, resident row block |
| `2004b80` | `RADIA_RLX_DEBUG` matrix-pass counter |
| `85228e9` | opt-in Galerkin assembly (+ `AmOfWarnings` 15->16 fix) |
| `cecf9b2` | **phase 1** — assembly emits the solver's layout directly |
| `00e123d` | **phase 2** — device hand-off; cold solves ~8x faster |

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

## 3. Phase 3 — the remaining work

**Goal:** stop materializing `InteractMatrix` when the GPU assembled the matrix
and the GPU will solve, so peak host memory drops from `2*36N^2` to `36N^2`.

The saving comes from **not allocating**, not merely not filling.

### 3.1 Every consumer of `InteractMatrix`

| site | when it runs | note |
|---|---|---|
| `radrlmet.cpp` — 22 direct uses | CPU relaxation only | all reachable **only** via §3.2 |
| `radgpurlx.cpp:41` (flatten) | GPU solve, `skipMatrix == 0` only | after phase 2 this is now rare |
| `radgpurlx.cpp:218` (self blocks) | **EVERY GPU solve** | see §3.3 — the trap |
| `radintrc.h:516` `ShowInteractMatrix` | on request | via `radapl2.cpp:1193` |
| `radintrc.cpp:377-426` | allocation | see §3.4 |

### 3.2 The chokepoints (verified)

Every relaxation-method object in the codebase is constructed in exactly two
functions, both in `radapl2.cpp`:

* `radTApplication::MakeManualRelax` (`:1220`) — builds `radTSimpleRelaxation`
* `radTApplication::MakeAutoRelax`   (`:1289`) — builds `radTRelaxationMethNo_3/4/a5/8/...`

Verified by grepping for every `RelaxMethNo_*` / `radTSimpleRelaxation`
construction outside `radrlmet.cpp` — there are none elsewhere. So guarding
these two, plus `ShowInteractMatrix` and the flatten path, covers all 22
downstream uses **without touching Chubar's `radrlmet.cpp` at all**.

That makes it ~4 guard sites, not 22. Re-verify this grep before relying on it.

### 3.3 The trap: self-blocks run on every solve

`radGPU_PackInteractionData` reads `InteractMatrix[i][i]` for the
self-interaction diagonal (`radgpurlx.cpp:218`) **outside** the `skipMatrix`
guard, so it runs on every GPU solve. If that triggers materialization, phase 3
buys *nothing* — every solve would rebuild the whole host matrix.

Fix first: pull the N diagonal blocks from the device matrix (9N floats, O(N)).
This is a prerequisite, not an optional extra.

### 3.4 Deferred allocation

`InteractMatrix` is allocated in `AllocateMemory` (`radintrc.cpp:377-426`)
*before* it is known whether GPU assembly will succeed, and the CPU fallback
assembly writes straight into it. The allocation must move to after the GPU has
either taken the matrix or declined. This is the riskiest edit in phase 3 —
it changes interaction-object lifetime.

### 3.5 Suggested shape

```
radTInteraction::EnsureInteractMatrix()   // alloc if needed + fill from the
                                          // device cache (D2H, layout read as
                                          // in the current radGPU_UnpackMatrix)
```
called at the four chokepoints. Needs a getter from `radgpurlx.cu` exposing the
cached device pointer/stamp/dim.

---

## 4. Verification

**Non-negotiable:** bit-identity against the previous build. See
`fingerprint`-style harness (scratch; reproduce it):

* four cases — plain, **rotated**, mirrored, rotated+mirrored
* full precision (`%.17e`) B at several points, plus misfit and iteration count
* build reference, `git stash` the change, rebuild, diff

**The rotated case is the one that matters.** It is the only one with a
non-identity `MainTransPtrArray`, so it is the only one that exercises the row
transform at all. With a plain block that code path never runs.

**Gap to close in phase 3:** nothing currently exercises **GPU-assembled ->
CPU-solved**, which is exactly the path phase 3 can break (null `InteractMatrix`).
Reachable via `rad.UtiGpuFallback('cpu')` on an oversized model, or the
anisotropic-material (Warning022) / staged-relaxation (Warning023) fallbacks.
Write that test *before* the phase-3 edits.

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
