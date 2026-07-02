# tests/test_gpu_fld_chunking.py
#
# Validates issue #12: the GPU field-eval path chunks observation points to bound
# the O(n_obs * n_faces) partial buffer within VRAM, and chunking is a *pure
# partition* -- it must not change results.
#
# How it's made decisive without a huge grid:
#   The sizer radGPU_FldMaxObsChunk honors an env var RADIA_GPU_FLD_MAX_OBS_CHUNK
#   that clamps the chunk to a small value (a test hook, radgpu_fld.cu). Forcing a
#   tiny chunk on a modest grid drives MANY chunk iterations, exercising the loop,
#   the per-chunk offset arithmetic (arCoord/arB + off*3), and per-chunk alloc/free.
#
#   Each observation point's field is independent of which chunk it lands in (one
#   thread per obs point; the reduction sums the same source blocks regardless), so
#   a chunked GPU result must be BIT-IDENTICAL to the unchunked GPU result. Any
#   chunking bug (wrong offset, dropped/duplicated last chunk, stale buffer) breaks
#   that equality. We also check GPU == CPU within tolerance, and that the run stays
#   on the GPU (backend 'gpu', i.e. it chunked -- it did not silently fall back).
#
# MPI-safe (see tests/_mpi.py): all rad.Fld calls run on every rank (so Radia's
# internal MPI collectives stay balanced); only rank 0 asserts and decides pass/fail.
# Runs single-process (`python tests/test_gpu_fld_chunking.py`) or under
# `mpiexec -n N python tests/test_gpu_fld_chunking.py`.
#
# Note: the hook relies on the C runtime's getenv() observing os.environ changes.
# On Windows this holds because Python and radia.pyd share the UCRT.
#
# Dependency-free (numpy + radia only); non-zero exit on failure, no pytest needed.

import os
import sys
import numpy as np
import radia as rad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `_mpi`
from _mpi import MpiTest

mpi = MpiTest()                 # init MPI iff under mpiexec; must precede any rad.Fld
rad.FldLenRndSw('off')          # deterministic CPU field (see test_gpu_fallback.py)

TOL = dict(rtol=1e-5, atol=1e-8)   # GPU (fp64 kernel) vs CPU divergence
FIELD_FLOOR = 1e-7                 # guard against a both-sides-zero false pass
CHUNKS = (1, 7, 13)               # forced chunk sizes; none divides len(OBS) evenly

ENV = "RADIA_GPU_FLD_MAX_OBS_CHUNK"

# Observation grid: 8^3 = 512 points in a shell |coord| >= 7.1 around compact
# sources near the origin (all sources within radius ~2.6). Non-integer / non-half-
# integer coordinates keep points off object face planes (singularities). 512 is not
# divisible by 7 or 13, so those chunk sizes exercise a short final chunk.
_C = [-16.7, -13.3, -10.1, -7.3, 7.1, 10.9, 13.7, 16.3]
OBS = [[x, y, z] for x in _C for y in _C for z in _C]


def fld(obj, use_gpu):
    # rad.Fld is issued on every rank so Radia's internal MPI collectives stay balanced.
    # Under MPI it returns Python None on non-root ranks (consistently for the CPU and GPU
    # paths, after the RadFld/binding fix); only rank 0 gets the field. Consume on rank 0.
    raw = rad.Fld(obj, 'b', OBS, use_gpu=use_gpu)
    if raw is None or not mpi.is_root:
        return None
    return np.array(raw, dtype=float).reshape(-1, 3)


def fld_chunked(obj, chunk):
    """Evaluate on the GPU with the chunk sizer clamped to `chunk` (None = unclamped)."""
    if chunk is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = str(chunk)
    try:
        return fld(obj, True)
    finally:
        os.environ.pop(ENV, None)


def backend():
    return rad.UtiFldLastBackend()


def maxabs(a):
    return float(np.max(np.abs(a))) if a.size else 0.0


def maxdiff(a, b):
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# --- GPU availability probe (runs on every rank; balanced) -------------------
rad.UtiDelAll()
_probe = rad.ObjRecMag([0, 0, 0], [2, 2, 2], [0, 0, 1])
_ = fld(_probe, True)
GPU_AVAILABLE = (backend() == 'gpu')
mpi.say(f"\nGPU backend available: {GPU_AVAILABLE} (probe backend={backend()!r}, rank={mpi.rank})")
mpi.say(f"Observation points: {len(OBS)}; forced chunk sizes: {CHUNKS}\n")


def build_recmag():
    # RecMag source -> exercises the RecMag chunk loop.
    return rad.ObjRecMag([0, 0, 0], [3, 3, 3], [0.3, 0.0, 1.0])


def build_poly():
    # Tetrahedron (polyhedron) -> exercises the polygon-face chunk loop.
    verts = [[-2, -2, -2], [2, -2, -2], [-2, 2, -2], [-2, -2, 2]]
    faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    return rad.ObjPolyhdr(verts, faces, [0.0, 0.5, 1.0])


def build_mixed():
    # Both source kinds in one eval -> both chunk loops run in a single Fld call.
    return rad.ObjCnt([build_recmag(), build_poly()])


def run_case(name, build):
    rad.UtiDelAll()
    obj = build()

    # --- field computations: run on EVERY rank to keep MPI collectives balanced ---
    cpu = fld(obj, False)
    gpu_ref = fld_chunked(obj, None)          # unchunked GPU baseline
    ref_backend = backend()
    chunk_results = []
    for chunk in CHUNKS:
        g = fld_chunked(obj, chunk)
        chunk_results.append((chunk, g, backend()))

    # --- assertions: rank 0 only (non-root holds no valid field data) ---
    if not mpi.is_root:
        return
    mpi.check(f"{name}: field non-trivial", maxabs(cpu) > FIELD_FLOOR, f"max|CPU|={maxabs(cpu):.3e}")
    if not GPU_AVAILABLE:
        return
    mpi.check(f"{name}: baseline ran on GPU", ref_backend == 'gpu', f"backend={ref_backend!r}")
    mpi.check(f"{name}: baseline GPU==CPU", np.allclose(gpu_ref, cpu, **TOL),
              f"max|d|={maxdiff(gpu_ref, cpu):.2e}")
    for chunk, g, used in chunk_results:
        # Must have stayed on the GPU (chunked, not silently fallen back to CPU).
        mpi.check(f"{name}: chunk={chunk} stayed on GPU", used == 'gpu', f"backend={used!r}")
        # Decisive: chunking is a pure partition -> bit-identical to unchunked GPU.
        mpi.check(f"{name}: chunk={chunk} bit-identical to baseline",
                  np.array_equal(g, gpu_ref), f"max|d|={maxdiff(g, gpu_ref):.2e}")
        # And still matches CPU within tolerance.
        mpi.check(f"{name}: chunk={chunk} == CPU", np.allclose(g, cpu, **TOL),
                  f"max|d|={maxdiff(g, cpu):.2e}")


def main():
    mpi.say("=== GPU field-eval observation-point chunking tests (issue #12) ===")
    run_case("RecMag", build_recmag)
    run_case("polyhedron", build_poly)
    run_case("mixed RecMag+polyhedron", build_mixed)
    mpi.say("")

    if mpi.failures:
        mpi.say(f"FAILED ({len(mpi.failures)}): " + ", ".join(mpi.failures))
    elif GPU_AVAILABLE:
        mpi.say("ALL PASSED")
    else:
        mpi.say("PASSED (CPU non-triviality only) — GPU path NOT exercised "
                "(CPU-only build/run); chunking assertions skipped.")
    return mpi.finish()


if __name__ == "__main__":
    sys.exit(main())
