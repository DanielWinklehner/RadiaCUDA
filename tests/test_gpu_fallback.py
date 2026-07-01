# tests/test_gpu_fallback.py
#
# Validates issue #3: the GPU field path must fall back to the CPU path (and
# produce correct results) when a model contains element types the GPU cannot
# evaluate, while still running on the GPU for supported geometry.
#
# Element handling on the GPU field path:
#   - extruded polygons (radTExtrPolygon, via ObjThckPgn)  -> decomposed to a polyhedron
#     and run on the GPU (issue #2); subdivided objects are containers of polyhedra and
#     already run on the GPU.
#   - current sources (ObjRecCur / arcs / racetracks):
#       * directly placed             -> kept on GPU path, current field added on CPU
#       * under a symmetry/transform  -> CPU fallback (coil helper is not symmetry-aware)
#
# Dependency-free assertion script (numpy + radia only): plain asserts via a
# small checker, non-zero exit on failure, so it works in CI without pytest.
#
# Backend introspection: rad.UtiFldLastBackend() reports 'gpu' / 'cpu' / 'none'
# for the most recent Fld B-field call. This lets the positive-control cases
# confirm the GPU actually ran (numeric GPU==CPU agreement alone cannot, since a
# silent CPU fallback would also match).
#
# Run:  python tests/test_gpu_fallback.py

import sys
import numpy as np
import radia as rad
# rad.UtiMPI('on')

# Disable Radia's built-in length "reproducibility rounding" — a ~1e-9 rand()-based
# perturbation of length values (ON by default; see radTConvergRepair / DoublePlus)
# that nudges distances off on-edge singularities but makes extruded-polygon / arc
# field values non-reproducible from call to call. Turning it off makes the CPU field
# deterministic, so the fallback (CPU-vs-CPU) comparisons below are bit-exact and the
# test is stable. (RecMag and polyhedra are unaffected either way.)
rad.FldLenRndSw('off')

# Tolerance for the GPU-vs-CPU comparison. With length randomization disabled above,
# the fallback cases (both sides run the same CPU code) are effectively bit-exact; the
# tolerance mainly covers divergence between the two independent double-precision
# kernels (GPU vs CPU) for the supported-geometry positive controls.
TOL = dict(rtol=1e-5, atol=1e-8)

# Floor below which the reference (CPU) field is considered trivially zero. Guards
# against a both-sides-all-zero regression (empty model / everything dropped),
# which a bare np.allclose would otherwise pass for the wrong reason.
FIELD_FLOOR = 1e-7

# Observation points kept off all coordinate planes AND off integer / half-integer
# values: with length randomization disabled, a point lying exactly on a symmetry
# plane OR on an axis-aligned object face plane hits an unregularized singularity
# (NaN). Non-integer coordinates avoid the block faces used by the test models.
OBS = [
    [7.3, 3.1, 9.7],
    [11.2, -5.4, 4.3],
    [4.6, 8.2, 13.1],
    [-9.1, 6.3, -4.4],
    [6.4, -7.2, 16.5],
]

_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""),
          flush=True)
    if not ok:
        _failures.append(name)


def fld(obj, use_gpu):
    return np.array(rad.Fld(obj, 'b', OBS, use_gpu=use_gpu), dtype=float).reshape(-1, 3)


def backend():
    return rad.UtiFldLastBackend()


def maxabs(a):
    return float(np.max(np.abs(a))) if a.size else 0.0


def maxdiff(a, b):
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# --- Detect whether this build actually has a working GPU path ---------------
# Probe with a plain axis-aligned ObjRecMag: a non-current RecMag with identity
# rotation, so it avoids both unsupported types (#2/#5) and the RecMag mag-frame
# bug (#1). If the GPU path is active it reports backend 'gpu'.
rad.UtiDelAll()
_probe = rad.ObjRecMag([0, 0, 0], [2, 2, 2], [0, 0, 1])
_ = fld(_probe, True)
GPU_AVAILABLE = (backend() == 'gpu')
print(f"\nGPU backend available: {GPU_AVAILABLE} (probe backend={backend()!r})\n", flush=True)


def run_case(name, build, expect_backend):
    """build() returns the object index. expect_backend is 'gpu' (supported,
    stays on GPU) or 'cpu' (unsupported, must fall back).

    Call order matters: CPU eval first, GPU eval last, backend() read immediately
    after the GPU eval — because g_LastFldBackend is last-write-wins and must
    reflect the use_gpu=True call under test."""
    rad.UtiDelAll()
    obj = build()
    cpu = fld(obj, False)
    gpu = fld(obj, True)
    used = backend()

    # Non-triviality: the reference field must be clearly non-zero, else a
    # both-sides-zero regression would make the equality check pass spuriously.
    # Unconditional so even a CPU-only build still validates the field is real.
    check(f"{name}: field non-trivial", maxabs(cpu) > FIELD_FLOOR,
          f"max|CPU|={maxabs(cpu):.3e}")

    check(f"{name}: GPU == CPU", np.allclose(gpu, cpu, **TOL),
          f"max|d|={maxdiff(gpu, cpu):.2e}")

    if GPU_AVAILABLE:
        label = "ran on GPU" if expect_backend == 'gpu' else "fell back to CPU"
        check(f"{name}: {label}", used == expect_backend, f"backend={used!r}")


def build_tet(mag=(0, 0, 1)):
    verts = [[10, 0, 10], [20, 0, 10], [10, 10, 10], [10, 0, 20]]
    faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    return rad.ObjPolyhdr(verts, faces, list(mag))


def build_rotated_recmag():
    # Rotated RecMag: exercises the local-vs-lab magnetization fix (#1). The GPU
    # RecMag kernel evaluates the box field locally and rotates it to lab, so it must
    # be fed the LOCAL magnetization; a non-trivial rotation with a tilted M makes the
    # old lab-frame double-rotation bug visible (GPU != CPU before the fix).
    m = rad.ObjRecMag([0, 0, 0], [2, 3, 4], [0.5, 0.7, 1.0])
    rot = rad.TrfRot([0, 0, 0], [1, 1, 0], 0.6)   # rotation transform (about a tilted axis)
    rad.TrfOrnt(m, rot)                            # apply it to the block once
    return m


def build_mirror_recmag():
    # Mirror-symmetric RecMag: exercises the mirror-parity half of the #1 fix
    # (rot = -Mtx, so rot*Magn must give the reflected axial magnetization).
    m = rad.ObjRecMag([0, 0, 5], [2, 3, 4], [0.5, 0.7, 1.0])
    rad.TrfZerPara(m, [0, 0, 0], [0, 0, 1])
    return m


def build_extruded():
    # Extruded polygon (radTExtrPolygon), now decomposed to a polyhedron and run on
    # the GPU (#2). The cross-section MUST be non-rectangular: Radia collapses a
    # rectangular extruded polygon into a radTRecMag, so a square would exercise the
    # RecMag path instead. A triangle yields a real radTExtrPolygon (UtiDmp reports
    # "Relaxable: ThckPgn").
    return rad.ObjThckPgn(0.0, 2.0, [[-1, -1], [1, -1], [0, 1]], 'x', [0, 0, 1])


def build_mixed():
    # Tet + extruded polygon in one container: both are GPU-supported now (the
    # extruded polygon is decomposed to a polyhedron), so the whole eval runs on GPU.
    return rad.ObjCnt([build_tet(), build_extruded()])


def build_subdiv_recmag():
    # Subdividing a RecMag yields a container of polyhedra ("Subdivided Polyhedron"),
    # already handled by group recursion + the polyhedron kernel -> runs on GPU.
    m = rad.ObjRecMag([0, 0, 0], [4, 4, 4], [0, 0, 1])
    rad.ObjDivMag(m, [2, 2, 2])
    return m


def build_subdiv_extruded():
    # Subdividing an extruded polygon likewise yields a container of polyhedra.
    e = rad.ObjThckPgn(0.0, 2.0, [[-1, -1], [1, -1], [0, 1]], 'x', [0, 0, 1])
    rad.ObjDivMag(e, [2, 2, 2])
    return e


def build_current_direct():
    # Directly-placed current RecMag (ObjRecCur): no symmetry/transform, so the GPU
    # path keeps it and adds its current field on the CPU (ComputeCoilFieldCPU). No
    # magnetics here, so only the coil-add runs, but radGPU_ComputeField still
    # succeeds -> backend reported as 'gpu' (i.e. no full RadFld fallback).
    return rad.ObjRecCur([0, 0, 0], [2, 3, 4], [0.0, 0.0, 1.0e6])


def build_tet_plus_current():
    # Hybrid, no symmetry: magnetized tet (GPU kernel) + directly-placed current
    # block (CPU coil-add). The whole eval should stay on the GPU path and match CPU.
    tet = build_tet()
    coil = rad.ObjRecCur([8, 0, 0], [2, 3, 4], [0.0, 0.0, 1.0e6])
    return rad.ObjCnt([tet, coil])


def build_current_sym():
    # Current RecMag UNDER a plane symmetry: the CPU coil helper is not symmetry-
    # aware, so this must fall back to the full CPU field path (not silently drop the
    # symmetry copy).
    c = rad.ObjRecCur([0, 0, 5], [2, 3, 4], [0.0, 0.0, 1.0e6])
    rad.TrfZerPara(c, [0, 0, 0], [0, 0, 1])   # mirror across z=0
    return c


def build_arc_sym():
    # Non-relaxable current source (arc) UNDER symmetry: same requirement -> fallback.
    a = rad.ObjArcCur([0, 0, 5], [10, 15], [0.0, 2.5], 8, 10, 5.0, 'man', 'z')
    rad.TrfZerPara(a, [0, 0, 0], [0, 0, 1])
    return a


def main():
    print("=== GPU fallback / dispatch tests (issue #3) ===", flush=True)
    run_case("supported RecMag", lambda: rad.ObjRecMag([0, 0, 0], [2, 3, 4], [0.3, 0.0, 1.0]), 'gpu')
    run_case("supported tet", build_tet, 'gpu')
    run_case("rotated RecMag", build_rotated_recmag, 'gpu')     # issue #1
    run_case("mirror RecMag", build_mirror_recmag, 'gpu')       # issue #1
    run_case("extruded polygon", build_extruded, 'gpu')
    run_case("subdivided RecMag", build_subdiv_recmag, 'gpu')
    run_case("subdivided extruded", build_subdiv_extruded, 'gpu')
    # current sources: directly placed -> kept on GPU path via CPU coil-add;
    # under a symmetry/transform -> full CPU fallback (issue #5 + coil-symmetry).
    run_case("current RecMag (direct)", build_current_direct, 'gpu')
    run_case("tet + current (hybrid)", build_tet_plus_current, 'gpu')
    run_case("current RecMag under symmetry", build_current_sym, 'cpu')
    run_case("arc under symmetry", build_arc_sym, 'cpu')
    run_case("mixed tet+extruded", build_mixed, 'gpu')
    print(flush=True)

    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures), flush=True)
        return 1
    if GPU_AVAILABLE:
        print("ALL PASSED", flush=True)
    else:
        # Honest status: with no GPU path, the fallback cases reduce to CPU==CPU
        # and the 'ran on GPU' / 'fell back' assertions were skipped. The numeric
        # and non-triviality checks still ran, but the GPU dispatch was NOT exercised.
        print("PASSED (numeric + non-triviality only) — GPU path NOT exercised "
              "(CPU-only build/run); backend assertions skipped.", flush=True)

    rad.UtiMPI('off')
    return 0


if __name__ == "__main__":
    sys.exit(main())
