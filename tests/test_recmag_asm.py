# tests/test_recmag_asm.py
#
# GPU interaction-matrix assembly parity for RecMag and MIXED
# (RecMag + polyhedron) models — the new assemble_mixed_kernel path
# (issue #8: the RecMag kernel used to be an empty stub, and mixed models
# were rejected by the packing).
#
# Method: build the same model twice, RlxPre with use_gpu=True (GPU
# assembly) and use_gpu=False (CPU assembly), relax both with the SAME
# deterministic CPU method (4 = Gauss-Seidel), evaluate B on the CPU
# (use_gpu=False) — so the ONLY difference between the two solves is which
# backend assembled the interaction matrix.  rad.UtiAsmLastBackend()
# confirms the intended backend actually ran (a silent CPU fallback would
# otherwise make the comparison trivially pass CPU-vs-CPU).
#
# Tolerances: the GPU stores the IM in float32 (vs CPU double), so relative
# deviations of order 1e-7..1e-6 in B are expected; a wrong Q-tensor shows
# up at 1e-3..1e0.  Linear-material cases use a tight tolerance (linear
# solve, no nonlinear amplification); the saturating-material case is
# slightly looser.
#
# Run:  python tests/test_recmag_asm.py

import sys
import numpy as np
import radia as rad

FAILED = []


def check(name, cond, msg=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {msg}", flush=True)
    if not cond:
        FAILED.append(name)


def tet_ids(rng, n, lo, hi):
    """n small random tetrahedra (polyhedron elements) in box [lo, hi]."""
    faces = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]
    ids = []
    for _ in range(n):
        base = rng.uniform(lo, hi)
        verts = base + rng.uniform(2, 8, size=(4, 3))
        ids.append(rad.ObjPolyhdr(verts.tolist(), faces))
    return ids


def recmag_grid_ids(nx, ny, nz, pitch, dims, origin):
    """Regular grid of RecMag cuboids."""
    ids = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [origin[0] + i * pitch[0],
                     origin[1] + j * pitch[1],
                     origin[2] + k * pitch[2]]
                ids.append(rad.ObjRecMag(c, list(dims)))
    return ids


def build_model(kind, material, symmetry):
    rad.UtiDelAll()
    rng = np.random.default_rng(7)

    ids = []
    if kind in ("recmag", "mixed"):
        ids += recmag_grid_ids(3, 2, 2, [12.0, 11.0, 9.0], [10.0, 8.0, 7.0],
                               [30.0, 10.0, 8.0])
    if kind in ("poly", "mixed"):
        ids += tet_ids(rng, 12, [25, 5, 5], [70, 35, 30])
    if kind == "touching":
        # Deliberate on-face-plane geometry: the second block's center lies
        # exactly in the plane of the first block's +x face (x = 25), far
        # away in y — exercises the AbsRandMagnitude jitter guards in the
        # cuboid closed form on both backends identically.
        ids.append(rad.ObjRecMag([20.0, 10.0, 8.0], [10.0, 8.0, 7.0]))
        ids.append(rad.ObjRecMag([25.0, 40.0, 8.0], [10.0, 8.0, 7.0]))
        ids.append(rad.ObjRecMag([45.0, 10.0, 8.0], [10.0, 8.0, 7.0]))

    iron = rad.ObjCnt(ids)
    if symmetry:
        rad.TrfZerPerp(iron, [0, 0, 0], [1, 0, 0])
        rad.TrfZerPara(iron, [0, 0, 0], [0, 0, 1])

    if material == "lin":
        # isotropic linear susceptibility + remanence (as in
        # test_gpu_asm_rowtransform.py): linear solve, deterministic
        mat = rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8])
    else:
        mat = rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2])
    rad.MatApl(iron, mat)

    coil = rad.ObjRaceTrk([0, 0, 25], [60, 80], [0, 0], 12, 8, 3.0, "man", "z")
    return rad.ObjCnt([iron, coil])


# Observation points chosen OFF every cuboid face-extension plane of the
# grids below (x !in {25,35,37,47,49,59}, y !in {6,14,17,25}, z !in
# {4.5,11.5,13.5,20.5} and their symmetry mirrors): with FldLenRndSw('off')
# the CPU RecMag field eval has no on-plane jitter repair and returns NaN
# for points exactly on such a plane (0/0) — that is expected Radia
# behavior, not an assembly property.
PTS = [[22.3, 8.1, 0.7], [50.2, 19.7, 3.9], [34.6, -13.8, 10.8]]


def solve_b(kind, material, symmetry, gpu_asm):
    model = build_model(kind, material, symmetry)
    im = rad.RlxPre(model, use_gpu=gpu_asm)
    backend = rad.UtiAsmLastBackend()
    rad.RlxAuto(im, 1e-7, 4000, 4)           # deterministic CPU relax
    b = np.array(rad.Fld(model, 'b', PTS, use_gpu=False))
    return backend, b


def run_case(name, kind, material, symmetry, rtol):
    print(f"case: {name}", flush=True)
    be_gpu, b_gpu = solve_b(kind, material, symmetry, gpu_asm=True)
    be_cpu, b_cpu = solve_b(kind, material, symmetry, gpu_asm=False)
    check(f"{name}: GPU assembly ran", be_gpu == "gpu", f"(backend={be_gpu})")
    check(f"{name}: CPU assembly ran", be_cpu == "cpu", f"(backend={be_cpu})")
    scale = np.abs(b_cpu).max()
    check(f"{name}: field is nontrivial", scale > 1e-7, f"(|B|max={scale:.2e})")
    dev = np.abs(b_gpu - b_cpu).max() / scale
    check(f"{name}: GPU-vs-CPU asm parity", dev < rtol,
          f"(rel dev {dev:.3e} < {rtol:.0e})")


def main():
    rad.FldLenRndSw('off')   # determinism (see test_gpu_fallback.py)

    # Pure RecMag, linear material, no symmetry: sharpest probe of the
    # cuboid Q-tensor (incl. self-blocks) with no nonlinear amplification.
    run_case("recmag_lin", "recmag", "lin", symmetry=False, rtol=5e-6)

    # Pure RecMag with TrfZerPerp+Para: exercises the per-copy point/field
    # transforms around the cuboid closed form.
    run_case("recmag_lin_sym", "recmag", "lin", symmetry=True, rtol=5e-6)

    # Mixed RecMag + tets with symmetry: cross blocks in both directions
    # (poly field at RecMag centers and vice versa).
    run_case("mixed_lin_sym", "mixed", "lin", symmetry=True, rtol=5e-6)

    # Pure polyhedron control: the refactor must not disturb the existing
    # poly-only path (same kernel, is_rec all zero).
    run_case("poly_lin_sym", "poly", "lin", symmetry=True, rtol=5e-6)

    # Saturating material (production-like): nonlinear solve amplifies IM
    # noise somewhat -> looser tolerance.
    run_case("mixed_sat_sym", "mixed", "sat", symmetry=True, rtol=5e-5)

    # On-face-plane centers: jitter-guard (AbsRandMagnitude) parity. Needs
    # randomization ON — with it off the guard returns 0 and BOTH backends
    # produce the same 0/0 garbage (AbsRandMagnitude itself is deterministic,
    # so 'on' does not hurt reproducibility of the RecMag closed form).
    rad.FldLenRndSw('on')
    run_case("touching_lin", "touching", "lin", symmetry=False, rtol=5e-6)
    rad.FldLenRndSw('off')

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURE(S): {FAILED}")
        return 1
    print("ALL RECMAG/MIXED GPU-ASSEMBLY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
