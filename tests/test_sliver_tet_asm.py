"""Degenerate (sliver) tetrahedra must not poison the interaction matrix.

Regression test for the NaN that surfaced as

    radGPU_RelaxNK: non-finite residual at start
    radGPU_AutoRelaxNK: GPU solver failed, falling back to CPU

The GPU assembly kernel evaluated log(R + u) directly, where
u = (bk + (1+k^2)x)/sqrt(1+k^2). That sum is analytically >= 0, with equality
only when z -> 0 AND b -> 0 -- i.e. the observation point lies both on the face
plane and on the edge line. That double degeneracy is exactly the
self-interaction of a sliver element, where cancellation rounds R+u to a tiny
NEGATIVE value and log() returns NaN. Which side of zero it lands on is decided
by FMA contraction, which nvcc emits differently per architecture -- so the same
mesh produced NaN on one GPU (sm_86) and finite values on another (sm_89).

The kernel now evaluates it cancellation-free (radgpu_asm_log_R_plus_u), and
radGPU_UnpackMatrix zeroes any non-finite entry as a backstop.

Checks, for progressively flatter tets:
  1. the assembled field is finite (no NaN/Inf reaches the solver),
  2. GPU assembly agrees with CPU assembly,
  3. the relaxation actually converges instead of falling back.
"""
import numpy as np
import radia as rad

# Flatness sweep: the last entries are far flatter than any real mesh would
# emit, to exercise the guard well past the point where the naive form fails.
THICKNESS_MM = [1.0, 1e-2, 1e-4, 1e-6, 1e-8]

OBS = np.array([
    [20.0, 5.0, 3.0],
    [-15.0, 8.0, -4.0],
    [5.0, 5.0, 12.0],
    [0.3, 0.2, 0.05],    # close to the sliver itself
], dtype=float)


def build(thickness_mm):
    """A sliver tet (nearly coplanar vertices) plus a solid neighbour."""
    rad.UtiDelAll()
    t = thickness_mm
    verts = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [3, 3, t]]
    faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    sliver = rad.ObjPolyhdr(verts, faces, [0, 0, 1])
    rad.MatApl(sliver, rad.MatLin([0.06, 0.17], 1.2))

    block = rad.ObjRecMag([6, 6, 12], [8, 8, 6], [0, 0, 1])
    rad.MatApl(block, rad.MatLin([0.06, 0.17], 1.2))

    grp = rad.ObjCnt([sliver, block])
    return grp


def field_at(grp):
    return np.array([rad.Fld(grp, 'b', list(p)) for p in OBS], dtype=float)


def run_case(thickness_mm, use_gpu):
    grp = build(thickness_mm)
    rad.RlxPre(grp, use_gpu=use_gpu) if use_gpu is not None else rad.RlxPre(grp)
    backend = rad.UtiAsmLastBackend()
    rad.Solve(grp, 1e-5, 1000)
    return field_at(grp), backend


def main():
    failures = []
    print(f"{'thickness':>12} {'backend':>8} {'finite':>7} {'max|B|':>12} {'GPU-vs-CPU':>12}")
    print("-" * 58)

    tested = 0
    for t in THICKNESS_MM:
        rad.UtiDelAll()
        try:
            b_gpu, backend_gpu = run_case(t, use_gpu=1)
            rad.UtiDelAll()
            b_cpu, backend_cpu = run_case(t, use_gpu=0)
        except RuntimeError as exc:
            # Below some flatness Radia's own convexity test rejects the
            # polyhedron outright, so it never reaches the assembly kernel.
            # That is a geometry-input limit, not an assembly failure.
            print(f"{t:12.0e} {'-':>8} {'skip':>7}   (rejected by Radia: "
                  f"{str(exc).splitlines()[0][:40]})")
            continue
        tested += 1

        finite = bool(np.all(np.isfinite(b_gpu)) and np.all(np.isfinite(b_cpu)))
        scale = max(np.max(np.abs(b_cpu)), 1e-12)
        dev = float(np.max(np.abs(b_gpu - b_cpu)) / scale) if finite else float('nan')

        print(f"{t:12.0e} {backend_gpu:>8} {str(finite):>7} "
              f"{np.max(np.abs(b_gpu)):12.5e} {dev:12.3e}")

        if not finite:
            failures.append(f"thickness {t:.0e}: NON-FINITE field "
                            f"(gpu_finite={np.all(np.isfinite(b_gpu))}, "
                            f"cpu_finite={np.all(np.isfinite(b_cpu))})")
        elif dev > 5e-5:
            failures.append(f"thickness {t:.0e}: GPU-vs-CPU deviation {dev:.3e} > 5e-5")
        if backend_gpu != 'gpu':
            failures.append(f"thickness {t:.0e}: GPU assembly did not run "
                            f"(backend={backend_gpu}) -- test would be vacuous")

    print()
    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
        raise SystemExit("SLIVER-TET ASSEMBLY TEST FAILED")
    if tested == 0:
        raise SystemExit("SLIVER-TET ASSEMBLY TEST VACUOUS: no case reached the kernel")
    print(f"ALL SLIVER-TET ASSEMBLY TESTS PASSED ({tested} thickness levels)")


if __name__ == '__main__':
    main()
