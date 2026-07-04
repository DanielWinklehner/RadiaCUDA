"""Mixed-symmetry regression: one model containing elements with FOUR
symmetry planes, ONE symmetry plane, and NO symmetry must solve and
field-evaluate identically on the GPU and CPU paths.

Run:  python tests/test_mixed_symmetry.py
"""

import numpy as np
import radia as rad

FACES = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]


def _tet_cluster(rng, n, lo, hi):
    ids = []
    for _ in range(n):
        base = rng.uniform(lo, hi)
        verts = base + rng.uniform(2, 8, size=(4, 3))
        ids.append(rad.ObjPolyhdr(verts.tolist(), FACES))
    return rad.ObjCnt(ids)


def build_model():
    rad.UtiDelAll()
    rng = np.random.default_rng(3)

    # A: fundamental wedge, FULL 8-fold + z-mirror (4 planes -> 16 copies)
    grp_a = _tet_cluster(rng, 20, [40, 5, 8], [90, 30, 35])
    rad.TrfZerPerp(grp_a, [0, 0, 0], [1, -1, 0])
    rad.TrfZerPerp(grp_a, [0, 0, 0], [1, 0, 0])
    rad.TrfZerPerp(grp_a, [0, 0, 0], [0, 1, 0])
    rad.TrfZerPara(grp_a, [0, 0, 0], [0, 0, 1])

    # B: ONE mirror plane only (y = 0)
    grp_b = _tet_cluster(rng, 15, [-140, 10, -20], [-100, 40, 20])
    rad.TrfZerPerp(grp_b, [0, 0, 0], [0, 1, 0])

    # C: NO symmetry
    grp_c = _tet_cluster(rng, 10, [100, 100, -30], [140, 140, 10])

    mat = rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2])
    for g in (grp_a, grp_b, grp_c):
        rad.MatApl(g, mat)

    coil = rad.ObjRaceTrk([0, 0, 40], [160, 180], [0, 0], 20, 12, 2.0, "man", "z")
    return rad.ObjCnt([grp_a, grp_b, grp_c, coil])


def solve_and_eval(model, *, gpu):
    im = rad.RlxPre(model, use_gpu=gpu)
    res = rad.RlxAuto(im, 1e-6, 5000, 9 if gpu else 4, 'ZeroM->True')
    rng = np.random.default_rng(9)
    pts = np.column_stack([rng.uniform(-160, 160, 300),
                           rng.uniform(-160, 160, 300),
                           rng.uniform(-50, 50, 300)]).tolist()
    b = np.array(rad.Fld(model, 'b', pts, use_gpu=gpu))
    return res[0], b


def main():
    model = build_model()
    misfit_cpu, b_cpu = solve_and_eval(model, gpu=False)
    misfit_gpu, b_gpu = solve_and_eval(model, gpu=True)

    scale = np.abs(b_cpu).max()
    rel = np.abs(b_gpu - b_cpu).max() / scale
    print(f"CPU solve misfit: {misfit_cpu:.3e}   GPU solve misfit: {misfit_gpu:.3e}")
    print(f"|B| scale: {scale:.4e} T   GPU vs CPU max rel diff: {rel:.3e}")
    assert rel < 1e-6, f"mixed-symmetry GPU/CPU divergence: {rel:.3e}"
    print("MIXED-SYMMETRY GPU/CPU TEST PASSED")


if __name__ == "__main__":
    main()
