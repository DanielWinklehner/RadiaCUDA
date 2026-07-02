"""fp32 GPU face-kernel test: precision='single' vs fp64 GPU vs CPU.

Builds a small magnetized tet model with the full 8-fold cyclotron symmetry
(TrfZerPerp x3 + TrfZerPara) plus a coil, then compares:
  - GPU fp64 vs CPU        -> must agree to ~1e-9 (regression guard)
  - GPU fp32 vs GPU fp64   -> visualization-grade, ~1e-4 relative

Run:  python tests/test_gpu_fld_fp32.py
"""

import numpy as np
import radia as rad


def build_model():
    rad.UtiDelAll()

    # A few tets in the fundamental wedge (0..45 deg), magnetized
    tets = [
        [[60, 5, 10], [90, 10, 10], [75, 30, 10], [70, 12, 40]],
        [[100, 20, 15], [130, 25, 15], [115, 45, 15], [110, 28, 50]],
        [[150, 10, 20], [180, 15, 20], [165, 40, 20], [160, 22, 55]],
        [[200, 30, 10], [230, 35, 10], [215, 60, 10], [210, 42, 45]],
    ]
    faces = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]
    ids = []
    for t in tets:
        oid = rad.ObjPolyhdr(t, faces)
        rad.ObjSetM(oid, [0.3, 0.1, 1.2])
        ids.append(oid)
    iron = rad.ObjCnt(ids)

    # full 8-fold cyclotron symmetry + midplane mirror
    rad.TrfZerPerp(iron, [0, 0, 0], [1, -1, 0])
    rad.TrfZerPerp(iron, [0, 0, 0], [1, 0, 0])
    rad.TrfZerPerp(iron, [0, 0, 0], [0, 1, 0])
    rad.TrfZerPara(iron, [0, 0, 0], [0, 0, 1])

    coil = rad.ObjRaceTrk([0, 0, 80], [250, 300], [0, 0], 40, 15, 2.5, "man", "z")

    return rad.ObjCnt([iron, coil])


def main():
    model = build_model()

    rng = np.random.default_rng(42)
    pts = np.column_stack([
        rng.uniform(-350, 350, 400),
        rng.uniform(-350, 350, 400),
        rng.uniform(-60, 60, 400),
    ]).tolist()

    b_cpu = np.array(rad.Fld(model, 'b', pts, use_gpu=False))
    print(f"CPU backend:      {rad.UtiFldLastBackend()}")

    b_gpu64 = np.array(rad.Fld(model, 'b', pts, use_gpu=True))
    print(f"GPU fp64 backend: {rad.UtiFldLastBackend()}")

    b_gpu32 = np.array(rad.Fld(model, 'b', pts, use_gpu=True, precision='single'))
    print(f"GPU fp32 backend: {rad.UtiFldLastBackend()}")

    scale = np.abs(b_cpu).max()
    err64 = np.abs(b_gpu64 - b_cpu).max() / scale
    err32 = np.abs(b_gpu32 - b_gpu64).max() / scale

    print(f"\n|B| scale: {scale:.4e} T")
    print(f"GPU fp64 vs CPU      max rel err: {err64:.3e}")
    print(f"GPU fp32 vs GPU fp64 max rel err: {err32:.3e}")

    assert err64 < 1e-9, f"fp64 GPU regression: {err64:.3e}"
    assert err32 < 5e-4, f"fp32 kernel out of tolerance: {err32:.3e}"

    # precision='double' must be identical to the default
    b_gpu64b = np.array(rad.Fld(model, 'b', pts, use_gpu=True, precision='double'))
    assert np.array_equal(b_gpu64, b_gpu64b), "precision='double' differs from default"

    print("\nALL FP32 KERNEL TESTS PASSED")


if __name__ == "__main__":
    main()
