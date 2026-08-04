# tests/_galerkin_probe.py
#
# Helper for test_galerkin_asm.py and for the bit-exactness A/B against a
# reference build: solves a small mixed model and prints B at fixed points with
# full precision, so two builds (or two backends) can be compared byte for byte.
#
# Usage:  python tests/_galerkin_probe.py <case> <gpu|cpu>
#   case: recmag_lin | poly_lin | mixed_lin | mixed_sat | poly_sym
#
# Import radia from a specific build with, e.g.
#   PYTHONPATH=build_dev/cpp python tests/_galerkin_probe.py mixed_lin gpu

import sys

import numpy as np
import radia as rad

PTS = [[22.3, 8.1, 0.7], [50.2, 19.7, 3.9], [34.6, -13.8, 10.8],
       [12.9, 3.3, -6.1]]


def tet_ids(rng, n, lo, hi):
    faces = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]
    ids = []
    for _ in range(n):
        base = rng.uniform(lo, hi)
        verts = base + rng.uniform(2, 8, size=(4, 3))
        ids.append(rad.ObjPolyhdr(verts.tolist(), faces))
    return ids


def recmag_grid_ids(nx, ny, nz, pitch, dims, origin):
    ids = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ids.append(rad.ObjRecMag(
                    [origin[0] + i * pitch[0], origin[1] + j * pitch[1],
                     origin[2] + k * pitch[2]], list(dims)))
    return ids


def build_model(case):
    rad.UtiDelAll()
    rng = np.random.default_rng(7)
    kind, material = case.rsplit("_", 1)
    sym = kind.endswith("_sym") or case.endswith("_sym")
    kind = kind.replace("_sym", "")

    ids = []
    if kind in ("recmag", "mixed"):
        ids += recmag_grid_ids(3, 2, 2, [12.0, 11.0, 9.0], [10.0, 8.0, 7.0],
                               [30.0, 10.0, 8.0])
    if kind in ("poly", "mixed"):
        ids += tet_ids(rng, 12, [25, 5, 5], [70, 35, 30])

    iron = rad.ObjCnt(ids)
    if sym:
        rad.TrfZerPerp(iron, [0, 0, 0], [1, 0, 0])
        rad.TrfZerPara(iron, [0, 0, 0], [0, 0, 1])

    if material == "sat":
        mat = rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2])
    else:
        mat = rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8])
    rad.MatApl(iron, mat)

    coil = rad.ObjRaceTrk([0, 0, 25], [60, 80], [0, 0], 12, 8, 3.0, "man", "z")
    return rad.ObjCnt([iron, coil])


CASES = {
    "recmag_lin": ("recmag", False),
    "poly_lin": ("poly", False),
    "mixed_lin": ("mixed", False),
    "mixed_sat": ("mixed", False),
    "poly_sym_lin": ("poly", True),
    "mixed_sym_lin": ("mixed", True),
}


def solve(case, gpu_asm):
    model = build_model(case)
    im = rad.RlxPre(model, use_gpu=gpu_asm)
    backend = rad.UtiAsmLastBackend()
    rad.RlxAuto(im, 1e-7, 4000, 4)             # deterministic CPU relax
    b = np.array(rad.Fld(model, 'b', PTS, use_gpu=False), dtype=float)
    return backend, b


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "mixed_lin"
    gpu = (len(sys.argv) < 3) or (sys.argv[2] != "cpu")
    rad.FldLenRndSw('on')
    backend, b = solve(case, gpu)
    print(f"module {rad.__file__}")
    print(f"case {case} backend {backend}")
    for v in b.ravel():
        print(v.hex())


if __name__ == "__main__":
    main()
