"""MPI solve regression: RlxPre/RlxAuto/Fld must give identical results under
mpiexec -n N as in a serial run, for all four assembly/relaxation combinations.

Exercises the RadiaCUDA MPI fixes:
  - GPU IM assembly under MPI (rank 0 assembles, workers wait via Bcast
    agreement) instead of silently switching to distributed-CPU assembly.
  - The 2-rank whole-matrix MPI_Send int-overflow fix (packeted sends for
    large models).

Usage:
  python tests/test_mpi_solve.py --reference          # writes reference values
  mpiexec -n 2 python tests/test_mpi_solve.py         # compares against them
  mpiexec -n 2 python tests/test_mpi_solve.py --large # >=1000-element model
                                                      # (packeted-branch check)

Import order matters: radia BEFORE mpi4py, then rad.UtiMPI('in').
"""

import json
import os
import sys

import radia as rad          # noqa: E402  must precede mpi4py
from mpi4py import MPI       # noqa: E402

import numpy as np

REF_FILE = os.path.join(os.path.dirname(__file__), "_mpi_solve_reference.json")


def build_model(large=False):
    rad.UtiDelAll()
    rng = np.random.default_rng(11)

    if large:
        # >= 1000 relaxable elements (RecMags: cheap analytic B_comp) to hit
        # the packeted MPI-assembly branch at nProc=2.
        n = 1100
        ids = []
        for _ in range(n):
            center = rng.uniform([-80, -80, -40], [80, 80, 40]).tolist()
            dims = rng.uniform(2.0, 4.0, 3).tolist()
            ids.append(rad.ObjRecMag(center, dims))
        iron = rad.ObjCnt(ids)
    else:
        faces = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]
        ids = []
        for _ in range(60):
            base = rng.uniform([20, 5, 5], [90, 40, 40])
            verts = base + rng.uniform(2, 10, size=(4, 3))
            ids.append(rad.ObjPolyhdr(verts.tolist(), faces))
        iron = rad.ObjCnt(ids)
        rad.TrfZerPerp(iron, [0, 0, 0], [1, 0, 0])
        rad.TrfZerPara(iron, [0, 0, 0], [0, 0, 1])

    mat = rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2])
    rad.MatApl(iron, mat)
    coil = rad.ObjRaceTrk([0, 0, 30], [70, 90], [0, 0], 15, 10, 3.0, "man", "z")
    return rad.ObjCnt([iron, coil])


def solve(model, *, gpu_asm, gpu_relax):
    im = rad.RlxPre(model, use_gpu=gpu_asm)
    method = 9 if gpu_relax else 4
    res = rad.RlxAuto(im, 1e-5, 3000, method)
    bz = rad.Fld(model, 'b', [[25.0, 10.0, 0.0], [60.0, 30.0, 5.0]],
                 use_gpu=gpu_relax)
    return {"misfit": res[0], "b": bz}


def main():
    rank = comm.Get_rank()
    large = "--large" in sys.argv
    write_ref = "--reference" in sys.argv

    if comm.Get_size() > 1:
        rad.UtiMPI('in')

    combos = ([("cpu_asm_cpu_relax", False, False)] if large else
              [("gpu_asm_gpu_relax", True, True),
               ("gpu_asm_cpu_relax", True, False),
               ("cpu_asm_cpu_relax", False, False),
               ("cpu_asm_gpu_relax", False, True)])

    results = {}
    for name, gpu_asm, gpu_relax in combos:
        model = build_model(large=large)
        out = solve(model, gpu_asm=gpu_asm, gpu_relax=gpu_relax)
        if rank == 0:
            results[name] = out
            print(f"[rank 0] {name}: misfit={out['misfit']:.3e} "
                  f"bz(p0)={out['b'][0][2]:.8f}", flush=True)

    if rank == 0:
        key = "large" if large else "small"
        if write_ref:
            ref = {}
            if os.path.exists(REF_FILE):
                ref = json.load(open(REF_FILE))
            ref[key] = results
            json.dump(ref, open(REF_FILE, "w"), indent=1)
            print("REFERENCE WRITTEN")
        else:
            ref = json.load(open(REF_FILE))[key]
            for name, out in results.items():
                expect = ref[name]
                db = np.abs(np.array(out["b"]) - np.array(expect["b"])).max()
                scale = np.abs(np.array(expect["b"])).max()
                print(f"  {name}: max|dB|/scale = {db / scale:.3e}")
                assert db / scale < 1e-6, f"{name} deviates from serial reference"
            print("ALL MPI SOLVE TESTS PASSED")

    if comm.Get_size() > 1:
        rad.UtiMPI('off')


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    main()
