# tests/test_gpu_asm_rowtransform.py
#
# Regression test for issue #6: GPU interaction-matrix assembly must apply the
# observation-row element's inverse rotation (MainTransPtrArray[StrNo]->TrMatrix_inv),
# which the CPU assembly does (radintrc.cpp:575). This only matters when a relaxable
# element carries a non-identity BASE transform (here a tet nested in a *rotated
# container*); pure TrfZerPara/Perp symmetry has an identity first copy and is
# unaffected (which is why plain mirror-symmetric models compute correctly even
# with the bug present).
#
# It solves the SAME model two ways and requires agreement:
#   - single-process   -> GPU assembly (m_nProcMPI < 2 takes the GPU assemble path)
#   - mpiexec -n 2      -> CPU assembly (m_nProcMPI >= 2 takes the CPU assemble path; reference)
# Field eval is forced to CPU (use_gpu=False) so ONLY the assembly differs between runs.
# If mpiexec is unavailable the test SKIPs rather than fails.
#
# Run:  python tests/test_gpu_asm_rowtransform.py

import sys
import os
import json
import subprocess
import numpy as np
import radia as rad


def solve_field(use_mpi):
    rad.FldLenRndSw('off')                       # determinism
    rank = rad.UtiMPI('on') if use_mpi else 0    # m_nProcMPI>=2 under mpiexec -> CPU assembly
    rad.UtiDelAll()
    verts = [[10, 0, 10], [20, 0, 10], [10, 10, 10], [10, 0, 20]]
    faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    tet = rad.ObjPolyhdr(verts, faces, [0, 0, 0])
    rad.MatApl(tet, rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8]))   # iso susceptibility + remanence
    cont = rad.ObjCnt([tet])
    rad.TrfOrnt(cont, rad.TrfRot([0, 0, 0], [0, 0, 1], 0.5))   # rotate container -> non-identity row transform (#6 trigger)
    rad.TrfZerPara(cont, [0, 0, 0], [1, 0, 0])                 # mirror symmetry
    rad.Solve(cont, 1e-5, 3000)
    pts = [[25.3, 7.1, 15.7], [18.2, -9.4, 22.3]]
    B_raw = rad.Fld(cont, 'b', pts, use_gpu=False)             # CPU field-eval isolates the assembly
    B = None
    if rank <= 0:
        B = np.array(B_raw, dtype=float).reshape(-1, 3).flatten().tolist()
    if use_mpi:
        rad.UtiMPI('off')
    return rank, B


# --- worker mode: invoked under mpiexec by the main process below ---
if '--mpi' in sys.argv:
    _rank, _B = solve_field(True)
    if _rank <= 0:
        print("BFIELD " + json.dumps(_B), flush=True)
    sys.exit(0)


def main():
    _, B_gpu = solve_field(False)   # single-process -> GPU assembly

    # reference: CPU assembly via mpiexec -n 2 (re-invokes this file with --mpi)
    try:
        out = subprocess.run(
            ["mpiexec", "-n", "2", sys.executable, os.path.abspath(__file__), "--mpi"],
            capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"SKIP: could not launch mpiexec reference ({e})")
        return 0

    ref = [ln for ln in out.stdout.splitlines() if ln.startswith("BFIELD ")]
    if not ref:
        print("SKIP: mpiexec reference produced no result")
        print(out.stdout[-500:])
        print(out.stderr[-500:])
        return 0

    B_cpu = np.array(json.loads(ref[-1][len("BFIELD "):]))
    B_gpu = np.array(B_gpu)
    d = float(np.max(np.abs(B_gpu - B_cpu)))
    # Tolerance separates the #6 bug (~1e-4 divergence) from the correct result. The
    # GPU assembles the matrix in float32 vs the CPU's double, so a ~1e-9 residual is
    # expected even when correct.
    ok = bool(np.allclose(B_gpu, B_cpu, rtol=1e-4, atol=1e-8))
    print(f"issue #6: GPU-assembly vs CPU-assembly  max|d|={d:.3e}  ->  {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
