# studies/galerkin_asm_timing.py
#
# Assembly wall-clock cost of the opt-in Galerkin (volume-averaged) interaction
# matrix, at several N and for several rules, on whichever backend RlxPre picks.
#
# The base rule is applied to EVERY pair (STEP 1 showed a distance cutoff is not
# usable for it -- see studies/GALERKIN_STEP1.md), so the expected cost is
# roughly K times the collocation assembly. The near pass adds an O(N) term that
# this script measures separately.
#
# Run (each configuration in its own process, since the switch is read from the
# environment once):
#   python studies/galerkin_asm_timing.py            # sweep, spawns children
#   python studies/galerkin_asm_timing.py <n_div>    # one measurement

import os
import subprocess
import sys
import time

import numpy as np


def build(n_div, tets=True):
    import radia as rad
    rad.UtiDelAll()
    if tets:
        # A tet lattice: subdivide a cube into cubes, each into 6 tets, so the
        # element population resembles the all-tet production models.
        n = n_div
        a = 10.0 / n
        ids = []
        cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        # 6-tet Kuhn decomposition of the unit cube
        kuhn = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
        faces = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    base = np.array([i, j, k], float) * a
                    for t in kuhn:
                        v = [(base + np.array(cube[q], float) * a).tolist() for q in t]
                        ids.append(rad.ObjPolyhdr(v, faces))
    else:
        ids = []
        n = n_div
        a = 10.0 / n
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ids.append(rad.ObjRecMag(
                        [(i + .5) * a, (j + .5) * a, (k + .5) * a], [a, a, a]))
    iron = rad.ObjCnt(ids)
    rad.MatApl(iron, rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2]))
    coil = rad.ObjRaceTrk([5, 5, 30], [20, 30], [0, 0], 12, 8, 3.0, "man", "z")
    return rad.ObjCnt([iron, coil]), len(ids)


def measure(n_div, tets=True):
    import radia as rad
    model, n_elem = build(n_div, tets)
    t0 = time.perf_counter()
    rad.RlxPre(model, use_gpu=True)
    dt = time.perf_counter() - t0
    return n_elem, dt, rad.UtiAsmLastBackend()


CONFIGS = [
    ("collocation (default)", {}),
    ("K=4",                   {"RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "4",
                               "RADIA_GALERKIN_CUTOFF": "0"}),
    ("K=4 + near K14x8",      {"RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "4",
                               "RADIA_GALERKIN_CUTOFF": "1.5",
                               "RADIA_GALERKIN_KNEAR": "14",
                               "RADIA_GALERKIN_NEARLEV": "1"}),
    ("K=14",                  {"RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "14",
                               "RADIA_GALERKIN_CUTOFF": "0"}),
]


def main():
    if len(sys.argv) > 1:
        n_div = int(sys.argv[1])
        tets = (len(sys.argv) < 3) or (sys.argv[2] != "rec")
        n_elem, dt, backend = measure(n_div, tets)
        print(f"RESULT {n_elem} {dt:.4f} {backend}")
        return 0

    divs = [4, 6, 8, 10, 12]
    print("=" * 78)
    print("Galerkin assembly timing (RlxPre wall clock, GPU backend)")
    print("=" * 78)
    hdr = f"{'N':>8s}"
    for label, _ in CONFIGS:
        hdr += f" {label:>22s}"
    print(hdr)
    print("-" * len(hdr))
    for d in divs:
        row = {}
        n_elem = None
        for label, env in CONFIGS:
            e = dict(os.environ)
            for k in list(e):
                if k.startswith("RADIA_GALERKIN"):
                    del e[k]
            e.update(env)
            # Best of two: GPU clocks drift enough between runs that a single
            # measurement can be off by tens of percent at these sizes.
            best = None
            for _rep in range(2):
                out = subprocess.run([sys.executable, os.path.abspath(__file__), str(d)],
                                     capture_output=True, text=True, env=e)
                line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
                if not line:
                    print(f"  {d}: {label} FAILED\n{out.stderr[-500:]}")
                    break
                _, ne, dt, backend = line[0].split()
                n_elem = int(ne)
                if (best is None) or (float(dt) < best[0]):
                    best = (float(dt), backend)
            row[label] = best
        base = row.get("collocation (default)")
        s = f"{n_elem:8d}"
        for label, _ in CONFIGS:
            v = row.get(label)
            if v is None:
                s += f" {'--':>22s}"
            else:
                mult = f" ({v[0] / base[0]:.2f}x)" if base else ""
                s += f" {v[1][:3] + ' ' + format(v[0], '.2f') + 's' + mult:>22s}"
        print(s, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
