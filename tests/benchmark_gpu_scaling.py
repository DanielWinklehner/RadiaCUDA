# examples/benchmark_gpu_scaling.py
import radia as rad
import numpy as np
import time
from PyRadia import flatten
from PyRadia.field_kernel import fld_gpu

print("=" * 70)
print(f"{'Elements':>10s} {'Points':>10s} {'CPU (s)':>10s} {'GPU (s)':>10s} {'Speedup':>10s} {'MaxErr':>10s}")
print("=" * 70)

for n_div in [2, 4, 6, 8]:
    rad.UtiDelAll()

    # Build geometry with increasing subdivision
    verts = [[0,0,0], [10,0,0], [0,10,0], [0,0,10]]
    faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
    tet = rad.ObjPolyhdr(verts, faces, [0,0,1])
    rad.ObjDivMag(tet, [n_div, n_div, n_div])

    geo = flatten(tet)
    n_total = geo.n_elem + geo.n_rec

    # Warm up GPU
    _ = fld_gpu(geo, np.array([[20.0, 0, 0]]))

    for Np in [1000, 10000, 100000]:
        pts = np.random.uniform(-20, 30, size=(Np, 3))

        # CPU
        if Np <= 100000:
            t0 = time.time()
            B_ref = np.array([rad.Fld(tet, 'b', p.tolist()) for p in pts])
            t_cpu = time.time() - t0
        else:
            t_cpu = None

        # GPU
        t0 = time.time()
        B_gpu = fld_gpu(geo, pts)
        t_gpu = time.time() - t0

        # Validate
        max_err = 0.0
        if t_cpu is not None:
            for i in range(min(20, Np)):
                err = np.linalg.norm(B_ref[i] - B_gpu[i])
                norm = np.linalg.norm(B_ref[i])
                if norm > 1e-15:
                    max_err = max(max_err, err / norm)

        cpu_str = f"{t_cpu:.3f}" if t_cpu else "skip"
        speedup = f"{t_cpu/t_gpu:.0f}x" if t_cpu else "N/A"
        err_str = f"{max_err:.1e}" if t_cpu else "N/A"
        print(f"{n_total:>10d} {Np:>10d} {cpu_str:>10s} {t_gpu:>10.4f} {speedup:>10s} {err_str:>10s}")

print("=" * 70)