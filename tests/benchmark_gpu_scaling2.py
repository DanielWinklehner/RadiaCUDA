# examples/benchmark_gpu_scaling2.py
import radia as rad
import numpy as np
import time
from PyRadia.radia_flatten import flatten
from PyRadia.field_kernel import fld_gpu, GPUGeometry

print("=" * 75)
print(f"{'Elements':>10s} {'Points':>10s} {'CPU (s)':>10s} {'GPU (s)':>10s} {'Flatten':>10s} {'Speedup':>10s}")
print("=" * 75)

for n_div in [2, 4, 6, 8, 10]:
    rad.UtiDelAll()

    verts = [[0,0,0], [10,0,0], [0,10,0], [0,0,10]]
    faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
    tet = rad.ObjPolyhdr(verts, faces, [0,0,1])
    rad.ObjDivMag(tet, [n_div, n_div, n_div])

    t0 = time.time()
    geo = flatten(tet)
    t_flat = time.time() - t0
    n_total = geo.n_elem + geo.n_rec

    # Cache geometry on GPU once
    gpu_geo = GPUGeometry(geo)

    # Warm up
    _ = fld_gpu(geo, np.array([[20.0, 0, 0]]), gpu_geo=gpu_geo)

    for Np in [1000, 10000, 100000]:
        pts = np.random.uniform(-20, 30, size=(Np, 3))

        # CPU
        if Np <= 100000 and n_total <= 220:
            t0 = time.time()
            B_ref = np.array([rad.Fld(tet, 'b', p.tolist()) for p in pts])
            t_cpu = time.time() - t0
        else:
            t_cpu = None

        # GPU (geometry already on GPU)
        t0 = time.time()
        B_gpu = fld_gpu(geo, pts, gpu_geo=gpu_geo)
        t_gpu = time.time() - t0

        cpu_str = f"{t_cpu:.3f}" if t_cpu else "skip"
        speedup = f"{t_cpu/t_gpu:.0f}x" if t_cpu else "N/A"
        flat_str = f"{t_flat:.3f}" if Np == 1000 else ""
        print(f"{n_total:>10d} {Np:>10d} {cpu_str:>10s} {t_gpu:>10.4f} {flat_str:>10s} {speedup:>10s}")

print("=" * 75)