# examples/benchmark_gpu.py
import radia as rad
import numpy as np
import time
from PyRadia import flatten
from PyRadia.field_kernel import fld_gpu

rad.UtiDelAll()

# Build a geometry with many polyhedra
verts = [[0,0,0], [10,0,0], [0,10,0], [0,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
tet = rad.ObjPolyhdr(verts, faces, [0,0,1])
rad.ObjDivMag(tet, [5,5,5])

geo = flatten(tet)
geo.summary()

# Generate observation points
for Np in [100, 1000, 10000, 100000]:
    pts = np.random.uniform(-20, 30, size=(Np, 3))

    # Radia CPU (only for small Np)
    if Np <= 100000:
        t0 = time.time()
        B_ref = np.array([rad.Fld(tet, 'b', p.tolist()) for p in pts])
        t_cpu = time.time() - t0
    else:
        t_cpu = None

    # GPU - warm up
    _ = fld_gpu(geo, pts[:10])

    # GPU - timed
    t0 = time.time()
    B_gpu = fld_gpu(geo, pts)
    t_gpu = time.time() - t0

    # Validate a few points
    if Np <= 100000:
        max_err = 0.0
        for i in range(min(10, Np)):
            ref = B_ref[i]
            err = np.linalg.norm(ref - B_gpu[i])
            norm = np.linalg.norm(ref)
            if norm > 1e-15:
                max_err = max(max_err, err/norm)

    cpu_str = f"{t_cpu:.3f}s" if t_cpu else "skipped"
    speedup = f"{t_cpu/t_gpu:.0f}x" if t_cpu else "N/A"
    print(f"Np={Np:>7d}  CPU: {cpu_str:>10s}  GPU: {t_gpu:.4f}s  Speedup: {speedup}")
    if Np <= 100000:
        print(f"           Max rel error: {max_err:.2e}")