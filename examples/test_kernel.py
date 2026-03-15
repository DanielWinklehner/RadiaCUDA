# examples/test_kernel.py
import radia as rad
import numpy as np
import time
from PyRadia import flatten
from PyRadia.field_kernel import fld_cpu, fld_gpu

rad.UtiDelAll()

# Single tet with magnetization
verts = [[0,0,0], [10,0,0], [0,10,0], [0,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
tet = rad.ObjPolyhdr(verts, faces, [0,0,1])

geo = flatten(tet)
geo.summary()

# Test points outside the tet
test_pts = np.array([
    [20, 0, 0],
    [0, 20, 0],
    [0, 0, 20],
    [5, 5, 20],
    [-10, -10, -10],
])

# Reference from Radia
print("\n=== Radia reference ===", flush=True)
for pt in test_pts:
    B = rad.Fld(tet, 'b', pt.tolist())
    print(f"  {pt} -> B = [{B[0]:.8f}, {B[1]:.8f}, {B[2]:.8f}]", flush=True)

# CPU fallback
print("\n=== CPU kernel ===", flush=True)
t0 = time.time()
B_cpu = fld_cpu(geo, test_pts)
print(f"Time: {time.time()-t0:.4f}s", flush=True)
for i, pt in enumerate(test_pts):
    print(f"  {pt} -> B = [{B_cpu[i,0]:.8f}, {B_cpu[i,1]:.8f}, {B_cpu[i,2]:.8f}]", flush=True)

# Compare
print("\n=== Comparison ===", flush=True)
for i, pt in enumerate(test_pts):
    B_ref = np.array(rad.Fld(tet, 'b', pt.tolist()))
    err = np.linalg.norm(B_ref - B_cpu[i])
    ref_norm = np.linalg.norm(B_ref)
    rel_err = err / ref_norm if ref_norm > 1e-15 else err
    status = "PASS" if rel_err < 1e-6 else "FAIL"
    print(f"  Point {i}: rel_err={rel_err:.2e}  {status}", flush=True)

# GPU if available
try:
    print("\n=== GPU kernel ===", flush=True)
    t0 = time.time()
    B_gpu = fld_gpu(geo, test_pts)
    print(f"Time: {time.time()-t0:.4f}s (includes compilation)", flush=True)
    for i, pt in enumerate(test_pts):
        B_ref = np.array(rad.Fld(tet, 'b', pt.tolist()))
        err = np.linalg.norm(B_ref - B_gpu[i])
        ref_norm = np.linalg.norm(B_ref)
        rel_err = err / ref_norm if ref_norm > 1e-15 else err
        status = "PASS" if rel_err < 1e-6 else "FAIL"
        print(f"  Point {i}: rel_err={rel_err:.2e}  {status}", flush=True)

    # Second call to measure without compilation overhead
    t0 = time.time()
    B_gpu2 = fld_gpu(geo, test_pts)
    print(f"Time (cached kernel): {time.time()-t0:.4f}s", flush=True)

except Exception as e:
    print(f"GPU not available: {e}", flush=True)
