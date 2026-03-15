# examples/test_recmag.py
import radia as rad
import numpy as np
from PyRadia import flatten
from PyRadia.field_kernel import fld_gpu

rad.UtiDelAll()

# Single RecMag block
r = rad.ObjRecMag([0,0,0], [2,3,4], [0,0,1])

geo = flatten(r)
geo.summary()

pts = np.array([
    [5.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 5.0],
    [3.0, 3.0, 3.0],
    [-5.0, -5.0, -5.0],
])

print("\n=== Comparison ===")
for i, pt in enumerate(pts):
    B_ref = np.array(rad.Fld(r, 'b', pt.tolist()))
    B_gpu = fld_gpu(geo, pt.reshape(1,3))[0]
    err = np.linalg.norm(B_ref - B_gpu)
    norm = np.linalg.norm(B_ref)
    rel_err = err / norm if norm > 1e-15 else err
    status = "PASS" if rel_err < 1e-6 else "FAIL"
    print(f"  Ref:  [{B_ref[0]:+.8f}, {B_ref[1]:+.8f}, {B_ref[2]:+.8f}]")
    print(f"  GPU:  [{B_gpu[0]:+.8f}, {B_gpu[1]:+.8f}, {B_gpu[2]:+.8f}]")
    print(f"  err={rel_err:.2e}  {status}\n")
