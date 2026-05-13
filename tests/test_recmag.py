# examples/test_recmag.py
import radia as rad
import numpy as np
from PyRadia import flatten
from PyRadia.field_kernel import fld_gpu

use_sym = False

rad.UtiDelAll()

# Single RecMag block
r = rad.ObjRecMag([0,-3,0], [2, 3, 4], [0,0,1])

if use_sym:
    rad.TrfZerPerp(r, [0, 0, 0], [1, -1, 0])   # Mirror across x=y plane
    model_symmetries = [
        ('perp', [0, 0, 0], [1, -1, 0])
    ]

geo = flatten(r)
geo.summary()

test_pts = np.array([
    [5.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 5.0],
    [3.0, 3.0, 3.0],
    [-5.0, -5.0, -5.0],
])

print("\n=== CPU ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist(), use_gpu=False)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU C++ ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist())
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU CuPY ===", flush=True)
B = fld_gpu(geo, test_pts)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)
