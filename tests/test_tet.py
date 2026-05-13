# examples/test_kernel.py
import radia as rad
import numpy as np
import time
from PyRadia import flatten
from PyRadia.field_kernel import fld_gpu

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

print("\n=== CPU ===", flush=True)
B = rad.Fld(tet, 'b', test_pts.tolist(), use_gpu=False)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU C++ ===", flush=True)
B = rad.Fld(tet, 'b', test_pts.tolist())
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU CuPY ===", flush=True)
B = fld_gpu(geo, test_pts)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)
