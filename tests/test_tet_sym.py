# tests/test_recmag_sym.py
import radia as rad
import numpy as np
from PyRadia import flatten
from PyRadia.field_kernel import fld_gpu

print("\n=== X=Y Plane perpendicular symmetry ===\n")

rad.UtiDelAll()

# Single tet with magnetization
verts = [[10,0,0], [20,0,0], [10,10,0], [10,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
r = rad.ObjPolyhdr(verts, faces, [0,0,1])

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
    [5.0, 5.0, 0.0],
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
B = fld_gpu(geo, test_pts, symmetries=model_symmetries)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== Z=0 Plane parallel symmetry ===\n")

rad.UtiDelAll()

# Single tet with magnetization
verts = [[10,0,0], [20,0,0], [10,10,0], [10,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
r = rad.ObjPolyhdr(verts, faces, [0,0,1])

rad.TrfZerPara(r, [0, 0, 0], [0, 0, 1])    # Mirror across z=0 plane

model_symmetries = [
    ('para', [0, 0, 0], [0, 0, 1])
]

geo = flatten(r)
geo.summary()

print("\n=== CPU ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist(), use_gpu=False)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU C++ ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist())
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU CuPY ===", flush=True)
B = fld_gpu(geo, test_pts, symmetries=model_symmetries)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== X=Y Plane perp + Z=0 Plane parallel symmetry ===\n")

rad.UtiDelAll()

# Single tet with magnetization
verts = [[10,0,0], [20,0,0], [10,10,0], [10,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
r = rad.ObjPolyhdr(verts, faces, [0,0,1])

rad.TrfZerPerp(r, [0, 0, 0], [1, -1, 0])   # Mirror across x=y plane
rad.TrfZerPara(r, [0, 0, 0], [0, 0, 1])    # Mirror across z=0 plane

model_symmetries = [
    ('perp', [0, 0, 0], [1, -1, 0]),
    ('para', [0, 0, 0], [0, 0, 1])
]

geo = flatten(r)
geo.summary()

print("\n=== CPU ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist(), use_gpu=False)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU C++ ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist())
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU CuPY ===", flush=True)
B = fld_gpu(geo, test_pts, symmetries=model_symmetries)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== Full Cyclotron Symmetry: X=Y Plane perp + Z=0 Plane parallel + X=0 Plane perp + Y=0 Plane perp ===\n")

rad.UtiDelAll()

# Single tet with magnetization
verts = [[10,0,10], [20,0,10], [10,10,10], [10,0,20]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
r = rad.ObjPolyhdr(verts, faces, [0,0,1])

rad.TrfZerPerp(r, [0, 0, 0], [1, -1, 0])   # Mirror across x=y plane
rad.TrfZerPerp(r, [0, 0, 0], [1, 0, 0])   # Mirror across x=0 plane
rad.TrfZerPerp(r, [0, 0, 0], [0, 1, 0])   # Mirror across y=0 plane
rad.TrfZerPara(r, [0, 0, 0], [0, 0, 1])    # Mirror across z=0 plane

model_symmetries = [
    ('perp', [0, 0, 0], [1, -1, 0]),
    ('perp', [0, 0, 0], [1, 0, 0]),
    ('perp', [0, 0, 0], [0, 1, 0]),
    ('para', [0, 0, 0], [0, 0, 1])
]

geo = flatten(r)
geo.summary()

print("\n=== CPU ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist(), use_gpu=False)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU C++ ===", flush=True)
B = rad.Fld(r, 'b', test_pts.tolist())
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== GPU CuPY ===", flush=True)
B = fld_gpu(geo, test_pts, symmetries=model_symmetries)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)
