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

# Radia C++ CUDA
print("\n=== C++ CUDA ===", flush=True)
B = rad.Fld(tet, 'b', test_pts.tolist())
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)

print("\n=== CuPY ===", flush=True)
print("=== CuPy face data ===")
for i in range(geo.n_faces_total):
    e_start = geo.edge_offsets[i]
    e_end = geo.edge_offsets[i+1]
    nv = e_end - e_start
    print(f"  Face {i}: nv={nv}")
    print(f"    normal={geo.face_normals[i]}")
    print(f"    origin={geo.face_origins[i]}")
    print(f"    coord_z={geo.face_coord_z[i]}")
    print(f"    3D verts:")
    f_start = geo.face_offsets[0]  # first element
    for v in range(nv):
        print(f"      v{v}={geo.face_vertices_3d[e_start + v]}")
    print(f"    2D edges:")
    for v in range(nv):
        print(f"      e{v}={geo.face_edges_2d[e_start + v]}")
    print(f"    transform=\n{geo.face_transforms[i]}")

B = fld_gpu(geo, test_pts)
for b_, pt in zip(B, test_pts):
    print(f"  {pt} -> B = [{b_[0]:.8f}, {b_[1]:.8f}, {b_[2]:.8f}]", flush=True)
