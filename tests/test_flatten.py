import radia as rad
import numpy as np
from PyRadia import flatten

rad.UtiDelAll()

verts = [[0,0,0], [10,0,0], [0,10,0], [0,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
tet = rad.ObjPolyhdr(verts, faces, [0,0,1])

geo = flatten(tet)
geo.summary()

print("\nMagnetization:", geo.magnetizations)
print("Center:", geo.centers)
print("Faces per element:", np.diff(geo.face_offsets))
print("Edges per face:", np.diff(geo.edge_offsets))
print("Face normals:\n\n\n", geo.face_normals)

rad.UtiDelAll()

verts = [[0,0,0], [10,0,0], [0,10,0], [0,0,10]]
faces = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
tet = rad.ObjPolyhdr(verts, faces, [0,0,1])
rad.ObjDivMag(tet, [3,3,3])

geo = flatten(tet)
geo.summary()