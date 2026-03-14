# examples/RADIA_Example04.py

#############################################################################
# RADIA Python Example #4: Uniformly magnetized polyhedron (sphere approx.)
#############################################################################

import math
import radia as rad
from uti_plot import uti_plot1d, uti_plot_show

print("RADIA Python Example #4:")
print("Uniformly magnetized polyhedron created with radObjMltExtPgn.")
print("Field inside is uniform and equal to 2/3 Tesla as expected.")
print("")


def SphericalVolume(_r, _n_phi, _nz, _M):
    dz = 2. * _r / _nz
    z = -_r + dz
    dPhi = 2. * math.pi / _n_phi
    allSlicePgns = [[[[0., 0.]], -_r]]
    for i in range(1, _nz):
        theta = math.asin(z / _r)
        cosTheta = math.cos(theta)
        phi = dPhi
        slicePgn = [[_r * cosTheta, 0.]]
        for k in range(1, _n_phi):
            slicePgn.append([_r * math.cos(phi) * cosTheta,
                             _r * math.sin(phi) * cosTheta])
            phi += dPhi
        allSlicePgns.append([slicePgn, z])
        z += dz
    allSlicePgns.append([[[0., 0.]], _r])
    return rad.ObjMltExtPgn(allSlicePgns, _M)


if __name__ == "__main__":

    aSpherMag = SphericalVolume(1, 15, 15, [1, 0, 0])
    rad.ObjDrwAtr(aSpherMag, [0, 0.5, 0.8])

    try:
        rad.ObjDrwOpenGL(aSpherMag)
    except Exception:
        print("OpenGL viewer not available, skipping 3D display")

    print("Field in the Center =", rad.Fld(aSpherMag, "b", [0, 0, 0]))

    yMin = -0.99
    yMax = 0.99
    ny = 301
    yStep = (yMax - yMin) / (ny - 1)
    BxVsY = rad.Fld(aSpherMag, "bx", [[0, yMin + i * yStep, 0] for i in range(ny)])

    uti_plot1d(BxVsY, [yMin, yMax, ny], ["Longitudinal Position", "Bx", "Horizontal Magnetic Field"], ["mm", "T"])
    uti_plot_show()