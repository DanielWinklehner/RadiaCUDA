# examples/RADIA_Example02.py

#############################################################################
# RADIA Python Example #2: Racetrack and circular coils
# (4T superconducting wiggler geometry)
#############################################################################

import radia as rad
from uti_plot import uti_plot1d, uti_plot_show


def BuildGeometry():

    j1 = 128
    j2 = 256

    n1 = 3
    n2 = 6
    c2 = [1, 0, 0]
    c1 = [0, 1, 1]
    thcn = 0.001

    Rt1 = rad.ObjRaceTrk([0., 0., 38.], [9.5, 24.5], [120., 0.], 36, n1, j1)
    rad.ObjDrwAtr(Rt1, c1, thcn)
    Rt3 = rad.ObjRaceTrk([0., 0., 76.], [10., 25.], [90., 0.], 24, n1, j1)
    rad.ObjDrwAtr(Rt3, c1, thcn)
    Rt2 = rad.ObjRaceTrk([0., 0., 38.], [24.5, 55.5], [120., 0.], 36, n1, j2)
    rad.ObjDrwAtr(Rt2, c2, thcn)
    Rt4 = rad.ObjRaceTrk([0., 0., 76.], [25., 55.], [90., 0.], 24, n1, j2)
    rad.ObjDrwAtr(Rt4, c2, thcn)
    Rt5 = rad.ObjRaceTrk([0., 0., 60.], [150., 166.3], [0., 0.], 39, n2, -j2)
    rad.ObjDrwAtr(Rt5, c2, thcn)

    Grp = rad.ObjCnt([Rt1, Rt2, Rt3, Rt4, Rt5])

    rad.TrfZerPara(Grp, [0, 0, 0], [0, 0, 1])

    return Grp


def CalcField(g):

    yMin = 0.
    yMax = 300.
    ny = 301
    yStep = (yMax - yMin) / (ny - 1)
    xc = 0.
    zc = 0.
    BzVsY = rad.Fld(g, "bz", [[xc, yMin + iy * yStep, zc] for iy in range(ny)])

    xMin = 0.
    xMax = 400.
    nx = 201
    xStep = (xMax - xMin) / (nx - 1)
    zc = 0.
    IBzVsX = [
        rad.FldInt(g, "inf", "ibz", [xMin + ix * xStep, -300., zc], [xMin + ix * xStep, 300., zc])
        for ix in range(nx)
    ]

    return BzVsY, [yMin, yMax, ny], IBzVsX, [xMin, xMax, nx]


if __name__ == "__main__":

    g = BuildGeometry()
    print("SCW Geometry Index:", g)

    try:
        rad.ObjDrwOpenGL(g)
    except Exception:
        print("OpenGL viewer not available, skipping 3D display")

    BzVsY, MeshY, IBzVsX, MeshX = CalcField(g)

    print("Field in Center:", BzVsY[0], "T")
    print("Field Integral in Center:", IBzVsX[0], "T.mm")

    uti_plot1d(BzVsY, MeshY, ["Longitudinal Position", "Bz", "Vertical Magnetic Field"], ["mm", "T"])
    uti_plot1d(IBzVsX, MeshX, ["Horizontal Position", "Integral of Bz", "Vertical Magnetic Field Integral"], ["mm", "T.mm"])
    uti_plot_show()