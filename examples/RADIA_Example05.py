# examples/RADIA_Example05.py

#############################################################################
# RADIA Python Example #5: Simple Dipole Magnet
#############################################################################

import math
import time
import radia as rad
from uti_plot import uti_plot1d, uti_plot2d1d, uti_plot_show

print("RADIA Python Example #5:")
print("Iron dominated dipole steerer.")
print("")


def Geom(circ):

    eps = 0
    ironcolor = [0, 0.5, 1]
    coilcolor = [1, 0, 0]

    lx1 = thick / 2
    ly1 = width
    lz1 = 20
    k1 = [[thick / 4 - chamfer / 2, 0, gap / 2], [thick / 2 - chamfer, ly1 - 2 * chamfer]]
    k2 = [[thick / 4, 0, gap / 2 + chamfer], [thick / 2, ly1]]
    k3 = [[thick / 4, 0, gap / 2 + lz1], [thick / 2, ly1]]
    g1 = rad.ObjMltExtRtg([k1, k2, k3])
    rad.ObjDivMag(g1, n1)

    lx2 = thick / 2
    ly2 = ly1
    lz2 = 30
    l2 = [lx2, ly2, lz2]
    p2 = [thick / 4, 0, lz1 + gap / 2 + lz2 / 2 + 1 * eps]
    g2 = rad.ObjRecMag(p2, l2)
    rad.ObjDivMag(g2, n2)

    lx3 = thick / 2
    ly3 = ly2
    lz3 = ly2 * 1.25
    l3 = [lx3, ly3, lz3]
    p3 = [thick / 4, 0, lz1 + gap / 2 + lz2 + lz3 / 2 + 2 * eps]
    g3 = rad.ObjRecMag(p3, l3)

    typ = [[p3[0], p3[1] + ly3 / 2, p3[2] - lz3 / 2], [1, 0, 0],
           [p3[0], p3[1] - ly3 / 2, p3[2] - lz3 / 2], lz3 / ly3]

    if circ == 1:
        rad.ObjDivMag(g3, [nbr, nbp, n3[1]], "cyl", typ)
    else:
        rad.ObjDivMag(g3, n3)

    lx4 = thick / 2
    ly4 = 80
    lz4 = lz3
    l4 = [lx4, ly4, lz4]
    p4 = [thick / 4, ly3 / 2 + eps + ly4 / 2, p3[2]]
    g4 = rad.ObjRecMag(p4, l4)
    rad.ObjDivMag(g4, n4)

    lx5 = thick / 2
    ly5 = lz4 * 1.25
    lz5 = lz4
    l5 = [lx5, ly5, lz5]
    p5 = [thick / 4, p4[1] + eps + (ly4 + ly5) / 2, p4[2]]
    g5 = rad.ObjRecMag(p5, l5)

    typ = [[p5[0], p5[1] - ly5 / 2, p5[2] - lz5 / 2], [1, 0, 0],
           [p5[0], p5[1] + ly5 / 2, p5[2] - lz5 / 2], lz5 / ly5]

    if circ == 1:
        rad.ObjDivMag(g5, [nbr, nbp, n5[0]], "cyl", typ)
    else:
        rad.ObjDivMag(g5, n5)

    lx6 = thick / 2
    ly6 = ly5
    lz6 = gap / 2 + lz1 + lz2
    l6 = [lx6, ly6, lz6]
    p6 = [thick / 4, p5[1], p5[2] - (lz6 + lz5) / 2 - eps]
    g6 = rad.ObjRecMag(p6, l6)
    rad.ObjDivMag(g6, n6)

    Rmin = 5
    Rmax = 40
    Nseg = 4
    H = 2 * lz6 - 5
    CurDens = current / H / (Rmax - Rmin)
    pc = [0, p6[1], 0]
    coil = rad.ObjRaceTrk(pc, [Rmin, Rmax], [thick, ly6], H, 3, CurDens)
    rad.ObjDrwAtr(coil, coilcolor)

    g = rad.ObjCnt([g1, g2, g3, g4, g5, g6])
    rad.ObjDrwAtr(g, ironcolor)
    rad.MatApl(g, ironmat)
    t = rad.ObjCnt([g, coil])

    rad.TrfZerPerp(g, [0, 0, 0], [1, 0, 0])
    rad.TrfZerPara(g, [0, 0, 0], [0, 0, 1])
    return t


if __name__ == "__main__":

    gap = 10
    thick = 50
    width = 40
    chamfer = 8
    current = -2000

    nx = 2
    nbp = 2
    nbr = 2

    n1 = [nx, 3, 2]
    n2 = [nx, 2, 2]
    n3 = [nx, 2, 2]
    n4 = [nx, 2, 2]
    n5 = [nx, 2, 2]
    n6 = [nx, 2, 2]

    t0 = time.time()
    rad.UtiDelAll()
    ironmat = rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2])
    t = Geom(1)
    size = rad.ObjDegFre(t)

    try:
        rad.ObjDrwOpenGL(t)
    except Exception:
        print("OpenGL viewer not available, skipping 3D display")

    t1 = time.time()
    res = rad.Solve(t, 0.0001, 1500, 4)
    t2 = time.time()

    print(res[0])

    b0 = rad.Fld(t, "Bz", [0, 0, 0])
    bampere = (-4 * math.pi * current / gap) / 10000
    r = b0 / bampere

    print("Solving results for the segmentation by elliptical cylinders in the corners:")
    print("Mag_Max  H_Max  N_Iter =", round(res[1], 5), "T ", round(res[2], 5), "T ", round(res[3]))
    print("Built & Solved in", round(t1 - t0, 2), "&", round(t2 - t1, 2), "seconds")
    print("Interaction Matrix :", size, "X", size, "or", round(size * size * 4 / 1000000, 3), "MBytes")
    print("Bz =", round(b0, 4), "T,   Bz Computed / Bz Ampere Law =", round(r, 4))

    t1 = time.time()
    z = 1
    rmax = 30
    np = 40
    rstep = 2 * rmax / (np - 1)
    BzVsXY = rad.Fld(t, "bz", [[-rmax + ix * rstep, -rmax + iy * rstep, z]
                                 for iy in range(np) for ix in range(np)])

    uti_plot2d1d(BzVsXY, [-rmax, rmax, np], [-rmax, rmax, np], x=0, y=0,
                 labels=("X", "Y", "Bz in Magnet Gap at Z = " + repr(z) + " mm"),
                 units=["mm", "mm", "T"])

    z = 3
    IBzVsY = [rad.FldInt(t, "inf", "ibz", [-1, -rmax + iy * rstep, z],
                          [1, -rmax + iy * rstep, z]) for iy in range(np)]
    print("Field calculation after solving done in", round(t1 - t0, 2), "seconds")

    uti_plot1d(IBzVsY, [-rmax, rmax, np],
               ["Y", "Vertical Field Integral",
                "Vertical Field Integral along X at Z = " + repr(z) + " mm"], ["mm", "T"])

    print("")
    print("Close all magnetic field graphs to continue this example.")
    uti_plot_show()

    t = Geom(0)

    try:
        rad.ObjDrwOpenGL(t)
    except Exception:
        print("OpenGL viewer not available, skipping 3D display")

    t1 = time.time()
    res = rad.Solve(t, 0.0001, 1500, 4)
    t2 = time.time()

    b0 = rad.Fld(t, "Bz", [0, 0, 0])
    bampere = (-4 * math.pi * current / gap) / 10000
    r = b0 / bampere

    print("")
    print("Solving results for the rectangular segmentation in the corners:")
    print("Mag_Max  H_Max  N_Iter =", round(res[1], 5), "T ", round(res[2], 5), "T ", round(res[3]))
    print("Built & Solved in", round(t1 - t0, 2), "&", round(t2 - t1, 2), "seconds")
    print("Bz =", round(b0, 4), "T,   Bz Computed / Bz Ampere Law =", round(r, 4))