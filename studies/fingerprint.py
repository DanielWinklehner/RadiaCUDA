# studies/fingerprint.py
#
# Bit-exact fingerprint of the assembly -> solve pipeline.
#
# The suites in tests/ compare GPU against CPU at ~1e-11, which is the right
# tolerance for "these two independent implementations agree" but blind to a
# change that shifts the LAST couple of digits of the GPU path itself. Phase 1
# of the assembly hand-off (studies/ASM_SOLVE_HANDOFF.md) shipped exactly such a
# regression -- nvcc contracting x*y+z into an FMA, rounding once where the host
# rounds twice -- and every suite passed with it present. Only a bit-exact diff
# against the previous build could see it.
#
# So: print full-precision (%.17e) results for a fixed set of cases, run it on
# the reference build, apply the change, rebuild, run it again, and require the
# two outputs to be IDENTICAL.
#
#   python studies/fingerprint.py > before.txt
#   ... edit, rebuild ...
#   python studies/fingerprint.py > after.txt
#   diff before.txt after.txt      # must be empty
#
# The FOUR geometry cases are plain / rotated / mirrored / rotated+mirrored.
# **The rotated case is the one that matters**: a base transform on the
# container is what makes MainTransPtrArray non-identity, and that is the only
# thing that exercises the row transform (s*M_inv) the assembly kernel applies
# at store time. With a plain block that code path never runs, and a mirror
# symmetry has an identity first copy, so it does not run there either.
#
# Each geometry is solved four ways, because the assembly's output reaches the
# solvers by different routes and phase 3 changes those routes:
#   m4  CPU relaxation      -- reads radTInteraction::InteractMatrix
#   m9  GPU Jacobi          -- reads the device-resident matrix
#   m11 GPU Newton-Krylov   -- same, different solver
#   man RlxMan (method 3)   -- the manual CPU relaxation entry point
#
# Field evaluation is forced to the CPU (use_gpu=False) so only the assembly and
# the solve are under test, and length randomization is off so the CPU field is
# deterministic.
#
# Run from PowerShell; point PYTHONPATH at build_dev/cpp to test the dev build
# rather than the installed site-packages radia.pyd.

import sys

import radia as rad

PTS = [[25.3, 7.1, 15.7], [18.2, -9.4, 22.3], [-11.6, 13.8, 9.2]]


def build(rotated, mirrored):
    """A subdivided block (linear material with remanence, so every element is a
    real source) plus a tet, in a container. `rotated` puts a base transform on
    the container -> non-identity MainTransPtrArray for every element."""
    rad.UtiDelAll()
    rad.FldLenRndSw('off')

    blk = rad.ObjRecMag([0, 0, 0], [12, 12, 12])
    rad.ObjDivMag(blk, [3, 3, 3])
    rad.MatApl(blk, rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8]))

    verts = [[10, 0, 10], [20, 0, 10], [10, 10, 10], [10, 0, 20]]
    faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    tet = rad.ObjPolyhdr(verts, faces, [0, 0, 0])
    rad.MatApl(tet, rad.MatLin([0.2, 0.2], [0.1, 0.2, 0.4]))

    cont = rad.ObjCnt([blk, tet])
    if rotated:
        rad.TrfOrnt(cont, rad.TrfRot([0, 0, 0], [0, 0, 1], 0.5))
    if mirrored:
        rad.TrfZerPara(cont, [0, 0, 0], [1, 0, 0])
    return cont


def fmt(vals):
    return " ".join("%.17e" % v for v in vals)


def flat(x):
    out = []
    if isinstance(x, (list, tuple)):
        for v in x:
            out.extend(flat(v))
    else:
        out.append(float(x))
    return out


def run(tag, rotated, mirrored, solver):
    cont = build(rotated, mirrored)
    im = rad.RlxPre(cont)
    asm = rad.UtiAsmLastBackend()

    if solver == 'man':
        res = rad.RlxMan(im, 3, 200, 1.0)
    else:
        res = rad.RlxAuto(im, 1e-6, 3000, int(solver[1:]))

    B = rad.Fld(cont, 'b', PTS, use_gpu=False)
    print(f"{tag}/{solver} asm={asm} res={fmt(flat(res))} B={fmt(flat(B))}",
          flush=True)


def main():
    print("# RadiaCUDA assembly->solve fingerprint", flush=True)
    for tag, rotated, mirrored in (("plain", False, False),
                                   ("rotated", True, False),
                                   ("mirrored", False, True),
                                   ("rot+mir", True, True)):
        for solver in ("m4", "m9", "m11", "man"):
            run(tag, rotated, mirrored, solver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
