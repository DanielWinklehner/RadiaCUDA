# tests/test_gpu_asm_cpu_solve.py
#
# The assembly backend and the solve backend are chosen INDEPENDENTLY, so all
# four combinations occur in practice. Three of them were already covered:
# GPU->GPU by most suites, CPU->CPU by the MPI reference in
# test_gpu_asm_rowtransform.py, CPU->GPU incidentally. The fourth --
# **GPU-assembled, CPU-solved** -- was not covered anywhere, and it is the one
# that breaks first if the host-side interaction matrix stops being
# materialized (studies/ASM_SOLVE_HANDOFF.md section 3): the CPU relaxation in
# radrlmet.cpp reads radTInteraction::InteractMatrix directly, so a null there
# is an immediate crash rather than a wrong number.
#
# Routes into it, all reachable from Python:
#   * anisotropic linear material (KsiPar != KsiPerp) -- the GPU SOLVER cannot
#     represent the full susceptibility tensor, so radGPU_PackInteractionData
#     warns (Warning022) and methods 9/11 hand off to CPU method 10, while the
#     assembly itself still ran on the GPU. This is the sharpest case: the
#     fallback runs the *same algorithm* as an explicit method-10 call, so the
#     two must agree BIT FOR BIT, not merely to a tolerance.
#   * RlxMan -- the manual CPU relaxation entry point, on a GPU-assembled matrix.
#
# The reverse direction (the GPU assembly declining, so the CPU assembly must
# still produce a usable matrix) is covered too, because the same change moves
# where that matrix is allocated:
#   * RlxPre(use_gpu=False)  -- the explicit switch
#   * an extruded polygon    -- no GPU assembly kernel for it (Warning020), so
#                               the assembly declines on its own. Same
#                               fall-through as an oversized model, without the
#                               O(N^2) CPU assembly a genuinely oversized model
#                               would cost.
#   * a genuinely VRAM-oversized model under UtiGpuFallback('cpu') -- the real
#     thing, opt-in via RADIA_TEST_OVERSIZED=1 (it needs ~12 GB of host RAM and
#     about a minute of single-threaded CPU assembly).
#
# Run:  python tests/test_gpu_asm_cpu_solve.py
#       $env:RADIA_TEST_OVERSIZED=1; python tests/test_gpu_asm_cpu_solve.py

import os
import subprocess
import sys

import radia as rad

# Determinism: Radia's length "reproducibility rounding" is a ~1e-9 rand()-based
# perturbation, on by default, which would make the bit-identity checks below
# meaningless.
rad.FldLenRndSw('off')

PTS = [[25.3, 7.1, 15.7], [18.2, -9.4, 22.3], [-11.6, 13.8, 9.2]]

# The reference (CPU) field must be clearly non-zero: a both-sides-zero
# regression would otherwise satisfy every equality check below.
FIELD_FLOOR = 1e-6

_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""),
          flush=True)
    if not ok:
        _failures.append(name)


def exact(vals):
    """Full-precision text form -- bit-identity, not a tolerance."""
    return " ".join("%.17e" % v for v in flat(vals))


def flat(x):
    out = []
    if isinstance(x, (list, tuple)):
        for v in x:
            out.extend(flat(v))
    else:
        out.append(float(x))
    return out


def maxabs(vals):
    f = flat(vals)
    return max(abs(v) for v in f) if f else 0.0


def maxdiff(a, b):
    fa, fb = flat(a), flat(b)
    return max(abs(x - y) for x, y in zip(fa, fb)) if fa else 0.0


def allfinite(vals):
    return all(v == v and abs(v) != float('inf') for v in flat(vals))


def build(ksi=(0.2, 0.2), rotated=True):
    """Subdivided block, linear material with remanence so every element is a
    real source. `rotated` puts a base transform on the container, which is what
    makes MainTransPtrArray non-identity -- the only thing that exercises the
    assembly's row transform."""
    rad.UtiDelAll()
    blk = rad.ObjRecMag([0, 0, 0], [12, 12, 12])
    rad.ObjDivMag(blk, [3, 3, 3])
    rad.MatApl(blk, rad.MatLin(list(ksi), [0.3, 0.5, 0.8]))
    cont = rad.ObjCnt([blk])
    if rotated:
        rad.TrfOrnt(cont, rad.TrfRot([0, 0, 0], [0, 0, 1], 0.5))
    return cont


def fld(obj):
    # CPU field eval: the field path is not what is under test here, and the CPU
    # one is deterministic.
    return rad.Fld(obj, 'b', PTS, use_gpu=False)


# --- GPU-assembled -> CPU-solved --------------------------------------------

def test_aniso_material_falls_back_to_cpu_solver():
    """A genuinely anisotropic linear material: assembled on the GPU, solved on
    the CPU. Methods 9 and 11 must both land on CPU method 10 and reproduce it
    exactly."""
    ref_res, ref_B = None, None
    for meth in (10, 9, 11):
        c = build(ksi=(0.6, 0.05))
        im = rad.RlxPre(c)
        asm = rad.UtiAsmLastBackend()
        res = rad.RlxAuto(im, 1e-6, 3000, meth)
        B = fld(c)

        if meth == 10:
            ref_res, ref_B = res, B
            check("aniso/m10: assembled on the GPU", asm == 'gpu', asm)
            check("aniso/m10: field non-trivial", maxabs(B) > FIELD_FLOOR,
                  f"max|B|={maxabs(B):.3e}")
            continue

        check(f"aniso/m{meth}: assembled on the GPU", asm == 'gpu', asm)
        check(f"aniso/m{meth}: solved on the CPU (== method 10, bit-identical)",
              exact(res) == exact(ref_res) and exact(B) == exact(ref_B),
              f"max|dB|={maxdiff(B, ref_B):.3e}")


def test_manual_relax_on_a_gpu_assembled_matrix():
    """RlxMan is CPU-only and reads InteractMatrix directly."""
    c = build()
    im = rad.RlxPre(c)
    asm_gpu = rad.UtiAsmLastBackend()
    rad.RlxMan(im, 3, 200, 1.0)
    B_gpu = fld(c)

    c = build()
    im = rad.RlxPre(c, use_gpu=False)
    asm_cpu = rad.UtiAsmLastBackend()
    rad.RlxMan(im, 3, 200, 1.0)
    B_cpu = fld(c)

    check("RlxMan: matrix assembled on the GPU", asm_gpu == 'gpu', asm_gpu)
    check("RlxMan: reference matrix assembled on the CPU", asm_cpu == 'cpu', asm_cpu)
    check("RlxMan: field non-trivial", maxabs(B_gpu) > FIELD_FLOOR,
          f"max|B|={maxabs(B_gpu):.3e}")
    # The two assemblies are independent implementations (float32 GPU vs double
    # CPU), so this is a tolerance check -- the point is that the CPU relaxation
    # got a real matrix, not a null or a zeroed one.
    rel = maxdiff(B_gpu, B_cpu) / max(maxabs(B_cpu), 1e-30)
    check("RlxMan: GPU-assembled result matches the CPU-assembled one",
          rel < 1e-4, f"max rel d={rel:.2e}")


def test_cpu_solve_then_gpu_solve_on_the_same_matrix():
    """Both representations of the same matrix, one after the other, on one
    interaction object: a CPU relaxation first (needs the host matrix), then a
    GPU solve (needs the device one)."""
    c = build()
    im = rad.RlxPre(c)
    rad.RlxMan(im, 3, 200, 1.0)
    B_man = fld(c)
    res = rad.RlxAuto(im, 1e-6, 3000, 9)
    B_auto = fld(c)

    check("CPU-then-GPU: manual pass produced a field",
          maxabs(B_man) > FIELD_FLOOR, f"max|B|={maxabs(B_man):.3e}")
    check("CPU-then-GPU: GPU solve after it produced a field",
          maxabs(B_auto) > FIELD_FLOOR and allfinite(B_auto),
          f"max|B|={maxabs(B_auto):.3e}")
    check("CPU-then-GPU: GPU solve converged", res[0] < 1e-5, f"misfit={res[0]:.2e}")


def test_repeat_gpu_solve_is_stable():
    """Two RlxAuto calls on one interaction: the second finds the device matrix
    cache warm, so it takes the skip-the-flatten path. Same answer."""
    c = build()
    im = rad.RlxPre(c)
    res1 = rad.RlxAuto(im, 1e-6, 3000, 9)
    B1 = fld(c)
    res2 = rad.RlxAuto(im, 1e-6, 3000, 9)
    B2 = fld(c)
    check("repeat GPU solve: field non-trivial", maxabs(B1) > FIELD_FLOOR,
          f"max|B|={maxabs(B1):.3e}")
    check("repeat GPU solve: bit-identical to the first",
          exact(res1) == exact(res2) and exact(B1) == exact(B2),
          f"max|dB|={maxdiff(B1, B2):.3e}")


# --- the GPU assembly declining ---------------------------------------------

def test_explicit_cpu_assembly():
    """RlxPre(use_gpu=False): the CPU assembly must produce a matrix usable by
    both solvers."""
    for meth, label in ((10, 'CPU'), (9, 'GPU')):
        c = build()
        im = rad.RlxPre(c, use_gpu=False)
        asm = rad.UtiAsmLastBackend()
        res = rad.RlxAuto(im, 1e-6, 3000, meth)
        B = fld(c)
        check(f"CPU assembly -> {label} solve: assembled on the CPU", asm == 'cpu', asm)
        check(f"CPU assembly -> {label} solve: converged to a real field",
              maxabs(B) > FIELD_FLOOR and allfinite(B) and res[0] < 1e-5,
              f"max|B|={maxabs(B):.3e} misfit={res[0]:.2e}")


def test_declined_gpu_assembly():
    """An extruded polygon has no GPU assembly kernel, so the assembly declines
    by itself (Warning020) and falls through to the CPU one -- the same
    fall-through an oversized model takes."""
    rad.UtiDelAll()
    e = rad.ObjThckPgn(0.0, 2.0, [[-4, -4], [4, -4], [0, 4]], 'x', [0, 0, 1])
    rad.MatApl(e, rad.MatLin([0.2, 0.2], [0.3, 0.5, 0.8]))
    im = rad.RlxPre(e)
    asm = rad.UtiAsmLastBackend()
    res = rad.RlxAuto(im, 1e-6, 3000, 9)
    B = fld(e)
    check("declined GPU assembly: fell through to the CPU assembly", asm == 'cpu', asm)
    check("declined GPU assembly: solved to a real field",
          maxabs(B) > FIELD_FLOOR and allfinite(B) and res[0] < 1e-5,
          f"max|B|={maxabs(B):.3e} misfit={res[0]:.2e}")


def _free_vram_bytes():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.split()[0]) * 1024 * 1024
    except Exception:
        return 0


def test_oversized_model_under_cpu_policy():
    """The real oversized case: a dense matrix too big for VRAM with the
    fallback policy left at 'cpu'. Opt-in -- it costs ~12 GB of host RAM and
    about a minute of single-threaded CPU assembly."""
    if not os.environ.get("RADIA_TEST_OVERSIZED"):
        print("  [SKIP] oversized model under 'cpu' policy "
              "(set RADIA_TEST_OVERSIZED=1 to run; ~12 GB RAM, ~1 min)", flush=True)
        return

    free = _free_vram_bytes()
    if free <= 0:
        print("  [SKIP] oversized model: could not read free VRAM", flush=True)
        return

    # The assembly declines when 36*N^2 * 1.15 > 0.90 * free. Aim well past it
    # so the branch is taken even if VRAM use shifts under us.
    n_thresh = (0.90 * free / (36.0 * 1.15)) ** 0.5
    n_elem = int(1.6 * n_thresh)
    n_side = max(2, int(round(n_elem ** (1.0 / 3.0))))
    n_elem = n_side ** 3
    print(f"  ... free VRAM {free/1e9:.1f} GB -> declines above ~{n_thresh:.0f} "
          f"elements; using {n_side}^3 = {n_elem} ({36.0*n_elem**2/1e9:.1f} GB matrix)",
          flush=True)

    rad.UtiGpuFallback('cpu')
    try:
        rad.UtiDelAll()
        c = rad.ObjRecMag([0, 0, 0], [100, 100, 100])
        rad.ObjDivMag(c, [n_side] * 3)
        rad.MatApl(c, rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8]))
        im = rad.RlxPre(c)
        asm = rad.UtiAsmLastBackend()
        if asm != 'cpu':
            print(f"  [SKIP] oversized model: GPU took it anyway (backend={asm!r}); "
                  "more VRAM free than the sizing assumed", flush=True)
            return
        check("oversized: GPU declined, CPU assembled", asm == 'cpu', asm)
        # A few passes only -- convergence is not the point, surviving the
        # assembly -> solve hand-off is.
        rad.RlxAuto(im, 1e-6, 20, 9)
        B = fld(c)
        check("oversized: CPU solve produced a real field",
              maxabs(B) > FIELD_FLOOR and allfinite(B), f"max|B|={maxabs(B):.3e}")
    finally:
        rad.UtiGpuFallback('cpu')
        rad.UtiDelAll()


def main():
    print("=== GPU-assembled -> CPU-solved (and the assembly declining) ===",
          flush=True)
    for fn in (test_aniso_material_falls_back_to_cpu_solver,
               test_manual_relax_on_a_gpu_assembled_matrix,
               test_cpu_solve_then_gpu_solve_on_the_same_matrix,
               test_repeat_gpu_solve_is_stable,
               test_explicit_cpu_assembly,
               test_declined_gpu_assembly,
               test_oversized_model_under_cpu_policy):
        fn()
    print(flush=True)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures), flush=True)
        return 1
    print("ALL PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
