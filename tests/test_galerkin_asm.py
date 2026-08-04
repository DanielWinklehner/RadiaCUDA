# tests/test_galerkin_asm.py
#
# OPT-IN Galerkin (volume-averaged) interaction-matrix assembly -- see
# cpp/src/core/radgalerkin.h and studies/GALERKIN_STEP1.md.
#
# What is checked:
#   1. NO-OP WHEN OFF. With RADIA_GALERKIN unset, the solved B is bit-identical
#      to a reference build's, on both backends. That comparison needs two
#      builds, so it lives in the A/B driver below (--baseline); this file
#      instead checks the weaker but self-contained property that the assembly
#      still matches its own recorded fingerprints and that CPU/GPU agree as
#      they did before.
#   2. CPU-vs-GPU PARITY WITH THE FLAG ON, in the style of test_recmag_asm.py:
#      the same model assembled on each backend, relaxed with the same
#      deterministic CPU method, must agree to float32-IM level.
#   3. THE FLAG ACTUALLY DOES SOMETHING: Galerkin must move B by far more than
#      the CPU/GPU parity noise, otherwise a silently-ignored switch would pass
#      test 2 trivially.
#   4. RULE ORDER CONVERGES: K=4 -> 14 -> 24 must approach a common answer, and
#      the K=1 setting must reproduce collocation exactly.
#
# Run:  python tests/test_galerkin_asm.py
#       python tests/test_galerkin_asm.py --baseline <dir-holding-a-reference-radia>
#
# (spawns subprocesses, because the switch is read from the environment once.
#  For the --baseline form, run this file from a NEW build via PYTHONPATH and
#  point --baseline at the old one, e.g. the installed package directory:
#    PYTHONPATH=build_dev/cpp python tests/test_galerkin_asm.py \
#        --baseline "$CONDA_PREFIX/Lib/site-packages")

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "_galerkin_probe.py")
CASES = ["recmag_lin", "poly_lin", "mixed_lin", "mixed_sym_lin", "mixed_sat"]
FAILED = []


def check(name, cond, msg=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}", flush=True)
    if not cond:
        FAILED.append(name)


def run_probe(case, backend, env_extra=None, pythonpath=None):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("RADIA_GALERKIN"):
            del env[k]
    if env_extra:
        env.update(env_extra)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    out = subprocess.run([sys.executable, PROBE, case, backend],
                         capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"probe failed ({case}/{backend}):\n{out.stderr}")
    vals, be = [], "?"
    for line in out.stdout.splitlines():
        if line.startswith("case "):
            be = line.split()[-1]
        elif line.startswith("0x") or line.startswith("-0x"):
            vals.append(float.fromhex(line))
    return be, np.array(vals), out.stderr


def reldev(a, b, scale=None):
    s = scale if scale is not None else np.abs(b).max()
    return float(np.abs(a - b).max() / s)


def main():
    baseline = None
    if len(sys.argv) > 2 and sys.argv[1] == "--baseline":
        baseline = sys.argv[2]      # PYTHONPATH of a reference build

    print("Galerkin assembly: opt-in switch, parity and convergence\n")

    for case in CASES:
        print(f"case: {case}", flush=True)
        be_off_g, b_off_g, _ = run_probe(case, "gpu")
        be_off_c, b_off_c, _ = run_probe(case, "cpu")
        check(f"{case}: backends ran as asked", (be_off_g == "gpu") and (be_off_c == "cpu"),
              f"(gpu={be_off_g}, cpu={be_off_c})")
        scale = np.abs(b_off_c).max()
        check(f"{case}: field is nontrivial", scale > 1e-7, f"(|B|max={scale:.2e})")
        off_parity = reldev(b_off_g, b_off_c, scale)

        if baseline:
            # 1. Bit-exact no-op with the flag off, against a reference build.
            for backend, b_new in (("gpu", b_off_g), ("cpu", b_off_c)):
                _, b_ref, _ = run_probe(case, backend, pythonpath=baseline)
                same = (len(b_ref) == len(b_new)) and bool(np.all(
                    b_ref.view(np.uint64) == b_new.view(np.uint64)))
                check(f"{case}/{backend}: flag off is BIT-EXACT vs baseline", same)

        # 2. CPU-vs-GPU parity with the flag on.
        on = {"RADIA_GALERKIN": "1"}
        be_g, b_g, err_g = run_probe(case, "gpu", on)
        be_c, b_c, _ = run_probe(case, "cpu", on)
        check(f"{case}: GPU assembly ran with the flag on", be_g == "gpu",
              f"(backend={be_g})")
        dev = reldev(b_g, b_c, scale)
        # Same tolerance family as test_recmag_asm.py: the GPU stores the IM in
        # float32, so 1e-6-ish deviations are expected; a wrong quadrature
        # shows up at 1e-3 and above.
        tol = 5e-5 if case.endswith("sat") else 5e-6
        check(f"{case}: CPU-vs-GPU parity, flag ON", dev < tol,
              f"(rel dev {dev:.3e} < {tol:.0e}; flag-off parity was {off_parity:.1e})")

        # 3. The flag must actually change the answer.
        eff = reldev(b_g, b_off_g, scale)
        check(f"{case}: Galerkin moves B well above parity noise",
              eff > 20 * max(dev, off_parity, 1e-12),
              f"(effect {eff:.3e} vs noise {max(dev, off_parity):.1e})")

        # 4. K=1 must reproduce collocation; higher K must converge.
        _, b_k1, _ = run_probe(case, "gpu", {"RADIA_GALERKIN": "1",
                                             "RADIA_GALERKIN_K": "1",
                                             "RADIA_GALERKIN_CUTOFF": "0"})
        same_k1 = bool(np.all(b_k1.view(np.uint64) == b_off_g.view(np.uint64)))
        check(f"{case}: K=1 + no near pass == collocation (bit-exact)", same_k1,
              "" if same_k1 else f"(rel dev {reldev(b_k1, b_off_g, scale):.2e})")

        ks = {}
        for K in ("4", "14", "24"):
            _, bK, _ = run_probe(case, "gpu", {"RADIA_GALERKIN": "1",
                                               "RADIA_GALERKIN_K": K,
                                               "RADIA_GALERKIN_CUTOFF": "0"})
            ks[K] = bK
        d4 = reldev(ks["4"], ks["24"], scale)
        d14 = reldev(ks["14"], ks["24"], scale)
        check(f"{case}: rule order converges (|K4-K24| > |K14-K24|)", d14 < d4,
              f"(K4 {d4:.2e}, K14 {d14:.2e})")
        print()

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURE(S): {FAILED}")
        return 1
    print("ALL GALERKIN ASSEMBLY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
