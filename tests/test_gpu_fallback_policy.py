# tests/test_gpu_fallback_policy.py
#
# rad.UtiGpuFallback: the policy switch and the out-of-core (streaming) matvec.
# NOTE: distinct from tests/test_gpu_fallback.py, which covers the GPU *field*
# path falling back for unsupported element types (issue #3).
#
# The policy decides what happens when the GPU cannot service the interaction
# matrix -- almost always because the dense 36*N^2 matrix does not fit in VRAM.
# Nothing ever changes it by itself: dropping to the CPU can turn a seconds-long
# solve into an hours-long one, so leaving the GPU is always the caller's choice.
#
# The streaming path keeps rows [0, resident) on the device and moves only the
# remainder per matvec. Row order and the per-row accumulation are unchanged, so
# a streamed solve must be BIT-IDENTICAL to an in-core one. That is the property
# worth testing -- a streaming path that is fast but subtly different is useless
# -- and it is checked at several splits, including both boundaries.
#
# Run:  python tests/test_gpu_fallback_policy.py

import os
import subprocess
import sys

import radia as rad

PTS = [[30, 12, 7], [-25, 40, -18], [70, -30, 25]]
_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""),
          flush=True)
    if not ok:
        _failures.append(name)


def solve(n_side=10, method=11):
    """Relax a subdivided block and sample B. Remanent magnetization makes every
    element a source, so the relaxation has real work to do -- a soft-iron block
    with no source converges in 0 iterations and would make any comparison
    below vacuous."""
    rad.UtiDelAll()
    mat = rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8])
    c = rad.ObjRecMag([0, 0, 0], [100, 100, 100])
    rad.ObjDivMag(c, [n_side] * 3)
    rad.MatApl(c, mat)
    im = rad.RlxPre(c)
    res = rad.RlxAuto(im, 1e-6, 2000, method)
    return res, [rad.Fld(c, "b", p) for p in PTS]


# --- policy switch ----------------------------------------------------------
def test_policy_switch():
    check("default is 'cpu'", rad.UtiGpuFallback() == "cpu", rad.UtiGpuFallback())
    for mode in ("break", "gpu_streaming", "cpu"):
        ok = rad.UtiGpuFallback(mode) == mode and rad.UtiGpuFallback() == mode
        check(f"set + query {mode!r}", ok)
    for bad in ("CPU", "stream", "gpu-streaming", ""):
        try:
            rad.UtiGpuFallback(bad)
            check(f"reject {bad!r}", False, "accepted")
        except RuntimeError as e:
            check(f"reject {bad!r}", "expected" in str(e))
    check("rejection left the policy alone", rad.UtiGpuFallback() == "cpu")


def test_not_triggered_when_the_matrix_fits():
    """A model that fits must be unaffected whatever the policy is -- in
    particular 'break' must not fire on a healthy solve."""
    for mode in ("cpu", "break", "gpu_streaming"):
        rad.UtiGpuFallback(mode)
        solve(n_side=6)
        check(f"policy {mode!r} not triggered by a fitting model",
              rad.UtiAsmLastBackend() == "gpu", rad.UtiAsmLastBackend())
    rad.UtiGpuFallback("cpu")


def test_break_refuses_an_oversized_model():
    """30^3 elements -> 24.4 GiB matrix, past any consumer card. 'break' must
    raise rather than start a CPU assembly that would run for hours."""
    rad.UtiGpuFallback("break")
    try:
        rad.UtiDelAll()
        c = rad.ObjRecMag([0, 0, 0], [100, 100, 100])
        rad.ObjDivMag(c, [30, 30, 30])
        rad.MatApl(c, rad.MatStd("Steel42"))
        rad.RlxPre(c)
        check("'break' refuses an oversized model", False, "no error raised")
    except RuntimeError as e:
        check("'break' refuses an oversized model",
              "fallback policy is 'break'" in str(e), str(e)[:70])
    finally:
        rad.UtiGpuFallback("cpu")


# --- streaming: bit-identity across resident/streamed splits ----------------
# Each split runs in its own process: the split is chosen once, when the matrix
# is armed, from environment read at that moment.
_CHILD = """
import os, sys
sys.path.insert(0, r"{here}")
import radia as rad
frac = {frac!r}
if frac is not None:
    os.environ["RADGPU_FORCE_STREAM"] = "1"
    os.environ["RADGPU_STREAM_RESIDENT_FRAC"] = frac
    rad.UtiGpuFallback("gpu_streaming")
from test_gpu_fallback_policy import solve
print("RESULT", repr(solve()))
"""


def _run_child(frac):
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    env.pop("RADGPU_FORCE_STREAM", None)
    env.pop("RADGPU_STREAM_RESIDENT_FRAC", None)
    out = subprocess.run([sys.executable, "-c", _CHILD.format(here=here, frac=frac)],
                         capture_output=True, text=True, env=env, cwd=here)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            return line
    raise AssertionError(f"child produced no result:\n{out.stdout}\n{out.stderr}")


def test_streamed_is_bit_identical():
    ref = _run_child(None)
    for frac in ("0.0", "0.5", "1.0"):   # all streamed / split / all resident
        got = _run_child(frac)
        check(f"resident fraction {frac} == in-core (bit-identical)", got == ref,
              "" if got == ref else f"in-core {ref[:70]} vs stream {got[:70]}")


def main():
    print("=== UtiGpuFallback policy + out-of-core matvec ===", flush=True)
    for fn in (test_policy_switch, test_not_triggered_when_the_matrix_fits,
               test_break_refuses_an_oversized_model,
               test_streamed_is_bit_identical):
        fn()
    print(flush=True)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures), flush=True)
        return 1
    print("ALL PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
