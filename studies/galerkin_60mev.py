# studies/galerkin_60mev.py
#
# PHYSICS VALIDATION of the opt-in Galerkin (volume-averaged) interaction
# matrix: does the -6.7 kHz radia-vs-COMSOL mean_f gap on the 60 MeV cyclotron
# (HCHC-60 base magnetic model) close, shrink, or stay put?
#
# Runs ONE mesh rung twice -- collocation (today's default) and Galerkin -- in
# separate subprocesses, because the switch is read from the environment once,
# and because each rung wants a fresh CUDA/gmsh state.
#
# The rung itself comes from the (gitignored) cyclotron_optimizer driver
#   scripts/perturb_study/ladder_radia.py
# so nothing about the model recipe is duplicated here. Point --repo at that
# checkout if it is not at the default location.
#
# Usage:
#   python studies/galerkin_60mev.py <yoke_mm> <pole_mm> [--margin GB]
#   python studies/galerkin_60mev.py 260 160          # cheap reference point
#   python studies/galerkin_60mev.py --child <tag> <yoke> <pole>   # internal
#
# NOTE the results are saved under the ladder's own results/ directory with a
# "_GAL" / "_COL" tag suffix, so they never overwrite the user's saved rungs.

import argparse
import json
import os
import subprocess
import sys

DEFAULT_REPO = r"D:\Dropbox (Personal)\Code\Python\cyclotron_optimizer"

# COMSOL reference for the 60 MeV base magnetic model: 5-rung ladder,
# noise-limited to +-1.7 kHz (see the project's comsol_ladder.py).
COMSOL_MEAN_F = 7.726343


def child(repo, tag, yoke, pole):
    sys.path.insert(0, os.path.join(repo, "scripts", "perturb_study"))
    sys.path.insert(0, repo)
    import ladder_radia as lr
    lr.STRUCT_YOKE = None            # all-tet: the trustworthy family
    lr.STRUCT_POLE = None
    out = lr.run_rung(tag, float(yoke), float(pole))
    import numpy as np
    d = dict(np.load(out, allow_pickle=False))
    print("RESULT " + json.dumps({
        "tag": tag, "n_iron": int(d["n_iron"]),
        "mean_f": float(d["mean_f"]), "std_f": float(d["std_f"]),
        "misfit": float(d["misfit"]), "backend": str(d["backend"]),
        "t_asm": float(d["t_asm"]), "t_solve": float(d["t_solve"]),
        "npz": out}))
    return 0


CONFIGS = [
    ("COL", "collocation (default)", {}),
    ("GAL4", "Galerkin K=4 + near K14x8", {
        "RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "4",
        "RADIA_GALERKIN_CUTOFF": "1.5", "RADIA_GALERKIN_KNEAR": "14",
        "RADIA_GALERKIN_NEARLEV": "1", "RADIA_GALERKIN_DEBUG": "1"}),
    ("GAL14", "Galerkin K=14 + near K14x8", {
        "RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "14",
        "RADIA_GALERKIN_CUTOFF": "1.5", "RADIA_GALERKIN_KNEAR": "14",
        "RADIA_GALERKIN_NEARLEV": "1", "RADIA_GALERKIN_DEBUG": "1"}),
    ("GAL24", "Galerkin K=24 + near K14x8", {
        "RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "24",
        "RADIA_GALERKIN_CUTOFF": "1.5", "RADIA_GALERKIN_KNEAR": "14",
        "RADIA_GALERKIN_NEARLEV": "1", "RADIA_GALERKIN_DEBUG": "1"}),
    # STEP 1 says widening the near band beats raising the base order: the
    # 1.5-2.5 h pairs carry most of the base rule's remaining error and there
    # are few of them.
    ("GAL4C25", "Galerkin K=4 + near K14x8 to 2.5h", {
        "RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "4",
        "RADIA_GALERKIN_CUTOFF": "2.5", "RADIA_GALERKIN_KNEAR": "14",
        "RADIA_GALERKIN_NEARLEV": "1", "RADIA_GALERKIN_DEBUG": "1"}),
    ("GAL14C25", "Galerkin K=14 + near K14x8 to 2.5h", {
        "RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "14",
        "RADIA_GALERKIN_CUTOFF": "2.5", "RADIA_GALERKIN_KNEAR": "14",
        "RADIA_GALERKIN_NEARLEV": "1", "RADIA_GALERKIN_DEBUG": "1"}),
    # No near pass at all: isolates how much the near pass contributes, and
    # hence bounds the error from the near test not seeing symmetry images
    # (radgalerkin.h).
    ("GAL14N0", "Galerkin K=14, no near pass", {
        "RADIA_GALERKIN": "1", "RADIA_GALERKIN_K": "14",
        "RADIA_GALERKIN_CUTOFF": "0", "RADIA_GALERKIN_DEBUG": "1"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yoke", nargs="?", type=float, default=260.0)
    ap.add_argument("pole", nargs="?", type=float, default=160.0)
    ap.add_argument("--margin", default=None, help="VRAM safety margin [GB]")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--only", default=None, help="comma-separated config keys")
    ap.add_argument("--child", nargs=3, default=None,
                    help="internal: <tag> <yoke> <pole>")
    args = ap.parse_args()

    if args.child:
        return child(args.repo, args.child[0], args.child[1], args.child[2])

    keys = args.only.split(",") if args.only else None
    tagbase = f"GALSTUDY_{args.yoke:g}_{args.pole:g}"
    rows = []
    for key, label, env in CONFIGS:
        if keys and key not in keys:
            continue
        tag = f"{tagbase}_{key}"
        e = dict(os.environ)
        for k in list(e):
            if k.startswith("RADIA_GALERKIN"):
                del e[k]
        e.update(env)
        if args.margin:
            e["LADDER_IM_MARGIN_GB"] = str(args.margin)
        print(f"\n{'#' * 74}\n### {label}  (tag {tag})\n{'#' * 74}", flush=True)
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", tag,
             str(args.yoke), str(args.pole), "--repo", args.repo],
            env=e, text=True, capture_output=True)
        sys.stdout.write(p.stdout)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-4000:])
            print(f"### {label}: FAILED (rc={p.returncode})")
            continue
        # the Galerkin packing diagnostics go to stderr
        for line in p.stderr.splitlines():
            if "Galerkin" in line:
                print("    " + line)
        res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if res:
            rows.append((key, label, json.loads(res[-1][7:])))

    if not rows:
        return 1
    print(f"\n{'=' * 96}")
    print(f"60 MeV base magnetic model, all-tet yoke {args.yoke:g} / "
          f"pole {args.pole:g} mm")
    print(f"{'=' * 96}")
    print(f"{'scheme':>28} {'N_iron':>7} {'asm s':>7} {'misfit':>9} "
          f"{'mean_f [MHz]':>13} {'std_f [MHz]':>12} {'vs COMSOL':>11}")
    base = None
    for key, label, d in rows:
        gap = 1e3 * (d["mean_f"] - COMSOL_MEAN_F)
        print(f"{label:>28} {d['n_iron']:>7} {d['t_asm']:>7.1f} "
              f"{d['misfit']:>9.1e} {d['mean_f']:>13.6f} {d['std_f']:>12.6f} "
              f"{gap:>+10.3f} kHz")
        if key == "COL":
            base = d
    if base:
        print(f"\nchange vs collocation (COMSOL reference {COMSOL_MEAN_F:.6f} "
              f"MHz, +-1.7 kHz):")
        for key, label, d in rows:
            if key == "COL":
                continue
            print(f"  {label:>28}: d(mean_f) = "
                  f"{1e3 * (d['mean_f'] - base['mean_f']):+8.3f} kHz   "
                  f"d(std_f) = {1e3 * (d['std_f'] - base['std_f']):+8.3f} kHz   "
                  f"assembly {d['t_asm'] / base['t_asm']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
