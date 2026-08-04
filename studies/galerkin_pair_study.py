# studies/galerkin_pair_study.py
#
# STEP 1 of the Galerkin investigation: how expensive would volume-averaged
# (Galerkin) interaction-matrix assembly be?  Answered at PAIR level -- no
# solver, no GPU, no kernel.
#
# Radia enforces M = M_mat(H) at ONE point per element (the centroid), so the
# interaction entry between an observation element i and a source element j is
#
#     Q_ij = Q_j(r_i^centroid)                      "center collocation", K=1
#
# where Q_j(r) is the source's 3x3 field-per-unit-magnetization tensor
# (column c = H(r) for M = e_c).  Galerkin replaces that with the volume
# average over the OBSERVATION element,
#
#     Qbar_ij = (1/V_i) Integral_{V_i} Q_j(r) dV .
#
# Questions:
#   (a) relative error of K=1 vs the exact volume average as a function of the
#       separation ratio d/h  (d = centroid distance, h = mean V^(1/3), the
#       same length the cost model counts pairs with);
#   (b) the cutoff R = d/h beyond which K=1 is accurate enough;
#   (c) the quadrature order K needed inside R, including the SELF block and
#       face/edge/vertex-adjacent pairs.
#
# Q_j(r) comes from Radia itself -- one magnetized element and
# rad.Fld(obj, 'h', pts) returns exactly the tensor column the assembly kernel
# computes.  The reference volume average uses adaptive 1->8 tet subdivision
# with a degree-6 rule, error-estimated against degree-3, driven to a
# per-pair RELATIVE target so far-field entries are referenced as tightly as
# near ones.
#
# THEORY (verified numerically below).  For a source that does not overlap the
# observation element, each component of H is harmonic there, so
#
#     Qbar = Q(c) + (1/2) Mdev_ab d^2Q/dx_a dx_b (c) + O(4th moment)
#
# where Mdev is the DEVIATORIC part of the element's normalized second-moment
# tensor -- the isotropic part drops out because the Hessian of a harmonic
# function is traceless (this is the mean-value property).  Two consequences:
#   * elements whose second-moment tensor is isotropic (cube, regular tet)
#     have NO leading correction at all;
#   * any degree-2-exact quadrature rule (4 points on a tet) reproduces the
#     whole leading term exactly, so the residual is 4th order.
#
# Run:  python studies/galerkin_pair_study.py [--quick]

import argparse
import json
import os
import sys
import time

import numpy as np
import radia as rad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from galerkin_quad import (TET_FACES, TET_RULES, TET_RULE_NAMES, hex_rule,
                           tet_centroid, tet_rule_points, tet_subdivide,
                           tet_volume)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# d/h bins used everywhere (h = mean element V^(1/3) = mean spacing)
BINS = [0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 40.0]

# Composite (subdivided) rules for the near band: (label, single-cell rule,
# refinement levels, total point count).
COMPOSITES = [("K4x8", "K4", 1, 32), ("K14x8", "K14", 1, 112),
              ("K4x64", "K4", 2, 256)]
COMPOSITE_NAMES = [c[0] for c in COMPOSITES]


# ==================================================================
# Source tensor Q(r) from Radia
# ==================================================================

class SourceQ:
    """Q(r) of one uniformly magnetized element, via three unit-M objects."""

    def __init__(self, kind, geom):
        self.ids = []
        for c in range(3):
            m = [0.0, 0.0, 0.0]
            m[c] = 1.0
            if kind == "tet":
                self.ids.append(rad.ObjPolyhdr(np.asarray(geom, float).tolist(),
                                               TET_FACES, m))
            elif kind == "rec":
                cen, dims = geom
                self.ids.append(rad.ObjRecMag(list(map(float, cen)),
                                              list(map(float, dims)), m))
            else:
                raise ValueError(kind)
        self.n_evals = 0

    def q(self, pts):
        """(n,3) points -> (n,3,3) tensors; out[p,i,c] = H_i at p for M = e_c."""
        pts = np.asarray(pts, float)
        out = np.empty((len(pts), 3, 3))
        P = pts.tolist()
        for c in range(3):
            h = np.array(rad.Fld(self.ids[c], 'h', P), dtype=float).reshape(-1, 3)
            out[:, :, c] = h
        self.n_evals += 3 * len(pts)
        return out


# ==================================================================
# Quadrature of Q over an observation element
# ==================================================================

def tet_entry(src, verts, rule_name):
    pts, w = tet_rule_points(verts, rule_name)
    return np.einsum("q,qij->ij", w, src.q(pts))


def hex_entry(src, center, dims, K):
    P, W = hex_rule(K)
    pts = np.asarray(center, float) + P * (0.5 * np.asarray(dims, float))
    return np.einsum("q,qij->ij", W, src.q(pts))


def tet_entry_composite(src, verts, rule_name, levels):
    """Composite rule: subdivide the tet `levels` times (8^levels sub-tets) and
    apply `rule_name` on each.  The practical answer for near/touching pairs,
    where the integrand has a boundary layer at the shared face and raising the
    single-cell order helps far less than refining does."""
    cells = [np.asarray(verts, float)]
    for _ in range(levels):
        cells = [s for c in cells for s in tet_subdivide(c)]
    vols = np.array([tet_volume(c) for c in cells])
    bary, w, _ = TET_RULES[rule_name]
    pts = np.vstack([bary @ c for c in cells])
    Q = src.q(pts).reshape(len(cells), len(w), 3, 3)
    per = np.einsum("q,cqij->cij", w, Q)
    return np.einsum("c,cij->ij", vols / vols.sum(), per)


def adaptive_tet_average(fn, verts, tol_abs, max_pts=250000, hi="K24", lo="K8"):
    """Volume average of a vector/tensor field over a tet, adaptively.

    `fn(pts) -> (n, ...)`.  Refines the sub-tets carrying 80% of the estimated
    error until the total estimate drops below `tol_abs`.  Returns
    (average, abs_err_estimate, n_points).
    """
    verts = np.asarray(verts, float)
    cells = [verts]
    vols = np.array([tet_volume(verts)])
    Vtot = vols[0]
    n_pts = 0
    bary_hi, w_hi, _ = TET_RULES[hi]
    bary_lo, w_lo, _ = TET_RULES[lo]
    nh, nl = len(w_hi), len(w_lo)
    while True:
        allpts = np.empty((len(cells) * (nh + nl), 3))
        for k, c in enumerate(cells):
            allpts[k * (nh + nl):k * (nh + nl) + nh] = bary_hi @ c
            allpts[k * (nh + nl) + nh:(k + 1) * (nh + nl)] = bary_lo @ c
        V = np.asarray(fn(allpts), float)
        n_pts += len(allpts)
        tail = V.shape[1:]
        V = V.reshape(len(cells), nh + nl, *tail)
        Vhi = np.tensordot(w_hi, V[:, :nh], axes=([0], [1]))
        Vlo = np.tensordot(w_lo, V[:, nh:], axes=([0], [1]))
        ax = tuple(range(1, 1 + len(tail)))
        cell_err = np.sqrt((np.abs(Vhi - Vlo) ** 2).sum(axis=ax)) * vols / Vtot
        avg = np.tensordot(vols / Vtot, Vhi, axes=([0], [0]))
        err = float(cell_err.sum())
        if err < tol_abs or n_pts > max_pts:
            return avg, err, n_pts

        order = np.argsort(cell_err)[::-1]
        cum = np.cumsum(cell_err[order])
        n_ref = max(1, int(np.searchsorted(cum, 0.8 * cum[-1]) + 1))
        refine = set(order[:n_ref].tolist())
        new_cells, new_vols = [], []
        for k, c in enumerate(cells):
            if k in refine:
                for s in tet_subdivide(c):
                    new_cells.append(s)
                    new_vols.append(tet_volume(s))
            else:
                new_cells.append(c)
                new_vols.append(vols[k])
        cells, vols = new_cells, np.array(new_vols)


def tet_reference(src, verts, rel_target=1e-7, max_pts=250000, floor=1e-16):
    """Adaptive volume average of Q over a tet, to a target RELATIVE to the
    entry's own magnitude (a far-field entry of size 1e-6 is referenced to
    1e-13 absolute; otherwise the reference is no better than the rule it is
    meant to judge).  Returns (Qbar, abs_err, rel_err, n_points)."""
    scale = np.linalg.norm(tet_entry(src, verts, "K1"))
    tol_abs = max(rel_target * max(scale, 1e-12), floor)
    Qbar, err, n = adaptive_tet_average(src.q, verts, tol_abs, max_pts)
    return Qbar, err, err / max(np.linalg.norm(Qbar), 1e-300), n


def hex_reference(src, center, dims, rel_target=1e-7, max_pts=250000):
    """Volume average of Q over a cuboid: Gauss-Legendre with octree fallback."""
    cen = np.asarray(center, float)
    dim = np.asarray(dims, float)
    scale = np.linalg.norm(hex_entry(src, cen, dim, 1))
    tol_abs = max(rel_target * max(scale, 1e-12), 1e-16)

    cells = [(cen, dim)]
    n_pts = 0
    Phi, Whi = hex_rule(6)
    Plo, Wlo = hex_rule(4)
    nh, nl = len(Whi), len(Wlo)
    while True:
        pts = []
        for c, d in cells:
            pts.append(c + Phi * (0.5 * d))
            pts.append(c + Plo * (0.5 * d))
        Q = src.q(np.vstack(pts))
        n_pts += len(Q)
        Qhi = np.empty((len(cells), 3, 3))
        Qlo = np.empty((len(cells), 3, 3))
        off = 0
        for k in range(len(cells)):
            Qhi[k] = np.einsum("q,qij->ij", Whi, Q[off:off + nh]); off += nh
            Qlo[k] = np.einsum("q,qij->ij", Wlo, Q[off:off + nl]); off += nl
        vols = np.array([float(np.prod(d)) for _, d in cells])
        Vtot = vols.sum()
        cell_err = np.linalg.norm(Qhi - Qlo, axis=(1, 2)) * vols / Vtot
        Qbar = np.einsum("c,cij->ij", vols / Vtot, Qhi)
        err = float(cell_err.sum())
        if err < tol_abs or n_pts > max_pts:
            mag = np.linalg.norm(Qbar)
            return Qbar, err, err / max(mag, 1e-300), n_pts
        order = np.argsort(cell_err)[::-1]
        cum = np.cumsum(cell_err[order])
        n_ref = max(1, int(np.searchsorted(cum, 0.8 * cum[-1]) + 1))
        refine = set(order[:n_ref].tolist())
        new = []
        for k, (c, d) in enumerate(cells):
            if k in refine:
                hd = 0.5 * d
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            new.append((c + 0.5 * hd * np.array([sx, sy, sz]), hd))
            else:
                new.append((c, d))
        cells = new


# ==================================================================
# Element shape descriptors
# ==================================================================

def second_moment(verts):
    """(1/V) Integral (r-c)(x)(r-c) dV over a tet, exactly.

    For a tetrahedron with vertices v_i and centroid c,
        (1/V) Int (r-c)(x)(r-c) dV = (1/20) Sum_i (v_i-c)(x)(v_i-c).
    """
    v = np.asarray(verts, float)
    c = v.mean(axis=0)
    u = v - c
    return (u.T @ u) / 20.0


def dev_ratio(verts):
    """||deviatoric second moment|| / V^(2/3): the leading Galerkin error
    coefficient of an observation element, normalized to be shape-only.

    0 for any element whose second-moment tensor is isotropic (cube, regular
    tetrahedron); grows with anisotropy.
    """
    M = second_moment(verts)
    dev = M - np.trace(M) / 3.0 * np.eye(3)
    return float(np.linalg.norm(dev) / tet_volume(verts) ** (2.0 / 3.0))


def aspect(verts):
    v = np.asarray(verts, float)
    e = [np.linalg.norm(v[a] - v[b]) for a in range(4) for b in range(a + 1, 4)]
    return float(max(e) / tet_volume(v) ** (1.0 / 3.0))


# ==================================================================
# Geometry generators
# ==================================================================

def random_rotation(rng):
    Qm, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Qm * np.sign(np.diag(R))


REG_TET = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)


def unit_volume_tet(verts):
    v = np.asarray(verts, float)
    v = v - v.mean(axis=0)
    return v / tet_volume(v) ** (1.0 / 3.0)


def gmsh_tet_mesh(target=0.18, verbose=False):
    """A production-representative tet mesh: gmsh defaults on a box, i.e. the
    same mesher and settings the cyclotron model's geometry layer uses.

    Returns tets scaled so the MEAN element volume is 1 (so h = 1 and d/h is
    just the centroid distance, matching the cost model's pair counting).
    """
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.Verbosity", 5 if verbose else 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", target)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target)
        gmsh.model.mesh.generate(3)
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        coord = {int(t): ncoord[3 * i:3 * i + 3] for i, t in enumerate(ntags)}
        et, etags, enodes = gmsh.model.mesh.getElements(3)
        tets = []
        for typ, nds in zip(et, enodes):
            if typ != 4:                       # 4 = 4-node tetrahedron
                continue
            nds = np.array(nds, dtype=np.int64).reshape(-1, 4)
            for row in nds:
                tets.append([coord[int(t)] for t in row])
        tets = np.array(tets, float)
    finally:
        gmsh.finalize()
    vols = np.array([tet_volume(t) for t in tets])
    return tets / vols.mean() ** (1.0 / 3.0)


def scipy_tet_mesh(n=8, seed=3):
    """Fallback (and a deliberately WORSE-quality mesh for contrast)."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, n)
    P = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    inner = (P > 1e-9).all(axis=1) & (P < 1 - 1e-9).all(axis=1)
    P[inner] += rng.uniform(-0.3, 0.3, size=(inner.sum(), 3)) / (n - 1)
    tets = P[Delaunay(P).simplices]
    vols = np.array([tet_volume(t) for t in tets])
    tets = tets[vols > 1e-3 * vols.mean()]
    vols = np.array([tet_volume(t) for t in tets])
    return tets / vols.mean() ** (1.0 / 3.0)


def mesh_interior(tets, frac=0.30):
    """Indices of elements away from the mesh boundary (works for any scaling)."""
    cens = np.array([tet_centroid(t) for t in tets])
    lo, hi = cens.min(axis=0), cens.max(axis=0)
    pad = frac * (hi - lo)
    sel = ((cens > lo + pad).all(axis=1) & (cens < hi - pad).all(axis=1))
    return np.where(sel)[0]


# ==================================================================
# Statistics helpers
# ==================================================================

def stats(v):
    v = np.asarray(v, float)
    return {"med": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max()), "mean": float(v.mean())}


def frob(a):
    return float(np.linalg.norm(a))


# ==================================================================
# Part A -- reference validation
# ==================================================================

def check_reciprocity(tets, n_pairs=6, seed=17):
    """Galerkin reciprocity: V_i Qbar_ij == V_j Qbar_ji exactly (the kernel
    G(r,r') is symmetric), while collocation is NOT symmetric.  This is an
    independent check that the reference integrator is right -- the two sides
    are computed from completely different quadratures."""
    rng = np.random.default_rng(seed)
    inner = mesh_interior(tets)
    cens = np.array([tet_centroid(t) for t in tets])
    rows = []
    for _ in range(n_pairs):
        i = int(rng.choice(inner))
        d = np.linalg.norm(cens - cens[i], axis=1)
        cand = np.where((d > 1.2) & (d < 6.0))[0]
        if not len(cand):
            continue
        j = int(rng.choice(cand))
        Vi, Vj = tet_volume(tets[i]), tet_volume(tets[j])
        rad.UtiDelAll()
        sj = SourceQ("tet", tets[j])
        Qij, _, _, _ = tet_reference(sj, tets[i], rel_target=1e-8)
        Q1ij = tet_entry(sj, tets[i], "K1")
        rad.UtiDelAll()
        si = SourceQ("tet", tets[i])
        Qji, _, _, _ = tet_reference(si, tets[j], rel_target=1e-8)
        Q1ji = tet_entry(si, tets[j], "K1")
        scale = 0.5 * (frob(Vi * Qij) + frob(Vj * Qji))
        rows.append({
            "i": i, "j": j, "d": float(d[j]),
            "galerkin_asym": frob(Vi * Qij - (Vj * Qji).T) / scale,
            "collocation_asym": frob(Vi * Q1ij - (Vj * Q1ji).T) / scale,
        })
    return rows


# ==================================================================
# Part B -- controlled separation sweep, several observation shapes
# ==================================================================

def make_shapes(tets):
    """Representative observation-element shapes, unit volume."""
    dr = np.array([dev_ratio(t) for t in tets])
    order = np.argsort(dr)
    shapes = {"regular": (unit_volume_tet(REG_TET), dev_ratio(REG_TET))}
    for lbl, q in (("mesh_p25", 0.25), ("mesh_med", 0.50),
                   ("mesh_p90", 0.90), ("mesh_worst", 1.0)):
        idx = order[min(int(q * (len(order) - 1)), len(order) - 1)]
        v = unit_volume_tet(tets[idx])
        shapes[lbl] = (v, dev_ratio(v))
    return shapes


def study_separation_sweep(shapes, ratios, n_orient=4, seed=99, verbose=True):
    rng = np.random.default_rng(seed)
    src_shape = unit_volume_tet(REG_TET)
    rows = []
    rad.UtiDelAll()
    src = SourceQ("tet", src_shape)
    for sname, (shape, dr) in shapes.items():
        for io in range(n_orient):
            Ro = random_rotation(rng)
            obs_shape = shape @ Ro.T
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            for d in ratios:
                obs = obs_shape + u * d
                Qref, eabs, erel, npts = tet_reference(src, obs)
                row = {"shape": sname, "dev": dr, "orient": io, "dh": float(d),
                       "ref_rel": erel, "ref_pts": npts, "Qref": frob(Qref)}
                for name in TET_RULE_NAMES:
                    row[name] = frob(tet_entry(src, obs, name) - Qref)
                rows.append(row)
        if verbose:
            sub = [r for r in rows if r["shape"] == sname]
            print(f"    {sname:11s} dev={dr:6.4f}  "
                  f"K1 rel @ d/h=3: "
                  f"{np.max([r['K1'] / r['Qref'] for r in sub if abs(r['dh'] - 3) < .01]):.2e}"
                  f"   @ d/h=10: "
                  f"{np.max([r['K1'] / r['Qref'] for r in sub if abs(r['dh'] - 10) < .01]):.2e}",
                  flush=True)
    return rows


# ==================================================================
# Part C -- realistic mesh pairs, bin-balanced
# ==================================================================

def study_mesh_pairs(tets, n_src=10, per_bin=6, seed=11, verbose=True):
    rng = np.random.default_rng(seed)
    cens = np.array([tet_centroid(t) for t in tets])
    inner = mesh_interior(tets)
    srcs = rng.choice(inner, size=min(n_src, len(inner)), replace=False)
    rows = []
    for si in srcs:
        rad.UtiDelAll()
        src = SourceQ("tet", tets[si])
        d = np.linalg.norm(cens - cens[si], axis=1)
        picks = [si]                                  # self block
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            cand = np.where((d >= max(lo, 1e-9)) & (d < hi))[0]
            if not len(cand):
                continue
            k = min(per_bin, len(cand))
            picks += rng.choice(cand, size=k, replace=False).tolist()
        for oi in picks:
            Qref, eabs, erel, npts = tet_reference(src, tets[oi])
            row = {"src": int(si), "obs": int(oi), "dh": float(d[oi]),
                   "shared": count_shared_vertices(tets[si], tets[oi]),
                   "dev": dev_ratio(tets[oi]), "ref_rel": erel,
                   "ref_pts": npts, "Qref": frob(Qref)}
            for name in TET_RULE_NAMES:
                row[name] = frob(tet_entry(src, tets[oi], name) - Qref)
            for lbl, rule, lev, _ in COMPOSITES:
                row[lbl] = frob(tet_entry_composite(src, tets[oi], rule, lev) - Qref)
            rows.append(row)
        if verbose:
            print(f"    source {si:5d}: {len(picks):3d} pairs", flush=True)
    return rows


def count_shared_vertices(a, b, tol=1e-9):
    return int(sum(np.min(np.linalg.norm(b - va, axis=1)) < tol for va in a))


# ==================================================================
# Part C2 -- the DECISIVE cutoff measurement: signed total correction
# ==================================================================

def container_field(cnt, chunk=20000):
    """H of a Radia container at many points, in bounded-size batches."""
    def fn(pts):
        pts = np.asarray(pts, float)
        out = np.empty((len(pts), 3))
        for a in range(0, len(pts), chunk):
            b = min(a + chunk, len(pts))
            out[a:b] = np.array(rad.Fld(cnt, 'h', pts[a:b].tolist()),
                                float).reshape(-1, 3)
        return out
    return fn


def study_total_correction(tets, radii, modes=("uniform", "gradient", "random"),
                           n_obs=3, seed=23, verbose=True):
    """(b) How far out does the quadrature actually have to reach?

    For any SET S of source elements carrying magnetizations M_j,

        Sum_{j in S} Qbar_ij M_j = (1/V_i) Integral_{V_i} H_S dV
        Sum_{j in S} Q_ij(c_i) M_j = H_S(c_i)

    because fields simply add: H_S is the field of the UNION of S.  So the
    exact SIGNED Galerkin correction contributed by a shell of sources needs no
    per-pair quadrature at all -- one Radia container per shell.

    This is the measurement that decides the cutoff, because the per-pair
    errors mostly CANCEL.  Collocation is exact for uniform M (Sum_j Q_j(r) =
    Q_body(r) identically, so H(c_i) is the true field there), which means a
    sum of |dQ_ij| magnitudes grossly overstates the real correction: the large
    self-block correction is cancelled by the neighbours', and only the
    residual survives.

    Because everything is linear in the sources, the same trick prices any
    QUADRATURE RULE applied to a set S: sum_q w_q H_S(x_q) - H_S(c_i) is the
    correction that rule produces for all of S at once.  Applying a rule inside
    R and plain K=1 outside therefore gives exactly dH_rule(S_R), since the K=1
    part contributes no correction at all.  So one pass over disjoint shells
    yields the whole (cutoff R) x (rule) error table.

    M modes: 'uniform' (the physical, strongly-cancelling case), 'gradient'
    (smoothly varying), 'random' (per-element random directions: the
    pathological no-cancellation case).
    """
    rng = np.random.default_rng(seed)
    cens = np.array([tet_centroid(t) for t in tets])
    vols = np.array([tet_volume(t) for t in tets])
    mid = 0.5 * (cens.min(axis=0) + cens.max(axis=0))
    obs_idx = np.argsort(np.linalg.norm(cens - mid, axis=1))[:n_obs]
    side = float(np.mean(cens.max(axis=0) - cens.min(axis=0)))
    if verbose:
        print(f"    mesh box ~{side:.1f} h on a side -> complete shells to "
              f"R ~ {side / 2:.1f} h")

    out = {}
    for mode in modes:
        if mode == "uniform":
            M = np.tile([0.0, 0.0, 1.0], (len(tets), 1))
        elif mode == "gradient":
            z = (cens[:, 2] - cens[:, 2].min()) / np.ptp(cens[:, 2])
            M = np.stack([0.3 * z, np.zeros_like(z), 1.0 - 0.6 * z], axis=1)
        elif mode.startswith("wave"):
            # M varying on a length scale of a few elements: the realistic
            # "sharp feature" case (saturation front, pole edge), which is what
            # decides whether a smooth-M argument is good enough.
            lam = float(mode[4:]) if len(mode) > 4 else 5.0
            k = 2.0 * np.pi / lam
            ph = k * (cens[:, 0] + 0.7 * cens[:, 1] + 0.4 * cens[:, 2])
            M = np.stack([0.2 * np.sin(ph), np.zeros(len(tets)),
                          1.0 + 0.5 * np.cos(ph)], axis=1)
        elif mode == "random":
            M = rng.normal(size=(len(tets), 3))
            M /= np.linalg.norm(M, axis=1, keepdims=True)
        else:
            raise ValueError(mode)

        rows = []
        for oi in obs_idx:
            d = np.linalg.norm(cens - cens[oi], axis=1)
            # The SELF element gets a shell of its own (lo = hi = 0): its
            # integrand has a boundary layer at the element's own faces and
            # needs a far bigger point budget than the neighbours, and lumping
            # it into the first shell made the reference the accuracy limit.
            bnds = [(0.0, 0.0)]
            edges = [0.0] + list(radii) + [1e18]
            bnds += list(zip(edges[:-1], edges[1:]))
            shells, Hc_tot, dH_tot = [], np.zeros(3), np.zeros(3)
            rad.UtiDelAll()
            for lo, hi in bnds:
                if (lo == 0.0) and (hi == 0.0):
                    sel = np.array([oi])
                else:
                    sel = np.where((d >= max(lo, 1e-12)) & (d < hi))[0]
                    sel = sel[sel != oi]
                if not len(sel):
                    shells.append({"lo": lo, "hi": hi, "n": 0,
                                   "dH": [0.0] * 3, "Hc": [0.0] * 3, "err": 0.0})
                    continue
                ids = [rad.ObjPolyhdr(tets[k].tolist(), TET_FACES, M[k].tolist())
                       for k in sel]
                cnt = rad.ObjCnt(ids)
                fn = container_field(cnt)

                Hc = fn(cens[oi][None, :])[0]
                # Field evaluation costs n_elem * n_pts, so spend the points
                # where they are cheap: the self/near shells hold a handful of
                # elements and need adaptive refinement (the integrand has a
                # boundary layer), the far shells hold thousands but their
                # field is smooth over the observation element.
                budget = int(max(2000, 600000 / len(sel)))
                tol = 1e-9 * max(np.linalg.norm(Hc), 1e-12)
                Hbar, err, npts = adaptive_tet_average(fn, tets[oi], tol,
                                                       max_pts=budget)
                # What each candidate rule would produce for THIS whole shell.
                rule_dH = {}
                for name in TET_RULE_NAMES:
                    pts, w = tet_rule_points(tets[oi], name)
                    rule_dH[name] = (w @ fn(pts) - Hc).tolist()
                for lbl, rule, lev, _ in COMPOSITES:
                    cells = [tets[oi]]
                    for _ in range(lev):
                        cells = [s for c in cells for s in tet_subdivide(c)]
                    cv = np.array([tet_volume(c) for c in cells])
                    bary, w, _ = TET_RULES[rule]
                    p = np.vstack([bary @ c for c in cells])
                    per = np.einsum("q,cqi->ci", w,
                                    fn(p).reshape(len(cells), len(w), 3))
                    rule_dH[lbl] = ((cv / cv.sum()) @ per - Hc).tolist()
                shells.append({"lo": lo, "hi": hi, "n": int(len(sel)),
                               "dH": (Hbar - Hc).tolist(), "Hc": Hc.tolist(),
                               "err": float(err), "pts": npts, "rule_dH": rule_dH})
                Hc_tot += Hc
                dH_tot += Hbar - Hc
                rad.UtiDelAll()
            # Independent cross-check for uniform M: the union of the tets IS
            # the meshed box, so a single RecMag with the same magnetization
            # must give the same total -- validating the whole shell
            # decomposition against one closed-form object.
            xchk = None
            if mode == "uniform":
                lo3 = np.min(tets.reshape(-1, 3), axis=0)
                hi3 = np.max(tets.reshape(-1, 3), axis=0)
                rad.UtiDelAll()
                box = rad.ObjRecMag((0.5 * (lo3 + hi3)).tolist(),
                                    (hi3 - lo3).tolist(), [0.0, 0.0, 1.0])
                fnb = container_field(box)
                Hcb = fnb(cens[oi][None, :])[0]
                Hbb, errb, _ = adaptive_tet_average(
                    fnb, tets[oi], 1e-8 * max(np.linalg.norm(Hcb), 1e-9),
                    max_pts=200000)
                xchk = {"Hc": Hcb.tolist(), "dH": (Hbb - Hcb).tolist(),
                        "err": float(errb),
                        "dHc_dev": float(np.linalg.norm(Hbb - Hcb - dH_tot)),
                        "Hc_dev": float(np.linalg.norm(Hcb - Hc_tot))}
                rad.UtiDelAll()
            rows.append({"obs": int(oi), "vol": float(vols[oi]),
                         "dev": dev_ratio(tets[oi]),
                         "shells": shells, "Hc_tot": Hc_tot.tolist(),
                         "dH_tot": dH_tot.tolist(), "xcheck": xchk})
            if verbose:
                msg = (f"    M={mode:9s} obs {oi:5d}: |H|={np.linalg.norm(Hc_tot):.4f} "
                       f"|dH_total|={np.linalg.norm(dH_tot):.3e} "
                       f"({np.linalg.norm(dH_tot) / np.linalg.norm(Hc_tot) * 100:.3f}% of H)")
                if xchk:
                    msg += (f"   [whole-box check: H dev {xchk['Hc_dev']:.1e}, "
                            f"dH dev {xchk['dHc_dev']:.1e}]")
                print(msg, flush=True)
        out[mode] = rows
    return out


def report_total_correction(res, verbose=True):
    """Cumulative |dH(R)| / |dH(total)| -- the fraction of the true Galerkin
    correction captured by quadrature out to R."""
    summary = {}
    for mode, rows in res.items():
        bnds = [(s["lo"], s["hi"]) for s in rows[0]["shells"]]
        print(f"\n  M = {mode}")
        # How well the shell decomposition itself resolves dH_total: the sum of
        # the per-shell reference error estimates, as a % of |dH_total|. Every
        # percentage below is meaningless at or under this floor.
        for r in rows:
            err = sum(s["err"] for s in r["shells"])
            tot = np.linalg.norm(r["dH_tot"])
            note = ""
            if r.get("xcheck"):
                note = (f", whole-box cross-check disagrees by "
                        f"{r['xcheck']['dHc_dev'] / max(tot, 1e-300) * 100:.1f}%")
            print(f"    obs {r['obs']}: reference floor "
                  f"{err / max(tot, 1e-300) * 100:.2f}% of |dH_total|{note}")
        print(f"  {'shell d/h':>13s} {'n':>6s} " +
              "".join(f" {'obs ' + str(r['obs']):>26s}" for r in rows))
        print(f"  {'':>13s} {'':>6s} " +
              "".join(f" {'|dH_shell|   cum/|dH_tot|':>26s}" for r in rows))
        print("  " + "-" * (21 + 27 * len(rows)))
        cum = [np.zeros(3) for _ in rows]
        recs = []
        for k, (lo, hi) in enumerate(bnds):
            line = (f"  {lo:5.1f}-{hi if hi < 1e17 else 999:5.1f} "
                    f"{rows[0]['shells'][k]['n']:6d} ")
            rec = {"lo": lo, "hi": hi if hi < 1e17 else None}
            for m, r in enumerate(rows):
                s = r["shells"][k]
                cum[m] = cum[m] + np.array(s["dH"])
                tot = np.linalg.norm(r["dH_tot"])
                frac = np.linalg.norm(cum[m]) / max(tot, 1e-300)
                line += f" {np.linalg.norm(s['dH']):12.3e} {frac * 100:11.1f}%"
                rec[f"obs{r['obs']}_shell"] = float(np.linalg.norm(s["dH"]))
                rec[f"obs{r['obs']}_cumfrac"] = float(frac)
            print(line)
            recs.append(rec)
        print("  " + "-" * (21 + 27 * len(rows)))
        line = f"  {'TOTAL':>13s} {'':>6s} "
        for r in rows:
            line += (f" {np.linalg.norm(r['dH_tot']):12.3e} "
                     f"{'(' + format(np.linalg.norm(r['dH_tot']) / np.linalg.norm(r['Hc_tot']) * 100, '.3f') + '% of H)':>11s}")
        print(line)
        summary[mode] = {"shells": recs,
                         "rule_cutoff": _rule_cutoff_table(rows, bnds),
                         "graded": _graded_table(rows, bnds)}
    return summary


# Graded schemes: (label, base rule applied to EVERY pair, near-band rule,
# near-band radius in d/h).  The base rule sets the N^2 cost; the near-band
# rule is applied to the O(N) pairs inside the radius, which is nearly free.
GRADED = [
    ("K4 everywhere",              "K4", None,    0.0),
    ("K4 + K24 inside 1.5",        "K4", "K24",   1.5),
    ("K4 + K14x8 inside 1.5",      "K4", "K14x8", 1.5),
    ("K4 + K4x64 inside 1.5",      "K4", "K4x64", 1.5),
    ("K4 + K14x8 inside 2.5",      "K4", "K14x8", 2.5),
    ("K14 everywhere",             "K14", None,   0.0),
    ("K14 + K14x8 inside 1.5",     "K14", "K14x8", 1.5),
]


def _graded_table(rows, bnds):
    """(c) The practical schemes: one cheap rule on every pair (the N^2 cost),
    optionally upgraded on the O(N) near-pair list."""
    print(f"\n    graded schemes -- error vs the EXACT Galerkin correction,")
    print(f"    in % of that correction (K=1 everywhere would be 100%):")
    print(f"      {'scheme':>24s} {'pts/pair':>9s} {'error':>8s}")
    print("      " + "-" * 43)
    out = []
    for label, base, near, R in GRADED:
        errs = []
        for r in rows:
            tot = np.array(r["dH_tot"])
            acc = np.zeros(3)
            for s in r["shells"]:
                if not s["n"]:
                    continue
                k = near if (near and s["hi"] <= R + 1e-9) else base
                acc = acc + np.array(s["rule_dH"][k])
            errs.append(np.linalg.norm(acc - tot) / max(np.linalg.norm(tot), 1e-300))
        v = float(np.mean(errs))
        npts = len(TET_RULES[base][1]) if base in TET_RULES else 0
        print(f"      {label:>24s} {npts:9d} {v * 100:7.2f}%")
        out.append({"scheme": label, "base": base, "near": near, "R": R,
                    "error": v})
    return out


def _rule_cutoff_table(rows, bnds):
    """(b)+(c) together: error of 'rule inside R, K=1 outside' against the exact
    Galerkin correction, as a percentage of that correction.

    K=1 everywhere is 100% by construction (it produces no correction at all),
    so a number below 100% means the scheme captures part of the effect and a
    number ABOVE 100% means it overshoots -- which is what truncating a
    cancelling sum does."""
    keys = TET_RULE_NAMES + COMPOSITE_NAMES
    Rs = [hi for _, hi in bnds]
    print(f"\n    error of 'rule inside R, K=1 outside' vs the EXACT Galerkin")
    print(f"    correction, in % of that correction (100% = today's scheme):")
    print(f"      {'R (d/h)':>9s}" + "".join(f" {k:>8s}" for k in keys))
    print("      " + "-" * (9 + 9 * len(keys)))
    out = []
    for ri, R in enumerate(Rs):
        line = f"      {R if R < 1e17 else 999:9.1f}"
        rec = {"R": R if R < 1e17 else None}
        for k in keys:
            errs = []
            for r in rows:
                tot = np.array(r["dH_tot"])
                acc = np.zeros(3)
                for s in r["shells"][:ri + 1]:
                    if s["n"]:
                        acc = acc + np.array(s["rule_dH"][k])
                errs.append(np.linalg.norm(acc - tot) / max(np.linalg.norm(tot), 1e-300))
            v = float(np.mean(errs))
            line += f" {v * 100:7.1f}%"
            rec[k] = v
        print(line)
        out.append(rec)
    return out


# ==================================================================
# Part D -- self blocks
# ==================================================================

def study_self_blocks(tets, n=25, seed=5, verbose=True):
    rng = np.random.default_rng(seed)
    inner = mesh_interior(tets, frac=0.2)
    pick = rng.choice(inner, size=min(n, len(inner)), replace=False)
    rows = []
    for i in pick:
        rad.UtiDelAll()
        src = SourceQ("tet", tets[i])
        Q1 = tet_entry(src, tets[i], "K1")
        Qref, eabs, erel, npts = tet_reference(src, tets[i], rel_target=3e-5,
                                               max_pts=400000)
        row = {"elem": int(i), "aspect": aspect(tets[i]), "dev": dev_ratio(tets[i]),
               "trace1": float(np.trace(Q1)), "traceref": float(np.trace(Qref)),
               "ref_rel": erel, "ref_pts": npts,
               "Q1": frob(Q1), "Qref": frob(Qref), "d_frob": frob(Qref - Q1)}
        for name in TET_RULE_NAMES[1:]:
            row[name] = frob(tet_entry(src, tets[i], name) - Qref)
        for lbl, rule, lev, _ in COMPOSITES:
            row[lbl] = frob(tet_entry_composite(src, tets[i], rule, lev) - Qref)
        rows.append(row)
        if verbose:
            print(f"    elem {i:5d} aspect {row['aspect']:5.2f} dev {row['dev']:6.4f} "
                  f"|dQ|/|Q| {row['d_frob'] / row['Q1'] * 100:6.2f}%  "
                  f"tr(K1)={row['trace1']:+.6f} tr(ref)={row['traceref']:+.6f} "
                  f"(ref {erel:.1e})", flush=True)

    rad.UtiDelAll()
    src = SourceQ("rec", ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    Q1 = hex_entry(src, [0, 0, 0], [1, 1, 1], 1)
    Qref, _, erel, _ = hex_reference(src, [0, 0, 0], [1, 1, 1], rel_target=3e-5,
                                     max_pts=400000)
    cube = {"d_frob": frob(Qref - Q1), "Q1": frob(Q1),
            "trace1": float(np.trace(Q1)), "traceref": float(np.trace(Qref)),
            "ref_rel": erel}
    # a non-cubic RecMag: does the structured lattice's aspect ratio matter?
    slabs = {}
    for lbl, dims in (("2:1:1", [1.5874, 0.7937, 0.7937]),
                      ("4:1:1", [2.5198, 0.62996, 0.62996])):
        rad.UtiDelAll()
        s = SourceQ("rec", ([0.0, 0.0, 0.0], dims))
        q1 = hex_entry(s, [0, 0, 0], dims, 1)
        qr, _, er, _ = hex_reference(s, [0, 0, 0], dims, rel_target=3e-5,
                                     max_pts=400000)
        slabs[lbl] = {"d_frob": frob(qr - q1), "Q1": frob(q1), "ref_rel": er}
    return rows, cube, slabs


# ==================================================================
# Part E -- RecMag separation sweep
# ==================================================================

def study_recmag(ratios, verbose=True):
    dims = np.array([1.0, 1.0, 1.0])
    rad.UtiDelAll()
    src = SourceQ("rec", ([0.0, 0.0, 0.0], dims))
    dirs = {"axis": np.array([1.0, 0, 0]),
            "face_diag": np.array([1.0, 1, 0]) / np.sqrt(2),
            "body_diag": np.array([1.0, 1, 1]) / np.sqrt(3)}
    rows = []
    for dname, dv in dirs.items():
        for d in ratios:
            cen = dv * d
            Qref, _, erel, npts = hex_reference(src, cen, dims)
            row = {"dir": dname, "dh": float(d), "ref_rel": erel,
                   "ref_pts": npts, "Qref": frob(Qref)}
            for K in (1, 2, 3, 4):
                row[f"K{K ** 3}"] = frob(hex_entry(src, cen, dims, K) - Qref)
            rows.append(row)
        if verbose:
            print(f"    {dname} done", flush=True)
    return rows


# ==================================================================
# Reporting
# ==================================================================

def bin_table(rows, keys, label, stat="max", bins=None, dhkey="dh"):
    bins = bins or BINS
    dh = np.array([r[dhkey] for r in rows])
    print(f"\n{label}   [{stat} over each bin]")
    hdr = f"  {'d/h bin':>12s} {'n':>4s} {'|Qref| med':>11s} {'ref err':>9s}"
    hdr += "".join(f" {k:>10s}" for k in keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (dh >= lo) & (dh < hi)
        if not sel.any():
            continue
        sub = [r for r, s in zip(rows, sel) if s]
        qref = np.array([r["Qref"] for r in sub])
        name = f"{lo:5.1f}-{hi:5.1f}"
        line = (f"  {name:>12s} {len(sub):4d} {np.median(qref):11.3e} "
                f"{np.max([r['ref_rel'] for r in sub]):9.1e}")
        rec = {"lo": lo, "hi": hi, "n": len(sub), "Qref_med": float(np.median(qref))}
        for k in keys:
            rel = np.array([r[k] for r in sub]) / qref
            val = {"max": rel.max(), "med": np.median(rel),
                   "p90": np.percentile(rel, 90)}[stat]
            line += f" {val:10.2e}"
            rec[k] = float(val)
        print(line)
        out.append(rec)
    return out


def cutoff_table(rows, keys, label, bins=None):
    """(b) Where can the quadrature stop?

    Shell weighting: in a mesh of unit mean element volume there are
    4/3 pi (hi^3 - lo^3) source elements in the shell [lo, hi).  The
    'coherent' sum n_shell * <|dQ|> is the worst case in which every source's
    entry error pushes H the same way -- which is exactly what the leading
    Mdev:grad^2 Q term does, since it factors out of the sum over sources.
    """
    bins = bins or BINS
    dh = np.array([r["dh"] for r in rows])
    shells = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (dh >= lo) & (dh < hi)
        if not sel.any():
            continue
        sub = [r for r, s in zip(rows, sel) if s]
        n_shell = 4.0 / 3.0 * np.pi * (hi ** 3 - lo ** 3)
        shells.append((lo, hi, n_shell,
                       {k: n_shell * float(np.mean([r[k] for r in sub])) for k in keys}))

    print(f"\n{label}")
    print(f"  {'d/h bin':>12s} {'n_shell':>9s}" + "".join(f" {k:>12s}" for k in keys))
    print("  " + "-" * (23 + 13 * len(keys)))
    for lo, hi, n, c in shells:
        print(f"  {lo:5.1f}-{hi:5.1f} {n:9.1f}" +
              "".join(f" {c[k]:12.3e}" for k in keys))
    tot = {k: sum(c[k] for _, _, _, c in shells) for k in keys}
    print("  " + "-" * (23 + 13 * len(keys)))
    print(f"  {'TOTAL':>12s} {'':>9s}" + "".join(f" {tot[k]:12.3e}" for k in keys))

    # What fraction of the FULL correction (= the K1 total) is left behind if
    # quadrature is applied only inside R and plain K1 outside?
    print(f"\n  residual error after 'quadrature inside R, K=1 outside',")
    print(f"  as a fraction of the total Galerkin correction "
          f"({tot['K1']:.3e}):")
    print(f"  {'R (d/h)':>9s}" + "".join(f" {k:>12s}" for k in keys[1:]))
    print("  " + "-" * (10 + 13 * (len(keys) - 1)))
    residual = {}
    for R in [b for b in bins[1:-1]]:
        line = f"  {R:9.1f}"
        residual[R] = {}
        for k in keys[1:]:
            inside = sum(c[k] for lo, hi, _, c in shells if hi <= R)
            outside = sum(c["K1"] for lo, hi, _, c in shells if hi > R)
            frac = (inside + outside) / tot["K1"]
            line += f" {frac * 100:11.2f}%"
            residual[R][k] = float(frac)
        print(line)
    return {"shells": [{"lo": lo, "hi": hi, "n": n, **c} for lo, hi, n, c in shells],
            "total": tot, "residual": residual}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mesh", choices=["gmsh", "scipy"], default="gmsh")
    ap.add_argument("--size", type=float, default=0.16,
                    help="gmsh target size for the pair mesh")
    ap.add_argument("--big-size", type=float, default=0.085,
                    help="gmsh target size for the total-correction mesh "
                         "(needs to be wide enough for complete shells)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rad.FldLenRndSw('on')       # keep Radia's on-plane jitter repair
    t0 = time.time()
    R = {}

    ratios = ([1.2, 2.0, 3.0, 5.0, 10.0, 20.0] if args.quick else
              [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 22.0, 32.0])

    print("=" * 84)
    print("STEP 1: pair-level cost/accuracy of Galerkin (volume-averaged) assembly")
    print("=" * 84)
    print("h = mean element V^(1/3) (= mean spacing, the length the cost model counts")
    print("with); d = centroid distance.  Errors are |Q_K - Qbar_exact| / |Qbar_exact|.")

    if args.mesh == "gmsh":
        tets = gmsh_tet_mesh(target=args.size)
    else:
        tets = scipy_tet_mesh()
    dr = np.array([dev_ratio(t) for t in tets])
    asp = np.array([aspect(t) for t in tets])
    print(f"\nmesh: {args.mesh}, {len(tets)} tets (mean volume normalized to 1)")
    print(f"  deviatoric second moment ||Mdev||/V^(2/3): "
          f"med {np.median(dr):.4f}  p90 {np.percentile(dr, 90):.4f}  "
          f"max {dr.max():.4f}   (regular tet: {dev_ratio(REG_TET):.4f}, cube: 0)")
    print(f"  aspect (longest edge / V^(1/3)):           "
          f"med {np.median(asp):.2f}  p90 {np.percentile(asp, 90):.2f}  "
          f"max {asp.max():.2f}     (regular tet: {aspect(REG_TET):.2f})")
    R["mesh_info"] = {"kind": args.mesh, "n": len(tets),
                      "dev": stats(dr), "aspect": stats(asp)}

    print("\n[1/6] reference validation: Galerkin reciprocity V_i Qbar_ij = V_j Qbar_ji")
    rec = check_reciprocity(tets, n_pairs=3 if args.quick else 6)
    R["reciprocity"] = rec
    if rec:
        g = np.array([r["galerkin_asym"] for r in rec])
        c = np.array([r["collocation_asym"] for r in rec])
        print(f"    Galerkin    asymmetry: max {g.max():.2e}  (must be ~ reference tol)")
        print(f"    collocation asymmetry: max {c.max():.2e}  med {np.median(c):.2e}"
              f"   <- what today's matrix has")

    print("\n[2/6] controlled sweep: regular-tet source, observation shapes from the mesh")
    shapes = make_shapes(tets)
    srows = study_separation_sweep(shapes, ratios,
                                   n_orient=2 if args.quick else 4)
    R["sweep"] = srows
    for sname in shapes:
        sub = [r for r in srows if r["shape"] == sname]
        R[f"sweep_{sname}"] = bin_table(
            sub, TET_RULE_NAMES,
            f"  observation shape '{sname}' (||Mdev||/V^(2/3) = {shapes[sname][1]:.4f})",
            stat="max")

    print("\n[3/6] realistic mesh pairs (bin-balanced sampling)")
    allkeys = TET_RULE_NAMES + COMPOSITE_NAMES
    mrows = study_mesh_pairs(tets, n_src=4 if args.quick else 10,
                             per_bin=3 if args.quick else 6)
    R["mesh_pairs"] = mrows
    nonself = [r for r in mrows if r["dh"] > 1e-9]
    R["mesh_bins_med"] = bin_table(nonself, allkeys,
                                   "  mesh pairs, MEDIAN relative entry error", "med")
    R["mesh_bins_max"] = bin_table(nonself, allkeys,
                                   "  mesh pairs, WORST relative entry error", "max")

    print("\n  adjacency breakdown (shared vertices; 4 = self, 3 = shared face)")
    print(f"  {'shared':>7s} {'n':>4s} {'d/h med':>8s} {'|Qref| med':>11s}" +
          "".join(f" {k:>10s}" for k in allkeys))
    adj = []
    for s in (4, 3, 2, 1, 0):
        sub = [r for r in mrows if r["shared"] == s]
        if not sub:
            continue
        qref = np.array([r["Qref"] for r in sub])
        line = (f"  {s:7d} {len(sub):4d} {np.median([r['dh'] for r in sub]):8.2f} "
                f"{np.median(qref):11.3e}")
        rec_ = {"shared": s, "n": len(sub)}
        for k in allkeys:
            v = float(np.max(np.array([r[k] for r in sub]) / qref))
            line += f" {v:10.2e}"
            rec_[k] = v
        print(line)
        adj.append(rec_)
    R["adjacency"] = adj

    print("\n[4/6] no-cancellation UPPER BOUND on the truncation error")
    R["cutoff_bound"] = cutoff_table(nonself, TET_RULE_NAMES,
        "  shell-integrated |dQ| assuming every source's error adds coherently\n"
        "  (a strict upper bound -- see [5/6] for the real, signed answer)")

    print("\n[5/6] SIGNED total Galerkin correction and its convergence with R")
    print("      (the decisive cutoff measurement: per-pair errors largely cancel)")
    big = gmsh_tet_mesh(target=args.big_size) if not args.quick else tets
    print(f"    mesh for this part: {len(big)} tets")
    radii = [1.5, 2.5, 4.0, 6.0, 9.0, 13.0]
    tot = study_total_correction(big, radii,
                                 modes=("uniform", "wave5", "random") if args.quick
                                 else ("uniform", "gradient", "wave8", "wave4",
                                       "random"),
                                 n_obs=2 if args.quick else 3)
    R["total_correction"] = tot
    R["total_correction_summary"] = report_total_correction(tot)

    print("\n[6/6] SELF blocks and the structured (RecMag) case")
    selfrows, cube, slabs = study_self_blocks(tets, n=6 if args.quick else 25)
    R["self"] = selfrows
    R["self_cube"] = cube
    R["self_slabs"] = slabs
    d = np.array([r["d_frob"] for r in selfrows])
    q1 = np.array([r["Q1"] for r in selfrows])
    print(f"\n  tet self-block |Qbar - Q_centroid| / |Q_centroid|:  "
          f"med {np.median(d / q1) * 100:.2f}%  p90 {np.percentile(d / q1, 90) * 100:.2f}%  "
          f"max {np.max(d / q1) * 100:.2f}%")
    print(f"  trace must be -1: centroid max dev "
          f"{np.abs(np.array([r['trace1'] for r in selfrows]) + 1).max():.2e}, "
          f"volume-averaged max dev "
          f"{np.abs(np.array([r['traceref'] for r in selfrows]) + 1).max():.2e}")
    print(f"  self-block residual after quadrature (relative to |Q_centroid|):")
    for k in TET_RULE_NAMES[1:] + COMPOSITE_NAMES:
        v = np.array([r[k] for r in selfrows]) / q1
        print(f"    {k:>6s}: med {np.median(v) * 100:7.3f}%  max {v.max() * 100:7.3f}%")
    print(f"  cube RecMag self-block: |dQ|/|Q| = {cube['d_frob'] / cube['Q1'] * 100:.4f}% "
          f"(ref {cube['ref_rel']:.1e})  <- exactly zero by cubic symmetry")
    for lbl, s in slabs.items():
        print(f"  RecMag {lbl} self-block: |dQ|/|Q| = "
              f"{s['d_frob'] / s['Q1'] * 100:.4f}% (ref {s['ref_rel']:.1e})")

    print("\n  RecMag (structured lattice) separation sweep")
    rrows = study_recmag(ratios)
    R["recmag"] = rrows
    R["recmag_bins"] = bin_table(rrows, ["K1", "K8", "K27", "K64"],
        "  cube source, cube observation (K = number of GL points)", "max")

    with open(os.path.join(OUT_DIR, "galerkin_pair_study.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(OUT_DIR, 'galerkin_pair_study.json')}")
    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
