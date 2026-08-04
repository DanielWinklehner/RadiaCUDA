# studies/galerkin_quad.py
#
# Quadrature machinery shared by the Galerkin pair study (STEP 1) and used as
# the reference for the kernel rules implemented later in radgpu_asm.cu /
# radintrc.cpp.
#
# Everything here is about ONE question: given a source element and an
# OBSERVATION element, what is
#
#     Qbar = (1/V_obs) * Integral_{V_obs} Q(r) dV
#
# where Q(r) is Radia's 3x3 "field per unit magnetization" tensor of the source
# (column j = H(r) for M = e_j).  Today's assembly uses Q(r_centroid) -- the
# K=1 rule.  Galerkin uses Qbar.
#
# Tetrahedron rules are given in barycentric orbit form and VALIDATED at import
# time against exact monomial integrals, so a mistyped constant cannot silently
# poison the study.

import numpy as np
from math import factorial

# ------------------------------------------------------------------
# Tetrahedron rules, barycentric orbits.
#
# Orbit types (standard notation):
#   S4    : (1/4,1/4,1/4,1/4)                          -- 1 point
#   S31(a): (1-3a, a, a, a) and permutations           -- 4 points
#   S22(a): (a, a, 1/2-a, 1/2-a) and permutations      -- 6 points
#   S211(a,b): (a, a, b, 1-2a-b) and permutations      -- 12 points
# Weights below are per POINT and sum to 1 (i.e. they integrate f over the
# tet as V * sum_q w_q f(x_q)).
# ------------------------------------------------------------------

_TET_ORBITS = {
    # name: (nominal polynomial degree, [(type, params, weight_per_point), ...])
    "K1": (1, [
        ("S4", (), 1.0),
    ]),
    "K4": (2, [
        ("S31", (0.1381966011250105,), 0.25),
    ]),
    # Two-orbit degree-3 rules form a one-parameter family; this member was
    # picked (studies/README.md) for positive weights, interior points and the
    # smallest leading degree-4 error.  Verified degree-3 exact below.
    "K8": (3, [
        ("S31", (0.3284461152882206,), 0.1310962232055796),
        ("S31", (0.1103685211074304,), 0.1189037767944204),
    ]),
    "K14": (5, [
        ("S31", (0.3108859192633005,), 0.1126879257180162),
        ("S31", (0.0927352503108912,), 0.0734930431163619),
        ("S22", (0.0455037041256497,), 0.0425460207770812),
    ]),
    "K24": (6, [
        ("S31", (0.2146028712591517,), 0.0399227502581679),
        ("S31", (0.0406739585346113,), 0.0100772110553207),
        ("S31", (0.3223378901422757,), 0.0553571815436544),
        ("S211", (0.0636610018750175, 0.2696723314583159), 0.0482142857142857),
    ]),
}


def _expand_orbit(kind, params):
    """Barycentric coordinates of one symmetry orbit, as an (n, 4) array."""
    if kind == "S4":
        return np.array([[0.25, 0.25, 0.25, 0.25]])
    if kind == "S31":
        a = params[0]
        b = 1.0 - 3.0 * a
        pts = []
        for i in range(4):
            p = [a] * 4
            p[i] = b
            pts.append(p)
        return np.array(pts)
    if kind == "S22":
        a = params[0]
        b = 0.5 - a
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                p = [b] * 4
                p[i] = a
                p[j] = a
                pts.append(p)
        return np.array(pts)          # 6 points
    if kind == "S211":
        a, b = params
        c = 1.0 - 2.0 * a - b
        pts = []
        # (a,a,b,c) over all distinct assignments: choose the pair holding a,
        # then which of the remaining two slots holds b -> 6*2 = 12 points
        for i in range(4):
            for j in range(i + 1, 4):
                rest = [k for k in range(4) if k not in (i, j)]
                for bi in range(2):
                    p = [0.0] * 4
                    p[i] = a
                    p[j] = a
                    p[rest[bi]] = b
                    p[rest[1 - bi]] = c
                    pts.append(p)
        return np.array(pts)          # 12 points
    raise ValueError(kind)


def _build_tet_rules():
    rules = {}
    for name, (deg, orbits) in _TET_ORBITS.items():
        bary, wts = [], []
        for kind, params, w in orbits:
            pts = _expand_orbit(kind, params)
            bary.append(pts)
            wts.append(np.full(len(pts), w))
        rules[name] = (np.vstack(bary), np.concatenate(wts), deg)
    return rules


TET_RULES = _build_tet_rules()


def _validate_tet_rules():
    """Exactness check on the reference tet (0,0,0),(1,0,0),(0,1,0),(0,0,1).

    Integral of x^i y^j z^k over it is i!j!k! / (i+j+k+3)!.
    """
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for name, (bary, w, deg) in TET_RULES.items():
        assert abs(w.sum() - 1.0) < 1e-13, f"{name}: weights sum to {w.sum()}"
        assert (w > 0).all(), f"{name}: has non-positive weights"
        pts = bary @ verts                              # (n,3)
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                for k in range(deg + 1 - i - j):
                    exact = (factorial(i) * factorial(j) * factorial(k)
                             / factorial(i + j + k + 3))
                    approx = (w * pts[:, 0] ** i * pts[:, 1] ** j
                              * pts[:, 2] ** k).sum() / 6.0   # V_ref = 1/6
                    assert abs(approx - exact) < 1e-12 * max(1.0, abs(exact)), (
                        f"{name}: not exact for x^{i}y^{j}z^{k} "
                        f"({approx:.16e} vs {exact:.16e})")


_validate_tet_rules()

TET_RULE_NAMES = ["K1", "K4", "K8", "K14", "K24"]


# ------------------------------------------------------------------
# Hexahedron (RecMag) rules: tensor-product Gauss-Legendre, K^3 points.
# K=1 is the midpoint rule, i.e. exactly today's centroid collocation.
# ------------------------------------------------------------------

def hex_rule(K):
    """(pts_local, weights) on [-1,1]^3, weights summing to 1."""
    x, w = np.polynomial.legendre.leggauss(K)
    w = w / 2.0                                        # normalize to sum 1 per axis
    P = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    W = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return P, W


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

TET_FACES = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]   # Radia 1-based, outward


def tet_volume(v):
    v = np.asarray(v, float)
    return abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0], v[3] - v[0]]))) / 6.0


def tet_centroid(v):
    return np.asarray(v, float).mean(axis=0)


def tet_subdivide(v):
    """1 -> 8 conforming refinement (4 corner tets + 4 from the octahedron)."""
    v = np.asarray(v, float)
    m01 = 0.5 * (v[0] + v[1]); m02 = 0.5 * (v[0] + v[2]); m03 = 0.5 * (v[0] + v[3])
    m12 = 0.5 * (v[1] + v[2]); m13 = 0.5 * (v[1] + v[3]); m23 = 0.5 * (v[2] + v[3])
    return np.array([
        [v[0], m01, m02, m03],
        [v[1], m01, m12, m13],
        [v[2], m02, m12, m23],
        [v[3], m03, m13, m23],
        # octahedron split along the m01-m23 diagonal (best aspect ratio)
        [m01, m02, m03, m23],
        [m01, m02, m12, m23],
        [m01, m03, m13, m23],
        [m01, m12, m13, m23],
    ])


def tet_rule_points(verts, rule_name):
    """Absolute quadrature points and weights (summing to 1) for one tet."""
    bary, w, _ = TET_RULES[rule_name]
    return bary @ np.asarray(verts, float), w
