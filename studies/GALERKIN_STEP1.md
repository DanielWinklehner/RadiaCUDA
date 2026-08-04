# Galerkin (volume-averaged) interaction-matrix assembly — STEP 1 and results

**Status: implemented behind `RADIA_GALERKIN` (default off, bit-exact no-op).
The physics question it was built to answer is answered: NO — Galerkin does not
close the −6.7 kHz radia-vs-COMSOL `mean_f` gap on the 60 MeV cyclotron. It
moves `mean_f` DOWN, i.e. the wrong way, by −6.3 kHz at N = 5880 and −4.9 kHz at
N = 10850, extrapolating to roughly −3 kHz at the converged mesh.**

Two findings from STEP 1 also invalidate the cost model the work was planned
around, and both are load-bearing for anyone who picks this up again:

1. **A distance cutoff cannot be used for the base rule.** Center collocation
   is *exact* for uniform M, so the per-pair Galerkin corrections very nearly
   cancel — the self block alone is 1000× the total for uniform M. Truncating
   that sum therefore *overshoots*: applying the 4-point rule only to the self
   block lands 8.4× further from full Galerkin than doing nothing at all, only
   inside 1.5 h it is 1.9× worse than nothing, and it does not even break even
   until R ≈ 4 h. The quadrature has to be applied to every pair.
2. **The affordable quadrature order is not accurate enough.** With M varying
   over a few elements (the realistic case near a saturation front), a 4-point
   rule on every pair captures only 30–55 % of the Galerkin correction. On the
   real 60 MeV model the K = 4 answer is **22.5 kHz** away from the K = 14 one —
   3× the discrepancy being chased. K = 14 and K = 24 agree to 0.7 kHz, so the
   scheme's answer *is* well determined from K = 14 up, but that costs **8×**
   the assembly.

So Galerkin is usable as an experiment only at K ≥ 14 and 8× the assembly cost;
the cheap variant is worse than useless because its own quadrature error exceeds
the effect. Everything below is the measurement behind those statements.

Reproduce with (radiacuda2 env, and the physics driver needs the conda env
*activated* — Intel MPI's activation vars, or `MPI_Init` aborts):

```bash
python studies/galerkin_pair_study.py        # STEP 1 (~30 min)
python studies/galerkin_asm_timing.py        # assembly cost
python studies/galerkin_60mev.py 260 160     # physics
python tests/test_galerkin_asm.py            # correctness
```

---

## 0. What the two schemes are

Radia enforces `M = M_mat(H)` at ONE point per element — the centroid. The
interaction entry between observation element *i* and source element *j* is

```
    Q_ij = Q_j(c_i)                                    "center collocation"
```

where `Q_j(r)` is the source's 3×3 field-per-unit-magnetization tensor. Galerkin
enforces the constitutive law in a volume-averaged sense instead:

```
    Qbar_ij = (1/V_i) ∫_{V_i} Q_j(r) dV
```

For a source that does not overlap the observation element every component of H
is harmonic there, so

```
    Qbar = Q(c) + ½ Mdev_ab ∂_a∂_b Q(c) + O(4th moment)
```

where `Mdev` is the **deviatoric** part of the element's normalized
second-moment tensor. The isotropic part drops out because the Hessian of a
harmonic function is traceless — that is the mean-value property, and it is why
collocation works as well as it does. Two consequences, both confirmed below:

* an element whose second-moment tensor is isotropic (a cube, a regular
  tetrahedron) has **no leading correction at all**;
* any degree-2-exact rule (4 points on a tet) reproduces the whole leading term,
  so its residual is 4th order.

`Q(r)` for the study is taken from Radia itself: one magnetized element and
`rad.Fld(obj, 'h', pts)` returns exactly the tensor column the assembly kernel
computes, so nothing about the field formulas is re-derived.

## 1. The reference integrator is validated, not assumed

The Galerkin matrix is symmetric in the V-weighted norm — `V_i Qbar_ij =
V_j Qbar_ji` exactly, because the kernel `G(r,r')` is symmetric. The two sides
are computed from completely different quadratures, so this is a real check:

| quantity | measured |
|---|---|
| Galerkin asymmetry `‖V_i Qbar_ij − (V_j Qbar_ji)ᵀ‖` / scale | **3.9e-13** (= reference tolerance) |
| collocation asymmetry, same pairs | **2.9e-2** median, 4.3e-2 max |

The 3 % asymmetry on the right is a property of today's matrix, not an error.

Two more identities hold to machine precision: `tr Q = −1` for every self block
under both schemes (max deviation 3e-15), and for uniform M the sum over all
mesh elements of the per-shell corrections reproduces a single closed-form
`ObjRecMag` covering the same box (`H` agrees to 1.3e-10).

## 2. (a) Accuracy of K = 1 vs separation — and it is a *shape* effect

Controlled sweep: regular-tet source, observation element translated to
`d/h` (h = mean element V^(1/3), the length the pair-count cost model uses),
4 random orientations, worst relative entry error `‖Q_K − Qbar‖/‖Qbar‖`:

Observation shape = **regular tetrahedron**, `‖Mdev‖/V^(2/3) = 0` :

| d/h | K1 | K4 | K14 | K24 |
|---|---|---|---|---|
| 2.0–2.5 | 8.2e-2 | 4.7e-2 | 1.7e-2 | 1.6e-2 |
| 3–4 | 2.0e-2 | 6.9e-3 | 2.0e-4 | 4.4e-5 |
| 5–7 | 4.6e-3 | 1.6e-3 | 6.1e-6 | 3.9e-7 |
| 10–15 | 6.0e-4 | 2.0e-4 | 1.0e-7 | 2.5e-9 |
| 25–40 | 1.9e-5 | 6.4e-6 | 9.7e-11 | 3.3e-12 |

Observation shape = **median gmsh tet**, `‖Mdev‖/V^(2/3) = 0.122` :

| d/h | K1 | K4 | K14 | K24 |
|---|---|---|---|---|
| 2.0–2.5 | 3.3e-1 | 9.9e-2 | 1.5e-1 | 9.1e-2 |
| 3–4 | 1.1e-1 | 1.1e-2 | 9.3e-4 | 3.2e-4 |
| 5–7 | 3.4e-2 | 2.7e-3 | 2.6e-5 | 3.2e-6 |
| 10–15 | 8.1e-3 | 3.5e-4 | 3.8e-7 | 1.8e-8 |
| 25–40 | 7.7e-4 | 1.1e-5 | 3.7e-10 | 2.0e-11 |

The two K1 columns differ by 4× at d/h = 2, 13× at d/h = 10 and 40× at d/h = 30 —
the ratio *grows* because the isotropic element has no second-order term at all
and decays two orders faster. And K1 tracks `‖Mdev‖` across all five shapes
tested (0, 0.092, 0.122, 0.178, 0.319 → K1 error at d/h = 10 of 6.0e-4, 3.9e-3,
8.1e-3, 8.7e-3, 1.3e-2). That is the `½ Mdev:∂∂Q` term, measured. A production
gmsh mesh of a box has `‖Mdev‖/V^(2/3)` median 0.122, p90 0.178.

The same sweep for a **cube source and cube observation** (the structured-lattice
case) gives K1 errors of 2.6e-3 at d/h = 3 and 2.2e-5 at d/h = 10 — a factor 40
better than the tets, again because a cube's second moment is isotropic.

## 3. (b) The cutoff — this is where the plan breaks

Two independent measurements say the same thing.

**Per-pair upper bound.** Summing `|dQ|` over a shell with
`n_shell = 4/3 π (hi³−lo³)` sources (uniform mesh, |M| = 1) — the pessimistic
case in which every source's error pushes H the same way — gives a truncation
residual that plateaus and never gets small:

| R (d/h) | K4 inside | K14 inside | K24 inside |
|---|---|---|---|
| 3.0 | 37.7 % | 22.5 % | 19.9 % |
| 5.0 | 28.3 % | 12.0 % | 9.3 % |
| 10.0 | 23.5 % | 6.9 % | 4.3 % |
| ∞ | 22.6 % | 5.9 % | 3.3 % |

**Signed total correction — the decisive one.** Because everything is linear in
the sources, the exact signed correction contributed by a *set* S of sources is

```
    Σ_{j∈S} (Qbar_ij − Q_ij(c_i)) M_j = (1/V_i) ∫_{V_i} H_S dV − H_S(c_i)
```

i.e. one Radia container per shell — no per-pair quadrature at all. The same
trick prices any quadrature rule applied to S. Measured on an 8137-tet gmsh mesh
(box ≈ 19.6 h across, so shells are complete to R ≈ 9.8 h), for the realistic
case where M varies over ~4 elements:

| R (d/h) | K1 | K4 | K14 | K24 | K14×8 |
|---|---|---|---|---|---|
| self only | 100 % | 842 % | 1133 % | 1184 % | 1219 % |
| 1.5 | 100 % | 192 % | 224 % | 227 % | 233 % |
| 2.5 | 100 % | 115 % | 73 % | 73 % | 70 % |
| 4.0 | 100 % | 78 % | 35 % | 27 % | 20 % |
| 9.0 | 100 % | 71 % | 29 % | 20 % | 11 % |
| ∞ | 100 % | 70 % | 28 % | 18 % | 9 % |

(100 % = today's scheme, which produces no correction at all; **above** 100 %
means the scheme is further from full Galerkin than doing nothing.)

The reason is visible in the per-shell magnitudes: the self block alone
contributes 5.4e-3 and the 0–1.5 h shell 8.9e-3, while the **total is 8.1e-3** —
the shells cancel. For uniform M the cancellation is a factor of 1000 (self
6.1e-3, total 3.2e-5). Collocation is exact for uniform M — `Σ_j Q_j(r) =
Q_body(r)` identically — so any partial application of Galerkin breaks a
cancellation that the full scheme relies on.

**There is no small R.** Per the original instruction, that is the negative
result: it was much cheaper to learn here than after writing the kernel.

The same table for uniform and smooth-gradient M is *not* reported, because
there the total is so small (3e-5 against a self-block term of 6e-3) that the
adaptive reference itself cannot resolve it — the script prints a per-mode
"reference floor" and it exceeds 100 % of `|dH_total|` for those two modes. What
that says is only that the correction is negligible for smooth M, which is the
useful part.

### How big is the correction at all?

`|dH_total| / |H|` at an interior element, same mesh, by how fast M varies:

| M field | correction |
|---|---|
| uniform | 0.009 – 0.041 % |
| smooth gradient | 0.056 – 0.088 % |
| wave, λ = 8 h | 0.56 – 1.06 % |
| wave, λ = 4 h | 0.37 – 2.59 % |
| random per element | 6.1 – 13.5 % |

Consistent with `½ Mdev:∂∂H`: the correction scales as `(h/L)²` where L is the
length scale of M. In a converged model's bulk it is ~1e-4; near a saturation
front it is ~1 %. A 1 % change in H where M varies fast is easily enough to move
`mean_f` by 1e-3 — which is why this was worth testing.

## 4. (c) K needed, including self and adjacent blocks

Per-pair worst-case relative entry errors over 550 pairs drawn from a gmsh mesh,
by adjacency (`K4x8` = the 4-point rule on 8 sub-tets = 32 points, `K14x8` =
112 points, `K4x64` = 256 points):

| shared vertices | n | d/h med | K1 | K4 | K14 | K24 | K4×8 | K14×8 | K4×64 |
|---|---|---|---|---|---|---|---|---|---|
| 4 (self) | 10 | 0 | 1.3e-1 | 4.4e-2 | 1.4e-2 | 8.2e-3 | 1.1e-2 | 4.6e-3 | 2.7e-3 |
| 3 (shared face) | 28 | 0.93 | 5.8e-1 | 1.9e-1 | 8.6e-2 | 4.8e-2 | 5.2e-2 | 2.7e-2 | 1.2e-2 |
| 2 (shared edge) | 82 | 1.58 | 4.8e-1 | 1.9e-1 | 9.1e-2 | 5.4e-2 | 3.9e-2 | 3.2e-2 | 8.5e-3 |
| 1 (shared vertex) | 113 | 2.43 | 2.3e-1 | 6.3e-2 | 2.6e-2 | 5.0e-3 | 1.1e-2 | 3.7e-3 | 1.0e-3 |
| 0 | 317 | 5.65 | 1.6e-1 | 2.0e-2 | 1.4e-3 | 3.3e-4 | 1.9e-3 | 4.5e-5 | 1.2e-4 |

Subdividing beats raising the single-cell order for the touching pairs, as
expected — the integrand has a boundary layer at the shared face and a
log singularity along shared edges.

**Self blocks.** Radia's analytic centroid demagnetization tensor differs from
the volume-averaged one by, over 25 gmsh tets,

```
  |Qbar − Q_centroid| / |Q_centroid|:   median 5.85 %   p90 18.1 %   max 19.6 %
```

and the correction is purely deviatoric (`tr Q = −1` holds for both to 3e-15).
The residual after quadrature, relative to `|Q_centroid|`:

| rule | K4 | K8 | K14 | K24 | K4×8 | K14×8 | K4×64 |
|---|---|---|---|---|---|---|---|
| median | 1.98 % | 0.44 % | 0.55 % | 0.33 % | 0.54 % | 0.16 % | 0.14 % |
| max | 6.61 % | 2.94 % | 2.97 % | 1.68 % | 1.80 % | 1.04 % | 0.34 % |

Radia's analytic self-demagnetization is **not** broken by any of this: the
K = 1 setting reproduces it bit-exactly (`tests/test_galerkin_asm.py`), and the
Galerkin self block is a different, well-defined quantity with the same trace.

For **RecMag** elements the self block is special: a **cube needs no correction
at all** (`|dQ|/|Q|` = 0 to 1e-13, forced by cubic symmetry), but a 2:1:1 slab
needs 13.6 % and a 4:1:1 slab 12.8 %. So structured lattices of near-cubes are
almost unaffected while anisotropic ones are not.

## 5. Cost

Assembly wall clock (`RlxPre`, GPU backend, RTX 4070 Ti Super), tet lattice.
**Caveat: another GPU job was resident throughout these measurements**, so the
absolute numbers are contended and the small-N rows are dominated by fixed
overhead; the multipliers at the largest N are the trustworthy part.

| N | collocation | K=4 | K=14 |
|---|---|---|---|
| 3072 | 0.61 s | 0.87 s (1.4×) | 4.96 s (8.1×) |
| 6000 | 2.02 s | 4.32 s (2.1×) | 18.8 s (9.3×) |
| 10368 | 6.45 s | 17.8 s (2.8×) | 49.5 s (7.7×) |

(The near-pass column from that sweep is omitted: contention made it come out
*faster* than K = 4 alone, which is impossible, so the run is not usable. Redo
`studies/galerkin_asm_timing.py` on an idle GPU for clean numbers.)

On the real 60 MeV rung (N = 5880, same GPU, same contention):

| scheme | assembly | × collocation |
|---|---|---|
| collocation | 27.5 s | 1.0 |
| K=4, near to 1.5 h | 95.5 s | 3.5 |
| K=14, near to 1.5 h | 224.7 s | 8.2 |
| K=4, near to 2.5 h | 245.7 s | 8.9 |
| K=14, near to 2.5 h | 584.1 s | 21.2 |
| K=24, near to 1.5 h | 633.9 s | 23.1 |

and at N = 10850: collocation 196.0 s, K = 14 with the 1.5 h near band 905.2 s
(4.6× — lower than the 8.2× above only because that collocation baseline was
itself badly contended).

The near pass with cutoff 1.5 h touches 95 886 pairs = 0.277 % of N², 16.3 per
element — cheap, as designed; widening it to 2.5 h is what makes the fourth row
cost as much as K = 14 everywhere.

The multiplier is below K because assembly also pays fixed costs (packing,
the N² host memcpy, unpack). The two-pass structure works: the N² bulk runs at
the base rule's speed with no warp divergence, and the near pass is a separate
sparse kernel over a precomputed pair list.

## 6. Physics: the 60 MeV benchmark

60 MeV base magnetic model (HCHC-60), all-tet, yoke 260 / pole 160 mm,
N = 5880, m11, `num_segments: 50`. COMSOL reference 7.726343 MHz ± 1.7 kHz.

| scheme | asm | ×  | misfit | matvecs | mean_f [MHz] | vs COMSOL | Δ vs collocation |
|---|---|---|---|---|---|---|---|
| collocation (default) | 27.5 s | 1.0 | 2.3e-06 | 3321 | 7.710118 | −16.225 kHz | — |
| Galerkin K=4, near to 1.5 h | 95.5 s | 3.5 | 3.2e-06 | 11490 | 7.681298 | −45.045 kHz | **−28.820 kHz** |
| Galerkin K=4, near to 2.5 h | 245.7 s | 8.9 | 5.1e-07 | 2010 | 7.703213 | −23.130 kHz | **−6.905 kHz** |
| Galerkin K=14, near to 1.5 h | 224.7 s | 8.2 | 4.3e-08 | 2630 | 7.703782 | −22.561 kHz | **−6.337 kHz** |
| Galerkin K=24, near to 1.5 h | 633.9 s | 23.1 | 4.0e-07 | 2307 | 7.704476 | −21.867 kHz | **−5.643 kHz** |
| Galerkin K=14, near to 2.5 h | 584.1 s | 21.2 | 1.1e-06 | 1992 | 7.705000 | −21.343 kHz | **−5.118 kHz** |

Every variant moves `mean_f` **down, away from COMSOL**. The rule sequence
converges from both directions — raising the base order (K = 4 → 14 → 24 at a
1.5 h near band: −28.8, −6.34, −5.64 kHz) and widening the near band (K = 4 and
K = 14 at 2.5 h: −6.91, −5.12 kHz) — on **≈ −5 kHz**. That shift is therefore a
property of the scheme, not of the quadrature. K = 4 with the narrow near band is
23 kHz off it: its own quadrature error is 4× the effect, exactly as STEP 1 §3
predicts for a mesh where M varies over a few elements.

STEP 1's *other* prediction — that widening the near band beats raising the base
order — **does not survive contact with the real model.** Taking K = 14 out to
2.5 h (−5.118 kHz) as the best-resolved answer, the errors at roughly equal cost
are 1.22 kHz for K = 14 out to 1.5 h (224.7 s) and 1.79 kHz for K = 4 out to
2.5 h (245.7 s): the base order wins. The idealized study said the opposite
(§3: 18 % vs 28 % for λ = 4 h). The likely reason is the limitation noted in
`radgalerkin.h` — the near test compares elements' own centroids and so misses
pairs that are adjacent only *through a symmetry plane*, and this model has both
8-fold rotation and a median plane. Widening a near band that is incomplete buys
less than the study's symmetry-free geometry predicts.

The collocation baseline reproduces the project's saved 7.709706 (taken with
`num_segments: 25`) plus the expected +0.41 kHz from the config's move to
`num_segments: 50`, so the rung is the intended one.

### Mesh trend

The same K = 14 comparison at a second, finer rung:

| rung | N | collocation | Galerkin K=14 | shift | vs COMSOL (Galerkin) |
|---|---|---|---|---|---|
| yoke 260 / pole 160 | 5 880 | 7.710118 | 7.703782 | **−6.337 kHz** | −22.561 kHz |
| yoke 180 / pole 110 | 10 850 | 7.717036 | 7.712148 | **−4.888 kHz** | −14.195 kHz |

The shift does shrink with refinement, but **more slowly than the `O(h²)` the
theory predicts**: 1.85× the elements (1.23× finer linearly) cut it by only 1.30×,
an implied exponent of ≈1.3 rather than 2. Two rungs and a rule-convergence
uncertainty of ~1.2 kHz do not determine that exponent well, so take it as a
range: extrapolating to the ladder's converged rung (yoke 120 / pole 72,
N = 27 820) gives **−2.6 kHz at p = 2 and −3.3 kHz at p = 1.3**.

So at the converged mesh Galerkin would take the gap from −6.7 kHz to roughly
**−9 to −10 kHz**. Not negligible, and firmly the wrong direction.

Center collocation is therefore **not** the source of the −6.7 kHz level
discrepancy, and this rules out the last standing candidate from the 2026-07-25
elimination exercise. Whatever produces the gap is still unidentified.

### A possible side benefit — but only half-confirmed

The m11 solve needed **fewer matvecs at both rungs** with Galerkin K = 14
(2630 vs 3321 at 260/160; 2365 vs 3335 at 180/110), which is the direction the
V-weighted-symmetric operator predicts. The residual *floor*, however, did not
behave consistently:

| rung | collocation misfit | Galerkin K=14 misfit |
|---|---|---|
| 260/160 | 2.3e-06 | **4.3e-08** (50× better) |
| 180/110 | 3.1e-06 | 6.7e-06 (2× worse) |

One rung showed a spectacular floor improvement and the other did not, so this is
a lead, not a result. If the m11 floor work wants to chase it, the cheap probe is
to run both rungs again at several `RADIA_GALERKIN_K` and see whether the floor
tracks how close the operator is to exactly symmetric (finite-order quadrature
only makes `V_i Q̄_ij = V_j Q̄_ji` approximate).

## 7. Using it

**Not installed.** Everything above was measured against the build in
`build_dev/cpp` (`PYTHONPATH=…/RadiaCUDA/build_dev/cpp`), because
`optimize_shims.py` was running against the installed `radia.pyd` throughout.
Install when convenient:

```bash
pip install . --no-build-isolation
```

Default OFF. Verified bit-exact when off: solved B is byte-identical to the
pre-change build on both backends for 6 model families
(`tests/test_galerkin_asm.py --baseline <PYTHONPATH-of-reference-build>`), and
the whole existing suite passes unchanged.

| variable | default | meaning |
|---|---|---|
| `RADIA_GALERKIN` | off | `1` enables |
| `RADIA_GALERKIN_K` | 14 | base rule, tet points per element: 1, 4, 8, 14, 24 — applied to EVERY pair. Defaults to 14, not to the cheapest rule, because K = 4 is measurably not converged (§6) and would hand back a number that looks like Galerkin and is not |
| `RADIA_GALERKIN_CUTOFF` | 1.5 | near-band radius in units of h = V^(1/3); 0 disables the near pass |
| `RADIA_GALERKIN_KNEAR` | 14 | near-band rule |
| `RADIA_GALERKIN_NEARLEV` | 1 | near-band subdivision levels (8^lev sub-cells), so the default near rule is 14 × 8 = 112 points |
| `RADIA_GALERKIN_DEBUG` | off | print rule, points/element, near-pair count |

`RADIA_GALERKIN_K=1` with `RADIA_GALERKIN_CUTOFF=0` is collocation, bit-exactly.

RecMag elements use tensor Gauss-Legendre at an order matching the tet rule's
degree; tetrahedral polyhedra use the tet rule directly; any other polyhedron
uses a composite rule over the star decomposition from its centroid, which is
correct but costs more points. An element type with no volume quadrature
(neither polyhedron nor RecMag) declines the GPU path and reports
`Radia::Warning024` before falling back to collocation.

**Galerkin shifts every result.** Every baseline in the cyclotron project —
shim solutions, both mesh ladders, the COMSOL comparison — would need
revalidation before it could be adopted. Given §6, it should not be.

## 8. Files

| file | what |
|---|---|
| `cpp/src/core/radgalerkin.{h,cpp}` | config + observation-element quadrature, shared by both backends |
| `cpp/src/core/radgpu_asm.{h,cpp,cu}` | CSR obs-quadrature packing, near-pair list, `assemble_near_kernel` |
| `cpp/src/core/radintrc.{h,cpp}` | `PrepGalerkinQuad`, `GalerkinInteractBlock`, hooks in both CPU loops |
| `tests/test_galerkin_asm.py` | opt-in no-op, CPU/GPU parity, K-convergence |
| `studies/galerkin_quad.py` | tet/hex rules, validated against exact monomial integrals |
| `studies/galerkin_pair_study.py` | STEP 1 |
| `studies/galerkin_asm_timing.py` | assembly cost |
| `studies/galerkin_60mev.py` | physics validation driver |
