/*-------------------------------------------------------------------------
*
* File name:      radgalerkin.h
*
* Project:        RADIA (RadiaCUDA)
*
* Description:    OPT-IN volume-averaged ("Galerkin") interaction-matrix
*                 assembly: the constitutive law is enforced in a
*                 volume-averaged sense over the OBSERVATION element instead
*                 of at its centroid.
*
*                 Radia's default is center collocation --
*                     Q_ij = Q_j(c_i)
*                 -- while the Galerkin entry is
*                     Qbar_ij = (1/V_i) Integral_{V_i} Q_j(r) dV.
*                 The two differ at second order in the element size; see
*                 studies/GALERKIN_STEP1.md for the measured cost/accuracy.
*
*                 This header supplies the pieces BOTH backends need: the env
*                 configuration and the observation-element quadrature. Only
*                 flat arrays cross into the CUDA translation unit.
*
*                 DEFAULT IS OFF and, when off, every code path is the
*                 pre-existing one (see radgpu_asm.cu: a separate kernel is
*                 used for the Galerkin path so the collocation kernel is
*                 untouched).
*
* Env switches (same style as RADIA_ANDERSON / RADIA_NK_*):
*   RADIA_GALERKIN=1          enable (default off)
*   RADIA_GALERKIN_K=14       base rule, tet points per observation element;
*                             one of 1, 4, 8, 14, 24. Applied to EVERY pair --
*                             STEP 1 showed a distance cutoff does not work for
*                             the base rule, because the per-pair corrections
*                             very nearly cancel for smooth M.
*                             Defaults to 14, not to the cheapest rule: K=4 is
*                             measurably NOT converged (22.5 kHz off the K>=14
*                             answer on the 60 MeV model), so it would return a
*                             number that looks like Galerkin and is not.
*   RADIA_GALERKIN_CUTOFF=1.5 radius (in units of h = V^(1/3)) of the near band
*                             that gets the higher-order rule below; 0 disables
*                             the near pass.
*                             LIMITATION: the near test compares the elements'
*                             own centroids, so a pair that is only adjacent
*                             THROUGH A SYMMETRY PLANE is not recognised and
*                             keeps the base rule. That costs some accuracy near
*                             symmetry planes but is never wrong -- both rules
*                             are valid quadratures of the same integral. It is
*                             the suspected reason widening the near band helped
*                             LESS on the 8-fold-symmetric 60 MeV model than the
*                             symmetry-free study predicted (GALERKIN_STEP1.md
*                             section 6). NOT yet isolated -- to bound it, run
*                             studies/galerkin_60mev.py --only GAL14N0 (no near
*                             pass) against GAL14 and see how much the near pass
*                             is worth at all.
*   RADIA_GALERKIN_KNEAR=14   near-band rule, tet points per sub-cell
*   RADIA_GALERKIN_NEARLEV=1  near-band subdivision levels (8^lev sub-cells),
*                             so the default near rule is 14 x 8 = 112 points
*   RADIA_GALERKIN_DEBUG=1    print the rule, point counts and near-pair count
*
-------------------------------------------------------------------------*/

#ifndef __RADGALERKIN_H
#define __RADGALERKIN_H

#include "gmvect.h"
#include <vector>

//-------------------------------------------------------------------------

struct radTGalerkinCfg {
	bool On;            // master switch
	int  K;             // base tet rule point count (1, 4, 8, 14, 24)
	int  KNear;         // near-band tet rule point count
	int  NearLevels;    // near-band subdivision levels (0 = none)
	double Cutoff;      // near-band radius in units of h = V^(1/3); 0 = off
	bool Debug;
};

// Parsed once from the environment on first use.
const radTGalerkinCfg& radGalerkinCfg();

// Observation-element quadrature in the element's OWN frame.
//
// Fills `pts`/`wts` (weights sum to 1) for element `el` using the tet rule with
// `K` points per cell and `levels` rounds of 1->8 subdivision. Returns the
// number of points, or 0 if the element type is not supported (the caller must
// then fall back to collocation for it).
//
//  * radTRecMag       -> tensor Gauss-Legendre, order chosen to match or
//                        exceed the tet rule's polynomial degree
//  * radTPolyhedron   -> the tet rule directly when the element IS a
//                        tetrahedron (4 triangular faces, the all-tet
//                        production case), otherwise a composite rule over the
//                        star decomposition from the centroid through every
//                        face triangle
//
// K == 1 and levels == 0 always returns exactly the element's centroid with
// weight 1, i.e. today's collocation.
int radGalerkinElemQuad(class radTg3dRelax* el, int K, int levels,
                        std::vector<TVector3d>& pts, std::vector<double>& wts);

// Characteristic size h = V^(1/3) of an element, used for the near-band
// radius. Returns 0 if the volume cannot be determined.
double radGalerkinElemSize(class radTg3dRelax* el);

//-------------------------------------------------------------------------

#endif
