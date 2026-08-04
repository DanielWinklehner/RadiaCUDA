/*-------------------------------------------------------------------------
*
* File name:      radgpu_asm.cpp
*
* Project:        RADIA
*
* Description:    Pack Radia geometry into flat arrays for GPU assembly
*
-------------------------------------------------------------------------*/

#ifdef RADIA_WITH_CUDA

#include "radgpu_asm.h"
#include "radintrc.h"
#include "radsend.h"
#include "radcast.h"
#include "radrec.h"
#include "radvlpgn.h"
#include "radplnr.h"
#include "radtrans.h"
#include "radgalerkin.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// (Removed dead, buggy ExtractPolyFaceData helper: it was never called, and its
//  face_rot fill wrote the same diagonal-only row three times. The live packing in
//  radGPU_PackGeometryForAsm computes face_rot correctly inline. See issue #11.)

// ============================================================
// Observation-element quadrature (Galerkin; radgalerkin.h).
//
// Collocation is packed as a single point per element -- the transformed
// centroid -- with weight 1, so the kernel's quadrature loop degenerates to
// exactly the old arithmetic.
//
// Returns 1 on success, 0 if some element's quadrature could not be built (the
// caller then declines the GPU path, as it does for unknown element types).
// ============================================================
int radGPU_PackObsQuadForAsm(radTInteraction* intrct, RadGPU_ObsQuadData* qd)
{
    const radTGalerkinCfg& cfg = radGalerkinCfg();
    int N = intrct->AmOfMainElem;
    memset(qd, 0, sizeof(RadGPU_ObsQuadData));
    qd->on = cfg.On ? 1 : 0;
    qd->n_elem = N;

    int Kbase = cfg.On ? cfg.K : 1;
    int LevBase = 0;                       // the base rule is never subdivided

    std::vector<int> off(N + 1, 0);
    std::vector<double> pts, wts;
    std::vector<TVector3d> ep;
    std::vector<double> ew;
    pts.reserve((size_t)N * 4 * 3);
    wts.reserve((size_t)N * 4);

    for(int i = 0; i < N; i++) {
        off[i] = (int)wts.size();
        radTg3dRelax* el = intrct->g3dRelaxPtrVect[i];
        if(!radGalerkinElemQuad(el, Kbase, LevBase, ep, ew)) return 0;
        radTrans* tr = intrct->MainTransPtrArray[i];
        for(size_t k = 0; k < ep.size(); k++) {
            TVector3d p = (tr != 0) ? tr->TrPoint(ep[k]) : ep[k];
            pts.push_back(p.x); pts.push_back(p.y); pts.push_back(p.z);
            wts.push_back(ew[k]);
        }
    }
    off[N] = (int)wts.size();

    qd->q_total = (int)wts.size();
    qd->q_offsets = new int[N + 1];
    qd->q_pts = new double[3 * (size_t)qd->q_total];
    qd->q_w = new double[(size_t)qd->q_total];
    memcpy(qd->q_offsets, &off[0], (size_t)(N + 1) * sizeof(int));
    memcpy(qd->q_pts, &pts[0], 3 * (size_t)qd->q_total * sizeof(double));
    memcpy(qd->q_w, &wts[0], (size_t)qd->q_total * sizeof(double));

    if(cfg.Debug) {
        fprintf(stderr, "Galerkin: base rule K=%d -> %.2f points/element "
                        "(N=%d, total %d)\n",
                Kbase, (double)qd->q_total / (double)N, N, qd->q_total);
    }

    // --- near pass ---------------------------------------------------
    if(!cfg.On || (cfg.Cutoff <= 0.) ||
       ((cfg.KNear == Kbase) && (cfg.NearLevels <= 0))) return 1;

    // Near-band quadrature, same layout as the base rule.
    std::vector<int> noff(N + 1, 0);
    std::vector<double> npts, nwts;
    for(int i = 0; i < N; i++) {
        noff[i] = (int)nwts.size();
        radTg3dRelax* el = intrct->g3dRelaxPtrVect[i];
        if(!radGalerkinElemQuad(el, cfg.KNear, cfg.NearLevels, ep, ew)) return 0;
        radTrans* tr = intrct->MainTransPtrArray[i];
        for(size_t k = 0; k < ep.size(); k++) {
            TVector3d p = (tr != 0) ? tr->TrPoint(ep[k]) : ep[k];
            npts.push_back(p.x); npts.push_back(p.y); npts.push_back(p.z);
            nwts.push_back(ew[k]);
        }
    }
    noff[N] = (int)nwts.size();

    // Near-pair list. Centroids in the same (transformed) frame the kernel
    // observes in, and h_i = V^(1/3) per element; a pair is "near" when
    // |c_i - c_j| < cutoff * max(h_i, h_j). Bucketed on a uniform grid of
    // cutoff*h_max so the scan is O(N), not O(N^2).
    std::vector<double> cx(N), cy(N), cz(N), hh(N);
    double hMax = 0.;
    for(int i = 0; i < N; i++) {
        radTg3dRelax* el = intrct->g3dRelaxPtrVect[i];
        radTrans* tr = intrct->MainTransPtrArray[i];
        TVector3d c = (tr != 0) ? tr->TrPoint(el->ReturnCentrPoint())
                                : el->ReturnCentrPoint();
        cx[i] = c.x; cy[i] = c.y; cz[i] = c.z;
        hh[i] = radGalerkinElemSize(el);
        if(hh[i] > hMax) hMax = hh[i];
    }
    if(hMax <= 0.) return 0;

    double cell = cfg.Cutoff * hMax;
    if(!(cell > 0.)) return 0;
    double xlo = *std::min_element(cx.begin(), cx.end());
    double ylo = *std::min_element(cy.begin(), cy.end());
    double zlo = *std::min_element(cz.begin(), cz.end());
    double xhi = *std::max_element(cx.begin(), cx.end());
    double yhi = *std::max_element(cy.begin(), cy.end());
    double zhi = *std::max_element(cz.begin(), cz.end());
    long long nx = (long long)((xhi - xlo) / cell) + 1;
    long long ny = (long long)((yhi - ylo) / cell) + 1;
    long long nz = (long long)((zhi - zlo) / cell) + 1;
    // Keep the grid from exploding on very non-uniform meshes.
    while(nx * ny * nz > 8LL * N + 64LL) {
        cell *= 1.5;
        nx = (long long)((xhi - xlo) / cell) + 1;
        ny = (long long)((yhi - ylo) / cell) + 1;
        nz = (long long)((zhi - zlo) / cell) + 1;
    }
    std::vector<std::vector<int> > bucket((size_t)(nx * ny * nz));
    std::vector<long long> bi(N);
    for(int i = 0; i < N; i++) {
        long long ix = (long long)((cx[i] - xlo) / cell); if(ix >= nx) ix = nx - 1;
        long long iy = (long long)((cy[i] - ylo) / cell); if(iy >= ny) iy = ny - 1;
        long long iz = (long long)((cz[i] - zlo) / cell); if(iz >= nz) iz = nz - 1;
        bi[i] = (ix * ny + iy) * nz + iz;
        bucket[(size_t)bi[i]].push_back(i);
    }

    std::vector<int> rows, cols;
    for(int i = 0; i < N; i++) {
        long long ix = bi[i] / (ny * nz);
        long long rem = bi[i] - ix * ny * nz;
        long long iy = rem / nz, iz = rem - iy * nz;
        for(long long dx = -1; dx <= 1; dx++)
        for(long long dy = -1; dy <= 1; dy++)
        for(long long dz = -1; dz <= 1; dz++) {
            long long jx = ix + dx, jy = iy + dy, jz = iz + dz;
            if((jx < 0) || (jy < 0) || (jz < 0) || (jx >= nx) || (jy >= ny) || (jz >= nz))
                continue;
            const std::vector<int>& b = bucket[(size_t)((jx * ny + jy) * nz + jz)];
            for(size_t k = 0; k < b.size(); k++) {
                int j = b[k];
                double ddx = cx[i] - cx[j], ddy = cy[i] - cy[j], ddz = cz[i] - cz[j];
                double d = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                double hij = (hh[i] > hh[j]) ? hh[i] : hh[j];
                if(d < cfg.Cutoff * hij) { rows.push_back(i); cols.push_back(j);}
            }
        }
    }

    qd->n_total = (int)nwts.size();
    qd->n_offsets = new int[N + 1];
    memcpy(qd->n_offsets, &noff[0], (size_t)(N + 1) * sizeof(int));
    if(qd->n_total > 0) {
        qd->n_pts = new double[3 * (size_t)qd->n_total];
        qd->n_w = new double[(size_t)qd->n_total];
        memcpy(qd->n_pts, &npts[0], 3 * (size_t)qd->n_total * sizeof(double));
        memcpy(qd->n_w, &nwts[0], (size_t)qd->n_total * sizeof(double));
    }
    qd->n_pairs = (int)rows.size();
    if(qd->n_pairs > 0) {
        qd->pair_rows = new int[(size_t)qd->n_pairs];
        qd->pair_cols = new int[(size_t)qd->n_pairs];
        memcpy(qd->pair_rows, &rows[0], (size_t)qd->n_pairs * sizeof(int));
        memcpy(qd->pair_cols, &cols[0], (size_t)qd->n_pairs * sizeof(int));
    }

    if(cfg.Debug) {
        fprintf(stderr, "Galerkin: near rule K=%d x 8^%d -> %.1f points/element; "
                        "%d near pairs (cutoff %.2f h, %.3f%% of N^2, %.1f per element)\n",
                cfg.KNear, cfg.NearLevels, (double)qd->n_total / (double)N,
                qd->n_pairs, cfg.Cutoff,
                100. * (double)qd->n_pairs / ((double)N * (double)N),
                (double)qd->n_pairs / (double)N);
    }
    return 1;
}

// ============================================================
// Pack geometry from Radia interaction data
// ============================================================
int radGPU_PackGeometryForAsm(
    radTInteraction* intrct,
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_ObsQuadData* quadData)
{
    int N = intrct->AmOfMainElem;
    if(N <= 0) return 0;

    radTCast Cast;

    // --- Classify elements (mixed RecMag + polyhedron models are supported:
    //     the assembly kernel branches per SOURCE element type) ---
    int nRec = 0, nPoly = 0;
    for(int i = 0; i < N; i++) {
        radTg3dRelax* rel = intrct->g3dRelaxPtrVect[i];
        radTRecMag* recPtr = Cast.RecMagCast(rel);
        radTPolyhedron* polyPtr = Cast.PolyhedronCast(rel);
        if(recPtr) nRec++;
        else if(polyPtr) nPoly++;
    }
    if(nRec + nPoly != N) {
        // Some element is neither a RecMag nor a polyhedron (e.g. an extruded
        // polygon) -- no GPU kernel for it; warn and use the CPU assembly.
        radTSend::WarningMessage("Radia::Warning020");
        return 0;
    }

    // --- Extract observation centers (transformed by MainTransPtrArray) ---
    // These are the points where we evaluate the field FROM each source element
    double* obsCenters = new double[3 * N];
    for(int i = 0; i < N; i++) {
        TVector3d cp = intrct->MainTransPtrArray[i]->TrPoint(intrct->g3dRelaxPtrVect[i]->ReturnCentrPoint());
        obsCenters[3*i+0] = cp.x;
        obsCenters[3*i+1] = cp.y;
        obsCenters[3*i+2] = cp.z;
    }

    // --- Per-row finalizing transform: s*M_inv of MainTransPtrArray[i] -------
    // Extracted through the public virtual by transforming the identity:
    // TrMatrix_inv(I) == s*M_inv*I == s*M_inv exactly (the products are with
    // 1.0 and 0.0), and radIdentTrans overrides it to a no-op, so rows with no
    // base transform come back as the identity. Going through the virtual also
    // keeps composite transforms correct without touching radTrans internals.
    double* rowTrans = new double[9 * N];
    for(int i = 0; i < N; i++) {
        TMatrix3d m(TVector3d(1., 0., 0.), TVector3d(0., 1., 0.), TVector3d(0., 0., 1.));
        radTrans* tr = intrct->MainTransPtrArray[i];
        if(tr != 0) tr->TrMatrix_inv(m);
        double* d = rowTrans + 9*i;
        d[0] = m.Str0.x; d[1] = m.Str0.y; d[2] = m.Str0.z;
        d[3] = m.Str1.x; d[4] = m.Str1.y; d[5] = m.Str1.z;
        d[6] = m.Str2.x; d[7] = m.Str2.y; d[8] = m.Str2.z;
    }

    // --- RecMag packing (global element indexing; zeros where not a RecMag) ---
    memset(recData, 0, sizeof(RadGPU_RecMagData));
    recData->n_rec = nRec;
    recData->is_rec = new int[N]();
    recData->centers = new double[3 * N]();
    recData->dims = new double[3 * N]();
    recData->abs_rand = radCR.AbsRand;
    recData->rel_rand = radCR.RelRand;
    recData->zero_rand = radCR.ZeroRand;
    recData->act_on_doubles = radCR.ActOnDoubles;
    if(nRec > 0) {
        for(int i = 0; i < N; i++) {
            radTRecMag* rec = Cast.RecMagCast(intrct->g3dRelaxPtrVect[i]);
            if(!rec) continue;
            recData->is_rec[i] = 1;
            recData->centers[3*i+0] = rec->CentrPoint.x;
            recData->centers[3*i+1] = rec->CentrPoint.y;
            recData->centers[3*i+2] = rec->CentrPoint.z;
            recData->dims[3*i+0] = rec->Dimensions.x;
            recData->dims[3*i+1] = rec->Dimensions.y;
            recData->dims[3*i+2] = rec->Dimensions.z;
        }
    }

    // --- Polyhedron face packing (global element indexing; RecMag elements
    //     get an empty face range) ---
    memset(polyData, 0, sizeof(RadGPU_PolyData));
    {
        // First pass: count faces and edges
        int totalFaces = 0, totalEdges = 0;
        for(int i = 0; i < N; i++) {
            radTPolyhedron* poly = Cast.PolyhedronCast(intrct->g3dRelaxPtrVect[i]);
            if(!poly) continue;
            totalFaces += poly->AmOfFaces;
            for(int fi = 0; fi < poly->AmOfFaces; fi++) {
                totalEdges += poly->VectHandlePgnAndTrans[fi].PgnHndl.rep->AmOfEdgePoints;
            }
        }

        polyData->n_elem = N;
        polyData->n_faces_total = totalFaces;
        polyData->n_edges_total = totalEdges;
        polyData->centers = obsCenters;  // use transformed obs centers
        polyData->row_trans = rowTrans;  // s*M_inv per row element
        polyData->face_offsets = new int[N + 1];
        polyData->edge_offsets = new int[totalFaces + 1];
        polyData->face_cz = new double[totalFaces];
        polyData->face_rot = new double[9 * totalFaces];
        polyData->face_orig = new double[3 * totalFaces];
        polyData->edge_pts_2d = new double[2 * totalEdges];

        // Second pass: fill arrays
        int faceIdx = 0, edgeIdx = 0;
        for(int i = 0; i < N; i++) {
            polyData->face_offsets[i] = faceIdx;
            radTPolyhedron* poly = Cast.PolyhedronCast(intrct->g3dRelaxPtrVect[i]);
            if(!poly) continue;  // RecMag: empty face range

            for(int fi = 0; fi < poly->AmOfFaces; fi++) {
                radTHandlePgnAndTrans& hpt = poly->VectHandlePgnAndTrans[fi];
                radTPolygon* pgn = hpt.PgnHndl.rep;
                radTrans* tr = hpt.TransHndl.rep;

                // Edge offset for this face
                polyData->edge_offsets[faceIdx] = edgeIdx;

                // Face origin
                TVector3d origin(0., 0., 0.);
                origin = tr->TrBiPoint(origin);
                polyData->face_orig[3*faceIdx+0] = origin.x;
                polyData->face_orig[3*faceIdx+1] = origin.y;
                polyData->face_orig[3*faceIdx+2] = origin.z;

                // Rotation matrix: lab -> local (transpose of TrBiPoint rotation)
                TVector3d ex(1,0,0), ey(0,1,0), ez(0,0,1);
                TVector3d labEx = tr->TrBiPoint(ex) - origin;
                TVector3d labEy = tr->TrBiPoint(ey) - origin;
                TVector3d labEz = tr->TrBiPoint(ez) - origin;

                double* rot = &polyData->face_rot[9*faceIdx];
                rot[0] = labEx.x; rot[1] = labEx.y; rot[2] = labEx.z;
                rot[3] = labEy.x; rot[4] = labEy.y; rot[5] = labEy.z;
                rot[6] = labEz.x; rot[7] = labEz.y; rot[8] = labEz.z;

                // Coord Z
                polyData->face_cz[faceIdx] = pgn->CoordZ;

                // Edge points 2D
                int ne = pgn->AmOfEdgePoints;
                for(int ei = 0; ei < ne; ei++) {
                    TVector2d& ep = pgn->EdgePointsVector[ei];
                    polyData->edge_pts_2d[2*edgeIdx+0] = ep.x;
                    polyData->edge_pts_2d[2*edgeIdx+1] = ep.y;
                    edgeIdx++;
                }
                faceIdx++;
            }
        }
        polyData->face_offsets[N] = faceIdx;
        polyData->edge_offsets[totalFaces] = edgeIdx;
    }

    // --- Per-element symmetry transforms ---
    memset(symData, 0, sizeof(RadGPU_SymData));
    symData->n_elem = N;

    // First pass: count total copies
    int totalCopies = 0;
    std::vector<int> counts(N);
    for(int j = 0; j < N; j++)
    {
        intrct->TransPtrVect.clear();
        intrct->FillInTransPtrVectForElem(j, 'I');
        counts[j] = (int)intrct->TransPtrVect.size();
        totalCopies += counts[j];
        intrct->EmptyTransPtrVect();
    }

    symData->total_copies = totalCopies;
    symData->sym_counts = new int[N];
    symData->sym_offsets = new int[N + 1];
    symData->point_transforms = new double[totalCopies * 9];
    symData->field_transforms = new double[totalCopies * 9];

    // Build offsets
    symData->sym_offsets[0] = 0;
    for(int j = 0; j < N; j++)
    {
        symData->sym_counts[j] = counts[j];
        symData->sym_offsets[j + 1] = symData->sym_offsets[j] + counts[j];
    }

    // Second pass: extract transforms
    for(int j = 0; j < N; j++)
    {
        intrct->TransPtrVect.clear();
        intrct->FillInTransPtrVectForElem(j, 'I');

        int offset = symData->sym_offsets[j];
        for(int sc = 0; sc < counts[j]; sc++)
        {
            radTrans* trPtr = intrct->TransPtrVect[sc];
            double* pt = &symData->point_transforms[(offset + sc) * 9];
            double* ft = &symData->field_transforms[(offset + sc) * 9];

            // Point inverse transform matrix
            TVector3d zero(0,0,0);
            TVector3d o = trPtr->TrPoint_inv(zero);
            TVector3d ex(1,0,0), ey(0,1,0), ez(0,0,1);
            TVector3d tx = trPtr->TrPoint_inv(ex) - o;
            TVector3d ty = trPtr->TrPoint_inv(ey) - o;
            TVector3d tz = trPtr->TrPoint_inv(ez) - o;

            pt[0] = tx.x; pt[1] = ty.x; pt[2] = tz.x;
            pt[3] = tx.y; pt[4] = ty.y; pt[5] = tz.y;
            pt[6] = tx.z; pt[7] = ty.z; pt[8] = tz.z;

            // Field transform via TrVectField
            TVector3d fx = trPtr->TrVectField(ex);
            TVector3d fy = trPtr->TrVectField(ey);
            TVector3d fz = trPtr->TrVectField(ez);

            ft[0] = fx.x; ft[1] = fy.x; ft[2] = fz.x;
            ft[3] = fx.y; ft[4] = fy.y; ft[5] = fz.y;
            ft[6] = fx.z; ft[7] = fy.z; ft[8] = fz.z;
        }

        intrct->EmptyTransPtrVect();
    }

    // Observation-element quadrature (collocation = one centroid point, so
    // this is packed unconditionally). Must come last: it reads
    // g3dRelaxPtrVect, which the caller replaces with FormalIntrctMemberPtr
    // only after assembly.
    if(!radGPU_PackObsQuadForAsm(intrct, quadData)) {
        // Some element has no volume quadrature -- decline the GPU path. The
        // CPU fallback then reports Warning024 if Galerkin was requested.
        radTSend::WarningMessage("Radia::Warning020");
        return 0;
    }

    return 1;
}

// The eager unpack into radTInteraction::InteractMatrix that used to live here
// is gone (phase 3, studies/ASM_SOLVE_HANDOFF.md): matrix_blocks arrives in the
// solver's SCALAR row-major layout with the row transform
// (MainTransPtrArray[i]->TrMatrix_inv) and the non-finite backstop ALREADY
// applied by the kernel (store_block_rowmajor in radgpu_asm.cu), so the
// interaction now simply keeps this buffer. The same layout read, on demand and
// for the consumers that genuinely need radia's TMatrix3df form, is
// radTInteraction::EnsureInteractMatrix (radintrc.cpp).

void radGPU_FreeObsQuadData(RadGPU_ObsQuadData* qd)
{
    if(!qd) return;
    delete[] qd->q_offsets; qd->q_offsets = nullptr;
    delete[] qd->q_pts;     qd->q_pts = nullptr;
    delete[] qd->q_w;       qd->q_w = nullptr;
    delete[] qd->n_offsets; qd->n_offsets = nullptr;
    delete[] qd->n_pts;     qd->n_pts = nullptr;
    delete[] qd->n_w;       qd->n_w = nullptr;
    delete[] qd->pair_rows; qd->pair_rows = nullptr;
    delete[] qd->pair_cols; qd->pair_cols = nullptr;
    qd->q_total = qd->n_total = qd->n_pairs = 0;
}

void radGPU_FreeSymData(RadGPU_SymData* symData)
{
    if(symData) {
        delete[] symData->sym_counts;   symData->sym_counts = nullptr;
        delete[] symData->sym_offsets;   symData->sym_offsets = nullptr;
        delete[] symData->point_transforms; symData->point_transforms = nullptr;
        delete[] symData->field_transforms; symData->field_transforms = nullptr;
    }
}

#endif // RADIA_WITH_CUDA