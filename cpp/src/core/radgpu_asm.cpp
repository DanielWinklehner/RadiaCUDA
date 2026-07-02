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

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// (Removed dead, buggy ExtractPolyFaceData helper: it was never called, and its
//  face_rot fill wrote the same diagonal-only row three times. The live packing in
//  radGPU_PackGeometryForAsm computes face_rot correctly inline. See issue #11.)

// ============================================================
// Pack geometry from Radia interaction data
// ============================================================
int radGPU_PackGeometryForAsm(
    radTInteraction* intrct,
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData)
{
    int N = intrct->AmOfMainElem;
    if(N <= 0) return 0;

    radTCast Cast;

    // --- Classify elements ---
    int nRec = 0, nPoly = 0;
    for(int i = 0; i < N; i++) {
        radTg3dRelax* rel = intrct->g3dRelaxPtrVect[i];
        radTRecMag* recPtr = Cast.RecMagCast(rel);
        radTPolyhedron* polyPtr = Cast.PolyhedronCast(rel);
        if(recPtr) nRec++;
        else if(polyPtr) nPoly++;
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

    // --- RecMag packing ---
    memset(recData, 0, sizeof(RadGPU_RecMagData));
    if(nRec > 0 && nRec == N) {
        recData->n_elem = N;
        recData->centers = new double[3 * N];
        recData->dims = new double[3 * N];
        recData->obs_centers = obsCenters;  // transfer ownership

        for(int i = 0; i < N; i++) {
            radTRecMag* rec = Cast.RecMagCast(intrct->g3dRelaxPtrVect[i]);
            recData->centers[3*i+0] = rec->CentrPoint.x;
            recData->centers[3*i+1] = rec->CentrPoint.y;
            recData->centers[3*i+2] = rec->CentrPoint.z;
            recData->dims[3*i+0] = rec->Dimensions.x;
            recData->dims[3*i+1] = rec->Dimensions.y;
            recData->dims[3*i+2] = rec->Dimensions.z;
        }
    } else if(nRec > 0) {
        radTSend::WarningMessage("Radia::Warning020");
        delete[] obsCenters;
        return 0;
    }

    // --- Polyhedron packing ---
    memset(polyData, 0, sizeof(RadGPU_PolyData));
    if(nPoly > 0 && nPoly == N) {
        // First pass: count faces and edges
        int totalFaces = 0, totalEdges = 0;
        for(int i = 0; i < N; i++) {
            radTPolyhedron* poly = Cast.PolyhedronCast(intrct->g3dRelaxPtrVect[i]);
            totalFaces += poly->AmOfFaces;
            for(int fi = 0; fi < poly->AmOfFaces; fi++) {
                totalEdges += poly->VectHandlePgnAndTrans[fi].PgnHndl.rep->AmOfEdgePoints;
            }
        }

        polyData->n_elem = N;
        polyData->n_faces_total = totalFaces;
        polyData->n_edges_total = totalEdges;
        polyData->centers = obsCenters;  // use transformed obs centers
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
    } else if(nPoly > 0) {
        radTSend::WarningMessage("Radia::Warning020");
        if(nRec == 0) delete[] obsCenters;
        return 0;
    }

    // If no elements recognized
    if(nRec == 0 && nPoly == 0) {
        delete[] obsCenters;
        return 0;
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

    return 1;
}

// ============================================================
// Unpack GPU matrix into Radia's TMatrix3df format
// ============================================================
void radGPU_UnpackMatrix(
    RadGPU_AsmResult* result,
    radTInteraction* intrct)
{
    int N = result->N;
    if(N != intrct->AmOfMainElem) {
        fprintf(stderr, "GPU asm unpack: N mismatch (%d vs %d)\n", N, intrct->AmOfMainElem);
        return;
    }

    float* blocks = result->matrix_blocks;
    for(int i = 0; i < N; i++) {
        // Observation-row element's first-copy transform. The CPU assembly finalizes
        // each block with MainTransPtrArray[StrNo]->TrMatrix_inv (radintrc.cpp:575);
        // the GPU kernel omits it, so apply it here on unpack. For rows whose element
        // has no base transform MainTransPtrArray[i] is the identity and this is a
        // no-op (the common case: pure TrfZerPara/Perp symmetry). Issue #6.
        radTrans* rowTrans = intrct->MainTransPtrArray[i];
        for(int j = 0; j < N; j++) {
            long long idx = ((long long)i * N + j) * 9;
            TMatrix3d block(
                TVector3d(blocks[idx+0], blocks[idx+1], blocks[idx+2]),
                TVector3d(blocks[idx+3], blocks[idx+4], blocks[idx+5]),
                TVector3d(blocks[idx+6], blocks[idx+7], blocks[idx+8]));
            if(rowTrans != 0) rowTrans->TrMatrix_inv(block);
            intrct->InteractMatrix[i][j] = block;
        }
    }
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