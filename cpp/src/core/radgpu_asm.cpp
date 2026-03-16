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
#include "radcast.h"
#include "radrec.h"
#include "radvlpgn.h"
#include "radplnr.h"
#include "radtrans.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// ============================================================
// Helper: extract face data from a polyhedron
// ============================================================
static void ExtractPolyFaceData(
    radTPolyhedron* poly,
    std::vector<double>& face_rot,
    std::vector<double>& face_orig,
    std::vector<double>& face_cz,
    std::vector<int>& edge_offsets,
    std::vector<double>& edge_pts_2d,
    int& n_faces, int& n_edges)
{
    int nf = poly->AmOfFaces;
    n_faces = nf;
    n_edges = 0;

    for(int fi = 0; fi < nf; fi++) {
        radTHandlePgnAndTrans& hpt = poly->VectHandlePgnAndTrans[fi];
        radTPolygon* pgn = hpt.PgnHndl.rep;
        radTrans* tr = hpt.TransHndl.rep;

        // Edge offset
        edge_offsets.push_back((int)edge_pts_2d.size() / 2);

        // Face origin = transform origin (translation part)
        // The transform maps local 2D+z to lab frame
        // We need: origin in lab frame, rotation matrix lab->local
        // TrBiPoint maps local->lab, so rot = TrBiPoint matrix rows
        // But for the kernel we need lab->local, which is the transpose

        // Get the rotation matrix from the transform
        // radTrans stores: TrMatrix (3x3), and offset
        TVector3d origin(0., 0., 0.);
        origin = tr->TrBiPoint(origin);  // origin in lab frame

        face_orig.push_back(origin.x);
        face_orig.push_back(origin.y);
        face_orig.push_back(origin.z);

        // Rotation: lab->local. The transform's TrBiPoint does local->lab.
        // So lab->local is the inverse = transpose of the rotation part.
        // We extract by transforming unit vectors:
        TVector3d ex(1,0,0), ey(0,1,0), ez(0,0,1);
        TVector3d zero(0,0,0);
        TVector3d labEx = tr->TrBiPoint(ex) - origin;
        TVector3d labEy = tr->TrBiPoint(ey) - origin;
        TVector3d labEz = tr->TrBiPoint(ez) - origin;

        // rot[row][col]: rot * (lab - origin) = local
        // rows of rot are labEx, labEy, labEz expressed as how they map lab->local
        // Since TrBiPoint does local->lab, the inverse (lab->local) has rows = columns of TrBiPoint matrix
        // i.e., rot = transpose of [labEx labEy labEz]
        // Row 0 of rot (local x): components are labEx.x, labEy.x, labEz.x
        face_rot.push_back(labEx.x); face_rot.push_back(labEy.y); face_rot.push_back(labEz.z);
        face_rot.push_back(labEx.x); face_rot.push_back(labEy.y); face_rot.push_back(labEz.z);
        face_rot.push_back(labEx.x); face_rot.push_back(labEy.y); face_rot.push_back(labEz.z);

        // Face coord z in local frame
        face_cz.push_back(pgn->CoordZ);

        // Edge points in 2D local frame
        int ne = pgn->AmOfEdgePoints;
        for(int ei = 0; ei < ne; ei++) {
            TVector2d& ep = pgn->EdgePointsVector[ei];
            edge_pts_2d.push_back(ep.x);
            edge_pts_2d.push_back(ep.y);
        }
        n_edges += ne;
    }
    // Final edge offset
    edge_offsets.push_back((int)edge_pts_2d.size() / 2);
}

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

    fprintf(stderr, "GPU asm pack: N=%d, nRec=%d, nPoly=%d\n", N, nRec, nPoly);

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
        fprintf(stderr, "GPU asm: mixed element types not yet supported\n");
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
        fprintf(stderr, "GPU asm: mixed element types not yet supported\n");
        if(nRec == 0) delete[] obsCenters;
        return 0;
    }

    // If no elements recognized
    if(nRec == 0 && nPoly == 0) {
        delete[] obsCenters;
        return 0;
    }

    // --- Symmetry transforms ---
    // Build all symmetry copies from the transform list of element 0
    // In Radia, all elements in an interaction share the same symmetry structure
    // TransPtrVect is populated by FillInTransPtrVectForElem
    memset(symData, 0, sizeof(RadGPU_SymData));

    // Use intrct's own method to enumerate symmetry copies for element 0
    intrct->FillInTransPtrVectForElem(0, 'I');
    int nCopies = (int)intrct->TransPtrVect.size();

    if(nCopies > RADGPU_MAX_SYM_COPIES) {
        fprintf(stderr, "GPU asm: too many symmetry copies (%d > %d)\n", nCopies, RADGPU_MAX_SYM_COPIES);
        intrct->EmptyTransPtrVect();
        return 0;
    }

    symData->n_copies = nCopies;

    // For field transforms: we need to know how the field transforms under each symmetry.
    // In the CPU code, TransPtrVect[i]->TrMatrix(SubMatrix) does the field transform.
    // And TransPtrVect[i]->TrPoint_inv(obs) does the point inverse transform.
    // For the GPU:
    //   point_transform: maps obs point from lab to this copy's frame (TrPoint_inv)
    //   field_transform: maps the computed field back to lab frame (TrMatrix)

    for(int sc = 0; sc < nCopies; sc++) {
        radTrans* trPtr = intrct->TransPtrVect[sc];
        double* pt = &symData->point_transforms[sc * 9];
        double* ft = &symData->field_transforms[sc * 9];

        // Extract point inverse transform matrix
        // TrPoint_inv(P) applies the inverse rotation
        // We extract by transforming unit vectors
        TVector3d zero(0,0,0);
        TVector3d o = trPtr->TrPoint_inv(zero);
        TVector3d ex(1,0,0), ey(0,1,0), ez(0,0,1);
        TVector3d tx = trPtr->TrPoint_inv(ex) - o;
        TVector3d ty = trPtr->TrPoint_inv(ey) - o;
        TVector3d tz = trPtr->TrPoint_inv(ez) - o;

        // Point transform: column-major storage as row-major 3x3
        // P_transformed = M * P, so row i = how unit vector maps
        pt[0] = tx.x; pt[1] = ty.x; pt[2] = tz.x;
        pt[3] = tx.y; pt[4] = ty.y; pt[5] = tz.y;
        pt[6] = tx.z; pt[7] = ty.z; pt[8] = tz.z;

        // Extract field transform matrix
        // TrMatrix(M) transforms a 3x3 matrix: result = R * M * R^T
        // For a vector field: B_lab = R * B_local
        // Extract R by transforming unit matrices
        TMatrix3d unitX(TVector3d(1,0,0), TVector3d(0,0,0), TVector3d(0,0,0));
        TMatrix3d unitY(TVector3d(0,0,0), TVector3d(0,1,0), TVector3d(0,0,0));
        TMatrix3d unitZ(TVector3d(0,0,0), TVector3d(0,0,0), TVector3d(0,0,1));

        // Actually, TrVectField gives us what we need directly:
        TVector3d fx = trPtr->TrVectField(ex);
        TVector3d fy = trPtr->TrVectField(ey);
        TVector3d fz = trPtr->TrVectField(ez);

        ft[0] = fx.x; ft[1] = fy.x; ft[2] = fz.x;
        ft[3] = fx.y; ft[4] = fy.y; ft[5] = fz.y;
        ft[6] = fx.z; ft[7] = fy.z; ft[8] = fz.z;
    }


    // DEBUG: force identity only
    symData->n_copies = 1;
    symData->point_transforms[0] = 1; symData->point_transforms[1] = 0; symData->point_transforms[2] = 0;
    symData->point_transforms[3] = 0; symData->point_transforms[4] = 1; symData->point_transforms[5] = 0;
    symData->point_transforms[6] = 0; symData->point_transforms[7] = 0; symData->point_transforms[8] = 1;
    symData->field_transforms[0] = 1; symData->field_transforms[1] = 0; symData->field_transforms[2] = 0;
    symData->field_transforms[3] = 0; symData->field_transforms[4] = 1; symData->field_transforms[5] = 0;
    symData->field_transforms[6] = 0; symData->field_transforms[7] = 0; symData->field_transforms[8] = 1;
    fprintf(stderr, "DEBUG: forced 1 identity symmetry copy\n");
    //


// DEBUG: dump face data for element 1
    if(nPoly > 0 && nPoly == N) {
        int testElem = 1;
        int fStart = polyData->face_offsets[testElem];
        int fEnd = polyData->face_offsets[testElem + 1];
        fprintf(stderr, "DEBUG elem %d: %d faces, center=%.6f %.6f %.6f\n",
                testElem, fEnd - fStart,
                polyData->centers[3*testElem+0],
                polyData->centers[3*testElem+1],
                polyData->centers[3*testElem+2]);

        for(int fi = fStart; fi < fEnd && fi < fStart + 2; fi++) {
            int eStart = polyData->edge_offsets[fi];
            int eEnd = polyData->edge_offsets[fi + 1];
            fprintf(stderr, "  face %d: %d edges, cz=%.6f, orig=%.6f %.6f %.6f\n",
                    fi - fStart, eEnd - eStart, polyData->face_cz[fi],
                    polyData->face_orig[3*fi+0], polyData->face_orig[3*fi+1], polyData->face_orig[3*fi+2]);
            fprintf(stderr, "    rot: [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]\n",
                    polyData->face_rot[9*fi+0], polyData->face_rot[9*fi+1], polyData->face_rot[9*fi+2],
                    polyData->face_rot[9*fi+3], polyData->face_rot[9*fi+4], polyData->face_rot[9*fi+5],
                    polyData->face_rot[9*fi+6], polyData->face_rot[9*fi+7], polyData->face_rot[9*fi+8]);
            for(int ei = eStart; ei < eEnd; ei++) {
                fprintf(stderr, "    edge %d: %.6f %.6f\n",
                        ei - eStart, polyData->edge_pts_2d[2*ei+0], polyData->edge_pts_2d[2*ei+1]);
            }
        }
    }

    // DEBUG: also dump what Radia sees internally for element 1
    if(nPoly > 0) {
        int testElem = 1;
        radTPolyhedron* poly = Cast.PolyhedronCast(intrct->g3dRelaxPtrVect[testElem]);
        fprintf(stderr, "RADIA elem %d: %d faces, CentrPoint=%.6f %.6f %.6f\n",
                testElem, poly->AmOfFaces,
                poly->CentrPoint.x, poly->CentrPoint.y, poly->CentrPoint.z);

        for(int fi = 0; fi < poly->AmOfFaces && fi < 2; fi++) {
            radTHandlePgnAndTrans& hpt = poly->VectHandlePgnAndTrans[fi];
            radTPolygon* pgn = hpt.PgnHndl.rep;
            radTrans* tr = hpt.TransHndl.rep;

            TVector3d orig(0,0,0);
            orig = tr->TrBiPoint(orig);

            TVector3d ex(1,0,0), ey(0,1,0), ez(0,0,1);
            TVector3d labEx = tr->TrBiPoint(ex) - orig;
            TVector3d labEy = tr->TrBiPoint(ey) - orig;
            TVector3d labEz = tr->TrBiPoint(ez) - orig;

            fprintf(stderr, "  face %d: %d edges, CoordZ=%.6f, orig=%.6f %.6f %.6f\n",
                    fi, pgn->AmOfEdgePoints, pgn->CoordZ,
                    orig.x, orig.y, orig.z);
            fprintf(stderr, "    labEx: %.4f %.4f %.4f\n", labEx.x, labEx.y, labEx.z);
            fprintf(stderr, "    labEy: %.4f %.4f %.4f\n", labEy.x, labEy.y, labEy.z);
            fprintf(stderr, "    labEz: %.4f %.4f %.4f\n", labEz.x, labEz.y, labEz.z);
            for(int ei = 0; ei < pgn->AmOfEdgePoints; ei++) {
                TVector2d& ep = pgn->EdgePointsVector[ei];
                fprintf(stderr, "    edge %d: %.6f %.6f\n", ei, ep.x, ep.y);
            }
        }
    }


    intrct->EmptyTransPtrVect();

    fprintf(stderr, "GPU asm pack: %d symmetry copies\n", nCopies);

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
        for(int j = 0; j < N; j++) {
            long long idx = ((long long)i * N + j) * 9;
            TMatrix3df& m = intrct->InteractMatrix[i][j];
            m.Str0.x = blocks[idx+0]; m.Str0.y = blocks[idx+1]; m.Str0.z = blocks[idx+2];
            m.Str1.x = blocks[idx+3]; m.Str1.y = blocks[idx+4]; m.Str1.z = blocks[idx+5];
            m.Str2.x = blocks[idx+6]; m.Str2.y = blocks[idx+7]; m.Str2.z = blocks[idx+8];
        }
    }
}

#endif // RADIA_WITH_CUDA