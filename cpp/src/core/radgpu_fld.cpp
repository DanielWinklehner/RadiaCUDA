/************************************************************************//**
 * File: radgpu_fld.cpp
 * Description: Host-side geometry packing for GPU-accelerated field eval.
 * Project: RadiaCUDA
 * First release: 2026
 *
 * @authors D. Winklehner, Claude
 ***************************************************************************/

#ifdef RADIA_WITH_CUDA

#include "radgpu_fld.h"
#include "radappl.h"
#include "radcast.h"
#include "radg3d.h"
#include "radg3dgr.h"
#include "radrec.h"
#include "radvlpgn.h"
#include "radplnr.h"
#include "radgroup.h"
#include "radtrans.h"

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdio>

#ifdef _WITH_MPI
#include <mpi.h>
#endif

//=========================================================================
// Internal data structures
//=========================================================================

struct FldFaceInfo
{
    double verts2d[RADGPU_FLD_MAX_VERTS * 2];  // 2D vertices in face local frame
    int nverts;
    double coordz;         // z coordinate of polygon in face local frame
    double transform[9];   // local->lab rotation (row-major)
    double inv_transform[9]; // lab->local rotation (row-major)
    double origin[3];      // face origin in lab frame
    double mag[3];         // magnetization in lab frame
};

struct FldRecMagInfo
{
    double center[3];   // center in lab frame
    double dims[3];     // dimensions in LOCAL frame (always axis-aligned)
    double mag[3];      // magnetization in LOCAL frame
    double rot[9];      // rotation matrix: lab_vec = rot * local_vec (row-major)
    double origin[3];   // translation: lab_point = rot * local_point + origin
};

//=========================================================================
// Transform chain — applies transforms sequentially using Radia's methods
//=========================================================================

struct TransformChain
{
    std::vector<radTrans*> chain;  // outer-to-inner order

    TVector3d TransformPoint(const TVector3d& p) const
    {
        TVector3d result = p;
        for (int i = (int)chain.size() - 1; i >= 0; i--)
        {
            if (chain[i] != nullptr)
                result = chain[i]->TrPoint(result);
        }
        return result;
    }

    TVector3d TransformDirection(const TVector3d& v) const
    {
        TVector3d zero(0, 0, 0);
        TVector3d result_v = v;
        TVector3d result_z = zero;
        for (int i = (int)chain.size() - 1; i >= 0; i--)
        {
            if (chain[i] != nullptr)
            {
                result_v = chain[i]->TrBiPoint(result_v);
                result_z = chain[i]->TrBiPoint(result_z);
            }
        }
        result_v.x -= result_z.x;
        result_v.y -= result_z.y;
        result_v.z -= result_z.z;
        return result_v;
    }

    TVector3d TransformField(const TVector3d& v) const
    {
        TVector3d result = v;
        for (int i = (int)chain.size() - 1; i >= 0; i--)
        {
            if (chain[i] != nullptr)
                result = chain[i]->TrVectField(result);
        }
        return result;
    }

    TVector3d TransformFieldAdjusted(const TVector3d& v, bool mirrored) const
    {
        TVector3d result = v;
        for (int i = (int)chain.size() - 1; i >= 0; i--)
        {
            if (chain[i] != nullptr)
                result = chain[i]->TrVectField(result);
        }
        // Radia's TrVectField for mirror (det=-1) is a simple reflection (s=1).
        // For pseudovectors like B-field (and magnetization), we need an extra minus sign
        // when parity is negative.
        if (mirrored) { result.x = -result.x; result.y = -result.y; result.z = -result.z; }
        return result;
    }

    // Get the full rotation matrix and translation for this chain.
    // Used for RecMag kernel which needs to transform obs points to local frame.
    void GetRotationAndTranslation(double rot[9], double origin[3], bool mirrored) const
    {
        TVector3d ex(1, 0, 0), ey(0, 1, 0), ez(0, 0, 1), zero(0, 0, 0);

        TVector3d labEx = TransformDirection(ex);
        TVector3d labEy = TransformDirection(ey);
        TVector3d labEz = TransformDirection(ez);

        if (mirrored)
        {
            labEx.x = -labEx.x; labEx.y = -labEx.y; labEx.z = -labEx.z;
            labEy.x = -labEy.x; labEy.y = -labEy.y; labEy.z = -labEy.z;
            labEz.x = -labEz.x; labEz.y = -labEz.y; labEz.z = -labEz.z;
        }

        TVector3d labOrigin = TransformPoint(zero);

        // Precompute values to avoid repeating struct field accesses
        double lExx = labEx.x, lExy = labEx.y, lExz = labEx.z;
        double lEyx = labEy.x, lEyy = labEy.y, lEyz = labEy.z;
        double lEzx = labEz.x, lEzy = labEz.y, lEzz = labEz.z;

        // rot transforms local -> lab: lab_vec = rot * local_vec
        // Row-major: rot[row*3 + col]
        rot[0] = lExx; rot[1] = lEyx; rot[2] = lEzx;
        rot[3] = lExy; rot[4] = lEyy; rot[5] = lEzy;
        rot[6] = lExz; rot[7] = lEyz; rot[8] = lEzz;

        origin[0] = labOrigin.x;
        origin[1] = labOrigin.y;
        origin[2] = labOrigin.z;
    }
};

//=========================================================================
// Forward declarations
//=========================================================================

static void CollectElementsRecursive(
    radTg3d* g3dPtr,
    radTCast& Cast,
    TransformChain& tChain,
    std::vector<FldFaceInfo>& faces,
    std::vector<FldRecMagInfo>& recmags,
    bool& hasCurrentSources,
    bool& hasUnsupported,
    bool mirrored);

static void AddPolyhedronFaces(
    radTPolyhedron* poly,
    const TransformChain& tChain,
    const TVector3d& labMag,
    std::vector<FldFaceInfo>& faces,
    bool mirrored);

static void AddRecMag(
    radTRecMag* rec,
    const TransformChain& tChain,
    std::vector<FldRecMagInfo>& recmags,
    bool mirrored);

static bool TryAddPolyhedronFromConvertible(
    radTg3dRelax* relaxPtr,
    radTCast& Cast,
    const TransformChain& tChain,
    bool mirrored,
    std::vector<FldFaceInfo>& faces);

//=========================================================================
// Recursive tree walk
//=========================================================================

static void CollectElementsRecursive(
    radTg3d* g3dPtr,
    radTCast& Cast,
    TransformChain& tChain,
    std::vector<FldFaceInfo>& faces,
    std::vector<FldRecMagInfo>& recmags,
    bool& hasCurrentSources,
    bool& hasUnsupported,
    bool mirrored = false)
{
    if (g3dPtr == nullptr) return;

    // We process each transform in the current object's transform list
    // Radia iterates over g3dListOfTransform from end to beginning (reverse_iterator)
    // to match the order of application.

    radTvhg vhFlatTrfs;
    g3dPtr->FlattenSpaceTransforms(vhFlatTrfs);

    if (g3dPtr->g3dListOfTransform.empty() || (vhFlatTrfs.size() == 1 && ((radTrans*)(vhFlatTrfs[0].rep))->IsIdent(1e-12)))
    {
        // No real transforms on this object.
        // If it is a group, recurse. If it is a leaf, collect.
        radTGroup* groupPtr = Cast.GroupCast(g3dPtr);
        if (groupPtr != nullptr)
        {
            for (radTmhg::const_iterator iter = groupPtr->GroupMapOfHandlers.begin();
                 iter != groupPtr->GroupMapOfHandlers.end(); ++iter)
            {
                radTg3d* childPtr = Cast.g3dCast(((*iter).second).rep);
                if (childPtr != nullptr)
                    CollectElementsRecursive(childPtr, Cast, tChain, faces, recmags, hasCurrentSources, hasUnsupported, mirrored);
            }
        }
        else
        {
            // Leaf
            radTg3dRelax* relaxPtr = Cast.g3dRelaxCast(g3dPtr);
            if (relaxPtr != nullptr)
            {
                radTPolyhedron* poly = Cast.PolyhedronCast(relaxPtr);
                if (poly != nullptr)
                {
                    TVector3d localMag = relaxPtr->Magn;
                    TVector3d labMag = tChain.TransformFieldAdjusted(localMag, mirrored);
                    AddPolyhedronFaces(poly, tChain, labMag, faces, mirrored);
                }
                else
                {
                    radTRecMag* rec = Cast.RecMagCast(relaxPtr);
                    if (rec != nullptr)
                    {
                        if (rec->J_IsNotZero)
                        {
                            // Current-carrying RecMag (ObjRecCur): a fixed Biot-Savart
                            // source with zero magnetization. If it is not under any
                            // symmetry/transform, add its current field on the CPU via
                            // ComputeCoilFieldCPU (below) while the magnetics stay on the
                            // GPU (issue #5). If it IS under a transform, that CPU coil
                            // helper (single-copy B_comp) would drop the symmetry copies,
                            // so fall back to the full CPU field path instead.
                            if (!tChain.chain.empty() || mirrored) hasUnsupported = true;
                            else                                   hasCurrentSources = true;
                        }
                        else
                        {
                            AddRecMag(rec, tChain, recmags, mirrored);
                        }
                    }
                    else
                    {
                        // Relaxable leaf that is neither a polyhedron nor a RecMag
                        // (e.g. an extruded polygon, radTExtrPolygon). Decompose it
                        // to a polyhedron (2 caps + mantle faces) and feed its faces
                        // to the GPU kernel (issue #2); if it cannot be converted,
                        // fall back to the CPU field path.
                        if (!TryAddPolyhedronFromConvertible(relaxPtr, Cast, tChain, mirrored, faces))
                            hasUnsupported = true;
                    }
                }
            }
            else
            {
                // Non-relaxable leaf = current source (arc, filament, racetrack
                // segment). ComputeCoilFieldCPU adds it on the CPU but is not
                // symmetry-aware, so route it there only when it is not under a
                // symmetry/transform; otherwise fall back to the full CPU field path.
                if (!tChain.chain.empty() || mirrored) hasUnsupported = true;
                else                                   hasCurrentSources = true;
            }
        }
    }
    else
    {
        // We need to handle the product of symmetries.
        // Radia's NestedFor_B handles this by recursing through the list of transforms.
        // For simplicity, we can use FlattenSpaceTransforms which already expands the product.

        for (size_t si = 0; si < vhFlatTrfs.size(); si++)
        {
            radTrans* pTr = (radTrans*)(vhFlatTrfs[si].rep);
            bool nextMirrored = mirrored;
            bool pushTr = false;

            if (pTr != nullptr && !pTr->IsIdent(1e-12))
            {
                tChain.chain.push_back(pTr);
                if (pTr->ShowParity() < 0) nextMirrored = !mirrored;
                pushTr = true;
            }

            // Temporarily clear the transforms of g3dPtr to avoid infinite recursion
            // and call CollectElementsRecursive again as if it had no transforms.
            radTlphg savedTrfs;
            savedTrfs.splice(savedTrfs.begin(), g3dPtr->g3dListOfTransform);

            CollectElementsRecursive(g3dPtr, Cast, tChain, faces, recmags, hasCurrentSources, hasUnsupported, nextMirrored);

            g3dPtr->g3dListOfTransform.splice(g3dPtr->g3dListOfTransform.begin(), savedTrfs);

            if (pushTr) tChain.chain.pop_back();
        }
    }
}

//=========================================================================
// Add a RecMag to the list (store local-frame data + rotation)
//=========================================================================

static void AddRecMag(
    radTRecMag* rec,
    const TransformChain& tChain,
    std::vector<FldRecMagInfo>& recmags,
    bool mirrored)
{
    FldRecMagInfo info;

    // Dimensions in local frame (always axis-aligned)
    info.dims[0] = rec->Dimensions.x;
    info.dims[1] = rec->Dimensions.y;
    info.dims[2] = rec->Dimensions.z;

    // Magnetization in the block's LOCAL frame (issue #1). The RecMag kernel
    // evaluates the box field in local coordinates and rotates the result into the
    // lab frame via info.rot, so it needs the LOCAL magnetization, not the lab one.
    // info.rot already encodes rotation AND mirror parity (rot = -Mtx for a
    // reflection, det +1), so rot*Magn reproduces the correct effective lab
    // magnetization for both rotated and mirror-symmetric copies. Passing the
    // pre-rotated lab magnetization here would rotate it a second time.
    info.mag[0] = rec->Magn.x;
    info.mag[1] = rec->Magn.y;
    info.mag[2] = rec->Magn.z;

    // The RecMag's own center is in its "parent" frame.
    // We need the rotation and translation that maps local to lab.
    // But the RecMag center is part of the local frame geometry.
    // The kernel needs: obs_local = rot^T * (obs_lab - center_lab)
    // where center_lab = tChain.TransformPoint(CentrPoint)
    //
    // For the kernel: we store the lab-frame center and orientation.
    // obs_local = rot^T * (obs_lab - lab_center)
    // where lab_center = tChain.TransformPoint(rec->CentrPoint)

    TVector3d labCenter = tChain.TransformPoint(rec->CentrPoint);
    info.center[0] = labCenter.x;
    info.center[1] = labCenter.y;
    info.center[2] = labCenter.z;

    // Rotation matrix and origin from chain
    // We need rot such that: lab_vec = rot * local_vec
    // tChain.GetRotationAndTranslation gives rot: lab = rot * local
    // The kernel uses rot^T to go from lab to local.
    tChain.GetRotationAndTranslation(info.rot, info.origin, mirrored);

    recmags.push_back(info);
}

//=========================================================================
// Add polygon faces from a Polyhedron (unchanged from before)
//=========================================================================

static void AddPolyhedronFaces(
    radTPolyhedron* poly,
    const TransformChain& tChain,
    const TVector3d& labMag,
    std::vector<FldFaceInfo>& faces,
    bool mirrored)
{
    for (int f = 0; f < (int)poly->VectHandlePgnAndTrans.size(); f++)
    {
        radTHandlePgnAndTrans& hpt = poly->VectHandlePgnAndTrans[f];
        radTPolygon* pgn = (radTPolygon*)(hpt.PgnHndl.rep);
        if (pgn == nullptr) continue;

        radTrans* faceTr = (radTrans*)(hpt.TransHndl.rep);

        int nv = pgn->AmOfEdgePoints;
        if (nv < 3 || nv > RADGPU_FLD_MAX_VERTS) continue;

        //------------------------------------------------------------------
        // Step 1: Get 3D lab-frame vertices (already verified correct)
        //------------------------------------------------------------------
        TVector3d labVerts[RADGPU_FLD_MAX_VERTS];
        double coordZ = pgn->CoordZ;
        for (int v = 0; v < nv; v++)
        {
            TVector2d edgePt = pgn->EdgePointsVector[v];
            TVector3d localPt(edgePt.x, edgePt.y, coordZ);
            TVector3d polyPt = (faceTr != nullptr) ? faceTr->TrPoint(localPt) : localPt;
            labVerts[v] = tChain.TransformPoint(polyPt);
        }

        //------------------------------------------------------------------
        // Step 2: Build face-local coordinate system from 3D vertices
        //         (matching CuPy flatten approach)
        //         Origin = first vertex
        //         X axis = along first edge (v0 -> v1), normalized
        //         Normal = from cross product of first two edges
        //         Y axis = normal × X (to form right-handed frame)
        //------------------------------------------------------------------
        TVector3d origin = labVerts[0];

        // First edge
        double e01x = labVerts[1].x - origin.x;
        double e01y = labVerts[1].y - origin.y;
        double e01z = labVerts[1].z - origin.z;
        double len01 = sqrt(e01x * e01x + e01y * e01y + e01z * e01z);
        if (len01 < 1e-30) continue;

        double axx = e01x / len01;
        double axy = e01y / len01;
        double axz = e01z / len01;

        // Second edge (v0 -> v2)
        double e02x = labVerts[2].x - origin.x;
        double e02y = labVerts[2].y - origin.y;
        double e02z = labVerts[2].z - origin.z;

        // Normal = e01 × e02
        double nx = e01y * e02z - e01z * e02y;
        double ny = e01z * e02x - e01x * e02z;
        double nz = e01x * e02y - e01y * e02x;
        double lenN = sqrt(nx * nx + ny * ny + nz * nz);
        if (lenN < 1e-30) continue;

        double azx = nx / lenN;
        double azy = ny / lenN;
        double azz = nz / lenN;

        // Y axis = Z × X (right-handed)
        double ayx = azy * axz - azz * axy;
        double ayy = azz * axx - azx * axz;
        double ayz = azx * axy - azy * axx;
        // Should already be normalized since ax and az are orthogonal unit vectors
        // but normalize for safety
        double lenY = sqrt(ayx * ayx + ayy * ayy + ayz * ayz);
        if (lenY < 1e-30) continue;
        ayx /= lenY; ayy /= lenY; ayz /= lenY;

        //------------------------------------------------------------------
        // Step 3: Build transform matrices
        //         Transform (local->lab): columns are ax, ay, az
        //         Inverse (lab->local): rows are ax, ay, az (transpose)
        //------------------------------------------------------------------
        FldFaceInfo fi;
        memset(&fi, 0, sizeof(fi));
        fi.nverts = nv;
        fi.coordz = 0.0;  // polygon is at z=0 in our local frame (origin at first vertex)

        // Transform: local->lab (column-major stored as row-major)
        // T * [lx, ly, lz]^T = lx*ax + ly*ay + lz*az
        // Row-major: T[row][col] = T[row*3+col]
        fi.transform[0] = axx; fi.transform[1] = ayx; fi.transform[2] = azx;
        fi.transform[3] = axy; fi.transform[4] = ayy; fi.transform[5] = azy;
        fi.transform[6] = axz; fi.transform[7] = ayz; fi.transform[8] = azz;

        // Inverse: lab->local (transpose)
        fi.inv_transform[0] = axx; fi.inv_transform[1] = axy; fi.inv_transform[2] = axz;
        fi.inv_transform[3] = ayx; fi.inv_transform[4] = ayy; fi.inv_transform[5] = ayz;
        fi.inv_transform[6] = azx; fi.inv_transform[7] = azy; fi.inv_transform[8] = azz;

        fi.origin[0] = origin.x;
        fi.origin[1] = origin.y;
        fi.origin[2] = origin.z;

        //------------------------------------------------------------------
        // Step 4: Project 3D vertices to 2D local frame
        //         For each vertex: dp = vert - origin, then
        //         x_local = dp · ax, y_local = dp · ay
        //         (z_local should be ~0 for all vertices)
        //------------------------------------------------------------------
        for (int v = 0; v < nv; v++)
        {
            double dpx = labVerts[v].x - origin.x;
            double dpy = labVerts[v].y - origin.y;
            double dpz = labVerts[v].z - origin.z;

            fi.verts2d[v * 2 + 0] = dpx * axx + dpy * axy + dpz * axz;
            fi.verts2d[v * 2 + 1] = dpx * ayx + dpy * ayy + dpz * ayz;
        }

        // Use the original lab-frame magnetization passed in
        fi.mag[0] = labMag.x;
        fi.mag[1] = labMag.y;
        fi.mag[2] = labMag.z;

        faces.push_back(fi);
    }
}

//=========================================================================
// Convert a relaxable leaf that supports ConvertToPolyhedron (e.g. an extruded
// polygon, radTExtrPolygon) into a temporary polyhedron and collect its faces
// for the GPU polygon kernel. Returns false if the type cannot be converted, in
// which case the caller falls back to CPU. Issue #2.
//=========================================================================

static bool TryAddPolyhedronFromConvertible(
    radTg3dRelax* relaxPtr,
    radTCast& Cast,
    const TransformChain& tChain,
    bool mirrored,
    std::vector<FldFaceInfo>& faces)
{
    // ConvertToPolyhedron is a no-op returning 0 for types that don't support it
    // (radg3d.h default), and a pure in-memory conversion for radTExtrPolygon
    // (radexpgn.cpp) with no global side effects. The radThg handle owns the
    // temporary polyhedron and frees it on scope exit.
    radThg polyHandle;
    if (!relaxPtr->ConvertToPolyhedron(polyHandle, nullptr, 0)) return false;
    if (polyHandle.rep == nullptr) return false;

    radTg3dRelax* polyRelax = Cast.g3dRelaxCast((radTg3d*)(polyHandle.rep));
    radTPolyhedron* poly = (polyRelax != nullptr) ? Cast.PolyhedronCast(polyRelax) : nullptr;
    if (poly == nullptr) return false;

    // The object's own transforms were spliced into tChain before recursing to
    // this leaf, so relaxPtr->Magn is the local magnetization; adjust to the lab
    // frame exactly as for a native polyhedron.
    TVector3d labMag = tChain.TransformFieldAdjusted(relaxPtr->Magn, mirrored);
    AddPolyhedronFaces(poly, tChain, labMag, faces, mirrored);
    return true;
}

//=========================================================================
// Compute field from current-carrying objects on CPU
//=========================================================================

static void ComputeCoilFieldCPU(
    radTg3d* topObj,
    radTCast& Cast,
    double* arCoord, int nP,
    double* arB_additive)
{
    struct StackEntry { radTg3d* ptr; };
    std::vector<StackEntry> stack;
    stack.push_back({topObj});

    while (!stack.empty())
    {
        StackEntry entry = stack.back();
        stack.pop_back();

        radTg3d* obj = entry.ptr;
        if (obj == nullptr) continue;

        radTGroup* grp = Cast.GroupCast(obj);
        if (grp != nullptr)
        {
            for (radTmhg::const_iterator it = grp->GroupMapOfHandlers.begin();
                 it != grp->GroupMapOfHandlers.end(); ++it)
            {
                radTg3d* child = Cast.g3dCast(((*it).second).rep);
                if (child != nullptr) stack.push_back({child});
            }
            continue;
        }

        // Skip relaxable objects (handled by GPU)
        radTg3dRelax* relaxPtr = Cast.g3dRelaxCast(obj);
        if (relaxPtr != nullptr)
        {
            radTRecMag* rec = Cast.RecMagCast(relaxPtr);
            if (rec != nullptr && rec->J_IsNotZero)
            {
                // Current-carrying RecMag (ObjRecCur): fixed Biot-Savart source with
                // zero magnetization. Add its current field on the CPU, exactly like the
                // non-relaxable coil objects below. Only reached for directly-placed
                // current RecMags (symmetrized ones force a full CPU fallback upstream),
                // so single-copy B_comp is exact and no transform handling is needed.
                for (int ip = 0; ip < nP; ip++)
                {
                    TVector3d obsP(arCoord[ip * 3], arCoord[ip * 3 + 1], arCoord[ip * 3 + 2]);
                    radTField field;
                    memset(&field, 0, sizeof(field));
                    field.P = obsP;
                    field.FieldKey.B_ = 1;
                    rec->B_comp(&field);
                    arB_additive[ip * 3 + 0] += field.B.x;
                    arB_additive[ip * 3 + 1] += field.B.y;
                    arB_additive[ip * 3 + 2] += field.B.z;
                }
            }
            continue;
        }

        // Non-relaxable leaf = current source
        for (int ip = 0; ip < nP; ip++)
        {
            TVector3d obsP(arCoord[ip * 3], arCoord[ip * 3 + 1], arCoord[ip * 3 + 2]);

            radTField field;
            memset(&field, 0, sizeof(field));
            field.P = obsP;
            field.FieldKey.B_ = 1;

            obj->B_comp(&field);

            arB_additive[ip * 3 + 0] += field.B.x;
            arB_additive[ip * 3 + 1] += field.B.y;
            arB_additive[ip * 3 + 2] += field.B.z;
        }
    }
}

//=========================================================================
// MAIN ENTRY POINT
//=========================================================================

int radGPU_ComputeField(int indObj, double* arCoord, int nP, double* arB, int use_gpu)
{
    int mpiRank = 0;
    int mpiSize = 1;
#ifdef _WITH_MPI
    // Only query MPI when it has actually been initialized (e.g. via UtiMPI('on')
    // or mpi4py). This lets the GPU field path run standalone (plain `python`,
    // no mpiexec / no UtiMPI) without tripping "MPI routine before init". When MPI
    // is not initialized we run serially as rank 0 of 1.
    int mpiInited = 0;
    MPI_Initialized(&mpiInited);
    if (mpiInited)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    }
#endif

    int gpuSuccess = 0;

    if (use_gpu && mpiRank == 0)
    {
        extern radTApplication rad;
        radTApplication* pApp = &rad;

        radTmhg::const_iterator iter = pApp->GlobalMapOfHandlers.find(indObj);
        if (iter == pApp->GlobalMapOfHandlers.end())
        {
            gpuSuccess = -1;
        }

        radTCast Cast;
        radTg3d* g3dPtr = nullptr;
        if (gpuSuccess == 0)
        {
            radThg hg = (*iter).second;
            g3dPtr = Cast.g3dCast(hg.rep);
            if (g3dPtr == nullptr) gpuSuccess = -1;
        }

        std::vector<FldFaceInfo> faces;
        std::vector<FldRecMagInfo> recmags;
        bool hasCurrentSources = false;
        bool hasUnsupported = false;

        if (gpuSuccess == 0)
        {
            TransformChain tChain;
            CollectElementsRecursive(g3dPtr, Cast, tChain, faces, recmags,
                                     hasCurrentSources, hasUnsupported, false);
        }

        if (gpuSuccess == 0 && hasUnsupported)
        {
            // The model contains element types the GPU field path cannot evaluate
            // correctly (extruded polygons -> issue #2, current-carrying RecMag ->
            // issue #5). Signal failure so RadFld falls back to the full CPU field
            // computation rather than returning a wrong/partial result. Under MPI
            // this gpuSuccess is broadcast below, so every rank falls back together.
            fprintf(stderr, "radGPU_Fld: unsupported element type on GPU path, falling back to CPU\n");
            gpuSuccess = -1;
        }

        if (gpuSuccess == 0)
        {
            int nFaces = (int)faces.size();
            int nRecMags = (int)recmags.size();

            // Initialize output to zero
            memset(arB, 0, (size_t)nP * 3 * sizeof(double));

            //-----------------------------------------------------------------
            // GPU: RecMag kernel
            //-----------------------------------------------------------------
            if (nRecMags > 0)
            {
                RadGPUFieldRecMagData rmData;
                rmData.n_recmags = nRecMags;
                rmData.n_obs = nP;
                rmData.n_src_blocks = (nRecMags + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE;

                rmData.h_centers = new double[(size_t)nRecMags * 3];
                rmData.h_dims = new double[(size_t)nRecMags * 3];
                rmData.h_mag = new double[(size_t)nRecMags * 3];
                rmData.h_obs = arCoord;
                rmData.h_result_B = arB;
                rmData.h_rot = new double[(size_t)nRecMags * 9];

                for (int r = 0; r < nRecMags; r++)
                {
                    const FldRecMagInfo& ri = recmags[r];
                    memcpy(&rmData.h_centers[r * 3], ri.center, 3 * sizeof(double));
                    memcpy(&rmData.h_dims[r * 3], ri.dims, 3 * sizeof(double));
                    memcpy(&rmData.h_mag[r * 3], ri.mag, 3 * sizeof(double));
                    memcpy(&rmData.h_rot[r * 9], ri.rot, 9 * sizeof(double));
                }

                int rc = radGPU_FldRecMagAllocAndCopy(&rmData);
                if (rc == 0) rc = radGPU_FldRecMagLaunchKernel(&rmData);
                if (rc == 0) rc = radGPU_FldRecMagRetrieveAndFree(&rmData);

                delete[] rmData.h_centers;
                delete[] rmData.h_dims;
                delete[] rmData.h_mag;
                delete[] rmData.h_rot;

                if (rc != 0) gpuSuccess = -1;
            }

            //-----------------------------------------------------------------
            // GPU: Polygon face kernel (for polyhedra)
            //-----------------------------------------------------------------
            if (gpuSuccess == 0 && nFaces > 0)
            {
                RadGPUFieldFaceData fData;
                fData.n_faces_total = nFaces;
                fData.n_obs = nP;
                fData.n_src_blocks = (nFaces + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE;

                size_t v2d_count = (size_t)nFaces * RADGPU_FLD_MAX_VERTS * 2;
                fData.h_verts2d = new double[v2d_count]();
                fData.h_nverts = new int[nFaces];
                fData.h_coordz = new double[nFaces];
                fData.h_transform = new double[(size_t)nFaces * 9];
                fData.h_inv_transform = new double[(size_t)nFaces * 9];
                fData.h_origin = new double[(size_t)nFaces * 3];
                fData.h_mag = new double[(size_t)nFaces * 3];
                fData.h_obs = arCoord;

                double* polyB = new double[(size_t)nP * 3]();
                fData.h_result_B = polyB;

                for (int f = 0; f < nFaces; f++)
                {
                    const FldFaceInfo& fi = faces[f];
                    fData.h_nverts[f] = fi.nverts;
                    fData.h_coordz[f] = fi.coordz;

                    int vbase = f * RADGPU_FLD_MAX_VERTS * 2;
                    memcpy(&fData.h_verts2d[vbase], fi.verts2d, fi.nverts * 2 * sizeof(double));

                    memcpy(&fData.h_transform[f * 9], fi.transform, 9 * sizeof(double));
                    memcpy(&fData.h_inv_transform[f * 9], fi.inv_transform, 9 * sizeof(double));
                    memcpy(&fData.h_origin[f * 3], fi.origin, 3 * sizeof(double));
                    memcpy(&fData.h_mag[f * 3], fi.mag, 3 * sizeof(double));
                }


               int rc = radGPU_FldAllocAndCopy(&fData);
               if (rc == 0) rc = radGPU_FldLaunchKernel(&fData);
               if (rc == 0) rc = radGPU_FldRetrieveAndFree(&fData);

                // Add polygon contribution to total
                if (rc == 0)
                {
                    for (int i = 0; i < nP * 3; i++)
                        arB[i] += polyB[i];
                }

                delete[] fData.h_verts2d;
                delete[] fData.h_nverts;
                delete[] fData.h_coordz;
                delete[] fData.h_transform;
                delete[] fData.h_inv_transform;
                delete[] fData.h_origin;
                delete[] fData.h_mag;
                delete[] polyB;

                if (rc != 0) gpuSuccess = -1;
            }

            //-----------------------------------------------------------------
            // CPU: add field from current-carrying objects
            //-----------------------------------------------------------------
            if (gpuSuccess == 0 && hasCurrentSources)
            {
                ComputeCoilFieldCPU(g3dPtr, Cast, arCoord, nP, arB);
            }
        }
    }
    else
    {
        memset(arB, 0, (size_t)nP * 3 * sizeof(double));
    }

#ifdef _WITH_MPI
    if (mpiInited && mpiSize > 1)
    {
        MPI_Bcast(&gpuSuccess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
#endif

    return gpuSuccess;
}

#endif // RADIA_WITH_CUDA