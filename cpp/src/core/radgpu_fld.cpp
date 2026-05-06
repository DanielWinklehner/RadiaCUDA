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

#ifdef RADIA_WITH_MPI
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

    // Get the full rotation matrix and translation for this chain.
    // Used for RecMag kernel which needs to transform obs points to local frame.
    void GetRotationAndTranslation(double rot[9], double origin[3]) const
    {
        TVector3d ex(1, 0, 0), ey(0, 1, 0), ez(0, 0, 1), zero(0, 0, 0);

        TVector3d labEx = TransformDirection(ex);
        TVector3d labEy = TransformDirection(ey);
        TVector3d labEz = TransformDirection(ez);
        TVector3d labOrigin = TransformPoint(zero);

        // rot transforms local -> lab: lab_vec = rot * local_vec
        // Row-major: rot[row*3 + col]
        rot[0] = labEx.x; rot[1] = labEy.x; rot[2] = labEz.x;
        rot[3] = labEx.y; rot[4] = labEy.y; rot[5] = labEz.y;
        rot[6] = labEx.z; rot[7] = labEy.z; rot[8] = labEz.z;

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
    std::vector<FldRecMagInfo>& recmags);

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
    bool mirrored = false)
{
    if (g3dPtr == nullptr) return;

    // Check if it's a group
    radTGroup* groupPtr = Cast.GroupCast(g3dPtr);
    if (groupPtr != nullptr)
    {
        for (radTmhg::const_iterator iter = groupPtr->GroupMapOfHandlers.begin();
             iter != groupPtr->GroupMapOfHandlers.end(); ++iter)
        {
            radTg3d* childPtr = Cast.g3dCast(((*iter).second).rep);
            if (childPtr != nullptr)
            {
                CollectElementsRecursive(childPtr, Cast, tChain,
                                         faces, recmags, hasCurrentSources, mirrored);
            }
        }
        return;
    }

    // Leaf element.
    // Radia's FlattenSpaceTransforms(vhFlatTrfs) on a leaf returns ALL symmetry transformations
    // of the object relative to the root if called on the root, but here we just want
    // all copies of THIS leaf.
    radTvhg vhFlatTrfs;
    g3dPtr->FlattenSpaceTransforms(vhFlatTrfs);

    std::vector<radTrans*> symCopies;
    if (vhFlatTrfs.empty())
    {
        symCopies.push_back(nullptr);
    }
    else
    {
        for (size_t i = 0; i < vhFlatTrfs.size(); i++)
        {
            radTrans* pTr = (radTrans*)(vhFlatTrfs[i].rep);
            symCopies.push_back(pTr);
        }
    }

    size_t nCopies = symCopies.size();
    for (size_t si = 0; si < nCopies; si++)
    {
        bool nextMirrored = mirrored;
        if (symCopies[si] != nullptr)
        {
            tChain.chain.push_back(symCopies[si]);
            if (symCopies[si]->ShowParity() < 0) nextMirrored = !mirrored;
        }

        // Check if magnetized relaxable
        radTg3dRelax* relaxPtr = Cast.g3dRelaxCast(g3dPtr);
        if (relaxPtr != nullptr)
        {
            // Polyhedron
            radTPolyhedron* poly = Cast.PolyhedronCast(relaxPtr);
            if (poly != nullptr)
            {
                TVector3d localMag = relaxPtr->Magn;
                TVector3d labMag = tChain.TransformField(localMag);
                AddPolyhedronFaces(poly, tChain, labMag, faces, nextMirrored);
            }
            else
            {
                // RecMag
                radTRecMag* rec = Cast.RecMagCast(relaxPtr);
                if (rec != nullptr)
                {
                    if (rec->J_IsNotZero) hasCurrentSources = true;
                    AddRecMag(rec, tChain, recmags);
                }
            }
        }
        else
        {
            hasCurrentSources = true;
        }

        if (symCopies[si] != nullptr) tChain.chain.pop_back();
    }
}

//=========================================================================
// Add a RecMag to the list (store local-frame data + rotation)
//=========================================================================

static void AddRecMag(
    radTRecMag* rec,
    const TransformChain& tChain,
    std::vector<FldRecMagInfo>& recmags)
{
    FldRecMagInfo info;

    // Dimensions in local frame (always axis-aligned)
    info.dims[0] = rec->Dimensions.x;
    info.dims[1] = rec->Dimensions.y;
    info.dims[2] = rec->Dimensions.z;

    // Magnetization in local frame
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
    tChain.GetRotationAndTranslation(info.rot, info.origin);

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
        for (int v = 0; v < nv; v++)
        {
            TVector2d edgePt = pgn->EdgePointsVector[v];
            double coordZ = pgn->CoordZ;
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
        TVector3d e01;
        e01.x = labVerts[1].x - labVerts[0].x;
        e01.y = labVerts[1].y - labVerts[0].y;
        e01.z = labVerts[1].z - labVerts[0].z;
        double len01 = sqrt(e01.x * e01.x + e01.y * e01.y + e01.z * e01.z);
        if (len01 < 1e-30) continue;

        TVector3d ax;  // local X axis
        ax.x = e01.x / len01;
        ax.y = e01.y / len01;
        ax.z = e01.z / len01;

        // Second edge (v0 -> v2)
        TVector3d e02;
        e02.x = labVerts[2].x - labVerts[0].x;
        e02.y = labVerts[2].y - labVerts[0].y;
        e02.z = labVerts[2].z - labVerts[0].z;

        // Normal = e01 × e02
        TVector3d normal;
        normal.x = e01.y * e02.z - e01.z * e02.y;
        normal.y = e01.z * e02.x - e01.x * e02.z;
        normal.z = e01.x * e02.y - e01.y * e02.x;
        double lenN = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (lenN < 1e-30) continue;

        TVector3d az;  // local Z axis = normal direction
        az.x = normal.x / lenN;
        az.y = normal.y / lenN;
        az.z = normal.z / lenN;

        if (mirrored)
        {
            az.x = -az.x; az.y = -az.y; az.z = -az.z;
        }

        // Y axis = Z × X (right-handed)
        TVector3d ay;
        ay.x = az.y * ax.z - az.z * ax.y;
        ay.y = az.z * ax.x - az.x * ax.z;
        ay.z = az.x * ax.y - az.y * ax.x;
        // Should already be normalized since ax and az are orthogonal unit vectors
        // but normalize for safety
        double lenY = sqrt(ay.x * ay.x + ay.y * ay.y + ay.z * ay.z);
        if (lenY < 1e-30) continue;
        ay.x /= lenY; ay.y /= lenY; ay.z /= lenY;

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
        fi.transform[0] = ax.x; fi.transform[1] = ay.x; fi.transform[2] = az.x;
        fi.transform[3] = ax.y; fi.transform[4] = ay.y; fi.transform[5] = az.y;
        fi.transform[6] = ax.z; fi.transform[7] = ay.z; fi.transform[8] = az.z;

        // Inverse: lab->local (transpose)
        fi.inv_transform[0] = ax.x; fi.inv_transform[1] = ax.y; fi.inv_transform[2] = ax.z;
        fi.inv_transform[3] = ay.x; fi.inv_transform[4] = ay.y; fi.inv_transform[5] = ay.z;
        fi.inv_transform[6] = az.x; fi.inv_transform[7] = az.y; fi.inv_transform[8] = az.z;

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

            fi.verts2d[v * 2 + 0] = dpx * ax.x + dpy * ax.y + dpz * ax.z;
            fi.verts2d[v * 2 + 1] = dpx * ay.x + dpy * ay.y + dpz * ay.z;
        }

        // Use the original lab-frame magnetization passed in
        fi.mag[0] = labMag.x;
        fi.mag[1] = labMag.y;
        fi.mag[2] = labMag.z;

        faces.push_back(fi);
    }
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
                // TODO: separate J contribution from RecMag
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

int radGPU_ComputeField(int indObj, double* arCoord, int nP, double* arB)
{
    int mpiRank = 0;
    int mpiSize = 1;
#ifdef RADIA_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#endif

    int gpuSuccess = 0;

    if (mpiRank == 0)
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

        if (gpuSuccess == 0)
        {
            TransformChain tChain;
            CollectElementsRecursive(g3dPtr, Cast, tChain, faces, recmags,
                                     hasCurrentSources, false);
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

//                // DEBUG: dump packed face data
//                fprintf(stderr, "DEBUG: nFaces=%d\n", nFaces);
//                for (int f = 0; f < nFaces; f++)
//                {
//                    fprintf(stderr, "  Face %d: nv=%d coordz=%.6f\n", f, fData.h_nverts[f], fData.h_coordz[f]);
//                    fprintf(stderr, "    origin=[%.6f, %.6f, %.6f]\n",
//                        fData.h_origin[f*3+0], fData.h_origin[f*3+1], fData.h_origin[f*3+2]);
//                    fprintf(stderr, "    mag=[%.6f, %.6f, %.6f]\n",
//                        fData.h_mag[f*3+0], fData.h_mag[f*3+1], fData.h_mag[f*3+2]);
//                    fprintf(stderr, "    transform=\n");
//                    for (int r = 0; r < 3; r++)
//                        fprintf(stderr, "      [%.8f, %.8f, %.8f]\n",
//                            fData.h_transform[f*9+r*3+0], fData.h_transform[f*9+r*3+1], fData.h_transform[f*9+r*3+2]);
//                    fprintf(stderr, "    inv_transform=\n");
//                    for (int r = 0; r < 3; r++)
//                        fprintf(stderr, "      [%.8f, %.8f, %.8f]\n",
//                            fData.h_inv_transform[f*9+r*3+0], fData.h_inv_transform[f*9+r*3+1], fData.h_inv_transform[f*9+r*3+2]);
//                    int vbase = f * RADGPU_FLD_MAX_VERTS * 2;
//                    for (int v = 0; v < fData.h_nverts[f]; v++)
//                        fprintf(stderr, "    v2d_%d=[%.8f, %.8f]\n", v,
//                            fData.h_verts2d[vbase+v*2+0], fData.h_verts2d[vbase+v*2+1]);
//                }

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

#ifdef RADIA_WITH_MPI
    if (mpiSize > 1)
    {
        MPI_Bcast(&gpuSuccess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
#endif

    return gpuSuccess;
}

#endif // RADIA_WITH_CUDA