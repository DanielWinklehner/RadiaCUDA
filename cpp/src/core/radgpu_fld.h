/************************************************************************//**
 * File: radgpu_fld.h
 * Description: Data structures and declarations for GPU-accelerated
 *              magnetic field evaluation.
 * Project: RadiaCUDA
 * First release: 2026
 *
 * @authors D. Winklehner, Claude
 ***************************************************************************/

#ifndef RADGPU_FLD_H
#define RADGPU_FLD_H

#ifdef RADIA_WITH_CUDA

//-------------------------------------------------------------------------
// Maximum vertices per polygon face.
// Matches the constraint in radgpu_asm: polyhedron faces rarely exceed 6.
// RecMag faces are always quads (4).
//-------------------------------------------------------------------------
#define RADGPU_FLD_MAX_VERTS 8

//-------------------------------------------------------------------------
// Block size for CUDA kernel launch (threads per block).
//-------------------------------------------------------------------------
#define RADGPU_FLD_BLOCK_SIZE 128

//-------------------------------------------------------------------------
// Structure holding packed face data for GPU field evaluation.
// All face geometry is in the LAB frame (symmetries pre-expanded on host).
//-------------------------------------------------------------------------
struct RadGPUFieldFaceData
{
    // Face vertices: flat array [n_faces_total * MAX_VERTS * 3]
    double* h_verts;
    double* d_verts;

    // Number of vertices per face: [n_faces_total]
    int* h_nverts;
    int* d_nverts;

    // Face outward normals: flat array [n_faces_total * 3]
    double* h_normals;
    double* d_normals;

    // Magnetization vector per face (pre-transformed): [n_faces_total * 3]
    double* h_mag;
    double* d_mag;

    // Total number of faces (after symmetry expansion)
    int n_faces_total;

    // Observation points: flat array [n_obs * 3]
    double* h_obs;
    double* d_obs;
    int n_obs;

    // Partial results buffer on device: [n_obs * n_src_blocks * 3]
    double* d_partial_B;
    int n_src_blocks;

    // Final result device buffer: [n_obs * 3]
    double* d_result_B;

    // Final result on host: [n_obs * 3] — points to caller's output array
    double* h_result_B;

    RadGPUFieldFaceData()
        : h_verts(nullptr), d_verts(nullptr)
        , h_nverts(nullptr), d_nverts(nullptr)
        , h_normals(nullptr), d_normals(nullptr)
        , h_mag(nullptr), d_mag(nullptr)
        , n_faces_total(0)
        , h_obs(nullptr), d_obs(nullptr)
        , n_obs(0)
        , d_partial_B(nullptr)
        , n_src_blocks(0)
        , d_result_B(nullptr)
        , h_result_B(nullptr)
    {}
};

//-------------------------------------------------------------------------
// CUDA kernel launch/memory functions (implemented in radgpu_fld.cu)
//-------------------------------------------------------------------------

int radGPU_FldLaunchKernel(RadGPUFieldFaceData* data);
int radGPU_FldAllocAndCopy(RadGPUFieldFaceData* data);
int radGPU_FldRetrieveAndFree(RadGPUFieldFaceData* data);

//-------------------------------------------------------------------------
// Host-side entry point (implemented in radgpu_fld.cpp)
// Called from the RadFld path in radentry.cpp.
//
// Packs geometry from Radia's internal structures, launches GPU kernel
// for magnetized elements, then adds CPU-computed coil contributions.
//
// Returns 0 on success, nonzero on failure (caller should fall back to CPU).
//-------------------------------------------------------------------------
int radGPU_ComputeField(int indObj, double* arCoord, int nP, double* arB);

#endif // RADIA_WITH_CUDA
#endif // RADGPU_FLD_H