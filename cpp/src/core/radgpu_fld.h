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
    // Face vertices in 2D local frame: [n_faces_total * MAX_VERTS * 2]
    double* h_verts2d;
    double* d_verts2d;

    // Number of vertices per face: [n_faces_total]
    int* h_nverts;
    int* d_nverts;

    // Face coordinate z in local frame: [n_faces_total]
    double* h_coordz;
    double* d_coordz;

    // Face transform (local->lab): [n_faces_total * 9] row-major 3x3
    double* h_transform;
    double* d_transform;

    // Face inverse transform (lab->local): [n_faces_total * 9] row-major 3x3
    double* h_inv_transform;
    double* d_inv_transform;

    // Face origin in lab frame: [n_faces_total * 3]
    double* h_origin;
    double* d_origin;

    // Magnetization per face (lab frame): [n_faces_total * 3]
    double* h_mag;
    double* d_mag;

    // Total number of faces
    int n_faces_total;

    // Observation points: [n_obs * 3]
    double* h_obs;
    double* d_obs;
    int n_obs;

    // Partial results: [n_obs * n_src_blocks * 3]
    double* d_partial_B;
    int n_src_blocks;

    // Result: [n_obs * 3]
    double* d_result_B;
    double* h_result_B;

    RadGPUFieldFaceData()
        : h_verts2d(nullptr), d_verts2d(nullptr)
        , h_nverts(nullptr), d_nverts(nullptr)
        , h_coordz(nullptr), d_coordz(nullptr)
        , h_transform(nullptr), d_transform(nullptr)
        , h_inv_transform(nullptr), d_inv_transform(nullptr)
        , h_origin(nullptr), d_origin(nullptr)
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
// Structure holding packed RecMag data for GPU field evaluation.
// Each entry is one RecMag copy (after symmetry expansion).
//-------------------------------------------------------------------------
struct RadGPUFieldRecMagData
{
    double* h_centers;
    double* d_centers;

    double* h_dims;
    double* d_dims;

    double* h_mag;
    double* d_mag;

    // Rotation matrices: flat array [n_recmags * 9]
    // Row-major: rot[row*3+col], transforms local->lab: lab = rot * local
    double* h_rot;
    double* d_rot;

    int n_recmags;

    double* h_obs;
    double* d_obs;
    int n_obs;

    double* d_partial_B;
    int n_src_blocks;

    double* d_result_B;
    double* h_result_B;

    RadGPUFieldRecMagData()
        : h_centers(nullptr), d_centers(nullptr)
        , h_dims(nullptr), d_dims(nullptr)
        , h_mag(nullptr), d_mag(nullptr)
        , h_rot(nullptr), d_rot(nullptr)
        , n_recmags(0)
        , h_obs(nullptr), d_obs(nullptr)
        , n_obs(0)
        , d_partial_B(nullptr)
        , n_src_blocks(0)
        , d_result_B(nullptr)
        , h_result_B(nullptr)
    {}
};

//-------------------------------------------------------------------------
// RecMag kernel launch/memory functions (implemented in radgpu_fld.cu)
//-------------------------------------------------------------------------
int radGPU_FldRecMagAllocAndCopy(RadGPUFieldRecMagData* data);
int radGPU_FldRecMagLaunchKernel(RadGPUFieldRecMagData* data);
int radGPU_FldRecMagRetrieveAndFree(RadGPUFieldRecMagData* data);

//-------------------------------------------------------------------------
// Host-side entry point (implemented in radgpu_fld.cpp)
// Called from the RadFld path in radentry.cpp.
//
// Packs geometry from Radia's internal structures, launches GPU kernel
// for magnetized elements, then adds CPU-computed coil contributions.
//
// Returns 0 on success, nonzero on failure (caller should fall back to CPU).
//-------------------------------------------------------------------------
int radGPU_ComputeField(int indObj, double* arCoord, int nP, double* arB, int use_gpu=1);

#endif // RADIA_WITH_CUDA
#endif // RADGPU_FLD_H
