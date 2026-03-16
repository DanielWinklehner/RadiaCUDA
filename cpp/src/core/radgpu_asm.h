/*-------------------------------------------------------------------------
*
* File name:      radgpu_asm.h
*
* Project:        RADIA
*
* Description:    GPU-accelerated interaction matrix assembly
*
-------------------------------------------------------------------------*/

#ifndef __RADGPU_ASM_H
#define __RADGPU_ASM_H

#ifdef RADIA_WITH_CUDA

// Maximum symmetry copies: 2^MAX_SYM_PLANES
#define RADGPU_MAX_SYM_COPIES 64

// ============================================================
// Flat geometry for GPU: polyhedron elements
// ============================================================
struct RadGPU_PolyData {
    int n_elem;
    int n_faces_total;
    int n_edges_total;

    // Per-element
    double* centers;        // [3 * n_elem] element centers
    int* face_offsets;      // [n_elem + 1] CSR into face arrays

    // Per-face
    int* edge_offsets;      // [n_faces_total + 1] CSR into edge arrays
    double* face_cz;        // [n_faces_total] z-coord in local frame
    double* face_rot;       // [9 * n_faces_total] rotation matrices (row-major)
    double* face_orig;      // [3 * n_faces_total] face origins

    // Per-edge
    double* edge_pts_2d;    // [2 * n_edges_total] 2D edge vertices
};

// ============================================================
// Flat geometry for GPU: RecMag elements
// ============================================================
struct RadGPU_RecMagData {
    int n_elem;
    double* centers;        // [3 * n_elem]
    double* dims;           // [3 * n_elem] half-widths
    double* obs_centers;    // [3 * n_elem] observation centers (may differ with symmetry)
};

// ============================================================
// Symmetry transform data
// ============================================================
struct RadGPU_SymData {
    int n_copies;                               // total symmetry copies (including identity)
    double point_transforms[RADGPU_MAX_SYM_COPIES * 9];  // [n_copies][3x3] point transforms
    double field_transforms[RADGPU_MAX_SYM_COPIES * 9];  // [n_copies][3x3] field sign transforms
};

// ============================================================
// Assembly output: flat interaction matrix blocks
// ============================================================
struct RadGPU_AsmResult {
    int N;                  // number of elements
    float* matrix_blocks;   // [N * N * 9] row-major 3x3 blocks, row-major within each block
};

// ============================================================
// GPU assembly functions
// ============================================================

// Pack geometry from Radia interaction data
int radGPU_PackGeometryForAsm(
    class radTInteraction* intrct,
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData);

// Run GPU assembly — fills result->matrix_blocks
int radGPU_AssembleMatrix(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_AsmResult* result);

// Unpack GPU matrix into Radia's TMatrix3df format
void radGPU_UnpackMatrix(
    RadGPU_AsmResult* result,
    class radTInteraction* intrct);

// Free all GPU assembly data
void radGPU_FreeAsmData(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_AsmResult* result);

#endif // RADIA_WITH_CUDA
#endif // __RADGPU_ASM_H