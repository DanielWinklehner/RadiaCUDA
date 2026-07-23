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
//#define RADGPU_MAX_SYM_COPIES 64

// ============================================================
// Flat geometry for GPU: polyhedron faces (mixed models supported).
// Arrays are indexed by the GLOBAL element index 0..n_elem-1; RecMag
// elements simply have an empty face range in face_offsets.
// ============================================================
struct RadGPU_PolyData {
    int n_elem;             // TOTAL relaxable elements N (poly + RecMag)
    int n_faces_total;
    int n_edges_total;

    // Per-element
    double* centers;        // [3 * n_elem] transformed observation centers (all elements)
    int* face_offsets;      // [n_elem + 1] CSR into face arrays (empty range for RecMags)

    // Per-face
    int* edge_offsets;      // [n_faces_total + 1] CSR into edge arrays
    double* face_cz;        // [n_faces_total] z-coord in local frame
    double* face_rot;       // [9 * n_faces_total] rotation matrices (row-major)
    double* face_orig;      // [3 * n_faces_total] face origins

    // Per-edge
    double* edge_pts_2d;    // [2 * n_edges_total] 2D edge vertices
};

// ============================================================
// Flat geometry for GPU: RecMag elements (mixed models supported).
// Arrays indexed by GLOBAL element index; entries of non-RecMag
// elements are zero and never read (guarded by is_rec).
// ============================================================
struct RadGPU_RecMagData {
    int n_rec;              // number of RecMag elements (0 = pure-polyhedron model)
    int* is_rec;            // [n_elem] 1 if element is a RecMag
    double* centers;        // [3 * n_elem] cuboid centers, element's own frame
    double* dims;           // [3 * n_elem] FULL edge lengths (radTRecMag::Dimensions)
    // Snapshot of radCR (radTConvergRepair) tolerances: the closed-form cuboid
    // Q-tensor uses AbsRandMagnitude-based guards on face/edge coincidences,
    // which must match the CPU B_comp exactly for parity.
    double abs_rand;
    double rel_rand;
    double zero_rand;
    int act_on_doubles;
};

// ============================================================
// Symmetry transform data
// ============================================================
struct RadGPU_SymData {
    int n_elem;                // number of elements
    int total_copies;          // sum of all sym copies across elements
    int* sym_counts;           // [n_elem] copies per element
    int* sym_offsets;          // [n_elem+1] offset into transform arrays
    double* point_transforms;  // [total_copies * 9] flattened
    double* field_transforms;  // [total_copies * 9] flattened
};
//struct RadGPU_SymData {
//    int n_copies;                               // total symmetry copies (including identity)
//    double point_transforms[RADGPU_MAX_SYM_COPIES * 9];  // [n_copies][3x3] point transforms
//    double field_transforms[RADGPU_MAX_SYM_COPIES * 9];  // [n_copies][3x3] field sign transforms
//};

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

void radGPU_FreeSymData(RadGPU_SymData* symData);

#endif // RADIA_WITH_CUDA
#endif // __RADGPU_ASM_H