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
    double* centers;        // [3 * n_elem] transformed observation centers (all
                            // elements). The assembly kernel no longer reads
                            // this -- it takes its observation points from
                            // RadGPU_ObsQuadData, which carries the same
                            // centroids in the collocation (default) case.
                            // Kept because it documents/defines the frame.
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
// Observation-element quadrature (OPT-IN Galerkin assembly; radgalerkin.h).
//
// Collocation (the default) is the degenerate case of this: exactly one point
// per element -- the transformed centroid -- with weight 1. The kernel then
// computes 0 + 1.0*block, which is bit-identical to the old single-point code
// (multiplication by 1.0 and addition to 0 are exact in IEEE-754), so the flag
// being off is a true no-op. tests/test_galerkin_asm.py checks this.
//
// `on` only controls the NEAR pass and the diagnostics; the base arrays are
// always used.
// ============================================================
struct RadGPU_ObsQuadData {
    int on;                 // 1 = Galerkin enabled (near pass may be present)
    int n_elem;

    // Base rule -- applied to EVERY pair. STEP 1 (studies/GALERKIN_STEP1.md)
    // showed a distance cutoff cannot be used here: the per-pair corrections
    // very nearly cancel for smooth M, so truncating the sum overshoots the
    // true correction by more than the correction itself.
    int* q_offsets;         // [n_elem + 1] CSR into q_pts / q_w
    double* q_pts;          // [3 * q_total] observation points, LAB frame
                            //               (already through MainTransPtrArray)
    double* q_w;            // [q_total] weights; sum to 1 per element
    int q_total;

    // Near pass -- a higher-order rule on the O(N) near-pair list, which
    // replaces those entries after the base pass.
    int* n_offsets;         // [n_elem + 1] CSR into n_pts / n_w
    double* n_pts;          // [3 * n_total]
    double* n_w;            // [n_total]
    int n_total;
    int* pair_rows;         // [n_pairs] observation element index
    int* pair_cols;         // [n_pairs] source element index
    int n_pairs;
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

// Pack the observation-element quadrature (called by radGPU_PackGeometryForAsm;
// separate only so it can be a friend of radTInteraction). Returns 0 if some
// element type has no volume quadrature.
int radGPU_PackObsQuadForAsm(
    class radTInteraction* intrct,
    RadGPU_ObsQuadData* quadData);

// Pack geometry from Radia interaction data
int radGPU_PackGeometryForAsm(
    class radTInteraction* intrct,
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_ObsQuadData* quadData);

// Run GPU assembly — fills result->matrix_blocks
int radGPU_AssembleMatrix(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_ObsQuadData* quadData,
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

void radGPU_FreeObsQuadData(RadGPU_ObsQuadData* quadData);

void radGPU_FreeSymData(RadGPU_SymData* symData);

#endif // RADIA_WITH_CUDA
#endif // __RADGPU_ASM_H