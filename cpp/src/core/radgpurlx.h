/*-------------------------------------------------------------------------
*
* File name:      radgpurlx.h
*
* Project:        RADIA
*
* Description:    GPU-accelerated relaxation (Jacobi iteration)
*
* Author(s):      GPU implementation
*
* First release:  2025
*
-------------------------------------------------------------------------*/

#ifndef __RADGPURLX_H
#define __RADGPURLX_H

#ifdef RADIA_WITH_CUDA

struct RadGPURelaxData {
    int numElem;
    int matrixDim;         // 3 * numElem
    double omega;              // initial omega, negative = use default (0.3)

    // Identity stamp of the interaction matrix (radTInteraction::mGpuMatrixStamp).
    // When it matches the device-side cache, h_matrix may be null (flatten +
    // upload skipped; the resident device matrix is reused).
    unsigned long long matrixStamp;

    // Interaction matrix flattened to row-major float[matrixDim x matrixDim]
    // (null when the device cache holds this stamp already)
    float* h_matrix;
    // 1 when the pack allocated h_matrix and radGPU_FreeData must release it;
    // 0 when it is BORROWED from the interaction object -- the GPU assembly's
    // own output is already in exactly this layout, so the pack points at it
    // rather than making a second host copy of the 36*N^2 matrix.
    int h_matrixOwned;

    // Working arrays (double)
    double* h_magn;        // [matrixDim] current magnetization
    double* h_extField;    // [matrixDim] external field
    double* h_field;       // [matrixDim] total H field

    // Per-element material info
    int* h_matType;        // [numElem] 0=lin_iso, 1=nonlin_iso, 2=lin_aniso, 3=nonlin_iso_formula
    double* h_remMagn;     // [3*numElem] remanent magnetization

    // Nonlinear isotropic (tabulated): concatenated M-H curves
    double* h_mhH;         // concatenated H values
    double* h_mhM;         // concatenated M values
    double* h_mhdMdH;      // concatenated dM/dH values
    int* h_mhOffset;       // [numElem] start offset
    int* h_mhLen;          // [numElem] number of points
    int totalMHPoints;

    // Nonlinear isotropic (formula): M = sum ms_i * tanh(ks_i * H / ms_i)
    double* h_formulaMs;   // [3*numElem]
    double* h_formulaKs;   // [3*numElem]
    int* h_formulaLen;     // [numElem] (0, 1, 2, or 3)

    // Linear isotropic
    double* h_linKsi;      // [numElem] scalar susceptibility

    // Self-interaction diagonal blocks
    float* h_selfBlocks;   // [9*numElem]
};

// GPU solver — returns iteration count, or -1 on failure
int radGPU_RelaxAuto(
    RadGPURelaxData* data,
    double precision,
    int maxIter,
    double* outMisfitM,
    double* outMaxModM,
    double* outMaxModH);

// Method 11: Newton-Krylov (preconditioned GMRES on the analytic Jacobian).
// Reported misfit = RMS physics residual |M_mat(H) - M| in T; returned count
// = total matvecs (maxIter caps that budget). -1 on failure.
int radGPU_RelaxNK(
    RadGPURelaxData* data,
    double precision,
    int maxIter,
    double* outMisfitM,
    double* outMaxModM,
    double* outMaxModH);

// Data packing/unpacking. skipMatrix != 0 leaves h_matrix null (caller
// verified the device cache holds this interaction's matrix already).
int radGPU_PackInteractionData(
    class radTInteraction* intrct,
    RadGPURelaxData* gpuData,
    int skipMatrix = 0);

// True when the device-side matrix cache holds this (stamp, matrixDim).
int radGPU_MatrixCached(unsigned long long stamp, int matrixDim);

// Hand a device matrix produced by the GPU assembly straight to the solver's
// resident cache, taking ownership of it. The assembly already emits the
// solver's layout, so the first solve then skips BOTH the O(N^2) host flatten
// and the H2D upload. Any previously cached matrix is released.
void radGPU_AdoptMatrixCache(float* d_matrix, unsigned long long stamp,
                             int matrixDim);

void radGPU_UnpackMagnetization(
    RadGPURelaxData* gpuData,
    class radTInteraction* intrct);

void radGPU_FreeData(RadGPURelaxData* data);

#endif // RADIA_WITH_CUDA
#endif // __RADGPURLX_H