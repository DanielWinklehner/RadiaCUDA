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

    // Interaction matrix flattened to row-major float[matrixDim x matrixDim]
    float* h_matrix;

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
// GPU solver — returns iteration count, or -1 on failure
int radGPU_RelaxAuto(
    RadGPURelaxData* data,
    double precision,
    int maxIter,
    double* outMisfitM,
    double* outMaxModM,
    double* outMaxModH);

// Data packing/unpacking
int radGPU_PackInteractionData(
    class radTInteraction* intrct,
    RadGPURelaxData* gpuData);

void radGPU_UnpackMagnetization(
    RadGPURelaxData* gpuData,
    class radTInteraction* intrct);

void radGPU_FreeData(RadGPURelaxData* data);

#endif // RADIA_WITH_CUDA
#endif // __RADGPURLX_H