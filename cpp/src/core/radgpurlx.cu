/*-------------------------------------------------------------------------
*
* File name:      radgpurlx.cu
*
* Project:        RADIA
*
* Description:    CUDA kernels for GPU-accelerated relaxation
*                 Red-black Gauss-Seidel with implicit per-element solve
*
-------------------------------------------------------------------------*/

#ifdef RADIA_WITH_CUDA

#include "radgpurlx.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================
// Device helpers
// ============================================================
__device__ void mat3x3_vec_dev(const float* Q, const double* v, double* out)
{
    out[0] = (double)Q[0]*v[0] + (double)Q[1]*v[1] + (double)Q[2]*v[2];
    out[1] = (double)Q[3]*v[0] + (double)Q[4]*v[1] + (double)Q[5]*v[2];
    out[2] = (double)Q[6]*v[0] + (double)Q[7]*v[1] + (double)Q[8]*v[2];
}

__device__ bool inv_I_minus_ksiQ_dev(double ksi, const float* Q, double inv[9])
{
    double a[9];
    a[0] = 1.0 - ksi*(double)Q[0]; a[1] =     - ksi*(double)Q[1]; a[2] =     - ksi*(double)Q[2];
    a[3] =     - ksi*(double)Q[3]; a[4] = 1.0 - ksi*(double)Q[4]; a[5] =     - ksi*(double)Q[5];
    a[6] =     - ksi*(double)Q[6]; a[7] =     - ksi*(double)Q[7]; a[8] = 1.0 - ksi*(double)Q[8];

    double c00 = a[4]*a[8] - a[5]*a[7];
    double c01 = a[5]*a[6] - a[3]*a[8];
    double c02 = a[3]*a[7] - a[4]*a[6];
    double det = a[0]*c00 + a[1]*c01 + a[2]*c02;
    if(fabs(det) < 1e-30) return false;

    double idet = 1.0 / det;
    inv[0] = c00*idet; inv[1] = (a[2]*a[7]-a[1]*a[8])*idet; inv[2] = (a[1]*a[5]-a[2]*a[4])*idet;
    inv[3] = c01*idet; inv[4] = (a[0]*a[8]-a[2]*a[6])*idet; inv[5] = (a[2]*a[3]-a[0]*a[5])*idet;
    inv[6] = c02*idet; inv[7] = (a[1]*a[6]-a[0]*a[7])*idet; inv[8] = (a[0]*a[4]-a[1]*a[3])*idet;
    return true;
}

__device__ void mat3x3d_vec_dev(const double* M, const double* v, double* out)
{
    out[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
    out[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
    out[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
}

__device__ void cubpln_dev(double step, double f1, double f2, double fpr1, double fpr2, double* a)
{
    double inv = 1.0 / step;
    double d = (f2 - f1) * inv;
    a[0] = f1;
    a[1] = fpr1;
    a[2] = (3.0*d - 2.0*fpr1 - fpr2) * inv;
    a[3] = (-2.0*d + fpr1 + fpr2) * inv * inv;
}

__device__ double interp_mh_dev(double absH,
    const double* curveH, const double* curveM, const double* curveDMDH, int len)
{
    if(len <= 0) return 0.0;
    int idx = 0;
    for(int i = 0; i < len; i++) {
        if(curveH[i] > absH) break;
        idx = i;
    }
    if(idx >= len - 1) {
        return curveM[len-1] + (absH - curveH[len-1]) * curveDMDH[len-1];
    }
    double arg = absH - curveH[idx];
    double step = curveH[idx+1] - curveH[idx];
    double a[4];
    cubpln_dev(step, curveM[idx], curveM[idx+1], curveDMDH[idx], curveDMDH[idx+1], a);
    return a[0] + arg*(a[1] + arg*(a[2] + arg*a[3]));
}

__device__ double formula_absM_dev(double absH, const double* ms, const double* ks, int len)
{
    double absM = 0.0;
    for(int i = 0; i < len; i++) {
        if(ms[i] != 0.0) absM += ms[i] * tanh(ks[i] * absH / ms[i]);
    }
    return absM;
}

__device__ void get_ksi_and_absM_dev(
    double absH, int mtype, int elem,
    const double* linKsi,
    const double* mhH, const double* mhM, const double* mhdMdH,
    const int* mhOffset, const int* mhLen,
    const double* formulaMs, const double* formulaKs, const int* formulaLen,
    double* outKsi, double* outAbsM)
{
    const double absHZeroTol = 1e-10;
    if(mtype == 0) {
        *outKsi = linKsi[elem];
        *outAbsM = linKsi[elem] * absH;
    }
    else if(mtype == 1) {
        int off = mhOffset[elem];
        int len = mhLen[elem];
        if(absH <= absHZeroTol) {
            *outKsi = (len > 0) ? mhdMdH[off] : 0.0;
            *outAbsM = 0.0;
        } else {
            double am = interp_mh_dev(absH, &mhH[off], &mhM[off], &mhdMdH[off], len);
            *outAbsM = am;
            *outKsi = am / absH;
        }
    }
    else if(mtype == 3) {
        int flen = formulaLen[elem];
        if(absH <= absHZeroTol) {
            double k = 0.0;
            for(int i = 0; i < flen; i++) k += formulaKs[3*elem+i];
            *outKsi = k;
            *outAbsM = 0.0;
        } else {
            double am = formula_absM_dev(absH, &formulaMs[3*elem], &formulaKs[3*elem], flen);
            *outAbsM = am;
            *outKsi = am / absH;
        }
    }
    else {
        *outKsi = 0.0;
        *outAbsM = 0.0;
    }
}

// ============================================================
// Kernel: compute quasi-external field for a subset of elements
//   H_ext_eff[i] = sum_{j != i} A[i][j] * M[j] + H_ext[i]
// Each thread handles one element from the color set
// ============================================================
__global__ void compute_quasi_ext_field_kernel(
    const float* __restrict__ matrix,
    const double* __restrict__ magn,
    const double* __restrict__ extField,
    double* __restrict__ quasiExtField,   // [3 * numColor] output
    const int* __restrict__ colorIndices,  // element indices for this color
    int numColor,
    int N3)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if(ci >= numColor) return;

    int elem = colorIndices[ci];
    int r0 = 3 * elem;

    for(int comp = 0; comp < 3; comp++) {
        int row = r0 + comp;
        const float* matRow = matrix + (long long)row * N3;
        double sum = 0.0;

        // Sum over ALL elements (including self — we subtract self below)
        for(int j = 0; j < N3; j++) {
            sum += (double)matRow[j] * magn[j];
        }
        // Subtract self-interaction Q_ii * M_i
        // (will be handled implicitly in the solve kernel)
        const float* selfRow = matRow + r0;
        sum -= (double)selfRow[0] * magn[r0+0]
             + (double)selfRow[1] * magn[r0+1]
             + (double)selfRow[2] * magn[r0+2];

        quasiExtField[3*ci + comp] = sum + extField[row];
    }
}


// Full-matrix matvec (same as before)
__global__ void matvec_add_extfield_kernel(
    const float* __restrict__ matrix,
    const double* __restrict__ magn,
    const double* __restrict__ extField,
    double* __restrict__ field,
    int N3)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= N3) return;

    const float* matRow = matrix + (long long)row * N3;
    double sum = 0.0;
    for(int j = 0; j < N3; j++) {
        sum += (double)matRow[j] * magn[j];
    }
    field[row] = sum + extField[row];
}

// Under-relaxed update + residual
__global__ void under_relax_and_residual_kernel(
    double* __restrict__ magn,
    const double* __restrict__ magn_new,
    double* __restrict__ residual_buf,
    double omega,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;

    double resid = 0.0;
    for(int c = 0; c < 3; c++) {
        int idx = 3 * elem + c;
        double m_mixed = magn[idx] + omega * (magn_new[idx] - magn[idx]);
        double diff = m_mixed - magn[idx];
        resid += diff * diff;
        magn[idx] = m_mixed;
    }
    residual_buf[elem] = resid;
}


// ============================================================
// Kernel: implicit solve + update for a subset of elements
// Directly updates magn[] in place (Gauss-Seidel style)
// ============================================================
__global__ void implicit_solve_kernel(
    const double* __restrict__ field_full,
    const double* __restrict__ magn_old,
    double* __restrict__ magn_new,
    double* __restrict__ field_out,
    const float* __restrict__ selfBlocks,
    const int* __restrict__ matType,
    const double* __restrict__ linKsi,
    const double* __restrict__ remMagn,
    const double* __restrict__ mhH,
    const double* __restrict__ mhM,
    const double* __restrict__ mhdMdH,
    const int* __restrict__ mhOffset,
    const int* __restrict__ mhLen,
    const double* __restrict__ formulaMs,
    const double* __restrict__ formulaKs,
    const int* __restrict__ formulaLen,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;

    const float* Qii = selfBlocks + 9 * elem;
    int mtype = matType[elem];

    double Mold[3] = {magn_old[3*elem], magn_old[3*elem+1], magn_old[3*elem+2]};

    // Quasi-external field = full_field - Q_ii * M_old
    double QiiMold[3];
    mat3x3_vec_dev(Qii, Mold, QiiMold);

    double Hext_eff[3];
    Hext_eff[0] = field_full[3*elem+0] - QiiMold[0];
    Hext_eff[1] = field_full[3*elem+1] - QiiMold[1];
    Hext_eff[2] = field_full[3*elem+2] - QiiMold[2];

    double rmx = remMagn[3*elem+0];
    double rmy = remMagn[3*elem+1];
    double rmz = remMagn[3*elem+2];

    double H[3];

    if(mtype == 0) {
        // Linear isotropic: exact one-step
        double ksi = linKsi[elem];
        double mr[3] = {rmx, rmy, rmz};
        double QiiMr[3];
        mat3x3_vec_dev(Qii, mr, QiiMr);
        double rhs[3] = {Hext_eff[0]+QiiMr[0], Hext_eff[1]+QiiMr[1], Hext_eff[2]+QiiMr[2]};

        double inv[9];
        if(inv_I_minus_ksiQ_dev(ksi, Qii, inv)) {
            mat3x3d_vec_dev(inv, rhs, H);
        } else {
            H[0] = rhs[0]; H[1] = rhs[1]; H[2] = rhs[2];
        }

        magn_new[3*elem+0] = ksi*H[0] + rmx;
        magn_new[3*elem+1] = ksi*H[1] + rmy;
        magn_new[3*elem+2] = ksi*H[2] + rmz;
        field_out[3*elem+0] = H[0]; field_out[3*elem+1] = H[1]; field_out[3*elem+2] = H[2];
        return;
    }

    // Nonlinear: add Q_ii * Mr to quasi-external field
    double mr[3] = {rmx, rmy, rmz};
    double QiiMr[3];
    mat3x3_vec_dev(Qii, mr, QiiMr);
    double Hext_eff_full[3] = {
        Hext_eff[0] + QiiMr[0],
        Hext_eff[1] + QiiMr[1],
        Hext_eff[2] + QiiMr[2]
    };

    // Start from previous H if available, otherwise quasi-external field
    double prevH[3] = {field_out[3*elem], field_out[3*elem+1], field_out[3*elem+2]};
    double prevHmag = prevH[0]*prevH[0] + prevH[1]*prevH[1] + prevH[2]*prevH[2];
    if(prevHmag > 1e-30) {
        H[0] = prevH[0]; H[1] = prevH[1]; H[2] = prevH[2];
    } else {
        H[0] = Hext_eff_full[0]; H[1] = Hext_eff_full[1]; H[2] = Hext_eff_full[2];
    }

    double absH = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);
    double misfitM = 1e23;
    const int maxInner = 15;

    for(int it = 0; it < maxInner; it++) {
        double ksi, absM;
        get_ksi_and_absM_dev(absH, mtype, elem,
            linKsi, mhH, mhM, mhdMdH, mhOffset, mhLen,
            formulaMs, formulaKs, formulaLen,
            &ksi, &absM);

        double inv[9];
        if(!inv_I_minus_ksiQ_dev(ksi, Qii, inv)) break;

        double Hnew[3];
        mat3x3d_vec_dev(inv, Hext_eff_full, Hnew);

        double newAbsH = sqrt(Hnew[0]*Hnew[0] + Hnew[1]*Hnew[1] + Hnew[2]*Hnew[2]);
        double newMisfitM = absM - newAbsH * ksi;
        double absNewMisfitM = fabs(newMisfitM);
        double absMisfitM = fabs(misfitM);

        double probNew = absMisfitM;
        if(newMisfitM * misfitM > 0.0) probNew += 0.5 * absNewMisfitM;
        double probOld = absNewMisfitM;
        double alpha = probNew / (probNew + probOld + 1e-30);
        absH = alpha * newAbsH + (1.0 - alpha) * absH;
        misfitM = absM - absH * ksi;

        if(newAbsH > 1e-25) {
            double sc = absH / newAbsH;
            H[0] = sc*Hnew[0]; H[1] = sc*Hnew[1]; H[2] = sc*Hnew[2];
        } else {
            H[0] = Hnew[0]; H[1] = Hnew[1]; H[2] = Hnew[2];
        }

        if(misfitM*misfitM < 1e-20) break;
    }

    // Final M from converged H
    absH = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);
    double ksi_f, absM_f;
    get_ksi_and_absM_dev(absH, mtype, elem,
        linKsi, mhH, mhM, mhdMdH, mhOffset, mhLen,
        formulaMs, formulaKs, formulaLen,
        &ksi_f, &absM_f);
    double sc = (absH > 1e-25) ? absM_f / absH : 0.0;

    magn_new[3*elem+0] = sc*H[0] + rmx;
    magn_new[3*elem+1] = sc*H[1] + rmy;
    magn_new[3*elem+2] = sc*H[2] + rmz;
    field_out[3*elem+0] = H[0]; field_out[3*elem+1] = H[1]; field_out[3*elem+2] = H[2];
}

// ============================================================
// Kernel: compute per-element |M_new - M_old|^2
// ============================================================
__global__ void residual_kernel(
    const double* __restrict__ magn,
    const double* __restrict__ magn_prev,
    double* __restrict__ residual_buf,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;
    double r = 0.0;
    for(int c = 0; c < 3; c++) {
        double d = magn[3*elem+c] - magn_prev[3*elem+c];
        r += d*d;
    }
    residual_buf[elem] = r;
}

// ============================================================
// Main GPU relaxation: under-relaxed Jacobi
// ============================================================
int radGPU_RelaxAuto(
    RadGPURelaxData* data,
    double precision,
    int maxIter,
    double* outMisfitM,
    double* outMaxModM,
    double* outMaxModH)
{
    int N = data->numElem;
    int N3 = data->matrixDim;
    long long matSize = (long long)N3 * N3;
    double precE2 = precision * precision;
    int result = -1;

    float *d_matrix = nullptr;
    double *d_magn = nullptr, *d_magn_new = nullptr, *d_magn_prev = nullptr;
    double *d_field_full = nullptr, *d_field_out = nullptr;
    double *d_extField = nullptr, *d_residual = nullptr;
    int *d_matType = nullptr;
    double *d_linKsi = nullptr, *d_remMagn = nullptr;
    double *d_mhH = nullptr, *d_mhM = nullptr, *d_mhdMdH = nullptr;
    int *d_mhOffset = nullptr, *d_mhLen = nullptr;
    double *d_formulaMs = nullptr, *d_formulaKs = nullptr;
    int *d_formulaLen = nullptr;
    float *d_selfBlocks = nullptr;

    double *h_residual = new double[N];

    #define CUDA_CHK(call) do { \
        cudaError_t e = (call); \
        if(e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            goto cleanup; \
        } \
    } while(0)

    CUDA_CHK(cudaMalloc(&d_matrix, matSize * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_magn, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_magn_new, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_magn_prev, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_field_full, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_field_out, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_extField, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_residual, N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_matType, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_linKsi, N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_remMagn, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_mhOffset, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_mhLen, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_formulaMs, 3 * N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_formulaKs, 3 * N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_formulaLen, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_selfBlocks, 9 * N * sizeof(float)));
    if(data->totalMHPoints > 0) {
        CUDA_CHK(cudaMalloc(&d_mhH, data->totalMHPoints * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_mhM, data->totalMHPoints * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_mhdMdH, data->totalMHPoints * sizeof(double)));
    }

    CUDA_CHK(cudaMemcpy(d_matrix, data->h_matrix, matSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_magn, data->h_magn, N3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_extField, data->h_extField, N3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_matType, data->h_matType, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_linKsi, data->h_linKsi, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_remMagn, data->h_remMagn, N3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mhOffset, data->h_mhOffset, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mhLen, data->h_mhLen, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_formulaMs, data->h_formulaMs, 3 * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_formulaKs, data->h_formulaKs, 3 * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_formulaLen, data->h_formulaLen, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_selfBlocks, data->h_selfBlocks, 9 * N * sizeof(float), cudaMemcpyHostToDevice));
    if(data->totalMHPoints > 0) {
        CUDA_CHK(cudaMemcpy(d_mhH, data->h_mhH, data->totalMHPoints * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_mhM, data->h_mhM, data->totalMHPoints * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_mhdMdH, data->h_mhdMdH, data->totalMHPoints * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Initialize field from h_field (computed in pack function)
    CUDA_CHK(cudaMemcpy(d_field_out, data->h_field, N3 * sizeof(double), cudaMemcpyHostToDevice));

    {
        int tpb = 256;
        int blkMV = (N3 + tpb - 1) / tpb;
        int blkEl = (N + tpb - 1) / tpb;

        double omega = (data->omega > 0.0) ? data->omega : 0.3;
        double omegaCeiling = 1.0;       // upper bound, ratchets down
        const double omegaMin = 0.05;
        double prevMisfitMe2 = 1e30;
        double bestMisfitMe2 = 1e30;
        int divergeCount = 0;
        double instMisfitMe2 = 1e30;   //
        int iterDone = 0;              //

        for(int iter = 0; iter < maxIter; iter++) {

            // Save current M
            CUDA_CHK(cudaMemcpy(d_magn_prev, d_magn, N3 * sizeof(double), cudaMemcpyDeviceToDevice));

            // Step 1: H_full = A * M + H_ext (full matrix, including diagonal)
            matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                d_matrix, d_magn, d_extField, d_field_full, N3);

            // Step 2: implicit per-element solve → M_proposed
            implicit_solve_kernel<<<blkEl, tpb>>>(
                d_field_full, d_magn, d_magn_new, d_field_out,
                d_selfBlocks,
                d_matType, d_linKsi, d_remMagn,
                d_mhH, d_mhM, d_mhdMdH, d_mhOffset, d_mhLen,
                d_formulaMs, d_formulaKs, d_formulaLen,
                N);

            // Step 3: under-relaxed update: M = M_old + omega * (M_proposed - M_old)
            under_relax_and_residual_kernel<<<blkEl, tpb>>>(
                d_magn, d_magn_new, d_residual, omega, N);

            // Step 4: convergence check
            CUDA_CHK(cudaMemcpy(h_residual, d_residual, N * sizeof(double), cudaMemcpyDeviceToHost));
            double sumR = 0.0;
            for(int i = 0; i < N; i++) sumR += h_residual[i];
            instMisfitMe2 = sumR / N;
            iterDone = iter + 1;

            if(instMisfitMe2 <= precE2) break;

            // Adaptive omega with ratcheting ceiling
            if(instMisfitMe2 < prevMisfitMe2) {
                // Converging
                divergeCount = 0;
                if(instMisfitMe2 < bestMisfitMe2) bestMisfitMe2 = instMisfitMe2;

                // Slow growth
                omega *= 1.01;
                if(omega > omegaCeiling) omega = omegaCeiling;
            } else {
                // Diverging
                divergeCount++;
                if(divergeCount >= 2) {
                    // Current omega is too high — ratchet ceiling down
                    omegaCeiling = omega * 0.95;
                    omega *= 0.8;
                    if(omega < omegaMin) omega = omegaMin;
                    if(omegaCeiling < omegaMin) omegaCeiling = omegaMin;
                    divergeCount = 0;
                }
            }
            prevMisfitMe2 = instMisfitMe2;
        }

        // Copy results back
        CUDA_CHK(cudaMemcpy(data->h_magn, d_magn, N3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(data->h_field, d_field_out, N3 * sizeof(double), cudaMemcpyDeviceToHost));

        *outMisfitM = sqrt(instMisfitMe2);
        double maxModM = 0, maxModH = 0;
        for(int i = 0; i < N; i++) {
            double mx = data->h_magn[3*i], my = data->h_magn[3*i+1], mz = data->h_magn[3*i+2];
            double mm = sqrt(mx*mx + my*my + mz*mz);
            if(mm > maxModM) maxModM = mm;
            double hx = data->h_field[3*i], hy = data->h_field[3*i+1], hz = data->h_field[3*i+2];
            double hm = sqrt(hx*hx + hy*hy + hz*hz);
            if(hm > maxModH) maxModH = hm;
        }
        *outMaxModM = maxModM;
        *outMaxModH = maxModH;
        result = iterDone;
    }

cleanup:
    delete[] h_residual;
    if(d_matrix) cudaFree(d_matrix);
    if(d_magn) cudaFree(d_magn);
    if(d_magn_new) cudaFree(d_magn_new);
    if(d_magn_prev) cudaFree(d_magn_prev);
    if(d_field_full) cudaFree(d_field_full);
    if(d_field_out) cudaFree(d_field_out);
    if(d_extField) cudaFree(d_extField);
    if(d_residual) cudaFree(d_residual);
    if(d_matType) cudaFree(d_matType);
    if(d_linKsi) cudaFree(d_linKsi);
    if(d_remMagn) cudaFree(d_remMagn);
    if(d_mhH) cudaFree(d_mhH);
    if(d_mhM) cudaFree(d_mhM);
    if(d_mhdMdH) cudaFree(d_mhdMdH);
    if(d_mhOffset) cudaFree(d_mhOffset);
    if(d_mhLen) cudaFree(d_mhLen);
    if(d_formulaMs) cudaFree(d_formulaMs);
    if(d_formulaKs) cudaFree(d_formulaKs);
    if(d_formulaLen) cudaFree(d_formulaLen);
    if(d_selfBlocks) cudaFree(d_selfBlocks);

    #undef CUDA_CHK
    return result;
}

#endif // RADIA_WITH_CUDA