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
#include <cstdlib>

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
// Dense matvec + external field: warp-per-row with coalesced column reads
// and DOUBLE accumulation. The previous row-per-thread kernel made adjacent
// threads read addresses N3 apart (fully uncoalesced) and ran ~3.5x off
// memory bandwidth; here lane k of the row's warp reads columns k, k+32, ...
// (consecutive addresses across the warp), then a shuffle reduction sums.
__global__ void matvec_add_extfield_kernel(
    const float* __restrict__ matrix,
    const double* __restrict__ magn,
    const double* __restrict__ extField,
    double* __restrict__ field,
    int N3)
{
    int warpsPerBlock = blockDim.x >> 5;
    int row = blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if(row >= N3) return;

    const float* matRow = matrix + (long long)row * N3;
    double sum = 0.0;
    for(int j = lane; j < N3; j += 32) {
        sum += (double)matRow[j] * magn[j];
    }
    for(int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if(lane == 0) field[row] = sum + extField[row];
}

// Fixed-point residual: F = M_proposed - M (the UNDAMPED proposal step --
// the misfit; see the method-9 false-convergence fix) + per-element |F|^2.
__global__ void proposal_residual_kernel(
    const double* __restrict__ magn,
    const double* __restrict__ magn_new,
    double* __restrict__ f_vec,
    double* __restrict__ residual_buf,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;

    double resid = 0.0;
    for(int c = 0; c < 3; c++) {
        int idx = 3 * elem + c;
        double step = magn_new[idx] - magn[idx];
        f_vec[idx] = step;
        resid += step * step;
    }
    residual_buf[elem] = resid;
}

// Plain damped update: X += omega * F
__global__ void axpy_update_kernel(
    double* __restrict__ x,
    const double* __restrict__ f,
    double omega,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) x[i] += omega * f[i];
}

// out = a - b
__global__ void vec_diff_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) out[i] = a[i] - b[i];
}

// Batched dot products for the Anderson least squares: pair p of (pa, pb)
// index lists; block (x, p) reduces its grid-stride slice into
// out[p * gridDim.x + x]. The per-block partials are summed ON THE HOST in
// a FIXED ORDER: an atomicAdd accumulation is order-nondeterministic, and
// on these soft-mode-dominated problems a 1e-16 difference in gamma grows
// into a macroscopically different iteration path run-to-run.
__global__ void batched_dot_kernel(
    const double* __restrict__ base,
    const int* __restrict__ pa_off,
    const int* __restrict__ pb_off,
    double* __restrict__ partials,
    int numPairs,
    int n)
{
    int p = blockIdx.y;
    if(p >= numPairs) return;
    const double* a = base + pa_off[p];
    const double* b = base + pb_off[p];

    double sum = 0.0;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }
    for(int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    __shared__ double warpSums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(lane == 0) warpSums[warp] = sum;
    __syncthreads();
    if(warp == 0) {
        int nWarps = (blockDim.x + 31) >> 5;
        sum = (lane < nWarps) ? warpSums[lane] : 0.0;
        for(int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        }
        if(lane == 0) partials[(size_t)p * gridDim.x + blockIdx.x] = sum;
    }
}

// Anderson type-II update:
//   X <- X + beta*F - sum_j gamma_j * (dX_j + beta*dF_j)
// dX/dF columns live in ring buffers of pitch n; gamma is a small device
// array (histLen <= RADGPU_ANDERSON_M entries).
__global__ void anderson_update_kernel(
    double* __restrict__ x,
    const double* __restrict__ f,
    const double* __restrict__ dX,
    const double* __restrict__ dF,
    const double* __restrict__ gamma,
    int histLen,
    double beta,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double upd = beta * f[i];
    for(int j = 0; j < histLen; j++) {
        upd -= gamma[j] * (dX[(long long)j * n + i] + beta * dF[(long long)j * n + i]);
    }
    x[i] += upd;
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

    // Damped inner fixed point on H (mirrors the method-10 CPU port): the
    // previous alpha-mixed secant scheme limit-cycled on strongly saturating
    // elements -- the resulting proposal jitter both floored the outer
    // misfit and poisoned the Anderson dF history. The budget must be deep
    // enough for the 0.5-damped iteration to actually meet the tolerance on
    // hard saturating elements: leftover inner error is proposal noise that
    // the outer Anderson mixing amplifies.
    const int maxInner = 60;
    for(int it = 0; it < maxInner; it++) {
        double absHc = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);
        double ksi, absM;
        get_ksi_and_absM_dev(absHc, mtype, elem,
            linKsi, mhH, mhM, mhdMdH, mhOffset, mhLen,
            formulaMs, formulaKs, formulaLen,
            &ksi, &absM);

        double inv[9];
        if(!inv_I_minus_ksiQ_dev(ksi, Qii, inv)) break;

        double Hnew[3];
        mat3x3d_vec_dev(inv, Hext_eff_full, Hnew);

        double d0 = Hnew[0] - H[0], d1 = Hnew[1] - H[1], d2 = Hnew[2] - H[2];
        H[0] += 0.5 * d0; H[1] += 0.5 * d1; H[2] += 0.5 * d2;

        double dm = d0*d0 + d1*d1 + d2*d2;
        double nm = Hnew[0]*Hnew[0] + Hnew[1]*Hnew[1] + Hnew[2]*Hnew[2];
        if(dm <= 1e-18 * (1.0 + nm)) break;
    }

    // Final M from converged H
    double absH = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);
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
// Device-resident interaction-matrix cache: repeated RlxAuto calls on the
// SAME interaction matrix (identified by radTInteraction::mGpuMatrixStamp)
// reuse the uploaded float matrix -- skipping the O(N^2) host flatten, the
// device allocation, and the (up to multi-GB) PCIe upload. Exactly one
// matrix is held; it is replaced when a different stamp arrives.
// ============================================================
static unsigned long long g_matCacheStamp = 0;
static int g_matCacheDim = 0;
static float* g_d_matCache = nullptr;

int radGPU_MatrixCached(unsigned long long stamp, int matrixDim)
{
    return (g_d_matCache != nullptr && stamp != 0 &&
            g_matCacheStamp == stamp && g_matCacheDim == matrixDim) ? 1 : 0;
}

// Anderson-acceleration compile-time defaults (env-overridable for A/B runs:
// RADIA_NO_ANDERSON=1 disables; RADIA_ANDERSON_M / RADIA_ANDERSON_BETA tune).
#define RADGPU_ANDERSON_M_MAX 8
static const int    kAndersonMDefault    = 5;
// beta <= 0 means AUTO: use the current adaptive omega. This matters: plain
// Jacobi on these strongly-coupled maps is UNSTABLE at omega ~ O(1) (that is
// why the adaptive scheme lives at 0.05-0.3), so a fixed large Anderson
// damping makes the non-extrapolated part of every step divergent by itself.
static const double kAndersonBetaDefault = -1.0;
static const int    kAndersonWarmup      = 20;   // plain damped passes first
static const int    kAndersonCooloff     = 20;   // plain passes after a blow-up
static const int    kAndersonMaxBlowups  = 10;   // then disable for the run
                                                 // (decremented on real progress)

// Solve (G + lambda*I) gamma = b in-place, dim n <= RADGPU_ANDERSON_M_MAX.
// Returns 0 on a degenerate pivot. The regularization must be substantial:
// near the dominant slow mode the dF history columns become nearly
// collinear and the normal equations square that ill-conditioning.
static int solve_small_ls(double* G, double* b, double* gamma, int n)
{
    double maxDiag = 0.0;
    for(int i = 0; i < n; i++) if(G[i*n+i] > maxDiag) maxDiag = G[i*n+i];
    double lambda = 1e-10 * (maxDiag > 0.0 ? maxDiag : 1.0);
    for(int i = 0; i < n; i++) G[i*n+i] += lambda;

    for(int k = 0; k < n; k++) {
        int piv = k;
        for(int i = k+1; i < n; i++)
            if(fabs(G[i*n+k]) > fabs(G[piv*n+k])) piv = i;
        if(fabs(G[piv*n+k]) < 1e-300) return 0;
        if(piv != k) {
            for(int j = 0; j < n; j++) { double t = G[k*n+j]; G[k*n+j] = G[piv*n+j]; G[piv*n+j] = t; }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        for(int i = k+1; i < n; i++) {
            double f = G[i*n+k] / G[k*n+k];
            for(int j = k; j < n; j++) G[i*n+j] -= f * G[k*n+j];
            b[i] -= f * b[k];
        }
    }
    for(int i = n-1; i >= 0; i--) {
        double s = b[i];
        for(int j = i+1; j < n; j++) s -= G[i*n+j] * gamma[j];
        gamma[i] = s / G[i*n+i];
    }
    return 1;
}

// ============================================================
// Main GPU relaxation: Anderson-accelerated under-relaxed Jacobi
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

    // Anderson config (env-overridable). DEFAULT OFF (opt in with
    // RADIA_ANDERSON=1): benchmarked 3x fewer iterations to the same
    // endpoint on well-conditioned models (requires the deep damped inner
    // solve -- residual inner-solve noise poisons the dF history), but on
    // ill-conditioned production meshes the accelerated iteration stalls at
    // a HIGHER misfit floor than the plain damped one. Not ready as a
    // default; safeguards (gamma cap, predicted-residual check, revert,
    // beta controller, deterministic dot reduction) are all in place for
    // further tuning.
    int andersonM = kAndersonMDefault;
    double andersonBeta = kAndersonBetaDefault;
    bool andersonEnabled = false;
    {
        const char* e = getenv("RADIA_ANDERSON");
        if(e && *e && *e != '0') andersonEnabled = true;
        e = getenv("RADIA_NO_ANDERSON");
        if(e && *e && *e != '0') andersonEnabled = false;
        e = getenv("RADIA_ANDERSON_M");
        if(e && *e) { int v = atoi(e); if(v >= 1 && v <= RADGPU_ANDERSON_M_MAX) andersonM = v; }
        e = getenv("RADIA_ANDERSON_BETA");
        if(e && *e) { double v = atof(e); if(v > 0.0 && v <= 1.0) andersonBeta = v; }
        // andersonBeta <= 0 -> use the current adaptive omega each pass
    }
    int andersonDebug = 0;
    {
        const char* e = getenv("RADIA_ANDERSON_DEBUG");
        if(e && *e) andersonDebug = atoi(e);
    }

    float *d_matrix = nullptr;
    double *d_magn = nullptr, *d_magn_new = nullptr;
    double *d_field_full = nullptr, *d_field_out = nullptr;
    double *d_extField = nullptr, *d_residual = nullptr;
    int *d_matType = nullptr;
    double *d_linKsi = nullptr, *d_remMagn = nullptr;
    double *d_mhH = nullptr, *d_mhM = nullptr, *d_mhdMdH = nullptr;
    int *d_mhOffset = nullptr, *d_mhLen = nullptr;
    double *d_formulaMs = nullptr, *d_formulaKs = nullptr;
    int *d_formulaLen = nullptr;
    float *d_selfBlocks = nullptr;
    // Anderson workspace: [dF (m cols) | dX (m cols) | F | Fprev | Xprev],
    // each column N3 doubles; plus the small dot/gamma/pair buffers.
    double *d_work = nullptr, *d_dots = nullptr, *d_gamma = nullptr;
    int *d_pa = nullptr, *d_pb = nullptr;

    double *h_residual = new double[N];

    #define CUDA_CHK(call) do { \
        cudaError_t e = (call); \
        if(e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            goto cleanup; \
        } \
    } while(0)

    // --- interaction matrix: resident cache ---
    if(radGPU_MatrixCached(data->matrixStamp, N3)) {
        d_matrix = g_d_matCache;
    } else {
        if(g_d_matCache) {
            cudaFree(g_d_matCache);
            g_d_matCache = nullptr; g_matCacheStamp = 0; g_matCacheDim = 0;
        }
        if(data->h_matrix == nullptr) {
            fprintf(stderr, "radGPU_RelaxAuto: matrix neither packed nor cached\n");
            goto cleanup;
        }
        {// Pre-flight: refuse cleanly (with sizing guidance) instead of a
         // raw CUDA malloc failure when the dense matrix cannot fit.
            size_t freeB = 0, totalB = 0;
            cudaMemGetInfo(&freeB, &totalB);
            size_t needB = matSize * sizeof(float)
                           + (size_t)N3 * 24 * sizeof(double);  // work arrays
            if(needB > freeB) {
                double maxElem = (freeB > 0)
                    ? (sqrt((double)freeB / sizeof(float)) / 3.0) : 0.0;
                fprintf(stderr,
                    "radGPU_RelaxAuto: dense interaction matrix needs %.1f GB "
                    "but only %.1f of %.1f GB GPU memory is free (~%.0fk "
                    "elements max on this GPU). Falling back to the CPU "
                    "relaxation (method 10, OpenMP-parallel) -- expect "
                    "minutes-per-1000-iterations at this size; consider a "
                    "coarser mesh or structured elements.\n",
                    needB / 1e9, freeB / 1e9, totalB / 1e9, maxElem / 1e3);
                goto cleanup;  // returns -1 -> dispatch falls back to method 10
            }
        }
        CUDA_CHK(cudaMalloc(&d_matrix, matSize * sizeof(float)));
        CUDA_CHK(cudaMemcpy(d_matrix, data->h_matrix, matSize * sizeof(float), cudaMemcpyHostToDevice));
        g_d_matCache = d_matrix;
        g_matCacheStamp = data->matrixStamp;
        g_matCacheDim = N3;
    }

    CUDA_CHK(cudaMalloc(&d_magn, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_magn_new, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_field_full, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_field_out, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_extField, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_residual, N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_work, (2 * (size_t)andersonM + 3) * N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_dots, (RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1))
                                 * 32 * sizeof(double)));  // per-block partials
    CUDA_CHK(cudaMalloc(&d_gamma, RADGPU_ANDERSON_M_MAX * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_pa, (RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1)) * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_pb, (RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1)) * sizeof(int)));
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

    {
        int tpb = 256;
        int warpsPerBlock = tpb / 32;
        int blkMV = (N3 + warpsPerBlock - 1) / warpsPerBlock;  // warp-per-row
        int blkEl = (N + tpb - 1) / tpb;
        int blkV = (N3 + tpb - 1) / tpb;

        // Initialize the per-element H (inner-solve linearization start)
        // ON THE GPU: H = A*M + H_ext (was an O(N^2) CPU loop in Pack).
        matvec_add_extfield_kernel<<<blkMV, tpb>>>(
            d_matrix, d_magn, d_extField, d_field_out, N3);

        double omega = (data->omega > 0.0) ? data->omega : 0.3;
        double omegaCeiling = 1.0;       // upper bound, ratchets down
        const double omegaMin = 0.05;
        double prevMisfitMe2 = 1e30;
        double bestMisfitMe2 = 1e30;
        int divergeCount = 0;
        int convergeStreak = 0;        // sustained-convergence counter
        int sinceBest = 0;             // passes since the stagnation reference improved
        int omegaResets = 0;           // omega restarts since the reference improved
        double refMisfitMe2 = 1e30;    // FROZEN stagnation reference (snapshot)
        double instMisfitMe2 = 1e30;   //
        int iterDone = 0;              //

        // Anderson state (workspace layout: dF cols | dX cols | F | Fprev | Xprev)
        double* d_dF    = d_work;
        double* d_dX    = d_work + (size_t)andersonM * N3;
        double* d_F     = d_work + (size_t)2 * andersonM * N3;
        double* d_Fprev = d_F + N3;
        double* d_Xprev = d_Fprev + N3;
        int histLen = 0;
        bool havePrev = false;
        int cooloff = 0;               // plain passes remaining after a blow-up
        int blowups = 0;
        // Anderson's own damping controller, SEPARATE from omega: the omega
        // controller reacts to misfit trends, and Anderson's legitimately
        // non-monotonic misfit would ratchet omega (and a beta tied to it)
        // into the floor -- measured as a stall at beta = omega_min.
        double andersonBetaCur = (andersonBeta > 0.0) ? andersonBeta : 0.3;
        bool lastAnderson = false;     // last applied update was Anderson
        double hG[RADGPU_ANDERSON_M_MAX * RADGPU_ANDERSON_M_MAX];
        double hB[RADGPU_ANDERSON_M_MAX], hGamma[RADGPU_ANDERSON_M_MAX];
        double hDots[RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1)];
        double hPartials[RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1) * 32];
        int hPa[RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1)];
        int hPb[RADGPU_ANDERSON_M_MAX * (RADGPU_ANDERSON_M_MAX + 1)];

        for(int iter = 0; iter < maxIter; iter++) {

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

            // Step 3: fixed-point residual F = M_proposed - M (the misfit)
            proposal_residual_kernel<<<blkEl, tpb>>>(
                d_magn, d_magn_new, d_F, d_residual, N);

            // Step 4: convergence check
            CUDA_CHK(cudaMemcpy(h_residual, d_residual, N * sizeof(double), cudaMemcpyDeviceToHost));
            double sumR = 0.0;
            for(int i = 0; i < N; i++) sumR += h_residual[i];

            // Anderson blow-up safeguard: a big misfit jump right after an
            // accelerated step means the extrapolation left the basin --
            // REVERT to the pre-step state, drop the history, and run plain
            // damped passes for a while.
            double newMisfitMe2 = sumR / N;
            if(havePrev && histLen > 0 && newMisfitMe2 > 1.5 * instMisfitMe2) {
                CUDA_CHK(cudaMemcpy(d_magn, d_Xprev, N3 * sizeof(double),
                                    cudaMemcpyDeviceToDevice));
                histLen = 0;
                havePrev = false;
                lastAnderson = false;
                cooloff = kAndersonCooloff;
                andersonBetaCur *= 0.5;
                if(andersonBetaCur < 0.05) andersonBetaCur = 0.05;
                if(++blowups >= kAndersonMaxBlowups) andersonEnabled = false;
                iterDone = iter + 1;
                continue;  // recompute the proposal from the restored state
            }
            // Anderson beta feedback: grow while the accelerated steps
            // improve the misfit, back off (gently) when they don't.
            if(lastAnderson) {
                andersonBetaCur *= (newMisfitMe2 < instMisfitMe2) ? 1.01 : 0.95;
                if(andersonBetaCur > 1.0) andersonBetaCur = 1.0;
                if(andersonBetaCur < 0.05) andersonBetaCur = 0.05;
            }
            instMisfitMe2 = newMisfitMe2;
            iterDone = iter + 1;

            if(instMisfitMe2 <= precE2) break;

            // Stagnation handling: less than 0.01% CUMULATIVE misfit
            // improvement over 2000 passes. The reference is a FROZEN
            // snapshot, refreshed only when the cumulative improvement since
            // the snapshot crosses the threshold: comparing against the
            // continuously-updated running minimum would count
            // slow-but-steady convergence (per-pass improvement below the
            // threshold) as stagnation and abort mid-decay.
            // On the FIRST stagnation the omega state is RESET instead of
            // exiting: the one-way-ish ratchet can pin omega so low that
            // progress stops even though a fresh omega schedule resumes
            // converging (measured: a restarted call continued from a
            // "stagnant" state). Only a second stagnation with a fresh
            // omega is treated as a genuine floor.
            if(instMisfitMe2 < refMisfitMe2 * 0.9999) {
                refMisfitMe2 = instMisfitMe2;
                sinceBest = 0;
                omegaResets = 0;
                if(blowups > 0) blowups--;  // progress amnesty
            }
            else if(++sinceBest >= 2000) {
                if(omegaResets >= 1) break;  // fresh omega also stalled
                omega = (data->omega > 0.0) ? data->omega : 0.3;
                omegaCeiling = 1.0;
                divergeCount = 0;
                convergeStreak = 0;
                prevMisfitMe2 = 1e30;
                sinceBest = 0;
                omegaResets++;
            }

            // Adaptive omega with a ratcheting ceiling. The ceiling also
            // RECOVERS after sustained convergence: a one-way ratchet lets
            // routine nonlinear misfit fluctuations pin omega near omegaMin
            // for the rest of the run (slow creep). Runs ONLY after plain
            // passes: Anderson's misfit sequence is legitimately
            // non-monotonic and would ratchet omega into the floor.
            if(!lastAnderson)
            if(instMisfitMe2 < prevMisfitMe2) {
                // Converging
                divergeCount = 0;
                if(instMisfitMe2 < bestMisfitMe2) bestMisfitMe2 = instMisfitMe2;

                // Slow growth
                omega *= 1.01;
                if(omega > omegaCeiling) omega = omegaCeiling;
                if(++convergeStreak >= 20 && omegaCeiling < 1.0) {
                    omegaCeiling *= 1.01;
                    if(omegaCeiling > 1.0) omegaCeiling = 1.0;
                }
            } else {
                // Diverging
                convergeStreak = 0;
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

            // Step 5: apply the update -- Anderson-accelerated (type II)
            // when history is available, plain damped Jacobi otherwise.
            bool doAnderson = andersonEnabled && cooloff == 0
                              && iter >= kAndersonWarmup;
            if(cooloff > 0) cooloff--;

            if(doAnderson) {
                if(havePrev) {
                    // Append a history column: dF = F_k - F_{k-1},
                    // dX = X_k - X_{k-1} (oldest column dropped when full).
                    if(histLen == andersonM) {
                        for(int j = 0; j < andersonM - 1; j++) {
                            CUDA_CHK(cudaMemcpy(d_dF + (size_t)j * N3,
                                d_dF + (size_t)(j+1) * N3,
                                N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                            CUDA_CHK(cudaMemcpy(d_dX + (size_t)j * N3,
                                d_dX + (size_t)(j+1) * N3,
                                N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                        }
                        histLen--;
                    }
                    vec_diff_kernel<<<blkV, tpb>>>(d_F, d_Fprev,
                        d_dF + (size_t)histLen * N3, N3);
                    vec_diff_kernel<<<blkV, tpb>>>(d_magn, d_Xprev,
                        d_dX + (size_t)histLen * N3, N3);
                    histLen++;
                }
                // Snapshot F_k, X_k (pre-update) for the next column.
                CUDA_CHK(cudaMemcpy(d_Fprev, d_F, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                CUDA_CHK(cudaMemcpy(d_Xprev, d_magn, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                havePrev = true;
            }

            bool applied = false;
            if(doAnderson && histLen > 0) {
                // Regularized normal equations for min ||F - dF*gamma||:
                // G_ij = dF_i . dF_j, b_i = dF_i . F, via batched device dots.
                int np = 0;
                int offsetF = 2 * andersonM * N3;
                for(int i = 0; i < histLen; i++)
                    for(int j = i; j < histLen; j++) {
                        hPa[np] = i * N3; hPb[np] = j * N3; np++;
                    }
                int rhsStart = np;
                for(int i = 0; i < histLen; i++) {
                    hPa[np] = i * N3; hPb[np] = offsetF; np++;
                }
                CUDA_CHK(cudaMemcpy(d_pa, hPa, np * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHK(cudaMemcpy(d_pb, hPb, np * sizeof(int), cudaMemcpyHostToDevice));
                dim3 dotGrid(32, np);
                batched_dot_kernel<<<dotGrid, tpb>>>(d_work, d_pa, d_pb,
                                                     d_dots, np, N3);
                CUDA_CHK(cudaMemcpy(hPartials, d_dots, (size_t)np * 32 * sizeof(double), cudaMemcpyDeviceToHost));
                for(int p = 0; p < np; p++) {
                    double s = 0.0;  // fixed-order sum -> deterministic
                    for(int q = 0; q < 32; q++) s += hPartials[(size_t)p * 32 + q];
                    hDots[p] = s;
                }

                int k = 0;
                for(int i = 0; i < histLen; i++)
                    for(int j = i; j < histLen; j++) {
                        hG[i*histLen + j] = hDots[k];
                        hG[j*histLen + i] = hDots[k];
                        k++;
                    }
                double hB0[RADGPU_ANDERSON_M_MAX];
                for(int i = 0; i < histLen; i++) {
                    hB[i] = hDots[rhsStart + i];
                    hB0[i] = hB[i];  // solve_small_ls overwrites b
                }

                if(andersonDebug >= 2 && iter < kAndersonWarmup + 6) {
                    // Verify the GPU dots against host recomputation.
                    double* hdF = new double[(size_t)(histLen + 1) * N3];
                    CUDA_CHK(cudaMemcpy(hdF, d_dF, (size_t)histLen * N3 * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHK(cudaMemcpy(hdF + (size_t)histLen * N3, d_F, N3 * sizeof(double), cudaMemcpyDeviceToHost));
                    for(int i = 0; i < histLen; i++) {
                        double sb = 0.0;
                        for(int t = 0; t < N3; t++)
                            sb += hdF[(size_t)i * N3 + t] * hdF[(size_t)histLen * N3 + t];
                        double gpu = hDots[rhsStart + i];
                        fprintf(stderr, "[AND dbg] it=%d b[%d]: gpu=%.6e host=%.6e rel=%.1e\n",
                                iter, i, gpu, sb,
                                fabs(gpu - sb) / (fabs(sb) + 1e-300));
                    }
                    delete[] hdF;
                }

                int ok = solve_small_ls(hG, hB, hGamma, histLen);
                if(ok) {
                    // Predicted LS residual ||F - dF*gamma||^2 ~= ||F||^2 -
                    // gamma.b must stay in [0, ||F||^2]; outside = the
                    // normal equations went degenerate despite lambda.
                    double sumF2 = instMisfitMe2 * N;
                    double pred = sumF2;
                    for(int i = 0; i < histLen; i++) pred -= hGamma[i] * hB0[i];
                    if(!(pred >= -0.01 * sumF2 && pred <= 1.01 * sumF2)) ok = 0;
                }
                if(ok) {
                    // Damped extrapolation: cap max|gamma| by scaling the
                    // whole coefficient vector (wild gamma = collinear
                    // history; direction is still useful, magnitude not).
                    double gmax = 0.0;
                    for(int i = 0; i < histLen; i++)
                        if(fabs(hGamma[i]) > gmax) gmax = fabs(hGamma[i]);
                    if(gmax > 2.0) {
                        double sc = 2.0 / gmax;
                        for(int i = 0; i < histLen; i++) hGamma[i] *= sc;
                    }
                    if(andersonDebug >= 3 && iter < kAndersonWarmup + 6) {
                        // Verify the update kernel against a host recompute
                        // on the first few entries.
                        int nchk = 12;
                        double *hx0 = new double[nchk], *hx1 = new double[nchk];
                        double *hf = new double[nchk];
                        double *hdx = new double[(size_t)histLen * nchk];
                        double *hdf = new double[(size_t)histLen * nchk];
                        CUDA_CHK(cudaMemcpy(hx0, d_magn, nchk * sizeof(double), cudaMemcpyDeviceToHost));
                        CUDA_CHK(cudaMemcpy(hf, d_F, nchk * sizeof(double), cudaMemcpyDeviceToHost));
                        for(int j = 0; j < histLen; j++) {
                            CUDA_CHK(cudaMemcpy(hdx + (size_t)j*nchk, d_dX + (size_t)j*N3, nchk * sizeof(double), cudaMemcpyDeviceToHost));
                            CUDA_CHK(cudaMemcpy(hdf + (size_t)j*nchk, d_dF + (size_t)j*N3, nchk * sizeof(double), cudaMemcpyDeviceToHost));
                        }
                        CUDA_CHK(cudaMemcpy(d_gamma, hGamma, histLen * sizeof(double), cudaMemcpyHostToDevice));
                        anderson_update_kernel<<<blkV, tpb>>>(d_magn, d_F, d_dX,
                            d_dF, d_gamma, histLen, andersonBetaCur, N3);
                        CUDA_CHK(cudaMemcpy(hx1, d_magn, nchk * sizeof(double), cudaMemcpyDeviceToHost));
                        for(int i = 0; i < 3; i++) {
                            double exp_v = hx0[i] + andersonBetaCur * hf[i];
                            for(int j = 0; j < histLen; j++)
                                exp_v -= hGamma[j] * (hdx[(size_t)j*nchk + i]
                                         + andersonBetaCur * hdf[(size_t)j*nchk + i]);
                            fprintf(stderr, "[AND upd] it=%d x[%d]: gpu=%.12e host=%.12e\n",
                                    iter, i, hx1[i], exp_v);
                        }
                        delete[] hx0; delete[] hx1; delete[] hf;
                        delete[] hdx; delete[] hdf;
                        applied = true;
                    } else {
                        CUDA_CHK(cudaMemcpy(d_gamma, hGamma, histLen * sizeof(double), cudaMemcpyHostToDevice));
                        anderson_update_kernel<<<blkV, tpb>>>(d_magn, d_F, d_dX,
                            d_dF, d_gamma, histLen, andersonBetaCur, N3);
                        applied = true;
                    }
                    if(andersonDebug >= 1 &&
                       (iter < kAndersonWarmup + 6 || iter % 500 == 0)) {
                        fprintf(stderr, "[AND] it=%d misfit=%.3e omega=%.3f "
                                "hist=%d beta=%.3f g0=%.3e blow=%d\n",
                                iter, sqrt(instMisfitMe2), omega, histLen,
                                andersonBetaCur, hGamma[0], blowups);
                    }
                } else {
                    histLen = 0;  // degenerate LS: drop the history
                }
            }
            if(!applied) {
                axpy_update_kernel<<<blkV, tpb>>>(d_magn, d_F, omega, N3);
            }
            lastAnderson = applied;
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
    // The interaction matrix is OWNED BY THE RESIDENT CACHE (freed on
    // replacement); free it here only if it never made it into the cache.
    if(d_matrix && d_matrix != g_d_matCache) cudaFree(d_matrix);
    if(d_magn) cudaFree(d_magn);
    if(d_magn_new) cudaFree(d_magn_new);
    if(d_field_full) cudaFree(d_field_full);
    if(d_field_out) cudaFree(d_field_out);
    if(d_extField) cudaFree(d_extField);
    if(d_residual) cudaFree(d_residual);
    if(d_work) cudaFree(d_work);
    if(d_dots) cudaFree(d_dots);
    if(d_gamma) cudaFree(d_gamma);
    if(d_pa) cudaFree(d_pa);
    if(d_pb) cudaFree(d_pb);
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

// ============================================================
// Method 11: Newton-Krylov (right-preconditioned GMRES) solver.
//
// Solves the magnetostatic fixed point as a rootfinding problem
//     F(M) = M_mat(H_ext + Q*M) - M = 0
// with the ANALYTIC Jacobian-vector product
//     J*v = D * (Q*v) - v,     D_i = dM_mat/dH |_{H_i}  (3x3 per element)
// so each GMRES iteration costs exactly one cached-matrix matvec (the same
// dominant kernel as one method-9 pass) plus O(N) work. The Newton system
// (I - D*Q) delta = F is solved by GMRES(m) right-preconditioned with the
// per-element blocks (I - D_i*Q_ii)^{-1}, globalized by a backtracking line
// search on ||F||. Rationale: element-wise stationary iteration (methods
// 4/9/10) is marginally stable on high-permeability models (soft
// flux-redistribution modes with spectral radius 1 - O(1/chi), chi ~ 1e3+ in
// low-H iron) and floors above 1e-4; a Krylov method spends ~one iteration
// per soft outlier mode instead of O(chi) passes.
//
// Semantics: the reported misfit is RMS |F| in Tesla -- the PHYSICS residual
// (per-element deviation from the material law under the total field), the
// same quantity methods 4/9/10 drive to zero at their fixed point. The
// returned iteration count is the TOTAL number of matvecs (Newton + GMRES +
// line search + smoothing), directly comparable to method-9 pass counts.
// maxIter caps that matvec budget.
//
// Determinism: every reduction is a fixed-32-block partial kernel summed on
// the host in fixed order (the Anderson lesson: atomicAdd nondeterminism
// grows through the soft modes into macroscopic run-to-run differences).
//
// Env knobs: RADIA_NK_GMRES_M (restart, def 60), RADIA_NK_GMRES_MAX (Krylov
// iters per Newton step, def 400), RADIA_NK_NEWTON_MAX (def 60),
// RADIA_NK_ETA (fixed forcing term; def adaptive Eisenstat-Walker-lite),
// RADIA_NK_PRESMOOTH (damped-Jacobi passes before Newton, def 20),
// RADIA_NK_SMOOTH_FALLBACK (passes after a failed line search, def 50),
// RADIA_NK_DEBUG (1 = per-Newton-step trace).
// ============================================================

// Derivative of the tabulated M(|H|) cubic (same segmentation as
// interp_mh_dev; beyond the table the extrapolation is linear -> end slope).
__device__ double interp_mh_deriv_dev(double absH,
    const double* curveH, const double* curveM, const double* curveDMDH, int len)
{
    if(len <= 0) return 0.0;
    int idx = 0;
    for(int i = 0; i < len; i++) {
        if(curveH[i] > absH) break;
        idx = i;
    }
    if(idx >= len - 1) return curveDMDH[len-1];
    double arg = absH - curveH[idx];
    double step = curveH[idx+1] - curveH[idx];
    double a[4];
    cubpln_dev(step, curveM[idx], curveM[idx+1], curveDMDH[idx], curveDMDH[idx+1], a);
    return a[1] + arg*(2.0*a[2] + 3.0*a[3]*arg);
}

// Secant (absM/|H|) and tangent (d absM/d|H|) susceptibility of the
// isotropic material law; both -> chi0 as |H| -> 0.
__device__ void get_matlaw_dev(
    double absH, int mtype, int elem,
    const double* linKsi,
    const double* mhH, const double* mhM, const double* mhdMdH,
    const int* mhOffset, const int* mhLen,
    const double* formulaMs, const double* formulaKs, const int* formulaLen,
    double* outSc, double* outMp)
{
    const double absHZeroTol = 1e-10;
    if(mtype == 0) {
        *outSc = linKsi[elem];
        *outMp = linKsi[elem];
    }
    else if(mtype == 1) {
        int off = mhOffset[elem];
        int len = mhLen[elem];
        if(absH <= absHZeroTol) {
            double chi0 = (len > 0) ? mhdMdH[off] : 0.0;
            *outSc = chi0; *outMp = chi0;
        } else {
            double am = interp_mh_dev(absH, &mhH[off], &mhM[off], &mhdMdH[off], len);
            *outSc = am / absH;
            *outMp = interp_mh_deriv_dev(absH, &mhH[off], &mhM[off], &mhdMdH[off], len);
        }
    }
    else if(mtype == 3) {
        int flen = formulaLen[elem];
        if(absH <= absHZeroTol) {
            double k = 0.0;
            for(int i = 0; i < flen; i++) k += formulaKs[3*elem+i];
            *outSc = k; *outMp = k;
        } else {
            double am = formula_absM_dev(absH, &formulaMs[3*elem], &formulaKs[3*elem], flen);
            *outSc = am / absH;
            double mp = 0.0;
            for(int i = 0; i < flen; i++) {
                double ms = formulaMs[3*elem+i], ks = formulaKs[3*elem+i];
                if(ms != 0.0) {
                    double c = cosh(ks * absH / ms);
                    mp += ks / (c*c);
                }
            }
            *outMp = mp;
        }
    }
    else {
        *outSc = 0.0; *outMp = 0.0;
    }
}

__device__ bool inv3x3d_dev(const double* a, double* inv)
{
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

// Per-element: physics residual F = M_mat(H) - M (+ |F|^2), tangent blocks
// D = dM_mat/dH, and preconditioner blocks Pinv = (shift*I - D*Q_ii)^{-1}.
// shift = 1 + 1/dtau is the pseudo-transient (Psi-tc) diagonal shift of the
// Newton operator A_tau = shift*I - D*Q; shift = 1 is the pure Newton system.
// For the isotropic law M = sc(|H|)*H + Mr:
//   D = mp * hn hn^T + sc * (I - hn hn^T),  hn = H/|H|   (D = chi0*I at H=0).
__global__ void nk_material_kernel(
    const double* __restrict__ field_full,
    const double* __restrict__ magn,
    double* __restrict__ f_vec,
    double* __restrict__ residual_buf,
    double* __restrict__ Dblocks,
    double* __restrict__ Pinvblocks,
    double shift,
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

    double H[3] = {field_full[3*elem], field_full[3*elem+1], field_full[3*elem+2]};
    double absH = sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]);

    double sc, mp;
    get_matlaw_dev(absH, matType[elem], elem,
        linKsi, mhH, mhM, mhdMdH, mhOffset, mhLen,
        formulaMs, formulaKs, formulaLen, &sc, &mp);

    // Residual F = M_mat(H) - M
    double resid = 0.0;
    for(int c = 0; c < 3; c++) {
        double mmat = sc*H[c] + remMagn[3*elem+c];
        double f = mmat - magn[3*elem+c];
        f_vec[3*elem+c] = f;
        resid += f*f;
    }
    residual_buf[elem] = resid;

    // Tangent D
    double D[9];
    if(absH <= 1e-10) {
        D[0]=sc; D[1]=0;  D[2]=0;
        D[3]=0;  D[4]=sc; D[5]=0;
        D[6]=0;  D[7]=0;  D[8]=sc;
    } else {
        double hn0 = H[0]/absH, hn1 = H[1]/absH, hn2 = H[2]/absH;
        double d = mp - sc;
        D[0] = sc + d*hn0*hn0; D[1] =      d*hn0*hn1; D[2] =      d*hn0*hn2;
        D[3] =      d*hn1*hn0; D[4] = sc + d*hn1*hn1; D[5] =      d*hn1*hn2;
        D[6] =      d*hn2*hn0; D[7] =      d*hn2*hn1; D[8] = sc + d*hn2*hn2;
    }
    for(int k = 0; k < 9; k++) Dblocks[9*elem+k] = D[k];

    // Preconditioner block Pinv = (shift*I - D*Q_ii)^{-1}
    const float* Qii = selfBlocks + 9*elem;
    double A3[9];
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++) {
            double s = 0.0;
            for(int k = 0; k < 3; k++) s += D[3*r+k] * (double)Qii[3*k+c];
            A3[3*r+c] = ((r == c) ? shift : 0.0) - s;
        }
    double Pinv[9];
    if(!inv3x3d_dev(A3, Pinv)) {
        Pinv[0]=1; Pinv[1]=0; Pinv[2]=0;
        Pinv[3]=0; Pinv[4]=1; Pinv[5]=0;
        Pinv[6]=0; Pinv[7]=0; Pinv[8]=1;
    }
    for(int k = 0; k < 9; k++) Pinvblocks[9*elem+k] = Pinv[k];
}

// Recompute the preconditioner blocks Pinv = (shift*I - D*Q_ii)^{-1} from
// the stored tangent blocks D -- used when only dtau (the Psi-tc shift)
// changed, so no matvec / material evaluation is needed.
__global__ void nk_precond_kernel(
    const double* __restrict__ Dblocks,
    const float* __restrict__ selfBlocks,
    double shift,
    double* __restrict__ Pinvblocks,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;
    const double* D = Dblocks + 9*elem;
    const float* Qii = selfBlocks + 9*elem;
    double A3[9];
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++) {
            double s = 0.0;
            for(int k = 0; k < 3; k++) s += D[3*r+k] * (double)Qii[3*k+c];
            A3[3*r+c] = ((r == c) ? shift : 0.0) - s;
        }
    double Pinv[9];
    if(!inv3x3d_dev(A3, Pinv)) {
        Pinv[0]=1; Pinv[1]=0; Pinv[2]=0;
        Pinv[3]=0; Pinv[4]=1; Pinv[5]=0;
        Pinv[6]=0; Pinv[7]=0; Pinv[8]=1;
    }
    for(int k = 0; k < 9; k++) Pinvblocks[9*elem+k] = Pinv[k];
}

// out_i = shift*t_i - D_i * (Qt)_i   (A_tau = shift*I - D*Q, given Qt = Q*t)
__global__ void nk_apply_A_kernel(
    const double* __restrict__ t,
    const double* __restrict__ Qt,
    const double* __restrict__ Dblocks,
    double shift,
    double* __restrict__ out,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;
    const double* D = Dblocks + 9*elem;
    double q0 = Qt[3*elem], q1 = Qt[3*elem+1], q2 = Qt[3*elem+2];
    out[3*elem+0] = shift*t[3*elem+0] - (D[0]*q0 + D[1]*q1 + D[2]*q2);
    out[3*elem+1] = shift*t[3*elem+1] - (D[3]*q0 + D[4]*q1 + D[5]*q2);
    out[3*elem+2] = shift*t[3*elem+2] - (D[6]*q0 + D[7]*q1 + D[8]*q2);
}

// out_i = Pinv_i * v_i
__global__ void nk_apply_Pinv_kernel(
    const double* __restrict__ v,
    const double* __restrict__ Pinvblocks,
    double* __restrict__ out,
    int numElem)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if(elem >= numElem) return;
    const double* P = Pinvblocks + 9*elem;
    double v0 = v[3*elem], v1 = v[3*elem+1], v2 = v[3*elem+2];
    out[3*elem+0] = P[0]*v0 + P[1]*v1 + P[2]*v2;
    out[3*elem+1] = P[3]*v0 + P[4]*v1 + P[5]*v2;
    out[3*elem+2] = P[6]*v0 + P[7]*v1 + P[8]*v2;
}

__global__ void nk_scale_kernel(double* __restrict__ x, double s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) x[i] *= s;
}

// w -= sum_j coef[j] * V[j*pitch + i]
__global__ void nk_minus_lincomb_kernel(
    double* __restrict__ w,
    const double* __restrict__ V,
    const double* __restrict__ coef,
    int ncols, long long pitch, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double s = 0.0;
    for(int j = 0; j < ncols; j++) s += coef[j] * V[(long long)j*pitch + i];
    w[i] -= s;
}

// out = sum_j coef[j] * V[j*pitch + i]
__global__ void nk_lincomb_kernel(
    double* __restrict__ out,
    const double* __restrict__ V,
    const double* __restrict__ coef,
    int ncols, long long pitch, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double s = 0.0;
    for(int j = 0; j < ncols; j++) s += coef[j] * V[(long long)j*pitch + i];
    out[i] = s;
}

#define RADGPU_NK_M_CAP 512

int radGPU_RelaxNK(
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
    int result = -1;

    // --- env config ---
    // GMRES restart default is LARGE on purpose: the near-solution Jacobian
    // carries the near-singular collective soft modes, and restarted GMRES
    // with a small window stagnates completely on them (measured: m=60
    // pinned at relres ~ 1.0 where m=400 converged). Memory is (m+1)*3N
    // doubles -- ~50 MB at 20k elements for m=300, cheap next to the matrix.
    int mRestart = 400, gmresMax = 1200, newtonMax = 200;
    int nPresmooth = 20, nSmoothFallback = 300, nkDebug = 0;
    double etaFixed = -1.0;
    {
        const char* e;
        e = getenv("RADIA_NK_GMRES_M");
        if(e && *e) { int v = atoi(e); if(v >= 5 && v <= RADGPU_NK_M_CAP) mRestart = v; }
        e = getenv("RADIA_NK_GMRES_MAX");
        if(e && *e) { int v = atoi(e); if(v >= mRestart) gmresMax = v; }
        e = getenv("RADIA_NK_NEWTON_MAX");
        if(e && *e) { int v = atoi(e); if(v >= 1) newtonMax = v; }
        e = getenv("RADIA_NK_ETA");
        if(e && *e) { double v = atof(e); if(v > 0.0 && v < 1.0) etaFixed = v; }
        e = getenv("RADIA_NK_PRESMOOTH");
        if(e && *e) { int v = atoi(e); if(v >= 0) nPresmooth = v; }
        e = getenv("RADIA_NK_SMOOTH_FALLBACK");
        if(e && *e) { int v = atoi(e); if(v >= 0) nSmoothFallback = v; }
        e = getenv("RADIA_NK_DEBUG");
        if(e && *e) nkDebug = atoi(e);
    }
    double smoothOmega = (data->omega > 0.0) ? data->omega : 0.3;

    float *d_matrix = nullptr;
    double *d_magn = nullptr, *d_xtrial = nullptr, *d_delta = nullptr;
    double *d_magn_new = nullptr, *d_field_out = nullptr;
    double *d_hfull = nullptr, *d_qt = nullptr, *d_t = nullptr;
    double *d_F = nullptr, *d_resid = nullptr;
    double *d_D = nullptr, *d_Pinv = nullptr;
    double *d_V = nullptr, *d_coef = nullptr;
    double *d_extField = nullptr, *d_zeroExt = nullptr, *d_partials = nullptr;
    int *d_pa = nullptr, *d_pb = nullptr;
    int *d_matType = nullptr;
    double *d_linKsi = nullptr, *d_remMagn = nullptr;
    double *d_mhH = nullptr, *d_mhM = nullptr, *d_mhdMdH = nullptr;
    int *d_mhOffset = nullptr, *d_mhLen = nullptr;
    double *d_formulaMs = nullptr, *d_formulaKs = nullptr;
    int *d_formulaLen = nullptr;
    float *d_selfBlocks = nullptr;

    double *h_resid = new double[N];
    double *hPartials = new double[(RADGPU_NK_M_CAP + 2) * 32];
    int *hPa = new int[RADGPU_NK_M_CAP + 2];
    int *hPb = new int[RADGPU_NK_M_CAP + 2];
    double *hH = new double[(RADGPU_NK_M_CAP + 1) * RADGPU_NK_M_CAP];
    double *hcs = new double[RADGPU_NK_M_CAP + 1];
    double *hsn = new double[RADGPU_NK_M_CAP + 1];
    double *hg = new double[RADGPU_NK_M_CAP + 1];
    double *hy = new double[RADGPU_NK_M_CAP + 1];
    double *hcoef = new double[RADGPU_NK_M_CAP + 1];

    #define CUDA_CHK(call) do { \
        cudaError_t e = (call); \
        if(e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            goto cleanup; \
        } \
    } while(0)

    // --- interaction matrix: same resident cache as method 9 ---
    if(radGPU_MatrixCached(data->matrixStamp, N3)) {
        d_matrix = g_d_matCache;
    } else {
        if(g_d_matCache) {
            cudaFree(g_d_matCache);
            g_d_matCache = nullptr; g_matCacheStamp = 0; g_matCacheDim = 0;
        }
        if(data->h_matrix == nullptr) {
            fprintf(stderr, "radGPU_RelaxNK: matrix neither packed nor cached\n");
            goto cleanup;
        }
        CUDA_CHK(cudaMalloc(&d_matrix, matSize * sizeof(float)));
        CUDA_CHK(cudaMemcpy(d_matrix, data->h_matrix, matSize * sizeof(float), cudaMemcpyHostToDevice));
        g_d_matCache = d_matrix;
        g_matCacheStamp = data->matrixStamp;
        g_matCacheDim = N3;
    }

    CUDA_CHK(cudaMalloc(&d_magn, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_xtrial, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_delta, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_magn_new, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_field_out, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_hfull, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_qt, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_t, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_F, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_resid, N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_D, 9 * (size_t)N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_Pinv, 9 * (size_t)N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_V, (size_t)(mRestart + 1) * N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_coef, (RADGPU_NK_M_CAP + 1) * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_extField, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_zeroExt, N3 * sizeof(double)));
    CUDA_CHK(cudaMemset(d_zeroExt, 0, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_partials, (size_t)(RADGPU_NK_M_CAP + 2) * 32 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_pa, (RADGPU_NK_M_CAP + 2) * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_pb, (RADGPU_NK_M_CAP + 2) * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_matType, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_linKsi, N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_remMagn, N3 * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_mhOffset, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_mhLen, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_formulaMs, 3 * (size_t)N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_formulaKs, 3 * (size_t)N * sizeof(double)));
    CUDA_CHK(cudaMalloc(&d_formulaLen, N * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_selfBlocks, 9 * (size_t)N * sizeof(float)));
    if(data->totalMHPoints > 0) {
        CUDA_CHK(cudaMalloc(&d_mhH, data->totalMHPoints * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_mhM, data->totalMHPoints * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_mhdMdH, data->totalMHPoints * sizeof(double)));
        CUDA_CHK(cudaMemcpy(d_mhH, data->h_mhH, data->totalMHPoints * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_mhM, data->h_mhM, data->totalMHPoints * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_mhdMdH, data->h_mhdMdH, data->totalMHPoints * sizeof(double), cudaMemcpyHostToDevice));
    }

    CUDA_CHK(cudaMemcpy(d_magn, data->h_magn, N3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_extField, data->h_extField, N3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_matType, data->h_matType, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_linKsi, data->h_linKsi, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_remMagn, data->h_remMagn, N3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mhOffset, data->h_mhOffset, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mhLen, data->h_mhLen, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_formulaMs, data->h_formulaMs, 3 * (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_formulaKs, data->h_formulaKs, 3 * (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_formulaLen, data->h_formulaLen, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_selfBlocks, data->h_selfBlocks, 9 * (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    {
        int tpb = 256;
        int warpsPerBlock = tpb / 32;
        int blkMV = (N3 + warpsPerBlock - 1) / warpsPerBlock;
        int blkEl = (N + tpb - 1) / tpb;
        int blkV = (N3 + tpb - 1) / tpb;

        int matvecs = 0;

        // ---- deterministic dot helper (pairs of columns inside d_V, or any
        // pair of vectors expressed as offsets from d_V) ----
        // Implemented inline below via batched_dot_kernel + fixed-order sum.

        // ---- pre-smoothing: a few damped Jacobi passes (also initializes
        // the persistent per-element H for the implicit-solve kernel) ----
        if(nPresmooth > 0) {
            matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                d_matrix, d_magn, d_extField, d_field_out, N3);
            matvecs++;
            for(int s = 0; s < nPresmooth && matvecs < maxIter; s++) {
                matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                    d_matrix, d_magn, d_extField, d_hfull, N3);
                implicit_solve_kernel<<<blkEl, tpb>>>(
                    d_hfull, d_magn, d_magn_new, d_field_out,
                    d_selfBlocks, d_matType, d_linKsi, d_remMagn,
                    d_mhH, d_mhM, d_mhdMdH, d_mhOffset, d_mhLen,
                    d_formulaMs, d_formulaKs, d_formulaLen, N);
                proposal_residual_kernel<<<blkEl, tpb>>>(
                    d_magn, d_magn_new, d_F, d_resid, N);
                axpy_update_kernel<<<blkV, tpb>>>(d_magn, d_F, smoothOmega, N3);
                matvecs++;
            }
        }

        // ---- F, D, Pinv at the current state ----
        // Pseudo-transient continuation (Psi-tc / SER): the Newton operator
        // is shifted to A_tau = (1 + 1/dtau)*I - D*Q. Small dtau ~ heavily
        // damped implicit step (regularizes the near-singular soft modes and
        // keeps GMRES fast); dtau grows as ||F|| falls (SER), so the method
        // finishes as pure Newton with quadratic contraction.
        double dtau = 1.0;
        {
            const char* e = getenv("RADIA_NK_DTAU0");
            if(e && *e) { double v = atof(e); if(v > 0.0) dtau = v; }
        }
        const double dtauMin = 1e-3, dtauMax = 1e9, dtauGrowCap = 10.0;
        double shiftCur = 1.0 + 1.0/dtau;
        double misfit = 1e30, normF = 1e30, misfitPrev = 1e30;
        // Nonmonotone (Grippo) line-search memory: the center-collocation
        // Jacobian is indefinite around the pathological element clusters,
        // so ||F|| has local minima that are NOT roots; a monotone Armijo
        // search parks there (measured). Accepting against the max of the
        // last few misfits lets the iteration climb out while Newton still
        // contracts on average.
        const int kHistLen = 5;
        double misfitHist[kHistLen];
        int histN = 0, histPos = 0;
        // Forcing-term ceiling, tightened when a step is rejected: on an
        // indefinite system a LOOSELY solved Newton step can be an ascent
        // direction even when the exact step is fine.
        double etaCap = 0.1;
        #define NK_EVAL_FDP(XPTR, SHIFT) do { \
            matvec_add_extfield_kernel<<<blkMV, tpb>>>( \
                d_matrix, (XPTR), d_extField, d_hfull, N3); \
            nk_material_kernel<<<blkEl, tpb>>>( \
                d_hfull, (XPTR), d_F, d_resid, d_D, d_Pinv, (SHIFT), \
                d_selfBlocks, d_matType, d_linKsi, d_remMagn, \
                d_mhH, d_mhM, d_mhdMdH, d_mhOffset, d_mhLen, \
                d_formulaMs, d_formulaKs, d_formulaLen, N); \
            CUDA_CHK(cudaMemcpy(h_resid, d_resid, N * sizeof(double), cudaMemcpyDeviceToHost)); \
            double sumF2 = 0.0; \
            for(int i = 0; i < N; i++) sumF2 += h_resid[i]; \
            misfit = sqrt(sumF2 / N); \
            normF = sqrt(sumF2); \
            matvecs++; \
        } while(0)

        NK_EVAL_FDP(d_magn, shiftCur);
        misfitPrev = misfit;
        if(!(misfit == misfit)) { // NaN state
            fprintf(stderr, "radGPU_RelaxNK: non-finite residual at start\n");
            goto cleanup;
        }

        // ---- debug >= 5: finite-difference Jacobian self-test at the
        // current state. J*v (analytic: D*(Q*v) - v) is compared against
        // (F(X + eps*v) - F(X))/eps for a deterministic pseudo-random v. ----
        if(nkDebug >= 5) {
            // v: fixed pseudo-random direction, |v| ~ 1e-3 T per component
            double* hv = new double[N3];
            unsigned long long sd = 12345;
            for(int i = 0; i < N3; i++) {
                sd = sd * 6364136223846793005ULL + 1442695040888963407ULL;
                hv[i] = 1e-3 * (2.0 * ((sd >> 11) * (1.0/9007199254740992.0)) - 1.0);
            }
            CUDA_CHK(cudaMemcpy(d_t, hv, N3 * sizeof(double), cudaMemcpyHostToDevice));
            // analytic J*v = D*(Q*v) - v  -> store -(A v) = J v in d_delta
            // (shift = 1: the pure Newton operator is the physical Jacobian)
            matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                d_matrix, d_t, d_zeroExt, d_qt, N3);
            nk_apply_A_kernel<<<blkEl, tpb>>>(d_t, d_qt, d_D, 1.0, d_delta, N);
            nk_scale_kernel<<<blkV, tpb>>>(d_delta, -1.0, N3);
            // save F(X)
            CUDA_CHK(cudaMemcpy(d_V, d_F, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
            // F(X + eps v)
            const double eps = 1e-4;
            CUDA_CHK(cudaMemcpy(d_xtrial, d_magn, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
            axpy_update_kernel<<<blkV, tpb>>>(d_xtrial, d_t, eps, N3);
            double msave = misfit, nsave = normF;
            NK_EVAL_FDP(d_xtrial, shiftCur);
            misfit = msave; normF = nsave;
            // fd = (F1 - F0)/eps  vs  Jv (in d_delta)
            double* hfd = new double[N3];
            double* hf0 = new double[N3];
            double* hjv = new double[N3];
            CUDA_CHK(cudaMemcpy(hfd, d_F, N3 * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHK(cudaMemcpy(hf0, d_V, N3 * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHK(cudaMemcpy(hjv, d_delta, N3 * sizeof(double), cudaMemcpyDeviceToHost));
            double num = 0.0, den = 0.0;
            for(int i = 0; i < N3; i++) {
                double fd = (hfd[i] - hf0[i]) / eps;
                double d = fd - hjv[i];
                num += d * d;
                den += hjv[i] * hjv[i];
            }
            fprintf(stderr, "[NK fdtest] ||FD - Jv|| / ||Jv|| = %.3e (eps=%.1e)\n",
                    sqrt(num) / (sqrt(den) + 1e-300), eps);
            delete[] hv; delete[] hfd; delete[] hf0; delete[] hjv;
            // restore F/D/Pinv at X
            NK_EVAL_FDP(d_magn, shiftCur);
        }

        int nkFails = 0;
        int newton = 0;
        for(; newton < newtonMax; newton++) {
            if(misfit <= precision) break;
            if(matvecs >= maxIter) break;

            // forcing term (Eisenstat-Walker-lite; the Psi-tc shift keeps
            // the shifted systems cheap, so cap eta tightly)
            double eta;
            if(etaFixed > 0.0) eta = etaFixed;
            else {
                double r = misfit / misfitPrev;
                eta = 0.5 * r * r;
                if(newton == 0) eta = 0.01;
                if(eta < 1e-3) eta = 1e-3;
                if(eta > etaCap) eta = etaCap;
            }

            // ---- GMRES(m) on (I - D*Q) P^{-1} u = F;  delta = P^{-1} (V y) ----
            CUDA_CHK(cudaMemset(d_delta, 0, N3 * sizeof(double)));
            bool solved = false;
            int cycles = 0, gmIters = 0;
            double relres = 1e30;

            while(!solved && gmIters < gmresMax && matvecs < maxIter) {
                double* Vcol0 = d_V;
                if(cycles == 0) {
                    CUDA_CHK(cudaMemcpy(Vcol0, d_F, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                } else {
                    // r = F - A*delta
                    matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                        d_matrix, d_delta, d_zeroExt, d_qt, N3);
                    matvecs++;
                    nk_apply_A_kernel<<<blkEl, tpb>>>(d_delta, d_qt, d_D, shiftCur, Vcol0, N);
                    vec_diff_kernel<<<blkV, tpb>>>(d_F, Vcol0, Vcol0, N3);
                }
                // rnorm
                double rnorm;
                {
                    hPa[0] = 0; hPb[0] = 0;
                    CUDA_CHK(cudaMemcpy(d_pa, hPa, sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHK(cudaMemcpy(d_pb, hPb, sizeof(int), cudaMemcpyHostToDevice));
                    dim3 dotGrid(32, 1);
                    batched_dot_kernel<<<dotGrid, tpb>>>(d_V, d_pa, d_pb, d_partials, 1, N3);
                    CUDA_CHK(cudaMemcpy(hPartials, d_partials, 32 * sizeof(double), cudaMemcpyDeviceToHost));
                    double s = 0.0;
                    for(int q = 0; q < 32; q++) s += hPartials[q];
                    rnorm = sqrt(s > 0.0 ? s : 0.0);
                }
                if(rnorm <= eta * normF || rnorm < 1e-300) { solved = true; break; }
                nk_scale_kernel<<<blkV, tpb>>>(Vcol0, 1.0 / rnorm, N3);
                hg[0] = rnorm;
                for(int i = 1; i <= mRestart; i++) hg[i] = 0.0;

                int j = 0;
                for(; j < mRestart && gmIters < gmresMax && matvecs < maxIter; ) {
                    double* Vj = d_V + (size_t)j * N3;
                    double* Vj1 = d_V + (size_t)(j+1) * N3;
                    // w = A P^{-1} V_j
                    nk_apply_Pinv_kernel<<<blkEl, tpb>>>(Vj, d_Pinv, d_t, N);
                    matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                        d_matrix, d_t, d_zeroExt, d_qt, N3);
                    matvecs++;
                    gmIters++;
                    nk_apply_A_kernel<<<blkEl, tpb>>>(d_t, d_qt, d_D, shiftCur, Vj1, N);

                    // CGS2 orthogonalization against V_0..V_j
                    int np = j + 1;
                    double h1[RADGPU_NK_M_CAP + 1], h2[RADGPU_NK_M_CAP + 1];
                    for(int pass = 0; pass < 2; pass++) {
                        double* hout = (pass == 0) ? h1 : h2;
                        for(int i = 0; i < np; i++) { hPa[i] = i * N3; hPb[i] = (j+1) * N3; }
                        CUDA_CHK(cudaMemcpy(d_pa, hPa, np * sizeof(int), cudaMemcpyHostToDevice));
                        CUDA_CHK(cudaMemcpy(d_pb, hPb, np * sizeof(int), cudaMemcpyHostToDevice));
                        dim3 dotGrid(32, np);
                        batched_dot_kernel<<<dotGrid, tpb>>>(d_V, d_pa, d_pb, d_partials, np, N3);
                        CUDA_CHK(cudaMemcpy(hPartials, d_partials, (size_t)np * 32 * sizeof(double), cudaMemcpyDeviceToHost));
                        for(int p = 0; p < np; p++) {
                            double s = 0.0;
                            for(int q = 0; q < 32; q++) s += hPartials[(size_t)p * 32 + q];
                            hout[p] = s;
                        }
                        CUDA_CHK(cudaMemcpy(d_coef, hout, np * sizeof(double), cudaMemcpyHostToDevice));
                        nk_minus_lincomb_kernel<<<blkV, tpb>>>(Vj1, d_V, d_coef, np, N3, N3);
                    }
                    for(int i = 0; i < np; i++) hH[i * RADGPU_NK_M_CAP + j] = h1[i] + h2[i];

                    // ||w||
                    double wnorm;
                    {
                        hPa[0] = (j+1) * N3; hPb[0] = (j+1) * N3;
                        CUDA_CHK(cudaMemcpy(d_pa, hPa, sizeof(int), cudaMemcpyHostToDevice));
                        CUDA_CHK(cudaMemcpy(d_pb, hPb, sizeof(int), cudaMemcpyHostToDevice));
                        dim3 dotGrid(32, 1);
                        batched_dot_kernel<<<dotGrid, tpb>>>(d_V, d_pa, d_pb, d_partials, 1, N3);
                        CUDA_CHK(cudaMemcpy(hPartials, d_partials, 32 * sizeof(double), cudaMemcpyDeviceToHost));
                        double s = 0.0;
                        for(int q = 0; q < 32; q++) s += hPartials[q];
                        wnorm = sqrt(s > 0.0 ? s : 0.0);
                    }
                    hH[(j+1) * RADGPU_NK_M_CAP + j] = wnorm;
                    bool breakdown = (wnorm <= 1e-300);
                    if(!breakdown) nk_scale_kernel<<<blkV, tpb>>>(Vj1, 1.0 / wnorm, N3);

                    // Givens rotations
                    for(int i = 0; i < j; i++) {
                        double a = hH[i * RADGPU_NK_M_CAP + j];
                        double b = hH[(i+1) * RADGPU_NK_M_CAP + j];
                        hH[i * RADGPU_NK_M_CAP + j]     =  hcs[i]*a + hsn[i]*b;
                        hH[(i+1) * RADGPU_NK_M_CAP + j] = -hsn[i]*a + hcs[i]*b;
                    }
                    double a = hH[j * RADGPU_NK_M_CAP + j];
                    double b = hH[(j+1) * RADGPU_NK_M_CAP + j];
                    double denom = sqrt(a*a + b*b);
                    if(denom < 1e-300) break; // dependent direction: drop col j
                    hcs[j] = a / denom; hsn[j] = b / denom;
                    hH[j * RADGPU_NK_M_CAP + j] = denom;
                    hH[(j+1) * RADGPU_NK_M_CAP + j] = 0.0;
                    hg[j+1] = -hsn[j] * hg[j];
                    hg[j]   =  hcs[j] * hg[j];
                    relres = fabs(hg[j+1]);
                    j++;
                    if(relres <= eta * normF) break;
                    if(breakdown) break;
                }

                // solve y, accumulate delta += P^{-1} (V y)
                if(j > 0) {
                    for(int k = j - 1; k >= 0; k--) {
                        double s = hg[k];
                        for(int l = k + 1; l < j; l++) s -= hH[k * RADGPU_NK_M_CAP + l] * hy[l];
                        hy[k] = s / hH[k * RADGPU_NK_M_CAP + k];
                    }
                    for(int k = 0; k < j; k++) hcoef[k] = hy[k];
                    CUDA_CHK(cudaMemcpy(d_coef, hcoef, j * sizeof(double), cudaMemcpyHostToDevice));
                    nk_lincomb_kernel<<<blkV, tpb>>>(d_t, d_V, d_coef, j, N3, N3);
                    nk_apply_Pinv_kernel<<<blkEl, tpb>>>(d_t, d_Pinv, d_qt, N);
                    axpy_update_kernel<<<blkV, tpb>>>(d_delta, d_qt, 1.0, N3);
                } else break; // no Krylov progress possible

                if(relres <= eta * normF) solved = true;
                cycles++;
            }

            // ---- backtracking line search on ||F||: MONOTONE while the
            // continuation is healthy (a nonmonotone reference lets the
            // iteration wander -- measured); the relaxed Grippo reference
            // (max of recent accepted misfits) engages only while escaping
            // a rejection, to allow climbing out of merit-function minima
            // of the indefinite collocation Jacobian.
            double misfitRef = misfit;
            if(nkFails > 0)
                for(int i = 0; i < histN; i++)
                    if(misfitHist[i] > misfitRef) misfitRef = misfitHist[i];
            bool accepted = false;
            double alpha = 1.0, misfitT = misfit, alphaAcc = 0.0;
            for(int ls = 0; ls < 8 && matvecs < maxIter; ls++, alpha *= 0.5) {
                CUDA_CHK(cudaMemcpy(d_xtrial, d_magn, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                axpy_update_kernel<<<blkV, tpb>>>(d_xtrial, d_delta, alpha, N3);
                double misfitSave = misfit, normFSave = normF;
                NK_EVAL_FDP(d_xtrial, shiftCur);
                misfitT = misfit;
                misfit = misfitSave; normF = normFSave;
                if(misfitT == misfitT && misfitT < misfitRef * (1.0 - 1e-4 * alpha)) {
                    CUDA_CHK(cudaMemcpy(d_magn, d_xtrial, N3 * sizeof(double), cudaMemcpyDeviceToDevice));
                    misfitPrev = misfit;
                    misfit = misfitT;
                    normF = misfitT * sqrt((double)N);
                    accepted = true;
                    alphaAcc = alpha;
                    break;
                }
            }

            if(nkDebug) {
                fprintf(stderr, "[NK] newton=%d misfit=%.3e dtau=%.2e eta=%.2e gmres=%d "
                        "relres/||F||=%.2e alpha=%.3g acc=%d matvecs=%d\n",
                        newton, misfit, dtau, eta, gmIters,
                        (normF > 0.0) ? relres / normF : 0.0, alphaAcc, (int)accepted, matvecs);
            }

            if(accepted) {
                nkFails = 0;
                misfitHist[histPos] = misfit;
                histPos = (histPos + 1) % kHistLen;
                if(histN < kHistLen) histN++;
                if(etaCap < 0.1) { etaCap *= 1.3; if(etaCap > 0.1) etaCap = 0.1; }
                // SER: dtau grows with the residual reduction; a full,
                // undamped accepted step earns the full SER growth, damped
                // steps earn proportionally less.
                double grow = (misfitPrev / misfit) * (alphaAcc > 0.0 ? alphaAcc : 1.0);
                if(grow > dtauGrowCap) grow = dtauGrowCap;
                if(grow > 1.0) dtau *= grow;
                if(dtau > dtauMax) dtau = dtauMax;
                double newShift = 1.0 + 1.0/dtau;
                if(newShift != shiftCur) {
                    shiftCur = newShift;
                    // D at the accepted state is current; only Pinv depends
                    // on the shift -- refresh it without a matvec.
                    nk_precond_kernel<<<blkEl, tpb>>>(d_D, d_selfBlocks, shiftCur, d_Pinv, N);
                }
                continue;
            }

            // ---- line search failed even against the nonmonotone
            // reference: solve the next system tighter (an eta-loose step on
            // an indefinite system can be ascent) and shrink dtau. On the
            // 4th consecutive failure, jiggle: a block of damped Jacobi
            // passes moves the state off the merit-function saddle (Jacobi
            // does not descend on ||F||, which is exactly what is needed
            // here). Give up only when that too changes nothing. ----
            nkFails++;
            etaCap *= 0.3;
            if(etaCap < 2e-3) etaCap = 2e-3;
            dtau *= 0.25;
            if(dtau < dtauMin) dtau = dtauMin;
            shiftCur = 1.0 + 1.0/dtau;
            if(nkFails == 4 && matvecs + nSmoothFallback < maxIter) {
                double misfitBefore = misfit;
                matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                    d_matrix, d_magn, d_extField, d_field_out, N3);
                matvecs++;
                for(int s = 0; s < nSmoothFallback && matvecs < maxIter; s++) {
                    matvec_add_extfield_kernel<<<blkMV, tpb>>>(
                        d_matrix, d_magn, d_extField, d_hfull, N3);
                    implicit_solve_kernel<<<blkEl, tpb>>>(
                        d_hfull, d_magn, d_magn_new, d_field_out,
                        d_selfBlocks, d_matType, d_linKsi, d_remMagn,
                        d_mhH, d_mhM, d_mhdMdH, d_mhOffset, d_mhLen,
                        d_formulaMs, d_formulaKs, d_formulaLen, N);
                    proposal_residual_kernel<<<blkEl, tpb>>>(
                        d_magn, d_magn_new, d_F, d_resid, N);
                    axpy_update_kernel<<<blkV, tpb>>>(d_magn, d_F, 0.15, N3);
                    matvecs++;
                }
                dtau = 1.0;
                shiftCur = 1.0 + 1.0/dtau;
                NK_EVAL_FDP(d_magn, shiftCur);
                misfitPrev = misfit;
                // the jiggle moved the state: give the continuation a
                // fresh chance (nonmonotone history keeps the old highs)
                if(fabs(misfit - misfitBefore) > 0.01 * misfitBefore) nkFails = 0;
                if(nkDebug) fprintf(stderr, "[NK] jiggle: misfit %.3e -> %.3e\n",
                                    misfitBefore, misfit);
                continue;
            }
            NK_EVAL_FDP(d_magn, shiftCur);
            misfitPrev = misfit;
            if(nkFails >= 8 || matvecs >= maxIter) break;
        }

        // ---- final state: H at X for the h_field output ----
        matvec_add_extfield_kernel<<<blkMV, tpb>>>(
            d_matrix, d_magn, d_extField, d_hfull, N3);
        CUDA_CHK(cudaMemcpy(data->h_magn, d_magn, N3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(data->h_field, d_hfull, N3 * sizeof(double), cudaMemcpyDeviceToHost));

        *outMisfitM = misfit;
        double maxModM = 0, maxModH = 0;
        for(int i = 0; i < N; i++) {
            double mx = data->h_magn[3*i], my = data->h_magn[3*i+1], mz = data->h_magn[3*i+2];
            double mm = sqrt(mx*mx + my*my + mz*mz);
            if(mm > maxModM) maxModM = mm;
            double hx = data->h_field[3*i], hy_ = data->h_field[3*i+1], hz = data->h_field[3*i+2];
            double hm = sqrt(hx*hx + hy_*hy_ + hz*hz);
            if(hm > maxModH) maxModH = hm;
        }
        *outMaxModM = maxModM;
        *outMaxModH = maxModH;
        result = (matvecs > 0) ? matvecs : 1;

        #undef NK_EVAL_FDP
    }

cleanup:
    delete[] h_resid;
    delete[] hPartials;
    delete[] hPa;
    delete[] hPb;
    delete[] hH;
    delete[] hcs;
    delete[] hsn;
    delete[] hg;
    delete[] hy;
    delete[] hcoef;
    if(d_matrix && d_matrix != g_d_matCache) cudaFree(d_matrix);
    if(d_magn) cudaFree(d_magn);
    if(d_xtrial) cudaFree(d_xtrial);
    if(d_delta) cudaFree(d_delta);
    if(d_magn_new) cudaFree(d_magn_new);
    if(d_field_out) cudaFree(d_field_out);
    if(d_hfull) cudaFree(d_hfull);
    if(d_qt) cudaFree(d_qt);
    if(d_t) cudaFree(d_t);
    if(d_F) cudaFree(d_F);
    if(d_resid) cudaFree(d_resid);
    if(d_D) cudaFree(d_D);
    if(d_Pinv) cudaFree(d_Pinv);
    if(d_V) cudaFree(d_V);
    if(d_coef) cudaFree(d_coef);
    if(d_extField) cudaFree(d_extField);
    if(d_zeroExt) cudaFree(d_zeroExt);
    if(d_partials) cudaFree(d_partials);
    if(d_pa) cudaFree(d_pa);
    if(d_pb) cudaFree(d_pb);
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