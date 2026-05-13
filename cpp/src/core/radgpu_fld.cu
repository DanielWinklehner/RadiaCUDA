/************************************************************************//**
 * File: radgpu_fld.cu
 * Description: CUDA kernels for GPU-accelerated magnetic field evaluation.
 *              Computes B field from uniformly magnetized polyhedra and
 *              rectangular parallelepipeds decomposed into polygon faces.
 *
 *              Kernel uses the atan2-based solid angle formulation
 *              (matching field_kernel.py _POLY_KERNEL_FP64).
 *
 * Project: RadiaCUDA
 * First release: 2026
 *
 * @authors D. Winklehner, Claude
 ***************************************************************************/

#ifdef RADIA_WITH_CUDA

#include "radgpu_fld.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>


// ===================== NEW HELPERS (add above radGPU_FldKernel) =====================
__device__ inline double radgpu_log_R_plus_u_stable(double R, double u, double q)
{
    // Computes log(R + u) robustly.
    // q must satisfy: q = R^2 - u^2 >= 0 (analytically).
    double rp = R + u;
    if (rp > 1e-12 * R) return log(rp); // safe direct path

    double rm = R - u;
    if (!(rm > 0.0) || !isfinite(rm)) rm = 1.0e-300;
    if (!(q  > 0.0) || !isfinite(q))  q  = 1.0e-300;

    // log(R+u) = log(q) - log(R-u), avoids cancellation when R+u is tiny
    return log(q) - log(rm);
}

__device__ inline double radgpu_clamp(double v, double lo, double hi)
{
    return fmin(hi, fmax(lo, v));
}

__device__ inline double radgpu_face_scale(
    const double* __restrict__ verts2d, int vbase, int nv)
{
    double s = 0.0;
    for (int i = 0; i < nv; i++) {
        int j = (i + 1) % nv;
        double dx = verts2d[vbase + j*2 + 0] - verts2d[vbase + i*2 + 0];
        double dy = verts2d[vbase + j*2 + 1] - verts2d[vbase + i*2 + 1];
        double L = sqrt(dx*dx + dy*dy);
        if (L > s) s = L;
    }
    if (s < 1e-30) s = 1.0;
    return s;
}

__device__ inline bool radgpu_point_in_poly_2d(
    const double* __restrict__ verts2d, int vbase, int nv, double px, double py)
{
    bool inside = false;
    for (int i = 0, j = nv - 1; i < nv; j = i++) {
        double xi = verts2d[vbase + i*2 + 0];
        double yi = verts2d[vbase + i*2 + 1];
        double xj = verts2d[vbase + j*2 + 0];
        double yj = verts2d[vbase + j*2 + 1];

        bool crosses = ((yi > py) != (yj > py));
        if (crosses) {
            double t = (py - yi) / (yj - yi);
            double xint = xi + t * (xj - xi);
            if (xint > px) inside = !inside;
        }
    }
    return inside;
}

__device__ inline double radgpu_min_dist2_edges_2d(
    const double* __restrict__ verts2d, int vbase, int nv, double px, double py)
{
    double min_d2 = 1.0e300;
    for (int i = 0; i < nv; i++) {
        int j = (i + 1) % nv;
        double ax = verts2d[vbase + i*2 + 0];
        double ay = verts2d[vbase + i*2 + 1];
        double bx = verts2d[vbase + j*2 + 0];
        double by = verts2d[vbase + j*2 + 1];

        double vx = bx - ax, vy = by - ay;
        double wx = px - ax, wy = py - ay;

        double vv = vx*vx + vy*vy;
        double t = (vv > 0.0) ? (wx*vx + wy*vy) / vv : 0.0;
        t = radgpu_clamp(t, 0.0, 1.0);

        double qx = ax + t*vx;
        double qy = ay + t*vy;
        double dx = px - qx, dy = py - qy;
        double d2 = dx*dx + dy*dy;
        if (d2 < min_d2) min_d2 = d2;
    }
    return min_d2;
}

__device__ inline double radgpu_nudge_small(double v, double eps)
{
    if (fabs(v) < eps) return (v < 0.0 ? -eps : +eps);
    return v;
}

__device__ inline void radgpu_eval_face_integrals_at_z(
    const double* __restrict__ verts2d,
    int vbase, int nv,
    double lx, double ly, double z,
    double eps_xy, double eps_b,
    double& Sx_out, double& Sy_out, double& Sz_out)
{
    const double Max_k = 1.0e+09;
    const double RelRandMagn = 1.0e-13;
    const double MaxRelTolToSwitch = 1.0e-07;

    if (nv < 3) {
        Sx_out = 0.0; Sy_out = 0.0; Sz_out = 0.0;
        return;
    }

    double ze2 = z * z;
    double Sx = 0.0, Sy = 0.0, Sz = 0.0;
    double Sx_log_extra = 0.0; // replaces ArgSumLogs2 product path

    double x1 = radgpu_nudge_small(verts2d[vbase + 0] - lx, eps_xy);
    double y1 = radgpu_nudge_small(verts2d[vbase + 1] - ly, eps_xy);
    double x1e2 = x1 * x1;

    for (int ei = 0; ei < nv; ei++)
    {
        int vnext = vbase + ((ei + 1) % nv) * 2;
        double x2 = radgpu_nudge_small(verts2d[vnext + 0] - lx, eps_xy);
        double y2 = radgpu_nudge_small(verts2d[vnext + 1] - ly, eps_xy);
        double x2e2 = x2 * x2;

        double x2mx1 = x2 - x1;
        double y2my1 = y2 - y1;

        if (fabs(x2mx1) * Max_k > fabs(y2my1))
        {
            double k = y2my1 / x2mx1;
            double b = radgpu_nudge_small(y1 - k * x1, eps_b);

            double ke2 = k * k;
            double be2 = b * b;
            double ke2p1 = ke2 + 1.0;
            double sqrtke2p1 = sqrt(ke2p1);
            double bk = b * k;

            double bpkx1 = b + k * x1;
            double bpkx2 = b + k * x2;
            double bpkx1e2 = bpkx1 * bpkx1;
            double bpkx2e2 = bpkx2 * bpkx2;

            double R1 = sqrt(x1e2 + bpkx1e2 + ze2);
            double R2 = sqrt(x2e2 + bpkx2e2 + ze2);

            double R1pbpkx1 = bpkx1 + R1;
            double R2pbpkx2 = bpkx2 + R2;

            // keep your existing R+... protection (important for A1..A4 path)
            double AbsRandR1 = 100.0 * R1 * RelRandMagn;
            double AbsRandR2 = 100.0 * R2 * RelRandMagn;
            double MaxAbsRandR1 = MaxRelTolToSwitch * R1;
            double MaxAbsRandR2 = MaxRelTolToSwitch * R2;
            if (AbsRandR1 > MaxAbsRandR1) AbsRandR1 = MaxAbsRandR1;
            if (AbsRandR2 > MaxAbsRandR2) AbsRandR2 = MaxAbsRandR2;

            double x1e2pze2 = x1e2 + ze2;
            if (fabs(R1pbpkx1) < AbsRandR1 && R1 > 100.0 * AbsRandR1 &&
                x1e2pze2 < bpkx1e2 * MaxRelTolToSwitch)
                R1pbpkx1 = (bpkx1 != 0.0) ? 0.5 * x1e2pze2 / fabs(bpkx1) : 1.0e-50;

            if (fabs(R2pbpkx2) < AbsRandR2 && R2 > 100.0 * AbsRandR2 &&
                (x2e2 + ze2) < bpkx2e2 * MaxRelTolToSwitch)
                R2pbpkx2 = (bpkx2 != 0.0) ? 0.5 * (x2e2 + ze2) / fabs(bpkx2) : 1.0e-50;

            if (R1pbpkx1 == 0.0) R1pbpkx1 = 1.0e-50;
            if (R2pbpkx2 == 0.0) R2pbpkx2 = 1.0e-50;

            // Sz (atan2 accumulation) as before
            double kze2 = k * ze2;
            double ke2ze2 = k * kze2;
            double ke2ze2pbe2 = ke2ze2 + be2;
            double ke2ze2mbe2 = ke2ze2 - be2;
            double bx1 = b * x1, bx2 = b * x2;
            double twob = 2.0 * b;
            double kx1mb = k * x1 - b, kx2mb = k * x2 - b;

            double A1 = -(ke2ze2pbe2 * (bx1 + kze2) * R1pbpkx1 + kze2 * twob * (x1e2 + ze2));
            double A2 =  (ke2ze2pbe2 * kx1mb * R1pbpkx1 + ke2ze2mbe2 * (x1e2 + ze2)) * z;
            double A3 =   ke2ze2pbe2 * (bx2 + kze2) * R2pbpkx2 + kze2 * twob * (x2e2 + ze2);
            double A4 =  (ke2ze2pbe2 * kx2mb * R2pbpkx2 + ke2ze2mbe2 * (x2e2 + ze2)) * z;

            Sz += atan2(A1 * A4 + A3 * A2, A2 * A4 - A1 * A3);

            // -------- stable SL1 --------
            // u = (bk + (1+k^2)x)/sqrt(1+k^2)
            double u1 = (bk + ke2p1 * x1) / sqrtke2p1;
            double u2 = (bk + ke2p1 * x2) / sqrtke2p1;

            // qv = R^2 - u^2 = z^2 + b^2/(1+k^2)
            double qv = ze2 + be2 / ke2p1;
            if (!(qv > 0.0) || !isfinite(qv)) qv = 1.0e-300;

            double logv1 = radgpu_log_R_plus_u_stable(R1, u1, qv);
            double logv2 = radgpu_log_R_plus_u_stable(R2, u2, qv);

            double SL1 = (logv2 - logv1) / sqrtke2p1;

            Sx += -k * SL1;
            Sy +=  SL1;

            // -------- stable replacement for ArgSumLogs2 --------
            // log(R + (b+kx)) with q = R^2 - (b+kx)^2 = x^2 + z^2
            double qx1 = x1e2 + ze2;
            double qx2 = x2e2 + ze2;
            if (!(qx1 > 0.0) || !isfinite(qx1)) qx1 = 1.0e-300;
            if (!(qx2 > 0.0) || !isfinite(qx2)) qx2 = 1.0e-300;

            double logrp1 = radgpu_log_R_plus_u_stable(R1, bpkx1, qx1);
            double logrp2 = radgpu_log_R_plus_u_stable(R2, bpkx2, qx2);

            Sx_log_extra += (logrp2 - logrp1);
        }

        x1 = x2; y1 = y2; x1e2 = x2e2;
    }

    Sx += Sx_log_extra;

    if (!isfinite(Sx)) Sx = 0.0;
    if (!isfinite(Sy)) Sy = 0.0;
    if (!isfinite(Sz)) Sz = 0.0;

    Sx_out = Sx;
    Sy_out = Sy;
    Sz_out = Sz;
}

__global__
void radGPU_FldKernel(
    const double* __restrict__ verts2d,       // [n_faces * MAX_VERTS * 2]
    const int*    __restrict__ nverts,        // [n_faces]
    const double* __restrict__ coordz,        // [n_faces]
    const double* __restrict__ transform,     // [n_faces * 9] local->lab
    const double* __restrict__ inv_transform, // [n_faces * 9] lab->local
    const double* __restrict__ origin,        // [n_faces * 3]
    const double* __restrict__ mag,           // [n_faces * 3]
    int n_faces,
    const double* __restrict__ obs,           // [n_obs * 3]
    int n_obs,
    double* __restrict__ partial_B,           // [n_obs * n_src_blocks * 3]
    int n_src_blocks)
{
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_block_idx = blockIdx.y;
    if (obs_idx >= n_obs) return;

    double px = obs[obs_idx * 3 + 0];
    double py = obs[obs_idx * 3 + 1];
    double pz = obs[obs_idx * 3 + 2];

    double Bx = 0.0, By = 0.0, Bz = 0.0;

    int face_start = src_block_idx * blockDim.x;
    int face_end   = face_start + blockDim.x;
    if (face_end > n_faces) face_end = n_faces;

    const double PI = 3.14159265358979323846;
    const double ConstForH = 1.0 / (4.0 * PI);

    for (int fi = face_start; fi < face_end; fi++)
    {
        int nv = nverts[fi];
        if (nv < 3) continue;

        int tb = fi * 9;
        double T00 = transform[tb + 0], T01 = transform[tb + 1], T02 = transform[tb + 2];
        double T10 = transform[tb + 3], T11 = transform[tb + 4], T12 = transform[tb + 5];
        double T20 = transform[tb + 6], T21 = transform[tb + 7], T22 = transform[tb + 8];

        double I00 = inv_transform[tb + 0], I01 = inv_transform[tb + 1], I02 = inv_transform[tb + 2];
        double I10 = inv_transform[tb + 3], I11 = inv_transform[tb + 4], I12 = inv_transform[tb + 5];
        double I20 = inv_transform[tb + 6], I21 = inv_transform[tb + 7], I22 = inv_transform[tb + 8];

        int f3 = fi * 3;
        double ox = origin[f3 + 0], oy = origin[f3 + 1], oz = origin[f3 + 2];
        double mx = mag[f3 + 0],    my = mag[f3 + 1],    mz = mag[f3 + 2];

        // observation in local frame
        double dpx = px - ox, dpy = py - oy, dpz = pz - oz;
        double lx = I00 * dpx + I01 * dpy + I02 * dpz;
        double ly = I10 * dpx + I11 * dpy + I12 * dpz;
        double lz = I20 * dpx + I21 * dpy + I22 * dpz;

        // magnetization local z
        double mlz = I20 * mx + I21 * my + I22 * mz;

        int vbase = fi * RADGPU_FLD_MAX_VERTS * 2;
        double cz = coordz[fi];
        double z_raw = cz - lz;

        // scale-aware epsilons
        double face_scale = radgpu_face_scale(verts2d, vbase, nv);
        double eps_z  = fmax(1.0e-15, 1.0e-12 * face_scale);
        double eps_xy = fmax(1.0e-15, 1.0e-12 * face_scale);
        double eps_b  = fmax(1.0e-15, 1.0e-12 * face_scale);

        // regular z evaluation (nudged away from exactly 0)
        double z_eval = z_raw;
        if (fabs(z_eval) < eps_z) z_eval = (z_eval < 0.0) ? -eps_z : +eps_z;

        double Sx = 0.0, Sy = 0.0, Sz = 0.0;

        // near-plane handling with two-sided limit around z=0
        bool near_plane = (fabs(z_raw) <= 10.0 * eps_z);

        if (near_plane)
        {
            bool inside2d = radgpu_point_in_poly_2d(verts2d, vbase, nv, lx, ly);

            double edge_eps = fmax(1.0e-12, 1.0e-9 * face_scale);
            double min_d2 = radgpu_min_dist2_edges_2d(verts2d, vbase, nv, lx, ly);
            bool near_edge = (min_d2 <= edge_eps * edge_eps);

            if (inside2d || near_edge)
            {
                double Sx_p, Sy_p, Sz_p;
                double Sx_m, Sy_m, Sz_m;

                radgpu_eval_face_integrals_at_z(
                    verts2d, vbase, nv, lx, ly, +eps_z, eps_xy, eps_b, Sx_p, Sy_p, Sz_p);

                radgpu_eval_face_integrals_at_z(
                    verts2d, vbase, nv, lx, ly, -eps_z, eps_xy, eps_b, Sx_m, Sy_m, Sz_m);

                if (inside2d)
                {
                    // principal-value style on-face limit
                    Sx = 0.5 * (Sx_p + Sx_m);
                    Sy = 0.5 * (Sy_p + Sy_m);
                    Sz = 0.5 * (Sz_p + Sz_m);
                }
                else
                {
                    // off-face but near edge: one-sided from z_raw
                    if (z_raw >= 0.0) { Sx = Sx_p; Sy = Sy_p; Sz = Sz_p; }
                    else              { Sx = Sx_m; Sy = Sy_m; Sz = Sz_m; }
                }
            }
            else
            {
                radgpu_eval_face_integrals_at_z(
                    verts2d, vbase, nv, lx, ly, z_eval, eps_xy, eps_b, Sx, Sy, Sz);
            }
        }
        else
        {
            radgpu_eval_face_integrals_at_z(
                verts2d, vbase, nv, lx, ly, z_eval, eps_xy, eps_b, Sx, Sy, Sz);
        }

        if (!isfinite(Sx) || !isfinite(Sy) || !isfinite(Sz)) continue;

        double Hx_loc = -ConstForH * mlz * Sx;
        double Hy_loc = -ConstForH * mlz * Sy;
        double Hz_loc = -ConstForH * mlz * Sz;

        // local -> lab
        Bx += T00 * Hx_loc + T01 * Hy_loc + T02 * Hz_loc;
        By += T10 * Hx_loc + T11 * Hy_loc + T12 * Hz_loc;
        Bz += T20 * Hx_loc + T21 * Hy_loc + T22 * Hz_loc;
    }

    int out_idx = (obs_idx * n_src_blocks + src_block_idx) * 3;
    partial_B[out_idx + 0] = Bx;
    partial_B[out_idx + 1] = By;
    partial_B[out_idx + 2] = Bz;
}

//-------------------------------------------------------------------------
// Reduction kernel: sum partial_B across source blocks per obs point.
//-------------------------------------------------------------------------
__global__
void radGPU_FldReduceKernel(
    const double* __restrict__ partial_B,   // [n_obs * n_src_blocks * 3]
    double* __restrict__ result_B,          // [n_obs * 3]
    int n_obs,
    int n_src_blocks)
{
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (obs_idx >= n_obs) return;

    double Bx = 0.0, By = 0.0, Bz = 0.0;
    int base = obs_idx * n_src_blocks * 3;
    for (int s = 0; s < n_src_blocks; s++)
    {
        Bx += partial_B[base + s * 3 + 0];
        By += partial_B[base + s * 3 + 1];
        Bz += partial_B[base + s * 3 + 2];
    }

    result_B[obs_idx * 3 + 0] = Bx;
    result_B[obs_idx * 3 + 1] = By;
    result_B[obs_idx * 3 + 2] = Bz;
}

//-------------------------------------------------------------------------
// Allocate device memory and copy host data to device.
//-------------------------------------------------------------------------
int radGPU_FldAllocAndCopy(RadGPUFieldFaceData* data)
{
    cudaError_t err;
    int nf = data->n_faces_total;
    int nobs = data->n_obs;
    int nsb = data->n_src_blocks;

    // 2D vertices
    size_t v2d_bytes = (size_t)nf * RADGPU_FLD_MAX_VERTS * 2 * sizeof(double);
    err = cudaMalloc(&data->d_verts2d, v2d_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_verts2d, data->h_verts2d, v2d_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Nverts
    size_t nv_bytes = (size_t)nf * sizeof(int);
    err = cudaMalloc(&data->d_nverts, nv_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_nverts, data->h_nverts, nv_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // CoordZ
    size_t cz_bytes = (size_t)nf * sizeof(double);
    err = cudaMalloc(&data->d_coordz, cz_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_coordz, data->h_coordz, cz_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Transform matrices
    size_t mat_bytes = (size_t)nf * 9 * sizeof(double);
    err = cudaMalloc(&data->d_transform, mat_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_transform, data->h_transform, mat_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Inverse transform
    err = cudaMalloc(&data->d_inv_transform, mat_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_inv_transform, data->h_inv_transform, mat_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Origins
    size_t orig_bytes = (size_t)nf * 3 * sizeof(double);
    err = cudaMalloc(&data->d_origin, orig_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_origin, data->h_origin, orig_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Magnetization
    size_t mag_bytes = (size_t)nf * 3 * sizeof(double);
    err = cudaMalloc(&data->d_mag, mag_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_mag, data->h_mag, mag_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Observation points
    size_t obs_bytes = (size_t)nobs * 3 * sizeof(double);
    err = cudaMalloc(&data->d_obs, obs_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_obs, data->h_obs, obs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Partial results
    size_t partial_bytes = (size_t)nobs * nsb * 3 * sizeof(double);
    err = cudaMalloc(&data->d_partial_B, partial_bytes);
    if (err != cudaSuccess) return -1;

    // Result buffer
    size_t result_bytes = (size_t)nobs * 3 * sizeof(double);
    err = cudaMalloc(&data->d_result_B, result_bytes);
    if (err != cudaSuccess) return -1;

    return 0;
}

//-------------------------------------------------------------------------
// Launch main kernel + reduction kernel.
//-------------------------------------------------------------------------
int radGPU_FldLaunchKernel(RadGPUFieldFaceData* data)
{
    int nobs = data->n_obs;
    int nf = data->n_faces_total;
    int nsb = data->n_src_blocks;

    dim3 block(RADGPU_FLD_BLOCK_SIZE);
    dim3 grid(
        (nobs + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE,
        nsb
    );

    radGPU_FldKernel<<<grid, block>>>(
        data->d_verts2d, data->d_nverts, data->d_coordz,
        data->d_transform, data->d_inv_transform, data->d_origin,
        data->d_mag, nf, data->d_obs, nobs, data->d_partial_B, nsb
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "radGPU_Fld: kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 red_block(RADGPU_FLD_BLOCK_SIZE);
    dim3 red_grid((nobs + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE);

    radGPU_FldReduceKernel<<<red_grid, red_block>>>(
        data->d_partial_B, data->d_result_B, nobs, nsb
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) return -1;

    return 0;
}

//-------------------------------------------------------------------------
// Copy results from device to host, free all device memory.
//-------------------------------------------------------------------------
int radGPU_FldRetrieveAndFree(RadGPUFieldFaceData* data)
{
    cudaError_t err;
    size_t result_bytes = (size_t)data->n_obs * 3 * sizeof(double);

    err = cudaMemcpy(data->h_result_B, data->d_result_B, result_bytes, cudaMemcpyDeviceToHost);

    if (data->d_verts2d)       { cudaFree(data->d_verts2d);       data->d_verts2d = nullptr; }
    if (data->d_nverts)        { cudaFree(data->d_nverts);        data->d_nverts = nullptr; }
    if (data->d_coordz)        { cudaFree(data->d_coordz);        data->d_coordz = nullptr; }
    if (data->d_transform)     { cudaFree(data->d_transform);     data->d_transform = nullptr; }
    if (data->d_inv_transform) { cudaFree(data->d_inv_transform); data->d_inv_transform = nullptr; }
    if (data->d_origin)        { cudaFree(data->d_origin);        data->d_origin = nullptr; }
    if (data->d_mag)           { cudaFree(data->d_mag);           data->d_mag = nullptr; }
    if (data->d_obs)           { cudaFree(data->d_obs);           data->d_obs = nullptr; }
    if (data->d_partial_B)     { cudaFree(data->d_partial_B);     data->d_partial_B = nullptr; }
    if (data->d_result_B)      { cudaFree(data->d_result_B);      data->d_result_B = nullptr; }

    return (err == cudaSuccess) ? 0 : -1;
}


//=========================================================================
// RecMag kernel: analytical field from rectangular parallelepiped.
//
// Uses the standard formula with 8 corner contributions.
// Each corner contributes atan and log terms to the field.
//
// This matches the _RECMAG_KERNEL_FP64 in field_kernel.py.
//
// 2D grid:
//   grid.x = ceil(n_obs / BLOCK_SIZE)      — observation point blocks
//   grid.y = ceil(n_recmags / BLOCK_SIZE)   — source RecMag blocks
//=========================================================================

__global__
void radGPU_FldRecMagKernel(
    const double* __restrict__ centers,     // [n_recmags * 3]
    const double* __restrict__ dims,        // [n_recmags * 3]
    const double* __restrict__ mag,         // [n_recmags * 3] (LOCAL frame)
    const double* __restrict__ rot,         // [n_recmags * 9] (local->lab, row-major)
    int n_recmags,
    const double* __restrict__ obs,         // [n_obs * 3]
    int n_obs,
    double* __restrict__ partial_B,         // [n_obs * n_src_blocks * 3]
    int n_src_blocks)
{
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_block_idx = blockIdx.y;

    if (obs_idx >= n_obs) return;

    double px = obs[obs_idx * 3 + 0];
    double py = obs[obs_idx * 3 + 1];
    double pz = obs[obs_idx * 3 + 2];

    double Bx = 0.0, By = 0.0, Bz = 0.0;

    int rm_start = src_block_idx * blockDim.x;
    int rm_end = rm_start + blockDim.x;
    if (rm_end > n_recmags) rm_end = n_recmags;

    const double PI4 = 4.0 * 3.14159265358979323846;
    const double inv4pi = 1.0 / PI4;

    for (int ri = rm_start; ri < rm_end; ri++)
    {
        int f3 = ri * 3;
        double cx = centers[f3 + 0], cy = centers[f3 + 1], cz = centers[f3 + 2];
        double hx = dims[f3 + 0] * 0.5, hy = dims[f3 + 1] * 0.5, hz = dims[f3 + 2] * 0.5;
        double mx = mag[f3 + 0], my = mag[f3 + 1], mz = mag[f3 + 2];

        int rb = ri * 9;
        double R00 = rot[rb+0], R01 = rot[rb+1], R02 = rot[rb+2];
        double R10 = rot[rb+3], R11 = rot[rb+4], R12 = rot[rb+5];
        double R20 = rot[rb+6], R21 = rot[rb+7], R22 = rot[rb+8];

        double dpx = px - cx, dpy = py - cy, dpz = pz - cz;
        double rx = R00 * dpx + R10 * dpy + R20 * dpz;
        double ry = R01 * dpx + R11 * dpy + R21 * dpz;
        double rz = R02 * dpx + R12 * dpy + R22 * dpz;

        double x0 = rx - hx, x1 = rx + hx;
        double y0 = ry - hy, y1 = ry + hy;
        double z0 = rz - hz, z1 = rz + hz;

        double Hxl = 0.0, Hyl = 0.0, Hzl = 0.0;

        for (int ix = 0; ix < 2; ix++) {
            double x = (ix == 0) ? x0 : x1; double sx = (ix == 0) ? -1.0 : 1.0;
            double x2 = x * x;
            for (int iy = 0; iy < 2; iy++) {
                double y = (iy == 0) ? y0 : y1; double sy = (iy == 0) ? -1.0 : 1.0;
                double x2py2 = x2 + y * y;
                double sxy = sx * sy;
                for (int iz = 0; iz < 2; iz++) {
                    double z = (iz == 0) ? z0 : z1; double sz = (iz == 0) ? -1.0 : 1.0;
                    double sign = sxy * sz;
                    double R = sqrt(x2py2 + z*z);
                    if (R < 1e-20) R = 1e-20;

                    double zpR = z + R, ypR = y + R, xpR = x + R;
                    if (fabs(zpR) < 1e-20) zpR = 1e-20;
                    if (fabs(ypR) < 1e-20) ypR = 1e-20;
                    if (fabs(xpR) < 1e-20) xpR = 1e-20;

                    double lzpR = log(fabs(zpR)), lypR = log(fabs(ypR)), lxpR = log(fabs(xpR));
                    double xR = x * R, yR = y * R, zR = z * R;
                    double at_yz_xR = (fabs(xR) > 1e-30) ? atan2(y*z, xR) : 0.0;
                    double at_xz_yR = (fabs(yR) > 1e-30) ? atan2(x*z, yR) : 0.0;
                    double at_xy_zR = (fabs(zR) > 1e-30) ? atan2(x*y, zR) : 0.0;

                    Hxl += sign * (mx * at_yz_xR - my * lzpR - mz * lypR);
                    Hyl += sign * (-mx * lzpR + my * at_xz_yR - mz * lxpR);
                    Hzl += sign * (-mx * lypR - my * lxpR + mz * at_xy_zR);
                }
            }
        }

        double bxl = -Hxl * inv4pi, byl = -Hyl * inv4pi, bzl = -Hzl * inv4pi;
        Bx += R00 * bxl + R01 * byl + R02 * bzl;
        By += R10 * bxl + R11 * byl + R12 * bzl;
        Bz += R20 * bxl + R21 * byl + R22 * bzl;
    }

    int out_idx_rm = (obs_idx * n_src_blocks + src_block_idx) * 3;
    partial_B[out_idx_rm + 0] = Bx;
    partial_B[out_idx_rm + 1] = By;
    partial_B[out_idx_rm + 2] = Bz;
}

//-------------------------------------------------------------------------
// RecMag reduction kernel (same pattern as polygon)
//-------------------------------------------------------------------------
__global__
void radGPU_FldRecMagReduceKernel(
    const double* __restrict__ partial_B,
    double* __restrict__ result_B,
    int n_obs,
    int n_src_blocks)
{
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (obs_idx >= n_obs) return;

    double Bx = 0.0, By = 0.0, Bz = 0.0;
    int base = obs_idx * n_src_blocks * 3;
    for (int s = 0; s < n_src_blocks; s++)
    {
        Bx += partial_B[base + s * 3 + 0];
        By += partial_B[base + s * 3 + 1];
        Bz += partial_B[base + s * 3 + 2];
    }

    result_B[obs_idx * 3 + 0] = Bx;
    result_B[obs_idx * 3 + 1] = By;
    result_B[obs_idx * 3 + 2] = Bz;
}

//-------------------------------------------------------------------------
// RecMag: allocate and copy
//-------------------------------------------------------------------------
int radGPU_FldRecMagAllocAndCopy(RadGPUFieldRecMagData* data)
{
    cudaError_t err;
    int nrm = data->n_recmags;
    int nobs = data->n_obs;
    int nsb = data->n_src_blocks;

    size_t vec3_bytes = (size_t)nrm * 3 * sizeof(double);
    size_t obs_bytes = (size_t)nobs * 3 * sizeof(double);
    size_t partial_bytes = (size_t)nobs * nsb * 3 * sizeof(double);
    size_t result_bytes = (size_t)nobs * 3 * sizeof(double);

    err = cudaMalloc(&data->d_centers, vec3_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_centers, data->h_centers, vec3_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&data->d_dims, vec3_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_dims, data->h_dims, vec3_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&data->d_mag, vec3_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_mag, data->h_mag, vec3_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    size_t rot_bytes = (size_t)nrm * 9 * sizeof(double);
    err = cudaMalloc(&data->d_rot, rot_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_rot, data->h_rot, rot_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&data->d_obs, obs_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_obs, data->h_obs, obs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&data->d_partial_B, partial_bytes);
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&data->d_result_B, result_bytes);
    if (err != cudaSuccess) return -1;

    return 0;
}

//-------------------------------------------------------------------------
// RecMag: launch kernels
//-------------------------------------------------------------------------
int radGPU_FldRecMagLaunchKernel(RadGPUFieldRecMagData* data)
{
    int nobs = data->n_obs;
    int nrm = data->n_recmags;
    int nsb = data->n_src_blocks;

    dim3 block(RADGPU_FLD_BLOCK_SIZE);
    dim3 grid(
        (nobs + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE,
        nsb
    );

    radGPU_FldRecMagKernel<<<grid, block>>>(
        data->d_centers, data->d_dims, data->d_mag, data->d_rot,
        nrm, data->d_obs, nobs, data->d_partial_B, nsb
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "radGPU_FldRecMag: kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 red_block(RADGPU_FLD_BLOCK_SIZE);
    dim3 red_grid((nobs + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE);

    radGPU_FldRecMagReduceKernel<<<red_grid, red_block>>>(
        data->d_partial_B, data->d_result_B, nobs, nsb
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;

    return 0;
}

//-------------------------------------------------------------------------
// RecMag: retrieve and free
//-------------------------------------------------------------------------
int radGPU_FldRecMagRetrieveAndFree(RadGPUFieldRecMagData* data)
{
    cudaError_t err;
    size_t result_bytes = (size_t)data->n_obs * 3 * sizeof(double);

    err = cudaMemcpy(data->h_result_B, data->d_result_B, result_bytes, cudaMemcpyDeviceToHost);

    if (data->d_centers)   { cudaFree(data->d_centers);   data->d_centers = nullptr; }
    if (data->d_dims)      { cudaFree(data->d_dims);      data->d_dims = nullptr; }
    if (data->d_mag)       { cudaFree(data->d_mag);       data->d_mag = nullptr; }
    if (data->d_obs)       { cudaFree(data->d_obs);       data->d_obs = nullptr; }
    if (data->d_partial_B) { cudaFree(data->d_partial_B); data->d_partial_B = nullptr; }
    if (data->d_result_B)  { cudaFree(data->d_result_B);  data->d_result_B = nullptr; }
    if (data->d_rot)       { cudaFree(data->d_rot);       data->d_rot = nullptr; }

    return (err == cudaSuccess) ? 0 : -1;
}

#endif // RADIA_WITH_CUDA