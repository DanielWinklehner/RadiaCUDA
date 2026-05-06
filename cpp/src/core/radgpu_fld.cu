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

//=========================================================================
// Polygon face kernel — 2D face-local formulation matching CuPy.
//
// For each face:
//   1. Transform obs point to face local frame
//   2. Compute field in local frame using 2D polygon formula
//   3. Transform result back to lab frame
//
// In the local frame, the polygon lies in the z=coordZ plane with
// vertices given as 2D (x,y) coordinates. The observation point is
// at (x_local, y_local, z_local) where z_local is the signed distance
// from the polygon plane.
//
// The field formula uses edge-based log and atan2 terms, matching
// _POLY_KERNEL_FP64 in field_kernel.py.
//=========================================================================

__device__ double TransAtans_fld(double x, double y, double& PiMult)
{
    double buf = 1.0 - x * y;
    if(buf == 0.0)
    {
        PiMult = (x > 0.0) ? -0.5 : 0.5;
        return 1.0e+50;
    }
    PiMult = (buf > 0.0) ? 0.0 : ((x < 0.0) ? -1.0 : 1.0);
    return (x + y) / buf;
}

__device__ double Sign_fld(double x)
{
    return (x >= 0.0) ? 1.0 : -1.0;
}

__global__
void radGPU_FldKernel(
    const double* __restrict__ verts2d,       // [n_faces * MAX_VERTS * 2]
    const int*    __restrict__ nverts,         // [n_faces]
    const double* __restrict__ coordz,         // [n_faces]
    const double* __restrict__ transform,      // [n_faces * 9] local->lab
    const double* __restrict__ inv_transform,  // [n_faces * 9] lab->local
    const double* __restrict__ origin,         // [n_faces * 3] face origin in lab
    const double* __restrict__ mag,            // [n_faces * 3] magnetization in lab
    int n_faces,
    const double* __restrict__ obs,            // [n_obs * 3]
    int n_obs,
    double* __restrict__ partial_B,            // [n_obs * n_src_blocks * 3]
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
    int face_end = face_start + blockDim.x;
    if (face_end > n_faces) face_end = n_faces;

    const double PI = 3.14159265358979323846;
    const double ConstForH = 1.0 / (4.0 * PI);
    const double Max_k = 1.0e+09;
    const double RelRandMagn = 1.0e-13;
    const double MaxRelTolToSwitch = 1.0e-07;

    for (int fi = face_start; fi < face_end; fi++)
    {
        int nv = nverts[fi];
        if (nv < 3) continue;

        int tb = fi * 9;
        // Transform local->lab
        double T00 = transform[tb+0], T01 = transform[tb+1], T02 = transform[tb+2];
        double T10 = transform[tb+3], T11 = transform[tb+4], T12 = transform[tb+5];
        double T20 = transform[tb+6], T21 = transform[tb+7], T22 = transform[tb+8];

        // Inverse transform (lab->local)
        double I00 = inv_transform[tb+0], I01 = inv_transform[tb+1], I02 = inv_transform[tb+2];
        double I10 = inv_transform[tb+3], I11 = inv_transform[tb+4], I12 = inv_transform[tb+5];
        double I20 = inv_transform[tb+6], I21 = inv_transform[tb+7], I22 = inv_transform[tb+8];

        double ox = origin[fi * 3 + 0], oy = origin[fi * 3 + 1], oz = origin[fi * 3 + 2];
        double mx = mag[fi * 3 + 0], my = mag[fi * 3 + 1], mz = mag[fi * 3 + 2];

        // Transform obs point to face local frame
        double dpx = px - ox, dpy = py - oy, dpz = pz - oz;
        double lx = I00 * dpx + I01 * dpy + I02 * dpz;
        double ly = I10 * dpx + I11 * dpy + I12 * dpz;
        double lz = I20 * dpx + I21 * dpy + I22 * dpz;

        // Transform magnetization to local frame
        // double mlx = I00 * mx + I01 * my + I02 * mz;
        // double mly = I10 * mx + I11 * my + I12 * mz;
        double mlz = I20 * mx + I21 * my + I22 * mz;

        double cz = coordz[fi];
        double z = cz - lz;
        
        if(z == 0.0) z = RelRandMagn;
        double ze2 = z * z;

        int vbase = fi * RADGPU_FLD_MAX_VERTS * 2;
        double x1 = verts2d[vbase + 0] - lx;
        double y1 = verts2d[vbase + 1] - ly;
        if(x1 == 0.0) x1 = RelRandMagn;
        if(y1 == 0.0) y1 = RelRandMagn;
        double x1e2 = x1 * x1;

        double Sz_atan = 0.0;
        double Sx = 0.0, Sy = 0.0;
        double ArgSumLogs2 = 1.0;

        for (int ei = 0; ei < nv; ei++)
        {
            double x2, y2;
            int vnext = vbase + ((ei + 1) % nv) * 2;
            x2 = verts2d[vnext + 0] - lx;
            y2 = verts2d[vnext + 1] - ly;
            if(x2 == 0.0) x2 = RelRandMagn;
            if(y2 == 0.0) y2 = RelRandMagn;
            double x2e2 = x2 * x2;

            double x2mx1 = x2 - x1, y2my1 = y2 - y1;
            if(fabs(x2mx1) * Max_k > fabs(y2my1))
            {
                double k = y2my1 / x2mx1;
                double b = y1 - k * x1;
                if(b == 0.0) b = RelRandMagn;

                double ke2 = k * k, be2 = b * b, ke2p1 = ke2 + 1.0;
                double sqrtke2p1 = sqrt(ke2p1), bk = b * k;
                double bpkx1 = b + k * x1, bpkx2 = b + k * x2;
                double bpkx1e2 = bpkx1 * bpkx1, bpkx2e2 = bpkx2 * bpkx2;
                double R1 = sqrt(x1e2 + bpkx1e2 + ze2);
                double R2 = sqrt(x2e2 + bpkx2e2 + ze2);
                double x1e2pze2 = x1e2 + ze2;
                double kze2 = k * ze2, ke2ze2 = k * kze2;
                double ke2ze2pbe2 = ke2ze2 + be2, ke2ze2mbe2 = ke2ze2 - be2;
                double bx1 = b * x1, bx2 = b * x2;
                double R1pbpkx1 = bpkx1 + R1, R2pbpkx2 = bpkx2 + R2;

                double AbsRandR1 = 100.0 * R1 * RelRandMagn;
                double AbsRandR2 = 100.0 * R2 * RelRandMagn;
                double MaxAbsRandR1 = MaxRelTolToSwitch * R1;
                double MaxAbsRandR2 = MaxRelTolToSwitch * R2;
                if(AbsRandR1 > MaxAbsRandR1) AbsRandR1 = MaxAbsRandR1;
                if(AbsRandR2 > MaxAbsRandR2) AbsRandR2 = MaxAbsRandR2;

                if(fabs(R1pbpkx1) < AbsRandR1 && R1 > 100.0 * AbsRandR1 && (x1e2pze2) < bpkx1e2 * MaxRelTolToSwitch)
                    R1pbpkx1 = (bpkx1 != 0.0) ? 0.5 * (x1e2pze2) / fabs(bpkx1) : 1.0e-50;
                if(fabs(R2pbpkx2) < AbsRandR2 && R2 > 100.0 * AbsRandR2 && (x2e2 + ze2) < bpkx2e2 * MaxRelTolToSwitch)
                    R2pbpkx2 = (bpkx2 != 0.0) ? 0.5 * (x2e2 + ze2) / fabs(bpkx2) : 1.0e-50;

                if(R1pbpkx1 == 0.0) R1pbpkx1 = 1.0e-50;
                if(R2pbpkx2 == 0.0) R2pbpkx2 = 1.0e-50;

                double twob = 2.0 * b;
                double kx1mb = k * x1 - b, kx2mb = k * x2 - b;

                double A1 = -(ke2ze2pbe2*(bx1 + kze2)*R1pbpkx1 + kze2*twob*(x1e2 + ze2));
                double A2 = (ke2ze2pbe2*kx1mb*R1pbpkx1 + ke2ze2mbe2*(x1e2 + ze2))*z;
                double A3 = ke2ze2pbe2*(bx2 + kze2)*R2pbpkx2 + kze2*twob*(x2e2 + ze2);
                double A4 = (ke2ze2pbe2*kx2mb*R2pbpkx2 + ke2ze2mbe2*(x2e2 + ze2))*z;

                Sz_atan += atan2(A1*A4 + A3*A2, A2*A4 - A1*A3);

                double bkpx1pke2x1 = bk + ke2p1 * x1, bkpx2pke2x2 = bk + ke2p1 * x2;
                double v1 = bkpx1pke2x1 / sqrtke2p1 + R1, v2 = bkpx2pke2x2 / sqrtke2p1 + R2;
                double be2pze2 = be2 + ze2;
                if(fabs(v1) < AbsRandR1 && R1 > 100.0 * AbsRandR1 && (be2pze2 + 2*bk*x1) < x1e2*ke2p1*MaxRelTolToSwitch)
                    v1 = (x1 != 0.0) ? 0.5 * be2pze2 / (fabs(x1) * sqrtke2p1) : 1.0e-50;
                if(fabs(v2) < AbsRandR2 && R2 > 100.0 * AbsRandR2 && (be2pze2 + 2*bk*x2) < x2e2*ke2p1*MaxRelTolToSwitch)
                    v2 = (x2 != 0.0) ? 0.5 * be2pze2 / (fabs(x2) * sqrtke2p1) : 1.0e-50;

                if(v1 == 0.0) v1 = 1.0e-50;
                if(v2 == 0.0) v2 = 1.0e-50;
                double SL1 = log(v2 / v1) / sqrtke2p1;

                double lr2 = (R2pbpkx2 / R1pbpkx1);
                if(lr2 <= 0.0) lr2 = 1.0e-50;
                ArgSumLogs2 *= lr2;
                Sx += -k * SL1; Sy += SL1;
            }
            x1 = x2; y1 = y2; x1e2 = x2e2;
        }

        if(ArgSumLogs2 <= 0.0) ArgSumLogs2 = 1.0e-50;
        Sx += log(ArgSumLogs2);

        // H_local = -ConstForH * Mz * (Sx, Sy, Sz)
        double Hx_loc = -ConstForH * mlz * Sx;
        double Hy_loc = -ConstForH * mlz * Sy;
        double Hz_loc = -ConstForH * mlz * Sz_atan;

        // B_lab = T * H_loc (rotation from local to lab)
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
        double cx = centers[ri * 3 + 0], cy = centers[ri * 3 + 1], cz = centers[ri * 3 + 2];
        double hx = dims[ri * 3 + 0] * 0.5, hy = dims[ri * 3 + 1] * 0.5, hz = dims[ri * 3 + 2] * 0.5;
        double mx = mag[ri * 3 + 0], my = mag[ri * 3 + 1], mz = mag[ri * 3 + 2];

        int rb = ri * 9;
        double R00 = rot[rb+0], R01 = rot[rb+1], R02 = rot[rb+2];
        double R10 = rot[rb+3], R11 = rot[rb+4], R12 = rot[rb+5];
        double R20 = rot[rb+6], R21 = rot[rb+7], R22 = rot[rb+8];

        // Transform obs point to local frame: obs_local = R^T * (obs_lab - center_lab)
        double dpx = px - cx, dpy = py - cy, dpz = pz - cz;
        double rx = R00 * dpx + R10 * dpy + R20 * dpz;
        double ry = R01 * dpx + R11 * dpy + R21 * dpz;
        double rz = R02 * dpx + R12 * dpy + R22 * dpz;

        double x_vals[2] = {rx - hx, rx + hx};
        double y_vals[2] = {ry - hy, ry + hy};
        double z_vals[2] = {rz - hz, rz + hz};

        double Hxl = 0.0, Hyl = 0.0, Hzl = 0.0;

        for (int ix = 0; ix < 2; ix++) {
            double x = x_vals[ix]; double sx = (ix == 0) ? -1.0 : 1.0;
            for (int iy = 0; iy < 2; iy++) {
                double y = y_vals[iy]; double sy = (iy == 0) ? -1.0 : 1.0;
                for (int iz = 0; iz < 2; iz++) {
                    double z = z_vals[iz]; double sz = (iz == 0) ? -1.0 : 1.0;
                    double sign = sx * sy * sz;
                    double R = sqrt(x*x + y*y + z*z);
                    if (R < 1e-20) R = 1e-20;

                    double zpR = z + R, ypR = y + R, xpR = x + R;
                    if (fabs(zpR) < 1e-20) zpR = 1e-20;
                    if (fabs(ypR) < 1e-20) ypR = 1e-20;
                    if (fabs(xpR) < 1e-20) xpR = 1e-20;

                    double lzpR = log(fabs(zpR)), lypR = log(fabs(ypR)), lxpR = log(fabs(xpR));
                    double at_yz_xR = (fabs(x*R) > 1e-30) ? atan2(y*z, x*R) : 0.0;
                    double at_xz_yR = (fabs(y*R) > 1e-30) ? atan2(x*z, y*R) : 0.0;
                    double at_xy_zR = (fabs(z*R) > 1e-30) ? atan2(x*y, z*R) : 0.0;

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