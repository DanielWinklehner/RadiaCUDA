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

//-------------------------------------------------------------------------
// Constant: -1/(4π), the scalar potential prefactor in Radia's units.
// Radia works in Tesla and mm; this prefactor converts surface charge
// density σ = M·n̂ (Tesla) to field contribution (Tesla).
//-------------------------------------------------------------------------
static __constant__ double d_CONST_FOR_H = -0.07957747154594767;  // -1/(4π)

//-------------------------------------------------------------------------
// Main kernel: polygon face → B field contribution.
//
// 2D grid:
//   grid.x = ceil(n_obs / BLOCK_SIZE)      — observation point blocks
//   grid.y = ceil(n_faces / BLOCK_SIZE)     — source face blocks
//
// Each thread computes contributions from one block of source faces
// to one observation point. Partial results stored for later reduction.
//
// This kernel implements the same field formula as _POLY_KERNEL_FP64
// in field_kernel.py, including the atan2 solid-angle accumulation.
//-------------------------------------------------------------------------
__global__
void radGPU_FldKernel(
    const double* __restrict__ verts,       // [n_faces * MAX_VERTS * 3]
    const int*    __restrict__ nverts,       // [n_faces]
    const double* __restrict__ normals,      // [n_faces * 3]
    const double* __restrict__ mag,          // [n_faces * 3]
    int n_faces,
    const double* __restrict__ obs,          // [n_obs * 3]
    int n_obs,
    double* __restrict__ partial_B,          // [n_obs * n_src_blocks * 3]
    int n_src_blocks)
{
    int obs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_block_idx = blockIdx.y;

    if (obs_idx >= n_obs) return;

    // Observation point
    double px = obs[obs_idx * 3 + 0];
    double py = obs[obs_idx * 3 + 1];
    double pz = obs[obs_idx * 3 + 2];

    // Accumulate B from faces in this source block
    double Bx = 0.0, By = 0.0, Bz = 0.0;

    int face_start = src_block_idx * blockDim.x;
    int face_end = face_start + blockDim.x;
    if (face_end > n_faces) face_end = n_faces;

    for (int fi = face_start; fi < face_end; fi++)
    {
        int nv = nverts[fi];
        if (nv < 3) continue;

        // Face normal
        double nx = normals[fi * 3 + 0];
        double ny = normals[fi * 3 + 1];
        double nz = normals[fi * 3 + 2];

        // Magnetization of the parent element (pre-transformed)
        double mx = mag[fi * 3 + 0];
        double my = mag[fi * 3 + 1];
        double mz = mag[fi * 3 + 2];

        // Surface charge density: σ = M · n̂
        double sigma = mx * nx + my * ny + mz * nz;
        if (fabs(sigma) < 1.0e-30) continue;

        // Accumulate edge-based line integrals (Sx, Sy, Sz) and solid angle (omega)
        double Sx = 0.0, Sy = 0.0, Sz = 0.0;
        double omega = 0.0;

        int vbase = fi * RADGPU_FLD_MAX_VERTS * 3;

        // First vertex relative to obs point
        double r0x = verts[vbase + 0] - px;
        double r0y = verts[vbase + 1] - py;
        double r0z = verts[vbase + 2] - pz;
        double R0 = sqrt(r0x * r0x + r0y * r0y + r0z * r0z);

        // Save first for wraparound
        double r_first_x = r0x, r_first_y = r0y, r_first_z = r0z;
        double R_first = R0;

        double r1x = r0x, r1y = r0y, r1z = r0z;
        double R1 = R0;

        for (int ei = 0; ei < nv; ei++)
        {
            // Next vertex (wraparound on last edge)
            double r2x, r2y, r2z, R2;
            if (ei < nv - 1)
            {
                int vnext = vbase + (ei + 1) * 3;
                r2x = verts[vnext + 0] - px;
                r2y = verts[vnext + 1] - py;
                r2z = verts[vnext + 2] - pz;
            }
            else
            {
                r2x = r_first_x;
                r2y = r_first_y;
                r2z = r_first_z;
            }
            R2 = sqrt(r2x * r2x + r2y * r2y + r2z * r2z);

            // Edge vector and length
            double ex = r2x - r1x;
            double ey = r2y - r1y;
            double ez = r2z - r1z;
            double L = sqrt(ex * ex + ey * ey + ez * ez);

            if (L < 1.0e-30)
            {
                r1x = r2x; r1y = r2y; r1z = r2z;
                R1 = R2;
                continue;
            }

            double invL = 1.0 / L;
            double eux = ex * invL;
            double euy = ey * invL;
            double euz = ez * invL;

            // --- Log term: ln((R1 + R2 + L) / (R1 + R2 - L)) ---
            double sum_R = R1 + R2;
            double arg_p = sum_R + L;
            double arg_m = sum_R - L;
            double log_term = 0.0;
            if (fabs(arg_m) > 1.0e-30 && arg_p > 1.0e-30)
            {
                log_term = log(arg_p / arg_m);
            }

            // ê × n̂  (direction of line integral contribution)
            double cn_x = euy * nz - euz * ny;
            double cn_y = euz * nx - eux * nz;
            double cn_z = eux * ny - euy * nx;

            Sx += cn_x * log_term;
            Sy += cn_y * log_term;
            Sz += cn_z * log_term;

            // --- Solid angle via atan2 (tangent addition formula) ---
            // d = signed distance from obs to face plane = n̂ · r1
            double d_plane = nx * r1x + ny * r1y + nz * r1z;

            // n̂ × ê
            double ne_x = ny * euz - nz * euy;
            double ne_y = nz * eux - nx * euz;
            double ne_z = nx * euy - ny * eux;

            double Arg1 = ne_x * r1x + ne_y * r1y + ne_z * r1z;  // (n̂ × ê) · r1
            double Arg2 = d_plane * R1;                            // d * R1
            double Arg3 = ne_x * r2x + ne_y * r2y + ne_z * r2z;  // (n̂ × ê) · r2
            double Arg4 = d_plane * R2;                            // d * R2

            double atan_num = Arg1 * Arg4 + Arg3 * Arg2;
            double atan_den = Arg2 * Arg4 - Arg1 * Arg3;
            omega += atan2(atan_num, atan_den);

            // Advance
            r1x = r2x; r1y = r2y; r1z = r2z;
            R1 = R2;
        }

        // B_face = ConstForH * σ * (S + n̂ * Ω)
        double coeff = d_CONST_FOR_H * sigma;
        Bx += coeff * (Sx + nx * omega);
        By += coeff * (Sy + ny * omega);
        Bz += coeff * (Sz + nz * omega);
    }

    // Store partial result for this (obs_point, src_block) pair
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

    // Vertices
    size_t verts_bytes = (size_t)nf * RADGPU_FLD_MAX_VERTS * 3 * sizeof(double);
    err = cudaMalloc(&data->d_verts, verts_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "radGPU_Fld: cudaMalloc verts failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMemcpy(data->d_verts, data->h_verts, verts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "radGPU_Fld: cudaMemcpy verts failed\n"); return -1; }

    // Nverts per face
    size_t nv_bytes = (size_t)nf * sizeof(int);
    err = cudaMalloc(&data->d_nverts, nv_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_nverts, data->h_nverts, nv_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Normals
    size_t norm_bytes = (size_t)nf * 3 * sizeof(double);
    err = cudaMalloc(&data->d_normals, norm_bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(data->d_normals, data->h_normals, norm_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    // Magnetization per face
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

    // Partial results buffer
    size_t partial_bytes = (size_t)nobs * nsb * 3 * sizeof(double);
    err = cudaMalloc(&data->d_partial_B, partial_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "radGPU_Fld: partial_B alloc failed (%.1f MB)\n", partial_bytes / 1e6); return -1; }

    // Separate result buffer (reduction output)
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

    // Main kernel: 2D grid
    dim3 block(RADGPU_FLD_BLOCK_SIZE);
    dim3 grid(
        (nobs + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE,
        nsb
    );

    radGPU_FldKernel<<<grid, block>>>(
        data->d_verts, data->d_nverts, data->d_normals, data->d_mag,
        nf, data->d_obs, nobs, data->d_partial_B, nsb
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "radGPU_Fld: kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Reduction kernel: 1D grid
    dim3 red_block(RADGPU_FLD_BLOCK_SIZE);
    dim3 red_grid((nobs + RADGPU_FLD_BLOCK_SIZE - 1) / RADGPU_FLD_BLOCK_SIZE);

    radGPU_FldReduceKernel<<<red_grid, red_block>>>(
        data->d_partial_B, data->d_result_B, nobs, nsb
    );

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "radGPU_Fld: reduce kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "radGPU_Fld: sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

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
    if (err != cudaSuccess)
    {
        fprintf(stderr, "radGPU_Fld: result copy failed: %s\n", cudaGetErrorString(err));
    }

    // Free all device allocations
    if (data->d_verts)     { cudaFree(data->d_verts);     data->d_verts = nullptr; }
    if (data->d_nverts)    { cudaFree(data->d_nverts);    data->d_nverts = nullptr; }
    if (data->d_normals)   { cudaFree(data->d_normals);   data->d_normals = nullptr; }
    if (data->d_mag)       { cudaFree(data->d_mag);       data->d_mag = nullptr; }
    if (data->d_obs)       { cudaFree(data->d_obs);       data->d_obs = nullptr; }
    if (data->d_partial_B) { cudaFree(data->d_partial_B); data->d_partial_B = nullptr; }
    if (data->d_result_B)  { cudaFree(data->d_result_B);  data->d_result_B = nullptr; }

    return (err == cudaSuccess) ? 0 : -1;
}

#endif // RADIA_WITH_CUDA