/*-------------------------------------------------------------------------
*
* File name:      radgpu_asm.cu
*
* Project:        RADIA
*
* Description:    CUDA kernels for interaction matrix assembly
*
*                 Ported from CuPy field kernels (field_kernel.py)
*                 with restructured parallelism for (i,j) pair computation.
*
-------------------------------------------------------------------------*/

#ifdef RADIA_WITH_CUDA

#include "radgpu_asm.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================
// Device: field from a single RecMag element at a point
//   Computes 3x3 block: column k = field due to unit M in direction k
// ============================================================
__device__ void recmag_block_dev(
    double px, double py, double pz,
    double cx, double cy, double cz,
    double wx, double wy, double wz,
    float* block)  // [9] output row-major
{
    const double PI4 = 4.0 * 3.14159265358979323846;
    const double inv4pi = 1.0 / PI4;

    double rx = px - cx, ry = py - cy, rz = pz - cz;
    double hx = 0.5*wx, hy = 0.5*wy, hz = 0.5*wz;
    double x0 = rx - hx, x1_ = rx + hx;
    double y0 = ry - hy, y1_ = ry + hy;
    double z0 = rz - hz, z1_ = rz + hz;

    double xs[2] = {x0, x1_};
    double ys[2] = {y0, y1_};
    double zs[2] = {z0, z1_};

    // Accumulate kernels for unit magnetizations
    // H_x(M) = mx*atan(yz/xR) - my*log(z+R) - mz*log(y+R)
    // So column 0 (unit mx): Hx += atan(yz/xR), Hy += -log(z+R), Hz += -log(y+R)
    // etc.
    double K_atan_yz_xR = 0.0;  // atan(yz/xR) sum
    double K_atan_xz_yR = 0.0;  // atan(xz/yR) sum
    double K_atan_xy_zR = 0.0;  // atan(xy/zR) sum
    double K_log_zpR = 0.0;     // log(z+R) sum
    double K_log_ypR = 0.0;     // log(y+R) sum
    double K_log_xpR = 0.0;     // log(x+R) sum

    for(int ix = 0; ix < 2; ix++) {
        double x = xs[ix]; double sx = (ix == 0) ? -1.0 : 1.0;
        for(int iy = 0; iy < 2; iy++) {
            double y = ys[iy]; double sy = (iy == 0) ? -1.0 : 1.0;
            for(int iz = 0; iz < 2; iz++) {
                double z = zs[iz]; double sz = (iz == 0) ? -1.0 : 1.0;
                double sign = sx * sy * sz;

                double xe2 = x*x, ye2 = y*y, ze2 = z*z;
                double R = sqrt(xe2 + ye2 + ze2);
                if(R < 1e-20) R = 1e-20;

                double zpR = z + R, ypR = y + R, xpR = x + R;
                if(fabs(zpR) < 1e-20) zpR = 1e-20;
                if(fabs(ypR) < 1e-20) ypR = 1e-20;
                if(fabs(xpR) < 1e-20) xpR = 1e-20;

                double log_zpR = log(fabs(zpR));
                double log_ypR = log(fabs(ypR));
                double log_xpR = log(fabs(xpR));

                double xR = x * R;
                double at_yz_xR = (fabs(xR) > 1e-30) ? atan2(y*z, xR) : 0.0;
                double at_xz_yR = (fabs(y*R) > 1e-30) ? atan2(x*z, y*R) : 0.0;
                double at_xy_zR = (fabs(z*R) > 1e-30) ? atan2(x*y, z*R) : 0.0;

                K_atan_yz_xR += sign * at_yz_xR;
                K_atan_xz_yR += sign * at_xz_yR;
                K_atan_xy_zR += sign * at_xy_zR;
                K_log_zpR += sign * log_zpR;
                K_log_ypR += sign * log_ypR;
                K_log_xpR += sign * log_xpR;
            }
        }
    }

    // Build 3x3 block: row = field component (Bx,By,Bz), col = unit M direction
    // B = -H / (4*pi), and the formula gives H contributions
    // Column 0 (unit Mx): Hx = atan_yz_xR, Hy = -log_zpR, Hz = -log_ypR
    // Column 1 (unit My): Hx = -log_zpR,   Hy = atan_xz_yR, Hz = -log_xpR
    // Column 2 (unit Mz): Hx = -log_ypR,   Hy = -log_xpR,   Hz = atan_xy_zR
    block[0] = (float)(-inv4pi * K_atan_yz_xR);  // Bx from Mx
    block[1] = (float)(-inv4pi * (-K_log_zpR));   // Bx from My
    block[2] = (float)(-inv4pi * (-K_log_ypR));   // Bx from Mz
    block[3] = (float)(-inv4pi * (-K_log_zpR));   // By from Mx
    block[4] = (float)(-inv4pi * K_atan_xz_yR);   // By from My
    block[5] = (float)(-inv4pi * (-K_log_xpR));   // By from Mz
    block[6] = (float)(-inv4pi * (-K_log_ypR));   // Bz from Mx
    block[7] = (float)(-inv4pi * (-K_log_xpR));   // Bz from My
    block[8] = (float)(-inv4pi * K_atan_xy_zR);   // Bz from Mz
}

// ============================================================
// Device: field from a single polyhedron face at a point
//   Computes contribution to Hx, Hy, Hz for unit normal magnetization
//   (only the face-normal component of M contributes)
//   Ported directly from polyhedron_field_fp64 kernel
// ============================================================
__device__ void poly_face_field_dev(
    double loc_x, double loc_y, double loc_z_obs,
    double cz,
    const double* edge_pts_2d, int n_edges,
    double* Sx_out, double* Sy_out, double* Sz_out)
{
    const double PI = 3.14159265358979323846;
    const double Max_k = 1.0e+09;
    const double RelRandMagn = 1.0e-13;
    const double MaxRelTolToSwitch = 1.0e-07;

    double z = cz - loc_z_obs;
    if(z == 0.0) z = RelRandMagn;
    double ze2 = z * z;

    double Sx = 0.0, Sy = 0.0;
    double ArgSumAtans1 = 0.0, PiMultSumAtans1 = 0.0;
    double ArgSumLogs2 = 1.0;

    double x1 = edge_pts_2d[0] - loc_x;
    double y1 = edge_pts_2d[1] - loc_y;
    if(x1 == 0.0) x1 = RelRandMagn;
    if(y1 == 0.0) y1 = RelRandMagn;
    double x1e2 = x1*x1;

    for(int ei = 0; ei < n_edges; ei++) {
        int next_ei = (ei + 1) % n_edges;
        double x2 = edge_pts_2d[next_ei*2+0] - loc_x;
        double y2 = edge_pts_2d[next_ei*2+1] - loc_y;
        if(x2 == 0.0) x2 = RelRandMagn;
        if(y2 == 0.0) y2 = RelRandMagn;
        double x2e2 = x2*x2;

        double x2mx1 = x2 - x1, y2my1 = y2 - y1;

        if(fabs(x2mx1)*Max_k > fabs(y2my1)) {
            double k = y2my1 / x2mx1;
            double b = y1 - k*x1;
            if(b == 0.0) b = RelRandMagn;

            double ke2 = k*k, be2 = b*b, ke2p1 = ke2 + 1.0;
            double sqrtke2p1 = sqrt(ke2p1), bk = b*k;
            double bpkx1 = b + k*x1, bpkx2 = b + k*x2;
            double bpkx1e2 = bpkx1*bpkx1, bpkx2e2 = bpkx2*bpkx2;
            double R1 = sqrt(x1e2 + bpkx1e2 + ze2);
            double R2 = sqrt(x2e2 + bpkx2e2 + ze2);

            double R1pbpkx1 = bpkx1 + R1, R2pbpkx2 = bpkx2 + R2;

            double AbsRandR1 = 100.0*R1*RelRandMagn;
            double AbsRandR2 = 100.0*R2*RelRandMagn;
            double MaxAbsRandR1 = MaxRelTolToSwitch*R1;
            double MaxAbsRandR2 = MaxRelTolToSwitch*R2;
            if(AbsRandR1 > MaxAbsRandR1) AbsRandR1 = MaxAbsRandR1;
            if(AbsRandR2 > MaxAbsRandR2) AbsRandR2 = MaxAbsRandR2;

            if(fabs(R1pbpkx1) < AbsRandR1 && R1 > 100.0*AbsRandR1 && (x1e2+ze2) < bpkx1e2*MaxRelTolToSwitch)
                R1pbpkx1 = (bpkx1 != 0.0) ? 0.5*(x1e2+ze2)/fabs(bpkx1) : 1.0e-50;
            if(fabs(R2pbpkx2) < AbsRandR2 && R2 > 100.0*AbsRandR2 && (x2e2+ze2) < bpkx2e2*MaxRelTolToSwitch)
                R2pbpkx2 = (bpkx2 != 0.0) ? 0.5*(x2e2+ze2)/fabs(bpkx2) : 1.0e-50;
            if(R1pbpkx1 == 0.0) R1pbpkx1 = 1.0e-50;
            if(R2pbpkx2 == 0.0) R2pbpkx2 = 1.0e-50;

            double bkpx1pke2x1 = bk + ke2p1*x1, bkpx2pke2x2 = bk + ke2p1*x2;
            double kze2 = k*ze2, ke2ze2 = k*kze2;
            double ke2ze2pbe2 = ke2ze2 + be2, ke2ze2mbe2 = ke2ze2 - be2;
            double bx1 = b*x1, bx2 = b*x2;
            double x1e2pze2 = x1e2 + ze2, x2e2pze2 = x2e2 + ze2;
            double twob = 2.0*b;
            double kx1mb = k*x1 - b, kx2mb = k*x2 - b;

            double Arg1 = -(ke2ze2pbe2*(bx1+kze2)*R1pbpkx1 + kze2*twob*x1e2pze2);
            double Arg2 = (ke2ze2pbe2*kx1mb*R1pbpkx1 + ke2ze2mbe2*x1e2pze2)*z;
            double Arg3 = ke2ze2pbe2*(bx2+kze2)*R2pbpkx2 + kze2*twob*x2e2pze2;
            double Arg4 = (ke2ze2pbe2*kx2mb*R2pbpkx2 + ke2ze2mbe2*x2e2pze2)*z;

            if(Arg2 == 0.0) Arg2 = 1.0e-50;
            if(Arg4 == 0.0) Arg4 = 1.0e-50;

            double rat1 = Arg1/Arg2, rat2 = Arg3/Arg4;
            double denom = 1.0 - rat1*rat2;
            double PiMult1 = 0.0, CurArg = 0.0;
            if(fabs(denom) > 1.0e-14*(fabs(rat1)+fabs(rat2))) {
                CurArg = (rat1+rat2)/denom;
            } else {
                PiMult1 = (rat1 > 0.0) ? 1.0 : -1.0;
            }
            double PiMult2 = 0.0;
            double denom2 = 1.0 - ArgSumAtans1*CurArg;
            if(fabs(denom2) > 1.0e-14*(fabs(ArgSumAtans1)+fabs(CurArg))) {
                ArgSumAtans1 = (ArgSumAtans1+CurArg)/denom2;
            } else {
                PiMult2 = (ArgSumAtans1 < 0.0) ? -1.0 : 1.0;
                ArgSumAtans1 = 0.0;
            }
            PiMultSumAtans1 += PiMult1 + PiMult2;

            double val1 = bkpx1pke2x1/sqrtke2p1 + R1;
            double val2 = bkpx2pke2x2/sqrtke2p1 + R2;
            double be2pze2 = be2 + ze2;
            if(fabs(val1) < AbsRandR1 && R1 > 100.0*AbsRandR1 && (be2pze2+2*bk*x1) < x1e2*ke2p1*MaxRelTolToSwitch)
                val1 = (x1 != 0.0) ? 0.5*be2pze2/(fabs(x1)*sqrtke2p1) : 1.0e-50;
            if(fabs(val2) < AbsRandR2 && R2 > 100.0*AbsRandR2 && (be2pze2+2*bk*x2) < x2e2*ke2p1*MaxRelTolToSwitch)
                val2 = (x2 != 0.0) ? 0.5*be2pze2/(fabs(x2)*sqrtke2p1) : 1.0e-50;
            if(val1 == 0.0) val1 = 1.0e-50;
            if(val2 == 0.0) val2 = 1.0e-50;

            double log_ratio = val2/val1;
            if(log_ratio <= 0.0) log_ratio = 1.0e-50;
            double SumLogs1 = log(log_ratio);
            double SumLogs1dsqrtke2p1 = SumLogs1/sqrtke2p1;

            double log_ratio2 = R2pbpkx2/R1pbpkx1;
            if(log_ratio2 <= 0.0) log_ratio2 = 1.0e-50;
            ArgSumLogs2 *= log_ratio2;
            Sx += -k*SumLogs1dsqrtke2p1;
            Sy += SumLogs1dsqrtke2p1;
        }
        x1 = x2; y1 = y2; x1e2 = x2e2;
    }

    double Sz = atan(ArgSumAtans1) + PiMultSumAtans1 * PI;
    if(ArgSumLogs2 <= 0.0) ArgSumLogs2 = 1.0e-50;
    Sx += log(ArgSumLogs2);

    *Sx_out = Sx;
    *Sy_out = Sy;
    *Sz_out = Sz;
}

// ============================================================
// Device: 3x3 block from a single polyhedron element at a point
//   For each of 3 unit magnetization directions, compute the field
//   by dotting unit M with each face normal, then accumulating
// ============================================================
__device__ void poly_block_dev(
    double px, double py, double pz,
    const int* face_offsets, int elem_idx,
    const int* edge_offsets,
    const double* edge_pts_2d,
    const double* face_cz,
    const double* face_rot,
    const double* face_orig,
    float* block)  // [9] output
{
    const double ConstForH = 1.0 / (4.0 * 3.14159265358979323846);

    // Zero the block
    for(int k = 0; k < 9; k++) block[k] = 0.0f;

    int f_start = face_offsets[elem_idx];
    int f_end = face_offsets[elem_idx + 1];

    // For each unit magnetization direction (dx = 0,1,2)
    double unitM[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    for(int dx = 0; dx < 3; dx++) {
        double mx = unitM[dx][0], my = unitM[dx][1], mz = unitM[dx][2];
        double Hx_sum = 0.0, Hy_sum = 0.0, Hz_sum = 0.0;

        for(int fi = f_start; fi < f_end; fi++) {
            double ox = face_orig[fi*3+0], oy = face_orig[fi*3+1], oz = face_orig[fi*3+2];
            double ddx = px - ox, ddy = py - oy, ddz = pz - oz;

            double r00 = face_rot[fi*9+0], r01 = face_rot[fi*9+1], r02 = face_rot[fi*9+2];
            double r10 = face_rot[fi*9+3], r11 = face_rot[fi*9+4], r12 = face_rot[fi*9+5];
            double r20 = face_rot[fi*9+6], r21 = face_rot[fi*9+7], r22 = face_rot[fi*9+8];

            double loc_x = r00*ddx + r01*ddy + r02*ddz;
            double loc_y = r10*ddx + r11*ddy + r12*ddz;
            double loc_z_obs = r20*ddx + r21*ddy + r22*ddz;
            double loc_mz = r20*mx + r21*my + r22*mz;

            int e_start = edge_offsets[fi];
            int e_end = edge_offsets[fi + 1];
            int n_edges = e_end - e_start;

            double Sx, Sy, Sz;
            poly_face_field_dev(loc_x, loc_y, loc_z_obs, face_cz[fi],
                               &edge_pts_2d[e_start*2], n_edges,
                               &Sx, &Sy, &Sz);

            double Hx_loc = -ConstForH * loc_mz * Sx;
            double Hy_loc = -ConstForH * loc_mz * Sy;
            double Hz_loc = -ConstForH * loc_mz * Sz;

            Hx_sum += r00*Hx_loc + r10*Hy_loc + r20*Hz_loc;
            Hy_sum += r01*Hx_loc + r11*Hy_loc + r21*Hz_loc;
            Hz_sum += r02*Hx_loc + r12*Hy_loc + r22*Hz_loc;
        }

        // Column dx of the 3x3 block
        block[0*3 + dx] = (float)Hx_sum;
        block[1*3 + dx] = (float)Hy_sum;
        block[2*3 + dx] = (float)Hz_sum;
    }
}

// ============================================================
// Kernel: assemble RecMag interaction matrix blocks
//   One thread per (i,j) pair
//   Each thread computes one 3x3 block including symmetry copies
// ============================================================
__global__ void asm_recmag_kernel(
    const double* __restrict__ centers,     // [3*N] observation centers
    const double* __restrict__ src_centers, // [3*N] source centers
    const double* __restrict__ src_dims,    // [3*N] source dimensions
    int N,
    const double* __restrict__ pt_transforms,    // [n_copies * 9]
    const double* __restrict__ fld_transforms,   // [n_copies * 9]
    int n_copies,
    float* __restrict__ matrix_blocks            // [N * N * 9]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N * N) return;

    int i = idx / N;  // observation element
    int j = idx % N;  // source element

    double obs_x = centers[3*i+0];
    double obs_y = centers[3*i+1];
    double obs_z = centers[3*i+2];

    double sc_x = src_centers[3*j+0];
    double sc_y = src_centers[3*j+1];
    double sc_z = src_centers[3*j+2];
    double wd_x = src_dims[3*j+0];
    double wd_y = src_dims[3*j+1];
    double wd_z = src_dims[3*j+2];

    float accum[9] = {0,0,0,0,0,0,0,0,0};

    for(int sc = 0; sc < n_copies; sc++) {
        const double* T = &pt_transforms[sc * 9];
        const double* F = &fld_transforms[sc * 9];

        // Transform source element center: sc' = T * sc
        double tc_x = T[0]*sc_x + T[1]*sc_y + T[2]*sc_z;
        double tc_y = T[3]*sc_x + T[4]*sc_y + T[5]*sc_z;
        double tc_z = T[6]*sc_x + T[7]*sc_y + T[8]*sc_z;

        // For RecMag, dimensions may change sign under reflection
        // but the field formula uses half-widths which are always positive
        // The sign change is absorbed by the field transform F

        float blk[9];
        recmag_block_dev(obs_x, obs_y, obs_z,
                         tc_x, tc_y, tc_z,
                         wd_x, wd_y, wd_z,
                         blk);

        // Apply field transform: B_actual = F * B_computed
        // And magnetization transform: M_actual = T * M_unit
        // Combined: block_actual[r][c] = sum_r' sum_c' F[r][r'] * blk[r'][c'] * T[c'][c]
        // Since T is orthogonal: T^{-1} = T^T, and we need blk_transformed = F * blk * T^T
        for(int r = 0; r < 3; r++) {
            for(int c = 0; c < 3; c++) {
                double val = 0.0;
                for(int rr = 0; rr < 3; rr++) {
                    for(int cc = 0; cc < 3; cc++) {
                        val += F[r*3+rr] * (double)blk[rr*3+cc] * T[c*3+cc]; // T^T[cc][c] = T[c][cc]
                    }
                }
                accum[r*3+c] += (float)val;
            }
        }
    }

    // Store in output matrix
    long long base = (long long)idx * 9;
    for(int k = 0; k < 9; k++) {
        matrix_blocks[base + k] = accum[k];
    }
}

// ============================================================
// Kernel: assemble polyhedron interaction matrix blocks
//   One thread per (i,j) pair
// ============================================================
__global__ void asm_poly_kernel(
    const double* __restrict__ centers,       // [3*N] observation centers
    int N,
    const int* __restrict__ face_offsets,
    const int* __restrict__ edge_offsets,
    const double* __restrict__ edge_pts_2d,
    const double* __restrict__ face_cz,
    const double* __restrict__ face_rot,
    const double* __restrict__ face_orig,
    const double* __restrict__ pt_transforms,
    const double* __restrict__ fld_transforms,
    int n_copies,
    float* __restrict__ matrix_blocks
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N * N) return;

    int i = idx / N;
    int j = idx % N;

    double obs_x = centers[3*i+0];
    double obs_y = centers[3*i+1];
    double obs_z = centers[3*i+2];

    float accum[9] = {0,0,0,0,0,0,0,0,0};

    for(int sc = 0; sc < n_copies; sc++) {
        const double* T = &pt_transforms[sc * 9];
        const double* F = &fld_transforms[sc * 9];

        // Transform observation point: obs' = T^{-1} * obs = T^T * obs
        // Actually: we transform the SOURCE geometry, not the obs point
        // But equivalently, we can transform obs into the source's symmetric copy frame
        // For polyhedra with face data, it's easier to transform the obs point
        double tp_x = T[0]*obs_x + T[3]*obs_y + T[6]*obs_z;  // T^T * obs
        double tp_y = T[1]*obs_x + T[4]*obs_y + T[7]*obs_z;
        double tp_z = T[2]*obs_x + T[5]*obs_y + T[8]*obs_z;

        float blk[9];
        poly_block_dev(tp_x, tp_y, tp_z,
                       face_offsets, j,
                       edge_offsets, edge_pts_2d,
                       face_cz, face_rot, face_orig,
                       blk);

        // Apply transforms: block_actual = F * blk * T^T
        for(int r = 0; r < 3; r++) {
            for(int c = 0; c < 3; c++) {
                double val = 0.0;
                for(int rr = 0; rr < 3; rr++) {
                    for(int cc = 0; cc < 3; cc++) {
                        val += F[r*3+rr] * (double)blk[rr*3+cc] * T[c*3+cc];
                    }
                }
                accum[r*3+c] += (float)val;
            }
        }
    }

    long long base = (long long)idx * 9;
    for(int k = 0; k < 9; k++) {
        matrix_blocks[base + k] = accum[k];
    }
}

// ============================================================
// Host: run GPU assembly
// ============================================================
int radGPU_AssembleMatrix(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_AsmResult* result)
{
    // Determine N (use whichever is nonzero — currently support one type per model)
    int N = 0;
    bool usePoly = false, useRec = false;

    if(polyData != nullptr && polyData->n_elem > 0) {
        N = polyData->n_elem;
        usePoly = true;
    }
    if(recData != nullptr && recData->n_elem > 0) {
        if(N > 0 && recData->n_elem != N) {
            fprintf(stderr, "radGPU_AssembleMatrix: mixed poly/rec not yet supported in single assembly\n");
            return -1;
        }
        N = recData->n_elem;
        useRec = true;
    }

    if(N == 0) return -1;

    result->N = N;
    long long totalBlocks = (long long)N * N;
    long long matSize = totalBlocks * 9;

    int n_copies = (symData != nullptr) ? symData->n_copies : 1;

    // Allocate output
    result->matrix_blocks = new float[matSize];

    // Device pointers
    float* d_matrix = nullptr;
    double* d_pt_transforms = nullptr;
    double* d_fld_transforms = nullptr;

    #define CUDA_CHK(call) do { \
        cudaError_t e = (call); \
        if(e != cudaSuccess) { \
            fprintf(stderr, "CUDA asm error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            return -1; \
        } \
    } while(0)

    CUDA_CHK(cudaMalloc(&d_matrix, matSize * sizeof(float)));
    CUDA_CHK(cudaMemset(d_matrix, 0, matSize * sizeof(float)));

    // Upload symmetry transforms
    double identityTransform[9] = {1,0,0, 0,1,0, 0,0,1};
    if(n_copies > 0 && symData != nullptr) {
        CUDA_CHK(cudaMalloc(&d_pt_transforms, n_copies * 9 * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_fld_transforms, n_copies * 9 * sizeof(double)));
        CUDA_CHK(cudaMemcpy(d_pt_transforms, symData->point_transforms, n_copies * 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_fld_transforms, symData->field_transforms, n_copies * 9 * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        n_copies = 1;
        CUDA_CHK(cudaMalloc(&d_pt_transforms, 9 * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_fld_transforms, 9 * sizeof(double)));
        CUDA_CHK(cudaMemcpy(d_pt_transforms, identityTransform, 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_fld_transforms, identityTransform, 9 * sizeof(double), cudaMemcpyHostToDevice));
    }

    int tpb = 256;
    int numPairs = N * N;
    int blocks = (numPairs + tpb - 1) / tpb;

    if(useRec) {
        double* d_centers = nullptr;
        double* d_src_centers = nullptr;
        double* d_src_dims = nullptr;

        CUDA_CHK(cudaMalloc(&d_centers, 3 * N * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_src_centers, 3 * N * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_src_dims, 3 * N * sizeof(double)));
        CUDA_CHK(cudaMemcpy(d_centers, recData->centers, 3 * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_src_centers, recData->centers, 3 * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_src_dims, recData->dims, 3 * N * sizeof(double), cudaMemcpyHostToDevice));

        asm_recmag_kernel<<<blocks, tpb>>>(
            d_centers, d_src_centers, d_src_dims, N,
            d_pt_transforms, d_fld_transforms, n_copies,
            d_matrix);

        CUDA_CHK(cudaDeviceSynchronize());
        cudaFree(d_centers);
        cudaFree(d_src_centers);
        cudaFree(d_src_dims);
    }

    if(usePoly) {
        double* d_centers = nullptr;
        int* d_face_offsets = nullptr;
        int* d_edge_offsets = nullptr;
        double* d_edge_pts_2d = nullptr;
        double* d_face_cz = nullptr;
        double* d_face_rot = nullptr;
        double* d_face_orig = nullptr;

        CUDA_CHK(cudaMalloc(&d_centers, 3 * N * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_face_offsets, (N+1) * sizeof(int)));
        CUDA_CHK(cudaMalloc(&d_edge_offsets, (polyData->n_faces_total+1) * sizeof(int)));
        CUDA_CHK(cudaMalloc(&d_edge_pts_2d, 2 * polyData->n_edges_total * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_face_cz, polyData->n_faces_total * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_face_rot, 9 * polyData->n_faces_total * sizeof(double)));
        CUDA_CHK(cudaMalloc(&d_face_orig, 3 * polyData->n_faces_total * sizeof(double)));

        CUDA_CHK(cudaMemcpy(d_centers, polyData->centers, 3 * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_face_offsets, polyData->face_offsets, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_edge_offsets, polyData->edge_offsets, (polyData->n_faces_total+1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_edge_pts_2d, polyData->edge_pts_2d, 2 * polyData->n_edges_total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_face_cz, polyData->face_cz, polyData->n_faces_total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_face_rot, polyData->face_rot, 9 * polyData->n_faces_total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_face_orig, polyData->face_orig, 3 * polyData->n_faces_total * sizeof(double), cudaMemcpyHostToDevice));

        asm_poly_kernel<<<blocks, tpb>>>(
            d_centers, N,
            d_face_offsets, d_edge_offsets,
            d_edge_pts_2d, d_face_cz, d_face_rot, d_face_orig,
            d_pt_transforms, d_fld_transforms, n_copies,
            d_matrix);

        CUDA_CHK(cudaDeviceSynchronize());
        cudaFree(d_centers);
        cudaFree(d_face_offsets);
        cudaFree(d_edge_offsets);
        cudaFree(d_edge_pts_2d);
        cudaFree(d_face_cz);
        cudaFree(d_face_rot);
        cudaFree(d_face_orig);
    }

    // Copy result back
    CUDA_CHK(cudaMemcpy(result->matrix_blocks, d_matrix, matSize * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_matrix);
    cudaFree(d_pt_transforms);
    cudaFree(d_fld_transforms);

    #undef CUDA_CHK
    return 0;
}

// ============================================================
// Free assembly data
// ============================================================
void radGPU_FreeAsmData(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_AsmResult* result)
{
    if(polyData) {
        delete[] polyData->centers;
        delete[] polyData->face_offsets;
        delete[] polyData->edge_offsets;
        delete[] polyData->edge_pts_2d;
        delete[] polyData->face_cz;
        delete[] polyData->face_rot;
        delete[] polyData->face_orig;
        memset(polyData, 0, sizeof(RadGPU_PolyData));
    }
    if(recData) {
        delete[] recData->centers;
        delete[] recData->dims;
        delete[] recData->obs_centers;
        memset(recData, 0, sizeof(RadGPU_RecMagData));
    }
    if(result) {
        delete[] result->matrix_blocks;
        memset(result, 0, sizeof(RadGPU_AsmResult));
    }
}

#endif // RADIA_WITH_CUDA