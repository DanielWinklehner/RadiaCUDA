/************************************************************************//**
 * File: radgpu_fld.cu
 * Description: CUDA kernels for GPU-accelerated magnetic field evaluation.
 *              Computes B field from uniformly magnetized polyhedra and
 *              rectangular parallelepipeds decomposed into polygon faces.
 *
 *              Kernel uses the atan2-based solid angle formulation
 *              (matching field_kernel.py _POLY_KERNEL_FP64).
 *
 *              The polygon-face kernel is templated on the scalar type:
 *              the double instantiation reproduces the original fp64 kernel
 *              bit-for-bit (identical constants and operation order); the
 *              float instantiation trades ~1e-4 relative accuracy for the
 *              much higher fp32 transcendental throughput of consumer GPUs
 *              (intended for field-map visualization, NOT for isochronism).
 *              Select via RadGPUFieldFaceData::use_fp32 (Python:
 *              rad.Fld(..., use_gpu=True, precision='single')).
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
#include <cstdlib>   // getenv, atol
#include <cmath>
#include <vector>


//=========================================================================
// Precision traits for the polygon-face kernel.
//
// The double specialization MUST keep the exact literal values of the
// original fp64 kernel. The float specialization rescales the guards to
// fp32's ~1e-7 relative epsilon, and additionally normalizes the atan2
// arguments: the A1..A4 intermediates reach ~1e30 for steep edges and
// their pairwise products overflow float range (~3.4e38); atan2 is
// scale-invariant, so dividing all four by their max is exact.
//=========================================================================

template<typename T> struct RadGpuFldTr;

template<> struct RadGpuFldTr<double>
{
    static __device__ __forceinline__ double max_k()          { return 1.0e+09; }
    static __device__ __forceinline__ double rel_rand()       { return 1.0e-13; }
    static __device__ __forceinline__ double max_rel_tol()    { return 1.0e-07; }
    static __device__ __forceinline__ double tiny()           { return 1.0e-300; }
    static __device__ __forceinline__ double tiny_r()         { return 1.0e-50; }
    static __device__ __forceinline__ double log_switch()     { return 1.0e-12; }
    static __device__ __forceinline__ double eps_floor()      { return 1.0e-15; }
    static __device__ __forceinline__ double eps_rel()        { return 1.0e-12; }
    static __device__ __forceinline__ double edge_eps_floor() { return 1.0e-12; }
    static __device__ __forceinline__ double edge_eps_rel()   { return 1.0e-9; }
    static __device__ __forceinline__ double huge_val()       { return 1.0e300; }
    static __device__ __forceinline__ bool   normalize_atan2(){ return false; }
};

template<> struct RadGpuFldTr<float>
{
    static __device__ __forceinline__ float max_k()          { return 1.0e+05f; }
    static __device__ __forceinline__ float rel_rand()       { return 1.0e-06f; }
    static __device__ __forceinline__ float max_rel_tol()    { return 1.0e-03f; }
    static __device__ __forceinline__ float tiny()           { return 1.0e-30f; }
    static __device__ __forceinline__ float tiny_r()         { return 1.0e-18f; }
    static __device__ __forceinline__ float log_switch()     { return 1.0e-05f; }
    static __device__ __forceinline__ float eps_floor()      { return 1.0e-07f; }
    static __device__ __forceinline__ float eps_rel()        { return 1.0e-05f; }
    static __device__ __forceinline__ float edge_eps_floor() { return 1.0e-06f; }
    static __device__ __forceinline__ float edge_eps_rel()   { return 1.0e-04f; }
    static __device__ __forceinline__ float huge_val()       { return 1.0e30f; }
    static __device__ __forceinline__ bool  normalize_atan2(){ return true; }
};

// Compute R + u without cancellation when u < 0 (fp32 path): R+u = q/(R-u)
// with q = R^2 - u^2 supplied analytically. In fp32 the naive sum has
// relative error ~1e-7*R/(R+u), which near u ~ -R (observation point close
// to an edge-line extension -- a frequent geometric situation) reaches
// percent level and was the dominant fp32 error source. Not used in the
// fp64 instantiation (keeps the original kernel bit-exact).
template<typename T> struct RadGpuFldAccurateRp
{ static __device__ __forceinline__ bool enabled() { return false; } };
template<> struct RadGpuFldAccurateRp<float>
{ static __device__ __forceinline__ bool enabled() { return true; } };

template<typename T>
__device__ __forceinline__ T radgpu_r_plus_u(T R, T u, T q)
{
    if (RadGpuFldAccurateRp<T>::enabled() && u < T(0))
    {
        T rm = R - u;  // >= R, no cancellation
        if (!(rm > T(0)) || !isfinite(rm)) rm = RadGpuFldTr<T>::tiny();
        T qq = q;
        if (!(qq > T(0)) || !isfinite(qq)) qq = RadGpuFldTr<T>::tiny();
        return qq / rm;
    }
    return R + u;
}


// ===================== Templated helpers (face kernel) =====================

template<typename T>
__device__ inline T radgpu_log_R_plus_u_stable(T R, T u, T q)
{
    // Computes log(R + u) robustly.
    // q must satisfy: q = R^2 - u^2 >= 0 (analytically).
    if (RadGpuFldAccurateRp<T>::enabled())
    {
        // fp32: form R+u cancellation-free for u<0 via q/(R-u), then a single
        // log. Accurate over the WHOLE range (the fp64 threshold scheme has a
        // band just above the switch where the naive sum has already lost
        // most of its float digits).
        T rp = radgpu_r_plus_u(R, u, q);
        if (!(rp > T(0)) || !isfinite(rp)) rp = RadGpuFldTr<T>::tiny();
        return log(rp);
    }

    T rp = R + u;
    if (rp > RadGpuFldTr<T>::log_switch() * R) return log(rp); // safe direct path

    T rm = R - u;
    if (!(rm > T(0)) || !isfinite(rm)) rm = RadGpuFldTr<T>::tiny();
    if (!(q  > T(0)) || !isfinite(q))  q  = RadGpuFldTr<T>::tiny();

    // log(R+u) = log(q) - log(R-u), avoids cancellation when R+u is tiny
    return log(q) - log(rm);
}

template<typename T>
__device__ inline T radgpu_clamp(T v, T lo, T hi)
{
    return fmin(hi, fmax(lo, v));
}

template<typename T>
__device__ inline T radgpu_face_scale(
    const double* __restrict__ verts2d, int vbase, int nv)
{
    T s = T(0);
    for (int i = 0; i < nv; i++) {
        int j = (i + 1) % nv;
        T dx = T(verts2d[vbase + j*2 + 0]) - T(verts2d[vbase + i*2 + 0]);
        T dy = T(verts2d[vbase + j*2 + 1]) - T(verts2d[vbase + i*2 + 1]);
        T L = sqrt(dx*dx + dy*dy);
        if (L > s) s = L;
    }
    if (s < RadGpuFldTr<T>::tiny()) s = T(1);
    return s;
}

template<typename T>
__device__ inline bool radgpu_point_in_poly_2d(
    const double* __restrict__ verts2d, int vbase, int nv, T px, T py)
{
    bool inside = false;
    for (int i = 0, j = nv - 1; i < nv; j = i++) {
        T xi = T(verts2d[vbase + i*2 + 0]);
        T yi = T(verts2d[vbase + i*2 + 1]);
        T xj = T(verts2d[vbase + j*2 + 0]);
        T yj = T(verts2d[vbase + j*2 + 1]);

        bool crosses = ((yi > py) != (yj > py));
        if (crosses) {
            T t = (py - yi) / (yj - yi);
            T xint = xi + t * (xj - xi);
            if (xint > px) inside = !inside;
        }
    }
    return inside;
}

template<typename T>
__device__ inline T radgpu_min_dist2_edges_2d(
    const double* __restrict__ verts2d, int vbase, int nv, T px, T py)
{
    T min_d2 = RadGpuFldTr<T>::huge_val();
    for (int i = 0; i < nv; i++) {
        int j = (i + 1) % nv;
        T ax = T(verts2d[vbase + i*2 + 0]);
        T ay = T(verts2d[vbase + i*2 + 1]);
        T bx = T(verts2d[vbase + j*2 + 0]);
        T by = T(verts2d[vbase + j*2 + 1]);

        T vx = bx - ax, vy = by - ay;
        T wx = px - ax, wy = py - ay;

        T vv = vx*vx + vy*vy;
        T t = (vv > T(0)) ? (wx*vx + wy*vy) / vv : T(0);
        t = radgpu_clamp(t, T(0), T(1));

        T qx = ax + t*vx;
        T qy = ay + t*vy;
        T dx = px - qx, dy = py - qy;
        T d2 = dx*dx + dy*dy;
        if (d2 < min_d2) min_d2 = d2;
    }
    return min_d2;
}

template<typename T>
__device__ inline T radgpu_nudge_small(T v, T eps)
{
    if (fabs(v) < eps) return (v < T(0) ? -eps : +eps);
    return v;
}

template<typename T>
__device__ inline void radgpu_eval_face_integrals_at_z(
    const double* __restrict__ verts2d,
    int vbase, int nv,
    double lx, double ly, T z,
    T eps_xy, T eps_b,
    T& Sx_out, T& Sy_out, T& Sz_out)
{
    // NOTE: lx/ly are double and the vertex-minus-observer subtractions are
    // performed in double BEFORE narrowing to T. For T=double this is
    // identical to subtracting in T; for T=float it removes the dominant
    // fp32 noise source (quantization of O(100 mm) absolute coordinates,
    // ~1e-5 mm, which otherwise varies point-to-point and speckles the map).
    // The narrowed differences are small, well-scaled numbers.
    const T Max_k = RadGpuFldTr<T>::max_k();
    const T RelRandMagn = RadGpuFldTr<T>::rel_rand();
    const T MaxRelTolToSwitch = RadGpuFldTr<T>::max_rel_tol();

    if (nv < 3) {
        Sx_out = T(0); Sy_out = T(0); Sz_out = T(0);
        return;
    }

    T ze2 = z * z;
    T Sx = T(0), Sy = T(0), Sz = T(0);
    T Sx_log_extra = T(0); // replaces ArgSumLogs2 product path

    T x1 = radgpu_nudge_small(T(verts2d[vbase + 0] - lx), eps_xy);
    T y1 = radgpu_nudge_small(T(verts2d[vbase + 1] - ly), eps_xy);
    T x1e2 = x1 * x1;

    for (int ei = 0; ei < nv; ei++)
    {
        int vnext = vbase + ((ei + 1) % nv) * 2;
        T x2 = radgpu_nudge_small(T(verts2d[vnext + 0] - lx), eps_xy);
        T y2 = radgpu_nudge_small(T(verts2d[vnext + 1] - ly), eps_xy);
        T x2e2 = x2 * x2;

        T x2mx1 = x2 - x1;
        T y2my1 = y2 - y1;

        if (fabs(x2mx1) * Max_k > fabs(y2my1))
        {
            T k = y2my1 / x2mx1;
            T b = radgpu_nudge_small(y1 - k * x1, eps_b);

            T ke2 = k * k;
            T be2 = b * b;
            T ke2p1 = ke2 + T(1);
            T sqrtke2p1 = sqrt(ke2p1);
            T bk = b * k;

            T bpkx1 = b + k * x1;
            T bpkx2 = b + k * x2;
            T bpkx1e2 = bpkx1 * bpkx1;
            T bpkx2e2 = bpkx2 * bpkx2;

            T R1 = sqrt(x1e2 + bpkx1e2 + ze2);
            T R2 = sqrt(x2e2 + bpkx2e2 + ze2);

            // fp32: cancellation-free R + bpkx via q/(R - bpkx), with
            // q = R^2 - bpkx^2 = x^2 + z^2 (no-op in the fp64 instantiation)
            T R1pbpkx1 = radgpu_r_plus_u(R1, bpkx1, x1e2 + ze2);
            T R2pbpkx2 = radgpu_r_plus_u(R2, bpkx2, x2e2 + ze2);

            // keep your existing R+... protection (important for A1..A4 path)
            T AbsRandR1 = T(100) * R1 * RelRandMagn;
            T AbsRandR2 = T(100) * R2 * RelRandMagn;
            T MaxAbsRandR1 = MaxRelTolToSwitch * R1;
            T MaxAbsRandR2 = MaxRelTolToSwitch * R2;
            if (AbsRandR1 > MaxAbsRandR1) AbsRandR1 = MaxAbsRandR1;
            if (AbsRandR2 > MaxAbsRandR2) AbsRandR2 = MaxAbsRandR2;

            T x1e2pze2 = x1e2 + ze2;
            if (fabs(R1pbpkx1) < AbsRandR1 && R1 > T(100) * AbsRandR1 &&
                x1e2pze2 < bpkx1e2 * MaxRelTolToSwitch)
                R1pbpkx1 = (bpkx1 != T(0)) ? T(0.5) * x1e2pze2 / fabs(bpkx1) : RadGpuFldTr<T>::tiny_r();

            if (fabs(R2pbpkx2) < AbsRandR2 && R2 > T(100) * AbsRandR2 &&
                (x2e2 + ze2) < bpkx2e2 * MaxRelTolToSwitch)
                R2pbpkx2 = (bpkx2 != T(0)) ? T(0.5) * (x2e2 + ze2) / fabs(bpkx2) : RadGpuFldTr<T>::tiny_r();

            if (R1pbpkx1 == T(0)) R1pbpkx1 = RadGpuFldTr<T>::tiny_r();
            if (R2pbpkx2 == T(0)) R2pbpkx2 = RadGpuFldTr<T>::tiny_r();

            // Sz (atan2 accumulation) as before
            T kze2 = k * ze2;
            T ke2ze2 = k * kze2;
            T ke2ze2pbe2 = ke2ze2 + be2;
            T ke2ze2mbe2 = ke2ze2 - be2;
            T bx1 = b * x1, bx2 = b * x2;
            T twob = T(2) * b;
            T kx1mb = k * x1 - b, kx2mb = k * x2 - b;

            T A1 = -(ke2ze2pbe2 * (bx1 + kze2) * R1pbpkx1 + kze2 * twob * (x1e2 + ze2));
            T A2 =  (ke2ze2pbe2 * kx1mb * R1pbpkx1 + ke2ze2mbe2 * (x1e2 + ze2)) * z;
            T A3 =   ke2ze2pbe2 * (bx2 + kze2) * R2pbpkx2 + kze2 * twob * (x2e2 + ze2);
            T A4 =  (ke2ze2pbe2 * kx2mb * R2pbpkx2 + ke2ze2mbe2 * (x2e2 + ze2)) * z;

            if (RadGpuFldTr<T>::normalize_atan2())
            {
                // atan2(A1*A4 + A3*A2, A2*A4 - A1*A3) is invariant under a
                // common rescale of A1..A4 (both args scale by s^2). The A's
                // reach ~1e30 for steep edges, so in fp32 the raw products
                // would overflow to inf -> NaN; normalize by the max first.
                T s = fmax(fmax(fabs(A1), fabs(A2)), fmax(fabs(A3), fabs(A4)));
                if (s > RadGpuFldTr<T>::tiny())
                {
                    T inv = T(1) / s;
                    A1 *= inv; A2 *= inv; A3 *= inv; A4 *= inv;
                }
            }

            Sz += atan2(A1 * A4 + A3 * A2, A2 * A4 - A1 * A3);

            // -------- stable SL1 --------
            // u = (bk + (1+k^2)x)/sqrt(1+k^2)
            T u1 = (bk + ke2p1 * x1) / sqrtke2p1;
            T u2 = (bk + ke2p1 * x2) / sqrtke2p1;

            // qv = R^2 - u^2 = z^2 + b^2/(1+k^2)
            T qv = ze2 + be2 / ke2p1;
            if (!(qv > T(0)) || !isfinite(qv)) qv = RadGpuFldTr<T>::tiny();

            T logv1 = radgpu_log_R_plus_u_stable(R1, u1, qv);
            T logv2 = radgpu_log_R_plus_u_stable(R2, u2, qv);

            T SL1 = (logv2 - logv1) / sqrtke2p1;

            Sx += -k * SL1;
            Sy +=  SL1;

            // -------- stable replacement for ArgSumLogs2 --------
            // log(R + (b+kx)) with q = R^2 - (b+kx)^2 = x^2 + z^2
            T qx1 = x1e2 + ze2;
            T qx2 = x2e2 + ze2;
            if (!(qx1 > T(0)) || !isfinite(qx1)) qx1 = RadGpuFldTr<T>::tiny();
            if (!(qx2 > T(0)) || !isfinite(qx2)) qx2 = RadGpuFldTr<T>::tiny();

            T logrp1 = radgpu_log_R_plus_u_stable(R1, bpkx1, qx1);
            T logrp2 = radgpu_log_R_plus_u_stable(R2, bpkx2, qx2);

            Sx_log_extra += (logrp2 - logrp1);
        }

        x1 = x2; y1 = y2; x1e2 = x2e2;
    }

    Sx += Sx_log_extra;

    if (!isfinite(Sx)) Sx = T(0);
    if (!isfinite(Sy)) Sy = T(0);
    if (!isfinite(Sz)) Sz = T(0);

    Sx_out = Sx;
    Sy_out = Sy;
    Sz_out = Sz;
}

//=========================================================================
// Polygon-face kernel, templated on the evaluation scalar type T.
// Geometry/observation arrays stay double (the kernel is compute-bound on
// transcendentals, not bandwidth); values are cast to T at load. The
// per-face contributions are accumulated in double for both instantiations,
// so fp32 accuracy is limited by the per-term error (~1e-6 relative), not
// by cancellation over the (up to millions of) faces.
//=========================================================================

template<typename T>
__global__
void radGPU_FldKernelT(
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

    double Bx = 0.0, By = 0.0, Bz = 0.0;

    int face_start = src_block_idx * blockDim.x;
    int face_end   = face_start + blockDim.x;
    if (face_end > n_faces) face_end = n_faces;

    const T PI = T(3.14159265358979323846);
    const T ConstForH = T(1) / (T(4) * PI);

    for (int fi = face_start; fi < face_end; fi++)
    {
        int nv = nverts[fi];
        if (nv < 3) continue;

        int tb = fi * 9;
        T T00 = T(transform[tb + 0]), T01 = T(transform[tb + 1]), T02 = T(transform[tb + 2]);
        T T10 = T(transform[tb + 3]), T11 = T(transform[tb + 4]), T12 = T(transform[tb + 5]);
        T T20 = T(transform[tb + 6]), T21 = T(transform[tb + 7]), T22 = T(transform[tb + 8]);

        // The lab->local transform of the observation point is kept in DOUBLE
        // for both instantiations: it involves O(100 mm) absolute coordinates
        // whose fp32 quantization would dominate the error budget, it is a
        // handful of FMAs per face (negligible next to the per-edge
        // transcendentals), and for T=double it is the original code path.
        double I00 = inv_transform[tb + 0], I01 = inv_transform[tb + 1], I02 = inv_transform[tb + 2];
        double I10 = inv_transform[tb + 3], I11 = inv_transform[tb + 4], I12 = inv_transform[tb + 5];
        double I20 = inv_transform[tb + 6], I21 = inv_transform[tb + 7], I22 = inv_transform[tb + 8];

        int f3 = fi * 3;
        double ox = origin[f3 + 0], oy = origin[f3 + 1], oz = origin[f3 + 2];
        double mx = mag[f3 + 0],    my = mag[f3 + 1],    mz = mag[f3 + 2];

        // observation in local frame (double)
        double px_d = obs[obs_idx * 3 + 0];
        double py_d = obs[obs_idx * 3 + 1];
        double pz_d = obs[obs_idx * 3 + 2];
        double dpx = px_d - ox, dpy = py_d - oy, dpz = pz_d - oz;
        double lx = I00 * dpx + I01 * dpy + I02 * dpz;
        double ly = I10 * dpx + I11 * dpy + I12 * dpz;
        double lz = I20 * dpx + I21 * dpy + I22 * dpz;

        // magnetization local z
        T mlz = T(I20 * mx + I21 * my + I22 * mz);

        int vbase = fi * RADGPU_FLD_MAX_VERTS * 2;
        double cz = coordz[fi];
        T z_raw = T(cz - lz);

        // scale-aware epsilons
        T face_scale = radgpu_face_scale<T>(verts2d, vbase, nv);
        T eps_z  = fmax(RadGpuFldTr<T>::eps_floor(), RadGpuFldTr<T>::eps_rel() * face_scale);
        T eps_xy = fmax(RadGpuFldTr<T>::eps_floor(), RadGpuFldTr<T>::eps_rel() * face_scale);
        T eps_b  = fmax(RadGpuFldTr<T>::eps_floor(), RadGpuFldTr<T>::eps_rel() * face_scale);

        // regular z evaluation (nudged away from exactly 0)
        T z_eval = z_raw;
        if (fabs(z_eval) < eps_z) z_eval = (z_eval < T(0)) ? -eps_z : +eps_z;

        T Sx = T(0), Sy = T(0), Sz = T(0);

        // near-plane handling with two-sided limit around z=0
        bool near_plane = (fabs(z_raw) <= T(10) * eps_z);

        if (near_plane)
        {
            bool inside2d = radgpu_point_in_poly_2d<T>(verts2d, vbase, nv, T(lx), T(ly));

            T edge_eps = fmax(RadGpuFldTr<T>::edge_eps_floor(),
                              RadGpuFldTr<T>::edge_eps_rel() * face_scale);
            T min_d2 = radgpu_min_dist2_edges_2d<T>(verts2d, vbase, nv, T(lx), T(ly));
            bool near_edge = (min_d2 <= edge_eps * edge_eps);

            if (inside2d || near_edge)
            {
                T Sx_p, Sy_p, Sz_p;
                T Sx_m, Sy_m, Sz_m;

                radgpu_eval_face_integrals_at_z<T>(
                    verts2d, vbase, nv, lx, ly, +eps_z, eps_xy, eps_b, Sx_p, Sy_p, Sz_p);

                radgpu_eval_face_integrals_at_z<T>(
                    verts2d, vbase, nv, lx, ly, -eps_z, eps_xy, eps_b, Sx_m, Sy_m, Sz_m);

                if (inside2d)
                {
                    // principal-value style on-face limit
                    Sx = T(0.5) * (Sx_p + Sx_m);
                    Sy = T(0.5) * (Sy_p + Sy_m);
                    Sz = T(0.5) * (Sz_p + Sz_m);
                }
                else
                {
                    // off-face but near edge: one-sided from z_raw
                    if (z_raw >= T(0)) { Sx = Sx_p; Sy = Sy_p; Sz = Sz_p; }
                    else               { Sx = Sx_m; Sy = Sy_m; Sz = Sz_m; }
                }
            }
            else
            {
                radgpu_eval_face_integrals_at_z<T>(
                    verts2d, vbase, nv, lx, ly, z_eval, eps_xy, eps_b, Sx, Sy, Sz);
            }
        }
        else
        {
            radgpu_eval_face_integrals_at_z<T>(
                verts2d, vbase, nv, lx, ly, z_eval, eps_xy, eps_b, Sx, Sy, Sz);
        }

        if (!isfinite(Sx) || !isfinite(Sy) || !isfinite(Sz)) continue;

        T Hx_loc = -ConstForH * mlz * Sx;
        T Hy_loc = -ConstForH * mlz * Sy;
        T Hz_loc = -ConstForH * mlz * Sz;

        // local -> lab; accumulate in double
        Bx += double(T00 * Hx_loc + T01 * Hy_loc + T02 * Hz_loc);
        By += double(T10 * Hx_loc + T11 * Hy_loc + T12 * Hz_loc);
        Bz += double(T20 * Hx_loc + T21 * Hy_loc + T22 * Hz_loc);
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
// Choose the largest observation-point chunk that fits in VRAM (issue #12).
// perObs = (n_src_blocks + 2) * 3 doubles  (partial_B + obs + result per point).
// Uses 85% of currently-free VRAM as a margin for fragmentation/other allocations.
// Test hook: env var RADIA_GPU_FLD_MAX_OBS_CHUNK, if set to a positive integer,
// further clamps the chunk so the chunking path can be exercised deterministically
// on a small observation grid.
//-------------------------------------------------------------------------
int radGPU_FldMaxObsChunk(int n_src_blocks, size_t geom_bytes, int n_obs_total)
{
    if (n_obs_total <= 0) return 1;
    int cap = n_obs_total; // default: no VRAM-driven chunking

    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess)
    {
        size_t budget = (size_t)((double)freeB * 0.85);
        size_t perObs = (size_t)(n_src_blocks + 2) * 3 * sizeof(double);
        if (perObs == 0 || budget <= geom_bytes)
        {
            cap = 1; // geometry alone is tight: try one point at a time
        }
        else
        {
            unsigned long long c = (unsigned long long)((budget - geom_bytes) / perObs);
            if (c < (unsigned long long)cap) cap = (c < 1) ? 1 : (int)c;
        }
    }

    // Test hook (issue #12): force a smaller chunk to exercise the chunking path.
    const char* env = getenv("RADIA_GPU_FLD_MAX_OBS_CHUNK");
    if (env && *env)
    {
        long v = atol(env);
        if (v >= 1 && v < (long)cap) cap = (int)v;
    }

    // Diagnostic: chunking is benign (each chunk still launches
    // ~n_obs_chunk/128 x n_src_blocks blocks, far above GPU saturation, and
    // the total pair work is chunk-count-invariant), but make it VISIBLE so
    // slow evaluations can be attributed correctly.
    if (cap < n_obs_total)
    {
        int n_chunks = (n_obs_total + cap - 1) / cap;
        fprintf(stderr,
                "radGPU_Fld: %d obs points in %d chunks of <= %d "
                "(%d src blocks, free VRAM %.0f MB)\n",
                n_obs_total, n_chunks, cap, n_src_blocks,
                (double)freeB / (1024.0 * 1024.0));
    }

    return (cap < 1) ? 1 : cap;
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
// data->use_fp32 selects the float instantiation of the face kernel
// (visualization-grade accuracy at much higher throughput on GeForce-class
// GPUs); default (0) is the original double kernel.
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

    if (data->use_fp32)
    {
        radGPU_FldKernelT<float><<<grid, block>>>(
            data->d_verts2d, data->d_nverts, data->d_coordz,
            data->d_transform, data->d_inv_transform, data->d_origin,
            data->d_mag, nf, data->d_obs, nobs, data->d_partial_B, nsb
        );
    }
    else
    {
        radGPU_FldKernelT<double><<<grid, block>>>(
            data->d_verts2d, data->d_nverts, data->d_coordz,
            data->d_transform, data->d_inv_transform, data->d_origin,
            data->d_mag, nf, data->d_obs, nobs, data->d_partial_B, nsb
        );
    }

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
// NOTE: the RecMag kernel is fp64-only; the precision='single' option
// currently applies to the polygon-face kernel (the dominant cost for
// tet-meshed models). RecMag models are typically small enough in fp64.
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
