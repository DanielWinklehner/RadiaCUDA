/*-------------------------------------------------------------------------
*
* File name:      radgpu_asm.cu
*
* Project:        RADIA
*
* Description:    GPU-accelerated interaction matrix assembly
*                 Polyhedron field via flat polygon face integrals;
*                 RecMag (cuboid) field via the closed-form Q-tensor.
*                 Mixed models supported (per-source-type branch).
*
-------------------------------------------------------------------------*/

#ifdef RADIA_WITH_CUDA

#include "radgpu_asm.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <new>          // std::nothrow
#include <cuda_runtime.h>

// ============================================================
// Device helper: TransAtans (matches Radia's CPU version)
// ============================================================
__device__ double TransAtans_dev(double x, double y, double& PiMult)
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

__device__ double Sign_dev(double x)
{
    return (x >= 0.0) ? 1.0 : -1.0;
}

__device__ double Step_dev(double x)
{
    return (x >= 0.0) ? 1.0 : 0.0;
}

// ============================================================
// Device: Flat polygon B_comp for PreRelax_ mode
// Computes Sx, Sy, Sz for a single polygon face
// in the face's local coordinate frame
// ============================================================
__device__ void polygon_prerelax_dev(
    double obs_x, double obs_y, double obs_z, // obs point in face local frame
    double coordZ,                              // face z-coordinate in local frame
    const double* edge_pts,                     // 2D edge points [x0,y0, x1,y1, ...]
    int n_edges,
    double& out_Sx, double& out_Sy, double& out_Sz)
{
    const double PI = 3.14159265358979;
//     const double ConstForH = 1.0 / (4.0 * PI);
    const double Max_k = 1.0e+09;
//     const double TINY = 1.0e-15;

    // Compute size-dependent jitter (matches Radia's AbsRandMagnitude)
    double charSize = fabs(edge_pts[0] - obs_x);
    double testSize = fabs(edge_pts[1] - obs_y);
    if(testSize > charSize) charSize = testSize;
    testSize = fabs(coordZ);
    if(testSize > charSize) charSize = testSize;
    double TINY = charSize * 1.0e-13;
    if(TINY < 1.0e-20) TINY = 1.0e-20;

    double z = coordZ - obs_z;
    if(z == 0.0) z = TINY;
    double ze2 = z * z;

    // First edge point relative to observer
    double x1 = edge_pts[0] - obs_x;
    double y1 = edge_pts[1] - obs_y;
    if(x1 == 0.0) x1 = TINY;
    if(y1 == 0.0) y1 = TINY;

    double x1e2 = x1 * x1;
//     double y1e2 = y1 * y1;

    double Sx = 0.0, Sy = 0.0;
    double ArgSumAtans1 = 0.0, PiMultSumAtans1 = 0.0;
    double ArgSumLogs2 = 1.0;

    int n_mi_1 = n_edges - 1;

    for(int i = 0; i < n_edges; i++)
    {
        double x2, y2;
        if(i != n_mi_1)
        {
            x2 = edge_pts[2*(i+1)]     - obs_x;
            y2 = edge_pts[2*(i+1) + 1] - obs_y;
        }
        else
        {
            x2 = edge_pts[0] - obs_x;
            y2 = edge_pts[1] - obs_y;
        }
        if(x2 == 0.0) x2 = TINY;
        if(y2 == 0.0) y2 = TINY;

        double x2e2 = x2 * x2;
//         double y2e2 = y2 * y2;

        double x2mx1 = x2 - x1;
        double y2my1 = y2 - y1;
        double abs_x2mx1 = fabs(x2mx1);
        double abs_y2my1 = fabs(y2my1);

        if(abs_x2mx1 * Max_k > abs_y2my1)
        {
            double k = y2my1 / x2mx1;
            double b = y1 - k * x1;
            if(b == 0.0) b = TINY;

            double bk = b * k, ke2 = k * k, be2 = b * b, twob = 2.0 * b;
            double ke2p1 = ke2 + 1.0;
            double sqrtke2p1 = sqrt(ke2p1);

            double bpkx1 = y1, bpkx2 = y2;
            double bpkx1e2 = bpkx1 * bpkx1, bpkx2e2 = bpkx2 * bpkx2;
            double kx1mb = -b + k*x1, kx2mb = -b + k*x2;
            double R1 = sqrt(x1e2 + bpkx1e2 + ze2);
            double R2 = sqrt(x2e2 + bpkx2e2 + ze2);

            double x1e2pze2 = x1e2 + ze2, x2e2pze2 = x2e2 + ze2;
            double bkpx1pke2x1 = bk + ke2p1 * x1;
            double bkpx2pke2x2 = bk + ke2p1 * x2;
            double kze2 = k * ze2;
            double ke2ze2 = k * kze2;
            double ke2ze2mbe2 = ke2ze2 - be2, ke2ze2pbe2 = ke2ze2 + be2;
            double bx1 = b * x1, bx2 = b * x2;
            double R1pbpkx1 = bpkx1 + R1, R2pbpkx2 = bpkx2 + R2;

            // Flip repair for atan summation
            double FlpRep1ForSumAtans1 = 0.0;
            double four_be2ke2 = 4.0 * be2 * ke2;
            double four_be2be2ke2 = be2 * four_be2ke2;
            double be2mke2ze2 = be2 - ke2ze2, be2pke2ze2 = be2 + ke2ze2;
            double be2mke2ze2e2 = be2mke2ze2 * be2mke2ze2;
            double be2pke2ze2e2 = be2pke2ze2 * be2pke2ze2;
            double DFlipRep = (be2 + ke2p1*ze2) * (four_be2ke2*(be2+ke2ze2) - be2mke2ze2e2);
            double BufDen = four_be2be2ke2 - ke2p1 * be2mke2ze2e2;

            if((DFlipRep >= 0.0) && (BufDen != 0.0))
            {
                double Buf1Num = bk * be2pke2ze2e2;
                double Buf2Num = be2mke2ze2 * sqrt(DFlipRep);

                for(int iFlp = 0; iFlp < 2; iFlp++)
                {
                    double xFlp = (iFlp == 0) ?
                        (Buf1Num - Buf2Num) / BufDen :
                        (Buf1Num + Buf2Num) / BufDen;

                    bool inRange = (x1 < x2) ?
                        ((xFlp > x1) && (xFlp < x2)) :
                        ((xFlp < x1) && (xFlp > x2));
                    if(inRange)
                    {
                        double xFlpe2 = xFlp * xFlp;
                        double kxFlp = k * xFlp;
                        double kxFlppb = kxFlp + b, kxFlpmb = kxFlp - b;
                        double SqRoot = sqrt(xFlpe2 + kxFlppb*kxFlppb + ze2);

                        if(Sign_dev((xFlpe2+ze2)*(-be2mke2ze2) + (-be2+ke2*xFlpe2)*be2pke2ze2) == Sign_dev(-kxFlpmb))
                        {
                            double DenomSign = Sign_dev(-2.0*xFlp*be2mke2ze2 + kxFlpmb*be2pke2ze2*(k+(bk+ke2p1*xFlp)/SqRoot) + k*be2pke2ze2*(kxFlppb + SqRoot));
                            double NumSign = Sign_dev((2.0*bk*ze2*(xFlpe2+ze2) + (b*xFlp+kze2)*be2pke2ze2*(kxFlppb + SqRoot))/z);
                            FlpRep1ForSumAtans1 += -DenomSign * NumSign * Sign_dev(x2mx1);
                        }
                    }
                }
            }

            // Main atan arguments
            double Arg1 = -(ke2ze2pbe2*(bx1 + kze2)*R1pbpkx1 + kze2*twob*x1e2pze2);
            double Arg2 = (ke2ze2pbe2*kx1mb*R1pbpkx1 + ke2ze2mbe2*x1e2pze2)*z;
            double Arg3 = ke2ze2pbe2*(bx2 + kze2)*R2pbpkx2 + kze2*twob*x2e2pze2;
            double Arg4 = (ke2ze2pbe2*kx2mb*R2pbpkx2 + ke2ze2mbe2*x2e2pze2)*z;

            if(Arg2 == 0.0) Arg2 = 1.0e-50;
            if(Arg4 == 0.0) Arg4 = 1.0e-50;

            double PiMult1 = 0.0, PiMult2 = 0.0;
            double CurArg = TransAtans_dev(Arg1/Arg2, Arg3/Arg4, PiMult1);
            ArgSumAtans1 = TransAtans_dev(ArgSumAtans1, CurArg, PiMult2);
            PiMultSumAtans1 += PiMult1 + PiMult2 + FlpRep1ForSumAtans1;

            // Log terms
            double bkpx1_over_sqrt_pR1 = bkpx1pke2x1/sqrtke2p1 + R1;
            double bkpx2_over_sqrt_pR2 = bkpx2pke2x2/sqrtke2p1 + R2;

            if(bkpx1_over_sqrt_pR1 == 0.0) bkpx1_over_sqrt_pR1 = 1.0e-50;
            if(bkpx2_over_sqrt_pR2 == 0.0) bkpx2_over_sqrt_pR2 = 1.0e-50;

            double SumLogs1 = log(bkpx2_over_sqrt_pR2 / bkpx1_over_sqrt_pR1);
            double SumLogs1dsqrtke2p1 = SumLogs1 / sqrtke2p1;

            if(R1pbpkx1 == 0.0) R1pbpkx1 = 1.0e-50;
            ArgSumLogs2 *= (R2pbpkx2 / R1pbpkx1);

            Sx += -k * SumLogs1dsqrtke2p1;
            Sy += SumLogs1dsqrtke2p1;
        }

        x1 = x2; y1 = y2;
        x1e2 = x2e2; //y1e2 = y2e2;
    }

    double Sz_val = atan(ArgSumAtans1) + PiMultSumAtans1 * PI;
    if(ArgSumLogs2 <= 0.0) ArgSumLogs2 = 1.0e-50;
    Sx += log(ArgSumLogs2);

    out_Sx = Sx;
    out_Sy = Sy;
    out_Sz = Sz_val;
}

// ============================================================
// Device: TransAtans with radTg3d::TransAtans semantics (radg3d.h:938).
// NOTE: differs from TransAtans_dev above (which matches the CPU flat-polygon
// code) in the degenerate 1-x*y==0 branch. The RecMag closed form calls the
// radTg3d version on the CPU, so the port must use these exact semantics.
// ============================================================
__device__ double TransAtansRec_dev(double x, double y, double& PiMult)
{
    double Buf = 1. - x * y;
    if(Buf == 0.) Buf = 1.e-50;
    PiMult = (((Buf > 0)? 0.:1.) * ((x < 0)? -1.:1.));
    return (x + y) / Buf;
}

// ============================================================
// Device: radTConvergRepair::AbsRandMagnitude (radcnvrg.h:85) --
// deterministic despite the name: max(RelRand*A, AbsRand), with the
// ZeroRand floor when A==0. Tolerances are passed in from the host's
// radCR snapshot so runtime changes (rad.FldCmpCrt) are honored.
// ============================================================
__device__ double AbsRandMag_dev(double A,
    double AbsRand, double RelRand, double ZeroRand, int ActOnDoubles)
{
    if(!ActOnDoubles) return 0.;
    double AbsFromRel = RelRand * A;
    double AbsMax = (AbsFromRel < AbsRand)? AbsRand : AbsFromRel;
    return (A != 0.)? AbsMax : ((AbsMax < ZeroRand)? ZeroRand : AbsMax);
}

// ============================================================
// Device: RecMag (uniformly magnetized cuboid) PreRelax Q-tensor.
// Exact port of the radTRecMag::B_comp exact branch (radrec.cpp:80-301)
// for the assembly field key (B_ | H_ | PreRelax_; no A_/Phi_/J):
//   Q = [ T.x  -S.z  -S.y ]
//       [-S.z   T.y  -S.x ]     (rows = Field.B / Field.H / Field.A)
//       [-S.y  -S.x   T.z ]
// The multipole branch never fires during assembly (MltplThresh[] are all 0,
// radg3d.h:483), so only the exact closed form is needed.
// obs is the observation point in the SOURCE element's own frame.
// ============================================================
__device__ void recmag_prerelax_dev(
    double obs_x, double obs_y, double obs_z,
    double cen_x, double cen_y, double cen_z,
    double dim_x, double dim_y, double dim_z,   // FULL edge lengths
    double AbsRand, double RelRand, double ZeroRand, int ActOnDoubles,
    double* Q)                                   // [9] row-major output
{
    double Px = obs_x - cen_x;                   // P_min_CenPo
    double Py = obs_y - cen_y;
    double Pz = obs_z - cen_z;
    double HalfDimX = 0.5 * dim_x;
    double HalfDimY = 0.5 * dim_y;
    double HalfDimZ = 0.5 * dim_z;

    // BfSt: source-corner coordinates relative to the observation point,
    // with the on-border jitter repair (radrec.cpp:84-97)
    double x0 = -Px - HalfDimX, x1 = -Px + HalfDimX;
    double y0 = -Py - HalfDimY, y1 = -Py + HalfDimY;
    double z0 = -Pz - HalfDimZ, z1 = -Pz + HalfDimZ;
    if(x0 == 0.) x0 = AbsRandMag_dev(HalfDimX, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    if(y0 == 0.) y0 = AbsRandMag_dev(HalfDimY, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    if(z0 == 0.) z0 = AbsRandMag_dev(HalfDimZ, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    if(x1 == 0.) x1 = AbsRandMag_dev(HalfDimX, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    if(y1 == 0.) y1 = AbsRandMag_dev(HalfDimY, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    if(z1 == 0.) z1 = AbsRandMag_dev(HalfDimZ, AbsRand, RelRand, ZeroRand, ActOnDoubles);

    double x0e2 = x0*x0, x1e2 = x1*x1;
    double y0e2 = y0*y0, y1e2 = y1*y1;
    double z0e2 = z0*z0, z1e2 = z1*z1;

    double D000 = sqrt(x0e2+y0e2+z0e2);
    double D100 = sqrt(x1e2+y0e2+z0e2);
    double D010 = sqrt(x0e2+y1e2+z0e2);
    double D110 = sqrt(x1e2+y1e2+z0e2);
    double D001 = sqrt(x0e2+y0e2+z1e2);
    double D101 = sqrt(x1e2+y0e2+z1e2);
    double D011 = sqrt(x0e2+y1e2+z1e2);
    double D111 = sqrt(x1e2+y1e2+z1e2);

    const double Pi = 3.141592653589793238;
    double PiMult1, PiMult2, PiMult3;

    double T0x = atan(TransAtansRec_dev(TransAtansRec_dev(y0*z0/(x0*D000), -y0*z1/(x0*D001), PiMult1),
                       TransAtansRec_dev(-y1*z0/(x0*D010), y1*z1/(x0*D011), PiMult2), PiMult3))+Pi*(PiMult1+PiMult2+PiMult3);
    double T1x = atan(TransAtansRec_dev(TransAtansRec_dev(-y0*z0/(x1*D100), y0*z1/(x1*D101), PiMult1),
                       TransAtansRec_dev(y1*z0/(x1*D110), -y1*z1/(x1*D111), PiMult2), PiMult3))+Pi*(PiMult1+PiMult2+PiMult3);
    double T0y = atan(TransAtansRec_dev(TransAtansRec_dev(x0*z0/(y0*D000), -x0*z1/(y0*D001), PiMult1),
                       TransAtansRec_dev(-x1*z0/(y0*D100), x1*z1/(y0*D101), PiMult2), PiMult3))+Pi*(PiMult1+PiMult2+PiMult3);
    double T1y = atan(TransAtansRec_dev(TransAtansRec_dev(-x0*z0/(y1*D010), x0*z1/(y1*D011), PiMult1),
                       TransAtansRec_dev(x1*z0/(y1*D110), -x1*z1/(y1*D111), PiMult2), PiMult3))+Pi*(PiMult1+PiMult2+PiMult3);
    double T0z = atan(TransAtansRec_dev(TransAtansRec_dev(x0*y0/(z0*D000), -x1*y0/(z0*D100), PiMult1),
                       TransAtansRec_dev(-x0*y1/(z0*D010), x1*y1/(z0*D110), PiMult2), PiMult3))+Pi*(PiMult1+PiMult2+PiMult3);
    double T1z = atan(TransAtansRec_dev(TransAtansRec_dev(-x0*y0/(z1*D001), x1*y0/(z1*D101), PiMult1),
                       TransAtansRec_dev(x0*y1/(z1*D011), -x1*y1/(z1*D111), PiMult2), PiMult3))+Pi*(PiMult1+PiMult2+PiMult3);

    double AbsRandD000 = 10.*AbsRandMag_dev(D000, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD010 = 10.*AbsRandMag_dev(D010, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD001 = 10.*AbsRandMag_dev(D001, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD011 = 10.*AbsRandMag_dev(D011, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD100 = 10.*AbsRandMag_dev(D100, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD110 = 10.*AbsRandMag_dev(D110, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD101 = 10.*AbsRandMag_dev(D101, AbsRand, RelRand, ZeroRand, ActOnDoubles);
    double AbsRandD111 = 10.*AbsRandMag_dev(D111, AbsRand, RelRand, ZeroRand, ActOnDoubles);

    // Catastrophic-cancellation guards on the log arguments (radrec.cpp:156-181):
    // when c+D ~ 0 the exact expression is replaced by its series limit.
    double z0plD100 = z0+D100; if(z0plD100 < AbsRandD100) z0plD100 = 0.5*(x1e2 + y0e2)/fabs(z0);
    double z1plD101 = z1+D101; if(z1plD101 < AbsRandD101) z1plD101 = 0.5*(x1e2 + y0e2)/fabs(z1);
    double z1plD001 = z1+D001; if(z1plD001 < AbsRandD001) z1plD001 = 0.5*(x0e2 + y0e2)/fabs(z1);
    double z0plD000 = z0+D000; if(z0plD000 < AbsRandD000) z0plD000 = 0.5*(x0e2 + y0e2)/fabs(z0);
    double z0plD010 = z0+D010; if(z0plD010 < AbsRandD010) z0plD010 = 0.5*(x0e2 + y1e2)/fabs(z0);
    double z1plD011 = z1+D011; if(z1plD011 < AbsRandD011) z1plD011 = 0.5*(x0e2 + y1e2)/fabs(z1);
    double z1plD111 = z1+D111; if(z1plD111 < AbsRandD111) z1plD111 = 0.5*(x1e2 + y1e2)/fabs(z1);
    double z0plD110 = z0+D110; if(z0plD110 < AbsRandD110) z0plD110 = 0.5*(x1e2 + y1e2)/fabs(z0);

    double y0plD100 = y0+D100; if(y0plD100 < AbsRandD100) y0plD100 = 0.5*(x1e2 + z0e2)/fabs(y0);
    double y1plD110 = y1+D110; if(y1plD110 < AbsRandD110) y1plD110 = 0.5*(x1e2 + z0e2)/fabs(y1);
    double y1plD010 = y1+D010; if(y1plD010 < AbsRandD010) y1plD010 = 0.5*(x0e2 + z0e2)/fabs(y1);
    double y0plD000 = y0+D000; if(y0plD000 < AbsRandD000) y0plD000 = 0.5*(x0e2 + z0e2)/fabs(y0);
    double y0plD001 = y0+D001; if(y0plD001 < AbsRandD001) y0plD001 = 0.5*(x0e2 + z1e2)/fabs(y0);
    double y1plD011 = y1+D011; if(y1plD011 < AbsRandD011) y1plD011 = 0.5*(x0e2 + z1e2)/fabs(y1);
    double y1plD111 = y1+D111; if(y1plD111 < AbsRandD111) y1plD111 = 0.5*(x1e2 + z1e2)/fabs(y1);
    double y0plD101 = y0+D101; if(y0plD101 < AbsRandD101) y0plD101 = 0.5*(x1e2 + z1e2)/fabs(y0);

    double x0plD010 = x0+D010; if(x0plD010 < AbsRandD010) x0plD010 = 0.5*(y1e2 + z0e2)/fabs(x0);
    double x1plD110 = x1+D110; if(x1plD110 < AbsRandD110) x1plD110 = 0.5*(y1e2 + z0e2)/fabs(x1);
    double x1plD100 = x1+D100; if(x1plD100 < AbsRandD100) x1plD100 = 0.5*(y0e2 + z0e2)/fabs(x1);
    double x0plD000 = x0+D000; if(x0plD000 < AbsRandD000) x0plD000 = 0.5*(y0e2 + z0e2)/fabs(x0);
    double x0plD001 = x0+D001; if(x0plD001 < AbsRandD001) x0plD001 = 0.5*(y0e2 + z1e2)/fabs(x0);
    double x1plD101 = x1+D101; if(x1plD101 < AbsRandD101) x1plD101 = 0.5*(y0e2 + z1e2)/fabs(x1);
    double x1plD111 = x1+D111; if(x1plD111 < AbsRandD111) x1plD111 = 0.5*(y1e2 + z1e2)/fabs(x1);
    double x0plD011 = x0+D011; if(x0plD011 < AbsRandD011) x0plD011 = 0.5*(y1e2 + z1e2)/fabs(x0);

    double x0plD010_di_x1plD110 = x0plD010/x1plD110;
    double x1plD100_di_x0plD000 = x1plD100/x0plD000;
    double x0plD001_di_x1plD101 = x0plD001/x1plD101;
    double x1plD111_di_x0plD011 = x1plD111/x0plD011;
    double y0plD100_di_y1plD110 = y0plD100/y1plD110;
    double y1plD010_di_y0plD000 = y1plD010/y0plD000;
    double y0plD001_di_y1plD011 = y0plD001/y1plD011;
    double y1plD111_di_y0plD101 = y1plD111/y0plD101;
    double z0plD100_di_z1plD101 = z0plD100/z1plD101;
    double z1plD001_di_z0plD000 = z1plD001/z0plD000;
    double z0plD010_di_z1plD011 = z0plD010/z1plD011;
    double z1plD111_di_z0plD110 = z1plD111/z0plD110;

    // PreRelax has no A_/Phi_ keys -> product-then-log branch (radrec.cpp:281-283)
    double Sx = -log(x0plD010_di_x1plD110*x1plD100_di_x0plD000*x0plD001_di_x1plD101*x1plD111_di_x0plD011);
    double Sy = -log(y0plD100_di_y1plD110*y1plD010_di_y0plD000*y0plD001_di_y1plD011*y1plD111_di_y0plD101);
    double Sz = -log(z0plD100_di_z1plD101*z1plD001_di_z0plD000*z0plD010_di_z1plD011*z1plD111_di_z0plD110);

    const double dConst2 = 1./4./Pi;
    double Tx = dConst2*(T0x + T1x);
    double Ty = dConst2*(T0y + T1y);
    double Tz = dConst2*(T0z + T1z);
    Sx *= dConst2; Sy *= dConst2; Sz *= dConst2;

    Q[0] = Tx;  Q[1] = -Sz; Q[2] = -Sy;
    Q[3] = -Sz; Q[4] = Ty;  Q[5] = -Sx;
    Q[6] = -Sy; Q[7] = -Sx; Q[8] = Tz;
}

// ============================================================
// Device: 3x3 matrix multiply C = A * B (row-major)
// ============================================================
__device__ void matmul3x3(const double* A, const double* B, double* C)
{
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
        {
            double s = 0.0;
            for(int k = 0; k < 3; k++)
                s += A[3*i+k] * B[3*k+j];
            C[3*i+j] = s;
        }
}

// ============================================================
// Device: transpose 3x3
// ============================================================
__device__ void transpose3x3(const double* A, double* AT)
{
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            AT[3*i+j] = A[3*j+i];
}

// ============================================================
// Kernel: Assemble interaction matrix, mixed element types.
// One thread per (obs_elem, src_elem) pair. The Q block depends only on
// the SOURCE element's type (the observer contributes just its center
// point), so a per-source branch covers RecMag<->RecMag, poly<->poly and
// both cross blocks alike.
// ============================================================
__global__ void assemble_mixed_kernel(
    int N,
    const double* __restrict__ obs_centers,    // [N*3] transformed observation centers
    const double* __restrict__ src_centers,     // [N*3] raw element centers
    const int* __restrict__ face_offsets,       // [N+1] (empty range for RecMags)
    const int* __restrict__ edge_offsets,       // [n_faces_total+1]
    const double* __restrict__ face_cz,         // [n_faces_total]
    const double* __restrict__ face_rot,        // [n_faces_total*9] lab->local rotation
    const double* __restrict__ face_orig,       // [n_faces_total*3] face origin in lab
    const double* __restrict__ edge_pts_2d,     // [n_edges_total*2]
    const int* __restrict__ is_rec,             // [N] 1 = RecMag source
    const double* __restrict__ rec_centers,     // [N*3] cuboid centers (own frame)
    const double* __restrict__ rec_dims,        // [N*3] cuboid FULL dimensions
    double rec_abs_rand,                        // radCR tolerance snapshot
    double rec_rel_rand,
    double rec_zero_rand,
    int rec_act_on_doubles,
    const int* __restrict__ sym_counts,
    const int* __restrict__ sym_offsets,
    const double* __restrict__ sym_point_tr,
    const double* __restrict__ sym_field_tr,

    float* __restrict__ out_blocks              // [N*N*9]
)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * N;
    if(tid >= total) return;

    int obs_idx = (int)(tid / N);  // row (StrNo)
    int src_idx = (int)(tid % N);  // column (ColNo)

    const double PI = 3.14159265358979;
    const double ConstForH = 1.0 / (4.0 * PI);

    // Observation point in lab frame
    double obs_lab[3] = {
        obs_centers[3*obs_idx],
        obs_centers[3*obs_idx+1],
        obs_centers[3*obs_idx+2]
    };

    // Accumulate 3x3 block over symmetry copies
    double block[9] = {0,0,0, 0,0,0, 0,0,0};

    int srcIsRec = is_rec[src_idx];
    int fStart = face_offsets[src_idx];
    int fEnd   = face_offsets[src_idx + 1];

    int n_sym_j = sym_counts[src_idx];
    int sym_off = sym_offsets[src_idx];

    for(int sc = 0; sc < n_sym_j; sc++)
    {
        const double* ptMat = &sym_point_tr[(sym_off + sc) * 9];
        const double* ftMat = &sym_field_tr[(sym_off + sc) * 9];

        // Transform observation point: obs_copy = ptMat * obs_lab
        double obs_copy[3] = {
            ptMat[0]*obs_lab[0] + ptMat[1]*obs_lab[1] + ptMat[2]*obs_lab[2],
            ptMat[3]*obs_lab[0] + ptMat[4]*obs_lab[1] + ptMat[5]*obs_lab[2],
            ptMat[6]*obs_lab[0] + ptMat[7]*obs_lab[1] + ptMat[8]*obs_lab[2]
        };

        double sum_block[9] = {0,0,0, 0,0,0, 0,0,0};

        if(srcIsRec)
        {
            // RecMag source: closed-form cuboid Q at the transformed obs point,
            // exactly as the CPU's radTRecMag::B_comp PreRelax branch.
            recmag_prerelax_dev(
                obs_copy[0], obs_copy[1], obs_copy[2],
                rec_centers[3*src_idx], rec_centers[3*src_idx+1], rec_centers[3*src_idx+2],
                rec_dims[3*src_idx], rec_dims[3*src_idx+1], rec_dims[3*src_idx+2],
                rec_abs_rand, rec_rel_rand, rec_zero_rand, rec_act_on_doubles,
                sum_block);

            // Apply symmetry field transform and accumulate (same as poly path below)
            double result[9];
            matmul3x3(ftMat, sum_block, result);
            for(int k = 0; k < 9; k++) block[k] += result[k];
            continue;
        }

        // Polyhedron source:
        // For each unit magnetization direction, compute field from all faces
        // PreRelax_ mode: B_comp returns a matrix Q where column c = field from M = e_c
        // For polygon: Q has only z-column nonzero: Q = [0,0,-Sx*C; 0,0,-Sy*C; 0,0,-Sz*C]
        // Then B_comp_frM applies: TrMatrixLeft_inv(Q); TrMatrix(Q);
        // which gives: rot * Q * rot^T
        // But Q only has z-column, so Q * rot^T has column j = Q_z * rot^T[z][j] = Q_z * rot[j][z]
        // Then rot * (Q * rot^T) gives the full 3x3

        // Sum over faces of source element
        // Each face produces Sx, Sy, Sz in its local frame
        // Then we form Q_face, apply face transform, and sum

        for(int fi = fStart; fi < fEnd; fi++)
        {
            const double* rot = &face_rot[9 * fi];    // lab->local
            const double* orig = &face_orig[3 * fi];
            double cz = face_cz[fi];
            int eStart = edge_offsets[fi];
            int eEnd   = edge_offsets[fi + 1];
            int nEdges = eEnd - eStart;
            if(nEdges < 3) continue;

            // Transform obs_copy to face local frame:
            // local = rot * (obs_copy - orig)
            double dx = obs_copy[0] - orig[0];
            double dy = obs_copy[1] - orig[1];
            double dz = obs_copy[2] - orig[2];

            double local_x = rot[0]*dx + rot[1]*dy + rot[2]*dz;
            double local_y = rot[3]*dx + rot[4]*dy + rot[5]*dz;
            double local_z = rot[6]*dx + rot[7]*dy + rot[8]*dz;

            // Compute polygon Sx, Sy, Sz
            double Sx, Sy_val, Sz;
            polygon_prerelax_dev(
                local_x, local_y, local_z,
                cz,
                &edge_pts_2d[2 * eStart],
                nEdges,
                Sx, Sy_val, Sz);

            // Q in local frame (only z-column nonzero):
            // Q[0][2] = -ConstForH * Sx   (B.z)
            // Q[1][2] = -ConstForH * Sy   (H.z)
            // Q[2][2] = -ConstForH * Sz   (A.z)
            double Qz0 = -ConstForH * Sx;
            double Qz1 = -ConstForH * Sy_val;
            double Qz2 = -ConstForH * Sz;

            // Apply face transform: result = rot^T * Q_local * rot
            // Since Q_local only has z-column: Q_local * rot has:
            //   (Q_local * rot)[i][j] = Q_local[i][2] * rot[2][j]  (row 2 of rot = rot[6..8])
            // Then rot^T * (Q_local * rot):
            //   result[i][j] = sum_m rot[m][i] * Q_local[m][2] * rot[2][j]
            //                = (rot[0][i]*Qz0 + rot[1][i]*Qz1 + rot[2][i]*Qz2) * rot[2][j]

            // rot^T column i = rot row i reversed indexing
            // rot[m][i] = rot[3*m + i]
            double rotT_col_dot_Qz[3];
            for(int i = 0; i < 3; i++)
            {
                rotT_col_dot_Qz[i] = rot[0*3+i]*Qz0 + rot[1*3+i]*Qz1 + rot[2*3+i]*Qz2;
            }

            double rot2[3] = {rot[6], rot[7], rot[8]};  // row 2 of rot

            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 3; j++)
                    sum_block[3*i+j] += rotT_col_dot_Qz[i] * rot2[j];
        }

        // Apply symmetry field transform: result = ftMat * sum_block (left multiply only)
        double result[9];
        matmul3x3(ftMat, sum_block, result);

        for(int k = 0; k < 9; k++)
            block[k] += result[k];
    }

    // Store result
    long long outIdx = tid * 9;
    for(int k = 0; k < 9; k++)
        out_blocks[outIdx + k] = (float)block[k];
}

// ============================================================
// Host: Launch assembly
// ============================================================

// On CUDA error: report which call/line failed, set rc = -1, and jump to the
// function's `cleanup` label (all device pointers are nullptr-initialized, so the
// cudaFree() calls there are safe). CU_MALLOC additionally prints the requested
// size, so an out-of-VRAM interaction matrix reports a clear message instead of
// surfacing later as a cryptic "unspecified launch failure".
#define CU_TRY(call) do { cudaError_t _e = (call); if(_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n  at: %s (%s:%d)\n", \
            cudaGetErrorString(_e), #call, __FILE__, __LINE__); \
    rc = -1; goto cleanup; } } while(0)

#define CU_MALLOC(ptr, bytes) do { size_t _b = (bytes); \
    cudaError_t _e = cudaMalloc((void**)&(ptr), _b); \
    if(_e != cudaSuccess) { \
        fprintf(stderr, "CUDA malloc failed for %s: %.2f GB requested - %s (%s:%d)\n", \
                #ptr, (double)_b / 1073741824.0, cudaGetErrorString(_e), __FILE__, __LINE__); \
        rc = -1; goto cleanup; } } while(0)

int radGPU_AssembleMatrix(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_AsmResult* result)
{
    if(!polyData || polyData->n_elem <= 0 || !recData) return -1;
    int N = polyData->n_elem;  // total elements (poly + RecMag; see packing)

    long long totalPairs = (long long)N * N;

    // Pre-flight VRAM check (issue #13): the dense interaction matrix on the device
    // (d_out = totalPairs*9 floats = 36*N^2 bytes) dominates GPU memory. If it won't fit in
    // free VRAM (with a margin for the smaller geometry/symmetry buffers), skip GPU assembly
    // up front so the caller (radTInteraction::SetupInteractMatrix) falls back to CPU and
    // warns -- rather than allocating a 36*N^2 host buffer and then failing a huge cudaMalloc.
    {
        size_t freeB = 0, totalB = 0;
        if(cudaMemGetInfo(&freeB, &totalB) == cudaSuccess)
        {
            double needBytes = (double)totalPairs * 9.0 * sizeof(float) * 1.15; // +15% geom/sym
            if(needBytes > (double)freeB * 0.90)
            {
                fprintf(stderr, "GPU assembly: interaction matrix needs ~%.2f GB but only "
                                "%.2f GB VRAM free (N=%d); using CPU.\n",
                        needBytes / 1073741824.0, (double)freeB / 1073741824.0, N);
                return -1;
            }
        }
    }

    // Allocate output (host). This dense matrix is N*N*9 floats; use nothrow so a
    // too-large request reports cleanly instead of throwing std::bad_alloc.
    result->N = N;
    result->matrix_blocks = new (std::nothrow) float[totalPairs * 9];
    if(!result->matrix_blocks) {
        fprintf(stderr, "Host allocation failed for interaction matrix: %.2f GB requested "
                        "(N=%d) - not enough RAM.\n",
                (double)(totalPairs * 9 * sizeof(float)) / 1073741824.0, N);
        return -1;
    }

    int rc = 0;

    {
        // Upload geometry to GPU. All device pointers are nullptr-initialized so the
        // `cleanup` label can cudaFree() them unconditionally on any error path.
        double *d_obs=nullptr, *d_centers=nullptr, *d_face_cz=nullptr, *d_face_rot=nullptr,
               *d_face_orig=nullptr, *d_edge_pts=nullptr;
        int *d_face_offsets=nullptr, *d_edge_offsets=nullptr;
        int *d_is_rec=nullptr;
        double *d_rec_centers=nullptr, *d_rec_dims=nullptr;
        float *d_out=nullptr;
        int *d_sym_counts=nullptr, *d_sym_offsets=nullptr;
        double *d_sym_pt=nullptr, *d_sym_ft=nullptr;

        int nFaces = polyData->n_faces_total;   // 0 for a pure-RecMag model
        int nEdges = polyData->n_edges_total;
        int totalCopies = symData->total_copies;

        // Declared before any CU_MALLOC so the goto to `cleanup` never bypasses them.
        int blockSize = 64;
        long long gridSize = (totalPairs + blockSize - 1) / blockSize;

        // size_t casts keep the byte counts 64-bit (d_out is N*N*9 floats).
        // Face/edge counts can be 0 (pure-RecMag model): pad to 1 element so
        // cudaMalloc never sees a zero-byte request (a 0-byte cudaMalloc returns
        // a nullptr that CU_TRY-guarded cudaMemcpy would then reject).
        size_t nFacesAlloc = (nFaces > 0)? (size_t)nFaces : 1;
        size_t nEdgesAlloc = (nEdges > 0)? (size_t)nEdges : 1;

        CU_MALLOC(d_obs,          3*(size_t)N*sizeof(double));
        CU_MALLOC(d_centers,      3*(size_t)N*sizeof(double));
        CU_MALLOC(d_face_offsets, ((size_t)N+1)*sizeof(int));
        CU_MALLOC(d_edge_offsets, (nFacesAlloc+1)*sizeof(int));
        CU_MALLOC(d_face_cz,      nFacesAlloc*sizeof(double));
        CU_MALLOC(d_face_rot,     9*nFacesAlloc*sizeof(double));
        CU_MALLOC(d_face_orig,    3*nFacesAlloc*sizeof(double));
        CU_MALLOC(d_edge_pts,     2*nEdgesAlloc*sizeof(double));
        CU_MALLOC(d_is_rec,       (size_t)N*sizeof(int));
        CU_MALLOC(d_rec_centers,  3*(size_t)N*sizeof(double));
        CU_MALLOC(d_rec_dims,     3*(size_t)N*sizeof(double));
        CU_MALLOC(d_out,          totalPairs*9*sizeof(float));
        CU_MALLOC(d_sym_counts,   (size_t)N*sizeof(int));
        CU_MALLOC(d_sym_offsets,  ((size_t)N+1)*sizeof(int));
        CU_MALLOC(d_sym_pt,       (size_t)totalCopies*9*sizeof(double));
        CU_MALLOC(d_sym_ft,       (size_t)totalCopies*9*sizeof(double));

        CU_TRY(cudaMemcpy(d_obs,          polyData->centers,          3*(size_t)N*sizeof(double),           cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_centers,      polyData->centers,          3*(size_t)N*sizeof(double),           cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_face_offsets, polyData->face_offsets,     ((size_t)N+1)*sizeof(int),            cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_edge_offsets, polyData->edge_offsets,     ((size_t)nFaces+1)*sizeof(int),       cudaMemcpyHostToDevice));
        if(nFaces > 0) {
            CU_TRY(cudaMemcpy(d_face_cz,      polyData->face_cz,      (size_t)nFaces*sizeof(double),        cudaMemcpyHostToDevice));
            CU_TRY(cudaMemcpy(d_face_rot,     polyData->face_rot,     9*(size_t)nFaces*sizeof(double),      cudaMemcpyHostToDevice));
            CU_TRY(cudaMemcpy(d_face_orig,    polyData->face_orig,    3*(size_t)nFaces*sizeof(double),      cudaMemcpyHostToDevice));
        }
        if(nEdges > 0) {
            CU_TRY(cudaMemcpy(d_edge_pts,     polyData->edge_pts_2d,  2*(size_t)nEdges*sizeof(double),      cudaMemcpyHostToDevice));
        }
        CU_TRY(cudaMemcpy(d_is_rec,       recData->is_rec,            (size_t)N*sizeof(int),                cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_rec_centers,  recData->centers,           3*(size_t)N*sizeof(double),           cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_rec_dims,     recData->dims,              3*(size_t)N*sizeof(double),           cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_sym_counts,   symData->sym_counts,        (size_t)N*sizeof(int),                cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_sym_offsets,  symData->sym_offsets,       ((size_t)N+1)*sizeof(int),            cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_sym_pt,       symData->point_transforms,  (size_t)totalCopies*9*sizeof(double), cudaMemcpyHostToDevice));
        CU_TRY(cudaMemcpy(d_sym_ft,       symData->field_transforms,  (size_t)totalCopies*9*sizeof(double), cudaMemcpyHostToDevice));

        assemble_mixed_kernel<<<(unsigned int)gridSize, blockSize>>>(
            N, d_obs, d_centers,
            d_face_offsets, d_edge_offsets,
            d_face_cz, d_face_rot, d_face_orig, d_edge_pts,
            d_is_rec, d_rec_centers, d_rec_dims,
            recData->abs_rand, recData->rel_rand, recData->zero_rand,
            recData->act_on_doubles,
            d_sym_counts, d_sym_offsets, d_sym_pt, d_sym_ft,
            d_out);

        CU_TRY(cudaGetLastError());        // launch-configuration errors (grid/block, args)
        CU_TRY(cudaDeviceSynchronize());   // in-kernel errors (illegal access, TDR timeout, ...)

        CU_TRY(cudaMemcpy(result->matrix_blocks, d_out, totalPairs*9*sizeof(float), cudaMemcpyDeviceToHost));

    cleanup:
        cudaFree(d_obs); cudaFree(d_centers);
        cudaFree(d_face_offsets); cudaFree(d_edge_offsets);
        cudaFree(d_face_cz); cudaFree(d_face_rot); cudaFree(d_face_orig);
        cudaFree(d_edge_pts); cudaFree(d_sym_pt); cudaFree(d_sym_ft);
        cudaFree(d_is_rec); cudaFree(d_rec_centers); cudaFree(d_rec_dims);
        cudaFree(d_out);
        cudaFree(d_sym_counts); cudaFree(d_sym_offsets);
        if(rc != 0 && result->matrix_blocks) {
            delete[] result->matrix_blocks;
            result->matrix_blocks = nullptr;
        }
    }

    return rc;
}

// ============================================================
// Host: Free assembly data
// ============================================================
void radGPU_FreeAsmData(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_AsmResult* result)
{
    if(polyData) {
        delete[] polyData->face_offsets;  polyData->face_offsets = nullptr;
        delete[] polyData->edge_offsets;  polyData->edge_offsets = nullptr;
        delete[] polyData->face_cz;      polyData->face_cz = nullptr;
        delete[] polyData->face_rot;     polyData->face_rot = nullptr;
        delete[] polyData->face_orig;    polyData->face_orig = nullptr;
        delete[] polyData->edge_pts_2d;  polyData->edge_pts_2d = nullptr;
        delete[] polyData->centers;      polyData->centers = nullptr;
    }
    if(recData) {
        delete[] recData->is_rec;        recData->is_rec = nullptr;
        delete[] recData->centers;       recData->centers = nullptr;
        delete[] recData->dims;          recData->dims = nullptr;
    }
    if(result) {
        delete[] result->matrix_blocks;  result->matrix_blocks = nullptr;
    }
}

#endif // RADIA_WITH_CUDA