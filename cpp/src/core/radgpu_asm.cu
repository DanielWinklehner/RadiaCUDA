/*-------------------------------------------------------------------------
*
* File name:      radgpu_asm.cu
*
* Project:        RADIA
*
* Description:    GPU-accelerated interaction matrix assembly
*                 Polyhedron field via flat polygon face integrals
*
-------------------------------------------------------------------------*/

#ifdef RADIA_WITH_CUDA

#include "radgpu_asm.h"
#include <cstdio>
#include <cstring>
#include <cmath>
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
// Kernel: Assemble interaction matrix for polyhedra
// One thread per (obs_elem, src_elem) pair
// ============================================================
__global__ void assemble_poly_kernel(
    int N,
    const double* __restrict__ obs_centers,    // [N*3] transformed observation centers
    const double* __restrict__ src_centers,     // [N*3] raw element centers
    const int* __restrict__ face_offsets,       // [N+1]
    const int* __restrict__ edge_offsets,       // [n_faces_total+1]
    const double* __restrict__ face_cz,         // [n_faces_total]
    const double* __restrict__ face_rot,        // [n_faces_total*9] lab->local rotation
    const double* __restrict__ face_orig,       // [n_faces_total*3] face origin in lab
    const double* __restrict__ edge_pts_2d,     // [n_edges_total*2]
    int n_sym,
    const double* __restrict__ sym_point_tr,    // [n_sym*9]
    const double* __restrict__ sym_field_tr,    // [n_sym*9]
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

    int fStart = face_offsets[src_idx];
    int fEnd   = face_offsets[src_idx + 1];

    for(int sc = 0; sc < n_sym; sc++)
    {
        const double* ptMat = &sym_point_tr[sc * 9];  // maps obs into this copy's frame
        const double* ftMat = &sym_field_tr[sc * 9];   // maps field back to lab

        // Transform observation point: obs_copy = ptMat * obs_lab
        double obs_copy[3] = {
            ptMat[0]*obs_lab[0] + ptMat[1]*obs_lab[1] + ptMat[2]*obs_lab[2],
            ptMat[3]*obs_lab[0] + ptMat[4]*obs_lab[1] + ptMat[5]*obs_lab[2],
            ptMat[6]*obs_lab[0] + ptMat[7]*obs_lab[1] + ptMat[8]*obs_lab[2]
        };

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

        double sum_block[9] = {0,0,0, 0,0,0, 0,0,0};

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
// Host: RecMag kernel (placeholder - not yet implemented)
// ============================================================
__global__ void assemble_recmag_kernel(
    int N,
    const double* __restrict__ centers,
    const double* __restrict__ dims,
    const double* __restrict__ obs_centers,
    int n_sym,
    const double* __restrict__ sym_point_tr,
    const double* __restrict__ sym_field_tr,
    float* __restrict__ out_blocks)
{
    // TODO: implement RecMag kernel
}

// ============================================================
// Host: Launch assembly
// ============================================================
int radGPU_AssembleMatrix(
    RadGPU_PolyData* polyData,
    RadGPU_RecMagData* recData,
    RadGPU_SymData* symData,
    RadGPU_AsmResult* result)
{
    int N = 0;
    bool usePoly = false;

    if(polyData && polyData->n_elem > 0) { N = polyData->n_elem; usePoly = true; }
    else if(recData && recData->n_elem > 0) { N = recData->n_elem; }
    else return -1;

    long long totalPairs = (long long)N * N;

    // Allocate output
    result->N = N;
    result->matrix_blocks = new float[totalPairs * 9];
    if(!result->matrix_blocks) return -1;

    if(usePoly)
    {
        // Upload poly data to GPU
        double *d_obs, *d_centers, *d_face_cz, *d_face_rot, *d_face_orig, *d_edge_pts;
        int *d_face_offsets, *d_edge_offsets;
        double *d_sym_pt, *d_sym_ft;
        float *d_out;

        int nFaces = polyData->n_faces_total;
        int nEdges = polyData->n_edges_total;
        int nSym = symData->n_copies;

        cudaMalloc(&d_obs,          3*N*sizeof(double));
        cudaMalloc(&d_centers,      3*N*sizeof(double));
        cudaMalloc(&d_face_offsets, (N+1)*sizeof(int));
        cudaMalloc(&d_edge_offsets, (nFaces+1)*sizeof(int));
        cudaMalloc(&d_face_cz,     nFaces*sizeof(double));
        cudaMalloc(&d_face_rot,    9*nFaces*sizeof(double));
        cudaMalloc(&d_face_orig,   3*nFaces*sizeof(double));
        cudaMalloc(&d_edge_pts,    2*nEdges*sizeof(double));
        cudaMalloc(&d_sym_pt,      nSym*9*sizeof(double));
        cudaMalloc(&d_sym_ft,      nSym*9*sizeof(double));
        cudaMalloc(&d_out,         totalPairs*9*sizeof(float));

        cudaMemcpy(d_obs,          polyData->centers,      3*N*sizeof(double),         cudaMemcpyHostToDevice);
        cudaMemcpy(d_centers,      polyData->centers,      3*N*sizeof(double),         cudaMemcpyHostToDevice);
        cudaMemcpy(d_face_offsets, polyData->face_offsets,  (N+1)*sizeof(int),          cudaMemcpyHostToDevice);
        cudaMemcpy(d_edge_offsets, polyData->edge_offsets,  (nFaces+1)*sizeof(int),     cudaMemcpyHostToDevice);
        cudaMemcpy(d_face_cz,     polyData->face_cz,       nFaces*sizeof(double),      cudaMemcpyHostToDevice);
        cudaMemcpy(d_face_rot,    polyData->face_rot,      9*nFaces*sizeof(double),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_face_orig,   polyData->face_orig,     3*nFaces*sizeof(double),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_edge_pts,    polyData->edge_pts_2d,   2*nEdges*sizeof(double),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_sym_pt,      symData->point_transforms, nSym*9*sizeof(double),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_sym_ft,      symData->field_transforms, nSym*9*sizeof(double),    cudaMemcpyHostToDevice);

        int blockSize = 64;
        int gridSize = (int)((totalPairs + blockSize - 1) / blockSize);

        assemble_poly_kernel<<<gridSize, blockSize>>>(
            N, d_obs, d_centers,
            d_face_offsets, d_edge_offsets,
            d_face_cz, d_face_rot, d_face_orig, d_edge_pts,
            nSym, d_sym_pt, d_sym_ft,
            d_out);

        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
            cudaFree(d_obs); cudaFree(d_centers);
            cudaFree(d_face_offsets); cudaFree(d_edge_offsets);
            cudaFree(d_face_cz); cudaFree(d_face_rot); cudaFree(d_face_orig);
            cudaFree(d_edge_pts); cudaFree(d_sym_pt); cudaFree(d_sym_ft);
            cudaFree(d_out);
            delete[] result->matrix_blocks;
            result->matrix_blocks = nullptr;
            return -1;
        }

        cudaMemcpy(result->matrix_blocks, d_out, totalPairs*9*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_obs); cudaFree(d_centers);
        cudaFree(d_face_offsets); cudaFree(d_edge_offsets);
        cudaFree(d_face_cz); cudaFree(d_face_rot); cudaFree(d_face_orig);
        cudaFree(d_edge_pts); cudaFree(d_sym_pt); cudaFree(d_sym_ft);
        cudaFree(d_out);
    }
    else
    {
        // RecMag path - not yet implemented
        fprintf(stderr, "RecMag GPU assembly not yet implemented\n");
        delete[] result->matrix_blocks;
        result->matrix_blocks = nullptr;
        return -1;
    }

    return 0;
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
        delete[] recData->centers;       recData->centers = nullptr;
        delete[] recData->dims;          recData->dims = nullptr;
        delete[] recData->obs_centers;   recData->obs_centers = nullptr;
    }
    if(result) {
        delete[] result->matrix_blocks;  result->matrix_blocks = nullptr;
    }
}

#endif // RADIA_WITH_CUDA