/*-------------------------------------------------------------------------
*
* File name:      radgpurlx.cpp
*
* Project:        RADIA
*
* Description:    GPU relaxation - data packing/unpacking (plain C++)
*
-------------------------------------------------------------------------*/

#ifdef RADIA_WITH_CUDA

#include "radgpurlx.h"
#include "radintrc.h"
#include "radmater.h"
#include "radg3d.h"
#include <cstring>
#include <cstdio>
#include <cmath>

// ============================================================
// Pack interaction data into GPU-friendly flat arrays
// ============================================================
int radGPU_PackInteractionData(radTInteraction* intrct, RadGPURelaxData* data)
{
    int N = intrct->AmOfMainElem;
    int N3 = 3 * N;
    memset(data, 0, sizeof(RadGPURelaxData));
    data->numElem = N;
    data->matrixDim = N3;

    // --- Flatten interaction matrix: TMatrix3df[N][N] -> float[N3 x N3] row-major ---
    data->h_matrix = new float[(long long)N3 * N3];
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            TMatrix3df& blk = intrct->InteractMatrix[i][j];
            int r0 = 3 * i, c0 = 3 * j;
            data->h_matrix[(long long)(r0+0)*N3 + c0+0] = blk.Str0.x;
            data->h_matrix[(long long)(r0+0)*N3 + c0+1] = blk.Str0.y;
            data->h_matrix[(long long)(r0+0)*N3 + c0+2] = blk.Str0.z;
            data->h_matrix[(long long)(r0+1)*N3 + c0+0] = blk.Str1.x;
            data->h_matrix[(long long)(r0+1)*N3 + c0+1] = blk.Str1.y;
            data->h_matrix[(long long)(r0+1)*N3 + c0+2] = blk.Str1.z;
            data->h_matrix[(long long)(r0+2)*N3 + c0+0] = blk.Str2.x;
            data->h_matrix[(long long)(r0+2)*N3 + c0+1] = blk.Str2.y;
            data->h_matrix[(long long)(r0+2)*N3 + c0+2] = blk.Str2.z;
        }
    }

    // --- External field ---
    data->h_extField = new double[N3];
    for(int i = 0; i < N; i++) {
        TVector3d& ef = intrct->ExternFieldArray[i];
        data->h_extField[3*i+0] = ef.x;
        data->h_extField[3*i+1] = ef.y;
        data->h_extField[3*i+2] = ef.z;
    }

    // --- Current magnetization ---
    data->h_magn = new double[N3];
    for(int i = 0; i < N; i++) {
        TVector3d m = intrct->g3dRelaxPtrVect[i]->Magn;
        data->h_magn[3*i+0] = m.x;
        data->h_magn[3*i+1] = m.y;
        data->h_magn[3*i+2] = m.z;
    }

// --- Initialize H field from current external field + matrix*M ---
    data->h_field = new double[N3];
    for(int i = 0; i < N; i++) {
        double hx = intrct->ExternFieldArray[i].x;
        double hy = intrct->ExternFieldArray[i].y;
        double hz = intrct->ExternFieldArray[i].z;
        for(int j = 0; j < N; j++) {
            TMatrix3df& blk = intrct->InteractMatrix[i][j];
            TVector3d mj = intrct->g3dRelaxPtrVect[j]->Magn;
            hx += (double)blk.Str0.x*mj.x + (double)blk.Str0.y*mj.y + (double)blk.Str0.z*mj.z;
            hy += (double)blk.Str1.x*mj.x + (double)blk.Str1.y*mj.y + (double)blk.Str1.z*mj.z;
            hz += (double)blk.Str2.x*mj.x + (double)blk.Str2.y*mj.y + (double)blk.Str2.z*mj.z;
        }
        data->h_field[3*i+0] = hx;
        data->h_field[3*i+1] = hy;
        data->h_field[3*i+2] = hz;
    }

    // --- Allocate per-element material arrays ---
    data->h_matType = new int[N];
    data->h_linKsi = new double[N];
    data->h_remMagn = new double[N3];
    data->h_mhOffset = new int[N];
    data->h_mhLen = new int[N];
    data->h_formulaMs = new double[3 * N];
    data->h_formulaKs = new double[3 * N];
    data->h_formulaLen = new int[N];

    memset(data->h_linKsi, 0, N * sizeof(double));
    memset(data->h_formulaMs, 0, 3 * N * sizeof(double));
    memset(data->h_formulaKs, 0, 3 * N * sizeof(double));
    memset(data->h_formulaLen, 0, N * sizeof(int));

    // --- First pass: identify material types, count M-H curve points ---
    int totalPts = 0;
    for(int i = 0; i < N; i++) {
        data->h_mhOffset[i] = 0;
        data->h_mhLen[i] = 0;

        radTg3dRelax* relax = intrct->g3dRelaxPtrVect[i];
        radTMaterial* mat = (radTMaterial*)(relax->MaterHandle.rep);

        // Remanent magnetization
        data->h_remMagn[3*i+0] = mat->RemMagn.x;
        data->h_remMagn[3*i+1] = mat->RemMagn.y;
        data->h_remMagn[3*i+2] = mat->RemMagn.z;

        int mtype = mat->Type_Material();
        // Type_Material: 1=linear_aniso, 2=linear_iso, 3=nonlinear_iso

        if(mtype == 2) {
           // Linear isotropic
            data->h_matType[i] = 0;
            // Extract Ksi by evaluating M(H) at a known H and dividing
            // For linear isotropic: M(H) = Ksi*H + RemMagn
            TVector3d testH(1.0, 0.0, 0.0);
            TVector3d testM = mat->M(testH);
            data->h_linKsi[i] = testM.x - mat->RemMagn.x; // Ksi * 1.0
        }
        else if(mtype == 3) {
            // Nonlinear isotropic — need to check if tabulated or formula
            radTNonlinearIsotropMaterial* nlMat = dynamic_cast<radTNonlinearIsotropMaterial*>(mat);
            if(nlMat == nullptr) {
                // Shouldn't happen, but fall back
                data->h_matType[i] = 0;
                data->h_linKsi[i] = 0;
                continue;
            }

            // Access private members via the GPU extraction methods we'll add
            int curvePts = 0;
            int formulaLen = 0;
            nlMat->GetGPUData_Counts(&curvePts, &formulaLen);

            if(curvePts > 0) {
                data->h_matType[i] = 1; // tabulated
                data->h_mhOffset[i] = totalPts;
                data->h_mhLen[i] = curvePts;
                totalPts += curvePts;
            }
            else if(formulaLen > 0) {
                data->h_matType[i] = 3; // formula
                double ms[3] = {0,0,0}, ks[3] = {0,0,0};
                nlMat->GetGPUData_Formula(ms, ks, &formulaLen);
                for(int k = 0; k < 3; k++) {
                    data->h_formulaMs[3*i+k] = ms[k];
                    data->h_formulaKs[3*i+k] = ks[k];
                }
                data->h_formulaLen[i] = formulaLen;
            }
            else {
                data->h_matType[i] = 0;
                data->h_linKsi[i] = 0;
            }
        }
        else if(mtype == 1) {
            // Linear anisotropic — approximate as linear isotropic with avg Ksi
            // (Full anisotropic support is future work)
            data->h_matType[i] = 0;
            TVector3d testH(1.0, 0.0, 0.0);
            TVector3d testM = mat->M(testH);
            data->h_linKsi[i] = testM.x - mat->RemMagn.x;
        }
        else {
            data->h_matType[i] = 0;
            data->h_linKsi[i] = 0;
        }
    }

    // --- Allocate and fill M-H curve data ---
    data->totalMHPoints = totalPts;
    if(totalPts > 0) {
        data->h_mhH = new double[totalPts];
        data->h_mhM = new double[totalPts];
        data->h_mhdMdH = new double[totalPts];
    } else {
        data->h_mhH = nullptr;
        data->h_mhM = nullptr;
        data->h_mhdMdH = nullptr;
    }

    // Second pass: copy curve data
    for(int i = 0; i < N; i++) {
        if(data->h_matType[i] != 1) continue; // only tabulated

        radTg3dRelax* relax = intrct->g3dRelaxPtrVect[i];
        radTMaterial* mat = (radTMaterial*)(relax->MaterHandle.rep);
        radTNonlinearIsotropMaterial* nlMat = dynamic_cast<radTNonlinearIsotropMaterial*>(mat);
        if(nlMat == nullptr) continue;

        int off = data->h_mhOffset[i];
        int len = data->h_mhLen[i];
        nlMat->GetGPUData_Curve(data->h_mhH + off, data->h_mhM + off, data->h_mhdMdH + off, len);
    }

    // --- Self-interaction diagonal blocks ---
    data->h_selfBlocks = new float[9 * N];
    for(int i = 0; i < N; i++) {
        TMatrix3df& blk = intrct->InteractMatrix[i][i];
        float* sb = data->h_selfBlocks + 9 * i;
        sb[0] = blk.Str0.x; sb[1] = blk.Str0.y; sb[2] = blk.Str0.z;
        sb[3] = blk.Str1.x; sb[4] = blk.Str1.y; sb[5] = blk.Str1.z;
        sb[6] = blk.Str2.x; sb[7] = blk.Str2.y; sb[8] = blk.Str2.z;
    }

    return 1;
}

// ============================================================
// Unpack magnetization results back into Radia structures
// ============================================================
void radGPU_UnpackMagnetization(RadGPURelaxData* data, radTInteraction* intrct)
{
    int N = data->numElem;
    for(int i = 0; i < N; i++) {
        TVector3d m;
        m.x = data->h_magn[3*i+0];
        m.y = data->h_magn[3*i+1];
        m.z = data->h_magn[3*i+2];

        intrct->g3dRelaxPtrVect[i]->Magn = m;
        intrct->NewMagnArray[i] = m;

        TVector3d h;
        h.x = data->h_field[3*i+0];
        h.y = data->h_field[3*i+1];
        h.z = data->h_field[3*i+2];
        intrct->NewFieldArray[i] = h;
    }
}

// ============================================================
// Free host-side data
// ============================================================
void radGPU_FreeData(RadGPURelaxData* data)
{
    if(!data) return;
    delete[] data->h_matrix;       data->h_matrix = nullptr;
    delete[] data->h_magn;         data->h_magn = nullptr;
    delete[] data->h_field;        data->h_field = nullptr;
    delete[] data->h_extField;     data->h_extField = nullptr;
    delete[] data->h_matType;      data->h_matType = nullptr;
    delete[] data->h_linKsi;       data->h_linKsi = nullptr;
    delete[] data->h_remMagn;      data->h_remMagn = nullptr;
    delete[] data->h_mhH;          data->h_mhH = nullptr;
    delete[] data->h_mhM;          data->h_mhM = nullptr;
    delete[] data->h_mhdMdH;       data->h_mhdMdH = nullptr;
    delete[] data->h_mhOffset;     data->h_mhOffset = nullptr;
    delete[] data->h_mhLen;        data->h_mhLen = nullptr;
    delete[] data->h_formulaMs;    data->h_formulaMs = nullptr;
    delete[] data->h_formulaKs;    data->h_formulaKs = nullptr;
    delete[] data->h_formulaLen;   data->h_formulaLen = nullptr;
    delete[] data->h_selfBlocks;   data->h_selfBlocks = nullptr;
}

#endif // RADIA_WITH_CUDA