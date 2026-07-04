#ifdef RADIA_WITH_CUDA

#include "radrlmet.h"
#include "radintrc.h"
#include "radgpurlx.h"
#include <cstdio>
#include <cmath>

int radGPU_AutoRelax(radTInteraction* IntrctPtr, double PrecOnMagnetiz, int MaxIterNumber, char MagnResetIsNotNeeded, double omega)
{
    if(IntrctPtr == nullptr || IntrctPtr->AmOfMainElem <= 0) return 0;

    if(!MagnResetIsNotNeeded)
    {
        IntrctPtr->ResetM();
    }
    IntrctPtr->ResetAuxParam();

    RadGPURelaxData gpuData;

    // Skip the O(N^2) matrix flatten + device upload when the GPU cache
    // already holds this interaction's matrix (repeated RlxAuto calls on
    // the same IM, e.g. warm restarts / staged perturbative solves).
    int skipMatrix = radGPU_MatrixCached(IntrctPtr->mGpuMatrixStamp,
                                         3 * IntrctPtr->AmOfMainElem);

    if(!radGPU_PackInteractionData(IntrctPtr, &gpuData, skipMatrix)) {
        fprintf(stderr, "radGPU_AutoRelax: failed to pack interaction data\n");
        radGPU_FreeData(&gpuData);
        return -1;
    }
    // NOTE: set omega AFTER packing -- Pack memsets the struct (setting it
    // before, as the original code did, silently discarded the omega option).
    gpuData.omega = omega;  // pass through; negative means "use default"

    double misfitM = 0, maxModM = 0, maxModH = 0;
    int iterDone = radGPU_RelaxAuto(&gpuData, PrecOnMagnetiz, MaxIterNumber,
                                     &misfitM, &maxModM, &maxModH);

    if(iterDone < 0) {
        fprintf(stderr, "radGPU_AutoRelax: GPU solver failed, falling back to CPU\n");
        radGPU_FreeData(&gpuData);
        return -1;
    }

    radGPU_UnpackMagnetization(&gpuData, IntrctPtr);

    IntrctPtr->RelaxStatusParam.MisfitM = misfitM;
    IntrctPtr->RelaxStatusParam.MaxModM = maxModM;
    IntrctPtr->RelaxStatusParam.MaxModH = maxModH;

    radGPU_FreeData(&gpuData);
    return iterDone;
}

// Method 11: Newton-Krylov. Same pack/unpack path as method 9; the omega
// option (if given) sets the damping of the pre-/fallback-smoothing passes.
int radGPU_AutoRelaxNK(radTInteraction* IntrctPtr, double PrecOnMagnetiz, int MaxIterNumber, char MagnResetIsNotNeeded, double omega)
{
    if(IntrctPtr == nullptr || IntrctPtr->AmOfMainElem <= 0) return 0;

    if(!MagnResetIsNotNeeded)
    {
        IntrctPtr->ResetM();
    }
    IntrctPtr->ResetAuxParam();

    RadGPURelaxData gpuData;

    int skipMatrix = radGPU_MatrixCached(IntrctPtr->mGpuMatrixStamp,
                                         3 * IntrctPtr->AmOfMainElem);

    if(!radGPU_PackInteractionData(IntrctPtr, &gpuData, skipMatrix)) {
        fprintf(stderr, "radGPU_AutoRelaxNK: failed to pack interaction data\n");
        radGPU_FreeData(&gpuData);
        return -1;
    }
    gpuData.omega = omega;  // smoothing-pass damping; negative = default

    double misfitM = 0, maxModM = 0, maxModH = 0;
    int iterDone = radGPU_RelaxNK(&gpuData, PrecOnMagnetiz, MaxIterNumber,
                                  &misfitM, &maxModM, &maxModH);

    if(iterDone < 0) {
        fprintf(stderr, "radGPU_AutoRelaxNK: GPU solver failed, falling back to CPU\n");
        radGPU_FreeData(&gpuData);
        return -1;
    }

    radGPU_UnpackMagnetization(&gpuData, IntrctPtr);

    IntrctPtr->RelaxStatusParam.MisfitM = misfitM;
    IntrctPtr->RelaxStatusParam.MaxModM = maxModM;
    IntrctPtr->RelaxStatusParam.MaxModH = maxModH;

    radGPU_FreeData(&gpuData);
    return iterDone;
}

#endif