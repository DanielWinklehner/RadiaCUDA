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
    gpuData.omega = omega;  // pass through; negative means "use default"

    if(!radGPU_PackInteractionData(IntrctPtr, &gpuData)) {
        fprintf(stderr, "radGPU_AutoRelax: failed to pack interaction data\n");
        radGPU_FreeData(&gpuData);
        return -1;
    }

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

#endif