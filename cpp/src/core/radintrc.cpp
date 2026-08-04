/*-------------------------------------------------------------------------
*
* File name:      radintrc.cpp
*
* Project:        RADIA
*
* Description:    Magnetic interaction between "relaxable" field source objects
*
* Author(s):      Oleg Chubar
*
* First release:  1997
* 
* Copyright (C):  1997 by European Synchrotron Radiation Facility, France
*                 All Rights Reserved
*
-------------------------------------------------------------------------*/

#include "radintrc.h"
#include "radsbdrc.h"
#include "radgalerkin.h"

#include <cmath>    // sqrt (Galerkin near-band test); std::isfinite below
#include <cstdio>   // fprintf/stderr (Galerkin + GPU fallback diagnostics)

#ifdef RADIA_WITH_CUDA
#include "radgpu_asm.h"
#include "radgpu_fld.h"
#endif

#ifdef _WITH_MPI
#include <mpi.h>
#endif

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

char radTInteraction::gUseGpuAsm = 1; //RadiaCUDA: GPU IM assembly on by default
char radTInteraction::gLastAsmBackend = -1; //RadiaCUDA: -1 = no assembly yet, 0 = CPU, 1 = GPU
char radTInteraction::gGpuFallback = 0; //RadiaCUDA: 0 = 'cpu' (legacy), 1 = 'gpu_streaming', 2 = 'break'

//RadiaCUDA: C-linkage bridge so the entry layer (radentry.cpp) can set the
//GPU-assembly flag without pulling in this header's include chain.
extern "C" void RadSetGpuAsmEnabled(int on) { radTInteraction::gUseGpuAsm = on ? 1 : 0; }
extern "C" int RadGetLastAsmBackend() { return (int)radTInteraction::gLastAsmBackend; }
extern "C" void RadSetGpuFallbackMode(int mode) { radTInteraction::gGpuFallback = (char)mode; }
extern "C" int RadGetGpuFallbackMode() { return (int)radTInteraction::gGpuFallback; }

//-------------------------------------------------------------------------

radTInteraction::radTInteraction(const radThg& In_hg, const radThg& In_hgMoreExtSrc, const radTCompCriterium& InCompCriterium, short InMemAllocTotAtOnce, char ExtraExternFieldArrayIsNeeded, char KeepTransData, int rankMPI, int nProcMPI) //OC08012020
//radTInteraction::radTInteraction(const radThg& In_hg, const radThg& In_hgMoreExtSrc, const radTCompCriterium& InCompCriterium, short InMemAllocTotAtOnce, char ExtraExternFieldArrayIsNeeded, char KeepTransData)
{
	if(!Setup(In_hg, In_hgMoreExtSrc, InCompCriterium, InMemAllocTotAtOnce, ExtraExternFieldArrayIsNeeded, KeepTransData, rankMPI, nProcMPI)) //OC08012020
	//if(!Setup(In_hg, In_hgMoreExtSrc, InCompCriterium, InMemAllocTotAtOnce, ExtraExternFieldArrayIsNeeded, KeepTransData)) 
	{
		SomethingIsWrong = 1;
		Send.ErrorMessage("Radia::Error118");
		throw 0;
	}
}

//-------------------------------------------------------------------------

radTInteraction::radTInteraction()
{
	AmOfMainElem = 0;
	AmOfExtElem = 0;
	InteractMatrix = NULL;
	ExternFieldArray = NULL;
	AuxOldMagnArray = NULL;
	AuxOldFieldArray = NULL;

	NewMagnArray = NULL;
	NewFieldArray = NULL;
	IdentTransPtr = NULL;

	RelaxSubIntervArray = NULL; // New
	mKeepTransData = 0;
}

//-------------------------------------------------------------------------

int radTInteraction::Setup(const radThg& In_hg, const radThg& In_hgMoreExtSrc, const radTCompCriterium& InCompCriterium, short InMemAllocTotAtOnce, char AuxOldMagnArrayIsNeeded, char KeepTransData, int rankMPI, int nProcMPI) //OC08012020
//int radTInteraction::Setup(const radThg& In_hg, const radThg& In_hgMoreExtSrc, const radTCompCriterium& InCompCriterium, short InMemAllocTotAtOnce, char AuxOldMagnArrayIsNeeded, char KeepTransData)
{
	SomethingIsWrong = 0;

	AmOfMainElem = 0;
	AmOfExtElem = 0;
	InteractMatrix = NULL;
	ExternFieldArray = NULL;
	AuxOldMagnArray = NULL;
	AuxOldFieldArray = NULL;

	NewMagnArray = NULL;
	NewFieldArray = NULL;
	IdentTransPtr = NULL;

	RelaxSubIntervArray = NULL; // New
	AmOfRelaxSubInterv = 0; // New

	SourceHandle = In_hg;
	CompCriterium = InCompCriterium;
	FillInMainTransOnly = 0;
	RelaxationStarted = 0;

	MoreExtSourceHandle = In_hgMoreExtSrc;

	MemAllocTotAtOnce = InMemAllocTotAtOnce;

	IdentTransPtr = new radIdentTrans();

	radTlphgPtr NewListOfTransPtr;
	CountMainRelaxElems((radTg3d*)(SourceHandle.rep), &NewListOfTransPtr);

	if(!NotEmpty()) return 0;

	//m_rankMPI = -1; //OC20122019 (to set from Application?) 
	//m_nProcMPI = 0;
	m_rankMPI = rankMPI; //OC08012019 (to set from Application?) 
	m_nProcMPI = nProcMPI; 

	bool IntrctMatrMemAllocShouldBeDone = true;
	if(m_rankMPI > 0) IntrctMatrMemAllocShouldBeDone = false;

//#ifdef _WITH_MPI
//	if(MPI_Comm_size(MPI_COMM_WORLD, &m_nProcMPI) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); return 0;}
//	if(MPI_Comm_rank(MPI_COMM_WORLD, &m_rankMPI) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); return 0;} //Get the rank of the process
//	if(m_rankMPI > 0) IntrctMatrMemAllocShouldBeDone = false;
//#endif

	if(IntrctMatrMemAllocShouldBeDone) //OC20122019
	{
		AllocateMemory(AuxOldMagnArrayIsNeeded); //In case of MPI-parallelization, this has to be executed by master only

		if(SomethingIsWrong)
		{
			EmptyVectOfPtrToListsOfTrans(); return 0;
		}
		FillInRelaxSubIntervArray(); //New
	}
	FillInMainTransPtrArray();

	if(!SetupInteractMatrix()) { DeallocateMemory(); return 0;} //OC26122019 //Most CPU-intensive
	//SetupInteractMatrix(); //Most CPU-intensive

	if(IntrctMatrMemAllocShouldBeDone) //OC29122019
	{
		SetupExternFieldArray();
		AddExternFieldFromMoreExtSource();
		//ZeroAuxOldMagnArray();
		ZeroAuxOldArrays();

		InitAuxArrays();
	}

	mKeepTransData = KeepTransData;
	if(!KeepTransData) //OC021103
	{
        DestroyMainTransPtrArray();
        EmptyVectOfPtrToListsOfTrans();
	}

	////ResetM();
	//InitAuxArrays(); //OC30122019 (moved up)

#ifdef RADIA_WITH_CUDA
	{//Stamp this interaction matrix for the GPU-resident matrix cache.
		static unsigned long long sGpuMatrixStampCounter = 0;
		mGpuMatrixStamp = ++sGpuMatrixStampCounter;

		//If the GPU assembled this matrix and it fits on the device, hand it to
		//the solver's cache under that stamp instead of letting the first solve
		//flatten it again (O(N^2) host) and upload it. The assembly already
		//emits the solver's layout, so nothing has to be converted. No-op when
		//the assembly ran on the CPU or was tiled.
		radGPU_PublishAssembledMatrix(mGpuMatrixStamp);
	}
#endif

	return 1;
}

//-------------------------------------------------------------------------

radTInteraction::~radTInteraction()
{
	DeallocateMemory(); //OC27122019
}

//-------------------------------------------------------------------------

void radTInteraction::DeallocateMemory() //OC27122019
{
	if(MemAllocTotAtOnce)
	{
		if(InteractMatrix != NULL)
		{
			if(InteractMatrix[0] != NULL) delete[](InteractMatrix[0]);
			delete[] InteractMatrix;
		}
	}
	else
	{
		if(InteractMatrix != NULL)
		{
			for(int i=0; i<AmOfMainElem; i++)
			{
				TMatrix3df* Matrix3dPtr = InteractMatrix[i]; //OC250504
				//TMatrix3d* Matrix3dPtr = InteractMatrix[i]; //OC250504
				if(Matrix3dPtr != NULL) delete[] Matrix3dPtr;
			}
			delete[] InteractMatrix;
		}
	}

	g3dExternPtrVect.erase(g3dExternPtrVect.begin(), g3dExternPtrVect.end()); //OC240408, to enable current scaling/update

	if(ExternFieldArray != NULL) delete[] ExternFieldArray;
	if(AuxOldMagnArray != NULL) delete[] AuxOldMagnArray;
	if(AuxOldFieldArray != NULL) delete[] AuxOldFieldArray;

	if(NewMagnArray != NULL) delete[] NewMagnArray;
	if(NewFieldArray != NULL) delete[] NewFieldArray;

	if(RelaxSubIntervArray != NULL) delete[] RelaxSubIntervArray;

	if(mKeepTransData) //OC021103
	{
		DestroyMainTransPtrArray();
		EmptyVectOfPtrToListsOfTrans();
	}
	if(IdentTransPtr != NULL) delete IdentTransPtr; //required by EmptyVectOfPtrToListsOfTrans();
}

//-------------------------------------------------------------------------

void radTInteraction::CountMainRelaxElems(radTg3d* g3dPtr, radTlphgPtr* CurrListOfTransPtrPtr)
{
	radTGroup* GroupPtr = Cast.GroupCast(g3dPtr);
	if(GroupPtr == 0)
	{
		radTg3dRelax* g3dRelaxPtr = Cast.g3dRelaxCast(g3dPtr);
		if((g3dRelaxPtr != 0) && (g3dRelaxPtr->MaterHandle.rep != 0))
		{
			g3dRelaxPtrVect.push_back(g3dRelaxPtr);
			AmOfMainElem++;

			radTlphgPtr* TotalListOfElemTransPtrPtr = new radTlphgPtr(*CurrListOfTransPtrPtr);
			PushFrontNativeElemTransList(g3dRelaxPtr, TotalListOfElemTransPtrPtr);
			IntVectOfPtrToListsOfTransPtr.push_back(TotalListOfElemTransPtrPtr);
		}
		else 
		{
			g3dExternPtrVect.push_back(g3dPtr);
			AmOfExtElem++;

			radTlphgPtr* TotalListOfElemTransPtrPtr	= new radTlphgPtr(*CurrListOfTransPtrPtr);
			PushFrontNativeElemTransList(g3dPtr, TotalListOfElemTransPtrPtr);
			ExtVectOfPtrToListsOfTransPtr.push_back(TotalListOfElemTransPtrPtr);
		}
	}
	else
	{
		//--New
		radTSubdividedRecMag* SubdividedRecMagPtr = Cast.SubdividedRecMagCast(GroupPtr);
		if(SubdividedRecMagPtr != 0)
		{
			radTg3dRelax* g3dRelaxFromSbdRecMagPtr = (radTg3dRelax*)SubdividedRecMagPtr;

			radTRecMag* SubElRecMagPtr = Cast.RecMagCast((radTg3dRelax*)((*(SubdividedRecMagPtr->GroupMapOfHandlers.begin())).second.rep));

			if((g3dRelaxFromSbdRecMagPtr->MaterHandle.rep != 0) && (SubElRecMagPtr != 0))
			{
				int SubIntervStart = AmOfMainElem;
				if(SubdividedRecMagPtr->FldCmpMeth==1)
				{
					for(int ix=0; ix<int(SubdividedRecMagPtr->kx); ix++)
						for(int iy=0; iy<int(SubdividedRecMagPtr->ky); iy++)
							for(int iz=0; iz<int(SubdividedRecMagPtr->kz); iz++)
							{
								g3dRelaxPtrVect.push_back(g3dRelaxFromSbdRecMagPtr);
								AmOfMainElem++;

								radTlphgPtr* TotalListOfElemTransPtrPtr = new radTlphgPtr(*CurrListOfTransPtrPtr);
								PushFrontNativeElemTransList(g3dRelaxFromSbdRecMagPtr, TotalListOfElemTransPtrPtr);
								IntVectOfPtrToListsOfTransPtr.push_back(TotalListOfElemTransPtrPtr);
							}
				}
				int SubIntervFin = SubIntervStart + (int)(SubdividedRecMagPtr->GroupMapOfHandlers.size()) - 1;

				if(RelaxSubIntervConstrVect.empty())
				{
					radTRelaxSubInterval RlxSbIntrv(SubIntervStart, SubIntervFin, RelaxTogether);
					RelaxSubIntervConstrVect.push_back(RlxSbIntrv);
				}
				else
				{
					radTRelaxSubInterval& LastEnteredSubIntrv = RelaxSubIntervConstrVect.back();
					if((SubIntervStart != LastEnteredSubIntrv.StartNo) && (SubIntervFin != LastEnteredSubIntrv.FinNo))
					{
						radTRelaxSubInterval RlxSbIntrv(SubIntervStart, SubIntervFin, RelaxTogether);
						RelaxSubIntervConstrVect.push_back(RlxSbIntrv);
					}
				}
			}
		}
		if((SubdividedRecMagPtr == 0) || ((SubdividedRecMagPtr != 0) && (SubdividedRecMagPtr->FldCmpMeth != 1)))
		{
		//--EndNew
			radTlphgPtr* LocListOfTransPtrPtr = CurrListOfTransPtrPtr;
			
			short GroupListOfTransIsNotEmpty = 1;
			if(GroupPtr->g3dListOfTransform.empty()) GroupListOfTransIsNotEmpty = 0;

			if(GroupListOfTransIsNotEmpty) 
			{
				LocListOfTransPtrPtr = new radTlphgPtr(*CurrListOfTransPtrPtr);
				PushFrontNativeElemTransList(GroupPtr, LocListOfTransPtrPtr);
			}

			for(radTmhg::iterator iter = GroupPtr->GroupMapOfHandlers.begin();
				iter != GroupPtr->GroupMapOfHandlers.end(); ++iter) 
				CountMainRelaxElems((radTg3d*)((*iter).second.rep), LocListOfTransPtrPtr);

			if(GroupListOfTransIsNotEmpty) delete LocListOfTransPtrPtr;
		//--New
		}
		//--EndNew
	}
}

//-------------------------------------------------------------------------

void radTInteraction::FillInRelaxSubIntervArray() // New
{
	if(RelaxSubIntervConstrVect.size() == 0) return;

	int CurrentStartNo = 0;
	int PlainCount = -1;

	vector<radTRelaxSubInterval>::iterator Iter;

	for(Iter = RelaxSubIntervConstrVect.begin(); Iter != RelaxSubIntervConstrVect.end(); ++Iter)
	{
		int LocStartNo = (*Iter).StartNo;
		if(LocStartNo != CurrentStartNo)
		{
			RelaxSubIntervArray[++PlainCount] = radTRelaxSubInterval(CurrentStartNo, LocStartNo-1, RelaxApart);
		}
		RelaxSubIntervArray[++PlainCount] = *Iter;
		CurrentStartNo = (*Iter).FinNo + 1;
	}
	if(CurrentStartNo != AmOfMainElem)
		RelaxSubIntervArray[++PlainCount] = radTRelaxSubInterval(CurrentStartNo, AmOfMainElem-1, RelaxApart);
	
	AmOfRelaxSubInterv = ++PlainCount;

	RelaxSubIntervConstrVect.erase(RelaxSubIntervConstrVect.begin(), RelaxSubIntervConstrVect.end());
}

//-------------------------------------------------------------------------

void radTInteraction::AllocateMemory(char AuxOldMagnArrayIsNeeded)
{
	//try
	//{
		ExternFieldArray = new TVector3d[AmOfMainElem];
		if(AuxOldMagnArrayIsNeeded) 
		{
			AuxOldMagnArray = new TVector3d[AmOfMainElem];
			AuxOldFieldArray = new TVector3d[AmOfMainElem];
		}

		NewMagnArray = new TVector3d[AmOfMainElem];
		NewFieldArray = new TVector3d[AmOfMainElem];
		InteractMatrix = new TMatrix3df*[AmOfMainElem]; //OC250504
		//InteractMatrix = new TMatrix3d*[AmOfMainElem]; //OC250504

		for(int k=0; k<AmOfMainElem; k++) InteractMatrix[k] = NULL;
	//}
	//catch (radTException* radExceptionPtr)
	//{
	//	Send.ErrorMessage(radExceptionPtr->what());	return;
	//}
	//catch (...)
	//{
	//	Send.ErrorMessage("Radia::Error999"); return;
	//}

	if(MemAllocTotAtOnce)
	{
		TMatrix3df* GenMatrPtr = 0; //OC250504
		//TMatrix3d* GenMatrPtr = 0; //OC250504
		//try
		//{
			GenMatrPtr = new TMatrix3df[AmOfMainElem*AmOfMainElem]; //OC250504
			//GenMatrPtr = new TMatrix3d[AmOfMainElem*AmOfMainElem]; //OC250504
		//}
		//catch (radTException* radExceptionPtr)
		//{
		//	InteractMatrix[0] = NULL;
		//	SomethingIsWrong = 1;
		//	Send.ErrorMessage(radExceptionPtr->what());	return;
		//}
		//catch (...)
		//{
		//	Send.ErrorMessage("Radia::Error999"); return;
		//}

		if(GenMatrPtr != 0) // Check for allocation failure
			for(int i=0; i<AmOfMainElem; i++) InteractMatrix[i] = &(GenMatrPtr[i*AmOfMainElem]);
		else
		{
			InteractMatrix[0] = NULL;
			SomethingIsWrong = 1;
			Send.ErrorMessage("Radia::Error900"); return;
		}
	}
	else
	{
		for(int i=0; i<AmOfMainElem; i++)
		{
			InteractMatrix[i] = new TMatrix3df[AmOfMainElem]; //OC250504
			//InteractMatrix[i] = new TMatrix3d[AmOfMainElem]; //OC250504
			if(InteractMatrix[i] == 0) // Check for allocation failure
			{
				for(int k=0; k<i; k++) delete[] (InteractMatrix[i]);
				delete[] InteractMatrix;

				SomethingIsWrong = 1;
				Send.ErrorMessage("Radia::Error900"); return;
			}
		}
	}

	int MaxSubIntervArraySize = 2 * ((int)(RelaxSubIntervConstrVect.size())) + 1; // New
	//try
	//{
		if(MaxSubIntervArraySize > 1) RelaxSubIntervArray = new radTRelaxSubInterval[MaxSubIntervArraySize]; // New
	//}
	//catch (radTException* radExceptionPtr)
	//{
	//	Send.ErrorMessage(radExceptionPtr->what());	return;
	//}
	//catch (...)
	//{
	//	Send.ErrorMessage("Radia::Error999"); return;
	//}
}

//-------------------------------------------------------------------------

void radTInteraction::NestedFor_Trans(radTrans* BaseTransPtr, const radTlphgPtr::const_iterator& Iter, int ElemLocInd, char I_or_E)
{
	radTrans* TransPtr = (radTrans*)(((**Iter).Handler_g).rep);
	radTrans* LocTotTransPtr = BaseTransPtr;
	radTrans LocTotTrans;

	radTlphgPtr::const_iterator LocalNextIter = Iter;
	LocalNextIter++;
	int Mult = (**Iter).m;

	if(Mult == 1)
	{
		TrProduct(LocTotTransPtr, TransPtr, LocTotTrans);
		AddTransOrNestedFor(&LocTotTrans, LocalNextIter, ElemLocInd, I_or_E);
	}
	else
	{
		AddTransOrNestedFor(LocTotTransPtr, LocalNextIter, ElemLocInd, I_or_E);
		if(FillInMainTransOnly) return;
		for(int km = 1; km < Mult; km++)
		{
			TrProduct(LocTotTransPtr, TransPtr, LocTotTrans);
			LocTotTransPtr = &LocTotTrans;
			AddTransOrNestedFor(LocTotTransPtr, LocalNextIter, ElemLocInd, I_or_E);
		}
	}
}

//-------------------------------------------------------------------------

void radTInteraction::FillInMainTransPtrArray()
{
	MainTransPtrArray = new radTrans*[AmOfMainElem];
	FillInMainTransOnly = 1;

	for(int i=0; i<AmOfMainElem; i++)
	{
		FillInTransPtrVectForElem(i, 'I');
		if(Cast.IdentTransCast(TransPtrVect[0]) == 0) 
		{
			MainTransPtrArray[i] = new radTrans(*(TransPtrVect[0]));
		}
		else MainTransPtrArray[i] = IdentTransPtr;
		EmptyTransPtrVect();
	}
	FillInMainTransOnly = 0;
}

//-------------------------------------------------------------------------

int radTInteraction::CountRelaxElemsWithSym()
{
	int AmOfElemWithSym = 0;

	for(int i=0; i<AmOfMainElem; i++)
	{
		radTlphgPtr& Loc_lphgPtr = *(IntVectOfPtrToListsOfTransPtr[i]);
		int LocTotMult = 1;

		for(radTlphgPtr::iterator TrIter = Loc_lphgPtr.begin();	
			TrIter != Loc_lphgPtr.end(); ++TrIter)
		{
			LocTotMult *= (**TrIter).m;
		}
		AmOfElemWithSym += LocTotMult;
	}
	return AmOfElemWithSym;
}

//-------------------------------------------------------------------------

//RadiaCUDA: OPT-IN volume-averaged ("Galerkin") assembly -- see radgalerkin.h.
//Builds the per-row observation quadratures (base rule, and the near-band rule
//when a cutoff is set). Returns 0 if any element type has no quadrature, in
//which case the caller must not use the Galerkin path for this model.
int radTInteraction::PrepGalerkinQuad()
{
	const radTGalerkinCfg& cfg = radGalerkinCfg();
	m_galNearOn = 0;
	m_galQPts.assign((size_t)AmOfMainElem, vector<TVector3d>());
	m_galQWts.assign((size_t)AmOfMainElem, vector<double>());
	m_galCen.assign((size_t)AmOfMainElem, TVector3d(0.,0.,0.));
	m_galH.assign((size_t)AmOfMainElem, 0.);

	vector<TVector3d> ep; vector<double> ew;
	long long nPtsBase = 0;
	for(int i=0; i<AmOfMainElem; i++)
	{
		radTg3dRelax* el = g3dRelaxPtrVect[i];
		if(!radGalerkinElemQuad(el, cfg.K, 0, ep, ew)) return 0;
		radTrans* tr = MainTransPtrArray[i];
		m_galQPts[i].resize(ep.size());
		m_galQWts[i] = ew;
		for(size_t k=0; k<ep.size(); k++)
			m_galQPts[i][k] = (tr != 0)? tr->TrPoint(ep[k]) : ep[k];
		nPtsBase += (long long)ep.size();

		m_galCen[i] = (tr != 0)? tr->TrPoint(el->ReturnCentrPoint())
		                       : el->ReturnCentrPoint();
		m_galH[i] = radGalerkinElemSize(el);
	}

	bool wantNear = (cfg.Cutoff > 0.) &&
	                ((cfg.KNear != cfg.K) || (cfg.NearLevels > 0));
	if(wantNear)
	{
		m_galNPts.assign((size_t)AmOfMainElem, vector<TVector3d>());
		m_galNWts.assign((size_t)AmOfMainElem, vector<double>());
		for(int i=0; i<AmOfMainElem; i++)
		{
			radTg3dRelax* el = g3dRelaxPtrVect[i];
			if(!radGalerkinElemQuad(el, cfg.KNear, cfg.NearLevels, ep, ew)) return 0;
			radTrans* tr = MainTransPtrArray[i];
			m_galNPts[i].resize(ep.size());
			m_galNWts[i] = ew;
			for(size_t k=0; k<ep.size(); k++)
				m_galNPts[i][k] = (tr != 0)? tr->TrPoint(ep[k]) : ep[k];
		}
		for(int i=0; i<AmOfMainElem; i++) if(m_galH[i] <= 0.) { wantNear = false; break;}
		m_galNearOn = wantNear? 1 : 0;
	}

	if(cfg.Debug)
	{
		fprintf(stderr, "Galerkin (CPU): base K=%d -> %.2f points/element; "
		                "near K=%d x 8^%d %s (cutoff %.2f h)\n",
			cfg.K, (double)nPtsBase/(double)AmOfMainElem, cfg.KNear,
			cfg.NearLevels, m_galNearOn? "on" : "off", cfg.Cutoff);
	}
	return 1;
}

//-------------------------------------------------------------------------

//RadiaCUDA: the volume-averaged interaction block. TransPtrVect must already
//hold the COLUMN element's symmetry copies (FillInTransPtrVectForElem(ColNo)),
//exactly as for the collocation code this replaces. The only difference is that
//the row element contributes several weighted observation points instead of one.
void radTInteraction::GalerkinInteractBlock(int StrNo, int ColNo,
	radTg3dRelax* g3dRelaxPtrColNo, const radTFieldKey& FieldKeyInteract,
	int AmOfElemWithSym, TMatrix3d& SubMatrix)
{
	TVector3d ZeroVect(0.,0.,0.);
	SubMatrix = TMatrix3d(ZeroVect, ZeroVect, ZeroVect);

	//Near band: the higher-order rule, on the O(N) pairs whose centroids are
	//within cutoff*max(h_i,h_j). Straight distance test -- the CPU path only
	//ever runs on small models.
	const vector<TVector3d>* pPts = &m_galQPts[StrNo];
	const vector<double>* pWts = &m_galQWts[StrNo];
	if(m_galNearOn)
	{
		const radTGalerkinCfg& cfg = radGalerkinCfg();
		TVector3d dc = m_galCen[StrNo] - m_galCen[ColNo];
		double hij = (m_galH[StrNo] > m_galH[ColNo])? m_galH[StrNo] : m_galH[ColNo];
		if(sqrt(dc.x*dc.x + dc.y*dc.y + dc.z*dc.z) < cfg.Cutoff*hij)
		{
			pPts = &m_galNPts[StrNo];
			pWts = &m_galNWts[StrNo];
		}
	}

	TMatrix3d BufSubMatrix;
	for(size_t iq=0; iq<pPts->size(); iq++)
	{
		TVector3d InitObsPoiVect = (*pPts)[iq];
		double wq = (*pWts)[iq];
		TMatrix3d AccMatrix(ZeroVect, ZeroVect, ZeroVect);
		for(unsigned i=0; i<TransPtrVect.size(); i++)
		{
			TVector3d ObsPoiVect = TransPtrVect[i]->TrPoint_inv(InitObsPoiVect);
			radTField Field(FieldKeyInteract, CompCriterium, ObsPoiVect, ZeroVect, ZeroVect, ZeroVect, ZeroVect, 0.);
			Field.AmOfIntrctElemWithSym = AmOfElemWithSym;

			g3dRelaxPtrColNo->B_comp(&Field);
			BufSubMatrix.Str0 = Field.B;
			BufSubMatrix.Str1 = Field.H;
			BufSubMatrix.Str2 = Field.A;

			TransPtrVect[i]->TrMatrix(BufSubMatrix);
			AccMatrix += BufSubMatrix;
		}
		AccMatrix.Str0 = AccMatrix.Str0*wq;
		AccMatrix.Str1 = AccMatrix.Str1*wq;
		AccMatrix.Str2 = AccMatrix.Str2*wq;
		SubMatrix += AccMatrix;
	}
	MainTransPtrArray[StrNo]->TrMatrix_inv(SubMatrix);
}

//-------------------------------------------------------------------------

int radTInteraction::SetupInteractMatrix() //OC26122019
//void radTInteraction::SetupInteractMatrix()
{
	radTFieldKey FieldKeyInteract; FieldKeyInteract.B_=FieldKeyInteract.H_=FieldKeyInteract.PreRelax_=1;
	TVector3d ZeroVect(0.,0.,0.);

	gLastAsmBackend = 0; //RadiaCUDA diagnostic: set to 1 below iff GPU assembly succeeds

	//--New
	int AmOfElemWithSym = CountRelaxElemsWithSym();
	//--EndNew

#ifdef RADIA_WITH_CUDA
	// GPU interaction-matrix assembly (gUseGpuAsm; rad.RlxPre(obj, use_gpu=...)).
	// Serial: as before. Under MPI: rank 0 assembles on the GPU while the
	// workers wait, then all ranks AGREE (Bcast) whether it succeeded -- the
	// workers would otherwise deadlock in the distributed-CPU send/recv below.
	if(gUseGpuAsm)
	{
		int gpuAsmOK = 0;
		if((m_nProcMPI < 2) || (m_rankMPI <= 0))
		{
			RadGPU_PolyData polyData;
			RadGPU_RecMagData recData;
			RadGPU_SymData symData;
			RadGPU_ObsQuadData quadData;
			RadGPU_AsmResult result;
			memset(&polyData, 0, sizeof(polyData));
			memset(&recData, 0, sizeof(recData));
			memset(&symData, 0, sizeof(symData));
			memset(&quadData, 0, sizeof(quadData));
			memset(&result, 0, sizeof(result));

			if(radGPU_PackGeometryForAsm(this, &polyData, &recData, &symData, &quadData))
			{
				if(radGPU_AssembleMatrix(&polyData, &recData, &symData, &quadData, &result) == 0)
				{
					radGPU_UnpackMatrix(&result, this);
					radGPU_FreeAsmData(&polyData, &recData, &result);
					radGPU_FreeSymData(&symData);
					radGPU_FreeObsQuadData(&quadData);

					for(int ClNo=0; ClNo<AmOfMainElem; ClNo++)
					{
						radTg3dRelax* g3dRelaxPtrClNo = g3dRelaxPtrVect[ClNo];
						g3dRelaxPtrVect[ClNo] = g3dRelaxPtrClNo->FormalIntrctMemberPtr();
					}
					gpuAsmOK = 1;
				}
				else
				{
					radGPU_FreeAsmData(&polyData, &recData, &result);
					radGPU_FreeSymData(&symData);
					radGPU_FreeObsQuadData(&quadData);
					Send.WarningMessage("Radia::Warning021"); //GPU assembly could not complete (e.g. out of GPU memory or unimplemented element kernel); falling back to CPU
				}
			}
			else radGPU_FreeObsQuadData(&quadData);
		}
#ifdef _WITH_MPI
		if((m_rankMPI >= 0) && (m_nProcMPI >= 2))
		{
			if(MPI_Bcast(&gpuAsmOK, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); return 0; }
		}
#endif
		if(gpuAsmOK) { gLastAsmBackend = 1; return 1;}

		//The GPU could not service the matrix. What happens now is the USER's
		//choice (rad.UtiGpuFallback), never an automatic decision: dropping to
		//the CPU can turn a seconds-long solve into an hours-long one, so
		//'break' lets the caller find out immediately and reduce the model
		//size instead. Tested after the Bcast so every rank agrees.
		if(gGpuFallback == 2) { Send.ErrorMessage("Radia::Error602"); return 0;}

		//else: fall through to the CPU assembly (serial or MPI-distributed),
		//consistently on all ranks.
	}
#endif

	//RadiaCUDA: OPT-IN Galerkin (volume-averaged) assembly. Prepared once here
	//so both CPU loops below can use it; with the flag off nothing is built and
	//the collocation code runs exactly as before.
	char galOn = 0;
	if(radGalerkinCfg().On)
	{
		if(PrepGalerkinQuad()) galOn = 1;
		else Send.WarningMessage("Radia::Warning024"); //element type without a volume quadrature -> collocation
	}

	if(m_nProcMPI < 2) //OC01012020
	{
		//DEBUG
		//long iCntBcomp = 0;
		//END DEBUG

		for(int ColNo=0; ColNo<AmOfMainElem; ColNo++)
		{
			FillInTransPtrVectForElem(ColNo, 'I');
			radTg3dRelax* g3dRelaxPtrColNo = g3dRelaxPtrVect[ColNo];

			if(galOn)
			{
				for(int StrNo=0; StrNo<AmOfMainElem; StrNo++)
				{
					TMatrix3d SubMatrix;
					GalerkinInteractBlock(StrNo, ColNo, g3dRelaxPtrColNo,
						FieldKeyInteract, AmOfElemWithSym, SubMatrix);
					InteractMatrix[StrNo][ColNo] = SubMatrix;
				}
				EmptyTransPtrVect();
				continue;
			}

			for(int StrNo=0; StrNo<AmOfMainElem; StrNo++)
			{
				TVector3d InitObsPoiVect = MainTransPtrArray[StrNo]->TrPoint((g3dRelaxPtrVect[StrNo])->ReturnCentrPoint());

				TMatrix3d SubMatrix(ZeroVect, ZeroVect, ZeroVect), BufSubMatrix;
				for(unsigned i=0; i<TransPtrVect.size(); i++)
				{
					TVector3d ObsPoiVect = TransPtrVect[i]->TrPoint_inv(InitObsPoiVect);

					radTField Field(FieldKeyInteract, CompCriterium, ObsPoiVect, ZeroVect, ZeroVect, ZeroVect, ZeroVect, 0.);
					Field.AmOfIntrctElemWithSym = AmOfElemWithSym; // New, may be changed later

					g3dRelaxPtrColNo->B_comp(&Field);

					BufSubMatrix.Str0 = Field.B;
					BufSubMatrix.Str1 = Field.H;
					BufSubMatrix.Str2 = Field.A;

					//DEBUG
					//iCntBcomp++;
					//END DEBUG

					TransPtrVect[i]->TrMatrix(BufSubMatrix);
					SubMatrix += BufSubMatrix;
				}
				MainTransPtrArray[StrNo]->TrMatrix_inv(SubMatrix);
				InteractMatrix[StrNo][ColNo] = SubMatrix;
			}
			EmptyTransPtrVect();
		}

		//DEBUG
		//long long nTotMatrElem = ((long long)AmOfMainElem)*((long long)AmOfMainElem);
		//std::cout << "rank=" << m_rankMPI << ": iCntBcomp= " << iCntBcomp << "; nTotMatrElem=" << nTotMatrElem; //DEBUG
		//std::cout.flush();
		//END DEBUG

		//--New
		for(int ClNo=0; ClNo<AmOfMainElem; ClNo++)
		{
			radTg3dRelax* g3dRelaxPtrClNo = g3dRelaxPtrVect[ClNo];
			g3dRelaxPtrVect[ClNo] = g3dRelaxPtrClNo->FormalIntrctMemberPtr();
		}
		//--EndNew
	}
#ifdef _WITH_MPI
	else
	{
		//DEBUG
		//std::cout << "rank=" << m_rankMPI << ": Hello";
		//std::cout.flush(); 
		//END DEBUG

		vector<pair<long long, long long> > vPacketElemStartEnd;
		int nProc_mi_1 = m_nProcMPI - 1;
		const long long switchAmOfElem = 1000; //threshold to switch between different data packaging for sending via MPI
		
		int nPacketsTot = 0; //required for master process
		long long nMaxMatrElemInPacket = 0;

		//NOTE (RadiaCUDA): the 2-rank shortcut of sending the WHOLE matrix in one
		//packet overflows MPI_Send's int count for AmOfMainElem >~ 15450
		//(N*N*9 floats > INT_MAX) -> send error while the master blocks in
		//MPI_Recv = deadlock. Restrict it to small models; large 2-rank runs
		//use the same column-packet scheme as nProc >= 3.
		if((m_nProcMPI < 3) && (AmOfMainElem < switchAmOfElem))
		{
			long long nTotMainElem = ((long long)AmOfMainElem)*((long long)AmOfMainElem);

			if(m_rankMPI > 0)
			{
				pair<long long, long long> pairStartEnd(0, nTotMainElem);
				vPacketElemStartEnd.push_back(pairStartEnd);
			}
			else
			{//required for master process
				nPacketsTot = 1;
				nMaxMatrElemInPacket = nTotMainElem;
			}
		}
		else if((m_nProcMPI < AmOfMainElem + 1) && (AmOfMainElem >= switchAmOfElem))
		//else if(nProcMPI < AmOfMainElem + 1)
		{//Send matrix elements to master by packets of AmOfMainElem in length

			if(m_rankMPI > 0)
			{
				pair<long long, long long> pairStartEnd(0, 0);
				long long nPerGen = AmOfMainElem*nProc_mi_1;
				long long nPackets = (long long)(AmOfMainElem/nProc_mi_1 + 1.e-14);
				long long iStart = (m_rankMPI - 1)*AmOfMainElem;
				for(long long i=1; i<=nPackets; i++)
				{
					pairStartEnd.first = iStart;
					pairStartEnd.second = iStart + AmOfMainElem;
					vPacketElemStartEnd.push_back(pairStartEnd);
					iStart += nPerGen;
				}
				if(nPackets*nProc_mi_1 < AmOfMainElem)
				{
					//iStart += (AmOfMainElem - nPerGen);
					long long nTotMainElem = ((long long)AmOfMainElem)*((long long)AmOfMainElem);
					if((iStart + AmOfMainElem) <= nTotMainElem)
					{
						pairStartEnd.first = iStart;
						pairStartEnd.second = iStart + AmOfMainElem;
						vPacketElemStartEnd.push_back(pairStartEnd);
					}
				}
			}
			else
			{//required for master process
				//long long nTotMainElem = ((long long)AmOfMainElem)*((long long)AmOfMainElem); //required for master process
				//nPacketsTot = (int)(nTotMainElem/nProc_mi_1 + 1.e-14);
				//if(((long long)nPacketsTot)*((long long)nProc_mi_1) < nTotMainElem) nPacketsTot++;

				nPacketsTot = AmOfMainElem; //OC30122019
				nMaxMatrElemInPacket = AmOfMainElem;
			}
		}
		else
		{//Send matrix elements to master by one packet (by each worker process) of ~AmOfMainElem*AmOfMainElem/(nProcMPI - 1) in length
			long long nTotMatrElem = ((long long)AmOfMainElem)*((long long)AmOfMainElem);
			long long nElemPerProc = (long long)(nTotMatrElem/nProc_mi_1 + 1.e-14);
			long long nExtraLast = nTotMatrElem - nElemPerProc*nProc_mi_1;

			if(m_rankMPI > 0)
			{
				pair<long long, long long> pairStartEnd((m_rankMPI - 1)*nElemPerProc, m_rankMPI*nElemPerProc);
				if(nExtraLast > 0)
				{
					if(m_rankMPI == nProc_mi_1) pairStartEnd.second += nExtraLast;
				}
				vPacketElemStartEnd.push_back(pairStartEnd);
				
				//std::cout << "rank=" << rankMPI << " nTotMatrElem=" << nTotMatrElem << "\n"; //DEBUG
				//std::cout << "rank=" << rankMPI << " pairStartEnd.first=" << pairStartEnd.first << " pairStartEnd.second=" << pairStartEnd.second << "\n"; //DEBUG
				//std::cout.flush(); //DEBUG
			}
			else
			{//required for master process
				nPacketsTot = nProc_mi_1; //required for master process
				nMaxMatrElemInPacket = nElemPerProc + nExtraLast;
			}
		}

		if(m_rankMPI > 0)
		{//Workers: calculate Interactin Matrix elements and send them to master
			long long nBufElem=0;
			int nPackets = (int)vPacketElemStartEnd.size(), ii;
			for(ii=0; ii<nPackets; ii++)
			{
				pair<long long, long long> &pairStartEnd = vPacketElemStartEnd[ii];
				long long nElemCur = pairStartEnd.second - pairStartEnd.first;
				if(nBufElem < nElemCur) nBufElem = nElemCur;
			}

			float *arBufElem=0;
			if(nBufElem > 0)
			{
				arBufElem = new float[nBufElem*9 + 8]; //the first 8 float values encode long long iStart, iEnd, all other values - elements of interaction matrix
				if(arBufElem == 0) { Send.ErrorMessage("Radia::Error900"); return 0;}

				//DEBUG
				//long iCntBcomp = 0;
				//std::cout << "rank=" << m_rankMPI << ": nPackets=" << nPackets; //DEBUG
				//std::cout.flush(); //DEBUG
				//END DEBUG

				for(ii=0; ii<nPackets; ii++)
				{
					pair<long long, long long> &pairStartEnd = vPacketElemStartEnd[ii];
					long long iStart = pairStartEnd.first;
					long long iEnd = pairStartEnd.second;

					float *t_arBufElem = arBufElem; //the first 8 float values encode long long iStart, iEnd
					LongLongToFloatAr(iStart, t_arBufElem); t_arBufElem += 4;
					LongLongToFloatAr(iEnd, t_arBufElem); t_arBufElem += 4;

					int ColNoStart = (int)(iStart/AmOfMainElem + 1.e-14);
					int StrNoStart = (int)(iStart - ((long long)ColNoStart)*((long long)AmOfMainElem));
					int ColNoEnd = (int)(iEnd/AmOfMainElem + 1.e-14);
					int StrNoEnd = (int)(iEnd - ((long long)ColNoEnd)*((long long)AmOfMainElem));
					
					if(StrNoEnd > 0) ColNoEnd++; //OC29122019
					else if(ColNoEnd - ColNoStart > 0) StrNoEnd = AmOfMainElem;

					//std::cout << "Before sending, rank=" << rankMPI << " iStart=" << iStart << " iEnd=" << iEnd << "\n"; //DEBUG
					//std::cout << "Before sending, rank=" << rankMPI << " ColNoStart=" << ColNoStart << " StrNoStart=" << StrNoStart << " ColNoEnd=" << ColNoEnd << " StrNoEnd=" << StrNoEnd << " AmOfMainElem=" << AmOfMainElem << "\n"; //DEBUG
					//std::cout.flush(); //DEBUG

					int ColNoEnd_mi_1 = ColNoEnd - 1;

					long long iElem = 0;
					for(int iCol=ColNoStart; iCol<ColNoEnd; iCol++)
					{
						int iStrStart = (iCol > ColNoStart)? 0 : StrNoStart;
						int iStrEnd = (iCol < ColNoEnd_mi_1)? AmOfMainElem : StrNoEnd;

						FillInTransPtrVectForElem(iCol, 'I');
						radTg3dRelax* g3dRelaxPtrColNo = g3dRelaxPtrVect[iCol];

						for(int iStr=iStrStart; iStr<iStrEnd; iStr++)
						{
							TMatrix3d SubMatrix(ZeroVect, ZeroVect, ZeroVect), BufSubMatrix;
							if(galOn)
							{
								GalerkinInteractBlock(iStr, iCol, g3dRelaxPtrColNo,
									FieldKeyInteract, AmOfElemWithSym, SubMatrix);
								TVector3d &g0 = SubMatrix.Str0, &g1 = SubMatrix.Str1, &g2 = SubMatrix.Str2;
								*(t_arBufElem++) = (float)g0.x; *(t_arBufElem++) = (float)g0.y;  *(t_arBufElem++) = (float)g0.z;
								*(t_arBufElem++) = (float)g1.x; *(t_arBufElem++) = (float)g1.y;  *(t_arBufElem++) = (float)g1.z;
								*(t_arBufElem++) = (float)g2.x; *(t_arBufElem++) = (float)g2.y;  *(t_arBufElem++) = (float)g2.z;
								continue;
							}

							TVector3d InitObsPoiVect = MainTransPtrArray[iStr]->TrPoint((g3dRelaxPtrVect[iStr])->ReturnCentrPoint());

							//if((iCol == 0) && (iStr == 0)) //DEBUG
							//{
							//	std::cout << "SetupInteractMatrix, rank=" << rankMPI; // << " InitObsPoiVect:" << InitObsPoiVect.x << InitObsPoiVect.y << InitObsPoiVect.z; //DEBUG
							//	std::cout.flush(); //DEBUG
							//}

							for(unsigned i=0; i<TransPtrVect.size(); i++)
							{
								TVector3d ObsPoiVect = TransPtrVect[i]->TrPoint_inv(InitObsPoiVect);
								radTField Field(FieldKeyInteract, CompCriterium, ObsPoiVect, ZeroVect, ZeroVect, ZeroVect, ZeroVect, 0.);
								Field.AmOfIntrctElemWithSym = AmOfElemWithSym; // New, may be changed later

								//if((ColNo == 0) && (StrNo == 0) && (rankMPI != 0)) //DEBUG
								//{
								//	std::cout << "radTInteraction::SetupInteractMatrix, rank=" << rankMPI << " ObsPoiVect:" << ObsPoiVect.x << ObsPoiVect.y << ObsPoiVect.z; //DEBUG
								//	std::cout.flush(); //DEBUG
								//}

								g3dRelaxPtrColNo->B_comp(&Field);
								BufSubMatrix.Str0 = Field.B;
								BufSubMatrix.Str1 = Field.H;
								BufSubMatrix.Str2 = Field.A;

								//DEBUG
								//iCntBcomp++;

								TransPtrVect[i]->TrMatrix(BufSubMatrix);
								SubMatrix += BufSubMatrix;
							}
							MainTransPtrArray[iStr]->TrMatrix_inv(SubMatrix);
							TVector3d &v0 = SubMatrix.Str0, &v1 = SubMatrix.Str1, &v2 = SubMatrix.Str2;
							*(t_arBufElem++) = (float)v0.x; *(t_arBufElem++) = (float)v0.y;  *(t_arBufElem++) = (float)v0.z;
							*(t_arBufElem++) = (float)v1.x; *(t_arBufElem++) = (float)v1.y;  *(t_arBufElem++) = (float)v1.z;
							*(t_arBufElem++) = (float)v2.x; *(t_arBufElem++) = (float)v2.y;  *(t_arBufElem++) = (float)v2.z;
							//InteractMatrix[StrNo][ColNo] = SubMatrix; //To be done by master process
						}
						EmptyTransPtrVect();
					}
					//Send Interact. Matr. elem. data to master:
					long long nVal = t_arBufElem - arBufElem;

					if(MPI_Send(arBufElem, (int)nVal, MPI_FLOAT, 0, 0, MPI_COMM_WORLD) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); if(arBufElem != 0) delete[] arBufElem; return 0;}

					//std::cout << "Sending done by rank=" << rankMPI << " nVal=" << nVal; //DEBUG
					//std::cout.flush(); //DEBUG
				}
				if(arBufElem != 0) delete[] arBufElem;

				//DEBUG
				//long long nTotMatrElem = ((long long)AmOfMainElem)*((long long)AmOfMainElem);
				//std::cout << "rank=" << m_rankMPI << ": iCntBcomp= " << iCntBcomp << "; nTotMatrElem=" << nTotMatrElem; //DEBUG
				//std::cout.flush(); 
				//END DEBUG
			}
		}
		else if((nPacketsTot > 0) && (nMaxMatrElemInPacket > 0))
		{//Master: receive calculated Interactin Matrix elements from workers and store them

			long long nMaxValInPacket = nMaxMatrElemInPacket*9 + 8;
			float *arBufElemRecv = new float[nMaxValInPacket];
			if(arBufElemRecv == 0) { Send.ErrorMessage("Radia::Error900"); return 0;}

			MPI_Status statusMPI;
			int trueNumValInPacket = 0;

			//std::cout << "rank=" << rankMPI << " nPacketsTot=" << nPacketsTot << "\n"; //DEBUG
			//std::cout.flush(); //DEBUG

			for(int i=0; i<nPacketsTot; i++)
			{
				if(MPI_Recv(arBufElemRecv, (int)nMaxValInPacket, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &statusMPI) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); delete[] arBufElemRecv; return 0;}
				if(MPI_Get_count(&statusMPI, MPI_FLOAT, &trueNumValInPacket) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); delete[] arBufElemRecv; return 0;}

				if(trueNumValInPacket < 8) { Send.ErrorMessage("Radia::Error601"); delete[] arBufElemRecv; return 0;}

				float *t_arBufElemRecv = arBufElemRecv;
				long long iStart = FloatArToLongLong(t_arBufElemRecv);
				t_arBufElemRecv += 4;
				long long iEnd = FloatArToLongLong(t_arBufElemRecv);
				t_arBufElemRecv += 4;

				long long expectedNumValInPacket = (iEnd - iStart)*9 + 8;

				if(expectedNumValInPacket > trueNumValInPacket) { Send.ErrorMessage("Radia::Error601"); delete[] arBufElemRecv; return 0;}

				int ColNoStart = (int)(iStart/AmOfMainElem + 1.e-14);
				int StrNoStart = (int)(iStart - ((long long)ColNoStart)*((long long)AmOfMainElem));
				int ColNoEnd = (int)(iEnd/AmOfMainElem + 1.e-14);
				int StrNoEnd = (int)(iEnd - ((long long)ColNoEnd)*((long long)AmOfMainElem));

				if(StrNoEnd > 0) ColNoEnd++; //OC29122019
				else if(ColNoEnd - ColNoStart > 0) StrNoEnd = AmOfMainElem;

				//std::cout << "Received, rank=" << rankMPI << " iStart=" << iStart << " iEnd=" << iEnd << "\n"; //DEBUG
				//std::cout << "Received, rank=" << rankMPI << " ColNoStart=" << ColNoStart << " StrNoStart=" << StrNoStart << " ColNoEnd=" << ColNoEnd << " StrNoEnd=" << StrNoEnd << " AmOfMainElem=" << AmOfMainElem << "\n"; //DEBUG
				//std::cout.flush(); //DEBUG

				int ColNoEnd_mi_1 = ColNoEnd - 1;

				for(int iCol=ColNoStart; iCol<ColNoEnd; iCol++)
				{
					int iStrStart = (iCol > ColNoStart)? 0 : StrNoStart;
					int iStrEnd = (iCol < ColNoEnd_mi_1)? AmOfMainElem : StrNoEnd;

					//std::cout << "rank=" << rankMPI << " iCol=" << iCol << " iStrStart=" << iStrStart << " iStrEnd=" << iStrEnd << "\n"; //DEBUG
					//std::cout.flush(); //DEBUG

					for(int iStr=iStrStart; iStr<iStrEnd; iStr++)
					{
						//if((iCol == 0) && (iStr == 0)) //DEBUG
						//{
						//	std::cout << "SetupInteractMatrix, rank=" << rankMPI; // << " InitObsPoiVect:" << InitObsPoiVect.x << InitObsPoiVect.y << InitObsPoiVect.z; //DEBUG
						//	std::cout.flush(); //DEBUG
						//}

						TMatrix3df &SubMatrix = InteractMatrix[iStr][iCol];
						TVector3df &Str0 = SubMatrix.Str0, &Str1 = SubMatrix.Str1, &Str2 = SubMatrix.Str2;
						Str0.x = *(t_arBufElemRecv++); Str0.y = *(t_arBufElemRecv++); Str0.z = *(t_arBufElemRecv++);
						Str1.x = *(t_arBufElemRecv++); Str1.y = *(t_arBufElemRecv++); Str1.z = *(t_arBufElemRecv++);
						Str2.x = *(t_arBufElemRecv++); Str2.y = *(t_arBufElemRecv++); Str2.z = *(t_arBufElemRecv++);

						//std::cout << "rank=" << rankMPI << " iCol=" << iCol << " iStr=" << iStr << "\n"; //DEBUG
						//std::cout.flush(); //DEBUG
					}
				}
				//std::cout << "Packet number " << i << " received by rank=" << rankMPI << "\n"; //DEBUG
				//std::cout.flush(); //DEBUG
			}
			delete[] arBufElemRecv;
		}
		//To consider synchronization:
		//if(MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) { Send.ErrorMessage("Radia::Error601"); throw 0; } //OC18012020
	}
	//std::cout << "rank=" << rankMPI << " about to exit radTInteraction::SetupInteractMatrix\n"; //DEBUG
	//std::cout.flush(); //DEBUG

#endif
	return 1; //OC26122019
}

//-------------------------------------------------------------------------

void radTInteraction::SetupExternFieldArray()
{
	radTFieldKey FieldKeyExtern; FieldKeyExtern.H_=1;
	TVector3d ZeroVect(0.,0.,0.), InitObsPoiVect(0.,0.,0.), ObsPoiVect(0.,0.,0.);

	for(int k=0; k<AmOfMainElem; k++) ExternFieldArray[k] = ZeroVect;

	for(int ExtElNo=0; ExtElNo<AmOfExtElem; ExtElNo++)
	{
		FillInTransPtrVectForElem(ExtElNo, 'E');
		radTg3d* ExtElPtr = g3dExternPtrVect[ExtElNo];

		for(int StrNo=0; StrNo<AmOfMainElem; StrNo++) 
		{
			InitObsPoiVect = MainTransPtrArray[StrNo]->TrPoint((g3dRelaxPtrVect[StrNo])->CentrPoint);
			TVector3d BufVect(0.,0.,0.);
			for(unsigned i=0; i<TransPtrVect.size(); i++)
			{
				TVector3d ObsPoiVect = TransPtrVect[i]->TrPoint_inv(InitObsPoiVect);
				radTField Field(FieldKeyExtern, CompCriterium, ObsPoiVect, ZeroVect, ZeroVect, ZeroVect, ZeroVect, 0.); // Improve
				ExtElPtr->B_comp(&Field);
				BufVect += TransPtrVect[i]->TrVectField(Field.H);
			}
			ExternFieldArray[StrNo] += MainTransPtrArray[StrNo]->TrVectField_inv(BufVect);
		}
		EmptyTransPtrVect();
	}
	//g3dExternPtrVect.erase(g3dExternPtrVect.begin(), g3dExternPtrVect.end()); //OC240408, to enable current scaling/update
}

//-------------------------------------------------------------------------

void radTInteraction::AddExternFieldFromMoreExtSource()
{
	if(MoreExtSourceHandle.rep == 0) return;

#ifdef RADIA_WITH_CUDA
	// GPU field eval for the frozen external source (rad.RlxPre srcobj). The
	// CPU loop below evaluates the source's field at every relaxable element
	// one point at a time via radTg3d::B_genComp; when the source is large
	// (e.g. a whole solved machine as the frozen background for a small
	// perturbative part) this dominates the setup. radGPU_ComputeField sums
	// the same source on the GPU. It returns B, which equals the H the CPU
	// path adds at points OUTSIDE the source material (in radia's Tesla
	// convention B = H + M, and M = 0 outside the source) -- true for the
	// frozen-background use case, where the relaxable elements are disjoint
	// from the source. Gated to gpu-assembly-on AND a single MPI rank:
	// radGPU_ComputeField MPI_Bcasts (collective), but this method runs on
	// rank 0 only, so calling it under nProc>=2 would deadlock -- multi-rank
	// keeps the CPU path.
	if(gUseGpuAsm && AmOfMainElem > 0 && m_nProcMPI < 2)
	{
		double* arObs = new double[3*AmOfMainElem];
		double* arB = new double[3*AmOfMainElem];
		for(int StrNo=0; StrNo<AmOfMainElem; StrNo++)
		{
			TVector3d P = MainTransPtrArray[StrNo]->TrPoint((g3dRelaxPtrVect[StrNo])->CentrPoint);
			arObs[3*StrNo] = P.x; arObs[3*StrNo+1] = P.y; arObs[3*StrNo+2] = P.z;
		}
		int rc = radGPU_ComputeFieldFromSrcRep((void*)(MoreExtSourceHandle.rep),
		                                       arObs, AmOfMainElem, arB, 1); // fp64
		bool gpuUsable = (rc == 0);

		// Guard 1: finiteness. This array seeds the relaxation residual, and
		// nothing downstream checks it -- a single NaN here surfaces only much
		// later as "radGPU_RelaxNK: non-finite residual at start".
		if(gpuUsable)
		{
			for(long i = 0; i < 3L*AmOfMainElem; i++)
			{
				if(!std::isfinite(arB[i]))
				{
					fprintf(stderr, "AddExternFieldFromMoreExtSource: GPU source field "
					        "returned a non-finite value; using the CPU field loop.\n");
					gpuUsable = false; break;
				}
			}
		}

		// Guard 2: verify the B == H invariant instead of assuming it. The GPU
		// returns B, the CPU adds H, and B = H + M -- equal ONLY where M = 0,
		// i.e. at points outside the source material. That holds for the
		// intended frozen-background use (relaxable part disjoint from the
		// source), but nothing enforces it: a relaxable element whose centroid
		// falls inside/on a source body would be seeded with a wrong, large H,
		// which the nonlinear material law can then drive to NaN. Spot-check a
		// sample against the CPU and fall back wholesale if they disagree.
		if(gpuUsable)
		{
			radTFieldKey FldKeyChk; FldKeyChk.H_=1;
			TVector3d ZeroChk(0.,0.,0.);
			int nChk = (AmOfMainElem < 16)? AmOfMainElem : 16;
			int stepChk = AmOfMainElem/nChk; if(stepChk < 1) stepChk = 1;
			double maxAbsH = 0., maxDiff = 0.;

			for(int c = 0; c < nChk; c++)
			{
				int StrNo = c*stepChk;
				if(StrNo >= AmOfMainElem) break;
				TVector3d ObsChk(arObs[3*StrNo], arObs[3*StrNo+1], arObs[3*StrNo+2]);
				radTField FldChk(FldKeyChk, CompCriterium, ObsChk, ZeroChk, ZeroChk, ZeroChk, ZeroChk, 0.);
				((radTg3d*)(MoreExtSourceHandle.rep))->B_genComp(&FldChk);

				TVector3d Hcpu = FldChk.H;
				TVector3d Bgpu(arB[3*StrNo], arB[3*StrNo+1], arB[3*StrNo+2]);
				TVector3d dVec = Hcpu - Bgpu;
				double aH = Hcpu.Abs(), aD = dVec.Abs();
				if(aH > maxAbsH) maxAbsH = aH;
				if(aD > maxDiff)  maxDiff = aD;
			}

			double tolChk = 1.e-6*maxAbsH + 1.e-12;
			if(maxDiff > tolChk)
			{
				fprintf(stderr, "AddExternFieldFromMoreExtSource: GPU source field (B) "
				        "disagrees with the CPU field (H) by %.3e (tol %.3e) -- the "
				        "relaxable elements are not disjoint from the frozen source, "
				        "so B != H. Using the CPU field loop.\n", maxDiff, tolChk);
				gpuUsable = false;
			}
		}

		if(gpuUsable)
		{
			for(int StrNo=0; StrNo<AmOfMainElem; StrNo++)
			{
				TVector3d Blab(arB[3*StrNo], arB[3*StrNo+1], arB[3*StrNo+2]);
				ExternFieldArray[StrNo] += MainTransPtrArray[StrNo]->TrVectField_inv(Blab);
			}
			delete[] arObs; delete[] arB;
			return; // GPU path succeeded and was verified
		}
		delete[] arObs; delete[] arB;
		// GPU unavailable / unsupported / unverified -> CPU fallback below
	}
#endif

	radTFieldKey FieldKeyExtern; FieldKeyExtern.H_=1;
	TVector3d ZeroVect(0.,0.,0.), InitObsPoiVect(0.,0.,0.);

	for(int StrNo=0; StrNo<AmOfMainElem; StrNo++)
	{
		InitObsPoiVect = MainTransPtrArray[StrNo]->TrPoint((g3dRelaxPtrVect[StrNo])->CentrPoint);
		radTField Field(FieldKeyExtern, CompCriterium, InitObsPoiVect, ZeroVect, ZeroVect, ZeroVect, ZeroVect, 0.); // Improve

		((radTg3d*)(MoreExtSourceHandle.rep))->B_genComp(&Field);

		ExternFieldArray[StrNo] += MainTransPtrArray[StrNo]->TrVectField_inv(Field.H);
	}
}

//-------------------------------------------------------------------------

void radTInteraction::AddMoreExternField(const radThg& hExtraExtSrc)
{
	if(hExtraExtSrc.rep == 0) return;

	radTg3d* pExtraExtSrc = (radTg3d*)(hExtraExtSrc.rep);

	radTFieldKey FieldKeyExtern; FieldKeyExtern.H_=1;
	TVector3d ZeroVect(0.,0.,0.), InitObsPoiVect(0.,0.,0.);

	for(int StrNo=0; StrNo<AmOfMainElem; StrNo++) 
	{
		radTrans* aTransPtr = MainTransPtrArray[StrNo];
        InitObsPoiVect = MainTransPtrArray[StrNo]->TrPoint((g3dRelaxPtrVect[StrNo])->CentrPoint);

        radTField Field(FieldKeyExtern, CompCriterium, InitObsPoiVect, ZeroVect, ZeroVect, ZeroVect, ZeroVect, 0.); // Improve
        pExtraExtSrc->B_genComp(&Field);

        ExternFieldArray[StrNo] += MainTransPtrArray[StrNo]->TrVectField_inv(Field.H);
	}
}

//-------------------------------------------------------------------------

void radTInteraction::ZeroAuxOldArrays()
{
	if(AmOfMainElem <= 0) return;

	if(AuxOldMagnArray != NULL)
	{
		TVector3d *tAuxOldMagn = AuxOldMagnArray;
		for(int StrNo=0; StrNo<AmOfMainElem; StrNo++) 
		{
			tAuxOldMagn->x = 0;
			tAuxOldMagn->y = 0;
			(tAuxOldMagn++)->z = 0;
		}
	}
	if(AuxOldFieldArray != NULL)
	{
		TVector3d *tAuxOldField = AuxOldFieldArray;
		for(int StrNo=0; StrNo<AmOfMainElem; StrNo++) 
		{
			tAuxOldField->x = 0;
			tAuxOldField->y = 0;
			(tAuxOldField++)->z = 0;
		}
	}
}

//-------------------------------------------------------------------------

void radTInteraction::SubstractOldMagn()
{
	if((AuxOldMagnArray == NULL) || (AmOfMainElem <= 0)) return;

	TVector3d *tAuxOldMagn = AuxOldMagnArray;
	for(int StNo=0; StNo<AmOfMainElem; StNo++)
	{
		TVector3d &M = (g3dRelaxPtrVect[StNo])->Magn;
		M -= *(tAuxOldMagn++); 
    }
}

//-------------------------------------------------------------------------

void radTInteraction::AddOldMagn()
{
	if((AuxOldMagnArray == NULL) || (AmOfMainElem <= 0)) return;

	TVector3d *tAuxOldMagn = AuxOldMagnArray;
	for(int StNo=0; StNo<AmOfMainElem; StNo++)
	{
		TVector3d &M = (g3dRelaxPtrVect[StNo])->Magn;
		M += *(tAuxOldMagn++); 
    }
}

//-------------------------------------------------------------------------

double radTInteraction::CalcQuadNewOldMagnDif()
{
	if((AuxOldMagnArray == NULL) || (AmOfMainElem <= 0)) return 0;

	double SumE2 = 0;
	TVector3d *tAuxOldMagn = AuxOldMagnArray;
	for(int StNo=0; StNo<AmOfMainElem; StNo++)
	{
		TVector3d CurDifM = (g3dRelaxPtrVect[StNo])->Magn - *(tAuxOldMagn++); 
		SumE2 += CurDifM.AmpE2(); //CurDifM*CurDifM;
    }
	return SumE2;
}

//-------------------------------------------------------------------------

void radTInteraction::FindMaxModMandH(double& MaxModM, double& MaxModH)
{
	double BufMaxModMe2, BufMaxModHe2, TestBufMaxModMe2, TestBufMaxModHe2;
	BufMaxModMe2 = BufMaxModHe2 = TestBufMaxModMe2 = TestBufMaxModHe2 = 1.E-17;

	for(int i=0; i<AmOfMainElem; i++)
	{
		TVector3d &NewMagn = NewMagnArray[i];
        TestBufMaxModMe2 = NewMagn.x*NewMagn.x + NewMagn.y*NewMagn.y + NewMagn.z*NewMagn.z;
        if(BufMaxModMe2 < TestBufMaxModMe2) BufMaxModMe2 = TestBufMaxModMe2;

		TVector3d &NewField = NewFieldArray[i];
		TestBufMaxModHe2 = NewField.x*NewField.x + NewField.y*NewField.y + NewField.z*NewField.z;
        if(BufMaxModHe2 < TestBufMaxModHe2) BufMaxModHe2 = TestBufMaxModHe2;
	}
	MaxModM = sqrt(BufMaxModMe2);
	MaxModH = sqrt(BufMaxModHe2);
}

//-------------------------------------------------------------------------

void radTInteraction::DumpBinVectOfPtrToListsOfTransPtr(CAuxBinStrVect& oStr, radVectPtr_lphgPtr& VectOfPtrToListsOfTransPtr, map<int, radTHandle<radTg>, less<int> >& gMapOfHandlers)
{
	int sizeVectOfPtrToListsOfTransPtr = (int)VectOfPtrToListsOfTransPtr.size();
	oStr << sizeVectOfPtrToListsOfTransPtr;
	for(int i=0; i<sizeVectOfPtrToListsOfTransPtr; i++)
	{
		radTlphgPtr* curListOfElemTransPtrPtr = VectOfPtrToListsOfTransPtr[i];
		int size_curListOfElemTransPtr = 0;
		if(curListOfElemTransPtrPtr != 0) size_curListOfElemTransPtr = (int)curListOfElemTransPtrPtr->size();
		
		oStr << size_curListOfElemTransPtr;
		if(size_curListOfElemTransPtr > 0)
		{
			for(radTlphgPtr::iterator TrIter = curListOfElemTransPtrPtr->begin();	TrIter != curListOfElemTransPtrPtr->end(); ++TrIter)
			{
				radTPair_int_hg *p_m_hg = *TrIter;
				//int mult = 0;

				if(p_m_hg != 0) 
				{
					int mult = p_m_hg->m;
					radThg &hg = p_m_hg->Handler_g;

					int existKey = 0;
					for(radTmhg::iterator mit = gMapOfHandlers.begin(); mit != gMapOfHandlers.end(); ++mit)
					{
						if(mit->second == hg) { existKey = mit->first; break;}
					}
					oStr << mult;
					oStr << existKey;
				}
				else oStr << (int)0;
			}
		}
	}
}

//-------------------------------------------------------------------------

void radTInteraction::DumpBin(CAuxBinStrVect& oStr, vector<int>& vElemKeysOut, map<int, radTHandle<radTg>, less<int> >& gMapOfHandlers, int& gUniqueMapKey, int elemKey)
{
	//radThg SourceHandle;
	int existKeySource = 0;
	if(SourceHandle.rep != 0)
	{
		//oStr << (char)1;
		//int existKey = 0;
		//const radThg &cur_hg = iter->second;
		for(radTmhg::iterator mit = gMapOfHandlers.begin(); mit != gMapOfHandlers.end(); ++mit)
		{
			if(mit->second == SourceHandle) { existKeySource = mit->first; break;}
		}
		if(existKeySource == 0)
		{
			existKeySource = gUniqueMapKey; 
			gMapOfHandlers[gUniqueMapKey++] = SourceHandle;
		}
		int indExist = CAuxParse::FindElemInd(existKeySource, vElemKeysOut);
		if(indExist < 0) SourceHandle.rep->DumpBin(oStr, vElemKeysOut, gMapOfHandlers, gUniqueMapKey, existKeySource);
	}
	//else oStr << (char)0;

	//radThg MoreExtSourceHandle;
	int existKeyMoreExtSource = 0;
	if(MoreExtSourceHandle.rep != 0)
	{
		//oStr << (char)1;
		//int existKey = 0;
		//const radThg &cur_hg = iter->second;
		for(radTmhg::iterator mit = gMapOfHandlers.begin(); mit != gMapOfHandlers.end(); ++mit)
		{
			if(mit->second == MoreExtSourceHandle) { existKeyMoreExtSource = mit->first; break;}
		}
		if(existKeyMoreExtSource == 0)
		{
			existKeyMoreExtSource = gUniqueMapKey; 
			gMapOfHandlers[gUniqueMapKey++] = MoreExtSourceHandle;
		}
		int indExist = CAuxParse::FindElemInd(existKeyMoreExtSource, vElemKeysOut);
		if(indExist < 0) MoreExtSourceHandle.rep->DumpBin(oStr, vElemKeysOut, gMapOfHandlers, gUniqueMapKey, existKeyMoreExtSource);
	}
	//else oStr << (char)0;

	//radTVectPtrg3dRelax g3dRelaxPtrVect;
	vector<int> vInd_g3dRelax;
	int size_g3dRelaxPtrVect = (int)g3dRelaxPtrVect.size();
	//oStr << size_g3dRelaxPtrVect;
	for(int i=0; i<size_g3dRelaxPtrVect; i++)
	{
		radTg3dRelax *p_g3dRelax = g3dRelaxPtrVect[i];
		if(p_g3dRelax != 0)
		{
			radTg *p_g = (radTg*)p_g3dRelax;
			//try to find element in the global map by pointer
			int oldKey = 0;
			for(radTmhg::iterator mit = gMapOfHandlers.begin(); mit != gMapOfHandlers.end(); ++mit)
			{
				if(mit->second.rep == p_g) { oldKey = mit->first; break;}
			}
			if(oldKey == 0)
			{
				oldKey = gUniqueMapKey;
				radThg hg(p_g3dRelax);
				gMapOfHandlers[gUniqueMapKey++] = hg;
			}
			int indExist = CAuxParse::FindElemInd(oldKey, vElemKeysOut);
			if(indExist < 0) p_g3dRelax->DumpBin(oStr, vElemKeysOut, gMapOfHandlers, gUniqueMapKey, oldKey);

			vInd_g3dRelax.push_back(oldKey);
		}
	}

	//radTVectPtr_g3d g3dExternPtrVect;
	vector<int> vInd_g3dExternPtrVect;
	int size_g3dExternPtrVect = (int)g3dExternPtrVect.size();
	for(int i=0; i<size_g3dExternPtrVect; i++)
	{
		radTg3d *p_g3d = g3dExternPtrVect[i];
		if(p_g3d != 0)
		{
			radTg *p_g = (radTg*)p_g3d;

			//try to find this element in the global map by pointer
			int oldKey = 0;
			for(radTmhg::iterator mit = gMapOfHandlers.begin(); mit != gMapOfHandlers.end(); ++mit)
			{
				if(mit->second.rep == p_g) { oldKey = mit->first; break;}
			}
			if(oldKey == 0)
			{
				oldKey = gUniqueMapKey;
				radThg hg(p_g3d);
				gMapOfHandlers[gUniqueMapKey++] = hg;
			}
			int indExist = CAuxParse::FindElemInd(oldKey, vElemKeysOut);
			if(indExist < 0) p_g3d->DumpBin(oStr, vElemKeysOut, gMapOfHandlers, gUniqueMapKey, oldKey);

			vInd_g3dExternPtrVect.push_back(oldKey);
		}
	}

	//radTVectPtrTrans TransPtrVect; //not required?
	vector<int> vIndTransPtrVect;
	int size_TransPtrVect = (int)TransPtrVect.size();
	for(int i=0; i<size_TransPtrVect; i++)
	{
		radTrans *pTrans = TransPtrVect[i];
		if(pTrans != 0)
		{
			if(Cast.IdentTransCast(pTrans))
			{
				vIndTransPtrVect.push_back(-1); //indicator of IdentTrans
			}
			else
			{
				radTrans *pTransCopy = new radTrans(*pTrans);

				radThg hg(pTransCopy);
				int oldKey = gUniqueMapKey;
				gMapOfHandlers[gUniqueMapKey++] = hg;
				
				pTransCopy->DumpBin(oStr, vElemKeysOut, gMapOfHandlers, gUniqueMapKey, oldKey);
				vIndTransPtrVect.push_back(oldKey);
			}
		}
		else vIndTransPtrVect.push_back(0);
	}

	//radTrans** MainTransPtrArray; //required
	vector<int> vIndMainTrans;
	if(mKeepTransData && (MainTransPtrArray != 0))
	{
		for(int i=0; i<AmOfMainElem; i++)
		{
			radTrans *pTrans = MainTransPtrArray[i];
			if(pTrans != 0)
			{
				if(Cast.IdentTransCast(pTrans))
				{
					vIndTransPtrVect.push_back(-1); //indicator of IdentTrans
				}
				else
				{
					radTrans *pTransCopy = new radTrans(*pTrans);

					radThg hg(pTransCopy);
					int oldKey = gUniqueMapKey;
					gMapOfHandlers[gUniqueMapKey++] = hg;
				
					pTransCopy->DumpBin(oStr, vElemKeysOut, gMapOfHandlers, gUniqueMapKey, oldKey);
					vIndMainTrans.push_back(oldKey);
				}
			}
			else vIndMainTrans.push_back(0);
		}
	}

	vElemKeysOut.push_back(elemKey);
	oStr << elemKey;

	//Next 5 bytes define/encode element type:
	oStr << (char)Type_g();
	oStr << (char)0;
	oStr << (char)0;
	oStr << (char)0;
	oStr << (char)0;

	//int AmOfMainElem;
	oStr << AmOfMainElem;

	//int AmOfExtElem;
	oStr << AmOfExtElem;

	//radThg SourceHandle;
	oStr << existKeySource;

	//radThg MoreExtSourceHandle;
	oStr << existKeyMoreExtSource;

	//radTVectPtrg3dRelax g3dRelaxPtrVect;
	int size_vInd_g3dRelax = (int)vInd_g3dRelax.size();
	oStr << size_vInd_g3dRelax;
	for(int i=0; i<size_vInd_g3dRelax; i++) oStr << vInd_g3dRelax[i];

	//radTVectPtr_g3d g3dExternPtrVect;
	int size_vInd_g3dExternPtrVect = (int)vInd_g3dExternPtrVect.size();
	oStr << size_vInd_g3dExternPtrVect;
	for(int i=0; i<size_vInd_g3dExternPtrVect; i++) oStr << vInd_g3dExternPtrVect[i];

	//radTVectPtrTrans TransPtrVect; //not required?
	int size_vIndTransPtrVect = (int)vIndTransPtrVect.size();
	oStr << size_vIndTransPtrVect;
	for(int i=0; i<size_vIndTransPtrVect; i++) oStr << vIndTransPtrVect[i];

	//radTCompCriterium CompCriterium;
	//short BasedOnPrecLevel; // Actually this is used nowhere at the moment
	oStr << CompCriterium.BasedOnPrecLevel;
	//double AbsPrecB;
	oStr << CompCriterium.AbsPrecB;
	//double AbsPrecA;
	oStr << CompCriterium.AbsPrecA;
	//double AbsPrecB_int;
	oStr << CompCriterium.AbsPrecB_int;
	//double AbsPrecForce;
	oStr << CompCriterium.AbsPrecForce;
	//double AbsPrecTorque;
	oStr << CompCriterium.AbsPrecTorque;
	//double AbsPrecEnergy;
	oStr << CompCriterium.AbsPrecTorque;
	//double AbsPrecTrjCoord;
	oStr << CompCriterium.AbsPrecTrjCoord;
	//double AbsPrecTrjAngle;
	oStr << CompCriterium.AbsPrecTrjAngle;
	//double MltplThresh[4]; // Threshold ratios for 4 diff. orders of multipole approx. at field computation
	oStr << CompCriterium.MltplThresh[0] << CompCriterium.MltplThresh[1] << CompCriterium.MltplThresh[2] << CompCriterium.MltplThresh[3];
	//double WorstRelPrec;
	oStr << CompCriterium.WorstRelPrec;
	//char BasedOnWorstRelPrec; // Used at energy - force computation
	oStr << CompCriterium.BasedOnWorstRelPrec;

	//radTRelaxStatusParam RelaxStatusParam;
	//double MisfitM, MaxModM, MaxModH;
	oStr << RelaxStatusParam.MisfitM;
	oStr << RelaxStatusParam.MaxModM;
	oStr << RelaxStatusParam.MaxModH;

	//short RelaxationStarted;
	oStr << RelaxationStarted;

	//TMatrix3df** InteractMatrix; //OC250504
	////TMatrix3d** InteractMatrix; //OC250504
	if(InteractMatrix != NULL)
	{
		oStr << (char)1;
		for(int i=0; i<AmOfMainElem; i++)
		{
			TMatrix3df *pLineInteractMatrix = InteractMatrix[i];
			if(pLineInteractMatrix != NULL)
			{
				oStr << (char)1;
				for(int j=0; j<AmOfMainElem; j++)
				{
					oStr << pLineInteractMatrix[j];
				}
			}
			else oStr << (char)0;
		}
	}
	else oStr << (char)0;

	//TVector3d* ExternFieldArray;
	if(ExternFieldArray != NULL)
	{
		oStr << (char)1;
		for(int i=0; i<AmOfMainElem; i++) oStr << ExternFieldArray[i];
	}
	else oStr << (char)0;

	//TVector3d* NewMagnArray;
	if(NewMagnArray != NULL)
	{
		oStr << (char)1;
		for(int i=0; i<AmOfMainElem; i++) oStr << NewMagnArray[i];
	}
	else oStr << (char)0;

	//TVector3d* NewFieldArray;
	if(NewFieldArray != NULL)
	{
		oStr << (char)1;
		for(int i=0; i<AmOfMainElem; i++) oStr << NewFieldArray[i];
	}
	else oStr << (char)0;

	//TVector3d* AuxOldMagnArray;
	if(AuxOldMagnArray != NULL)
	{
		oStr << (char)1;
		for(int i=0; i<AmOfMainElem; i++) oStr << AuxOldMagnArray[i];
	}
	else oStr << (char)0;

	//TVector3d* AuxOldFieldArray;
	if(AuxOldFieldArray != NULL)
	{
		oStr << (char)1;
		for(int i=0; i<AmOfMainElem; i++) oStr << AuxOldFieldArray[i];
	}
	else oStr << (char)0;

	//radTVectRelaxSubInterval RelaxSubIntervConstrVect; // New
	int sizeRelaxSubIntervConstrVect = (int)RelaxSubIntervConstrVect.size();
	oStr << sizeRelaxSubIntervConstrVect;	
	if(sizeRelaxSubIntervConstrVect > 0)
	{
		for(int i=0; i<sizeRelaxSubIntervConstrVect; i++)
		{
			radTRelaxSubInterval &relaxSubInterval = RelaxSubIntervConstrVect[i];
			oStr << relaxSubInterval.StartNo;
			oStr << relaxSubInterval.FinNo;
			oStr << (int)(relaxSubInterval.SubIntervalID);
		}

		//radTRelaxSubInterval* RelaxSubIntervArray; // New 
		if(RelaxSubIntervArray != NULL)
		{
			int MaxSubIntervArraySize = 2*sizeRelaxSubIntervConstrVect + 1;
			oStr << (int)MaxSubIntervArraySize;
			radTRelaxSubInterval *t_RelaxSubIntervArray = RelaxSubIntervArray;
			for(int i=0; i<MaxSubIntervArraySize; i++)
			{
				oStr << (t_RelaxSubIntervArray->StartNo);
				oStr << (t_RelaxSubIntervArray->FinNo);
				oStr << (int)(t_RelaxSubIntervArray->SubIntervalID);
				t_RelaxSubIntervArray++;
			}
		}
		else oStr << (int)0;
	}

	//radVectPtr_lphgPtr IntVectOfPtrToListsOfTransPtr; //required
	DumpBinVectOfPtrToListsOfTransPtr(oStr, IntVectOfPtrToListsOfTransPtr, gMapOfHandlers);

	//radVectPtr_lphgPtr ExtVectOfPtrToListsOfTransPtr; //required
	DumpBinVectOfPtrToListsOfTransPtr(oStr, ExtVectOfPtrToListsOfTransPtr, gMapOfHandlers);

	//radIdentTrans* IdentTransPtr; //required, but doesn't need to be saved
	//radTCast Cast; //no members?
	//radTSend Send; //no members?

	//short FillInMainTransOnly;
	oStr << FillInMainTransOnly;

	//char mKeepTransData;
	oStr << mKeepTransData;

	//radTrans** MainTransPtrArray; //required
	int size_vIndMainTrans = (int)vIndMainTrans.size();
	oStr << size_vIndMainTrans;
	for(int k=0; k<size_vIndMainTrans; k++) oStr << vIndMainTrans[k];
	
	//int AmOfRelaxSubInterv;
	oStr << AmOfRelaxSubInterv;

	//short SomethingIsWrong;
	oStr << SomethingIsWrong;

	//short MemAllocTotAtOnce;
	oStr << MemAllocTotAtOnce;
}

//-------------------------------------------------------------------------

//void radTInteraction::DumpBinParseSourceHandle(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers, bool do_g3dCast, bool do_g3dRelaxCast, radThg& out_hg)
int radTInteraction::DumpBinParseSourceHandle(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers, bool do_g3dCast, bool do_g3dRelaxCast, radThg& out_hg)
{//move to g3d?
	int oldKey = 0;
	inStr >> oldKey;
	if(oldKey > 0)
	{
		map<int, int>::const_iterator itKey = mKeysOldNew.find(oldKey);
		if(itKey != mKeysOldNew.end())
		{
			int newKey = itKey->second;
			if(newKey > 0)
			{
				radTmhg::const_iterator iter = gMapOfHandlers.find(newKey);
				if(iter != gMapOfHandlers.end())
				{
					radThg hg = (*iter).second;
					if(hg.rep != 0)
					{
						if(do_g3dCast || do_g3dRelaxCast)
						{
							radTg3d *g3dPtr = radTCast::g3dCast(hg.rep);
							if(g3dPtr != 0)
							{
								if(do_g3dRelaxCast)
								{
									if(radTCast::g3dRelaxCast(g3dPtr) != 0) out_hg = hg;
								}
								else out_hg = hg;
							}
						}
						else out_hg = hg;
					}
				}
			}
		}
	}
	return oldKey;
}

//-------------------------------------------------------------------------

void radTInteraction::DumpBinParseVectOfPtrToListsOfTransPtr(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers, radVectPtr_lphgPtr& VectOfPtrToListsOfTransPtr)
{
	int sizeVectOfPtrToListsOfTransPtr = 0;
	inStr >> sizeVectOfPtrToListsOfTransPtr;

	for(int i=0; i<sizeVectOfPtrToListsOfTransPtr; i++)
	{
		int size_curListOfElemTransPtr = 0;
		inStr >> size_curListOfElemTransPtr;

		if(size_curListOfElemTransPtr > 0)
		{
			radTlphgPtr *pCurListOfElemTransPtr = new radTlphgPtr();
			for(int j=0; j<size_curListOfElemTransPtr; j++)
			{
				int mult = 0;
				inStr >> mult;
				if(mult > 0)
				{
					radThg hg;
					DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, false, false, hg);
					pCurListOfElemTransPtr->push_back(new radTPair_int_hg(mult, hg));
				}
			}
			VectOfPtrToListsOfTransPtr.push_back(pCurListOfElemTransPtr);
		}
	}
}

//-------------------------------------------------------------------------

radTInteraction::radTInteraction(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers)
{
	//radIdentTrans* IdentTransPtr; //required
	IdentTransPtr = new radIdentTrans();

	//int AmOfMainElem;
	inStr >> AmOfMainElem;

	//int AmOfExtElem;
	inStr >> AmOfExtElem;

	//radThg SourceHandle;
	DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, true, false, SourceHandle);

	//radThg MoreExtSourceHandle;
	DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, true, false, MoreExtSourceHandle);

	//radTVectPtrg3dRelax g3dRelaxPtrVect;
	int size_g3dRelaxPtrVect = 0;
	inStr >> size_g3dRelaxPtrVect;
	if(g3dRelaxPtrVect.size() > 0) g3dRelaxPtrVect.erase(g3dRelaxPtrVect.begin(), g3dRelaxPtrVect.end()); //?
	for(int i=0; i<size_g3dRelaxPtrVect; i++)
	{
		radThg hg;
		DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, true, true, hg);
		if(hg.rep != 0) g3dRelaxPtrVect.push_back((radTg3dRelax*)((radTg3d*)hg.rep));
	}

	//radTVectPtr_g3d g3dExternPtrVect;
	int size_g3dExternPtrVect = 0;
	inStr >> size_g3dExternPtrVect;
	if(g3dExternPtrVect.size() > 0) g3dExternPtrVect.erase(g3dExternPtrVect.begin(), g3dExternPtrVect.end()); //?
	for(int i=0; i<size_g3dExternPtrVect; i++)
	{
		radThg hg;
		DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, true, false, hg);
		if(hg.rep != 0) g3dExternPtrVect.push_back((radTg3d*)hg.rep);
	}

	//radTVectPtrTrans TransPtrVect; //not required?
	int sizeTransPtrVect = 0;
	inStr >> sizeTransPtrVect;
	if(TransPtrVect.size() > 0) TransPtrVect.erase(TransPtrVect.begin(), TransPtrVect.end()); //?
	for(int i=0; i<sizeTransPtrVect; i++)
	{
		radThg hg;
		int oldKey = DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, false, false, hg);
		if(oldKey < 0) TransPtrVect.push_back(IdentTransPtr);
		else if(hg.rep != 0) TransPtrVect.push_back(new radTrans(*((radTrans*)hg.rep))); //will be deleted at distraction
	}

	//radTCompCriterium CompCriterium;
	//short BasedOnPrecLevel; // Actually this is used nowhere at the moment
	inStr >> CompCriterium.BasedOnPrecLevel;
	//double AbsPrecB;
	inStr >> CompCriterium.AbsPrecB;
	//double AbsPrecA;
	inStr >> CompCriterium.AbsPrecA;
	//double AbsPrecB_int;
	inStr >> CompCriterium.AbsPrecB_int;
	//double AbsPrecForce;
	inStr >> CompCriterium.AbsPrecForce;
	//double AbsPrecTorque;
	inStr >> CompCriterium.AbsPrecTorque;
	//double AbsPrecEnergy;
	inStr >> CompCriterium.AbsPrecTorque;
	//double AbsPrecTrjCoord;
	inStr >> CompCriterium.AbsPrecTrjCoord;
	//double AbsPrecTrjAngle;
	inStr >> CompCriterium.AbsPrecTrjAngle;
	//double MltplThresh[4]; // Threshold ratios for 4 diff. orders of multipole approx. at field computation
	inStr >> CompCriterium.MltplThresh[0];
	inStr >> CompCriterium.MltplThresh[1];
	inStr >> CompCriterium.MltplThresh[2];
	inStr >> CompCriterium.MltplThresh[3];
	//double WorstRelPrec;
	inStr >> CompCriterium.WorstRelPrec;
	//char BasedOnWorstRelPrec; // Used at energy - force computation
	inStr >> CompCriterium.BasedOnWorstRelPrec;

	//radTRelaxStatusParam RelaxStatusParam;
	//double MisfitM, MaxModM, MaxModH;
	inStr >> RelaxStatusParam.MisfitM;
	inStr >> RelaxStatusParam.MaxModM;
	inStr >> RelaxStatusParam.MaxModH;

	//short RelaxationStarted;
	inStr >> RelaxationStarted;

	//TMatrix3df** InteractMatrix;
	char matrixExists = 0;
	inStr >> matrixExists;
	if(matrixExists && (AmOfMainElem > 0))
	{
		InteractMatrix = new TMatrix3df*[AmOfMainElem];
		TMatrix3df **pLineInteractMatrix = InteractMatrix;

		for(int i=0; i<AmOfMainElem; i++)
		{
			char matrixRowExists = 0;
			*pLineInteractMatrix = NULL;

			inStr >> matrixRowExists;
			if(matrixRowExists)
			{
				*pLineInteractMatrix = new TMatrix3df[AmOfMainElem];
				TMatrix3df *tLine = *(pLineInteractMatrix++);
				for(int j=0; j<AmOfMainElem; j++)
				{
					inStr >> *(tLine++);
				}
			}
		}
	}

	//TVector3d* ExternFieldArray;
	char externFieldArrayExists = 0;
	ExternFieldArray = 0;
	inStr >> externFieldArrayExists;
	if(externFieldArrayExists && (AmOfMainElem > 0))
	{
		ExternFieldArray = new TVector3d[AmOfMainElem];
		for(int i=0; i<AmOfMainElem; i++) inStr >> ExternFieldArray[i];
	}

	//TVector3d* NewMagnArray;
	char newMagnArrayExists = 0;
	NewMagnArray = 0;
	inStr >> newMagnArrayExists;
	if(newMagnArrayExists && (AmOfMainElem > 0))
	{
		NewMagnArray = new TVector3d[AmOfMainElem];
		for(int i=0; i<AmOfMainElem; i++) inStr >> NewMagnArray[i];
	}

	//TVector3d* NewFieldArray;
	char newFieldArrayExists = 0;
	NewFieldArray = 0;
	inStr >> newFieldArrayExists;
	if(newFieldArrayExists && (AmOfMainElem > 0))
	{
		NewFieldArray = new TVector3d[AmOfMainElem];
		for(int i=0; i<AmOfMainElem; i++) inStr >> NewFieldArray[i];
	}

	//TVector3d* AuxOldMagnArray;
	char auxOldMagnArrayExists = 0;
	AuxOldMagnArray = 0;
	inStr >> auxOldMagnArrayExists;
	if(auxOldMagnArrayExists && (AmOfMainElem > 0))
	{
		AuxOldMagnArray = new TVector3d[AmOfMainElem];
		for(int i=0; i<AmOfMainElem; i++) inStr >> AuxOldMagnArray[i];
	}

	//TVector3d* AuxOldFieldArray;
	char auxOldFieldArrayExists = 0;
	AuxOldFieldArray = 0;
	inStr >> auxOldFieldArrayExists;
	if(auxOldFieldArrayExists && (AmOfMainElem > 0))
	{
		AuxOldFieldArray = new TVector3d[AmOfMainElem];
		for(int i=0; i<AmOfMainElem; i++) inStr >> AuxOldFieldArray[i];
	}

	//radTVectRelaxSubInterval RelaxSubIntervConstrVect; // New
	int sizeRelaxSubIntervConstrVect = 0;
	RelaxSubIntervArray = 0;
	inStr >> sizeRelaxSubIntervConstrVect;
	if(sizeRelaxSubIntervConstrVect > 0)
	{
		for(int i=0; i<sizeRelaxSubIntervConstrVect; i++)
		{
			radTRelaxSubInterval relaxSubInterval;
			inStr >> relaxSubInterval.StartNo;
			inStr >> relaxSubInterval.FinNo;
			int subIntervalID = 0;
			inStr >> subIntervalID;
			relaxSubInterval.SubIntervalID = (TRelaxSubIntervalID)subIntervalID;

			RelaxSubIntervConstrVect.push_back(relaxSubInterval);
		}

		//radTRelaxSubInterval* RelaxSubIntervArray; // New 
		int MaxSubIntervArraySize = 0;
		inStr >> MaxSubIntervArraySize;
		if(MaxSubIntervArraySize > 0)
		{
			RelaxSubIntervArray = new radTRelaxSubInterval[MaxSubIntervArraySize];
			radTRelaxSubInterval *t_RelaxSubIntervArray = RelaxSubIntervArray;
			for(int i=0; i<MaxSubIntervArraySize; i++)
			{
				inStr >> (t_RelaxSubIntervArray->StartNo);
				inStr >> (t_RelaxSubIntervArray->FinNo);
				int subIntervalID = 0;
				inStr >> subIntervalID;
				t_RelaxSubIntervArray->SubIntervalID = (TRelaxSubIntervalID)subIntervalID;
				t_RelaxSubIntervArray++;
			}
		}
	}

	//radVectPtr_lphgPtr IntVectOfPtrToListsOfTransPtr; //required
	DumpBinParseVectOfPtrToListsOfTransPtr(inStr, mKeysOldNew, gMapOfHandlers, IntVectOfPtrToListsOfTransPtr);

	//radVectPtr_lphgPtr ExtVectOfPtrToListsOfTransPtr; //required
	DumpBinParseVectOfPtrToListsOfTransPtr(inStr, mKeysOldNew, gMapOfHandlers, ExtVectOfPtrToListsOfTransPtr);

	//radTCast Cast; //no members?
	//radTSend Send; //no members?

	//short FillInMainTransOnly;
	inStr >> FillInMainTransOnly;

	//char mKeepTransData;
	inStr >> mKeepTransData;

	//radTrans** MainTransPtrArray; //required
	MainTransPtrArray= 0;
	int size_vIndMainTrans = 0;
	inStr >> size_vIndMainTrans;
	if(size_vIndMainTrans > 0)
	{
		MainTransPtrArray = new radTrans*[AmOfMainElem];

		for(int i=0; i<AmOfMainElem; i++)
		{
			radThg hg;
			int oldKey = DumpBinParseSourceHandle(inStr, mKeysOldNew, gMapOfHandlers, false, false, hg);
			if(oldKey < 0) MainTransPtrArray[i] = IdentTransPtr;
			else if(hg.rep != 0) MainTransPtrArray[i] = new radTrans(*((radTrans*)hg.rep)); //will be deleted at distraction
		}
	}

	//int AmOfRelaxSubInterv;
	inStr >> AmOfRelaxSubInterv;

	//short SomethingIsWrong;
	inStr >> SomethingIsWrong;

	//short MemAllocTotAtOnce;
	inStr >> MemAllocTotAtOnce;
}

//-------------------------------------------------------------------------
