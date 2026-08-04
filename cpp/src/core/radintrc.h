/*-------------------------------------------------------------------------
*
* File name:      radintrc.h
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

#ifndef __RADINTRC_H
#define __RADINTRC_H

#include "radcast.h"
#include "gmvectf.h"
//#include "radtrans.h"
#include "gmtrans.h"
#include "radg3d.h"

#include <sstream>
#include <vector>

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

typedef list <radTPair_int_hg*> radTlphgPtr;

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

struct radTRelaxStatusParam {
	double MisfitM, MaxModM, MaxModH;
	radTRelaxStatusParam(double InMisfitM =0., double InMaxModM =0., double InMaxModH =0.) 
	{ 
		MisfitM=InMisfitM; MaxModM=InMaxModM; MaxModH=InMaxModH;
	}
};

//-------------------------------------------------------------------------

enum TRelaxSubIntervalID { RelaxTogether, RelaxApart };

//-------------------------------------------------------------------------

struct radTRelaxSubInterval {
	int StartNo, FinNo;
	TRelaxSubIntervalID SubIntervalID;

	radTRelaxSubInterval(int InStartNo, int InFinNo, TRelaxSubIntervalID InSubIntervalID)
	{
		StartNo = InStartNo; FinNo = InFinNo; SubIntervalID = InSubIntervalID;
	}
	radTRelaxSubInterval() {}

	inline friend int operator <(const radTRelaxSubInterval&, const radTRelaxSubInterval&);
	inline friend int operator ==(const radTRelaxSubInterval&, const radTRelaxSubInterval&);
};

//-------------------------------------------------------------------------

inline int operator <(const radTRelaxSubInterval&, const radTRelaxSubInterval&) { return 1;}

//-------------------------------------------------------------------------

inline int operator ==(const radTRelaxSubInterval& i1, const radTRelaxSubInterval& i2) 
{ 
	return (i1.StartNo == i2.StartNo) && (i1.FinNo == i2.FinNo) && (i1.SubIntervalID == i2.SubIntervalID);
}

//-------------------------------------------------------------------------

typedef vector<radTg3dRelax*> radTVectPtrg3dRelax;
typedef vector<radTg3d*> radTVectPtr_g3d;
typedef vector<radTrans*> radTVectPtrTrans;
typedef vector<radTlphgPtr*> radVectPtr_lphgPtr;
typedef vector<radTRelaxSubInterval> radTVectRelaxSubInterval;

//-------------------------------------------------------------------------

class radTInteraction : public radTg {

#ifdef RADIA_WITH_CUDA
	friend int radGPU_PackInteractionData(radTInteraction*, struct RadGPURelaxData*, int);
	friend void radGPU_UnpackMagnetization(struct RadGPURelaxData*, radTInteraction*);
	friend int radGPU_AutoRelax(radTInteraction*, double, int, char, double);
	friend int radGPU_AutoRelaxNK(radTInteraction*, double, int, char, double);
	friend int radGPU_PackGeometryForAsm(radTInteraction*, struct RadGPU_PolyData*, struct RadGPU_RecMagData*, struct RadGPU_SymData*, struct RadGPU_ObsQuadData*);
	friend int radGPU_PackObsQuadForAsm(radTInteraction*, struct RadGPU_ObsQuadData*);
	friend void radGPU_UnpackMatrix(struct RadGPU_AsmResult*, radTInteraction*);

public:
	// Unique identity of THIS interaction matrix (assigned once per Setup);
	// keys the GPU-resident matrix cache so repeated RlxAuto calls on the
	// same matrix skip the flatten + device upload. 0 = not set up.
	unsigned long long mGpuMatrixStamp = 0;
private:
#endif

	int AmOfMainElem;
	int AmOfExtElem;
	radThg SourceHandle;
	radThg MoreExtSourceHandle;
	radTCompCriterium CompCriterium;
	radTRelaxStatusParam RelaxStatusParam;
	short RelaxationStarted;

	TMatrix3df** InteractMatrix; //OC250504
	//TMatrix3d** InteractMatrix; //OC250504

	TVector3d* ExternFieldArray;
	TVector3d* NewMagnArray;
	TVector3d* NewFieldArray;
	TVector3d* AuxOldMagnArray;
	TVector3d* AuxOldFieldArray;

	radTRelaxSubInterval* RelaxSubIntervArray; // New 
	radTVectPtrg3dRelax g3dRelaxPtrVect;
	radTVectPtr_g3d g3dExternPtrVect;
	radTVectPtrTrans TransPtrVect;
	radVectPtr_lphgPtr IntVectOfPtrToListsOfTransPtr;
	radVectPtr_lphgPtr ExtVectOfPtrToListsOfTransPtr;
	radTVectRelaxSubInterval RelaxSubIntervConstrVect; // New
	radTrans** MainTransPtrArray;

	radTCast Cast;
	radTSend Send;
	radIdentTrans* IdentTransPtr;
	short FillInMainTransOnly;
	char mKeepTransData;

	int m_rankMPI; //21122019 (to set from Application?)
	int m_nProcMPI;

	//RadiaCUDA: OPT-IN volume-averaged ("Galerkin") assembly, radgalerkin.h.
	//All of these stay EMPTY unless RADIA_GALERKIN=1, and the collocation code
	//paths are then untouched, so the flag being off is a true no-op.
	//Per row element: the base-rule and near-rule observation quadratures,
	//already transformed by MainTransPtrArray[i]; the transformed centroid and
	//the characteristic size h = V^(1/3) used for the near-band radius.
	vector<vector<TVector3d> > m_galQPts, m_galNPts;
	vector<vector<double> > m_galQWts, m_galNWts;
	vector<TVector3d> m_galCen;
	vector<double> m_galH;
	char m_galNearOn = 0;

	//Builds the caches above; returns 0 if any element type has no quadrature.
	int PrepGalerkinQuad();
	//Interaction block for (row StrNo, the column element whose symmetry copies
	//are already in TransPtrVect), volume-averaged over the row element.
	void GalerkinInteractBlock(int StrNo, int ColNo, radTg3dRelax* g3dRelaxPtrColNo,
		const radTFieldKey& FieldKeyInteract, int AmOfElemWithSym,
		TMatrix3d& SubMatrix);

public:

	int AmOfRelaxSubInterv;

	short SomethingIsWrong;
	short MemAllocTotAtOnce;

	// RadiaCUDA: GPU interaction-matrix assembly switch (default on). Set per
	// RlxPre call via rad.RlxPre(obj, use_gpu=...). Under MPI, rank 0
	// assembles on the GPU while the workers wait; with the flag off the
	// classic MPI-distributed CPU assembly is used.
	static char gUseGpuAsm;
	// Which backend serviced the most recent interaction-matrix assembly:
	// -1 = none yet, 0 = CPU, 1 = GPU. Diagnostic (rad.UtiAsmLastBackend()).
	static char gLastAsmBackend;
	// RadiaCUDA: what to do when the GPU cannot service the interaction matrix
	// (typically: the dense matrix does not fit in VRAM). Set globally via
	// rad.UtiGpuFallback(...). NOTHING ever changes this by itself -- a run
	// only leaves the GPU if the user said it may.
	//   0 = 'cpu'            fall back to the CPU assembly (default; legacy)
	//   1 = 'gpu_streaming'  keep the matrix in host RAM and stream it
	//   2 = 'break'          raise an error instead of silently going slow
	static char gGpuFallback;

	radTInteraction(const radThg&, const radThg&, const radTCompCriterium&, short =0, char =0, char =0, int =-1, int =0); //OC08012020
	//radTInteraction(const radThg&, const radThg&, const radTCompCriterium&, short =0, char =0, char =0);
	radTInteraction(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers);
	radTInteraction();
	~radTInteraction();

	int Setup(const radThg& In_hg, const radThg& In_hgMoreExtSrc, const radTCompCriterium& InCompCriterium, short InMemAllocTotAtOnce, char AuxOldMagnArrayIsNeeded, char KeepTransData, int rankMPI=-1, int nProcMPI=0); //OC08012020
	//int Setup(const radThg& In_hg, const radThg& In_hgMoreExtSrc, const radTCompCriterium& InCompCriterium, short InMemAllocTotAtOnce, char AuxOldMagnArrayIsNeeded, char KeepTransData);

	void CountMainRelaxElems(radTg3d*, radTlphgPtr*);
	void AllocateMemory(char ExtraExternFieldArrayIsNeeded);
	void DeallocateMemory(); //OC27122019

	int SetupInteractMatrix(); //OC26122019
	//void SetupInteractMatrix();

	void SetupExternFieldArray();
	void AddExternFieldFromMoreExtSource();
	void AddMoreExternField(const radThg& hExtraExtSrc);
	//void ZeroAuxOldMagnArray();
	//void StoreAuxOldMagnArray();
	void SubstractOldMagn();
    void AddOldMagn();
	double CalcQuadNewOldMagnDif();
	int CountRelaxElemsWithSym();
	int OutAmOfRelaxObjs() { return AmOfMainElem;}
	void FindMaxModMandH(double& MaxModM, double& MaxModH);

	inline void PushFrontNativeElemTransList(radTg3d*, radTlphgPtr*);
	inline void EmptyVectOfPtrToListsOfTrans();

	inline void FillInTransPtrVectForElem(int, char);
	inline void EmptyTransPtrVect();

	void NestedFor_Trans(radTrans*, const radTlphgPtr::const_iterator&, int, char);
	inline void AddTransOrNestedFor(radTrans*, const radTlphgPtr::const_iterator&, int, char);

	void FillInMainTransPtrArray();
	inline void DestroyMainTransPtrArray();

	void FillInRelaxSubIntervArray(); //New

	int NotEmpty() { return (AmOfMainElem==0)? 0 : 1;}
	inline void Dump(std::ostream&, int =0); // Porting
	void DumpBin(CAuxBinStrVect& oStr, vector<int>& vElemKeysOut, map<int, radTHandle<radTg>, less<int> >& gMapOfHandlers, int& gUniqueMapKey, int elemKey);
	void DumpBinVectOfPtrToListsOfTransPtr(CAuxBinStrVect& oStr, radVectPtr_lphgPtr& VectOfPtrToListsOfTransPtr, map<int, radTHandle<radTg>, less<int> >& gMapOfHandlers);
	int DumpBinParseSourceHandle(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers, bool do_g3dCast, bool do_g3dRelaxCast, radThg& out_hg);
	void DumpBinParseVectOfPtrToListsOfTransPtr(CAuxBinStrVect& inStr, map<int, int>& mKeysOldNew, radTmhg& gMapOfHandlers, radVectPtr_lphgPtr& VectOfPtrToListsOfTransPtr);

	int Type_g() { return 4;}

	inline void ResetM();
	inline void ResetAuxParam();
	inline void InitAuxArrays();

	void ZeroAuxOldArrays(); //OC300504
	inline void StoreAuxOldArrays(); //OC300504
    inline void RestoreAuxOldArrays(); //OC300504

	inline void OutRelaxStatusParam(double*);
	inline void ShowInteractVector(char);
	inline void ShowInteractMatrix();

	inline int SizeOfThis();

	inline void UpdateExternalField();

	inline void LongLongToFloatAr(long long, float*); //OC24122019
	inline long long FloatArToLongLong(float*); //OC27122019

	template<class T> inline long OutMagnVals(T*& arMagnVals, int nExtraVals=0); //OC02012020
	template<class T> inline void SetRelaxObjMagnVals(T* arMagnVals); //OC02012020

	friend class radTIterativeRelaxMeth;
	friend class radTSimpleRelaxation;
	friend class radTRelaxationMethNo_2;
	friend class radTRelaxationMethNo_3;
	friend class radTRelaxationMethNo_4;
	friend class radTRelaxationMethNo_a5;
	friend class radTRelaxationMethNo_7;
	friend class radTRelaxationMethNo_8;
	friend class radTRelaxationMethNo_10;
};

//-------------------------------------------------------------------------

inline void radTInteraction::PushFrontNativeElemTransList(radTg3d* g3dPtr, radTlphgPtr* ListOfPtrToTransPtr)
{
	for(radTlphg::iterator TrIter = g3dPtr->g3dListOfTransform.begin();	
		TrIter != g3dPtr->g3dListOfTransform.end(); ++TrIter)
		ListOfPtrToTransPtr->push_back(&(*TrIter)); // Improve dereferentiation?
}

//-------------------------------------------------------------------------

inline void radTInteraction::EmptyVectOfPtrToListsOfTrans()
{
	for(unsigned i=1; i<IntVectOfPtrToListsOfTransPtr.size(); i++)
	//for(unsigned i=0; i<IntVectOfPtrToListsOfTransPtr.size(); i++) //OC30122019: this correction was suggested by per-gron
	{
		radTlphgPtr*& p_lphgPtr = IntVectOfPtrToListsOfTransPtr[i];
		if(p_lphgPtr != 0) delete p_lphgPtr;
		p_lphgPtr = 0;
	}
	IntVectOfPtrToListsOfTransPtr.erase(IntVectOfPtrToListsOfTransPtr.begin(), IntVectOfPtrToListsOfTransPtr.end());
	for(unsigned k=1; k<ExtVectOfPtrToListsOfTransPtr.size(); k++) 
	//for(unsigned k=0; k<ExtVectOfPtrToListsOfTransPtr.size(); k++) //OC30122019: this correction was suggested by per-gron
	{
		radTlphgPtr*& p_lphgPtr = ExtVectOfPtrToListsOfTransPtr[k];
		if(p_lphgPtr != 0) delete p_lphgPtr;
		p_lphgPtr = 0;
	}
	ExtVectOfPtrToListsOfTransPtr.erase(ExtVectOfPtrToListsOfTransPtr.begin(), ExtVectOfPtrToListsOfTransPtr.end());
}

//-------------------------------------------------------------------------

inline void radTInteraction::FillInTransPtrVectForElem(int ElemLocInd, char I_or_E)
{
	radTlphgPtr* PtrToListOfPtrToTrans = NULL;
	if(I_or_E == 'I') PtrToListOfPtrToTrans = IntVectOfPtrToListsOfTransPtr[ElemLocInd];
	else PtrToListOfPtrToTrans = ExtVectOfPtrToListsOfTransPtr[ElemLocInd];

	if(PtrToListOfPtrToTrans->empty()) TransPtrVect.push_back(IdentTransPtr);
	else NestedFor_Trans(IdentTransPtr, PtrToListOfPtrToTrans->begin(), ElemLocInd, I_or_E);
}

//-------------------------------------------------------------------------

inline void radTInteraction::EmptyTransPtrVect()
{
	if(Cast.IdentTransCast(TransPtrVect[0])==0) delete TransPtrVect[0];
	for(unsigned i=1; i<TransPtrVect.size(); i++) delete TransPtrVect[i];
	TransPtrVect.erase(TransPtrVect.begin(), TransPtrVect.end());
}

//-------------------------------------------------------------------------

inline void radTInteraction::AddTransOrNestedFor(radTrans* BaseTransPtr, const radTlphgPtr::const_iterator& Iter, int ElemLocInd, char I_or_E)
{
	radTlphgPtr* PtrToListOfPtrToTrans = NULL;
	if(I_or_E == 'I') PtrToListOfPtrToTrans = IntVectOfPtrToListsOfTransPtr[ElemLocInd];
	else PtrToListOfPtrToTrans = ExtVectOfPtrToListsOfTransPtr[ElemLocInd];

	if(Iter == PtrToListOfPtrToTrans->end()) 
	{
		if(Cast.IdentTransCast(BaseTransPtr) == 0) TransPtrVect.push_back(new radTrans(*BaseTransPtr));
		else TransPtrVect.push_back(BaseTransPtr);
	}
	else NestedFor_Trans(BaseTransPtr, Iter, ElemLocInd, I_or_E);
}

//-------------------------------------------------------------------------

inline void radTInteraction::DestroyMainTransPtrArray()
{
	if(MainTransPtrArray == 0) return;

	for(int i=0; i<AmOfMainElem; i++)
	{
		radTrans* MainTransPtr = MainTransPtrArray[i];
		if(MainTransPtr != 0)
		{
			if(Cast.IdentTransCast(MainTransPtr)==0) 
			{
				delete (MainTransPtr);
				MainTransPtr = 0;
			}
		}
	}
	delete[] MainTransPtrArray;
	MainTransPtrArray = 0;
}

//-------------------------------------------------------------------------

inline void radTInteraction::ResetM()
{
	for(int i=0; i<AmOfMainElem; i++)
	{
		radTg3dRelax* g3dRelaxPtr = g3dRelaxPtrVect[i];

		g3dRelaxPtr->Magn = ((radTMaterial*)(g3dRelaxPtrVect[i]->MaterHandle.rep))->RemMagn;
		NewMagnArray[i] = g3dRelaxPtr->Magn;
		NewFieldArray[i] = TVector3d(0.,0.,0.); // Or make it TVector3df
	}
}

//-------------------------------------------------------------------------

inline void radTInteraction::ResetAuxParam()
{
	for(int i=0; i<AmOfMainElem; i++)
	{
		radTg3dRelax* g3dRelaxPtr = g3dRelaxPtrVect[i];

		g3dRelaxPtr->AuxFloat1 = 0;
		g3dRelaxPtr->AuxFloat2 = 0;
		g3dRelaxPtr->AuxFloat3 = 0;
	}
}

//-------------------------------------------------------------------------

inline void radTInteraction::InitAuxArrays()
{
	for(int i=0; i<AmOfMainElem; i++)
	{
		NewMagnArray[i] = g3dRelaxPtrVect[i]->Magn;
		NewFieldArray[i] = TVector3d(0.,0.,0.); // Or make it TVector3df
	}
}

//-------------------------------------------------------------------------

//inline void radTInteraction::StoreOldMagnData() //OC300504
//{
//	if(AuxOldMagnArray == NULL) return;
//
//    TVector3d *tAuxOldMagnArray = AuxOldMagnArray;
//	for(int i=0; i<AmOfMainElem; i++)
//	{
//		*(tAuxOldMagnArray++) = g3dRelaxPtrVect[i]->Magn;
//	}
//}

//-------------------------------------------------------------------------

inline void radTInteraction::StoreAuxOldArrays()
{
	if(AmOfMainElem <= 0) return;
	
	if((AuxOldMagnArray != NULL) && (AuxOldFieldArray != NULL))
	{
        TVector3d *tAuxOldMagn = AuxOldMagnArray;
		TVector3d *tAuxOldField = AuxOldFieldArray;
		TVector3d *tNewFieldArray = NewFieldArray;

        for(int StNo=0; StNo<AmOfMainElem; StNo++)
		{
			TVector3d &M = (g3dRelaxPtrVect[StNo])->Magn; 
			*(tAuxOldMagn++) = M;
			*(tAuxOldField++) = *(tNewFieldArray++);
		}
	}
	else
	{
		if(AuxOldMagnArray != NULL)
		{
			TVector3d *tAuxOldMagn = AuxOldMagnArray;
			for(int StNo=0; StNo<AmOfMainElem; StNo++)
			{
				TVector3d &M = (g3dRelaxPtrVect[StNo])->Magn; 
				*(tAuxOldMagn++) = M;
			}
		}
		if(AuxOldFieldArray != NULL)
		{
			TVector3d *tAuxOldField = AuxOldFieldArray;
			TVector3d *tNewFieldArray = NewFieldArray;
			for(int StNo=0; StNo<AmOfMainElem; StNo++)
			{
				*(tAuxOldField++) = *(tNewFieldArray++);
			}
		}
	}
}

//-------------------------------------------------------------------------

inline void radTInteraction::RestoreAuxOldArrays() //OC300504
{
	if((AuxOldMagnArray == NULL) && (AuxOldFieldArray == NULL)) return;

    TVector3d *tAuxOldMagnArray = AuxOldMagnArray;
	TVector3d *tNewMagnArray = NewMagnArray;
    TVector3d *tAuxOldFieldArray = AuxOldFieldArray;
    TVector3d *tNewFieldArray = NewFieldArray;

	if((AuxOldMagnArray != NULL) && (AuxOldFieldArray != NULL))
	{
		for(int i=0; i<AmOfMainElem; i++)
		{
			g3dRelaxPtrVect[i]->Magn = *tAuxOldMagnArray;
			*(tNewMagnArray++) = *(tAuxOldMagnArray++);
			*(tNewFieldArray++) = *(tAuxOldFieldArray++);
		}
	}
	else
	{
		if(AuxOldMagnArray != NULL)
		{
			for(int i=0; i<AmOfMainElem; i++)
			{
				g3dRelaxPtrVect[i]->Magn = *tAuxOldMagnArray;
				*(tNewMagnArray++) = *(tAuxOldMagnArray++);
			}
		}
		if(AuxOldFieldArray != NULL)
		{
			for(int i=0; i<AmOfMainElem; i++)
			{
                *(tNewFieldArray++) = *(tAuxOldFieldArray++);
			}
		}
	}
}

//-------------------------------------------------------------------------

inline void radTInteraction::ShowInteractVector(char Ch)
{
	TVector3d* Vect3dPtr = NULL;
	switch(Ch) 
	{
		case 'E':
			Vect3dPtr = ExternFieldArray; break;
		case 'T':
			Vect3dPtr = NewFieldArray; break;
		case 'M':
			Vect3dPtr = NewMagnArray; break;
		default :
			Vect3dPtr = NewFieldArray; break;
	}
	Send.ArrayOfVector3d(Vect3dPtr, AmOfMainElem);
}

//-------------------------------------------------------------------------

inline void radTInteraction::ShowInteractMatrix()
{
	Send.MatrixOfMatrix3d(InteractMatrix, AmOfMainElem, AmOfMainElem);
}

//-------------------------------------------------------------------------

inline void radTInteraction::Dump(std::ostream& o, int ShortSign) // Porting
{
	radTg::Dump(o);
	o << "Interaction: ";

	if(ShortSign) return;

	o << endl;
	o << "   Number of \"atomic\" relaxable objects: " << AmOfMainElem << endl;
	o << "   Total number of degrees of freedom to relax on: " << AmOfMainElem*3 << endl;
	o << "   Number of external field sources in the general container: " << AmOfExtElem;

	o << endl;
	o << "   Memory occupied: " << SizeOfThis() << " bytes";
}

//-------------------------------------------------------------------------

inline int radTInteraction::SizeOfThis()
{
	long GenSize = sizeof(*this);
	GenSize += AmOfMainElem*AmOfMainElem*sizeof(TMatrix3d);
	GenSize += AmOfMainElem*sizeof(TMatrix3d*);
	GenSize += 3*AmOfMainElem*sizeof(TVector3d);
	GenSize += AmOfMainElem*sizeof(radTg3dRelax*);
	GenSize += AmOfRelaxSubInterv*sizeof(radTRelaxSubInterval);
	return GenSize;
}

//-------------------------------------------------------------------------

inline void radTInteraction::OutRelaxStatusParam(double* RelaxStatusParamArray)
{
	RelaxStatusParamArray[0] = RelaxStatusParam.MisfitM;
	RelaxStatusParamArray[1] = RelaxStatusParam.MaxModM;
	RelaxStatusParamArray[2] = RelaxStatusParam.MaxModH;
	// Add more members of radTRelaxStatusParam here, should they appear in Future
}

//-------------------------------------------------------------------------

inline void radTInteraction::UpdateExternalField()
{
	SetupExternFieldArray(); //zeros and then sets the ExternFieldArray from g3dExternPtrVect
	AddExternFieldFromMoreExtSource(); //adds field to ExternFieldArray from MoreExtSourceHandle
}

//-------------------------------------------------------------------------

template<class T> inline long radTInteraction::OutMagnVals(T*& arMagnVals, int nExtraVals) //OC02012020
{//Allocates arMagnVals!
 //This assumes that after the Relaxation, the resulting Magnetization data is stored in radTInteraction::g3dRelaxPtrVect
 //(but can be changed to take the Magnetization data from radTInteraction::NewMagnArray)
	//arMagnVals = 0;
	if(AmOfMainElem <= 0) return 0;
	long nMagnVals = AmOfMainElem*3 + nExtraVals; //nExtraVals can be used e.g. for attaching extra data to the Magnetization array

	if(arMagnVals == 0) arMagnVals = new T[nMagnVals];
	if(arMagnVals == 0) { Send.ErrorMessage("Radia::Error900"); return 0;}
	T *t_arMagnVals = arMagnVals + nExtraVals;
	for(long i=0; i<AmOfMainElem; i++)
	{
		TVector3d &M = (g3dRelaxPtrVect[i])->Magn;
		*(t_arMagnVals++) = (T)M.x; *(t_arMagnVals++) = (T)M.y; *(t_arMagnVals++) = (T)M.z;
	}
	return nMagnVals;
}

//-------------------------------------------------------------------------

template<class T> inline void radTInteraction::SetRelaxObjMagnVals(T* arMagnVals)
{
	if((AmOfMainElem <= 0) || (arMagnVals == 0)) return;

	T *t_arMagnVals = arMagnVals;
	for(long i=0; i<AmOfMainElem; i++)
	{
		TVector3d &M = (g3dRelaxPtrVect[i])->Magn;
		 M.x = *(t_arMagnVals++);  M.y = *(t_arMagnVals++); M.z = *(t_arMagnVals++);
	}
}

//-------------------------------------------------------------------------

inline void radTInteraction::LongLongToFloatAr(long long inVal, float outAr[4]) //24122019
{//Aux. function to encode one long long number to 4 floats (consider moving to some parser class)
	long long mask = 0xFFFF;
	for(int j=0; j<4; j++)
	{
		outAr[j] = (float)((inVal & mask) >> (j*16));
		mask <<= 16;
	}
}

//-------------------------------------------------------------------------

inline long long radTInteraction::FloatArToLongLong(float inAr[4]) //OC27122019
{//Aux. function to decode one long long number from 4 floats (consider moving to some parser class)
	long long res=0, aux;
	for(int j=0; j<4; j++)
	{
		aux = (long long)inAr[j];
		aux <<= (j*16);
		res |= aux;
	}
	return res;
}

//-------------------------------------------------------------------------

#endif
