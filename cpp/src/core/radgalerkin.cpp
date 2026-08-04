/*-------------------------------------------------------------------------
*
* File name:      radgalerkin.cpp
*
* Project:        RADIA (RadiaCUDA)
*
* Description:    Configuration and observation-element quadrature for the
*                 OPT-IN volume-averaged ("Galerkin") interaction matrix.
*                 See radgalerkin.h for the scheme and the env switches.
*
-------------------------------------------------------------------------*/

#include "radgalerkin.h"
#include "radg3d.h"
#include "radrec.h"
#include "radvlpgn.h"
#include "radplnr.h"
#include "radtrans.h"
#include "radcast.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

//-------------------------------------------------------------------------
// Configuration
//-------------------------------------------------------------------------

const radTGalerkinCfg& radGalerkinCfg()
{
	static radTGalerkinCfg cfg;
	static bool init = false;
	if(init) return cfg;

	//K defaults to 14, not to the cheapest usable rule: on the 60 MeV model K=4
	//lands 22.5 kHz away from the K>=14 answer in mean_f (3x the discrepancy the
	//scheme was built to probe), so a K=4 default would quietly hand back a
	//number that looks like Galerkin and is not. See studies/GALERKIN_STEP1.md.
	cfg.On = false; cfg.K = 14; cfg.KNear = 14; cfg.NearLevels = 1;
	cfg.Cutoff = 1.5; cfg.Debug = false;

	const char* e = getenv("RADIA_GALERKIN");
	if(e && *e && *e != '0') cfg.On = true;
	e = getenv("RADIA_GALERKIN_K");
	if(e && *e)
	{
		int v = atoi(e);
		if((v == 1) || (v == 4) || (v == 8) || (v == 14) || (v == 24)) cfg.K = v;
	}
	e = getenv("RADIA_GALERKIN_KNEAR");
	if(e && *e)
	{
		int v = atoi(e);
		if((v == 1) || (v == 4) || (v == 8) || (v == 14) || (v == 24)) cfg.KNear = v;
	}
	e = getenv("RADIA_GALERKIN_NEARLEV");
	if(e && *e) { int v = atoi(e); if((v >= 0) && (v <= 2)) cfg.NearLevels = v; }
	e = getenv("RADIA_GALERKIN_CUTOFF");
	if(e && *e) { double v = atof(e); if((v >= 0.) && (v < 1.e+03)) cfg.Cutoff = v; }
	e = getenv("RADIA_GALERKIN_DEBUG");
	if(e && *e && *e != '0') cfg.Debug = true;

	init = true;
	return cfg;
}

//-------------------------------------------------------------------------
// Fully symmetric tetrahedron rules, in barycentric orbit form.
//
// Orbits: S4 (1 point), S31(a) (4), S22(a) (6), S211(a,b) (12). Weights are
// per point and sum to 1. Validated against exact monomial integrals by
// studies/galerkin_quad.py, which carries the same tables.
//-------------------------------------------------------------------------

namespace {

struct TetOrbit { int kind; double a, b, w; };   // kind: 0=S4, 1=S31, 2=S22, 3=S211

// K = 1, degree 1
const TetOrbit gTet1[] = {{0, 0., 0., 1.}};
// K = 4, degree 2
const TetOrbit gTet4[] = {{1, 0.1381966011250105, 0., 0.25}};
// K = 8, degree 3 (two-orbit family member with the smallest degree-4 error)
const TetOrbit gTet8[] = {{1, 0.3284461152882206, 0., 0.1310962232055796},
                          {1, 0.1103685211074304, 0., 0.1189037767944204}};
// K = 14, degree 5
const TetOrbit gTet14[] = {{1, 0.3108859192633005, 0., 0.1126879257180162},
                           {1, 0.0927352503108912, 0., 0.0734930431163619},
                           {2, 0.0455037041256497, 0., 0.0425460207770812}};
// K = 24, degree 6
const TetOrbit gTet24[] = {{1, 0.2146028712591517, 0., 0.0399227502581679},
                           {1, 0.0406739585346113, 0., 0.0100772110553207},
                           {1, 0.3223378901422757, 0., 0.0553571815436544},
                           {3, 0.0636610018750175, 0.2696723314583159, 0.0482142857142857}};

struct TetRule { const TetOrbit* orb; int nOrb; int nPts; int deg; };

const TetRule* TetRuleFor(int K)
{
	static const TetRule r1  = {gTet1,  1, 1,  1};
	static const TetRule r4  = {gTet4,  1, 4,  2};
	static const TetRule r8  = {gTet8,  2, 8,  3};
	static const TetRule r14 = {gTet14, 3, 14, 5};
	static const TetRule r24 = {gTet24, 4, 24, 6};
	switch(K)
	{
		case 1:  return &r1;
		case 4:  return &r4;
		case 8:  return &r8;
		case 14: return &r14;
		case 24: return &r24;
	}
	return 0;
}

// Barycentric coordinates of one orbit; returns the number of points written.
int ExpandOrbit(const TetOrbit& o, double bc[][4])
{
	int n = 0;
	if(o.kind == 0)
	{
		bc[0][0] = bc[0][1] = bc[0][2] = bc[0][3] = 0.25;
		return 1;
	}
	if(o.kind == 1)
	{
		double a = o.a, b = 1. - 3.*o.a;
		for(int i=0; i<4; i++)
		{
			for(int k=0; k<4; k++) bc[n][k] = a;
			bc[n][i] = b;
			n++;
		}
		return n;
	}
	if(o.kind == 2)
	{
		double a = o.a, b = 0.5 - o.a;
		for(int i=0; i<4; i++) for(int j=i+1; j<4; j++)
		{
			for(int k=0; k<4; k++) bc[n][k] = b;
			bc[n][i] = a; bc[n][j] = a;
			n++;
		}
		return n;
	}
	// kind == 3: (a,a,b,c), c = 1 - 2a - b
	double a = o.a, b = o.b, c = 1. - 2.*o.a - o.b;
	for(int i=0; i<4; i++) for(int j=i+1; j<4; j++)
	{
		int rest[2], nr = 0;
		for(int k=0; k<4; k++) if((k != i) && (k != j)) rest[nr++] = k;
		for(int bi=0; bi<2; bi++)
		{
			bc[n][i] = a; bc[n][j] = a;
			bc[n][rest[bi]] = b; bc[n][rest[1-bi]] = c;
			n++;
		}
	}
	return n;
}

inline double TetVol(const TVector3d& v0, const TVector3d& v1,
                     const TVector3d& v2, const TVector3d& v3)
{
	TVector3d a = v1 - v0, b = v2 - v0, c = v3 - v0;
	double d = a.x*(b.y*c.z - b.z*c.y) - a.y*(b.x*c.z - b.z*c.x)
	         + a.z*(b.x*c.y - b.y*c.x);
	return fabs(d)/6.;
}

// Apply the rule of `nPts` points to one tet, appending to pts/wts with the
// caller-supplied volume weight factor.
void AddTetRule(const TetRule* rule, const TVector3d v[4], double wFac,
                std::vector<TVector3d>& pts, std::vector<double>& wts)
{
	for(int io=0; io<rule->nOrb; io++)
	{
		double bc[12][4];                       // largest orbit is S211 = 12
		int nOrb = ExpandOrbit(rule->orb[io], bc);
		for(int k=0; k<nOrb; k++)
		{
			pts.push_back(v[0]*bc[k][0] + v[1]*bc[k][1]
			            + v[2]*bc[k][2] + v[3]*bc[k][3]);
			wts.push_back(rule->orb[io].w * wFac);
		}
	}
}

// 1 -> 8 conforming tet refinement (4 corner tets + 4 from the octahedron,
// split along the shortest diagonal choice used by studies/galerkin_quad.py).
void SubdivideTet(const TVector3d v[4], TVector3d out[8][4])
{
	TVector3d m01 = (v[0] + v[1])*0.5, m02 = (v[0] + v[2])*0.5,
	          m03 = (v[0] + v[3])*0.5, m12 = (v[1] + v[2])*0.5,
	          m13 = (v[1] + v[3])*0.5, m23 = (v[2] + v[3])*0.5;
	const TVector3d tets[8][4] = {
		{v[0], m01, m02, m03}, {v[1], m01, m12, m13},
		{v[2], m02, m12, m23}, {v[3], m03, m13, m23},
		{m01, m02, m03, m23},  {m01, m02, m12, m23},
		{m01, m03, m13, m23},  {m01, m12, m13, m23}};
	for(int i=0; i<8; i++) for(int k=0; k<4; k++) out[i][k] = tets[i][k];
}

// Composite rule over one tet: `levels` rounds of subdivision, `rule` on each
// cell, volume-weighted. Weights accumulate to wFac.
void AddTetComposite(const TetRule* rule, const TVector3d v[4], int levels,
                     double wFac, std::vector<TVector3d>& pts,
                     std::vector<double>& wts)
{
	if(levels <= 0) { AddTetRule(rule, v, wFac, pts, wts); return;}

	std::vector<TVector3d> cells;              // flat, 4 vertices per cell
	cells.reserve(4*8);
	for(int k=0; k<4; k++) cells.push_back(v[k]);
	for(int lev=0; lev<levels; lev++)
	{
		std::vector<TVector3d> next;
		next.reserve(cells.size()*8);
		for(size_t c=0; c<cells.size(); c+=4)
		{
			TVector3d cur[4] = {cells[c], cells[c+1], cells[c+2], cells[c+3]};
			TVector3d sub[8][4];
			SubdivideTet(cur, sub);
			for(int i=0; i<8; i++) for(int k=0; k<4; k++) next.push_back(sub[i][k]);
		}
		cells.swap(next);
	}

	double volTot = 0.;
	size_t nCells = cells.size()/4;
	std::vector<double> vol(nCells);
	for(size_t c=0; c<nCells; c++)
	{
		vol[c] = TetVol(cells[4*c], cells[4*c+1], cells[4*c+2], cells[4*c+3]);
		volTot += vol[c];
	}
	if(volTot <= 0.) { AddTetRule(rule, v, wFac, pts, wts); return;}
	for(size_t c=0; c<nCells; c++)
	{
		TVector3d cur[4] = {cells[4*c], cells[4*c+1], cells[4*c+2], cells[4*c+3]};
		AddTetRule(rule, cur, wFac*vol[c]/volTot, pts, wts);
	}
}

// Gauss-Legendre nodes/weights on [-1,1] for n = 1..4 (weights normalized to
// sum to 1, i.e. already divided by 2).
struct GLRule { int n; double x[4]; double w[4]; };

const GLRule* GLRuleFor(int n)
{
	static const GLRule g1 = {1, {0., 0., 0., 0.}, {1., 0., 0., 0.}};
	static const GLRule g2 = {2, {-0.5773502691896257, 0.5773502691896257, 0., 0.},
	                             { 0.5, 0.5, 0., 0.}};
	static const GLRule g3 = {3, {-0.7745966692414834, 0., 0.7745966692414834, 0.},
	                             { 0.2777777777777778, 0.4444444444444444,
	                               0.2777777777777778, 0.}};
	static const GLRule g4 = {4, {-0.8611363115940526, -0.3399810435848563,
	                               0.3399810435848563,  0.8611363115940526},
	                             { 0.1739274225687269, 0.3260725774312731,
	                               0.3260725774312731, 0.1739274225687269}};
	switch(n) { case 1: return &g1; case 2: return &g2; case 3: return &g3;
	            case 4: return &g4;}
	return 0;
}

// Gauss-Legendre order per axis that matches or exceeds the tet rule's degree.
int GLOrderForTetDegree(int deg)
{
	if(deg <= 1) return 1;      // degree 1
	if(deg <= 3) return 2;      // degree 3
	if(deg <= 5) return 3;      // degree 5
	return 4;                   // degree 7
}

// 3D vertices of one polyhedron face, in the element's own frame.
void FaceVertices(radTHandlePgnAndTrans& hpt, std::vector<TVector3d>& out)
{
	radTPolygon* pgn = hpt.PgnHndl.rep;
	radTrans* tr = hpt.TransHndl.rep;
	out.clear();
	for(int i=0; i<pgn->AmOfEdgePoints; i++)
	{
		TVector2d& ep = pgn->EdgePointsVector[i];
		TVector3d loc(ep.x, ep.y, pgn->CoordZ);
		out.push_back(tr->TrBiPoint(loc));
	}
}

} // anonymous namespace

//-------------------------------------------------------------------------

double radGalerkinElemSize(radTg3dRelax* el)
{
	radTCast Cast;
	radTRecMag* rec = Cast.RecMagCast(el);
	if(rec != 0)
	{
		double V = fabs(rec->Dimensions.x * rec->Dimensions.y * rec->Dimensions.z);
		return (V > 0.)? pow(V, 1./3.) : 0.;
	}
	radTPolyhedron* poly = Cast.PolyhedronCast(el);
	if(poly != 0)
	{
		TVector3d C = poly->ReturnCentrPoint();
		double V = 0.;
		std::vector<TVector3d> fv;
		for(int f=0; f<poly->AmOfFaces; f++)
		{
			FaceVertices(poly->VectHandlePgnAndTrans[f], fv);
			for(size_t k=1; k+1<fv.size(); k++)
				V += TetVol(C, fv[0], fv[k], fv[k+1]);
		}
		return (V > 0.)? pow(V, 1./3.) : 0.;
	}
	return 0.;
}

//-------------------------------------------------------------------------

int radGalerkinElemQuad(radTg3dRelax* el, int K, int levels,
                        std::vector<TVector3d>& pts, std::vector<double>& wts)
{
	pts.clear(); wts.clear();
	if(el == 0) return 0;

	const TetRule* rule = TetRuleFor(K);
	if(rule == 0) return 0;

	// Collocation: exactly the centroid, weight 1. Kept as an explicit early
	// exit so the flag-off packing is trivially identical to the old one.
	if((K == 1) && (levels <= 0))
	{
		pts.push_back(el->ReturnCentrPoint());
		wts.push_back(1.);
		return 1;
	}

	radTCast Cast;

	// --- RecMag: tensor Gauss-Legendre ---------------------------------
	radTRecMag* rec = Cast.RecMagCast(el);
	if(rec != 0)
	{
		int n = GLOrderForTetDegree(rule->deg);
		// A subdivision level on a cuboid is just a higher GL order per axis;
		// clamp to the tabulated range (n <= 4, degree 7).
		if(levels > 0) { n += levels; if(n > 4) n = 4;}
		const GLRule* g = GLRuleFor(n);
		if(g == 0) return 0;
		TVector3d C = rec->CentrPoint;
		double hx = 0.5*rec->Dimensions.x, hy = 0.5*rec->Dimensions.y,
		       hz = 0.5*rec->Dimensions.z;
		for(int i=0; i<g->n; i++) for(int j=0; j<g->n; j++) for(int k=0; k<g->n; k++)
		{
			pts.push_back(TVector3d(C.x + hx*g->x[i], C.y + hy*g->x[j],
			                        C.z + hz*g->x[k]));
			wts.push_back(g->w[i]*g->w[j]*g->w[k]);
		}
		return (int)pts.size();
	}

	// --- Polyhedron ----------------------------------------------------
	radTPolyhedron* poly = Cast.PolyhedronCast(el);
	if(poly == 0) return 0;                     // unsupported element type

	// A tetrahedron (4 triangular faces) is the all-tet production case: run
	// the rule on the element itself rather than on a star decomposition, which
	// would cost 4x the points for the same accuracy.
	if(poly->AmOfFaces == 4)
	{
		bool allTri = true;
		for(int f=0; f<4; f++)
			if(poly->VectHandlePgnAndTrans[f].PgnHndl.rep->AmOfEdgePoints != 3)
			{ allTri = false; break;}
		if(allTri)
		{
			TVector3d v[4];
			int nv = 0;
			std::vector<TVector3d> fv;
			for(int f=0; f<4 && nv<4; f++)
			{
				FaceVertices(poly->VectHandlePgnAndTrans[f], fv);
				for(size_t k=0; k<fv.size() && nv<4; k++)
				{
					bool dup = false;
					for(int q=0; q<nv; q++)
					{
						TVector3d d = v[q] - fv[k];
						double s = fabs(v[q].x) + fabs(v[q].y) + fabs(v[q].z);
						double tol = 1.e-10*(s > 0.? s : 1.);
						if((fabs(d.x) < tol) && (fabs(d.y) < tol) && (fabs(d.z) < tol))
						{ dup = true; break;}
					}
					if(!dup) v[nv++] = fv[k];
				}
			}
			if((nv == 4) && (TetVol(v[0], v[1], v[2], v[3]) > 0.))
			{
				AddTetComposite(rule, v, levels, 1., pts, wts);
				return (int)pts.size();
			}
		}
	}

	// General polyhedron: star decomposition from the centroid through every
	// face triangle. Valid for any element star-shaped about its centroid,
	// which covers convex elements and every subdivision Radia produces.
	{
		TVector3d C = poly->ReturnCentrPoint();
		std::vector<TVector3d> cellV;           // flat, 4 per sub-tet
		std::vector<double> cellVol;
		double volTot = 0.;
		std::vector<TVector3d> fv;
		for(int f=0; f<poly->AmOfFaces; f++)
		{
			FaceVertices(poly->VectHandlePgnAndTrans[f], fv);
			for(size_t k=1; k+1<fv.size(); k++)
			{
				double V = TetVol(C, fv[0], fv[k], fv[k+1]);
				if(V <= 0.) continue;
				cellV.push_back(C); cellV.push_back(fv[0]);
				cellV.push_back(fv[k]); cellV.push_back(fv[k+1]);
				cellVol.push_back(V);
				volTot += V;
			}
		}
		if((volTot <= 0.) || cellVol.empty()) return 0;
		for(size_t c=0; c<cellVol.size(); c++)
		{
			TVector3d cur[4] = {cellV[4*c], cellV[4*c+1], cellV[4*c+2], cellV[4*c+3]};
			AddTetComposite(rule, cur, levels, cellVol[c]/volTot, pts, wts);
		}
		return (int)pts.size();
	}
}
