"""field_kernel.py — GPU-accelerated magnetic field computation for Radia geometries.

Supports:
  - Polyhedron elements (tetrahedral meshes)
  - RecMag elements (rectangular parallelepipeds)
  - Arbitrary mirror symmetries through origin
  - FP64 and mixed precision modes

Usage:
    from PyRadia.radia_flatten import flatten
    from PyRadia.field_kernel import fld_gpu, GPUGeometry

    geo = flatten(container)
    gpu_geo = GPUGeometry(geo)

    symmetries = [
        ('perp', [0,0,0], [1,0,0]),
        ('para', [0,0,0], [0,0,1]),
    ]

    B = fld_gpu(geo, points, gpu_geo=gpu_geo, symmetries=symmetries)
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ═══════════════════════════════════════════════════════════════
# CUDA Kernels — V2 architecture (one block per point,
# threads split elements, Python-side symmetry loop)
# ═══════════════════════════════════════════════════════════════

_POLY_KERNEL_FP64 = r"""
extern "C" __global__
void polyhedron_field_fp64(
    const double* __restrict__ obs_pts,
    const int Np,
    const double* __restrict__ magn,
    const int n_elem,
    const int* __restrict__ face_offsets,
    const int* __restrict__ edge_offsets,
    const double* __restrict__ edge_pts_2d,
    const double* __restrict__ face_cz,
    const double* __restrict__ face_rot,
    const double* __restrict__ face_orig,
    double* __restrict__ out_B
)
{
    int ip = blockIdx.x;
    if (ip >= Np) return;

    int tid = threadIdx.x;
    int nThreads = blockDim.x;

    double px = obs_pts[ip*3+0];
    double py = obs_pts[ip*3+1];
    double pz = obs_pts[ip*3+2];

    const double PI = 3.14159265358979323846;
    const double ConstForH = 1.0 / (4.0 * PI);
    const double Max_k = 1.0e+09;
    const double RelRandMagn = 1.0e-13;
    const double MaxRelTolToSwitch = 1.0e-07;

    double Bx = 0.0, By = 0.0, Bz = 0.0;

    for (int ie = tid; ie < n_elem; ie += nThreads)
    {
        double mx = magn[ie*3+0], my = magn[ie*3+1], mz = magn[ie*3+2];
        int f_start = face_offsets[ie], f_end = face_offsets[ie+1];
        double Hx_sum = 0.0, Hy_sum = 0.0, Hz_sum = 0.0;

        for (int fi = f_start; fi < f_end; fi++)
        {
            double ox=face_orig[fi*3+0],oy=face_orig[fi*3+1],oz=face_orig[fi*3+2];
            double dx=px-ox, dy=py-oy, dz=pz-oz;
            double r00=face_rot[fi*9+0],r01=face_rot[fi*9+1],r02=face_rot[fi*9+2];
            double r10=face_rot[fi*9+3],r11=face_rot[fi*9+4],r12=face_rot[fi*9+5];
            double r20=face_rot[fi*9+6],r21=face_rot[fi*9+7],r22=face_rot[fi*9+8];

            double loc_x=r00*dx+r01*dy+r02*dz;
            double loc_y=r10*dx+r11*dy+r12*dz;
            double loc_z_obs=r20*dx+r21*dy+r22*dz;
            double loc_mz=r20*mx+r21*my+r22*mz;

            double z = face_cz[fi] - loc_z_obs;
            if (z == 0.0) z = RelRandMagn;
            double ze2 = z*z;

            int e_start=edge_offsets[fi], e_end=edge_offsets[fi+1];
            int n_edges=e_end-e_start;

            double Sx=0.0, Sy=0.0, Sz=0.0;
            double ArgSumAtans1=0.0, PiMultSumAtans1=0.0;
            double ArgSumLogs2=1.0;

            double x1=edge_pts_2d[e_start*2+0]-loc_x;
            double y1=edge_pts_2d[e_start*2+1]-loc_y;
            if(x1==0.0) x1=RelRandMagn;
            if(y1==0.0) y1=RelRandMagn;
            double x1e2=x1*x1, y1e2=y1*y1;

            for(int ei=0; ei<n_edges; ei++)
            {
                int next_ei=(ei+1)%n_edges;
                double x2=edge_pts_2d[(e_start+next_ei)*2+0]-loc_x;
                double y2=edge_pts_2d[(e_start+next_ei)*2+1]-loc_y;
                if(x2==0.0) x2=RelRandMagn;
                if(y2==0.0) y2=RelRandMagn;
                double x2e2=x2*x2, y2e2=y2*y2;

                double x2mx1=x2-x1, y2my1=y2-y1;

                if(fabs(x2mx1)*Max_k > fabs(y2my1))
                {
                    double k=y2my1/x2mx1;
                    double b=y1-k*x1;
                    if(b==0.0) b=RelRandMagn;

                    double ke2=k*k, be2=b*b, ke2p1=ke2+1.0;
                    double sqrtke2p1=sqrt(ke2p1), bk=b*k;
                    double bpkx1=b+k*x1, bpkx2=b+k*x2;
                    double bpkx1e2=bpkx1*bpkx1, bpkx2e2=bpkx2*bpkx2;
                    double R1=sqrt(x1e2+bpkx1e2+ze2);
                    double R2=sqrt(x2e2+bpkx2e2+ze2);

                    double R1pbpkx1=bpkx1+R1, R2pbpkx2=bpkx2+R2;

                    double AbsRandR1=100.0*R1*RelRandMagn;
                    double AbsRandR2=100.0*R2*RelRandMagn;
                    double MaxAbsRandR1=MaxRelTolToSwitch*R1;
                    double MaxAbsRandR2=MaxRelTolToSwitch*R2;
                    if(AbsRandR1>MaxAbsRandR1) AbsRandR1=MaxAbsRandR1;
                    if(AbsRandR2>MaxAbsRandR2) AbsRandR2=MaxAbsRandR2;

                    if(fabs(R1pbpkx1)<AbsRandR1 && R1>100.0*AbsRandR1 && (x1e2+ze2)<bpkx1e2*MaxRelTolToSwitch)
                        R1pbpkx1=(bpkx1!=0.0)?0.5*(x1e2+ze2)/fabs(bpkx1):1.0e-50;
                    if(fabs(R2pbpkx2)<AbsRandR2 && R2>100.0*AbsRandR2 && (x2e2+ze2)<bpkx2e2*MaxRelTolToSwitch)
                        R2pbpkx2=(bpkx2!=0.0)?0.5*(x2e2+ze2)/fabs(bpkx2):1.0e-50;
                    if(R1pbpkx1==0.0) R1pbpkx1=1.0e-50;
                    if(R2pbpkx2==0.0) R2pbpkx2=1.0e-50;

                    double bkpx1pke2x1=bk+ke2p1*x1, bkpx2pke2x2=bk+ke2p1*x2;
                    double kze2=k*ze2, ke2ze2=k*kze2;
                    double ke2ze2pbe2=ke2ze2+be2, ke2ze2mbe2=ke2ze2-be2;
                    double bx1=b*x1, bx2=b*x2;
                    double x1e2pze2=x1e2+ze2, x2e2pze2=x2e2+ze2;
                    double twob=2.0*b;
                    double kx1mb=k*x1-b, kx2mb=k*x2-b;

                    double Arg1=-(ke2ze2pbe2*(bx1+kze2)*R1pbpkx1+kze2*twob*x1e2pze2);
                    double Arg2=(ke2ze2pbe2*kx1mb*R1pbpkx1+ke2ze2mbe2*x1e2pze2)*z;
                    double Arg3=ke2ze2pbe2*(bx2+kze2)*R2pbpkx2+kze2*twob*x2e2pze2;
                    double Arg4=(ke2ze2pbe2*kx2mb*R2pbpkx2+ke2ze2mbe2*x2e2pze2)*z;

                    if(Arg2==0.0) Arg2=1.0e-50;
                    if(Arg4==0.0) Arg4=1.0e-50;

                    double rat1=Arg1/Arg2, rat2=Arg3/Arg4;
                    double denom=1.0-rat1*rat2;
                    double PiMult1=0.0, CurArg=0.0;
                    if(fabs(denom)>1.0e-14*(fabs(rat1)+fabs(rat2))){
                        CurArg=(rat1+rat2)/denom;
                    } else {
                        PiMult1=(rat1>0.0)?1.0:-1.0;
                    }
                    double PiMult2=0.0;
                    double denom2=1.0-ArgSumAtans1*CurArg;
                    if(fabs(denom2)>1.0e-14*(fabs(ArgSumAtans1)+fabs(CurArg))){
                        ArgSumAtans1=(ArgSumAtans1+CurArg)/denom2;
                    } else {
                        PiMult2=(ArgSumAtans1<0.0)?-1.0:1.0;
                        ArgSumAtans1=0.0;
                    }
                    PiMultSumAtans1+=PiMult1+PiMult2;

                    double val1=bkpx1pke2x1/sqrtke2p1+R1;
                    double val2=bkpx2pke2x2/sqrtke2p1+R2;
                    double be2pze2=be2+ze2;
                    if(fabs(val1)<AbsRandR1 && R1>100.0*AbsRandR1 && (be2pze2+2*bk*x1)<x1e2*ke2p1*MaxRelTolToSwitch)
                        val1=(x1!=0.0)?0.5*be2pze2/(fabs(x1)*sqrtke2p1):1.0e-50;
                    if(fabs(val2)<AbsRandR2 && R2>100.0*AbsRandR2 && (be2pze2+2*bk*x2)<x2e2*ke2p1*MaxRelTolToSwitch)
                        val2=(x2!=0.0)?0.5*be2pze2/(fabs(x2)*sqrtke2p1):1.0e-50;
                    if(val1==0.0) val1=1.0e-50;
                    if(val2==0.0) val2=1.0e-50;

                    double log_ratio=val2/val1;
                    if(log_ratio<=0.0) log_ratio=1.0e-50;
                    double SumLogs1=log(log_ratio);
                    double SumLogs1dsqrtke2p1=SumLogs1/sqrtke2p1;

                    double log_ratio2=R2pbpkx2/R1pbpkx1;
                    if(log_ratio2<=0.0) log_ratio2=1.0e-50;
                    ArgSumLogs2*=log_ratio2;
                    Sx+=-k*SumLogs1dsqrtke2p1;
                    Sy+=SumLogs1dsqrtke2p1;
                }
                x1=x2; y1=y2; x1e2=x2e2; y1e2=y2e2;
            }

            Sz=atan(ArgSumAtans1)+PiMultSumAtans1*PI;
            if(ArgSumLogs2<=0.0) ArgSumLogs2=1.0e-50;
            Sx+=log(ArgSumLogs2);

            double Hx_loc=-ConstForH*loc_mz*Sx;
            double Hy_loc=-ConstForH*loc_mz*Sy;
            double Hz_loc=-ConstForH*loc_mz*Sz;

            Hx_sum+=r00*Hx_loc+r10*Hy_loc+r20*Hz_loc;
            Hy_sum+=r01*Hx_loc+r11*Hy_loc+r21*Hz_loc;
            Hz_sum+=r02*Hx_loc+r12*Hy_loc+r22*Hz_loc;
        }
        Bx+=Hx_sum; By+=Hy_sum; Bz+=Hz_sum;
    }

    __shared__ double sBx[256];
    __shared__ double sBy[256];
    __shared__ double sBz[256];
    sBx[tid]=Bx; sBy[tid]=By; sBz[tid]=Bz;
    __syncthreads();
    for(int s=nThreads/2;s>0;s>>=1){
        if(tid<s){sBx[tid]+=sBx[tid+s];sBy[tid]+=sBy[tid+s];sBz[tid]+=sBz[tid+s];}
        __syncthreads();
    }
    if(tid==0){
        out_B[ip*3+0]=sBx[0];
        out_B[ip*3+1]=sBy[0];
        out_B[ip*3+2]=sBz[0];
    }
}
"""

_RECMAG_KERNEL_FP64 = r"""
extern "C" __global__
void recmag_field_fp64(
    const double* __restrict__ obs_pts,
    const int Np,
    const double* __restrict__ centers,
    const double* __restrict__ dims,
    const double* __restrict__ magn,
    const int n_rec,
    double* __restrict__ out_B
)
{
    int ip = blockIdx.x;
    if (ip >= Np) return;
    int tid = threadIdx.x;
    int nThreads = blockDim.x;

    double px=obs_pts[ip*3+0], py=obs_pts[ip*3+1], pz=obs_pts[ip*3+2];
    const double PI4 = 4.0 * 3.14159265358979323846;
    double Bx=0.0, By=0.0, Bz=0.0;

    for(int ir=tid; ir<n_rec; ir+=nThreads)
    {
        double cx=centers[ir*3+0],cy=centers[ir*3+1],cz=centers[ir*3+2];
        double wx=dims[ir*3+0],wy=dims[ir*3+1],wz=dims[ir*3+2];
        double mx=magn[ir*3+0],my=magn[ir*3+1],mz=magn[ir*3+2];
        double rx=px-cx,ry=py-cy,rz=pz-cz;
        double hx=0.5*wx,hy=0.5*wy,hz=0.5*wz;
        double x0=rx-hx,x1_=rx+hx;
        double y0=ry-hy,y1_=ry+hy;
        double z0=rz-hz,z1_=rz+hz;
        double Hx=0.0,Hy=0.0,Hz=0.0;
        double xs[2]={x0,x1_}, ys[2]={y0,y1_}, zs[2]={z0,z1_};
        for(int ix=0;ix<2;ix++){
            double x=xs[ix]; double sx=(ix==0)?-1.0:1.0;
            for(int iy=0;iy<2;iy++){
                double y=ys[iy]; double sy=(iy==0)?-1.0:1.0;
                for(int iz=0;iz<2;iz++){
                    double z=zs[iz]; double sz=(iz==0)?-1.0:1.0;
                    double sign=sx*sy*sz;
                    double xe2=x*x,ye2=y*y,ze2=z*z;
                    double R=sqrt(xe2+ye2+ze2);
                    if(R<1e-20) R=1e-20;
                    double zpR=z+R,ypR=y+R,xpR=x+R;
                    if(fabs(zpR)<1e-20) zpR=1e-20;
                    if(fabs(ypR)<1e-20) ypR=1e-20;
                    if(fabs(xpR)<1e-20) xpR=1e-20;
                    double log_zpR=log(fabs(zpR));
                    double log_ypR=log(fabs(ypR));
                    double log_xpR=log(fabs(xpR));
                    double xR=x*R;
                    double at_yz_xR=(fabs(xR)>1e-30)?atan2(y*z,xR):0.0;
                    double at_xz_yR=(fabs(y*R)>1e-30)?atan2(x*z,y*R):0.0;
                    double at_xy_zR=(fabs(z*R)>1e-30)?atan2(x*y,z*R):0.0;
                    Hx+=sign*(mx*at_yz_xR-my*log_zpR-mz*log_ypR);
                    Hy+=sign*(-mx*log_zpR+my*at_xz_yR-mz*log_xpR);
                    Hz+=sign*(-mx*log_ypR-my*log_xpR+mz*at_xy_zR);
                }
            }
        }
        double inv4pi=1.0/PI4;
        Bx-=Hx*inv4pi; By-=Hy*inv4pi; Bz-=Hz*inv4pi;
    }

    __shared__ double sBx[256];
    __shared__ double sBy[256];
    __shared__ double sBz[256];
    sBx[tid]=Bx; sBy[tid]=By; sBz[tid]=Bz;
    __syncthreads();
    for(int s=nThreads/2;s>0;s>>=1){
        if(tid<s){sBx[tid]+=sBx[tid+s];sBy[tid]+=sBy[tid+s];sBz[tid]+=sBz[tid+s];}
        __syncthreads();
    }
    if(tid==0){
        out_B[ip*3+0]+=sBx[0];
        out_B[ip*3+1]+=sBy[0];
        out_B[ip*3+2]+=sBz[0];
    }
}
"""

# ═══════════════════════════════════════════════════════════════
# Kernel caching
# ═══════════════════════════════════════════════════════════════

def _get_poly_kernel():
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")
    if not hasattr(_get_poly_kernel, '_cached'):
        _get_poly_kernel._cached = cp.RawKernel(_POLY_KERNEL_FP64, 'polyhedron_field_fp64')
    return _get_poly_kernel._cached

def _get_recmag_kernel():
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")
    if not hasattr(_get_recmag_kernel, '_cached'):
        _get_recmag_kernel._cached = cp.RawKernel(_RECMAG_KERNEL_FP64, 'recmag_field_fp64')
    return _get_recmag_kernel._cached


# ═══════════════════════════════════════════════════════════════
# GPU geometry cache
# ═══════════════════════════════════════════════════════════════

class GPUGeometry:
    """Geometry data cached on GPU memory.

    Parameters
    ----------
    geo : FlatGeometry
        Flattened geometry from flatten().
    """

    def __init__(self, geo):
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")

        self.n_elem = geo.n_elem
        self.n_rec = geo.n_rec

        dtype = np.float64

        # Polyhedra
        if geo.n_elem > 0:
            self.d_magn = cp.asarray(geo.magnetizations.ravel().astype(dtype))
            self.d_face_offsets = cp.asarray(geo.face_offsets)
            self.d_edge_offsets = cp.asarray(geo.edge_offsets)
            self.d_edge_pts_2d = cp.asarray(geo.face_edges_2d.ravel().astype(dtype))
            self.d_face_cz = cp.asarray(geo.face_coord_z.astype(dtype))
            self.d_face_rot = cp.asarray(geo.face_transforms.reshape(-1, 9).ravel().astype(dtype))
            self.d_face_orig = cp.asarray(geo.face_origins.ravel().astype(dtype))
        else:
            self.d_magn = None
            self.d_face_offsets = None
            self.d_edge_offsets = None
            self.d_edge_pts_2d = None
            self.d_face_cz = None
            self.d_face_rot = None
            self.d_face_orig = None

        # RecMag
        if geo.n_rec > 0:
            self.d_rec_centers = cp.asarray(geo.rec_centers.ravel().astype(dtype))
            self.d_rec_dims = cp.asarray(geo.rec_dimensions.ravel().astype(dtype))
            self.d_rec_magn = cp.asarray(geo.rec_magnetizations.ravel().astype(dtype))
        else:
            self.d_rec_centers = None
            self.d_rec_dims = None
            self.d_rec_magn = None


# ═══════════════════════════════════════════════════════════════
# Symmetry handling
# ═══════════════════════════════════════════════════════════════

def _build_symmetry_transforms(symmetries):
    """Build all 2^N symmetry transform pairs.

    Parameters
    ----------
    symmetries : list of tuples
        [('perp', [0,0,0], [1,0,0]), ...]

    Returns
    -------
    list of (T, M) tuples
        T: 3x3 point transform matrix
        M: 3x3 field transform matrix

    Notes
    -----
    TrfZerPerp: B transforms as R @ B (normal component antisymmetric)
    TrfZerPara: B transforms as -R @ B (tangential components antisymmetric)

    TODO: Support mirror planes through arbitrary points (not just origin).
    For non-origin planes: translate to origin, reflect, translate back.
    """
    if symmetries is None:
        return [(np.eye(3), np.eye(3))]

    n_sym = len(symmetries)
    ref_matrices, ref_types = [], []
    for sym_type, point, normal in symmetries:
        n = np.array(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        R = np.eye(3) - 2.0 * np.outer(n, n)
        ref_matrices.append(R)
        ref_types.append(sym_type)

    transforms = []
    for combo in range(2 ** n_sym):
        T = np.eye(3)
        M = np.eye(3)
        for bit in range(n_sym):
            if combo & (1 << bit):
                R = ref_matrices[bit]
                T = R @ T
                if ref_types[bit] == 'perp':
                    M = R @ M
                else:
                    M = -R @ M
        transforms.append((T, M))

    return transforms


# ═══════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════

def fld_gpu(geo, points, component='b', gpu_geo=None, symmetries=None,
            coil_obj=None):
    """Compute magnetic field on GPU.

    Parameters
    ----------
    geo : FlatGeometry
        Flattened geometry from flatten().
    points : array_like, shape (Np, 3)
        Observation points in mm.
    component : str
        'b' for full B vector (Np,3), 'bx'/'by'/'bz' for components (Np,).
    gpu_geo : GPUGeometry, optional
        Pre-cached GPU data. Created automatically if None.
    symmetries : list of tuples, optional
        Mirror symmetries: [('perp', [0,0,0], [1,0,0]), ...]
        All planes must pass through origin (non-origin TODO).
    coil_obj : int, optional
        Radia object ID for coils/current sources. Computed on CPU
        and added to GPU result.

    Returns
    -------
    ndarray
        Field values. Shape (Np, 3) for 'b', (Np,) for components.

    Examples
    --------
    > geo = flatten(container)
    > gpu_geo = GPUGeometry(geo)
    > symmetries = [('perp', [0,0,0], [1,0,0]),
    ...              ('para', [0,0,0], [0,0,1])]
    > B = fld_gpu(geo, points, gpu_geo=gpu_geo, symmetries=symmetries,
    ...            coil_obj=coils)
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    Np = len(pts)

    if gpu_geo is None:
        gpu_geo = GPUGeometry(geo)

    sym_transforms = _build_symmetry_transforms(symmetries)

    block_size = 256
    B_total = np.zeros((Np, 3), dtype=np.float64)

    for T_pts, M_fld in sym_transforms:
        pts_sym = (pts @ T_pts.T).astype(np.float64)
        d_pts = cp.asarray(pts_sym.ravel())
        d_out_B = cp.zeros(Np * 3, dtype=np.float64)

        if gpu_geo.n_elem > 0:
            _get_poly_kernel()((Np,), (block_size,), (
                d_pts, np.int32(Np),
                gpu_geo.d_magn, np.int32(gpu_geo.n_elem),
                gpu_geo.d_face_offsets, gpu_geo.d_edge_offsets,
                gpu_geo.d_edge_pts_2d, gpu_geo.d_face_cz,
                gpu_geo.d_face_rot, gpu_geo.d_face_orig,
                d_out_B
            ))

        if gpu_geo.n_rec > 0:
            _get_recmag_kernel()((Np,), (block_size,), (
                d_pts, np.int32(Np),
                gpu_geo.d_rec_centers, gpu_geo.d_rec_dims, gpu_geo.d_rec_magn,
                np.int32(gpu_geo.n_rec),
                d_out_B
            ))

        B_sym = d_out_B.get().reshape(Np, 3)
        B_total += B_sym @ M_fld.T

    # Add coil contribution from Radia CPU (analytical, fast)
    if coil_obj is not None:
        import radia as rad
        B_coil = np.array(rad.Fld(coil_obj, 'b', pts.tolist()))
        B_total += B_coil

    if component == 'b':
        return B_total
    elif component == 'bx':
        return B_total[:, 0]
    elif component == 'by':
        return B_total[:, 1]
    elif component == 'bz':
        return B_total[:, 2]
    else:
        raise ValueError(f"Unknown component: {component}")
