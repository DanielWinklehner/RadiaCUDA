"""field_gpu.py — High-level GPU field computation for Radia models.

Provides a single function FldGPU() that mirrors Radia's rad.Fld() interface
but runs on GPU. Automatically handles geometry flattening, coil separation,
and GPU caching.

Usage:
    from PyRadia.field_gpu import FldGPU

    B = FldGPU(model_id, points, symmetries=[
        ('perp', [0,0,0], [1,0,0]),
        ('para', [0,0,0], [0,0,1]),
    ])
"""

import numpy as np
import radia as rad
from .radia_flatten import flatten
from .field_kernel import fld_gpu, GPUGeometry


# ═══════════════════════════════════════════════════════════════
# Cache — avoids re-flattening and re-uploading on repeated calls
# ═══════════════════════════════════════════════════════════════

_cache = {}


def invalidate_cache(obj_id=None):
    """Clear cached GPU data.

    Call after modifying geometry (e.g. re-solving).

    Parameters
    ----------
    obj_id : int, optional
        Clear cache for specific object. If None, clear all.
    """
    global _cache
    if obj_id is None:
        _cache.clear()
    elif obj_id in _cache:
        del _cache[obj_id]


# ═══════════════════════════════════════════════════════════════
# Geometry analysis
# ═══════════════════════════════════════════════════════════════

def _classify_objects(obj_id):
    """Traverse a Radia geometry tree and classify objects.

    Returns
    -------
    iron_ids : list of int
        IDs of magnetizable elements (Polyhedron, RecMag).
    coil_ids : list of int
        IDs of current-carrying elements (ArcCur, RaceTrk, etc.).
    skipped : set of str
        Description strings of unrecognized element types.
    """
    iron_ids = []
    coil_ids = []
    skipped = set()

    def traverse(oid):
        info = rad.UtiDmp(oid, 'asc')
        if "Magnetic field source object: Container" in info or \
                "Magnetic field source object: Subdivided Polyhedron" in info:
            for sub_id in rad.ObjCntStuf(oid):
                traverse(sub_id)
        elif "Magnetic field source object: Relaxable: Polyhedron" in info:
            iron_ids.append(oid)
        elif "Magnetic field source object: Relaxable: RecMag" in info:
            iron_ids.append(oid)
        elif "Magnetic field source object: Current carrying:" in info:
            coil_ids.append(oid)
        else:
            skipped.add(info.split('\n')[0].strip())

    traverse(obj_id)
    return iron_ids, coil_ids, skipped


def _find_coil_container(obj_id):
    """Find the highest-level container holding only current-carrying elements.

    Walks top-level children. If a child subtree contains ONLY coils
    (no iron), it's a coil container.

    Returns
    -------
    coil_obj : int or None
        Radia object ID for computing coil field via rad.Fld().
        None if no coils found.
    """
    info = rad.UtiDmp(obj_id, 'asc')

    if "Current carrying:" in info:
        return obj_id

    if "Container" not in info and "Subdivided" not in info:
        return None

    children = rad.ObjCntStuf(obj_id)
    coil_children = []

    for child_id in children:
        child_iron, child_coils, _ = _classify_objects(child_id)
        if len(child_coils) > 0 and len(child_iron) == 0:
            coil_children.append(child_id)

    if len(coil_children) == 0:
        _, all_coils, _ = _classify_objects(obj_id)
        if len(all_coils) == 0:
            return None
        coil_container = rad.ObjCnt(all_coils)
        return coil_container
    elif len(coil_children) == 1:
        return coil_children[0]
    else:
        coil_container = rad.ObjCnt(coil_children)
        return coil_container


# ═══════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════

def FldGPU(obj_id, points, component='b', symmetries=None,
           cache=True, verbose=False, rank=0, comm=None):
    """Compute magnetic field using GPU acceleration.

    Drop-in acceleration for rad.Fld(). On first call, analyzes the
    geometry, separates iron from coils, flattens iron for GPU, and
    caches everything. Subsequent calls reuse the cache.

    Iron field is computed on GPU (rank 0 only). Coil field (analytical)
    is computed on CPU via Radia using MPI across all ranks.

    All ranks must call this function when MPI is active, so that
    rad.Fld() for coils can operate collectively.

    Parameters
    ----------
    obj_id : int
        Radia object index (top-level container).
    points : array_like
        Observation points. Can be:
        - shape (3,) for a single point
        - shape (Np, 3) for multiple points
        - list of [x, y, z] lists (Radia-compatible format)
    component : str
        'b' for full B vector, 'bx'/'by'/'bz' for components.
    symmetries : list of tuples or None
        Mirror symmetries applied to the geometry. Must match the
        symmetries used in the Radia model. Pass None if no
        symmetries were applied.

        Format: [('perp', [px,py,pz], [nx,ny,nz]), ...]
          'perp' = TrfZerPerp (field normal to plane is zero)
          'para' = TrfZerPara (field parallel to plane is zero)

        NOTE: Only origin-centered planes ([0,0,0]) are currently
        supported. Non-origin planes are on the TODO list.

        Example for a cyclotron with 4 mirrors (1/16th model):
            symmetries = [
                ('perp', [0,0,0], [1,-1,0]),
                ('perp', [0,0,0], [1,0,0]),
                ('perp', [0,0,0], [0,1,0]),
                ('para', [0,0,0], [0,0,1]),
            ]
    cache : bool
        If True (default), cache flattened geometry and GPU data
        between calls. Call invalidate_cache() after re-solving
        or modifying geometry.
    verbose : bool
        Print diagnostic info on first call (use verbose=(rank==0)
        with MPI to avoid duplicate output).
    rank : int
        MPI rank. Only rank 0 performs GPU computation and geometry
        analysis. Default 0 (single-process mode).
    comm : MPI communicator or None
        MPI communicator from mpi4py (e.g. MPI.COMM_WORLD).
        Required when running with MPI so that coil object ID
        can be broadcast to all ranks.

    Returns
    -------
    ndarray or None
        On rank 0: field values. Shape (Np, 3) for 'b', (Np,) for
        components. For a single point with component='b', returns
        shape (3,).
        On rank > 0: returns None.

    Examples
    --------
    Single process:

    > B = FldGPU(model, pts, symmetries=symmetries)

    With MPI:

    > from mpi4py import MPI
    > comm = MPI.COMM_WORLD
    > rank = comm.Get_rank()
    > B = FldGPU(model, pts, symmetries=symmetries,
    ...            rank=rank, comm=comm, verbose=(rank==0))
    > if rank == 0:
    ...     print(B.shape)

    After re-solving:

    > invalidate_cache(model)
    > B = FldGPU(model, pts, symmetries=symmetries,
    ...            rank=rank, comm=comm)
    """
    global _cache

    # ── Normalize points (all ranks need this for rad.Fld) ──
    pts = np.asarray(points, dtype=np.float64)
    single_point = False
    if pts.ndim == 1:
        if len(pts) == 3:
            pts = pts.reshape(1, 3)
            single_point = True
        else:
            raise ValueError(f"Expected 3D point(s), got shape {pts.shape}")
    elif pts.ndim == 2:
        if pts.shape[1] != 3:
            raise ValueError(f"Expected shape (Np, 3), got {pts.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D array, got {pts.ndim}D")

    # ── Rank 0: analyze geometry, flatten, upload to GPU ──
    geo = None
    gpu_geo = None
    coil_id = -1

    if rank == 0:
        if cache and obj_id in _cache:
            cached = _cache[obj_id]
            geo = cached['geo']
            gpu_geo = cached['gpu_geo']
            coil_id = cached['coil_obj'] if cached['coil_obj'] is not None else -1
        else:
            if verbose:
                print(f"FldGPU: Analyzing Radia object {obj_id}...",
                      flush=True)

            iron_ids, coil_ids, skipped = _classify_objects(obj_id)

            if verbose:
                print(f"  Iron elements: {len(iron_ids)}", flush=True)
                print(f"  Coil elements: {len(coil_ids)}", flush=True)
                if skipped:
                    print(f"  Skipped:       {skipped}", flush=True)

            if len(iron_ids) == 0 and len(coil_ids) == 0:
                raise ValueError("No supported elements found in geometry")

            coil_obj = _find_coil_container(obj_id) if coil_ids else None
            coil_id = coil_obj if coil_obj is not None else -1

            if verbose and coil_obj is not None:
                print(f"  Coil object:   {coil_obj}", flush=True)

            if iron_ids:
                if verbose:
                    print("  Flattening...", flush=True)
                geo = flatten(obj_id)
                if verbose:
                    geo.summary()
            else:
                geo = None

            gpu_geo = GPUGeometry(geo) if geo is not None else None

            if verbose:
                print("  GPU ready.", flush=True)

            if cache:
                _cache[obj_id] = {
                    'geo': geo,
                    'gpu_geo': gpu_geo,
                    'coil_obj': coil_obj,
                }

    # ── Broadcast coil ID so all ranks can participate in rad.Fld ──
    if comm is not None:
        coil_id = comm.bcast(coil_id, root=0)

    # ── Coil field: ALL ranks participate (MPI collective) ──
    B_coil = None
    if coil_id >= 0:
        B_coil = np.array(rad.Fld(coil_id, 'b', pts.tolist()))

    # ── GPU iron field: rank 0 only ──
    if rank == 0:
        if geo is not None and gpu_geo is not None:
            result = fld_gpu(geo, pts, component='b', gpu_geo=gpu_geo,
                             symmetries=symmetries)
        else:
            result = np.zeros((len(pts), 3), dtype=np.float64)

        if B_coil is not None:
            result += B_coil

        if component == 'b':
            return result[0] if single_point else result
        elif component in ('bx', 'by', 'bz'):
            idx = {'bx': 0, 'by': 1, 'bz': 2}[component]
            col = result[:, idx]
            return float(col[0]) if single_point else col
        else:
            raise ValueError(f"Unknown component: {component}")

    # Ranks > 0: no result
    return None