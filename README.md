# Radia
3D Magnetostatics Computer Code

## Field reproducibility

By default Radia adds a tiny (~1e-9) `rand()`-based perturbation to length values in
some field formulas (e.g. extruded polygons, arcs) to avoid on-edge singularities, so
those field values are not bit-reproducible from call to call (rectangular blocks and
polyhedra are unaffected). Call `radia.FldLenRndSw('off')` for deterministic results —
recommended when comparing CPU vs GPU or writing regression tests; `radia.FldLenTol(...)`
tunes the magnitude.

## GPU field evaluation

`rad.Fld(..., use_gpu=True)` runs magnetized objects (blocks, polyhedra, extruded polygons)
on the GPU and adds current-source fields on the CPU. **Do not apply a symmetry or transform
to a current-carrying object** (`ObjRecCur`, `ObjArcCur`, `ObjRaceTrk`, `ObjFlmCur`) — or to a
container holding one — if you want GPU acceleration: the GPU coil path is not symmetry-aware,
so such a model falls back entirely to the CPU (results stay correct, just slower). For best
GPU performance, apply symmetries to the magnetized objects (e.g. the iron yoke) and keep
current-carrying objects (coils) directly placed.
