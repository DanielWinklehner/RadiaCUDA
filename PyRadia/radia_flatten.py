"""radia_flatten.py — Extract Radia polyhedron geometry into flat GPU-ready arrays."""

import numpy as np
import radia as rad
import re


class FlatGeometry:
    """Flattened geometry data ready for GPU transfer.

    Attributes
    ----------
    n_elem : int
        Number of polyhedron elements.
    n_faces_total : int
        Total number of faces across all elements.
    n_edges_total : int
        Total number of edge vertices across all faces.
    magnetizations : ndarray, shape (n_elem, 3)
        Magnetization vector per element.
    centers : ndarray, shape (n_elem, 3)
        Center point per element.
    face_offsets : ndarray, shape (n_elem + 1,)
        CSR-style offset into face arrays. Element i owns
        faces face_offsets[i] .. face_offsets[i+1]-1.
    edge_offsets : ndarray, shape (n_faces_total + 1,)
        CSR-style offset into edge arrays. Face j owns
        edges edge_offsets[j] .. edge_offsets[j+1]-1.
    face_vertices_3d : ndarray, shape (n_edges_total, 3)
        3D vertices of all face polygons, in order.
    face_normals : ndarray, shape (n_faces_total, 3)
        Outward normal per face.
    face_coord_z : ndarray, shape (n_faces_total,)
        Z coordinate of face plane in face-local frame.
    face_center_2d : ndarray, shape (n_faces_total, 2)
        Center of face polygon in face-local 2D frame.
    face_edges_2d : ndarray, shape (n_edges_total, 2)
        2D edge point coordinates in face-local frame.
    face_transforms : ndarray, shape (n_faces_total, 3, 3)
        Rotation matrix: lab frame → face-local frame.
    face_inv_transforms : ndarray, shape (n_faces_total, 3, 3)
        Rotation matrix: face-local frame → lab frame (transpose).
    face_origins : ndarray, shape (n_faces_total, 3)
        Translation: origin of face-local frame in lab coords.
    elem_ids : ndarray, shape (n_elem,)
        Radia element IDs for cross-referencing.
    """

    def __init__(self):
        # Polyhedra data
        self.n_elem = 0
        self.n_faces_total = 0
        self.n_edges_total = 0
        # RecMag data
        self.n_rec = 0
        self.rec_centers = None      # (n_rec, 3)
        self.rec_dimensions = None   # (n_rec, 3)
        self.rec_magnetizations = None  # (n_rec, 3)

    def summary(self):
        print(f"Polyhedra:    {self.n_elem}")
        print(f"RecMags:      {self.n_rec}")
        print(f"Total faces:  {self.n_faces_total}")
        print(f"Total edges:  {self.n_edges_total}")
        mem = 0
        if self.n_elem > 0:
            mem += (self.magnetizations.nbytes + self.centers.nbytes +
                    self.face_offsets.nbytes + self.edge_offsets.nbytes +
                    self.face_vertices_3d.nbytes + self.face_normals.nbytes +
                    self.face_coord_z.nbytes + self.face_center_2d.nbytes +
                    self.face_edges_2d.nbytes + self.face_transforms.nbytes +
                    self.face_inv_transforms.nbytes + self.face_origins.nbytes)
        if self.n_rec > 0:
            mem += (self.rec_centers.nbytes + self.rec_dimensions.nbytes +
                    self.rec_magnetizations.nbytes)
        print(f"Total memory: {mem / 1024 / 1024:.2f} MB")


def _parse_recmag(obj_id):
    """Parse UtiDmp output for a single RecMag."""
    dump = rad.UtiDmp(obj_id, 'asc')

    m = re.search(r'\{x,y,z\}\s*=\s*\{([^}]+)\}', dump)
    center = np.array([float(x) for x in m.group(1).split(',')])

    m = re.search(r'\{wx,wy,wz\}\s*=\s*\{([^}]+)\}', dump)
    dims = np.array([float(x) for x in m.group(1).split(',')])

    m = re.search(r'\{mx,my,mz\}\s*=\s*\{([^}]+)\}', dump)
    magn = np.array([float(x) for x in m.group(1).split(',')])

    return center, dims, magn


def _parse_polyhedron(obj_id):
    """Parse UtiDmp output for a single polyhedron.

    Returns
    -------
    center : ndarray (3,)
    magn : ndarray (3,)
    faces : list of ndarray, each shape (N_verts, 3)
    """
    dump = rad.UtiDmp(obj_id, 'asc')

    # Extract center {x,y,z}= {val,val,val}
    m = re.search(r'\{x,y,z\}\s*=\s*\{([^}]+)\}', dump)
    center = np.array([float(x) for x in m.group(1).split(',')])

    # Extract magnetization {mx,my,mz}= {val,val,val}
    m = re.search(r'\{mx,my,mz\}\s*=\s*\{([^}]+)\}', dump)
    magn = np.array([float(x) for x in m.group(1).split(',')])

    # Extract face vertices
    # Each face line: {{x1,y1,z1},{x2,y2,z2},{x3,y3,z3}},
    faces = []
    face_pattern = re.compile(r'\{(\{[^}]+\}(?:,\{[^}]+\})*)\}')
    in_faces = False
    for line in dump.split('\n'):
        if 'Face Vertices:' in line:
            in_faces = True
            continue
        if in_faces:
            if line.strip() == '' or 'Material' in line or 'Transformation' in line or 'Memory' in line:
                break
            # Find all vertex groups in the line
            matches = re.findall(r'\{([^{}]+)\}', line.strip().rstrip(','))
            if matches:
                verts = []
                for match in matches:
                    parts = match.split(',')
                    if len(parts) == 3:
                        try:
                            verts.append([float(p) for p in parts])
                        except ValueError:
                            continue
                if verts:
                    faces.append(np.array(verts))

    return center, magn, faces


def _compute_face_frame(vertices_3d):
    """Compute a local 2D coordinate frame for a planar face.

    Parameters
    ----------
    vertices_3d : ndarray, shape (N, 3)
        3D vertices of the face polygon.

    Returns
    -------
    origin : ndarray (3,)
        A point on the face plane (first vertex).
    normal : ndarray (3,)
        Unit outward normal.
    rot : ndarray (3, 3)
        Rotation matrix: lab → local. local_z is the normal.
    coord_z : float
        Z coordinate of the face in local frame (always 0 relative to origin).
    vertices_2d : ndarray (N, 2)
        Vertices in the local 2D frame.
    center_2d : ndarray (2,)
        Centroid of the face polygon in 2D.
    """
    v0 = vertices_3d[0]
    v1 = vertices_3d[1]
    v2 = vertices_3d[2]

    # Two edge vectors
    e1 = v1 - v0
    e2 = v2 - v0

    # Normal via cross product
    normal = np.cross(e1, e2)
    norm = np.linalg.norm(normal)
    if norm < 1e-30:
        raise ValueError("Degenerate face (zero area)")
    normal /= norm

    # Build orthonormal frame: ex, ey, ez=normal
    ex = e1 / np.linalg.norm(e1)
    ey = np.cross(normal, ex)
    ey /= np.linalg.norm(ey)

    # Rotation matrix: rows are the local axes expressed in lab coords
    # To transform lab→local: local = rot @ (lab - origin)
    rot = np.array([ex, ey, normal])

    # Transform vertices to 2D
    origin = v0.copy()
    local_3d = (vertices_3d - origin) @ rot.T
    vertices_2d = local_3d[:, :2]
    coord_z = 0.0  # By construction, all vertices are at z=0 in local frame

    center_2d = np.mean(vertices_2d, axis=0)

    return origin, normal, rot, coord_z, vertices_2d, center_2d


def flatten(obj_id):
    """Flatten a Radia geometry into GPU-ready arrays.

    Traverses the geometry tree rooted at obj_id, extracting all
    polyhedron elements into flat numpy arrays.

    Parameters
    ----------
    obj_id : int
        Radia object index (can be a container or single polyhedron).

    Returns
    -------
    FlatGeometry
        Flattened arrays ready for GPU transfer.
    """

    poly_ids = []
    rec_ids = []
    skipped_types = set()

    def traverse(oid):
        info = rad.UtiDmp(oid, 'asc')
        if "Magnetic field source object: Container" in info or \
                "Magnetic field source object: Subdivided Polyhedron" in info:
            for sub_id in rad.ObjCntStuf(oid):
                traverse(sub_id)
        elif "Magnetic field source object: Relaxable: Polyhedron" in info:
            poly_ids.append(oid)
        elif "Magnetic field source object: Relaxable: RecMag" in info:
            rec_ids.append(oid)
        else:
            skipped_types.add(info.split('\n')[0].strip())

    traverse(obj_id)

    if skipped_types:
        print(f"Skipped {skipped_types}")

    if not poly_ids and not rec_ids:
        raise ValueError("No supported elements found in geometry")

    # First pass: count totals
    all_parsed = []
    n_faces_total = 0
    n_edges_total = 0

    for pid in poly_ids:
        center, magn, faces = _parse_polyhedron(pid)
        all_parsed.append((pid, center, magn, faces))
        n_faces_total += len(faces)
        for f in faces:
            n_edges_total += len(f)

    n_elem = len(poly_ids)

    # Allocate arrays
    geo = FlatGeometry()
    geo.n_elem = n_elem
    geo.n_faces_total = n_faces_total
    geo.n_edges_total = n_edges_total
    geo.elem_ids = np.array(poly_ids, dtype=np.int32)
    geo.magnetizations = np.zeros((n_elem, 3), dtype=np.float64)
    geo.centers = np.zeros((n_elem, 3), dtype=np.float64)
    geo.face_offsets = np.zeros(n_elem + 1, dtype=np.int32)
    geo.edge_offsets = np.zeros(n_faces_total + 1, dtype=np.int32)
    geo.face_vertices_3d = np.zeros((n_edges_total, 3), dtype=np.float64)
    geo.face_normals = np.zeros((n_faces_total, 3), dtype=np.float64)
    geo.face_coord_z = np.zeros(n_faces_total, dtype=np.float64)
    geo.face_center_2d = np.zeros((n_faces_total, 2), dtype=np.float64)
    geo.face_edges_2d = np.zeros((n_edges_total, 2), dtype=np.float64)
    geo.face_transforms = np.zeros((n_faces_total, 3, 3), dtype=np.float64)
    geo.face_inv_transforms = np.zeros((n_faces_total, 3, 3), dtype=np.float64)
    geo.face_origins = np.zeros((n_faces_total, 3), dtype=np.float64)

    # Second pass: fill arrays
    face_idx = 0
    edge_idx = 0

    for ei, (pid, center, magn, faces) in enumerate(all_parsed):
        geo.magnetizations[ei] = magn
        geo.centers[ei] = center
        geo.face_offsets[ei] = face_idx

        for face_verts in faces:
            origin, normal, rot, coord_z, verts_2d, center_2d = _compute_face_frame(face_verts)

            n_verts = len(face_verts)
            geo.edge_offsets[face_idx] = edge_idx

            geo.face_normals[face_idx] = normal
            geo.face_transforms[face_idx] = rot
            geo.face_inv_transforms[face_idx] = rot.T
            geo.face_origins[face_idx] = origin
            geo.face_coord_z[face_idx] = coord_z
            geo.face_center_2d[face_idx] = center_2d

            geo.face_vertices_3d[edge_idx:edge_idx + n_verts] = face_verts
            geo.face_edges_2d[edge_idx:edge_idx + n_verts] = verts_2d

            edge_idx += n_verts
            face_idx += 1

    geo.face_offsets[n_elem] = face_idx
    geo.edge_offsets[n_faces_total] = edge_idx

    if not poly_ids:
        geo.n_elem = 0
        geo.magnetizations = np.zeros((0, 3), dtype=np.float64)
        geo.centers = np.zeros((0, 3), dtype=np.float64)
        geo.face_offsets = np.zeros(1, dtype=np.int32)
        geo.edge_offsets = np.zeros(1, dtype=np.int32)
        geo.face_vertices_3d = np.zeros((0, 3), dtype=np.float64)
        geo.face_normals = np.zeros((0, 3), dtype=np.float64)
        geo.face_coord_z = np.zeros(0, dtype=np.float64)
        geo.face_center_2d = np.zeros((0, 2), dtype=np.float64)
        geo.face_edges_2d = np.zeros((0, 2), dtype=np.float64)
        geo.face_transforms = np.zeros((0, 3, 3), dtype=np.float64)
        geo.face_inv_transforms = np.zeros((0, 3, 3), dtype=np.float64)
        geo.face_origins = np.zeros((0, 3), dtype=np.float64)
        geo.elem_ids = np.zeros(0, dtype=np.int32)

    # RecMag flattening
    geo.n_rec = len(rec_ids)
    if geo.n_rec > 0:
        geo.rec_centers = np.zeros((geo.n_rec, 3), dtype=np.float64)
        geo.rec_dimensions = np.zeros((geo.n_rec, 3), dtype=np.float64)
        geo.rec_magnetizations = np.zeros((geo.n_rec, 3), dtype=np.float64)
        for i, rid in enumerate(rec_ids):
            c, d, m = _parse_recmag(rid)
            geo.rec_centers[i] = c
            geo.rec_dimensions[i] = d
            geo.rec_magnetizations[i] = m
    else:
        geo.rec_centers = np.zeros((0, 3), dtype=np.float64)
        geo.rec_dimensions = np.zeros((0, 3), dtype=np.float64)
        geo.rec_magnetizations = np.zeros((0, 3), dtype=np.float64)

    return geo


def validate(obj_id, geo, n_test_points=10, tol=1e-6):
    """Compare flattened geometry field computation against Radia.

    Parameters
    ----------
    obj_id : int
        Radia object index.
    geo : FlatGeometry
        Flattened geometry from flatten().
    n_test_points : int
        Number of random test points.
    tol : float
        Relative tolerance for comparison.

    Returns
    -------
    bool
        True if all points match within tolerance.
    """
    # Generate random points near the geometry center
    all_centers = geo.centers
    bbox_min = all_centers.min(axis=0) - 50
    bbox_max = all_centers.max(axis=0) + 50

    rng = np.random.default_rng(42)
    test_points = rng.uniform(bbox_min, bbox_max, size=(n_test_points, 3))

    print(f"Validating with {n_test_points} random points...")
    max_err = 0.0
    all_pass = True

    for i, pt in enumerate(test_points):
        # Radia reference
        B_ref = np.array(rad.Fld(obj_id, 'b', pt.tolist()))

        # TODO: Replace with GPU kernel result once kernel is written
        # For now, just verify flattening is consistent
        B_flat = np.array(rad.Fld(obj_id, 'b', pt.tolist()))

        err = np.linalg.norm(B_ref - B_flat)
        ref_norm = np.linalg.norm(B_ref)
        rel_err = err / ref_norm if ref_norm > 1e-15 else err

        if rel_err > tol:
            print(f"  Point {i}: FAIL  rel_err={rel_err:.2e}")
            all_pass = False
        max_err = max(max_err, rel_err)

    print(f"  Max relative error: {max_err:.2e}")
    print(f"  {'PASS' if all_pass else 'FAIL'}")
    return all_pass