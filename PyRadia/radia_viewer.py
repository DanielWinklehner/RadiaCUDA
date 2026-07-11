"""Lightweight 3D viewer for Radia geometries using PyVista."""

import sys

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pv = None
    HAS_PYVISTA = False

_DPI_AWARE_SET = False


def ensure_dpi_aware():
    """Make the process per-monitor DPI-aware on Windows before a pyvista/VTK
    window opens, so the 3D view renders at NATIVE resolution (crisp) instead
    of being bitmap-upscaled by the OS on a scaled high-DPI display.

    This is the same process-wide setting that Qt-based toolkits (e.g.
    matplotlib's Qt backend) apply as a side effect of opening a window --
    which is why a Radia view was only sharp after a matplotlib figure had been
    shown. Call it before creating the render window. Idempotent; a no-op off
    Windows, if already set, or if the API is unavailable.
    """
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET or sys.platform != "win32":
        return
    _DPI_AWARE_SET = True
    import ctypes

    # Newest -> oldest; each raises AttributeError on Windows too old for it.
    # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (matches Qt6/PySide6).
    for setter in (
        lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)),
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),   # PER_MONITOR
        lambda: ctypes.windll.user32.SetProcessDPIAware(),        # system-aware
    ):
        try:
            setter()
            return
        except Exception:
            continue


def ObjDrwPyVista(obj, opacity=1.0, show_edges=True):
    """Display a Radia object in an interactive 3D viewer.

    Parameters
    ----------
    obj : int
        Radia object index.
    opacity : float
        Opacity of the surfaces (0.0 to 1.0).
    show_edges : bool
        Whether to draw mesh edges.
    """
    import radia as rad

    if not HAS_PYVISTA:
        print("PyVista not installed. Run: pip install pyvista")
        return

    data = rad.ObjDrwVTK(obj, 'EdgeLines->False')

    ensure_dpi_aware()  # crisp on scaled high-DPI Windows; must precede the window
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Draw polygons (solid volumes)
    pgns = data.get("polygons", {})
    _add_vtk_data(plotter, pgns, opacity=opacity, show_edges=show_edges)

    # Draw lines (coils, axes, etc.)
    lines = data.get("lines", {})
    _add_vtk_lines(plotter, lines)

    plotter.add_axes()
    plotter.show()


def _add_vtk_data(plotter, pgn_data, opacity=1.0, show_edges=True):
    """Convert Radia polygon data to PyVista meshes and add to plotter."""
    if not pgn_data:
        return

    vertices = pgn_data.get("vertices", [])
    lengths = pgn_data.get("lengths", [])
    colors = pgn_data.get("colors", [])

    if len(vertices) == 0 or len(lengths) == 0:
        return

    verts = np.array(vertices).reshape(-1, 3)
    lens = np.array(lengths)
    cols = np.array(colors).reshape(-1, 3) if len(colors) > 0 else None

    # Build PyVista faces array
    faces = []
    vert_offset = 0
    face_colors = []

    for i, n in enumerate(lens):
        face = [int(n)] + list(range(vert_offset, vert_offset + int(n)))
        faces.extend(face)
        vert_offset += int(n)
        if cols is not None and i < len(cols):
            face_colors.append(cols[i])

    faces = np.array(faces)
    mesh = pv.PolyData(verts, faces=faces)

    if face_colors:
        rgb = (np.array(face_colors) * 255).astype(np.uint8)
        mesh.cell_data["colors"] = rgb
        plotter.add_mesh(mesh, scalars="colors", rgb=True,
                         opacity=opacity, show_edges=show_edges)
    else:
        plotter.add_mesh(mesh, color="steelblue",
                         opacity=opacity, show_edges=show_edges)


def _add_vtk_lines(plotter, line_data):
    """Convert Radia line data to PyVista lines and add to plotter."""
    if not line_data:
        return

    vertices = line_data.get("vertices", [])
    lengths = line_data.get("lengths", [])
    colors = line_data.get("colors", [])

    if len(vertices) == 0 or len(lengths) == 0:
        return

    verts = np.array(vertices).reshape(-1, 3)
    lens = np.array(lengths)
    cols = np.array(colors).reshape(-1, 3) if len(colors) > 0 else None

    vert_offset = 0
    for i, n in enumerate(lens):
        n = int(n)
        if n < 2:
            vert_offset += n
            continue
        pts = verts[vert_offset:vert_offset + n]
        line = pv.lines_from_points(pts)
        color = cols[i].tolist() if cols is not None and i < len(cols) else [1, 0, 0]
        plotter.add_mesh(line, color=color, line_width=2)
        vert_offset += n