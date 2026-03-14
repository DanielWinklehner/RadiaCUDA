# examples/uti_plot.py
"""Minimal plotting utilities compatible with RADIA example scripts."""

import matplotlib.pyplot as plt
import numpy as np


def uti_plot1d(data, mesh, labels=None, units=None):
    """Plot a 1D dataset.

    Parameters
    ----------
    data : list or array
        Y-axis values.
    mesh : list
        [min, max, npoints] defining the X-axis grid.
    labels : list of str, optional
        [x_label, y_label, title].
    units : list of str, optional
        [x_unit, y_unit] appended to axis labels.
    """
    x_min, x_max, n = mesh[0], mesh[1], int(mesh[2])
    x = np.linspace(x_min, x_max, n)

    fig, ax = plt.subplots()
    ax.plot(x, data)
    _apply_labels(ax, labels, units)


def uti_plot1d_m(datasets, labels=None, units=None, styles=None,
                 legend=None):
    """Plot multiple 1D datasets on the same axes.

    Parameters
    ----------
    datasets : list of list
        Each element is a list of [x, y] pairs (as returned by
        ``rad.FldLst`` with ``'arg'``).
    labels : list of str, optional
        [x_label, y_label, title].
    units : list of str, optional
        [x_unit, y_unit].
    styles : list of str, optional
        Matplotlib format strings, one per dataset.
    legend : list of str, optional
        Legend entries.
    """
    fig, ax = plt.subplots()
    for i, ds in enumerate(datasets):
        arr = np.array(ds)
        x = arr[:, 0]
        y = arr[:, 1]
        fmt = styles[i] if styles and i < len(styles) else '-'
        fmt = fmt.replace('.', '')  # strip marker-dot shorthand
        ax.plot(x, y, fmt)
    _apply_labels(ax, labels, units)
    if legend:
        ax.legend(legend)


def uti_plot2d1d(data, mesh_x, mesh_y, x=0, y=0, labels=None,
                 units=None):
    """Plot a 2D color map with 1D cross-section cuts.

    Parameters
    ----------
    data : list
        Flat list of values in row-major order (y varies slowest).
    mesh_x : list
        [x_min, x_max, nx].
    mesh_y : list
        [y_min, y_max, ny].
    x : float
        X position for the vertical cross-section.
    y : float
        Y position for the horizontal cross-section.
    labels : tuple of str, optional
        (x_label, y_label, title).
    units : list of str, optional
        [x_unit, y_unit, z_unit].
    """
    nx = int(mesh_x[2])
    ny = int(mesh_y[2])
    xv = np.linspace(mesh_x[0], mesh_x[1], nx)
    yv = np.linspace(mesh_y[0], mesh_y[1], ny)
    z = np.array(data).reshape(ny, nx)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 2D map
    ax0 = axes[0]
    im = ax0.pcolormesh(xv, yv, z, shading="auto")
    fig.colorbar(im, ax=ax0)
    if labels and len(labels) >= 3:
        ax0.set_xlabel(labels[0])
        ax0.set_ylabel(labels[1])
        ax0.set_title(labels[2])

    # Horizontal cut at y
    iy = int(np.argmin(np.abs(yv - y)))
    axes[1].plot(xv, z[iy, :])
    axes[1].set_xlabel(labels[0] if labels else "X")
    axes[1].set_title("Cut at Y={:.3g}".format(yv[iy]))

    # Vertical cut at x
    ix = int(np.argmin(np.abs(xv - x)))
    axes[2].plot(yv, z[:, ix])
    axes[2].set_xlabel(labels[1] if labels else "Y")
    axes[2].set_title("Cut at X={:.3g}".format(xv[ix]))

    fig.tight_layout()


def uti_plot_show():
    """Display all pending figures."""
    plt.show()


def _apply_labels(ax, labels, units):
    """Apply axis labels and title to an Axes object."""
    if not labels:
        return
    xl = labels[0] if len(labels) > 0 else ""
    yl = labels[1] if len(labels) > 1 else ""
    title = labels[2] if len(labels) > 2 else ""
    if units:
        if len(units) > 0 and units[0]:
            xl += " [{}]".format(units[0])
        if len(units) > 1 and units[1]:
            yl += " [{}]".format(units[1])
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
