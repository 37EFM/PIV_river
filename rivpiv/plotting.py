"""Plots of the velocity field, on the image and in plan view."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .geometry import rectify, vectors_to_world


def show_field_on_image(field, image, scale=None, color="red", every=1, ax=None,
                        show_invalid=False, **kw):
    """Quiver plot of the pixel displacements on top of a frame.

    ``scale`` is the quiver scale in the sense of matplotlib with
    ``scale_units='xy'``: displacement [px] / scale = arrow length [px].
    Default draws the arrows ten times longer than the displacement.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(image, cmap="gray" if np.ndim(image) == 2 else None)
    u, v = field.masked()
    s = every
    ax.quiver(field.x[::s, ::s], field.y[::s, ::s], u[::s, ::s], v[::s, ::s],
              color=color, angles="xy", scale_units="xy", scale=0.1 if scale is None else scale,
              width=0.003, **kw)
    if show_invalid:
        bad = ~field.valid
        ax.plot(field.x[bad], field.y[bad], "x", color="cyan", ms=3, mew=0.6)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    return ax


def world_field(field, cam):
    """Positions [m] and velocities [m/s] of a validated field (NaN where invalid)."""
    u, v = field.masked()
    return vectors_to_world(cam, field.x, field.y, u, v, field.dt)


def show_plan_view(field, cam, image=None, resolution=0.05, extent=None, every=1,
                   ax=None, cmap="viridis", vmax=None, arrow_scale=None, arrows=True, dots=True):
    """Velocity magnitude in plan view with arrows, over the rectified frame.

    The plan view is in world coordinates: ``X`` across the view (camera's
    right is positive) and ``Y`` away from the camera.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 8))
    X, Y, U, V = world_field(field, cam)
    speed = np.hypot(U, V)
    if extent is None:
        ok = np.isfinite(speed)
        extent = (np.nanmin(X[ok]), np.nanmax(X[ok]), np.nanmin(Y[ok]), np.nanmax(Y[ok]))
    if image is not None:
        plan, ext = rectify(cam, image, extent=extent, resolution=resolution)
        ax.imshow(plan, extent=ext, cmap="gray" if plan.ndim == 2 else None,
                  origin="upper", interpolation="nearest",
                  vmin=0, vmax=255 if plan.ndim == 3 else None)
    s = every
    if dots:
        sc = ax.scatter(X, Y, c=speed, cmap=cmap, vmin=0, vmax=vmax, s=12, zorder=2)
        acolor = "w"
    else:
        acolor = None
    if arrows:
        q = ax.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s], *([] if dots else [speed[::s, ::s]]),
                      color=acolor, cmap=None if dots else cmap, clim=(0, vmax) if not dots else None,
                      angles="xy", scale_units="xy", scale=1.0 if arrow_scale is None else arrow_scale,
                      width=0.004, zorder=3)
        if not dots:
            sc = q
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]  (across the view)")
    ax.set_ylabel("Y [m]  (away from the camera)")
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("surface speed [m/s]")
    return ax


def show_shifts(shifts, fps, ax=None):
    """Camera drift in pixels against time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    t = np.arange(len(shifts)) / fps
    ax.plot(t, shifts[:, 0], label="dx")
    ax.plot(t, shifts[:, 1], label="dy")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("camera shift [px]")
    ax.legend()
    return ax


def show_correlation_map(field, j, i, ax=None):
    """The ensemble correlation map of window (j, i): a look under the hood."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    c = field.correlation[j, i]
    R = c.shape[0] // 2
    im = ax.imshow(c, extent=(-R - 0.5, R + 0.5, R + 0.5, -R - 0.5), cmap="magma")
    ax.plot(field.u[j, i], field.v[j, i], "c+", ms=10)
    ax.set_xlabel("dx [px]")
    ax.set_ylabel("dy [px]")
    ax.set_title(f"window ({j}, {i}), SNR = {field.snr[j, i]:.2f}", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax
