"""Removing the motion of a hand-held camera.

A phone held by hand drifts by tens of pixels over a few seconds, which
is comparable to the displacement of the water between frames, and it
also rolls slightly.  The motion is measured on parts of the image that
do not move -- the banks -- and every frame is mapped back onto the
first one.

With one static region only a translation can be found.  With two or
more, a rigid motion (translation plus a rotation about the image
centre) is fitted by least squares to the shifts of the regions, which
takes out the roll of the phone.  What remains is parallax: a bank close
to the camera and trees far away shift by different amounts when the
phone translates.  That cannot be removed without knowing the distances,
so static regions at about the same distance as the water of interest
are the best choice.
"""
from __future__ import annotations

import numpy as np
import scipy.fft
from scipy.ndimage import gaussian_filter, affine_transform

from .piv import _subpixel


def _xcorr_shift(a, b, max_shift):
    """Sub-pixel shift of ``b`` relative to ``a`` (dx, dy), and peak height."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    a = a - a.mean()
    b = b - b.mean()
    H, W = a.shape
    n = (2 * H, 2 * W)
    C = scipy.fft.irfft2(np.conj(scipy.fft.rfft2(a, s=n)) * scipy.fft.rfft2(b, s=n), s=n)
    C = np.fft.fftshift(C)
    cy, cx = H, W
    R = int(max_shift)
    C = C[cy - R:cy + R + 1, cx - R:cx + R + 1]
    d = np.arange(-R, R + 1)
    C = C / np.outer(H - np.abs(d), W - np.abs(d))
    C /= (a.std() * b.std() + 1e-6)
    py, px = np.unravel_index(C.argmax(), C.shape)
    m = C.shape[0]
    cm = lambda dy, dx: C[np.clip(py + dy, 0, m - 1), np.clip(px + dx, 0, m - 1)]
    dx = px - R + float(_subpixel(np.array(cm(0, -1)), np.array(cm(0, 0)), np.array(cm(0, 1))))
    dy = py - R + float(_subpixel(np.array(cm(-1, 0)), np.array(cm(0, 0)), np.array(cm(1, 0))))
    return dx, dy, float(C[py, px])


def estimate_motion(frames, regions, reference=0, max_shift=80, highpass_sigma=5.0, rigid=None):
    """Camera motion of every frame relative to the reference frame.

    Parameters
    ----------
    frames : (n, H, W) array
    regions : list of rectangles [x0, y0, x1, y1] on static parts of the
        scene (banks, vegetation, structures)
    reference : index of the reference frame
    max_shift : largest shift searched [pixels]
    highpass_sigma : Gaussian scale subtracted before correlating
    rigid : fit a rotation as well (default: yes if there are >= 2 regions)

    Returns
    -------
    motion : (n, 3) array of (dx, dy, angle) -- the scene in frame i
        appears translated by (dx, dy) pixels and rotated by ``angle``
        radians (about the image centre, positive = counter-clockwise
        on screen) with respect to the reference frame.
    region_shifts : (n, len(regions), 2) measured shift of each region
    quality : (n, len(regions)) correlation peak heights
    """
    regions = [tuple(int(round(v)) for v in r) for r in regions]
    rigid = len(regions) >= 2 if rigid is None else rigid
    H, W = frames.shape[1:]
    centre = np.array([W / 2.0, H / 2.0])
    crops_ref, centres = [], []
    for x0, y0, x1, y1 in regions:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        g = frames[reference, y0:y1, x0:x1].astype(np.float32)
        crops_ref.append(((x0, y0, x1, y1), g - gaussian_filter(g, highpass_sigma)))
        centres.append([(x0 + x1) / 2.0 - centre[0], (y0 + y1) / 2.0 - centre[1]])
    centres = np.array(centres)
    n = len(frames)
    motion = np.zeros((n, 3))
    region_shifts = np.zeros((n, len(regions), 2))
    quality = np.zeros((n, len(regions)))
    for i in range(n):
        for k, ((x0, y0, x1, y1), ref) in enumerate(crops_ref):
            g = frames[i, y0:y1, x0:x1].astype(np.float32)
            g = g - gaussian_filter(g, highpass_sigma)
            dx, dy, q = _xcorr_shift(ref, g, min(max_shift, (y1 - y0) // 3, (x1 - x0) // 3))
            region_shifts[i, k] = (dx, dy)
            quality[i, k] = q
        w = np.maximum(quality[i], 1e-3)
        if rigid:
            # d_k = t + phi * (-c_ky, c_kx)   (small-angle rotation about the centre)
            A = np.zeros((2 * len(regions), 3))
            b = np.zeros(2 * len(regions))
            for k, (cx_, cy_) in enumerate(centres):
                A[2 * k] = (1, 0, -cy_)
                A[2 * k + 1] = (0, 1, cx_)
                b[2 * k:2 * k + 2] = region_shifts[i, k]
            sw = np.sqrt(np.repeat(w, 2))
            sol = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
            motion[i] = sol
        else:
            motion[i, :2] = (region_shifts[i] * w[:, None]).sum(0) / w.sum()
    return motion, region_shifts, quality


def estimate_shifts(frames, regions, **kw):
    """Translation only; see :func:`estimate_motion`.  Returns (shifts, quality)."""
    motion, _, quality = estimate_motion(frames, regions, rigid=False, **kw)
    return motion[:, :2], quality


def apply_motion(frames, motion, order=1):
    """Map every frame back onto the reference frame."""
    motion = np.asarray(motion, float)
    if motion.shape[1] == 2:
        motion = np.column_stack([motion, np.zeros(len(motion))])
    H, W = frames.shape[1:]
    centre = np.array([H / 2.0, W / 2.0])          # (row, col)
    out = np.empty_like(frames)
    for i, (dx, dy, phi) in enumerate(motion):
        # output pixel p (row, col) comes from input pixel R(phi) (p - c) + c + (dy, dx)
        c, s = np.cos(phi), np.sin(phi)
        # x' = x cos - y sin + dx, y' = x sin + y cos + dy, written in (row, col) order
        R = np.array([[c, s], [-s, c]])
        offset = centre - R @ centre + np.array([dy, dx])
        f = affine_transform(frames[i].astype(np.float32), R, offset=offset, order=order, mode="nearest")
        out[i] = np.clip(np.rint(f), 0, 255).astype(frames.dtype) if frames.dtype == np.uint8 else f
    return out


apply_shifts = apply_motion
