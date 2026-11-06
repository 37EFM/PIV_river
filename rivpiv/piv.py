"""Cross-correlation PIV on a sequence of frames.

The algorithm is the textbook one (Adrian & Westerweel, *Particle Image
Velocimetry*, 2011):

1. each frame is high-pass filtered to remove the slowly varying glare
   and reflections, which do not move with the water;
2. the image is cut into square interrogation windows;
3. for each window, the cross-correlation between frame *i* and frame
   *i + dt* is computed by FFT, and its peak gives the displacement;
4. the peak position is refined to sub-pixel precision with a
   three-point Gaussian fit;
5. vectors that disagree with their neighbours (normalized median test,
   Westerweel & Scarano 2005) or whose peak is not clearly above the
   noise are discarded.

In *ensemble* mode the correlation maps of all frame pairs are averaged
before the peak is searched (Meinhart, Wereley & Santiago 2000).  This
is the right choice for a short clip of a river with few tracers: the
mean flow is obtained even where a single pair has no signal.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.fft
from scipy.ndimage import gaussian_filter, maximum_filter


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------
def preprocess(frame, highpass_sigma=8.0, normalize_sigma=None, clip=None):
    """High-pass filter a frame (and optionally equalize its contrast).

    Parameters
    ----------
    frame : 2-D uint8/float array
    highpass_sigma : float or None
        Standard deviation [pixels] of the Gaussian that is subtracted.
        Removes illumination gradients, glare and the reflections of the
        banks and sky, which would otherwise bias the velocity toward zero.
    normalize_sigma : float or None
        If given, divide by the local standard deviation computed with
        this Gaussian scale, so that dull and bright regions of the water
        weigh equally in the correlation.
    clip : float or None
        Clip the result to +-clip (after normalization) to limit the
        influence of a single bright leaf or glint.
    """
    g = np.asarray(frame, np.float32)
    if highpass_sigma:
        g = g - gaussian_filter(g, highpass_sigma)
    if normalize_sigma:
        var = gaussian_filter(g * g, normalize_sigma)
        g = g / np.sqrt(var + 1e-3)
    if clip:
        g = np.clip(g, -clip, clip)
    return g


# ----------------------------------------------------------------------
# window grid and correlation
# ----------------------------------------------------------------------
def window_grid(shape, window=64, overlap=0.5):
    """Top-left corners and centres of the interrogation windows."""
    H, W = shape
    step = max(int(round(window * (1 - overlap))), 1)
    ys = np.arange(0, H - window + 1, step)
    xs = np.arange(0, W - window + 1, step)
    return xs, ys, step


def correlate_pair(a, b, window=64, overlap=0.5, max_shift=None, chunk=400):
    """Normalized cross-correlation maps of all windows of a frame pair.

    Returns
    -------
    xs, ys : window corner coordinates
    corr : array (len(ys), len(xs), 2R+1, 2R+1) with R = max_shift;
        element [j, i, R + dy, R + dx] is the correlation coefficient of
        window (j, i) of ``a`` with the same window of ``b`` displaced
        by (dx, dy) pixels.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    xs, ys, step = window_grid(a.shape, window, overlap)
    R = window // 4 if max_shift is None else int(max_shift)
    n = 2 * window                             # zero padding: no wrap-around
    wa = sliding_window_view(a, (window, window))[::step, ::step]
    wb = sliding_window_view(b, (window, window))[::step, ::step]
    wa = wa[:len(ys), :len(xs)]
    wb = wb[:len(ys), :len(xs)]
    ny, nx = len(ys), len(xs)
    out = np.empty((ny, nx, 2 * R + 1, 2 * R + 1), np.float32)

    # number of overlapping pixels for each displacement (bias correction)
    d = np.arange(-R, R + 1)
    overlap_n = np.outer(window - np.abs(d), window - np.abs(d)).astype(np.float32)

    rows_per_chunk = max(chunk // nx, 1)
    for j0 in range(0, ny, rows_per_chunk):
        A = wa[j0:j0 + rows_per_chunk].reshape(-1, window, window)
        B = wb[j0:j0 + rows_per_chunk].reshape(-1, window, window)
        A = A - A.mean(axis=(1, 2), keepdims=True)
        B = B - B.mean(axis=(1, 2), keepdims=True)
        sa = np.sqrt((A * A).mean(axis=(1, 2)))
        sb = np.sqrt((B * B).mean(axis=(1, 2)))
        FA = scipy.fft.rfft2(A, s=(n, n), workers=-1)
        FB = scipy.fft.rfft2(B, s=(n, n), workers=-1)
        C = scipy.fft.irfft2(np.conj(FA) * FB, s=(n, n), workers=-1)   # C[dy, dx] = sum A(p) B(p + d)
        C = np.fft.fftshift(C, axes=(1, 2))                 # zero displacement at (n//2, n//2)
        c0 = n // 2
        C = C[:, c0 - R:c0 + R + 1, c0 - R:c0 + R + 1]
        C = C / overlap_n / (sa * sb + 1e-6)[:, None, None]  # correlation coefficient
        out[j0:j0 + rows_per_chunk] = C.reshape(-1, nx, 2 * R + 1, 2 * R + 1)
    return xs, ys, out


# ----------------------------------------------------------------------
# peak finding
# ----------------------------------------------------------------------
def _subpixel(cm, c0, cp):
    """Three-point Gaussian peak fit; falls back to a parabola."""
    with np.errstate(all="ignore"):
        ok = (cm > 0) & (c0 > 0) & (cp > 0)
        lm, l0, lp = np.log(np.where(ok, cm, 1)), np.log(np.where(ok, c0, 1)), np.log(np.where(ok, cp, 1))
        gauss = (lm - lp) / (2 * lm - 4 * l0 + 2 * lp)
        parab = (cm - cp) / (2 * cm - 4 * c0 + 2 * cp)
        delta = np.where(ok, gauss, parab)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    return np.clip(delta, -1, 1)


def find_peaks(corr):
    """Displacement (u, v), peak height and signal-to-noise ratio of each window.

    The signal-to-noise ratio is the ratio of the highest peak to the
    second highest *local maximum* of the map (the shoulder of a broad
    main peak does not count).  A ratio below about 1.3 means that the
    displacement is not clearly distinguished from noise.
    """
    ny, nx, m, _ = corr.shape
    R = m // 2
    flat = corr.reshape(ny, nx, -1)
    k = flat.argmax(axis=-1)
    py, px = np.unravel_index(k, (m, m))
    c0 = np.take_along_axis(flat, k[..., None], -1)[..., 0]

    # neighbours (clamped at the border of the search area)
    jj, ii = np.indices((ny, nx))
    def val(dy, dx):
        return corr[jj, ii, np.clip(py + dy, 0, m - 1), np.clip(px + dx, 0, m - 1)]
    u = px - R + _subpixel(val(0, -1), c0, val(0, 1))
    v = py - R + _subpixel(val(-1, 0), c0, val(1, 0))

    # second peak: the highest local maximum other than the main one
    is_max = corr >= maximum_filter(corr, size=(1, 1, 5, 5), mode="nearest")
    masked = np.where(is_max, corr, -np.inf)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            masked[jj, ii, np.clip(py + dy, 0, m - 1), np.clip(px + dx, 0, m - 1)] = -np.inf
    c1 = masked.reshape(ny, nx, -1).max(axis=-1)
    c1 = np.where(np.isfinite(c1), c1, 0.0)
    snr = c0 / np.maximum(c1, 1e-6)
    at_edge = (px == 0) | (px == m - 1) | (py == 0) | (py == m - 1)
    snr = np.where(at_edge, 0.0, snr)
    return u, v, c0, snr


# ----------------------------------------------------------------------
# PIV drivers
# ----------------------------------------------------------------------
class Field:
    """A velocity field on the window grid, in pixels per frame pair."""

    def __init__(self, x, y, u, v, peak, snr, window, dt_frames, fps):
        self.x, self.y = x, y            # window centres [pixels], 2-D arrays
        self.u, self.v = u, v            # displacements [pixels] over dt_frames
        self.peak, self.snr = peak, snr
        self.window, self.dt_frames, self.fps = window, dt_frames, fps
        self.valid = np.ones_like(u, bool)

    @property
    def dt(self):
        """Time between the two frames of a pair [s]."""
        return self.dt_frames / self.fps

    def masked(self):
        """u, v with invalid vectors set to NaN."""
        u = np.where(self.valid, self.u, np.nan)
        v = np.where(self.valid, self.v, np.nan)
        return u, v


def _grid(frame_shape, window, overlap):
    xs, ys, _ = window_grid(frame_shape, window, overlap)
    X, Y = np.meshgrid(xs + window / 2.0, ys + window / 2.0)
    return X, Y


def ensemble_piv(frames, fps, dt_frames=2, window=64, overlap=0.5, max_shift=None,
                 pairs=None, pair_step=1, prep=None, verbose=True):
    """Ensemble-averaged PIV over all frame pairs (i, i + dt_frames).

    Parameters
    ----------
    frames : (n, H, W) array, raw or stabilized grey levels
    fps : frames per second
    dt_frames : separation between the frames of a pair
    window, overlap : interrogation window size [px] and overlap fraction
    max_shift : largest displacement searched [px]; default window/4
    pairs : explicit list of first-frame indices; default all
    pair_step : stride between first frames (if ``pairs`` is None)
    prep : dict of keyword arguments for :func:`preprocess`
    """
    prep = {} if prep is None else prep
    n = len(frames)
    if pairs is None:
        pairs = range(0, n - dt_frames, pair_step)
    pairs = list(pairs)
    acc = None
    cache = {}

    def get(i):
        if i not in cache:
            cache.clear() if len(cache) > 4 else None
            cache[i] = preprocess(frames[i], **prep)
        return cache[i]

    import time
    t0 = time.time()
    for k, i in enumerate(pairs):
        xs, ys, corr = correlate_pair(get(i), get(i + dt_frames), window, overlap, max_shift)
        acc = corr if acc is None else acc + corr
    if verbose:
        print(f"ensemble PIV: {len(pairs)} pairs, {acc.shape[0]}x{acc.shape[1]} windows of {window} px, "
              f"{time.time() - t0:.0f} s")
    acc /= len(pairs)
    u, v, peak, snr = find_peaks(acc)
    X, Y = _grid(frames.shape[1:], window, overlap)
    fld = Field(X, Y, u, v, peak, snr, window, dt_frames, fps)
    fld.correlation = acc
    fld.n_pairs = len(pairs)
    return fld


def piv_pair(a, b, fps, dt_frames=1, window=64, overlap=0.5, max_shift=None, prep=None):
    """Instantaneous PIV of a single frame pair (already preprocessed if prep is None)."""
    if prep is not None:
        a, b = preprocess(a, **prep), preprocess(b, **prep)
    xs, ys, corr = correlate_pair(a, b, window, overlap, max_shift)
    u, v, peak, snr = find_peaks(corr)
    X, Y = _grid(np.shape(a), window, overlap)
    return Field(X, Y, u, v, peak, snr, window, dt_frames, fps)


def piv_sequence(frames, fps, dt_frames=2, window=64, overlap=0.5, max_shift=None,
                 pair_step=1, prep=None, validate_kw=None, verbose=True):
    """Instantaneous PIV of every pair; returns a list of validated Fields."""
    prep = {} if prep is None else prep
    validate_kw = {} if validate_kw is None else validate_kw
    import time
    t0 = time.time()
    fields = []
    pairs = list(range(0, len(frames) - dt_frames, pair_step))
    for k, i in enumerate(pairs):
        fld = piv_pair(preprocess(frames[i], **prep), preprocess(frames[i + dt_frames], **prep),
                       fps, dt_frames, window, overlap, max_shift)
        validate(fld, **validate_kw)
        fields.append(fld)
    if verbose:
        print(f"instantaneous PIV: {len(pairs)} pairs, {time.time() - t0:.0f} s")
    return fields


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
def normalized_median_test(u, v, threshold=2.0, eps=0.1):
    """Westerweel & Scarano (2005) outlier detection on a vector field.

    Returns a boolean array, True where the vector is an outlier.
    ``eps`` is the noise level in pixels (0.1 is their recommendation).
    """
    def neighbours(a):
        p = np.pad(a, 1, constant_values=np.nan)
        stack = [p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        return np.stack(stack)                       # (8, ny, nx)
    import warnings
    r2 = np.zeros_like(u, float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for comp in (u, v):
            nb = neighbours(comp)
            med = np.nanmedian(nb, axis=0)
            rm = np.nanmedian(np.abs(nb - med), axis=0)
            r = np.abs(comp - med) / (rm + eps)
            r2 += np.where(np.isfinite(r), r, 0) ** 2
    return np.sqrt(r2) > threshold


def validate(field, snr_min=1.3, median_threshold=2.0, mask=None, peak_min=None):
    """Flag bad vectors: low signal-to-noise, outliers, outside the water.

    ``mask`` is a boolean array on the window grid (True = keep) or a
    polygon [(x, y), ...] in pixels around the water surface.
    Sets ``field.valid`` in place and returns it.
    """
    valid = np.isfinite(field.u) & np.isfinite(field.v)
    if snr_min is not None:
        valid &= field.snr >= snr_min
    if peak_min is not None:
        valid &= field.peak >= peak_min
    if mask is not None:
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            valid &= mask
        else:
            valid &= polygon_mask(mask, field.x, field.y)
    if median_threshold is not None:
        u = np.where(valid, field.u, np.nan)
        v = np.where(valid, field.v, np.nan)
        valid &= ~normalized_median_test(u, v, median_threshold)
    field.valid = valid
    return valid


def polygon_mask(polygon, x, y):
    """True for the points (x, y) inside the polygon [(x, y), ...]."""
    from matplotlib.path import Path
    pts = np.column_stack([np.ravel(x), np.ravel(y)])
    return Path(np.asarray(polygon, float)).contains_points(pts).reshape(np.shape(x))


def fill_invalid(field, iterations=3):
    """Replace invalid vectors by the mean of their valid neighbours.

    Returns (u, v) arrays; vectors with no valid neighbour stay NaN.
    """
    u, v = field.masked()
    for _ in range(iterations):
        for a in (u, v):
            p = np.pad(a, 1, constant_values=np.nan)
            nb = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
            with np.errstate(all="ignore"):
                fillv = np.nanmean(nb, axis=0)
            hole = np.isnan(a) & np.isfinite(fillv)
            a[hole] = fillv[hole]
    return u, v
