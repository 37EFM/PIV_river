"""Reading the movie and its metadata.

Frames are returned as an array of shape (n_frames, height, width) of
8-bit grey levels.  The Apple QuickTime metadata (lens, focal length,
GPS position) is read directly from the file's ``moov/meta`` atoms, so
no external tool is needed.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import imageio.v3 as iio


# ----------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------
def read_frames(path, scale=1.0, start=0, stop=None, gray=True):
    """Read a movie into memory.

    Parameters
    ----------
    path : str or Path
    scale : float
        Spatial downscaling factor (1 = full resolution, 0.5 = half).
        Halving the resolution divides the memory by four.
    start, stop : int
        Frame range to keep.
    gray : bool
        Convert to grey levels (uint8).  PIV only needs intensity.

    Returns
    -------
    frames : ndarray, (n, H, W) uint8   (or (n, H, W, 3) if gray=False)
    info : dict with 'fps', 'width', 'height', 'n_frames', 'scale', 'path'
    """
    path = Path(path)
    meta = iio.immeta(path, plugin="FFMPEG")
    fps = float(meta["fps"])
    frames = []
    for i, fr in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if i < start:
            continue
        if stop is not None and i >= stop:
            break
        if gray:
            # luma weights; keep as float for the optional resize
            fr = (0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2])
        if scale != 1.0:
            fr = _resize(fr, scale)
        frames.append(np.clip(np.rint(fr), 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    info = dict(fps=fps, n_frames=len(frames), height=frames.shape[1],
                width=frames.shape[2], scale=scale, path=str(path))
    return frames, info


def _resize(img, scale):
    """Resize by block averaging (scale = 1/k) or linear interpolation."""
    from scipy.ndimage import zoom
    k = 1.0 / scale
    if abs(k - round(k)) < 1e-9 and k >= 1:
        k = int(round(k))
        if img.ndim == 2:
            h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
            return img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
        h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
        return img[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))
    factors = (scale, scale) if img.ndim == 2 else (scale, scale, 1)
    return zoom(img, factors, order=1)


# ----------------------------------------------------------------------
# metadata (QuickTime 'mdta' keys)
# ----------------------------------------------------------------------
def _boxes(buf, off, end):
    """Iterate over the (type, payload_start, box_end) of MP4/MOV boxes."""
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        if size == 0:
            size = end - off
        if size < hdr:
            return
        yield typ, off + hdr, off + size
        off += size


def _decode_value(dtype, payload):
    if dtype == 1:
        return payload.decode("utf-8", errors="replace")
    if dtype == 23:
        return struct.unpack(">f", payload)[0]
    if dtype == 24:
        return struct.unpack(">d", payload)[0]
    if dtype == 21:
        return int.from_bytes(payload, "big", signed=True)
    if dtype == 22:
        return int.from_bytes(payload, "big")
    return payload


def _read_meta_box(buf, a, b, out):
    sub = {t: (x, y) for t, x, y in _boxes(buf, a, b)}
    if b"keys" not in sub:            # MOV 'meta' has no version/flags, MP4 has 4 bytes
        sub = {t: (x, y) for t, x, y in _boxes(buf, a + 4, b)}
    if b"keys" not in sub or b"ilst" not in sub:
        return
    ka, kb = sub[b"keys"]
    keys = [buf[x:y].decode("utf-8", errors="replace") for _, x, y in _boxes(buf, ka + 8, kb)]
    for t, x, y in _boxes(buf, *sub[b"ilst"]):
        idx = struct.unpack(">I", t)[0]
        if not 1 <= idx <= len(keys):
            continue
        for t2, x2, y2 in _boxes(buf, x, y):
            if t2 != b"data":
                continue
            dtype = struct.unpack(">I", buf[x2:x2 + 4])[0]
            out[keys[idx - 1]] = _decode_value(dtype, buf[x2 + 8:y2])


def _walk(buf, off, end, out):
    for typ, a, b in _boxes(buf, off, end):
        if typ == b"meta":
            _read_meta_box(buf, a, b, out)
        elif typ in (b"moov", b"trak", b"udta"):
            _walk(buf, a, b, out)


def read_metadata(path):
    """Return the QuickTime metadata of a movie as a dict.

    For an iPhone movie this contains, among others,
    ``com.apple.quicktime.camera.focal_length.35mm_equivalent``,
    ``com.apple.quicktime.camera.lens_model``,
    ``com.apple.quicktime.location.ISO6709`` and the creation date.
    The dict also receives convenience keys ``focal_length_35mm`` (float
    or None), ``lens`` and ``location`` (latitude, longitude, altitude).
    """
    buf = Path(path).read_bytes()
    out = {}
    _walk(buf, 0, len(buf), out)

    f35 = out.get("com.apple.quicktime.camera.focal_length.35mm_equivalent")
    try:
        out["focal_length_35mm"] = float(f35) if f35 is not None else None
    except (TypeError, ValueError):
        out["focal_length_35mm"] = None
    out["lens"] = out.get("com.apple.quicktime.camera.lens_model")
    out["model"] = out.get("com.apple.quicktime.model")
    out["creation_date"] = out.get("com.apple.quicktime.creationdate")
    out["location"] = _parse_iso6709(out.get("com.apple.quicktime.location.ISO6709"))
    return out


def _parse_iso6709(s):
    """'+51.4458+005.4831+016.498/' -> (51.4458, 5.4831, 16.498)."""
    if not s:
        return None
    import re
    nums = re.findall(r"[+-]\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        return None
    vals = [float(v) for v in nums[:3]]
    return tuple(vals) if len(vals) == 3 else (vals[0], vals[1], None)


def focal_length_pixels(focal_length_35mm, width, height):
    """Convert a 35 mm-equivalent focal length to pixels.

    The convention used here is that the long side of the image
    corresponds to the 36 mm side of a 35 mm film frame, so that

        f_px = f35 / 36 mm * max(width, height).

    Phone makers do not all use the same convention, and video modes crop
    the sensor (electronic stabilization crops further), so this value is
    uncertain by roughly 10 %.  With three or more calibration segments
    the focal length can be fitted instead of assumed; see
    :func:`rivpiv.geometry.fit_camera`.
    """
    return float(focal_length_35mm) / 36.0 * max(width, height)
