"""From pixels to metres: the camera model and its calibration.

The water surface is assumed flat and horizontal.  The camera sits at a
height ``h`` above it and looks down at a pitch angle ``theta`` below
the horizontal, without roll (the horizon is horizontal in the image).
The lens is a pinhole with focal length ``f`` in pixels and principal
point ``(cx, cy)`` at the image centre.

Image coordinates ``(x, y)`` are pixel column and row, origin at the
top-left corner, ``y`` increasing downward.  World coordinates ``(X, Y)``
are metres on the water surface: ``X`` to the camera's right, ``Y``
horizontally away from the camera, origin directly below the camera.

With xi = x - cx and eta = y - cy, a ray through the pixel meets the
water at

    D = eta cos(theta) + f sin(theta)
    X = h xi / D
    Y = h (f cos(theta) - eta sin(theta)) / D

``D = 0`` is the horizon (``eta = -f tan(theta)``); only pixels below it
see the water.  The cross-stream scale ``dX/dx = h / D`` has an inverse
that is linear in the row, which is why two known lengths at two
different rows fix the whole mapping once ``f`` is known.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, least_squares


@dataclass
class CameraModel:
    h: float            # camera height above the water surface [m]
    theta: float        # pitch below the horizontal [rad]
    f: float            # focal length [pixels]
    cx: float           # principal point [pixels]
    cy: float

    # -- forward and inverse mapping -----------------------------------
    def pixel_to_world(self, x, y):
        """Map pixel coordinates to metres on the water surface."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        s, c = np.sin(self.theta), np.cos(self.theta)
        xi, eta = x - self.cx, y - self.cy
        D = eta * c + self.f * s
        with np.errstate(divide="ignore", invalid="ignore"):
            X = self.h * xi / D
            Y = self.h * (self.f * c - eta * s) / D
        X = np.where(D > 0, X, np.nan)
        Y = np.where(D > 0, Y, np.nan)
        return X, Y

    def world_to_pixel(self, X, Y):
        """Map metres on the water surface to pixel coordinates."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        s, c = np.sin(self.theta), np.cos(self.theta)
        eta = self.f * (self.h * c - Y * s) / (Y * c + self.h * s)
        xi = X * (eta * c + self.f * s) / self.h
        return xi + self.cx, eta + self.cy

    def metres_per_pixel(self, x, y):
        """Local scale factors (dX/dx, dY/dy) at a pixel, in m/pixel.

        ``dX/dx`` is the cross-view scale; ``dY/dy`` the along-view scale
        (negative because ``Y`` grows upward in the image).
        """
        y = np.asarray(y, float)
        s, c = np.sin(self.theta), np.cos(self.theta)
        eta = y - self.cy
        D = eta * c + self.f * s
        dXdx = self.h / D
        dYdy = -self.h * self.f / D ** 2
        return dXdx, dYdy

    @property
    def horizon_row(self):
        """Image row of the horizon (may be outside the image)."""
        return self.cy - self.f * np.tan(self.theta)

    @property
    def theta_deg(self):
        return np.degrees(self.theta)

    def __repr__(self):
        return (f"CameraModel(h={self.h:.2f} m, theta={self.theta_deg:.1f} deg, "
                f"f={self.f:.0f} px, horizon row={self.horizon_row:.0f})")


# ----------------------------------------------------------------------
# calibration from segments of known length
# ----------------------------------------------------------------------
def segment_lengths(cam, segments):
    """World length of each segment [(x1, y1, x2, y2, L), ...] under ``cam``."""
    seg = np.asarray([s[:4] for s in segments], float)
    X1, Y1 = cam.pixel_to_world(seg[:, 0], seg[:, 1])
    X2, Y2 = cam.pixel_to_world(seg[:, 2], seg[:, 3])
    return np.hypot(X2 - X1, Y2 - Y1)


def _unit_lengths(theta, f, cx, cy, seg):
    """Segment lengths for h = 1 (they scale linearly with h)."""
    cam = CameraModel(1.0, theta, f, cx, cy)
    return segment_lengths(cam, seg)


def _best_h(g, L):
    """Closed-form least-squares height given unit lengths g and targets L."""
    return float(np.sum(L * g) / np.sum(g * g))


def fit_camera(segments, f, width, height, fit_focal=False, theta_range=(2.0, 88.0)):
    """Fit the camera height and pitch (and optionally focal length).

    Parameters
    ----------
    segments : list of (x1, y1, x2, y2, length_m)
        Pixel end points of segments drawn on the water surface and their
        true lengths in metres.  Two segments at clearly different rows
        are the minimum; more give a least-squares fit with residuals.
    f : float
        Focal length in pixels (see :func:`rivpiv.video.focal_length_pixels`).
        Used as the fixed value, or as the initial guess if ``fit_focal``.
    width, height : int
        Image size in pixels; the principal point is taken at the centre.
    fit_focal : bool
        Also fit the focal length.  Needs at least three segments, not all
        parallel.
    theta_range : (deg, deg)
        Search interval for the pitch angle.

    Returns
    -------
    cam : CameraModel
    report : dict with per-segment measured and fitted lengths and the
        relative residuals.
    """
    seg = [tuple(map(float, s[:5])) for s in segments]
    if len(seg) < 2:
        raise ValueError("at least two segments are needed")
    if fit_focal and len(seg) < 3:
        raise ValueError("fitting the focal length needs at least three segments")
    L = np.array([s[4] for s in seg])
    cx, cy = width / 2.0, height / 2.0

    rows = np.array([(s[1] + s[3]) / 2 for s in seg])
    if np.ptp(rows) < 0.05 * height:
        print("warning: all segments lie at nearly the same image row; "
              "the pitch angle is poorly constrained")

    def residual(theta, fpx):
        with np.errstate(all="ignore"):
            g = _unit_lengths(theta, fpx, cx, cy, seg)
        if not np.all(np.isfinite(g)) or np.any(g <= 0):
            return np.inf, np.nan
        h = _best_h(g, L)
        return float(np.sum(((h * g - L) / L) ** 2)), h

    # coarse search in theta, then refinement
    thetas = np.radians(np.linspace(theta_range[0], theta_range[1], 400))
    costs = np.array([residual(t, f)[0] for t in thetas])
    if not np.any(np.isfinite(costs)):
        raise RuntimeError("no pitch angle puts all segment end points below the horizon")
    i = int(np.nanargmin(costs))
    lo = thetas[max(i - 1, 0)]
    hi = thetas[min(i + 1, len(thetas) - 1)]
    res = minimize_scalar(lambda t: residual(t, f)[0], bounds=(lo, hi), method="bounded")
    theta = float(res.x)
    fpx = float(f)

    if fit_focal:
        def fun(p):
            t, lf = p
            g = _unit_lengths(t, np.exp(lf), cx, cy, seg)
            if not np.all(np.isfinite(g)) or np.any(g <= 0):
                return np.full(len(seg), 1e3)
            h = _best_h(g, L)
            return (h * g - L) / L
        sol = least_squares(fun, [theta, np.log(fpx)],
                            bounds=([np.radians(theta_range[0]), np.log(fpx / 3)],
                                    [np.radians(theta_range[1]), np.log(fpx * 3)]))
        theta, fpx = float(sol.x[0]), float(np.exp(sol.x[1]))

    g = _unit_lengths(theta, fpx, cx, cy, seg)
    h = _best_h(g, L)
    cam = CameraModel(h=h, theta=theta, f=fpx, cx=cx, cy=cy)
    fitted = h * g
    report = dict(measured_m=L, fitted_m=fitted,
                  residual_percent=100 * (fitted - L) / L,
                  rms_percent=float(np.sqrt(np.mean((100 * (fitted - L) / L) ** 2))))
    return cam, report


# ----------------------------------------------------------------------
# applying the model to PIV results
# ----------------------------------------------------------------------
def vectors_to_world(cam, x, y, u, v, dt):
    """Convert pixel displacements to velocities in m/s on the water surface.

    Parameters
    ----------
    cam : CameraModel
    x, y : arrays of vector positions [pixels]
    u, v : arrays of displacements [pixels] over the time ``dt`` [s]
        (``v`` positive downward in the image, as in the PIV output)

    Returns
    -------
    X, Y : positions [m] in world coordinates
    U, V : velocities [m/s]; ``U`` to the camera's right, ``V`` away
        from the camera.
    """
    X0, Y0 = cam.pixel_to_world(x, y)
    X1, Y1 = cam.pixel_to_world(np.asarray(x) + np.asarray(u), np.asarray(y) + np.asarray(v))
    return X0, Y0, (X1 - X0) / dt, (Y1 - Y0) / dt


def rectify(cam, image, extent=None, resolution=0.05, order=1):
    """Resample an image onto a regular grid in metres (a plan view).

    Parameters
    ----------
    cam : CameraModel
    image : 2-D array (grey) or 3-D (colour)
    extent : (Xmin, Xmax, Ymin, Ymax) in metres.  Default: the world
        coordinates of the lower part of the image.
    resolution : grid spacing [m/pixel]
    order : interpolation order for scipy.ndimage.map_coordinates

    Returns
    -------
    plan : resampled image, row 0 at ``Ymax`` (so that ``imshow`` with
        ``extent`` and ``origin='upper'`` shows ``Y`` increasing upward)
    extent : the extent actually used (for ``imshow``)
    """
    from scipy.ndimage import map_coordinates
    H, W = image.shape[:2]
    if extent is None:
        # use the lower 85 % of the image below the horizon
        y_top = max(cam.horizon_row + 0.15 * (H - cam.horizon_row), 0)
        Xc, Yc = cam.pixel_to_world([0, W - 1, 0, W - 1], [y_top, y_top, H - 1, H - 1])
        extent = (np.nanmin(Xc), np.nanmax(Xc), np.nanmin(Yc), np.nanmax(Yc))
    Xmin, Xmax, Ymin, Ymax = extent
    Xg = np.arange(Xmin, Xmax, resolution)
    Yg = np.arange(Ymax, Ymin, -resolution)
    XX, YY = np.meshgrid(Xg, Yg)
    px, py = cam.world_to_pixel(XX, YY)
    if image.ndim == 2:
        plan = map_coordinates(image.astype(np.float32), [py, px], order=order, cval=np.nan)
    else:
        plan = np.stack([map_coordinates(image[..., k].astype(np.float32), [py, px],
                                         order=order, cval=np.nan) for k in range(image.shape[2])], -1)
    return plan, (Xg[0], Xg[-1], Yg[-1], Yg[0])


# ----------------------------------------------------------------------
# everything a movie needs, in one JSON file
# ----------------------------------------------------------------------
@dataclass
class Setup:
    """Calibration and regions for one movie, saved as JSON.

    segments : list of [x1, y1, x2, y2, length_m]   (pixels of the frames as read)
    static_regions : list of [x0, y0, x1, y1] rectangles on the banks,
        used to estimate the camera motion
    water_polygon : list of [x, y] vertices around the water surface
    scale : the ``scale`` the frames were read with (coordinates refer
        to those frames)
    focal_length_px : focal length actually used
    camera : the fitted CameraModel parameters (h, theta, f, cx, cy)
    """
    movie: str = ""
    scale: float = 1.0
    width: int = 0
    height: int = 0
    focal_length_px: float | None = None
    segments: list = field(default_factory=list)
    static_regions: list = field(default_factory=list)
    water_polygon: list = field(default_factory=list)
    camera: dict | None = None
    notes: str = ""

    def save(self, path):
        d = asdict(self)
        d["segments"] = [list(map(float, s)) for s in self.segments]
        # one shape per line, for a readable file
        text = json.dumps(d, indent=2)
        for key in ("segments", "static_regions", "water_polygon"):
            text = text.replace(json.dumps(d[key], indent=2).replace("\n", "\n  "),
                                "[\n    " + ",\n    ".join(json.dumps(s) for s in d[key]) + "\n  ]"
                                if d[key] else "[]")
        Path(path).write_text(text + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def camera_model(self):
        if self.camera is None:
            raise ValueError("no camera fitted yet; call fit()")
        return CameraModel(**self.camera)

    def fit(self, fit_focal=False):
        """Fit the camera from the stored segments; returns (cam, report)."""
        cam, report = fit_camera(self.segments, self.focal_length_px, self.width,
                                 self.height, fit_focal=fit_focal)
        self.camera = dict(h=cam.h, theta=cam.theta, f=cam.f, cx=cam.cx, cy=cam.cy)
        return cam, report
