"""rivpiv -- surface velocimetry of a river from a hand-held video.

A small, readable implementation of large-scale particle image
velocimetry (LSPIV) for the course 37EFM Environmental Fluid Mechanics
(TU/e).  The pipeline is

    video.read_frames      ->  stabilize.estimate_shifts / apply_shifts
    gui.ShapePicker        ->  geometry.fit_camera   (pixels -> metres)
    piv.ensemble_piv       ->  piv.validate          -> geometry.vectors_to_world
    plotting.*

Every step is a plain function on numpy arrays so that it can be read,
modified and questioned.
"""
from . import video, geometry, gui, stabilize, piv, plotting   # noqa: F401
from .geometry import CameraModel, fit_camera, Setup
from .video import read_frames, read_metadata, focal_length_pixels

__version__ = "0.1.0"
