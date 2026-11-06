# PIV_river — surface velocity of a river from a phone video

Course material for **37EFM Environmental Fluid Mechanics** (Eindhoven
University of Technology). A short video of a river, filmed from a
bridge with a phone, is turned into a map of the surface velocity in
metres per second. The method is *large-scale particle image
velocimetry* (LSPIV): foam, leaves and ripples are the particles, the
phone is the camera.

The repository contains

| Path | Content |
|---|---|
| `notebooks/river_surface_PIV.ipynb` | The walkthrough, from movie to velocity field, with questions at the end. Start here. |
| `rivpiv/` | The Python package the notebook uses: five short, commented modules. |
| `data/IMG_0332.mov` | 3.7 s of the Dommel in Eindhoven, 6 November 2026, 1080 × 1920 at 30 fps. |
| `data/IMG_0332_setup_example.json` | An example calibration file. **Its lengths are placeholders**, not measurements; see the notebook. |
| `tests/` | Checks of the geometry, the correlation and the stabilization on synthetic data (`pytest`). |

## Installation

With conda (Miniconda or Anaconda):

```bash
git clone https://github.com/37EFM/PIV_river.git
cd PIV_river
conda env create -f environment.yml
conda activate rivpiv
jupyter lab notebooks/river_surface_PIV.ipynb
```

Without conda, `pip install -e ".[notebook]"` in the repository does the
same. The interactive drawing tools need the `ipympl` package, which
both routes install; in the notebook they are switched on with
`%matplotlib widget`.

## What the notebook does

1. **Read** the movie. The phone's metadata gives the lens, its focal
   length and the GPS position.
2. **Stabilize.** A hand-held phone drifts by tens of pixels in a few
   seconds and rolls slightly. Rectangles drawn on the banks are tracked
   from frame to frame, a rigid motion is fitted, and every frame is
   mapped back onto the first.
3. **Calibrate.** Two or more segments of known length are drawn on the
   water surface. With the focal length, they fix the camera height and
   pitch, and hence the mapping from every pixel to a point on the
   water in metres. A plan view (orthophoto) shows whether it worked.
4. **PIV.** Interrogation windows are cross-correlated between frames by
   FFT. The correlation maps of all frame pairs are averaged (ensemble
   correlation) before the peak is located to sub-pixel precision.
   Vectors are validated by signal-to-noise ratio and by the normalized
   median test.
5. **Convert and look.** Displacements become velocities in m/s; the
   field is shown on the image and in plan view, a cross-section profile
   is extracted, and a discharge is estimated from an assumed depth
   profile. Instantaneous fields give a time series at a point.

## What to measure in the field

The calibration needs lengths on the water surface that are visible in
the frame, **both ends at water level** (waterline to waterline, or
between floating marks). Two segments at clearly different distances
from the camera are the minimum. A third one checks the fit and, if
wanted, lets the focal length be fitted rather than taken from the
metadata, which is uncertain by about 10 % because video modes crop the
sensor.

The river width at two places, measured with a tape or a laser
rangefinder from the bridge, is the easiest choice. Write down where
the ends are.

The discharge estimate also needs a depth profile across the section
you choose, which the notebook does not provide.

## Filming your own river

- Brace the phone on a railing. The stabilization corrects drift and
  roll, not a walk.
- Keep both banks in the frame, with some vegetation or ground on each:
  these are the static regions the stabilization tracks.
- Ten to twenty seconds at 30 fps is plenty. Lock the exposure if the
  phone allows it.
- Foam, leaves and ripples are the tracers. Still, glassy water with
  reflections of the sky gives nothing to track; wind ripples help.
- Measure two lengths on the water that you can see in the frame.

Then copy the movie into `data/`, change `MOVIE` in the notebook, draw
new regions and segments, and save the setup under a new name.

## Assumptions and limits

- The water surface is flat and horizontal, and the phone is not rolled
  relative to it (the horizon is horizontal in the image). The fitted
  roll from the stabilization is relative to the first frame only.
- The camera is a pinhole with the principal point at the image centre;
  lens distortion is ignored, which is adequate for the main camera of a
  phone away from the corners.
- Only translation and rotation of the camera are corrected. Parallax
  between near and far static regions is left; choose static regions
  close to the water you care about.
- The PIV is single-pass. Displacements larger than about a quarter of
  the window are lost; `dt_frames` and `window` must be chosen with the
  fastest part of the image in mind.

## The package

```
rivpiv/video.py      read_frames, read_metadata, focal_length_pixels
rivpiv/stabilize.py  estimate_motion, apply_motion
rivpiv/geometry.py   CameraModel, fit_camera, vectors_to_world, rectify, Setup
rivpiv/gui.py        ShapePicker (segments, rectangles, polygons), draw_shapes
rivpiv/piv.py        preprocess, correlate_pair, find_peaks, ensemble_piv, piv_pair,
                     piv_sequence, validate, normalized_median_test, fill_invalid
rivpiv/plotting.py   show_field_on_image, show_plan_view, show_correlation_map, ...
```

Dependencies: numpy, scipy, matplotlib, imageio and imageio-ffmpeg. No
OpenCV, no OpenPIV: the whole method is in these files, so that it can
be read and changed.

## References

- Adrian, R. J. and Westerweel, J. (2011). *Particle Image Velocimetry*.
  Cambridge University Press.
- Fujita, I., Muste, M. and Kruger, A. (1998). Large-scale particle image
  velocimetry for flow analysis in hydraulic engineering applications.
  *Journal of Hydraulic Research* 36, 397–414.
- Meinhart, C. D., Wereley, S. T. and Santiago, J. G. (2000). A PIV
  algorithm for estimating time-averaged velocity fields. *Journal of
  Fluids Engineering* 122, 285–289.
- Westerweel, J. and Scarano, F. (2005). Universal outlier detection for
  PIV data. *Experiments in Fluids* 39, 1096–1100.

## Licence

MIT, see `LICENSE`. The movie is by Matias Duran Matute and may be used
under the same terms.
