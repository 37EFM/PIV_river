"""Quick checks on synthetic data: run with ``pytest``."""
import numpy as np
from scipy.ndimage import gaussian_filter, shift

from rivpiv import geometry, piv, stabilize


def _camera():
    return geometry.CameraModel(h=4.0, theta=np.radians(50), f=1400.0, cx=540.0, cy=960.0)


def test_pixel_world_round_trip():
    cam = _camera()
    x = np.array([10.0, 500.0, 1000.0])
    y = np.array([300.0, 900.0, 1800.0])
    X, Y = cam.pixel_to_world(x, y)
    xb, yb = cam.world_to_pixel(X, Y)
    assert np.allclose(xb, x) and np.allclose(yb, y)


def test_points_above_horizon_are_nan():
    cam = _camera()
    X, Y = cam.pixel_to_world(540.0, cam.horizon_row - 10)
    assert np.isnan(X) and np.isnan(Y)


def test_fit_camera_recovers_synthetic_camera():
    cam = _camera()
    seg = [(240, 230, 830, 230), (230, 250, 30, 560), (1000, 1300, 1080, 1700)]
    L = geometry.segment_lengths(cam, seg)
    segments = [s + (l,) for s, l in zip(seg, L)]
    fitted, report = geometry.fit_camera(segments[:2], cam.f, 1080, 1920)
    assert abs(fitted.h - cam.h) < 0.01 and abs(fitted.theta - cam.theta) < 1e-3
    fitted3, report3 = geometry.fit_camera(segments, cam.f * 1.15, 1080, 1920, fit_focal=True)
    assert abs(fitted3.f - cam.f) / cam.f < 0.01
    assert report3["rms_percent"] < 0.1


def test_piv_finds_known_shift():
    rng = np.random.default_rng(0)
    a = gaussian_filter(rng.normal(size=(320, 256)).astype(np.float32), 1.5)
    b = shift(a, (2.3, -1.7), order=3)
    fld = piv.piv_pair(a, b, fps=30, window=64)
    piv.validate(fld, snr_min=1.0, median_threshold=None)
    assert abs(np.nanmedian(fld.u) + 1.7) < 0.1
    assert abs(np.nanmedian(fld.v) - 2.3) < 0.1


def test_median_test_flags_single_outlier():
    u = np.ones((7, 7))
    v = np.zeros((7, 7))
    u[3, 3] = 8.0
    bad = piv.normalized_median_test(u, v)
    assert bad[3, 3] and bad.sum() == 1


def test_stabilization_removes_translation_and_roll():
    rng = np.random.default_rng(1)
    base = gaussian_filter(rng.uniform(0, 255, size=(600, 400)), 2)
    frames = np.stack([base, stabilize.apply_motion(base[None], np.array([[-6.0, 3.5, np.radians(0.3)]]))[0]])
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    regions = [[20, 20, 180, 180], [220, 400, 380, 580]]
    motion, _, _ = stabilize.estimate_motion(frames, regions, highpass_sigma=3)
    # apply_motion with +m maps the scene by the inverse, so the recovered motion is -m
    assert abs(motion[1, 0] - 6.0) < 0.3 and abs(motion[1, 1] + 3.5) < 0.3
    assert abs(np.degrees(motion[1, 2]) + 0.3) < 0.05
    stab = stabilize.apply_motion(frames, motion)
    residual, _, _ = stabilize.estimate_motion(stab, regions, highpass_sigma=3)
    assert np.abs(residual[1, :2]).max() < 0.3
