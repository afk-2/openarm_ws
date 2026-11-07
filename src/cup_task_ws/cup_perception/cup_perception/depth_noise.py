"""
D435-style errors for a perfect simulated depth image (pure NumPy/SciPy, no ROS).

Gazebo's depth camera measures every pixel exactly. A real D435 (active IR
stereo) does not. This module adds its main error types, each one
switchable on its own (set its strength to 0):

  1. Depth noise growing with distance squared. Stereo measures disparity
     (the pixel shift between the two IR imagers); a fixed disparity error
     `subpixel_rms` gives a depth error
         sigma_z = z^2 * subpixel_rms / (focal_px * baseline)
     (Intel, "Tuning depth cameras for best performance"). The error is
     partly correlated over a few pixels, like the stereo matching window,
     so it does not simply average out over a surface.
  2. Edges:
       - shadow: background just left of a nearer object is hidden from the
         right imager, so it gets no depth (width grows with the depth jump)
       - along depth jumps, some pixels are missing and some are "flying
         pixels" with a depth somewhere between the front and the back
  3. Random single-pixel dropouts.
  4. Missing patches on shiny surfaces. A glossy surface (glazed ceramic)
     seen at a steep angle reflects the projected IR pattern away from the
     camera, so blobs of it get no depth. Approximated by: pixels whose
     surface is seen at more than `shiny_min_incidence_deg` from head-on
     can lose patches. With the camera looking down, this hits the cup
     walls but not the table top.
  5. The real driver's output format: uint16 millimetres (1 mm steps),
     0 = no depth, nothing closer than min_depth.
"""
from dataclasses import dataclass, fields
import math

import cv2
import numpy as np


@dataclass
class NoiseSettings:
    # 1. depth noise
    baseline: float = 0.050            # m, distance between the D435's IR imagers
    depth_focal_px: float = 383.0      # D435 depth imager focal length at 640x480
    subpixel_rms: float = 0.08         # disparity error (px); Intel's typical value
    correlated_fraction: float = 0.6   # share of the noise variance that is correlated
    correlation_px: float = 2.0        # size of the correlated noise blobs (px)
    # 2. edges
    shadow: bool = True
    edge_jump: float = 0.02            # m; a depth jump this big between neighbours = edge
    edge_invalid_prob: float = 0.3     # edge pixels with no depth
    edge_flying_prob: float = 0.3      # edge pixels with a depth between front and back
    # 3. random dropouts
    dropout_prob: float = 0.005
    # 4. shiny surfaces
    shiny_min_incidence_deg: float = 55.0
    shiny_seed_prob: float = 0.01      # chance per candidate pixel to start a patch
    shiny_patch_radius_px: int = 3
    # 5. output range
    min_depth: float = 0.28            # m; D435 at 640x480 sees nothing closer
    max_depth: float = 6.0

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]


def sigma_z(z, s):
    """Depth noise standard deviation (m) at depth z (m)."""
    return z ** 2 * s.subpixel_rms / (s.depth_focal_px * s.baseline)


def _unit_noise(shape, s, rng):
    """Zero-mean, unit-variance noise, partly correlated over correlation_px."""
    white = rng.standard_normal(shape, dtype=np.float32)
    if s.correlated_fraction <= 0 or s.correlation_px <= 0:
        return white
    smooth = cv2.GaussianBlur(rng.standard_normal(shape, dtype=np.float32), (0, 0),
                              s.correlation_px)
    smooth /= max(smooth.std(), 1e-12)
    a = math.sqrt(s.correlated_fraction)
    return a * smooth + math.sqrt(1 - a * a) * white


def shadow_mask(z, fx, baseline, max_px=40):
    """
    Pixels the right imager cannot see (D435 depth is relative to the left one).

    Background pixel u is hidden if a nearer pixel u+k to its right has a
    disparity at least k larger: seen from the right imager it covers u.
    """
    ok = np.isfinite(z) & (z > 0)
    disp = np.zeros(z.shape, dtype=np.float32)
    disp[ok] = fx * baseline / z[ok]
    hidden = np.zeros(z.shape, dtype=bool)
    if z.shape[1] < 2:
        return hidden
    # The widest shadow comes from the biggest jump between neighbouring pixels.
    steps = int(min(max_px, math.ceil(max(0.0, float((disp[:, 1:] - disp[:, :-1]).max())))))
    for k in range(1, steps + 1):
        hidden[:, :-k] |= disp[:, k:] - disp[:, :-k] >= k
    return hidden


def incidence_deg(z, fx, fy, cx, cy):
    """Angle between the surface normal and the camera ray, per pixel (deg)."""
    z = z.astype(np.float32)
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float32)
    p = [(u - cx) * z / fx, (v - cy) * z / fy, z]
    du = [np.gradient(c, axis=1) for c in p]
    dv = [np.gradient(c, axis=0) for c in p]
    n = [du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2],
         du[0] * dv[1] - du[1] * dv[0]]
    dot = np.abs(n[0] * p[0] + n[1] * p[1] + n[2] * p[2])
    norms = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) * np.sqrt(p[0] ** 2 + p[1] ** 2 + z ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = dot / norms
    return np.degrees(np.arccos(np.clip(np.nan_to_num(cos, nan=1.0), 0.0, 1.0)))


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def add_d435_noise(z, fx, fy, cx, cy, s, rng):
    """
    Return a copy of depth image z (metres, NaN = none) with D435-style errors.

    Also returns the masks of each effect, for checking.
    """
    valid = np.isfinite(z) & (z > 0)
    z = np.where(valid, z, np.nan)
    out = z + sigma_z(np.nan_to_num(z), s) * _unit_noise(z.shape, s, rng)
    masks = {}

    # Edges (from the true depth): flying pixels and missing pixels.
    box = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(np.where(valid, z, 0.0).astype(np.float32), box).astype(z.dtype)
    zmin = cv2.erode(np.where(valid, z, np.inf).astype(np.float32), box).astype(z.dtype)
    edge = valid & (zmax - zmin > s.edge_jump)
    r = rng.random(z.shape)
    flying = edge & (r >= s.edge_invalid_prob) & (r < s.edge_invalid_prob + s.edge_flying_prob)
    out[flying] = zmin[flying] + rng.random(int(flying.sum())) * (zmax - zmin)[flying]
    masks['edge_missing'] = edge & (r < s.edge_invalid_prob)
    masks['flying'] = flying
    masks['shadow'] = shadow_mask(z, fx, s.baseline) & valid if s.shadow else \
        np.zeros(z.shape, bool)
    masks['dropout'] = valid & (rng.random(z.shape) < s.dropout_prob)

    shiny = np.zeros(z.shape, bool)
    if s.shiny_seed_prob > 0:
        candidate = valid & (incidence_deg(z, fx, fy, cx, cy) > s.shiny_min_incidence_deg)
        seeds = candidate & (rng.random(z.shape) < s.shiny_seed_prob)
        patches = cv2.dilate(seeds.astype(np.uint8), _disk(int(s.shiny_patch_radius_px)))
        shiny = patches.astype(bool) & candidate
    masks['shiny'] = shiny

    missing = masks['edge_missing'] | masks['shadow'] | masks['dropout'] | shiny
    out[missing] = np.nan
    out[(out < s.min_depth) | (out > s.max_depth)] = np.nan
    return out, masks


def to_uint16_mm(z):
    """Depth in metres (NaN = none) -> uint16 millimetres (0 = none), like the driver."""
    mm = np.round(np.nan_to_num(z, nan=0.0) * 1000.0)
    return np.clip(mm, 0, 65535).astype(np.uint16)
