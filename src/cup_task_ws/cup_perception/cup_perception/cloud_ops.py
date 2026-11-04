"""
Point-cloud steps for finding the cup (pure NumPy/SciPy, no ROS).

    depth_to_points   depth image -> 3D points in the camera's optical frame
    transform         4x4 transform applied to N x 3 points
    crop_box          keep points inside an axis-aligned box
    fit_table_plane   RANSAC: the dominant, roughly horizontal plane
    dbscan            density clustering (points closer than eps join a cluster)
    select_cup        pick the cluster whose size looks like a cup
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def depth_to_metres(data, encoding, height, width, step):
    """
    Raw image bytes -> float32 depth in metres (NaN where invalid).

    Handles the two D435 conventions: 32FC1 in metres (Gazebo) and
    16UC1/mono16 in millimetres (the real realsense2_camera driver).
    """
    if encoding == '32FC1':
        depth = np.frombuffer(data, dtype=np.float32).reshape(height, step // 4)[:, :width]
        depth = depth.astype(np.float32, copy=True)
    elif encoding in ('16UC1', 'mono16'):
        raw = np.frombuffer(data, dtype=np.uint16).reshape(height, step // 2)[:, :width]
        depth = raw.astype(np.float32) * 0.001
        depth[raw == 0] = np.nan  # 0 = no measurement
    else:
        raise ValueError(f'unsupported depth encoding "{encoding}"')
    depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
    return depth


def depth_to_points(depth, fx, fy, cx, cy, stride=1, min_range=0.0, max_range=np.inf):
    """Deproject a depth image (metres) into N x 3 points (optical frame: z forward)."""
    d = depth[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    ok = np.isfinite(d) & (d >= min_range) & (d <= max_range)
    z = d[ok]
    return np.column_stack([(u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z]).astype(np.float64)


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def crop_box(points, lower, upper):
    keep = np.all((points >= np.asarray(lower)) & (points <= np.asarray(upper)), axis=1)
    return points[keep]


@dataclass
class Plane:
    normal: np.ndarray  # unit, pointing up
    offset: float       # plane: normal . p + offset = 0
    inliers: int

    def height_of(self, points):
        """Signed height of points above the plane."""
        return points @ self.normal + self.offset

    def z_at(self, x, y):
        n = self.normal
        return -(n[0] * x + n[1] * y + self.offset) / n[2]


def fit_table_plane(points, distance, max_tilt_deg, iterations, rng=None):
    """
    RANSAC for the largest plane whose normal is within max_tilt_deg of vertical.

    Repeatedly picks 3 random points, makes the plane through them and counts
    the points within `distance` of it; keeps the best and refines it with a
    least-squares fit on its inliers.
    """
    if len(points) < 3:
        return None
    rng = rng or np.random.default_rng(0)
    sample = points if len(points) <= 20000 else points[rng.choice(len(points), 20000, False)]
    idx = rng.integers(0, len(sample), size=(iterations, 3))
    a, b, c = sample[idx[:, 0]], sample[idx[:, 1]], sample[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    norm = np.linalg.norm(normals, axis=1)
    valid = norm > 1e-9
    normals[valid] /= norm[valid, None]
    normals[normals[:, 2] < 0] *= -1
    valid &= normals[:, 2] > math.cos(math.radians(max_tilt_deg))
    if not valid.any():
        return None
    normals, a = normals[valid], a[valid]
    offsets = -np.einsum('ij,ij->i', normals, a)
    counts = (np.abs(sample @ normals.T + offsets) < distance).sum(axis=0)
    best = int(np.argmax(counts))
    inliers = points[np.abs(points @ normals[best] + offsets[best]) < distance]
    # Refine: plane through the inliers' centroid, normal = least-variance direction.
    centroid = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - centroid, full_matrices=False)[2][2]
    if normal[2] < 0:
        normal = -normal
    return Plane(normal, -float(normal @ centroid), len(inliers))


def dbscan(points, eps, min_samples):
    """
    DBSCAN cluster labels for each point (-1 = noise).

    Core points have >= min_samples neighbours within eps (counting
    themselves); clusters are connected core points plus the border points
    next to them.
    """
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    pairs = cKDTree(points).query_pairs(eps, output_type='ndarray')
    degree = np.bincount(pairs.ravel(), minlength=n) + 1
    core = degree >= min_samples
    core_pairs = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]]
    graph = coo_matrix((np.ones(len(core_pairs)), (core_pairs[:, 0], core_pairs[:, 1])),
                       shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    labels[core] = comp[core]
    # Border points: non-core points next to a core point take its cluster.
    i, j = pairs[:, 0], pairs[:, 1]
    for border, owner in ((i, j), (j, i)):
        m = ~core[border] & core[owner]
        labels[border[m]] = labels[owner[m]]
    # Renumber clusters 0..k-1 (noise stays -1).
    clustered = labels >= 0
    labels[clustered] = np.unique(labels[clustered], return_inverse=True)[1]
    return labels


@dataclass
class Cluster:
    points: np.ndarray
    height: float      # top of the cluster above the table plane
    footprint: float   # largest horizontal extent


def describe_clusters(points, labels, plane):
    out = []
    for k in range(labels.max() + 1 if len(labels) else 0):
        pts = points[labels == k]
        xy = pts[:, :2]
        extent = xy.max(axis=0) - xy.min(axis=0)
        out.append(Cluster(pts, float(plane.height_of(pts).max()), float(np.hypot(*extent))))
    return out


def select_cup(clusters, min_points, height_range, footprint_range):
    """
    Pick the biggest cluster with a cup-like size (or None).

    Also returns why each rejected cluster was rejected, for logging.
    """
    good, reasons = [], []
    for i, c in enumerate(clusters):
        if len(c.points) < min_points:
            reasons.append(f'#{i}: {len(c.points)} points (< {min_points})')
        elif not height_range[0] <= c.height <= height_range[1]:
            reasons.append(f'#{i}: {c.height * 1000:.0f} mm tall')
        elif not footprint_range[0] <= c.footprint <= footprint_range[1]:
            reasons.append(f'#{i}: {c.footprint * 1000:.0f} mm wide')
        else:
            good.append(c)
    return (max(good, key=lambda c: len(c.points)) if good else None), reasons
