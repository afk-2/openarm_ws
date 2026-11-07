"""
Point-cloud steps for finding the cup (pure NumPy/SciPy, no ROS).

    depth_to_points   depth image -> 3D points in the camera's optical frame
    transform         4x4 transform applied to N x 3 points
    crop_box          keep points inside an axis-aligned box
    fit_table_plane   RANSAC: the dominant, roughly horizontal plane
    dbscan            density clustering (points closer than eps join a cluster)
    select_cups       keep the clusters whose size looks like a cup
    fit_outer_circle  centre of a cup from its outer wall
    estimate_handle   handle direction from the points outside the wall
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


def footprint(xy, trim=1.0):
    """
    Largest horizontal extent of a cluster, measured along its own main axes.

    The outermost `trim` percent of points on each side are ignored, so a
    few stray (flying) pixels do not count, and the result does not depend
    on how the cup is turned.
    """
    if len(xy) < 3:
        return 0.0
    c = xy - xy.mean(axis=0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    lo, hi = np.percentile(c @ axes.T, [trim, 100 - trim], axis=0)
    return float((hi - lo).max())


def describe_clusters(points, labels, plane):
    out = []
    for k in range(labels.max() + 1 if len(labels) else 0):
        pts = points[labels == k]
        out.append(Cluster(pts, float(plane.height_of(pts).max()), footprint(pts[:, :2])))
    return out


def select_cups(clusters, min_points, height_range, footprint_range):
    """
    All clusters with a cup-like size, biggest first.

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
    return sorted(good, key=lambda c: -len(c.points)), reasons


@dataclass
class Circle:
    center: np.ndarray   # x, y
    radius: float
    inliers: np.ndarray  # boolean mask over the points given to the fit
    arc_deg: float       # angular span of the inliers seen from the centre


def _circle_through(p1, p2, p3):
    """Centre and radius of the circle through three 2D points (None if collinear)."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.array([ux, uy]), math.hypot(ax - ux, ay - uy)


def _arc_span_deg(angles):
    """Smallest angular range (deg) covering all angles."""
    a = np.sort(np.mod(angles, 2 * math.pi))
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * math.pi]))
    return math.degrees(2 * math.pi - gaps.max())


def fit_outer_circle(xy, camera_xy, radius_range, tolerance, iterations, fixed_radius=0.0,
                     rng=None):
    """
    RANSAC circle fit for the cup's outer wall, seen from one side.

    Only circles with a radius in radius_range whose centre lies further from
    the camera than the sampled points are accepted, and only inliers on the
    circle's camera-facing half count: the camera can only see that half of
    the outer wall. That rules out the far inner wall, seen through the
    opening, which lies on the far half and curves the other way. With
    fixed_radius > 0
    the final centre is refined with the radius held at that value (more
    robust when only part of the wall is visible).
    """
    if len(xy) < 3:
        return None
    rng = rng or np.random.default_rng(0)

    def inlier_mask(center, r):
        near_side = (xy - center) @ (camera_xy - center) > 0
        return near_side & (np.abs(np.linalg.norm(xy - center, axis=1) - r) < tolerance)

    best, best_count = None, 0
    for _ in range(iterations):
        p = xy[rng.choice(len(xy), 3, replace=False)]
        fit = _circle_through(*p)
        if fit is None:
            continue
        center, r = fit
        if not radius_range[0] <= r <= radius_range[1]:
            continue
        if np.linalg.norm(center - camera_xy) <= np.linalg.norm(p.mean(axis=0) - camera_xy):
            continue  # centre on the camera's side: not the outer wall
        count = int(inlier_mask(center, r).sum())
        if count > best_count:
            best, best_count = (center, r), count
    if best is None:
        return None
    center, r = best
    for _ in range(3):  # refine on the inliers
        pts = xy[inlier_mask(center, r)]
        if len(pts) < 3:
            return None
        if fixed_radius > 0:
            r = fixed_radius
            for _ in range(10):  # Gauss-Newton on the centre, radius fixed
                d = pts - center
                dist = np.linalg.norm(d, axis=1)
                jac = -d / dist[:, None]
                step = np.linalg.lstsq(jac, -(dist - r), rcond=None)[0]
                center = center + step
        else:  # algebraic (Kasa) fit: x^2 + y^2 + D x + E y + F = 0
            a = np.column_stack([pts, np.ones(len(pts))])
            sol = np.linalg.lstsq(a, -(pts ** 2).sum(axis=1), rcond=None)[0]
            center = -sol[:2] / 2
            r = math.sqrt(max(center @ center - sol[2], 1e-12))
    inl = inlier_mask(center, r)
    angles = np.arctan2(xy[inl, 1] - center[1], xy[inl, 0] - center[0])
    return Circle(center, r, inl, _arc_span_deg(angles) if inl.any() else 0.0)


def estimate_handle(points, heights, circle, cup_height, gap, max_reach, height_band,
                    min_points, cluster_eps=0.006, cluster_min_samples=10, min_reach=0.0):
    """
    Handle direction from the points just outside the fitted outer wall.

    Returns (yaw or None, handle points). Candidates are points between
    `gap` and `max_reach` outside the wall and within height_band (fractions
    of the cup height) above the table. Only the biggest dense group of them
    counts as the handle: a real camera also leaves scattered stray points
    around a cup's outline (flying pixels), which must not look like a
    handle. Fewer than min_points in that group, or a group that does not
    stick out at least min_reach beyond the wall, means handle not visible:
    depth noise far from the camera pushes blobs of the front wall a few mm
    outward, but only a real handle sticks out ~18 mm.
    """
    d = points[:, :2] - circle.center
    dist = np.linalg.norm(d, axis=1)
    band = (heights >= height_band[0] * cup_height) & (heights <= height_band[1] * cup_height)
    sel = np.flatnonzero(band & (dist > circle.radius + gap) & (dist < circle.radius + max_reach))
    if len(sel) < min_points:
        return None, points[sel]
    labels = dbscan(points[sel], cluster_eps, cluster_min_samples)
    if labels.max() < 0:
        return None, points[sel[:0]]
    biggest = np.bincount(labels[labels >= 0]).argmax()
    sel = sel[labels == biggest]
    if len(sel) < min_points or (dist[sel] - circle.radius).max() < min_reach:
        return None, points[sel]
    v = d[sel] / dist[sel, None]
    mean = v.mean(axis=0)
    return math.atan2(mean[1], mean[0]), points[sel]
