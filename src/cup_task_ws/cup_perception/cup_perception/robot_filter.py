"""
Robot self-filter: remove camera points that lie on the robot itself.

The URDF's collision shapes (meshes, boxes, cylinders, spheres) are covered
once with sample points about `spacing` apart. Each frame, every link's
samples are moved to where TF says the link is, and camera points closer than
`margin` to any sample are removed. So an arm reaching over the table can
never be mistaken for a cup.
"""
import math
import os
import re
import struct
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import numpy as np
from scipy.spatial import cKDTree


def _rpy_matrix(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    m = np.eye(4)
    m[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    m[:3, 3] = xyz
    return m


def _resolve(uri):
    if uri.startswith('package://'):
        pkg, rel = uri[len('package://'):].split('/', 1)
        return os.path.join(get_package_share_directory(pkg), rel)
    if uri.startswith('file://'):
        return uri[len('file://'):]
    return uri


def _read_stl(path):
    data = open(path, 'rb').read()
    if data[:5] == b'solid' and b'facet' in data[:512]:
        v = re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data)
        return np.array(v, dtype=float).reshape(-1, 3, 3)
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')])
    return np.frombuffer(data[84:84 + n * 50], dtype=rec)['v'].reshape(-1, 3, 3).astype(float)


def _sample_triangles(tris, spacing, rng):
    """Area-weighted random points on a triangle mesh, ~1 per spacing^2."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    count = int(max(area.sum() / spacing ** 2, 1))
    pick = rng.choice(len(tris), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])
    return np.vstack([pts, tris.reshape(-1, 3)])  # include vertices for sharp corners


def _sample_box(size, spacing):
    axes = [np.linspace(-s / 2, s / 2, max(2, int(math.ceil(s / spacing)) + 1)) for s in size]
    g = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
    on_surface = np.any(np.isclose(np.abs(g), np.array(size) / 2), axis=1)
    return g[on_surface]


def _sample_cylinder(radius, length, spacing):
    n = max(8, int(2 * math.pi * radius / spacing))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    zs = np.linspace(-length / 2, length / 2, max(2, int(length / spacing) + 1))
    side = np.array([[radius * math.cos(a), radius * math.sin(a), z] for z in zs for a in t])
    rs = np.linspace(0, radius, max(2, int(radius / spacing) + 1))
    caps = np.array([[r * math.cos(a), r * math.sin(a), z] for z in (-length / 2, length / 2)
                     for r in rs for a in t])
    return np.vstack([side, caps])


def _sample_sphere(radius, spacing):
    n = max(20, int(4 * math.pi * radius ** 2 / spacing ** 2))
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), math.pi * (1 + 5 ** 0.5) * i
    return radius * np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                                     np.cos(phi)])


def link_samples(urdf_xml, spacing, logger=None):
    """{link name: N x 3 sample points in the link frame} for all collision shapes."""
    rng = np.random.default_rng(0)
    out = {}
    for link in ET.fromstring(urdf_xml).iter('link'):
        parts = []
        for col in link.findall('collision'):
            o = col.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            geom = col.find('geometry')
            try:
                if geom.find('mesh') is not None:
                    m = geom.find('mesh')
                    scale = np.array([float(v) for v in m.get('scale', '1 1 1').split()])
                    pts = _sample_triangles(_read_stl(_resolve(m.get('filename'))) * scale,
                                            spacing, rng)
                elif geom.find('box') is not None:
                    pts = _sample_box([float(v) for v in geom.find('box').get('size').split()],
                                      spacing)
                elif geom.find('cylinder') is not None:
                    c = geom.find('cylinder')
                    pts = _sample_cylinder(float(c.get('radius')), float(c.get('length')), spacing)
                elif geom.find('sphere') is not None:
                    pts = _sample_sphere(float(geom.find('sphere').get('radius')), spacing)
                else:
                    continue
            except (OSError, ValueError) as e:
                if logger:
                    logger.warn(f'self-filter: skipping a shape of {link.get("name")}: {e}')
                continue
            t = _rpy_matrix(xyz, rpy)
            parts.append(pts @ t[:3, :3].T + t[:3, 3])
        if parts:
            out[link.get('name')] = np.vstack(parts)
    return out


class RobotSelfFilter:

    def __init__(self, urdf_xml, spacing, margin, logger=None):
        self.margin = margin
        self.samples = link_samples(urdf_xml, spacing, logger)

    def robot_points(self, link_transforms, lower=None, upper=None):
        """All link samples placed in the world, optionally only inside a box."""
        pts = [s @ m[:3, :3].T + m[:3, 3] for name, s in self.samples.items()
               if (m := link_transforms.get(name)) is not None]
        if not pts:
            return np.zeros((0, 3))
        pts = np.vstack(pts)
        if lower is not None:
            grow = self.margin
            keep = np.all((pts >= np.asarray(lower) - grow) & (pts <= np.asarray(upper) + grow),
                          axis=1)
            pts = pts[keep]
        return pts

    def split(self, points, robot_pts):
        """(points not on the robot, points on the robot)."""
        if len(points) == 0 or len(robot_pts) == 0:
            return points, np.zeros((0, 3))
        dist, _ = cKDTree(robot_pts).query(points, distance_upper_bound=self.margin)
        on_robot = np.isfinite(dist)
        return points[~on_robot], points[on_robot]
