#!/usr/bin/env python3
"""
Generate the ceramic_cup Gazebo model from config/cup.yaml.

    generate_cup_model.py <cup.yaml> <output model dir>

Writes <output>/model.config, <output>/model.sdf and
<output>/meshes/ceramic_cup.stl:
  - visual: a lathed cup body plus a swept D-shaped handle (mesh)
  - collision: primitives only (base disc, wall slabs, handle segments), which
    keep contacts stable during grasping
  - inertia: computed from the body mesh, scaled to the configured mass
Cup frame: origin at the centre of the base, z up, handle along +x.
"""
import math
import os
import struct
import sys

import numpy as np
import yaml

SEGMENTS = 64
HANDLE_SEGMENTS = 32
HANDLE_SIDES = 16
WALL_SLABS = 16
HANDLE_BOXES = 5


def lathe(profile, segments):
    """Revolve a closed (r, z) profile around z; returns triangles (N, 3, 3)."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    tris = []
    for i in range(len(profile)):
        (r0, z0), (r1, z1) = profile[i], profile[(i + 1) % len(profile)]
        for j in range(segments):
            a0, a1 = angles[j], angles[(j + 1) % segments]
            p00 = (r0 * math.cos(a0), r0 * math.sin(a0), z0)
            p01 = (r0 * math.cos(a1), r0 * math.sin(a1), z0)
            p10 = (r1 * math.cos(a0), r1 * math.sin(a0), z1)
            p11 = (r1 * math.cos(a1), r1 * math.sin(a1), z1)
            if r0 > 0.0:
                tris.append((p00, p01, p11))
            if r1 > 0.0:
                tris.append((p00, p11, p10))
    return np.array(tris)


def handle_arc(cup, steps):
    """Centreline of the handle: a half circle bulging out along +x."""
    h = cup['handle']
    t = np.linspace(-math.pi / 2, math.pi / 2, steps + 1)
    # Start/end slightly inside the wall so the handle merges with the body.
    x0 = cup['outer_diameter'] / 2 - 0.25 * h['thickness']
    centers = np.stack([x0 + h['radius'] * np.cos(t), np.zeros_like(t),
                        h['center_height'] + h['radius'] * np.sin(t)], axis=1)
    tangents = np.stack([-np.sin(t), np.zeros_like(t), np.cos(t)], axis=1)
    return t, centers, tangents


def handle_mesh(cup):
    """Sweep a round tube along the handle arc, with capped ends."""
    tube = cup['handle']['thickness'] / 2
    _, centers, tangents = handle_arc(cup, HANDLE_SEGMENTS)
    binormal = np.array([0.0, 1.0, 0.0])
    phis = np.linspace(0.0, 2.0 * math.pi, HANDLE_SIDES, endpoint=False)
    rings = np.array([[c + tube * (math.cos(p) * np.cross(binormal, tan) + math.sin(p) * binormal)
                       for p in phis] for c, tan in zip(centers, tangents)])
    tris = []
    for i in range(len(rings) - 1):
        for j in range(HANDLE_SIDES):
            k = (j + 1) % HANDLE_SIDES
            a, b, c, d = rings[i, j], rings[i, k], rings[i + 1, j], rings[i + 1, k]
            tris += [(a, c, d), (a, d, b)]
    for ring, center, flip in ((rings[0], centers[0], True), (rings[-1], centers[-1], False)):
        for j in range(HANDLE_SIDES):
            k = (j + 1) % HANDLE_SIDES
            tris.append((center, ring[k], ring[j]) if flip else (center, ring[j], ring[k]))
    return np.array(tris)


def mass_properties(tris, mass):
    """Volume, centre of mass and inertia about the COM of a closed mesh."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    vol6 = np.einsum('ij,ij->i', a, np.cross(b, c))
    volume = vol6.sum() / 6.0
    com = ((a + b + c) * vol6[:, None]).sum(0) / (4.0 * vol6.sum())
    s = a + b + c
    cov = np.array([[(vol6 / 120.0 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j]
                                      + s[:, i] * s[:, j])).sum() for j in range(3)]
                    for i in range(3)])
    cov = cov * mass / volume - mass * np.outer(com, com)
    return volume, com, np.trace(cov) * np.eye(3) - cov


def write_stl(path, tris):
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with open(path, 'wb') as f:
        f.write(b'ceramic_cup (generated from cup.yaml)'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        for n, tri in zip(normals, tris):
            f.write(struct.pack('<12fH', *n, *tri[0], *tri[1], *tri[2], 0))


def collision(name, pose, geometry, mu):
    return (f'      <collision name="{name}">\n'
            f'        <pose>{" ".join(f"{v:.5f}" for v in pose)}</pose>\n'
            f'        <geometry>{geometry}</geometry>\n'
            f'        <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>'
            f'</surface>\n'
            f'      </collision>\n')


def collisions(cup):
    r_out = cup['outer_diameter'] / 2
    wall, bottom, height, mu = (cup['wall_thickness'], cup['bottom_thickness'],
                                cup['height'], cup['friction'])
    out = collision('base', (0, 0, bottom / 2, 0, 0, 0),
                    f'<cylinder><radius>{r_out}</radius><length>{bottom}</length></cylinder>', mu)
    # Wall: slabs around the circle, slightly wider than a chord so there are no gaps.
    r_mid = r_out - wall / 2
    width = 2 * r_out * math.tan(math.pi / WALL_SLABS) * 1.02
    wall_h = height - bottom
    for i in range(WALL_SLABS):
        a = 2 * math.pi * i / WALL_SLABS
        out += collision(f'wall_{i:02d}',
                         (r_mid * math.cos(a), r_mid * math.sin(a), bottom + wall_h / 2, 0, 0, a),
                         f'<box><size>{wall:.5f} {width:.5f} {wall_h:.5f}</size></box>', mu)
    # Handle: boxes along the arc, each aligned with the local tangent.
    thick = cup['handle']['thickness']
    t, centers, _ = handle_arc(cup, HANDLE_BOXES * 2)
    seg = math.pi * cup['handle']['radius'] / HANDLE_BOXES * 1.1
    for i in range(HANDLE_BOXES):
        k = 2 * i + 1  # midpoint of each segment
        out += collision(f'handle_{i}', (*centers[k], 0, -t[k], 0),
                         f'<box><size>{thick} {thick} {seg:.5f}</size></box>', mu)
    return out


def main():
    cup = yaml.safe_load(open(sys.argv[1]))['cup']
    out_dir = sys.argv[2]
    r_out = cup['outer_diameter'] / 2
    r_in = r_out - cup['wall_thickness']
    height, bottom = cup['height'], cup['bottom_thickness']
    if not 0 < r_in < r_out or not 0 < bottom < height:
        raise SystemExit('cup.yaml: wall/bottom thickness must be smaller than the cup')

    os.makedirs(os.path.join(out_dir, 'meshes'), exist_ok=True)
    profile = [(0.0, 0.0), (r_out, 0.0), (r_out, height),
               (r_in, height), (r_in, bottom), (0.0, bottom)]
    body = lathe(profile, SEGMENTS)
    write_stl(os.path.join(out_dir, 'meshes', 'ceramic_cup.stl'),
              np.concatenate([body, handle_mesh(cup)]))
    # Mass properties from the body (the handle is small and overlaps the wall).
    _, com, inertia = mass_properties(body, cup['mass'])
    capacity_ml = math.pi * r_in ** 2 * (height - bottom) * 1e6

    with open(os.path.join(out_dir, 'model.config'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>ceramic_cup</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>
    Ceramic espresso cup, generated from cup_description/config/cup.yaml:
    {cup['outer_diameter'] * 1000:.0f} mm diameter, {height * 1000:.0f} mm tall,
    ~{capacity_ml:.0f} ml, {cup['mass']} kg.
  </description>
</model>
""")
    with open(os.path.join(out_dir, 'model.sdf'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<!-- Generated by cup_description/scripts/generate_cup_model.py from config/cup.yaml.
     Do not edit; change cup.yaml and rebuild. -->
<sdf version="1.8">
  <model name="ceramic_cup">
    <!-- Origin at the centre of the cup's base; the handle points along +x. -->
    <link name="cup_link">
      <inertial>
        <pose>0 0 {com[2]:.5f} 0 0 0</pose>
        <mass>{cup['mass']}</mass>
        <inertia>
          <ixx>{inertia[0, 0]:.4e}</ixx><iyy>{inertia[1, 1]:.4e}</iyy><izz>{inertia[2, 2]:.4e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry><mesh><uri>model://ceramic_cup/meshes/ceramic_cup.stl</uri></mesh></geometry>
        <material>
          <ambient>0.85 0.85 0.82 1</ambient>
          <diffuse>0.92 0.92 0.90 1</diffuse>
          <specular>0.6 0.6 0.6 1</specular>
        </material>
      </visual>
{collisions(cup)}    </link>
  </model>
</sdf>
""")
    print(f'ceramic_cup: {cup["outer_diameter"] * 1000:.0f} mm x {height * 1000:.0f} mm, '
          f'~{capacity_ml:.0f} ml, COM z {com[2] * 1000:.1f} mm -> {out_dir}')


if __name__ == '__main__':
    main()
