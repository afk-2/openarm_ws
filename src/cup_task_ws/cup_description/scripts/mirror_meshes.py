#!/usr/bin/env python3
"""
Write y-mirrored copies of openarm_description's collision STLs.

OpenArm builds the left arm by mirroring the right arm's meshes with
scale="1 -1 1". DART (Gazebo's physics engine) rejects negative mesh scales,
so the Gazebo launch swaps those meshes for these pre-mirrored copies.

    mirror_meshes.py <openarm_description share dir> <output dir>

Each <share>/assets/.../collision/<name>.stl becomes
<output>/assets/.../collision/<name>.stl with y negated and the triangle
winding reversed (so normals still point outwards).
"""
import pathlib
import re
import struct
import sys

import numpy as np


def read_stl(data):
    if data[:5] == b'solid' and b'facet' in data[:512]:
        vertices = re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data)
        return np.array(vertices, dtype=float).reshape(-1, 3, 3)
    count = struct.unpack('<I', data[80:84])[0]
    record = np.dtype([('normal', '<3f4'), ('v', '<9f4'), ('attr', '<u2')])
    return np.frombuffer(data[84:84 + count * 50], dtype=record)['v'].reshape(-1, 3, 3)


def write_stl(path, tris):
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    record = np.zeros(len(tris), dtype=[('normal', '<3f4'), ('v', '<9f4'), ('attr', '<u2')])
    record['normal'] = normals
    record['v'] = tris.reshape(-1, 9)
    with open(path, 'wb') as f:
        f.write(b'mirrored by cup_description'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        f.write(record.tobytes())


def main():
    share, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    meshes = sorted(p for p in (share / 'assets').rglob('*.stl') if 'collision' in p.parts)
    for src in meshes:
        tris = read_stl(src.read_bytes()).copy()
        tris[:, :, 1] *= -1.0
        tris = tris[:, ::-1]  # reverse winding
        dst = out / src.relative_to(share)
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_stl(dst, tris)
    print(f'mirrored {len(meshes)} collision meshes into {out}')


if __name__ == '__main__':
    main()
