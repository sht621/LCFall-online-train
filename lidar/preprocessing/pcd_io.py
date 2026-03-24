"""Minimal PCD loader for binary xyz point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_xyz_pcd(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, 'rb') as f:
        content = f.read()

    marker = b'DATA binary\n'
    idx = content.find(marker)
    if idx < 0:
        raise ValueError(f'Unsupported PCD format: DATA binary header not found in {path}')

    header = content[: idx + len(marker)].decode('ascii', errors='replace').splitlines()
    meta = {}
    for line in header:
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        key, values = parts[0], parts[1:]
        meta[key] = values

    if meta.get('FIELDS') != ['x', 'y', 'z']:
        raise ValueError(f'Unsupported PCD fields in {path}: {meta.get("FIELDS")}')
    if meta.get('SIZE') != ['4', '4', '4'] or meta.get('TYPE') != ['F', 'F', 'F']:
        raise ValueError(f'Unsupported PCD dtype in {path}: SIZE={meta.get("SIZE")} TYPE={meta.get("TYPE")}')

    point_count = int(meta['POINTS'][0])
    payload = content[idx + len(marker):]
    points = np.frombuffer(payload, dtype=np.float32)
    if points.size != point_count * 3:
        raise ValueError(f'Point payload mismatch in {path}: expected {point_count * 3}, got {points.size}')
    return points.reshape(point_count, 3).astype(np.float32)
