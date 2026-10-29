"""Surface points of collision primitives (numpy only), so that Isaac and MuJoCo shapes are measured the same way.

All shapes are centred at the origin; capsules and cylinders lie along the local z axis (``along`` turns them).
"""

import numpy as np


def _sphere_dirs(n: int = 400) -> np.ndarray:
    """Evenly spread unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=-1
    )


def sphere(radius: float) -> np.ndarray:
    return radius * _sphere_dirs()


def capsule(radius: float, half_length: float) -> np.ndarray:
    """Cylinder of length 2 * half_length along z plus two hemispherical caps."""
    d = radius * _sphere_dirs()
    top, bottom = d[d[:, 2] >= 0], d[d[:, 2] < 0]
    return np.concatenate([top + [0.0, 0.0, half_length], bottom - [0.0, 0.0, half_length]])


def cylinder(radius: float, half_height: float, n: int = 64) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], axis=-1)
    return np.concatenate([ring + [0.0, 0.0, half_height], ring - [0.0, 0.0, half_height]])


def box(half_extents) -> np.ndarray:
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return corners * np.asarray(half_extents, dtype=float)


def along(points: np.ndarray, axis: str) -> np.ndarray:
    """Turn a z-aligned shape to the USD axis token "X", "Y" or "Z" (the shapes are symmetric about their axis)."""
    return {"X": points[:, [2, 1, 0]], "Y": points[:, [0, 2, 1]], "Z": points}[axis]


def aabb(points: np.ndarray) -> list[list[float]]:
    """Axis-aligned bounding box [[min x, y, z], [max x, y, z]]."""
    return [np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()]
