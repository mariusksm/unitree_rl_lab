"""Geometry of the G1's rubber hands, shared by the Isaac asset, the catch check and the MuJoCo model of ``sim2sim/``.

Only numpy is imported here, so ``sim2sim/`` can load this module by path without Isaac Sim.

All values are in the ``*_wrist_yaw_link`` frame (x along the hand, z across it with the thumb at +z; the right hand
mirrors the left in y). Taken from Phillip's g1-catch (mj-lab-pipeline, branch ``phillip``, ``env.py`` and
``mdp.py``), which uses the same rubber hand mesh.
"""

from __future__ import annotations

import numpy as np

PALM_POINTS = ((0.09, -0.01, 0.0), (0.09, 0.01, 0.0))
"""Centre of the palm surface of the [left, right] hand."""

PALM_NORMALS = ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
"""Normal of the [left, right] palm, pointing out of the palm (the fingers curl toward it)."""

PARTS = ("palm", "fingers", "fingertips", "thumb")
"""Convex parts of a hand collider. One convex hull over the whole hand fills the cup of the palm (1.6 cm above the
palm centre; a ∅15 cm ball rests 1.8 cm farther out than on the real hand); separate hulls keep it open."""


def part_masks(points: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask per part for rubber-hand vertices given in the wrist_yaw_link frame, shape (n, 3)."""
    x, z = points[:, 0], points[:, 2]
    thumb = z >= 0.035
    return {
        "palm": (x < 0.11) & ~thumb,
        "fingers": (x >= 0.11) & (x < 0.145) & ~thumb,
        "fingertips": (x >= 0.145) & ~thumb,
        "thumb": thumb,
    }
