"""Distance of the ball to the real arm surface in the drops of ``probe_arm_contacts.py``.

The real surface is the visual mesh of each arm link in ``g1.xml`` (same meshes as the USD), the rubber hand included.
Per drop: the distance from the ball centre to the nearest triangle minus the ball radius, at its smallest within
50 ms after the first contact (gap below 5 mm). Negative means the ball overlaps the real arm (the collider lets it
in), positive the gap the collider leaves. Whether the arm stopped the ball at all shows in its vertical speed over
these 50 ms (free fall: 0.5 m/s faster downward; a contact brakes, stops or reverses it).

    python sim2sim/tools/analyze_arm_contacts.py probe.npz ~/unitree_mjcf/g1
"""

import argparse
import os

import mujoco
import numpy as np


def closest_point_distance(p: np.ndarray, tri: np.ndarray) -> float:
    """Distance from point p to a triangle soup tri (n, 3, 3) (Ericson, Real-Time Collision Detection 5.1.5)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp, cp = p - b, p - c
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        q = a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]  # inside the face
        # edges and vertices (order matters: later assignments win where their region applies)
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        q[e_ab] = (a + ab * (d1 / (d1 - d3))[:, None])[e_ab]
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        q[e_ac] = (a + ac * (d2 / (d2 - d6))[:, None])[e_ac]
        e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q[e_bc] = (b + (c - b) * w[:, None])[e_bc]
    q[(d1 <= 0) & (d2 <= 0)] = a[(d1 <= 0) & (d2 <= 0)]
    q[(d3 >= 0) & (d4 <= d3)] = b[(d3 >= 0) & (d4 <= d3)]
    q[(d6 >= 0) & (d5 <= d6)] = c[(d6 >= 0) & (d5 <= d6)]
    return float(np.sqrt(np.nanmin(np.einsum("ij,ij->i", p - q, p - q))))  # degenerate triangles give nan


def body_triangles(m: mujoco.MjModel, body: str) -> np.ndarray:
    """All mesh triangles of a body's visual geoms, in the body frame."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    tris = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g]:
            continue
        mesh = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]].astype(float)
        f = m.mesh_face[m.mesh_faceadr[mesh] : m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        tris.append((v @ rot.reshape(3, 3).T + m.geom_pos[g])[f])
    return np.concatenate(tris)


def to_local(points: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return (points - pos) @ rot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe")
    parser.add_argument("mjcf_dir")
    args = parser.parse_args()
    data = np.load(args.probe)
    m = mujoco.MjModel.from_xml_path(os.path.join(args.mjcf_dir, "g1.xml"))
    radius, links = float(data["radius"]), [str(n) for n in data["links"]]
    tris = {f"{s}_{n}": body_triangles(m, f"{s}_{n}") for s in ("left", "right") for n in links}
    print(f"forearm colliders: {bool(data['forearm_colliders'])}, ball radius {radius * 100:.1f} cm")
    header = f"{'side':6s} {'target':17s} {'drop':>9s} {'impact':>7s} {'deepest':>9s} {'at link':18s}"
    print(header + " arm stopped the ball")
    for i in range(len(data["side"])):
        side, target = str(data["side"][i]), str(data["target"][i])
        drop = f"{float(data['height'][i]) * 100:.0f}cm+{float(data['speed'][i]):.0f}"
        ball = data["ball_pos"][i]
        gaps = []
        for t in range(len(ball)):
            best = (np.inf, "")
            for k, link in enumerate(links):
                local = to_local(ball[t], data["link_pos"][i, t, k], data["link_quat"][i, t, k])
                if np.linalg.norm(local) > 0.35:
                    continue  # far from this link
                best = min(best, (closest_point_distance(local, tris[f"{side}_{link}"]) - radius, link))
            gaps.append(best)
        touching = [t for t, g in enumerate(gaps) if g[0] < 0.005]
        if not touching:
            print(f"{side:6s} {target:17s} {drop:>9s}  no contact, closest {min(g[0] for g in gaps) * 1000:.1f} mm")
            continue
        t0 = touching[0]  # first impact; the deepest point within the next 50 ms
        window = gaps[t0 : t0 + 10]
        deepest = min(window)
        vz = np.diff(ball[:, 2]) / 0.005  # m/s, negative = down
        before, after = vz[max(t0 - 1, 0)], vz[min(t0 + 9, len(vz) - 1)]
        stopped = "yes" if after > before + 0.2 else "NO, fell through"  # free fall: 0.5 m/s faster in 50 ms
        print(f"{side:6s} {target:17s} {drop:>9s}  {-before:5.1f}  {deepest[0] * 1000:6.1f} mm  {deepest[1]:18s} "
              f"{stopped} (vz {before:+.1f} -> {after:+.1f} m/s)")

if __name__ == "__main__":
    main()
