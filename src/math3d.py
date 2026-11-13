"""
3D Math helpers, Euler/Quaternion orientation, ray-sphere/segment-sphere intersection,
and procedural Crimean Bridge 3D geometry builders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    ARCH_CENTER_Z,
    ARCH_HALF_SPAN_Z,
    ARCH_PEAK_Y,
    BRIDGE_FINISH_Z,
    BRIDGE_LENGTH,
    BRIDGE_START_Z,
    NUM_BARRIERS,
    PILLAR_SPACING,
    RAIL_CENTER_X,
    RAIL_DECK_Y,
    RAIL_HALF_WIDTH,
    ROAD_DECK_Y,
    ROAD_HALF_WIDTH,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def move_toward(current: float, target: float, max_delta: float) -> float:
    """Move `current` towards `target` by at most `max_delta` (used for self-levelling)."""
    if abs(target - current) <= max_delta:
        return target
    return current + (math.copysign(max_delta, target - current))


def wrap_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def euler_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Returns 3x3 rotation matrix transforming local coordinates (X=right, Y=up, Z=forward)
    into world coordinates using intrinsic Y (yaw) -> X (pitch) -> Z (roll).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # Ry(yaw) @ Rx(pitch) @ Rz(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return ry @ rx @ rz


def forward_from_angles(yaw: float, pitch: float) -> np.ndarray:
    """Unit forward vector (+Z local rotated by yaw and pitch)."""
    cp = math.cos(pitch)
    return np.array(
        [math.sin(yaw) * cp, -math.sin(pitch), math.cos(yaw) * cp],
        dtype=np.float32,
    )


def ray_sphere_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    max_dist: float,
) -> Optional[float]:
    """Returns distance along ray_dir (unit) if ray hits sphere within [0, max_dist], else None."""
    oc = ray_origin - sphere_center
    b = float(np.dot(oc, ray_dir))
    c = float(np.dot(oc, oc)) - sphere_radius * sphere_radius
    disc = b * b - c
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    t1 = -b - sqrt_d
    t2 = -b + sqrt_d
    t = t1 if t1 >= 0.0 else t2
    if 0.0 <= t <= max_dist:
        return t
    return None


def segment_sphere_intersect(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> Optional[float]:
    """
    Continuous swept collision test between a 3D segment [seg_start, seg_end] and a sphere.
    Returns normalized parameter t in [0.0, 1.0] of closest/first impact if hit, else None.
    """
    d = seg_end - seg_start
    len2 = float(np.dot(d, d))
    oc = seg_start - sphere_center
    r2 = sphere_radius * sphere_radius
    if len2 < 1e-10:
        return 0.0 if float(np.dot(oc, oc)) <= r2 else None

    b = float(np.dot(oc, d))
    c = float(np.dot(oc, oc)) - r2
    if c <= 0.0:
        return 0.0
    disc = b * b - len2 * c
    if disc < 0.0:
        return None
    t = (-b - math.sqrt(disc)) / len2
    if 0.0 <= t <= 1.0:
        return t
    return None


@dataclass
class AABB:
    min_pt: np.ndarray
    max_pt: np.ndarray
    tag: str = "obstacle"


class Mesh3D:
    """Compact triangle mesh stored as NumPy arrays for fast transformation and rendering."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
        two_sided: Optional[np.ndarray] = None,
        depth_bias: Optional[np.ndarray] = None,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float32)   # (V, 3)
        self.faces = np.asarray(faces, dtype=np.int32)           # (F, 3)
        self.colors = np.asarray(colors, dtype=np.float32)       # (F, 3) in 0..255
        if normals is None:
            self.normals = self._compute_normals()
        else:
            self.normals = np.asarray(normals, dtype=np.float32)
        if two_sided is None:
            self.two_sided = np.zeros(len(self.faces), dtype=bool)
        else:
            self.two_sided = np.asarray(two_sided, dtype=bool)
        if depth_bias is None:
            self.depth_bias = np.ones(len(self.faces), dtype=np.float32)
        else:
            self.depth_bias = np.asarray(depth_bias, dtype=np.float32)

    def _compute_normals(self) -> np.ndarray:
        if len(self.faces) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-6)
        return (n / lengths).astype(np.float32)


class MeshBuilder:
    """Helper to assemble boxes, prisms, cylinders, and arches into a single Mesh3D."""

    def __init__(self) -> None:
        self.verts: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, int, int]] = []
        self.colors: List[Tuple[int, int, int]] = []
        self.two_sided_flags: List[bool] = []
        self.depth_biases: List[float] = []

    def add_quad(
        self,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
        color: Tuple[int, int, int],
        two_sided: bool = False,
        depth_bias: float = 1.0,
    ) -> None:
        """
        Adds a quad (2 triangles).
        `depth_bias < 1.0` (e.g. 0.999) ensures flat surface decals (road markings)
        always draw cleanly on top of the coplanar road segment below them.
        """
        base = len(self.verts)
        self.verts.extend([p0, p1, p2, p3])
        self.faces.append((base, base + 1, base + 2))
        self.colors.append(color)
        self.two_sided_flags.append(two_sided)
        self.depth_biases.append(depth_bias)
        self.faces.append((base, base + 2, base + 3))
        self.colors.append(color)
        self.two_sided_flags.append(two_sided)
        self.depth_biases.append(depth_bias)

    def add_box(
        self,
        cx: float,
        cy: float,
        cz: float,
        sx: float,
        sy: float,
        sz: float,
        color: Tuple[int, int, int],
        skip_bottom: bool = False,
        skip_top: bool = False,
        skip_front: bool = False,
        skip_back: bool = False,
    ) -> None:
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = cz - hz, cz + hz

        c_top = (min(255, int(color[0] * 1.12)), min(255, int(color[1] * 1.12)), min(255, int(color[2] * 1.12)))
        c_side = color
        c_front = (int(color[0] * 0.9), int(color[1] * 0.9), int(color[2] * 0.9))
        c_bot = (int(color[0] * 0.65), int(color[1] * 0.65), int(color[2] * 0.65))

        if not skip_top:
            self.add_quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), c_top)
        if not skip_bottom:
            self.add_quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), c_bot)
        if not skip_front:
            self.add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), c_front)
        if not skip_back:
            self.add_quad((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), c_front)
        # Right (+X)
        self.add_quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), c_side)
        # Left (-X)
        self.add_quad((x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0), c_side)

    def add_beam(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        thickness: float,
        color: Tuple[int, int, int],
        two_sided: bool = False,
    ) -> None:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        ab = vb - va
        length = float(np.linalg.norm(ab))
        if length < 1e-4:
            return
        fwd = ab / length
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fwd, up_ref))) > 0.95:
            up_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(up_ref, fwd)
        right /= max(float(np.linalg.norm(right)), 1e-6)
        up = np.cross(fwd, right)
        h = thickness * 0.5

        corners_a = [
            va - right * h - up * h,
            va + right * h - up * h,
            va + right * h + up * h,
            va - right * h + up * h,
        ]
        corners_b = [
            vb - right * h - up * h,
            vb + right * h - up * h,
            vb + right * h + up * h,
            vb - right * h + up * h,
        ]
        for i in range(4):
            ni = (i + 1) % 4
            shade = 0.88 + 0.08 * i
            col = (
                min(255, int(color[0] * shade)),
                min(255, int(color[1] * shade)),
                min(255, int(color[2] * shade)),
            )
            self.add_quad(
                tuple(corners_a[i]),
                tuple(corners_a[ni]),
                tuple(corners_b[ni]),
                tuple(corners_b[i]),
                col,
                two_sided=two_sided,
            )

    def add_continuous_rib(
        self,
        points: List[Tuple[float, float, float]],
        thickness: float,
        color: Tuple[int, int, int],
    ) -> None:
        """
        Extrudes a continuous square-section beam along `points` with shared mitered rings
        so adjacent segments never overlap or z-fight at joints.
        """
        if len(points) < 2:
            return
        pts = [np.array(p, dtype=np.float32) for p in points]
        h = thickness * 0.5
        rings: List[List[np.ndarray]] = []
        for idx, pt in enumerate(pts):
            if idx == 0:
                tng = pts[1] - pts[0]
            elif idx == len(pts) - 1:
                tng = pts[-1] - pts[-2]
            else:
                tng = pts[idx + 1] - pts[idx - 1]
            tlen = float(np.linalg.norm(tng))
            fwd = tng / max(tlen, 1e-6)
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(fwd, up_ref))) > 0.95:
                up_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(up_ref, fwd)
            right /= max(float(np.linalg.norm(right)), 1e-6)
            up = np.cross(fwd, right)
            rings.append([
                pt - right * h - up * h,
                pt + right * h - up * h,
                pt + right * h + up * h,
                pt - right * h + up * h,
            ])

        for s in range(len(rings) - 1):
            ca = rings[s]
            cb = rings[s + 1]
            for i in range(4):
                ni = (i + 1) % 4
                shade = 0.88 + 0.08 * i
                col = (
                    min(255, int(color[0] * shade)),
                    min(255, int(color[1] * shade)),
                    min(255, int(color[2] * shade)),
                )
                self.add_quad(tuple(ca[i]), tuple(ca[ni]), tuple(cb[ni]), tuple(cb[i]), col)

    def build(self) -> Mesh3D:
        if not self.verts:
            return Mesh3D(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )
        return Mesh3D(
            np.array(self.verts, dtype=np.float32),
            np.array(self.faces, dtype=np.int32),
            np.array(self.colors, dtype=np.float32),
            two_sided=np.array(self.two_sided_flags, dtype=bool),
            depth_bias=np.array(self.depth_biases, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Procedural Models: Bridge, Fuel Truck, FPV Drone
# ---------------------------------------------------------------------------
def generate_barrier_layout(seed: int = 42) -> List[AABB]:
    """Deterministic chicane concrete blocks along the bridge deck."""
    rng = np.random.default_rng(seed)
    barriers: List[AABB] = []
    z_positions = np.linspace(BRIDGE_START_Z + 120.0, BRIDGE_FINISH_Z - 90.0, NUM_BARRIERS)
    for idx, z in enumerate(z_positions):
        lane_side = -1.0 if idx % 2 == 0 else 1.0
        jitter_x = float(rng.uniform(-2.2, 2.2))
        cx = lane_side * 5.8 + jitter_x
        w = 5.4
        h = 1.65
        d = 2.4
        barriers.append(
            AABB(
                min_pt=np.array([cx - w * 0.5, ROAD_DECK_Y, z - d * 0.5], dtype=np.float32),
                max_pt=np.array([cx + w * 0.5, ROAD_DECK_Y + h, z + d * 0.5], dtype=np.float32),
                tag="barrier",
            )
        )
    return barriers


def build_crimean_bridge_mesh(barriers: List[AABB]) -> Tuple[Mesh3D, List[AABB]]:
    """
    Constructs the Crimean Bridge 3D scene without intersecting/coplanar Z-fighting geometry:
    - 4-lane highway bridge deck (open internal joints) + guardrails + road markings + light poles
    - Parallel railway bridge deck
    - Concrete marine piers & main arch anchor piers terminating cleanly under the decks
    - Signature twin white continuous steel shipping arches + hangers & cross-braces
    - Finish checkpoint portal
    """
    mb = MeshBuilder()
    colliders: List[AABB] = list(barriers)

    # Distant Kerch & Taman coastline embankments flush with approach deck height
    land_h = ROAD_DECK_Y - 0.5
    mb.add_box(0.0, land_h * 0.5, -810.0, 520.0, land_h, 110.0, (108, 122, 86), skip_bottom=True)
    mb.add_box(0.0, land_h * 0.5, 810.0, 520.0, land_h, 110.0, (118, 126, 88), skip_bottom=True)

    # 2. Segmented Highway Deck + Guardrails + Markings + Parallel Rail Bridge
    #    60 shorter segments (25m) with internal +-Z end-caps omitted -> zero Painter's sort wedges!
    num_deck_segs = 60
    z0_bridge = -BRIDGE_LENGTH * 0.5
    seg_len = BRIDGE_LENGTH / num_deck_segs
    asphalt_col = (60, 64, 70)
    rail_col = (182, 188, 196)

    for i in range(num_deck_segs):
        cz = z0_bridge + (i + 0.5) * seg_len
        skip_back = i > 0
        skip_front = i < num_deck_segs - 1

        # Highway concrete box girder & asphalt top
        mb.add_box(
            0.0,
            ROAD_DECK_Y - 1.1,
            cz,
            ROAD_HALF_WIDTH * 2.0,
            2.2,
            seg_len,
            asphalt_col,
            skip_front=skip_front,
            skip_back=skip_back,
        )

        # White center divider & dashed lane lines (+Y single-sided decals with depth_bias=0.999)
        mb.add_quad(
            (-0.22, ROAD_DECK_Y + 0.05, cz - seg_len * 0.5),
            (-0.22, ROAD_DECK_Y + 0.05, cz + seg_len * 0.5),
            (0.22, ROAD_DECK_Y + 0.05, cz + seg_len * 0.5),
            (0.22, ROAD_DECK_Y + 0.05, cz - seg_len * 0.5),
            (230, 232, 235),
            two_sided=False,
            depth_bias=0.999,
        )
        for lane_x in (-6.5, 6.5):
            mb.add_quad(
                (lane_x - 0.15, ROAD_DECK_Y + 0.05, cz - seg_len * 0.24),
                (lane_x - 0.15, ROAD_DECK_Y + 0.05, cz + seg_len * 0.24),
                (lane_x + 0.15, ROAD_DECK_Y + 0.05, cz + seg_len * 0.24),
                (lane_x + 0.15, ROAD_DECK_Y + 0.05, cz - seg_len * 0.24),
                (215, 215, 210),
                two_sided=False,
                depth_bias=0.999,
            )

        # Side steel barrier rails resting cleanly on top of deck edge (no internal caps)
        mb.add_box(
            -ROAD_HALF_WIDTH + 0.35,
            ROAD_DECK_Y + 0.65,
            cz,
            0.6,
            1.3,
            seg_len,
            rail_col,
            skip_bottom=True,
            skip_front=skip_front,
            skip_back=skip_back,
        )
        mb.add_box(
            ROAD_HALF_WIDTH - 0.35,
            ROAD_DECK_Y + 0.65,
            cz,
            0.6,
            1.3,
            seg_len,
            rail_col,
            skip_bottom=True,
            skip_front=skip_front,
            skip_back=skip_back,
        )

        # Parallel Railway Deck (+X side)
        mb.add_box(
            RAIL_CENTER_X,
            RAIL_DECK_Y - 1.3,
            cz,
            RAIL_HALF_WIDTH * 2.0,
            2.6,
            seg_len,
            (86, 90, 96),
            skip_front=skip_front,
            skip_back=skip_back,
        )
        mb.add_box(
            RAIL_CENTER_X - RAIL_HALF_WIDTH + 0.3,
            RAIL_DECK_Y + 0.8,
            cz,
            0.45,
            1.6,
            seg_len,
            (145, 150, 158),
            skip_bottom=True,
            skip_front=skip_front,
            skip_back=skip_back,
        )
        mb.add_box(
            RAIL_CENTER_X + RAIL_HALF_WIDTH - 0.3,
            RAIL_DECK_Y + 0.8,
            cz,
            0.45,
            1.6,
            seg_len,
            (145, 150, 158),
            skip_bottom=True,
            skip_front=skip_front,
            skip_back=skip_back,
        )

        # Light poles mounted cleanly ON TOP of the side guardrails (no intersection!)
        if i % 3 == 0:
            for side_x in (-ROAD_HALF_WIDTH + 0.35, ROAD_HALF_WIDTH - 0.35):
                mb.add_box(side_x, ROAD_DECK_Y + 5.15, cz, 0.22, 7.7, 0.22, (150, 158, 168), skip_bottom=True)
                inward = -1.2 if side_x > 0 else 1.2
                mb.add_box(side_x + inward * 0.5, ROAD_DECK_Y + 8.95, cz, 1.4, 0.25, 0.35, (250, 238, 185))

    # 3. Concrete Marine Piers strictly UNDER both decks (never intersecting asphalt!)
    road_pier_top = ROAD_DECK_Y - 2.2
    road_pier_h = road_pier_top + 1.0
    road_pier_cy = (road_pier_top - 1.0) * 0.5
    rail_pier_top = RAIL_DECK_Y - 2.6
    rail_pier_h = rail_pier_top + 1.0
    rail_pier_cy = (rail_pier_top - 1.0) * 0.5

    z_pier = -BRIDGE_LENGTH * 0.5 + 45.0
    while z_pier <= BRIDGE_LENGTH * 0.5 - 45.0:
        if abs(z_pier - ARCH_CENTER_Z) > ARCH_HALF_SPAN_Z - 12.0:
            mb.add_box(-7.2, road_pier_cy, z_pier, 3.4, road_pier_h, 4.2, (162, 166, 170), skip_bottom=True, skip_top=True)
            mb.add_box(7.2, road_pier_cy, z_pier, 3.4, road_pier_h, 4.2, (162, 166, 170), skip_bottom=True, skip_top=True)
            mb.add_box(RAIL_CENTER_X - 3.5, rail_pier_cy, z_pier, 2.8, rail_pier_h, 4.0, (154, 158, 164), skip_bottom=True, skip_top=True)
            mb.add_box(RAIL_CENTER_X + 3.5, rail_pier_cy, z_pier, 2.8, rail_pier_h, 4.0, (154, 158, 164), skip_bottom=True, skip_top=True)
        z_pier += PILLAR_SPACING

    # Massive white-concrete main arch anchor piers UNDER the highway/rail decks
    for end_z in (ARCH_CENTER_Z - ARCH_HALF_SPAN_Z, ARCH_CENTER_Z + ARCH_HALF_SPAN_Z):
        mb.add_box(0.0, road_pier_cy, end_z, ROAD_HALF_WIDTH * 2.35, road_pier_h, 12.0, (215, 220, 225), skip_bottom=True, skip_top=True)
        mb.add_box(RAIL_CENTER_X, rail_pier_cy, end_z, RAIL_HALF_WIDTH * 2.5, rail_pier_h, 12.0, (210, 216, 222), skip_bottom=True, skip_top=True)
        # Side arch anchor pylons cleanly outside road deck (+-13.0)
        for side_sign in (-1.0, 1.0):
            px = side_sign * (ROAD_HALF_WIDTH + 1.55)
            mb.add_box(px, ROAD_DECK_Y + 4.9, end_z, 2.8, 14.2, 8.0, (232, 236, 240), skip_bottom=True)
            colliders.append(
                AABB(
                    min_pt=np.array([px - 1.5, ROAD_DECK_Y, end_z - 4.0], dtype=np.float32),
                    max_pt=np.array([px + 1.5, ROAD_DECK_Y + 12.0, end_z + 4.0], dtype=np.float32),
                    tag="arch_pylon",
                )
            )

    # 4. Iconic White Steel Shipping Arches (Continuous vertex-shared ribs!)
    arch_white = (242, 245, 250)
    arch_steps = 18

    def build_arch_span(center_x: float, half_w: float, base_y: float, peak_y: float, add_colliders: bool) -> None:
        pts_left: List[Tuple[float, float, float]] = []
        pts_right: List[Tuple[float, float, float]] = []
        for s in range(arch_steps + 1):
            u = -1.0 + 2.0 * (s / arch_steps)
            z = ARCH_CENTER_Z + u * ARCH_HALF_SPAN_Z
            y = base_y + (peak_y - base_y) * (1.0 - u * u)
            pts_left.append((center_x - half_w, y, z))
            pts_right.append((center_x + half_w, y, z))

        mb.add_continuous_rib(pts_left, 2.0, arch_white)
        mb.add_continuous_rib(pts_right, 2.0, arch_white)

        for s in range(arch_steps):
            if add_colliders:
                for p_a, p_b in ((pts_left[s], pts_left[s + 1]), (pts_right[s], pts_right[s + 1])):
                    mn = np.minimum(p_a, p_b) - np.array([1.2, 1.2, 1.2])
                    mx = np.maximum(p_a, p_b) + np.array([1.2, 1.2, 1.2])
                    colliders.append(AABB(min_pt=mn.astype(np.float32), max_pt=mx.astype(np.float32), tag="arch"))

            if s > 1 and s < arch_steps - 1 and s % 2 == 0:
                mb.add_beam(pts_left[s], pts_right[s], 1.15, (226, 232, 240))
                if s + 2 <= arch_steps - 1:
                    mb.add_beam(pts_left[s], pts_right[s + 2], 0.65, (212, 220, 230))
                if add_colliders and pts_left[s][1] > base_y + 10.0:
                    cy = pts_left[s][1]
                    cz = pts_left[s][2]
                    colliders.append(
                        AABB(
                            min_pt=np.array([center_x - half_w, cy - 0.8, cz - 0.8], dtype=np.float32),
                            max_pt=np.array([center_x + half_w, cy + 0.8, cz + 0.8], dtype=np.float32),
                            tag="arch_crossbeam",
                        )
                    )

            if 1 <= s <= arch_steps - 1:
                lx, ly, lz = pts_left[s]
                rx, ry, rz = pts_right[s]
                if ly > base_y + 2.5:
                    mb.add_beam((lx, base_y + 1.3, lz), (lx, ly - 0.85, lz), 0.22, (195, 202, 212), two_sided=True)
                    mb.add_beam((rx, base_y + 1.3, rz), (rx, ry - 0.85, rz), 0.22, (195, 202, 212), two_sided=True)

    build_arch_span(0.0, ROAD_HALF_WIDTH + 0.9, ROAD_DECK_Y, ARCH_PEAK_Y, add_colliders=True)
    build_arch_span(RAIL_CENTER_X, RAIL_HALF_WIDTH + 0.8, RAIL_DECK_Y, ARCH_PEAK_Y + 5.0, add_colliders=False)

    # 5. Concrete Chicane Barriers on the Roadway (orange-white hazard blocks)
    for idx, b in enumerate(barriers):
        cx = float((b.min_pt[0] + b.max_pt[0]) * 0.5)
        cy = float((b.min_pt[1] + b.max_pt[1]) * 0.5)
        cz = float((b.min_pt[2] + b.max_pt[2]) * 0.5)
        sx = float(b.max_pt[0] - b.min_pt[0])
        sy = float(b.max_pt[1] - b.min_pt[1])
        sz = float(b.max_pt[2] - b.min_pt[2])
        col = (228, 112, 34) if idx % 2 == 0 else (218, 220, 224)
        mb.add_box(cx, cy, cz, sx, sy, sz, col, skip_bottom=True)
        # Reflective top stripe resting cleanly on top of barrier
        mb.add_box(cx, cy + sy * 0.5 + 0.125, cz, sx * 0.92, 0.25, sz * 1.02, (245, 210, 45), skip_bottom=True)

    # 6. Finish Portal Banner at Z = BRIDGE_FINISH_Z (Evacuation / Safe Zone)
    fz = BRIDGE_FINISH_Z
    mb.add_box(-ROAD_HALF_WIDTH + 0.6, ROAD_DECK_Y + 5.5, fz, 1.2, 11.0, 1.2, (46, 185, 86), skip_bottom=True)
    mb.add_box(ROAD_HALF_WIDTH - 0.6, ROAD_DECK_Y + 5.5, fz, 1.2, 11.0, 1.2, (46, 185, 86), skip_bottom=True)
    mb.add_box(0.0, ROAD_DECK_Y + 10.6, fz, ROAD_HALF_WIDTH * 2.0, 1.5, 1.2, (40, 200, 92))
    # Checkered finish line on asphalt (+Y single-sided decal)
    for k in range(-6, 6):
        c_chk = (245, 245, 245) if k % 2 == 0 else (30, 30, 30)
        mb.add_quad(
            (k * 2.0, ROAD_DECK_Y + 0.06, fz - 1.5),
            (k * 2.0, ROAD_DECK_Y + 0.06, fz + 1.5),
            ((k + 1) * 2.0, ROAD_DECK_Y + 0.06, fz + 1.5),
            ((k + 1) * 2.0, ROAD_DECK_Y + 0.06, fz - 1.5),
            c_chk,
            two_sided=False,
            depth_bias=0.999,
        )

    # Side guardrail boundary colliders
    colliders.append(
        AABB(
            min_pt=np.array([-ROAD_HALF_WIDTH - 2.0, ROAD_DECK_Y, -BRIDGE_LENGTH * 0.5], dtype=np.float32),
            max_pt=np.array([-ROAD_HALF_WIDTH + 0.6, ROAD_DECK_Y + 2.2, BRIDGE_LENGTH * 0.5], dtype=np.float32),
            tag="guardrail",
        )
    )
    colliders.append(
        AABB(
            min_pt=np.array([ROAD_HALF_WIDTH - 0.6, ROAD_DECK_Y, -BRIDGE_LENGTH * 0.5], dtype=np.float32),
            max_pt=np.array([ROAD_HALF_WIDTH + 2.0, ROAD_DECK_Y + 2.2, BRIDGE_LENGTH * 0.5], dtype=np.float32),
            tag="guardrail",
        )
    )

    return mb.build(), colliders


def build_fuel_truck_mesh(has_cage: bool = False, twin_turret: bool = False) -> Mesh3D:
    """
    Builds a detailed 3D Fuel Tanker Truck (Бензовоз КАМАЗ/Урал):
    - Chassis frame + 6 wheel pairs
    - Cab + windshield + headlights
    - Long cylindrical/octagonal orange-red "ОГНЕОПАСНО" fuel cistern
    - Optional anti-drone cope-cage ("Мангал")
    """
    mb = MeshBuilder()

    # Chassis dark steel frame (local origin at road contact under truck center, +Z = forward)
    mb.add_box(0.0, 0.85, 0.0, 2.5, 0.55, 11.2, (42, 45, 50))

    # Wheels (3 axles: front Z=+3.8, mid Z=-1.6, rear Z=-3.8)
    for az in (3.8, -1.6, -3.8):
        for wx in (-1.35, 1.35):
            mb.add_box(wx, 0.58, az, 0.55, 1.15, 1.15, (24, 26, 28))
            mb.add_box(wx * 1.03, 0.58, az, 0.58, 0.55, 0.55, (130, 135, 142))

    # Cab at front (+Z)
    cab_col = (212, 68, 36)
    mb.add_box(0.0, 2.15, 3.65, 2.75, 2.15, 2.85, cab_col)
    # Windshield
    mb.add_box(0.0, 2.48, 5.05, 2.35, 0.88, 0.14, (125, 192, 232))
    # Front bumper & glowing headlights
    mb.add_box(0.0, 0.95, 5.18, 2.85, 0.55, 0.35, (65, 68, 74))
    mb.add_box(-0.98, 1.05, 5.32, 0.45, 0.28, 0.12, (255, 248, 195))
    mb.add_box(0.98, 1.05, 5.32, 0.45, 0.28, 0.12, (255, 248, 195))

    # Fuel Cistern
    tank_z = -1.35
    tank_len = 6.9
    mb.add_box(0.0, 2.25, tank_z, 2.85, 1.55, tank_len, (236, 128, 28))
    mb.add_box(0.0, 2.25, tank_z, 2.15, 2.15, tank_len * 0.98, (244, 142, 34))
    # Red hazard stripe cleanly wrapping outside tank width (2.94 > 2.85)
    mb.add_box(0.0, 2.28, tank_z, 2.94, 0.42, tank_len * 0.92, (215, 38, 32))
    mb.add_box(0.0, 3.45, tank_z, 0.95, 0.25, tank_len * 0.82, (165, 170, 178))
    mb.add_box(-0.95, 1.05, -5.55, 0.48, 0.25, 0.15, (240, 28, 28))
    mb.add_box(0.95, 1.05, -5.55, 0.48, 0.25, 0.15, (240, 28, 28))

    # EW Jammer dome on cab roof
    mb.add_box(-0.75, 3.42, 3.1, 0.48, 0.48, 0.48, (95, 195, 225))

    if has_cage:
        # Steel anti-drone cage ("Мангал") above tank and cab
        cage_col = (118, 126, 134)
        for cz in (-4.4, -1.3, 1.8, 4.6):
            mb.add_beam((-1.65, 1.4, cz), (-1.65, 4.15, cz), 0.12, cage_col, two_sided=True)
            mb.add_beam((1.65, 1.4, cz), (1.65, 4.15, cz), 0.12, cage_col, two_sided=True)
            mb.add_beam((-1.65, 4.15, cz), (1.65, 4.15, cz), 0.12, cage_col, two_sided=True)
        for sx in (-1.65, -0.8, 0.0, 0.8, 1.65):
            mb.add_beam((sx, 4.15, -4.5), (sx, 4.15, 4.7), 0.10, cage_col, two_sided=True)

    return mb.build()


def build_turret_mesh(twin: bool = False) -> Mesh3D:
    """Rotatable roof AA turret mounted on the truck cab."""
    mb = MeshBuilder()
    mb.add_box(0.0, 0.22, 0.0, 0.85, 0.42, 0.85, (72, 78, 86))
    if twin:
        mb.add_beam((-0.24, 0.34, 0.1), (-0.24, 0.34, 1.75), 0.13, (45, 50, 56), two_sided=True)
        mb.add_beam((0.24, 0.34, 0.1), (0.24, 0.34, 1.75), 0.13, (45, 50, 56), two_sided=True)
    else:
        mb.add_beam((0.0, 0.34, 0.1), (0.0, 0.34, 1.65), 0.15, (45, 50, 56), two_sided=True)
    return mb.build()


def build_fpv_drone_mesh(shaped_charge: bool = False) -> Mesh3D:
    """
    Quadcopter X-frame FPV kamikaze drone with 4 rotor discs, battery pack,
    FPV camera pod, and under-slung warhead (PG-7VL RPG rocket or frag cylinder).
    Scaled slightly so the truck gunner can visually acquire & lead it at 120m!
    """
    mb = MeshBuilder()
    # Carbon X-frame body
    mb.add_box(0.0, 0.0, 0.0, 0.38, 0.16, 0.72, (42, 46, 52))
    # Top blue LiPo battery
    mb.add_box(0.0, 0.15, -0.05, 0.26, 0.15, 0.42, (48, 136, 235))
    # Front FPV camera + red LED beacon for gameplay readability
    mb.add_box(0.0, 0.05, 0.40, 0.20, 0.18, 0.16, (240, 62, 48))

    # 4 carbon arms + bright rotor rings
    arm_coords = [(-0.58, 0.52), (0.58, 0.52), (-0.58, -0.52), (0.58, -0.52)]
    for ax, az in arm_coords:
        mb.add_beam((0.0, 0.0, 0.0), (ax, 0.02, az), 0.08, (60, 65, 72), two_sided=True)
        rotor_col = (245, 90, 45) if az > 0 else (90, 225, 110)
        mb.add_box(ax, 0.09, az, 0.48, 0.04, 0.48, rotor_col)

    # Under-slung warhead
    if shaped_charge:
        mb.add_box(0.0, -0.22, 0.18, 0.26, 0.26, 0.95, (92, 112, 62))
        mb.add_box(0.0, -0.22, 0.72, 0.14, 0.14, 0.28, (168, 172, 160))
    else:
        mb.add_box(0.0, -0.20, 0.08, 0.22, 0.22, 0.65, (115, 122, 84))

    return mb.build()
