"""
shapes_3d.py — interactive 3D wireframe shapes (sphere, pyramid)
════════════════════════════════════════════════════════════════
Each shape owns its geometry + interaction state and renders itself
onto the frame using the TouchDesigner bloom pipeline (td_render).
"""
import math

import cv2
import numpy as np

import td_render as td
from config import SHAPES, PHYSICS


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TOUCHDESIGNER-STYLE WIREFRAME PALETTES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Per-shape accent colours (float32 BGR, 0-1)
# TD palette: one hue per shape, varying luminosity only
_TD_PALETTES = {
    'sphere': {
        'edge_near': np.array([0.90, 0.95, 0.70], np.float32),  # near-white cyan
        'edge_far':  np.array([0.15, 0.22, 0.08], np.float32),  # deep indigo
        'node_rim':  np.array([0.80, 0.90, 0.55], np.float32),
        'grabbed':   np.array([0.60, 1.00, 0.70], np.float32),  # soft green when held
    },
    'pyramid': {
        'edge_near': np.array([0.90, 0.88, 0.60], np.float32),  # warm gold-white
        'edge_far':  np.array([0.14, 0.12, 0.05], np.float32),
        'node_rim':  np.array([0.80, 0.75, 0.45], np.float32),
        'grabbed':   np.array([0.40, 0.80, 1.00], np.float32),  # cool blue when held
    },
}


def _rot_matrices(angle_x, angle_y):
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


def _draw_td_wireframe(img, proj, depths, edges,
                       node_r=2, grabbed=False, palette='sphere'):
    """
    TouchDesigner-style wireframe renderer.

    Pipeline:
      1. Draw hair-thin lines + nodes onto a float32 emission layer
         — depth fades from bright near to dim far
      2. Gaussian bloom pass on emission
      3. Additive composite onto img (already dimmed by hand_detect)
    """
    if not proj:
        return

    pal = _TD_PALETTES.get(palette, _TD_PALETTES['sphere'])
    h, w = img.shape[:2]
    em = td.make_emission(h, w)

    d_min = min(depths)
    d_max = max(depths)
    d_range = max(d_max - d_min, 1e-6)

    rim_col = pal['grabbed'] if grabbed else pal['node_rim']

    # ── Edges ──────────────────────────────────────────────────────────────────
    for a, b in edges:
        depth_t = ((depths[a] + depths[b]) * 0.5 - d_min) / d_range
        t = float(depth_t)
        col = pal['edge_near'] * (1.0 - t) + pal['edge_far'] * t
        td.draw_line_em(em, proj[a], proj[b], color=col)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    for i, (px, py) in enumerate(proj):
        depth_t = (depths[i] - d_min) / d_range
        # Nodes closer to camera are slightly larger
        r = max(1, int(node_r * (1.1 - depth_t * 0.5)))
        rim = rim_col * (1.0 - depth_t * 0.6)
        td.draw_node_em(em, (px, py), r=r,
                        core=td.TD_WHITE * (0.9 - depth_t * 0.7),
                        rim=rim)

    # ── Bloom & composite ──────────────────────────────────────────────────────
    bloomed = td.bloom(em, kernel=25, sigma=10.0, strength=1.1)
    td.add_emission(img, bloomed)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SHAPE BASE CLASS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class _Shape3D:
    """Interaction state shared by all shapes (position, pinches, inertia)."""

    def __init__(self, x, y, radius, color):
        self.x, self.y = float(x), float(y)
        self.radius = float(radius)
        self.color = color
        self.vx = self.vy = 0.0
        self.angle_x = self.angle_y = 0.0
        self.pinches = []
        self.locked_hand_id = None
        self.is_two_handed = False
        self.is_grabbed_single = False
        self.is_rotating_manually = False
        self.initial_radius = float(radius)
        self.initial_pinch_dist = 0
        self.two_hand_offset = (0, 0)
        self.drag_offset = (0, 0)
        self.last_pinch_pos = (0, 0)
        self.pinch_count = 0
        self.last_pinch_time = 0.0
        self.was_pinched_last_frame = False
        self.to_delete = False
        self.rotation_speed_mult = 1.0  # controlled by spread gesture

    def draw(self, img):
        raise NotImplementedError


class MatrixSphere3D(_Shape3D):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, SHAPES['sphere_color'])
        self._build(SHAPES['sphere_rings'], SHAPES['sphere_segments'])

    def _build(self, rings, segments):
        nodes = []
        for i in range(rings + 1):
            phi = (i / rings) * math.pi
            for j in range(segments):
                theta = (j / segments) * 2 * math.pi
                nodes.append([
                    math.sin(phi) * math.cos(theta),
                    math.cos(phi),
                    math.sin(phi) * math.sin(theta),
                ])
        self.nodes = np.array(nodes)
        self.edges = []
        for i in range(rings):
            for j in range(segments):
                c = i * segments + j
                self.edges.append((c, i * segments + (j + 1) % segments))
                self.edges.append((c, (i + 1) * segments + j))

    def draw(self, img):
        if self.locked_hand_id is None:
            spd = PHYSICS['rotation_auto_sphere'] * self.rotation_speed_mult
            self.angle_y += spd
            self.angle_x += spd
        R = _rot_matrices(self.angle_x, self.angle_y)
        rotated = self.nodes @ R.T
        proj = [(int(nx * self.radius + self.x), int(ny * self.radius + self.y))
                for nx, ny, _ in rotated]
        depths = [nz for _, _, nz in rotated]
        _draw_td_wireframe(img, proj, depths, self.edges,
                           node_r=2, grabbed=bool(self.pinches), palette='sphere')


class MatrixPyramid3D(_Shape3D):
    EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]

    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, SHAPES['pyramid_color'])
        # Copie propre pour chaque instance (pas de partage de tableau)
        self.nodes = np.array([(0, -1, 0), (-0.866, 0.5, -0.5),
                               (0.866, 0.5, -0.5), (0, 0.5, 1)], dtype=float)
        self.edges = list(self.EDGES)

    def draw(self, img):
        if self.locked_hand_id is None:
            self.angle_y += PHYSICS['rotation_auto_pyramid_y'] * self.rotation_speed_mult
            self.angle_x += PHYSICS['rotation_auto_pyramid_x'] * self.rotation_speed_mult
        R = _rot_matrices(self.angle_x, self.angle_y)
        rotated = self.nodes @ R.T
        proj = [(int(nx * self.radius + self.x), int(ny * self.radius + self.y))
                for nx, ny, _ in rotated]
        depths = [nz for _, _, nz in rotated]
        _draw_td_wireframe(img, proj, depths, self.edges,
                           node_r=4, grabbed=bool(self.pinches), palette='pyramid')
