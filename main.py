"""
main.py — Vision AI : radial menu, shapes 3D, modes (mouse / glob)
══════════════════════════════════════════════════════════════════
Architecture :
  CameraStream      → background thread, frame producer
  DetectionThread   → background thread, hand detection (MediaPipe + 1€)
  main loop         → state machine : menu / shapes / mouse / glob
"""
import math
import sys
import threading
import time

import cv2
import numpy as np
import setup_check

import gestures as gest
import glob_metal
import virt_mouse
import td_render as td
import hand_detect as hand_detect_mod
from hand_detect import hand_detect, release, pick_detect_resolution
from config import CAM, MENU, SHAPES, PHYSICS, GESTURE
from shapes_3d import MatrixSphere3D, MatrixPyramid3D

DETECT_W,  DETECT_H  = CAM['detect_w'],  CAM['detect_h']
DISPLAY_W, DISPLAY_H = CAM['display_w'], CAM['display_h']

# ── Radial menu constants (from config) ─────────────────────────────────────────
MENU_RADIUS      = MENU['radius']
MENU_ICON_RADIUS = MENU['icon_radius']

# (label, angle_deg, color_BGR)
MENU_OPTIONS = [
    ("Shapes",  90,  (180, 100, 255)),
    ("Mouse",  270,  (200, 200, 200)),
    ("Glob",   180,  (160, 220, 200)),   # gauche — métal liquide
    ("Quit",     0,  (100, 120, 255)),
]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MINI-SHAPE ICON RENDERERS (dessinées dans les bulles du menu)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Géométrie sphère mini (moins de segments pour la lisibilité à petite taille)
_SPHERE_MINI_NODES = None
_SPHERE_MINI_EDGES = []


def _build_sphere_mini(rings=5, segments=8):
    global _SPHERE_MINI_NODES, _SPHERE_MINI_EDGES
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
    _SPHERE_MINI_NODES = np.array(nodes)
    edges = []
    for i in range(rings):
        for j in range(segments):
            c = i * segments + j
            edges.append((c, i * segments + (j + 1) % segments))
            edges.append((c, (i + 1) * segments + j))
    _SPHERE_MINI_EDGES = edges


_build_sphere_mini()

# Pyramide mini (mêmes proportions que MatrixPyramid3D)
_PYRAMID_MINI_NODES = np.array([(0, -1, 0), (-0.866, 0.5, -0.5),
                                 (0.866, 0.5, -0.5), (0, 0.5, 1)], dtype=float)
_PYRAMID_MINI_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]


def _rot_matrices(angle_x, angle_y):
    """Shared rotation matrices (Ry @ Rx) for icons and 3D shapes."""
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


def _draw_sphere_icon(img, ox, oy, r, color, angle_y, angle_x, hot):
    """Dessine un wireframe sphère miniature centré en (ox, oy) dans un rayon r."""
    icon_r = r * 0.60
    R = _rot_matrices(angle_x, angle_y)
    rotated = _SPHERE_MINI_NODES @ R.T
    proj = [(int(nx * icon_r + ox), int(ny * icon_r + oy)) for nx, ny, _ in rotated]
    line_color = (200, 140, 255) if hot else (130, 80, 180)
    node_color = (255, 80, 255)  if hot else (180, 60, 200)
    for a, b in _SPHERE_MINI_EDGES:
        cv2.line(img, proj[a], proj[b], line_color, 1, cv2.LINE_AA)
    for px, py in proj:
        cv2.circle(img, (px, py), 1, node_color, -1, cv2.LINE_AA)


def _draw_pyramid_icon(img, ox, oy, r, color, angle_y, angle_x, hot):
    """Dessine un wireframe pyramide miniature centré en (ox, oy) dans un rayon r."""
    icon_r = r * 0.65
    R = _rot_matrices(angle_x, angle_y)
    rotated = _PYRAMID_MINI_NODES @ R.T
    proj = [(int(nx * icon_r + ox), int(ny * icon_r + oy)) for nx, ny, _ in rotated]
    line_color = (120, 200, 255) if hot else (60, 130, 200)
    node_color = (200, 220, 255) if hot else (140, 170, 220)
    for a, b in _PYRAMID_MINI_EDGES:
        cv2.line(img, proj[a], proj[b], node_color, 2, cv2.LINE_AA)
    for px, py in proj:
        cv2.circle(img, (px, py), 3 if hot else 2, node_color, -1, cv2.LINE_AA)


def _draw_mouse_icon(img, ox, oy, r, color, hot):
    """Icône souris stylisée (contour + scroll wheel)."""
    w = int(r * 0.55)
    h = int(r * 0.80)
    x0, y0 = ox - w, oy - h // 2
    c = color if hot else (150, 150, 150)
    cv2.rectangle(img, (x0, y0), (x0 + w * 2, y0 + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w * 2, y0 + h), c, 1, cv2.LINE_AA)
    cv2.line(img, (ox, y0), (ox, y0 + h // 2), c, 1, cv2.LINE_AA)
    wh, ww = int(h * 0.18), int(w * 0.22)
    cv2.rectangle(img, (ox - ww, y0 + int(h * 0.12)),
                  (ox + ww, y0 + int(h * 0.12) + wh), c, -1)


def _draw_quit_icon(img, ox, oy, r, color, hot):
    """Icône X de fermeture."""
    c = color if hot else (130, 130, 160)
    sz = int(r * 0.42)
    th = 2 if hot else 1
    cv2.line(img, (ox - sz, oy - sz), (ox + sz, oy + sz), c, th, cv2.LINE_AA)
    cv2.line(img, (ox + sz, oy - sz), (ox - sz, oy + sz), c, th, cv2.LINE_AA)


def _draw_shapes_icon(img, ox, oy, r, color, hot):
    """Icône 'Shapes' : petit cube wireframe."""
    c = color if hot else (130, 100, 180)
    s = int(r * 0.38)
    o = int(s * 0.45)
    th = 1
    cv2.rectangle(img, (ox - s, oy - s), (ox + s, oy + s), c, th, cv2.LINE_AA)
    cv2.rectangle(img, (ox - s + o, oy - s - o), (ox + s + o, oy + s - o), c, th, cv2.LINE_AA)
    for dx, dy in [(-s, -s), (s, -s), (s, s), (-s, s)]:
        cv2.line(img, (ox + dx, oy + dy), (ox + dx + o, oy + dy - o), c, th, cv2.LINE_AA)


def _draw_glob_icon(img, ox, oy, r, color, hot):
    """Icône Glob : deux cercles qui se fondent (metaball stylisé)."""
    c = color if hot else (100, 160, 140)
    th = 1
    ro = int(r * 0.36)
    off = int(r * 0.22)
    cv2.circle(img, (ox - off, oy), ro, c, th, cv2.LINE_AA)
    cv2.circle(img, (ox + off, oy), ro, c, th, cv2.LINE_AA)
    if hot:
        cv2.circle(img, (ox - off - ro // 3, oy - ro // 3), max(1, ro // 4),
                   (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (ox + off - ro // 3, oy - ro // 3), max(1, ro // 4),
                   (255, 255, 255), -1, cv2.LINE_AA)


# Table d'icônes du menu principal : label → fonction de dessin
_MAIN_ICON_DRAWERS = {
    "Shapes": _draw_shapes_icon,
    "Mouse":  _draw_mouse_icon,
    "Glob":   _draw_glob_icon,
    "Quit":   _draw_quit_icon,
}

# Table d'icônes du menu shapes
_SHAPE_ICON_DRAWERS = {
    "Sphere":  None,  # spécial : wireframe 3D animé
    "Pyramid": None,  # spécial : wireframe 3D animé
    "Cancel":  _draw_quit_icon,
}

SHAPE_OPTIONS = [
    ("Sphere",  90,  SHAPES['sphere_color'],  MatrixSphere3D,  80),
    ("Pyramid", 270, SHAPES['pyramid_color'], MatrixPyramid3D, SHAPES['pyramid_radius_initial']),
    ("Cancel",    0, (100, 100, 100), None, 0),
]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RADIAL MENU RENDERER — batched overlay (1 img.copy per menu)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Angles animés des icônes 3D dans le menu (mis à jour dans la boucle principale)
_menu_sphere_ay = 0.0
_menu_sphere_ax = 0.0
_menu_pyramid_ay = 0.0
_menu_pyramid_ax = 0.0


def _draw_dynamic_menu(img, cx, cy, hovered_idx, carousel_theta,
                       options_list, icon_drawers=None):
    """
    Infinite 3D Wheel Carousel.
    The carousel rotates based on carousel_theta.
    Items closest to the 'front' (theta ≈ 270) are largest.
    """
    global _menu_sphere_ay, _menu_sphere_ax, _menu_pyramid_ay, _menu_pyramid_ax

    n_opts = len(options_list)
    if n_opts == 0: return

    # Rendu 3D settings
    RADIUS_X = MENU_RADIUS * 1.2
    RADIUS_Z = MENU_RADIUS * 0.8  # profondeur
    ITEM_MIN_SCALE = 0.5
    FRONT_ANGLE = 270 # Position sélectionnée (bas centre)

    rendered = []
    for i, opt in enumerate(options_list):
        label, base_angle, color = opt[0], opt[1], opt[2]
        
        # Current angle on the wheel
        angle_deg = (base_angle + carousel_theta) % 360
        rad = math.radians(angle_deg)
        
        # 3D projection: items follow an ellipse in X-Z, projected to X-Y
        # Z goes from -1 (back) to 1 (front)
        z = -math.sin(rad) 
        x = math.cos(rad)
        
        # screen pos
        ox = int(cx + x * RADIUS_X)
        oy = int(cy + z * (RADIUS_Z * 0.4)) # slight tilt
        
        # depth-based scale and brightness
        # depth_t: 0 (back) to 1 (front)
        depth_t = (z + 1) / 2.0
        scale = ITEM_MIN_SCALE + depth_t * (1.0 - ITEM_MIN_SCALE)
        
        r = int(MENU_ICON_RADIUS * scale)
        hot = (i == hovered_idx)

        rendered.append({
            'i': i, 'label': label, 'color': color,
            'ox': ox, 'oy': oy, 'r': r,
            'scale': scale, 'z': z, 'hot': hot,
            'depth_t': depth_t
        })

    # Sort back-to-front
    rendered.sort(key=lambda item: item['z'])

    ov = img.copy()
    for item in rendered:
        label, color, ox, oy, r, scale, z, hot, depth_t = \
            item['label'], item['color'], item['ox'], item['oy'], item['r'], \
            item['scale'], item['z'], item['hot'], item['depth_t']
            
        brightness = 0.3 + 0.7 * depth_t
        c_dim = (int(color[0]*brightness), int(color[1]*brightness), int(color[2]*brightness))
        
        # Glow for the front-most item
        if hot:
            for glow_r in (r + 15, r + 8, r + 4):
                cv2.circle(ov, (ox, oy), int(glow_r), color, -1)

        # Bubble
        cv2.circle(ov, (ox, oy), r, color if hot else c_dim, -1)
        
        # Outline
        border_val = int(200 * brightness)
        cv2.circle(img, (ox, oy), r, (border_val, border_val, border_val), 1, cv2.LINE_AA)

        # Icon
        if icon_drawers and label in icon_drawers:
            drawer = icon_drawers[label]
            if drawer is None:
                if label == "Sphere":
                    _draw_sphere_icon(img, ox, oy, r, color, _menu_sphere_ay, _menu_sphere_ax, hot)
                elif label == "Pyramid":
                    _draw_pyramid_icon(img, ox, oy, r, color, _menu_pyramid_ay, _menu_pyramid_ax, hot)
            else:
                drawer(img, ox, oy, r, color, hot)
        else:
            main_drawer = _MAIN_ICON_DRAWERS.get(label)
            if main_drawer:
                main_drawer(img, ox, oy, r, color, hot)

        # Text label
        if depth_t > 0.4:
            font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale
            (tw, th), _ = cv2.getTextSize(label, font, sc, 1)
            lx, ly = ox - tw // 2, oy + r + 20
            cv2.putText(img, label, (lx, ly), font, sc, (255, 255, 255) if hot else (150, 150, 150), 1, cv2.LINE_AA)

    cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)


def _draw_fps(img, fps, mode, hand_count, shapes_count=0):
    """HUD: FPS, current mode, hand count, shape count."""
    h, w = img.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.32
    pad, line_h = 8, 14

    # Background panel (semi-transparent dark)
    panel_w, panel_h = 180, line_h * 4 + pad * 3
    cv2.rectangle(img, (0, 0), (panel_w, panel_h), (10, 10, 10), -1)
    cv2.rectangle(img, (0, 0), (panel_w, panel_h), (60, 60, 60), 1, cv2.LINE_AA)

    # FPS (green if good, yellow if okay, red if low)
    fps_color = (0, 255, 0) if fps >= 30 else (255, 255, 0) if fps >= 15 else (0, 100, 255)
    cv2.putText(img, f"fps  {fps:.0f}", (pad, pad + line_h), font, scale, fps_color, 1, cv2.LINE_AA)

    # Mode indicator
    mode_color = (180, 180, 255) if mode == "menu" else (255, 200, 100) if mode == "shapes" else (100, 200, 255) if mode == "mouse" else (100, 255, 200)
    cv2.putText(img, f"mode {mode}", (pad, pad + line_h * 2), font, scale, mode_color, 1, cv2.LINE_AA)

    # Hands / shapes
    cv2.putText(img, f"hands {hand_count}", (pad, pad + line_h * 3), font, scale, (180, 180, 180), 1, cv2.LINE_AA)
    if shapes_count > 0:
        cv2.putText(img, f"shapes {shapes_count}", (pad, pad + line_h * 4), font, scale, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_gesture_hints(img, poing, pincement, double_fist_progress, spawn_cooldown_remaining):
    """Show gesture feedback in bottom-center."""
    h, w = img.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.30
    y_base = h - 40

    hints = []
    if poing:
        hints.append(("FIST", (180, 180, 255)))
    if pincement:
        hints.append(("PINCH", (100, 255, 150)))
    if double_fist_progress > 0:
        hints.append((f"DUAL {int(double_fist_progress*100)}%", (255, 200, 100)))
    if spawn_cooldown_remaining > 0:
        hints.append((f"CD {spawn_cooldown_remaining:.0f}", (150, 150, 150)))

    if not hints:
        return

    # Center the hints
    total_w = sum(cv2.getTextSize(h[0], font, scale, 1)[0][0] + 8 for h in hints) + 16
    x_start = max(8, (w - total_w) // 2)

    for label, color in hints:
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.putText(img, label, (x_start, y_base), font, scale, color, 1, cv2.LINE_AA)
        x_start += tw + 16


def _draw_hint(img):
    h, w = img.shape[:2]
    hint = "double-fist → open menu  |  point wrist  |  pinch to select"
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.36
    (tw, th), _ = cv2.getTextSize(hint, font, scale, 1)
    lx, ly = w // 2 - tw // 2, h - 14
    cv2.rectangle(img, (lx - 6, ly - th - 4), (lx + tw + 6, ly + 4), (12, 12, 12), -1)
    cv2.putText(img, hint, (lx, ly), font, scale, (110, 110, 110), 1, cv2.LINE_AA)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CAMERA / DETECTION THREADS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CameraStream:
    """Background thread that keeps one fresh (flipped) camera frame."""

    def __init__(self, src, width, height):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Request a higher frame rate for finer temporal sampling (ignored if
        # the camera can't do it).
        self.cap.set(cv2.CAP_PROP_FPS, CAM.get('camera_fps', 60))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM['buffer_size'])
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                f = cv2.flip(f, 1)
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class DetectionThread:
    """Background thread running MediaPipe on the small frame.
    Uses a threading.Event instead of busy-poll sleep."""

    def __init__(self):
        self._in = None
        self._in_lock = threading.Lock()
        self._out = None
        self._data = []
        self._out_lock = threading.Lock()
        self._event = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        while self.running:
            self._event.wait(timeout=0.05)
            self._event.clear()
            with self._in_lock:
                f = self._in
            if f is None:
                continue
            f, data = hand_detect(f)
            with self._out_lock:
                self._out = f
                self._data = data

    def submit(self, frame):
        with self._in_lock:
            self._in = frame.copy()
        self._event.set()

    def get(self):
        with self._out_lock:
            return (self._out.copy() if self._out is not None else None,
                    list(self._data))

    def stop(self):
        self.running = False
        self._event.set()
        self.thread.join(timeout=1.0)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS & PHYSICS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


LISSAGE    = PHYSICS['lissage']
FRICTION   = PHYSICS['friction']
V_MIN      = PHYSICS['vitesse_min']
ROT_SENS   = PHYSICS['rotation_sensibilite']
R_MIN      = PHYSICS['radius_min']
R_MAX      = PHYSICS['radius_max']


def _update_shape_physics(shape):
    """Move/rotate/resize a shape toward its pinch targets, with inertia."""
    shape.is_rotating_manually = False

    if len(shape.pinches) == 1:
        px, py, is_fist = shape.pinches[0]
        if not shape.is_grabbed_single:
            shape.is_grabbed_single = True
            shape.is_two_handed = False
            shape.drag_offset = (shape.x - px, shape.y - py)
            shape.last_pinch_pos = (px, py)

        if is_fist:
            # Rotation manuelle au poing
            shape.is_rotating_manually = True
            dx = px - shape.last_pinch_pos[0]
            dy = py - shape.last_pinch_pos[1]
            shape.angle_y += dx * ROT_SENS
            shape.angle_x -= dy * ROT_SENS
            shape.drag_offset = (shape.x - px, shape.y - py)
            shape.vx = shape.vy = 0
        else:
            # Drag & drop lissé
            tx = px + shape.drag_offset[0]
            ty = py + shape.drag_offset[1]
            shape.vx = (tx - shape.x) / LISSAGE
            shape.vy = (ty - shape.y) / LISSAGE
            shape.x += shape.vx
            shape.y += shape.vy
        shape.last_pinch_pos = (px, py)

    elif len(shape.pinches) == 2:
        shape.is_grabbed_single = False
        (p1x, p1y, _), (p2x, p2y, _) = shape.pinches[0], shape.pinches[1]
        ccx, ccy = (p1x + p2x) // 2, (p1y + p2y) // 2
        cpd = _dist((p1x, p1y), (p2x, p2y))

        if not shape.is_two_handed:
            shape.is_two_handed = True
            shape.initial_pinch_dist = cpd
            shape.initial_radius = shape.radius
            shape.two_hand_offset = (shape.x - ccx, shape.y - ccy)
        else:
            if shape.initial_pinch_dist > 0:
                sf = cpd / shape.initial_pinch_dist
                nr = max(R_MIN, min(R_MAX, int(shape.initial_radius * sf)))
                shape.radius += (nr - shape.radius) / LISSAGE

        tx = ccx + shape.two_hand_offset[0]
        ty = ccy + shape.two_hand_offset[1]
        shape.vx = (tx - shape.x) / LISSAGE
        shape.vy = (ty - shape.y) / LISSAGE
        shape.x += shape.vx
        shape.y += shape.vy

    else:
        shape.is_grabbed_single = shape.is_two_handed = False
        shape.x += shape.vx
        shape.y += shape.vy
        shape.vx *= FRICTION
        shape.vy *= FRICTION
        if abs(shape.vx) < V_MIN:
            shape.vx = 0
        if abs(shape.vy) < V_MIN:
            shape.vy = 0


def _get_front_option_idx(carousel_theta, options_list):
    """
    Returns the index of the option currently closest to the front (270 degrees).
    """
    FRONT_ANGLE = 270
    best_i, min_diff = -1, float('inf')
    for i, opt in enumerate(options_list):
        angle_deg = (opt[1] + carousel_theta) % 360
        diff = min(abs(angle_deg - FRONT_ANGLE), 360 - abs(angle_deg - FRONT_ANGLE))
        if diff < min_diff:
            min_diff = diff
            best_i = i
    return best_i

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

TRIPLE_WINDOW = SHAPES['triple_clic_fenetre']
GRAB_RANGE    = SHAPES['grab_range_mult']
SPAWN_DIST    = SHAPES['spawn_min_dist_mult']
MAX_SHAPES    = SHAPES['max_shapes']


def main():
    global _menu_sphere_ay, _menu_sphere_ax, _menu_pyramid_ay, _menu_pyramid_ax
    global DETECT_W, DETECT_H

    # ── Settings GUI (--settings) ─────────────────────────────────────────────
    # Standalone configurator: pick camera, frame rate, display + detection
    # resolution, smoothing. Saves to the settings file that config.py reads
    # on every launch, then exits — run main.py normally to use the settings.
    if "--settings" in sys.argv:
        import settings_gui
        settings_gui.open_settings()
        sys.exit(0)

    # ── Startup calibration screen ──────────────────────────────────────────────
    import startup as _startup
    if not _startup.run_startup():
        sys.exit(0)

    # ── Auto detection resolution (runs AFTER user sees camera feed) ─────────────
    if CAM.get('detect_auto', False):
        base_wh = (CAM['detect_w'], CAM['detect_h'])
        hi_wh   = (CAM['detect_hi_w'], CAM['detect_hi_h'])
        budget  = CAM['detect_latency_budget_ms']
        DETECT_W, DETECT_H, _ms, label = pick_detect_resolution(
            CAM['source'], base_wh, hi_wh, budget)
        print(f"Detection resolution: {DETECT_W}x{DETECT_H} ({label})")
    else:
        DETECT_W, DETECT_H = CAM['detect_w'], CAM['detect_h']
        print(f"Detection resolution: {DETECT_W}x{DETECT_H} (fixed)")

    # ── Hand calibration ────────────────────────────────────────────────────────
    # (removed — now done in startup.py GUI)

    cam = CameraStream(CAM['source'], DISPLAY_W, DISPLAY_H)
    det = DetectionThread()

    # ── Menu principal ───────────────────────────────────────────────────────────
    menu_phase = 0
    menu_cx = DISPLAY_W // 2
    menu_cy = DISPLAY_H // 2
    hovered_idx = -1
    action = None
    menu_rotation = 0.0
    menu_rotation_frozen = 0.0   # rotation figée pendant que le menu est ouvert
    menu_cursor_x = None          # wrist position used as the pointing cursor
    menu_cursor_y = None
    menu_double_fist_ctr = [0]    # double-fist counter for opening the menu
    menu_menu_state = 'idle'      # idle | open | close_requested

    # ── Mode shapes ──────────────────────────────────────────────────────────────
    shapes = []
    shape_menu_phase = 0
    shape_menu_cx = 0
    shape_menu_cy = 0
    shape_hovered_idx = -1
    shape_menu_rotation = 0.0
    shape_menu_rotation_frozen = 0.0
    shapes_active = False
    double_fist_counter = [0]
    last_touched_shape = None  # ref to most recently interacted shape
    shape_menu_cursor_x = None
    shape_menu_cursor_y = None
    shape_menu_state = 'idle'       # idle | open | close_requested

    SX = DISPLAY_W / DETECT_W
    SY = DISPLAY_H / DETECT_H

    print("Ready — close your fist to open the radial menu.")
    while cam.read() is None:
        time.sleep(0.01)

    # ── FPS tracking ────────────────────────────────────────────────────────────
    fps_frames = 0
    fps_time = time.monotonic()
    current_fps = 0
    spawn_cooldown_frames = 0  # visual cooldown for shape spawn

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            det.submit(cv2.resize(frame, (DETECT_W, DETECT_H)))
            res_frame, hand_data = det.get()

            display = cv2.resize(res_frame, (DISPLAY_W, DISPLAY_H)) \
                      if res_frame is not None else frame

            now = time.time()

            # ── FPS calculation ─────────────────────────────────────────────────
            fps_frames += 1
            fps_now = time.monotonic()
            if fps_now - fps_time >= 0.5:
                current_fps = fps_frames / (fps_now - fps_time)
                fps_frames = 0
                fps_time = fps_now

            # ── Spawn cooldown decrement ────────────────────────────────────────
            if spawn_cooldown_frames > 0:
                spawn_cooldown_frames -= 1

            # ── Animation des icônes 3D dans les menus ───────────────────────────
            _menu_sphere_ay = (_menu_sphere_ay + 0.04) % (2 * math.pi)
            _menu_sphere_ax = (_menu_sphere_ax + 0.02) % (2 * math.pi)
            _menu_pyramid_ay = (_menu_pyramid_ay + 0.05) % (2 * math.pi)
            _menu_pyramid_ax = (_menu_pyramid_ax + 0.02) % (2 * math.pi)

            # ── Rotation continue du menu (fond) ─────────────────────────────────
            # Le menu principal et le sous-menu shapes tournent quand ils sont
            # fermés, et la rotation est GELÉE dès qu'ils s'ouvrent (FIX).
            if menu_phase != 1:
                menu_rotation = (menu_rotation + 0.4) % 360
            if shape_menu_phase != 1:
                shape_menu_rotation = (shape_menu_rotation + 0.4) % 360

            # ── FSM Menu principal (3-state: idle → open → close_requested) ─────
            both_fists_now = (hand_data is not None
                              and len(hand_data) >= 2
                              and hand_data[0]['gestes']['poing']
                              and hand_data[1]['gestes']['poing'])

            if menu_phase == 0:
                if menu_menu_state == 'idle':
                    if both_fists_now:
                        menu_double_fist_ctr[0] += 1
                        if menu_double_fist_ctr[0] >= GESTURE['double_fist_frames']:
                            menu_cx, menu_cy = DISPLAY_W // 2, DISPLAY_H // 2
                            menu_phase = 1
                            menu_menu_state = 'open'
                            hovered_idx = -1
                            menu_cursor_x = menu_cursor_y = None
                            menu_rotation_frozen = menu_rotation
                            menu_double_fist_ctr[0] = 0
                    else:
                        menu_double_fist_ctr[0] = 0
                        _draw_hint(display)
                        gest.draw_double_fist_hint(display, menu_double_fist_ctr)

                elif menu_menu_state == 'close_requested':
                    if both_fists_now:
                        menu_double_fist_ctr[0] += 1
                        if menu_double_fist_ctr[0] >= GESTURE['double_fist_frames']:
                            menu_phase = 0
                            menu_menu_state = 'idle'
                            menu_cursor_x = menu_cursor_y = None
                            menu_double_fist_ctr[0] = 0
                    else:
                        # Released → go back to open (user changed mind)
                        menu_menu_state = 'open'
                        menu_double_fist_ctr[0] = 0

            elif menu_phase == 1:
                # ── Menu open: wrist = carousel rotator, pinch = confirm ───────────────
                if hand_data:
                    pincement = hand_data[0]['gestes']['pincement']
                    pts = [(int(x * SX), int(y * SY)) for x, y in hand_data[0]['points']]
                    
                    # Carousel rotation controlled by wrist X position
                    wrist_x = pts[0][0]
                    rot_delta = (wrist_x - DISPLAY_W/2) / (DISPLAY_W/2) * MENU['carousel_sensitivity']
                    menu_rotation = (menu_rotation + rot_delta) % 360
                    
                    hovered_idx = _get_front_option_idx(menu_rotation, MENU_OPTIONS)

                    _draw_dynamic_menu(display, menu_cx, menu_cy,
                                       hovered_idx, menu_rotation,
                                       MENU_OPTIONS)

                    if pincement:
                        # Confirm selection
                        if hovered_idx != -1:
                            action = MENU_OPTIONS[hovered_idx][0]
                        menu_phase = 0
                        menu_menu_state = 'idle'
                        menu_cursor_x = menu_cursor_y = None
                    elif not both_fists_now:
                        # Released fists → enter close_requested (ignore held fists)
                        menu_menu_state = 'close_requested'
                        menu_double_fist_ctr[0] = 0
                else:
                    # No hand → assume fists released
                    if menu_menu_state == 'open':
                        menu_menu_state = 'close_requested'
                        menu_double_fist_ctr[0] = 0

            # ── Action dispatch ──────────────────────────────────────────────────
            if action == "Quit":
                break

            elif action == "Mouse":
                action = None
                cv2.destroyAllWindows()
                virt_mouse.run_mouse_mode(cam, det)
                menu_phase = 0
                hovered_idx = -1

            elif action == "Glob":
                action = None
                cv2.destroyAllWindows()
                glob_metal.run_glob_mode(cam, det)
                menu_phase = 0
                hovered_idx = -1

            elif action == "Shapes":
                action = None
                shapes_active = True
                menu_phase = -1

            # ── Shapes mode ──────────────────────────────────────────────────────
            if shapes_active and hand_data:
                # ── Two-hand scissor spread → last-touched shape rotation speed ──
                spread_active, speed_mult, mid0, mid1 = \
                    gest.read_two_hand_spread(hand_data, SX, SY)

                if spread_active and last_touched_shape is not None \
                        and last_touched_shape in shapes:
                    last_touched_shape.rotation_speed_mult = speed_mult

                gest.draw_spread_indicator(display, mid0, mid1, speed_mult, spread_active)

                # 1. Récupération des pincements et des poings
                active_pinches = []
                fists = []
                for hi, hand in enumerate(hand_data):
                    pts = hand['points']
                    tx = int(pts[4][0] * SX);  ty = int(pts[4][1] * SY)
                    ix = int(pts[8][0] * SX);  iy = int(pts[8][1] * SY)
                    ccx = (tx + ix) // 2;      ccy = (ty + iy) // 2
                    is_fist = hand['gestes']['poing']
                    is_pinch = hand['gestes']['pincement']
                    if is_pinch or is_fist:
                        active_pinches.append((ccx, ccy, is_fist, hand['id']))
                        cv2.circle(display, (ccx, ccy), 8,
                                   (0, 0, 255) if is_fist else (0, 255, 0), -1)
                    if is_fist:
                        fists.append((ccx, ccy, hand['id']))

                # 2. Gestion de l'état "saisie" des formes
                hand_by_id = {hand['id']: hand for hand in hand_data}
                for shape in shapes:
                    shape.pinches = []
                    if shape.locked_hand_id is not None:
                        lid = shape.locked_hand_id
                        hand = hand_by_id.get(lid)
                        if hand is not None and hand['gestes']['poing']:
                            pts = hand['points']
                            tx = int(pts[4][0] * SX);  ty = int(pts[4][1] * SY)
                            ix = int(pts[8][0] * SX);  iy = int(pts[8][1] * SY)
                            ccx = (tx + ix) // 2;      ccy = (ty + iy) // 2
                            shape.pinches.append((ccx, ccy, True))
                        else:
                            shape.locked_hand_id = None

                for (px, py, is_fist, hid) in active_pinches:
                    for shape in shapes:
                        if shape.locked_hand_id == hid:
                            continue
                        if _dist((px, py), (shape.x, shape.y)) < shape.radius * GRAB_RANGE:
                            shape.pinches.append((px, py, is_fist))
                            if is_fist:
                                shape.locked_hand_id = hid
                            last_touched_shape = shape  # track for spread gesture
                            break

                # 3. Menu radial shapes (invisible carousel: double-fist → pinch to select)
                grabbing_something = any(s.pinches for s in shapes)
                both_fists_now = (hand_data is not None
                                  and len(hand_data) >= 2
                                  and hand_data[0]['gestes']['poing']
                                  and hand_data[1]['gestes']['poing'])

                if shape_menu_phase == 0:
                    # ── Double-fist → open shapes spawn menu OR close shapes mode ─────
                    if not grabbing_something and len(shapes) < MAX_SHAPES:
                        if shape_menu_state == 'idle':
                            if both_fists_now:
                                double_fist_counter[0] += 1
                                if double_fist_counter[0] >= GESTURE['double_fist_frames']:
                                    shape_menu_cx, shape_menu_cy = DISPLAY_W // 2, DISPLAY_H // 2
                                    shape_menu_phase = 1
                                    shape_menu_state = 'open'
                                    shape_hovered_idx = -1
                                    shape_menu_cursor_x = shape_menu_cursor_y = None
                                    shape_menu_rotation_frozen = shape_menu_rotation
                                    double_fist_counter[0] = 0
                            else:
                                double_fist_counter[0] = 0
                                gest.draw_double_fist_hint(display, double_fist_counter)
                        elif shape_menu_state == 'close_requested':
                            if both_fists_now:
                                double_fist_counter[0] += 1
                                if double_fist_counter[0] >= GESTURE['double_fist_frames']:
                                    shapes_active = False
                                    menu_phase = 0
                                    shape_menu_phase = 0
                                    shape_menu_state = 'idle'
                                    shape_hovered_idx = -1
                                    last_touched_shape = None
                                    double_fist_counter[0] = 0
                            else:
                                # Released → go back to open
                                shape_menu_state = 'open'
                                double_fist_counter[0] = 0
                    else:
                        double_fist_counter[0] = 0

                elif shape_menu_phase == 1:
                    if hand_data:
                        pincement = hand_data[0]['gestes']['pincement']
                        pts = [(int(x * SX), int(y * SY)) for x, y in hand_data[0]['points']]
                        
                        # Carousel rotation
                        wrist_x = pts[0][0]
                        rot_delta = (wrist_x - DISPLAY_W/2) / (DISPLAY_W/2) * MENU['carousel_sensitivity']
                        shape_menu_rotation = (shape_menu_rotation + rot_delta) % 360
                        
                        shape_hovered_idx = _get_front_option_idx(shape_menu_rotation, SHAPE_OPTIONS)

                        _draw_dynamic_menu(
                            display,
                            shape_menu_cx, shape_menu_cy,
                            shape_hovered_idx, shape_menu_rotation,
                            SHAPE_OPTIONS,
                            icon_drawers=_SHAPE_ICON_DRAWERS)

                        # Pinch to confirm selection
                        if pincement:
                            if shape_hovered_idx != -1:
                                label, _, _, ShapeClass, default_radius = \
                                    SHAPE_OPTIONS[shape_hovered_idx]
                                if ShapeClass is not None:
                                    shapes.append(ShapeClass(
                                        shape_menu_cx, shape_menu_cy, default_radius))
                                    spawn_cooldown_frames = 30
                            shape_menu_phase = 0
                            shape_menu_state = 'idle'
                            shape_hovered_idx = -1
                        elif not both_fists_now:
                            # Released fists → enter close_requested (ignore held fists)
                            shape_menu_state = 'close_requested'
                            double_fist_counter[0] = 0
                    else:
                        if shape_menu_state == 'open':
                            shape_menu_state = 'close_requested'
                            double_fist_counter[0] = 0

                # 4. Suppression par triple pincement
                for shape in shapes:
                    pinched = len(shape.pinches) > 0
                    if pinched and not shape.was_pinched_last_frame:
                        shape.pinch_count = (shape.pinch_count + 1
                                             if now - shape.last_pinch_time < TRIPLE_WINDOW
                                             else 1)
                        shape.last_pinch_time = now
                        if shape.pinch_count >= 3:
                            shape.to_delete = True
                    shape.was_pinched_last_frame = pinched
                shapes = [s for s in shapes if not s.to_delete]

                # 5. Physique et dessin
                for shape in shapes:
                    _update_shape_physics(shape)
                    shape.draw(display)

            elif shapes_active:
                # Pas de main détectée mais shapes_active : on dessine quand même
                for shape in shapes:
                    shape.pinches = []
                    _update_shape_physics(shape)
                    shape.draw(display)

            # ── HUD overlay ─────────────────────────────────────────────────────
            _draw_fps(display, current_fps,
                      "menu" if menu_phase >= 0 else ("shapes" if shapes_active else "?"),
                      len(hand_data) if hand_data else 0,
                      len(shapes) if shapes_active else 0)
            if hand_data:
                _draw_gesture_hints(display,
                                    hand_data[0]['gestes']['poing'],
                                    hand_data[0]['gestes']['pincement'],
                                    double_fist_counter[0] / GESTURE['double_fist_frames'] if hand_data else 0,
                                    spawn_cooldown_frames)
            elif shapes_active:
                _draw_gesture_hints(display, False, False, 0, spawn_cooldown_frames)

            cv2.imshow("Vision AI", display)
            if cv2.waitKey(1) == 27:
                break

    finally:
        print("Cleaning up threads and camera...")
        det.stop()
        cam.stop()
        release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if not setup_check.check():
        sys.exit(1)
    main()

# ── Pre-flight check (runs before main loop) ──────────────────────────────────
import setup_check
if not setup_check.check():
    sys.exit(1)
