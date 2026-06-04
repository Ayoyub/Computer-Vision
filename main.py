import sys
import os
import cv2
import threading
import time
import math
import numpy as np
import virt_mouse

from hand_detect import hand_detect, release
from config import CAM

DETECT_W,  DETECT_H  = CAM['detect_w'],  CAM['detect_h']
DISPLAY_W, DISPLAY_H = CAM['display_w'], CAM['display_h']

# ── Radial menu constants ────────────────────────────────────────────────────────
MENU_RADIUS      = 130
MENU_ICON_RADIUS = 38
SEUIL_SELECT     = 48
FRAMES_CONFIRM   = 22

# (label, angle_deg, color_BGR)
MENU_OPTIONS = [
    ("Shapes",  90,  (180, 100, 255)),
    ("Mouse",  270,  (200, 200, 200)),
    ("Quit",     0,  (100, 120, 255)),
]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MINI-SHAPE ICON RENDERERS (dessinées dans les bulles du menu)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Angles pré-calculés pour les icônes (évite de les recalculer chaque frame)
_SPHERE_ICON_ANGLES_Y = 0.0
_SPHERE_ICON_ANGLES_X = 0.0
_PYRAMID_ICON_ANGLE_Y = 0.0
_PYRAMID_ICON_ANGLE_X = 0.0

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
_PYRAMID_MINI_NODES = np.array([(0,-1,0), (-0.866,0.5,-0.5), (0.866,0.5,-0.5), (0,0.5,1)], dtype=float)
_PYRAMID_MINI_EDGES = [(0,1),(0,2),(0,3),(1,2),(2,3),(3,1)]


def _draw_sphere_icon(img, ox, oy, r, color, angle_y, angle_x, hot):
    """Dessine un wireframe sphère miniature centré en (ox, oy) dans un rayon r."""
    icon_r = r * 0.60  # la sphère occupe 60% du rayon de la bulle
    cx_f, sx_f = math.cos(angle_x), math.sin(angle_x)
    cy_f, sy_f = math.cos(angle_y), math.sin(angle_y)
    Rx = np.array([[1,0,0],[0,cx_f,-sx_f],[0,sx_f,cx_f]])
    Ry = np.array([[cy_f,0,sy_f],[0,1,0],[-sy_f,0,cy_f]])
    R  = Ry @ Rx
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
    cx_f, sx_f = math.cos(angle_x), math.sin(angle_x)
    cy_f, sy_f = math.cos(angle_y), math.sin(angle_y)
    Rx = np.array([[1,0,0],[0,cx_f,-sx_f],[0,sx_f,cx_f]])
    Ry = np.array([[cy_f,0,sy_f],[0,1,0],[-sy_f,0,cy_f]])
    R  = Ry @ Rx
    rotated = _PYRAMID_MINI_NODES @ R.T
    proj = [(int(nx * icon_r + ox), int(ny * icon_r + oy)) for nx, ny, _ in rotated]
    line_color = (120, 200, 255) if hot else (60, 130, 200)
    node_color = (200, 220, 255) if hot else (140, 170, 220)
    for a, b in _PYRAMID_MINI_EDGES:
        cv2.line(img, proj[a], proj[b], line_color, 2, cv2.LINE_AA)
    for px, py in proj:
        cv2.circle(img, (px, py), 3 if hot else 2, node_color, -1, cv2.LINE_AA)


def _draw_mouse_icon(img, ox, oy, r, color, hot):
    """Icône souris stylisée (contour + scroll wheel)."""
    w  = int(r * 0.55)
    h  = int(r * 0.80)
    br = int(w * 0.5)
    x0, y0 = ox - w, oy - h // 2
    c = color if hot else (150, 150, 150)
    # Corps de la souris
    cv2.rectangle(img, (x0, y0), (x0 + w*2, y0 + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w*2, y0 + h), c, 1, cv2.LINE_AA)
    # Séparateur boutons
    cv2.line(img, (ox, y0), (ox, y0 + h//2), c, 1, cv2.LINE_AA)
    # Molette
    wh, ww = int(h * 0.18), int(w * 0.22)
    cv2.rectangle(img, (ox - ww, y0 + int(h*0.12)),
                  (ox + ww, y0 + int(h*0.12) + wh), c, -1)


def _draw_quit_icon(img, ox, oy, r, color, hot):
    """Icône X de fermeture."""
    c  = color if hot else (130, 130, 160)
    sz = int(r * 0.42)
    th = 2 if hot else 1
    cv2.line(img, (ox-sz, oy-sz), (ox+sz, oy+sz), c, th, cv2.LINE_AA)
    cv2.line(img, (ox+sz, oy-sz), (ox-sz, oy+sz), c, th, cv2.LINE_AA)


def _draw_shapes_icon(img, ox, oy, r, color, hot):
    """Icône 'Shapes' : petit cube wireframe."""
    c  = color if hot else (130, 100, 180)
    s  = int(r * 0.38)
    o  = int(s * 0.45)
    th = 1
    # Face avant
    cv2.rectangle(img, (ox-s, oy-s), (ox+s, oy+s), c, th, cv2.LINE_AA)
    # Décalage perspective
    cv2.rectangle(img, (ox-s+o, oy-s-o), (ox+s+o, oy+s-o), c, th, cv2.LINE_AA)
    # 4 arêtes de profondeur
    for dx, dy in [(-s,-s),(s,-s),(s,s),(-s,s)]:
        cv2.line(img, (ox+dx, oy+dy), (ox+dx+o, oy+dy-o), c, th, cv2.LINE_AA)


# Table d'icônes du menu principal : label → fonction de dessin
_MAIN_ICON_DRAWERS = {
    "Shapes": _draw_shapes_icon,
    "Mouse":  _draw_mouse_icon,
    "Quit":   _draw_quit_icon,
}

# Table d'icônes du menu shapes
_SHAPE_ICON_DRAWERS = {
    "Sphere":  None,  # spécial : wireframe 3D animé
    "Pyramid": None,  # spécial : wireframe 3D animé
    "Cancel":  _draw_quit_icon,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3D SHAPES                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class _Shape3D:
    def __init__(self, x, y, radius, color):
        self.x, self.y             = float(x), float(y)
        self.radius                = float(radius)
        self.color                 = color
        self.vx = self.vy          = 0.0
        self.angle_x = self.angle_y = 0.0
        self.pinches               = []
        self.locked_hand_id        = None
        self.is_two_handed         = False
        self.is_grabbed_single     = False
        self.is_rotating_manually  = False
        self.initial_radius        = float(radius)
        self.initial_pinch_dist    = 0
        self.two_hand_offset       = (0, 0)
        self.drag_offset           = (0, 0)
        self.last_pinch_pos        = (0, 0)
        self.pinch_count           = 0
        self.last_pinch_time       = 0.0
        self.was_pinched_last_frame = False
        self.to_delete             = False


class MatrixSphere3D(_Shape3D):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, (106, 50, 159))
        self.nodes, self.edges = [], []
        self._build(rings=10, segments=14)

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
        for i in range(rings):
            for j in range(segments):
                c = i * segments + j
                self.edges.append((c, i * segments + (j+1) % segments))
                self.edges.append((c, (i+1) * segments + j))

    def draw(self, img):
        if not self.locked_hand_id:
            self.angle_y += 0.003
            self.angle_x += 0.003
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        R  = Ry @ Rx
        rotated = self.nodes @ R.T
        proj = [(int(nx * self.radius + self.x), int(ny * self.radius + self.y))
                for nx, ny, _ in rotated]
        for a, b in self.edges:
            cv2.line(img, proj[a], proj[b], self.color, 1, cv2.LINE_AA)
        for px, py in proj:
            cv2.circle(img, (px, py), 2, (255, 0, 255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), (0,255,0), 1)


class MatrixPyramid3D(_Shape3D):
    EDGES = [(0,1),(0,2),(0,3),(1,2),(2,3),(3,1)]

    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, (0, 150, 255))
        # FIX: copie propre pour chaque instance (évite le partage de tableau)
        self.nodes = np.array([(0,-1,0), (-0.866,0.5,-0.5),
                                (0.866,0.5,-0.5), (0,0.5,1)], dtype=float)
        self.edges = self.EDGES

    def draw(self, img):
        if not self.locked_hand_id:
            self.angle_y += 0.04
            self.angle_x += 0.02
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        R  = Ry @ Rx
        rotated = self.nodes @ R.T
        proj = [(int(nx * self.radius + self.x), int(ny * self.radius + self.y))
                for nx, ny, _ in rotated]
        for a, b in self.edges:
            cv2.line(img, proj[a], proj[b], self.color, 2, cv2.LINE_AA)
        for px, py in proj:
            cv2.circle(img, (px, py), 4, (200,200,255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.color, 1)


SHAPE_OPTIONS = [
    ("Sphere",  90,  (106, 50, 159), MatrixSphere3D,  80),
    ("Pyramid", 270, (0, 150, 255),  MatrixPyramid3D, 80),
    ("Cancel",    0, (100, 100, 100), None, 0),
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RADIAL MENU RENDERER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Angles animés des icônes 3D dans le menu (mis à jour dans la boucle principale)
_menu_sphere_ay = 0.0
_menu_sphere_ax = 0.0
_menu_pyramid_ay = 0.0
_menu_pyramid_ax = 0.0


def _draw_dynamic_menu(img, cx, cy, hovered, cursor_x, cursor_y,
                       rotation_offset, options_list, icon_drawers=None):
    """
    Dessine le menu rotatif avec icônes personnalisées par option.
    icon_drawers : dict {label: callable} ou None pour les icônes standard.
    """
    global _menu_sphere_ay, _menu_sphere_ax, _menu_pyramid_ay, _menu_pyramid_ax

    # Centre du menu
    cv2.circle(img, (cx, cy), 15, (20, 20, 20), -1)
    cv2.circle(img, (cx, cy), 15, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    if cursor_x and cursor_y:
        # Un seul addWeighted pour la ligne de visée
        ov = img.copy()
        cv2.line(ov, (cx, cy), (cursor_x, cursor_y), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.4, img, 0.6, 0, img)
        cv2.circle(img, (cursor_x, cursor_y), 8, (255, 255, 255), -1, cv2.LINE_AA)

    for i, opt in enumerate(options_list):
        label, base_angle, color = opt[0], opt[1], opt[2]

        angle_deg = (base_angle + rotation_offset) % 360
        rad       = math.radians(angle_deg)
        ox  = int(cx + MENU_RADIUS * math.cos(rad))
        oy  = int(cy - MENU_RADIUS * math.sin(rad))
        hot = (i == hovered)
        r   = MENU_ICON_RADIUS + (8 if hot else 0)

        cv2.line(img, (cx, cy), (ox, oy), (80, 80, 80), 1, cv2.LINE_AA)

        # Glow (hot only) — 1 seul addWeighted par anneau
        if hot:
            for glow_r, alpha in [(r+12, 0.12), (r+6, 0.14), (r+3, 0.16)]:
                ov = img.copy()
                cv2.circle(ov, (ox, oy), glow_r, color, -1)
                cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

        # Fond de bulle (1 addWeighted)
        ov = img.copy()
        cv2.circle(ov, (ox, oy), r, color if hot else (40, 40, 40), -1)
        cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)

        # Contour
        cv2.circle(img, (ox, oy), r, color if hot else (120, 120, 120),
                   2 if hot else 1, cv2.LINE_AA)

        # ── Icône dans la bulle ──────────────────────────────────────────────────
        if icon_drawers and label in icon_drawers:
            drawer = icon_drawers[label]
            if drawer is None:
                # Icônes 3D animées (Sphere / Pyramid dans le menu shapes)
                if label == "Sphere":
                    _draw_sphere_icon(img, ox, oy, r, color,
                                      _menu_sphere_ay, _menu_sphere_ax, hot)
                elif label == "Pyramid":
                    _draw_pyramid_icon(img, ox, oy, r, color,
                                       _menu_pyramid_ay, _menu_pyramid_ax, hot)
            else:
                drawer(img, ox, oy, r, color, hot)
        else:
            # Menu principal : icônes statiques selon le label
            main_drawer = _MAIN_ICON_DRAWERS.get(label)
            if main_drawer:
                main_drawer(img, ox, oy, r, color, hot)

        # Label texte sous la bulle
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45 if hot else 0.40
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        lx, ly = ox - tw//2, oy + th//2 + r + 15

        ov = img.copy()
        cv2.rectangle(ov, (lx-6, ly-th-6), (lx+tw+6, ly+6), (10,10,10), -1)
        cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
        cv2.putText(img, label, (lx, ly), font, scale,
                    (255,255,255) if hot else (170,170,170), 1, cv2.LINE_AA)


def _draw_hint(img):
    h, w = img.shape[:2]
    hint = "close fist to open radial menu"
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.36
    (tw, th), _ = cv2.getTextSize(hint, font, scale, 1)
    lx, ly = w//2 - tw//2, h - 14
    ov = img.copy()
    cv2.rectangle(ov, (lx-6, ly-th-4), (lx+tw+6, ly+4), (12,12,12), -1)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    cv2.putText(img, hint, (lx, ly), font, scale, (110,110,110), 1, cv2.LINE_AA)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CAMERA / DETECTION THREADS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CameraStream:
    def __init__(self, src, width, height):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAM['buffer_size'])
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = cv2.flip(f, 1)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class DetectionThread:
    def __init__(self):
        self._in        = None
        self._in_lock   = threading.Lock()
        self._out       = None
        self._data      = []
        self._out_lock  = threading.Lock()
        self._event     = threading.Event()   # FIX: remplace busy-poll sleep(0.001)
        self.running    = True
        threading.Thread(target=self._work, daemon=True).start()

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
                self._out  = f
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS & PHYSICS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def _update_shape_physics(shape, LISSAGE=3.0):
    shape.is_rotating_manually = False

    if len(shape.pinches) == 1:
        px, py, is_fist = shape.pinches[0]
        if not shape.is_grabbed_single:
            shape.is_grabbed_single = True
            shape.is_two_handed     = False
            shape.drag_offset       = (shape.x - px, shape.y - py)
            shape.last_pinch_pos    = (px, py)

        if is_fist:
            shape.is_rotating_manually = True
            dx = px - shape.last_pinch_pos[0]
            dy = py - shape.last_pinch_pos[1]
            shape.angle_y += dx * 0.015
            shape.angle_x -= dy * 0.015
            shape.drag_offset = (shape.x - px, shape.y - py)
            shape.vx = shape.vy = 0
        else:
            tx = px + shape.drag_offset[0]
            ty = py + shape.drag_offset[1]
            shape.vx = (tx - shape.x) / LISSAGE
            shape.vy = (ty - shape.y) / LISSAGE
            shape.x += shape.vx
            shape.y += shape.vy
        shape.last_pinch_pos = (px, py)

    elif len(shape.pinches) == 2:
        shape.is_grabbed_single = False
        (p1x,p1y,_), (p2x,p2y,_) = shape.pinches[0], shape.pinches[1]
        ccx, ccy = (p1x+p2x)//2, (p1y+p2y)//2
        cpd = _dist((p1x,p1y),(p2x,p2y))

        if not shape.is_two_handed:
            shape.is_two_handed      = True
            shape.initial_pinch_dist = cpd
            shape.initial_radius     = shape.radius
            shape.two_hand_offset    = (shape.x - ccx, shape.y - ccy)
        else:
            if shape.initial_pinch_dist > 0:
                sf = cpd / shape.initial_pinch_dist
                nr = max(30, min(400, int(shape.initial_radius * sf)))
                shape.radius += (nr - shape.radius) / LISSAGE

        tx = ccx + shape.two_hand_offset[0]
        ty = ccy + shape.two_hand_offset[1]
        shape.vx = (tx - shape.x) / LISSAGE
        shape.vy = (ty - shape.y) / LISSAGE
        shape.x += shape.vx
        shape.y += shape.vy

    else:
        shape.is_grabbed_single = shape.is_two_handed = False
        shape.x  += shape.vx;  shape.y  += shape.vy
        shape.vx *= 0.85;      shape.vy *= 0.85
        if abs(shape.vx) < 0.1: shape.vx = 0
        if abs(shape.vy) < 0.1: shape.vy = 0


def _hovered_option_dynamic(x, y, cx, cy, rotation_offset, options_list):
    angle_rad = math.atan2(-(y - cy), x - cx)
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    best_i, min_diff = -1, float('inf')
    for i, opt in enumerate(options_list):
        opt_angle = (opt[1] + rotation_offset) % 360
        diff = min(abs(angle_deg - opt_angle), 360 - abs(angle_deg - opt_angle))
        if diff < min_diff:
            min_diff = diff
            best_i   = i
    return best_i if min_diff <= 45 else -1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    global _menu_sphere_ay, _menu_sphere_ax, _menu_pyramid_ay, _menu_pyramid_ax

    cam = CameraStream(CAM['source'], DISPLAY_W, DISPLAY_H)
    det = DetectionThread()

    # ── Menu principal ───────────────────────────────────────────────────────────
    menu_phase     = 0
    menu_cx        = DISPLAY_W // 2
    menu_cy        = DISPLAY_H // 2
    hovered_idx    = -1
    action         = None
    menu_rotation  = 0.0          # FIX: gelée à l'ouverture → voir ci-dessous


    # ── Mode shapes ──────────────────────────────────────────────────────────────
    shapes              = []
    shape_menu_phase    = 0
    shape_menu_cx       = 0
    shape_menu_cy       = 0
    shape_hovered_idx   = -1
    shape_menu_rotation = 0.0

    shapes_active       = False
    LISSAGE             = 3.0
    MAX_SHAPES          = 8       # FIX: limite pour éviter le ralentissement

    print("Ready — close your fist to open the radial menu.")
    while cam.read() is None:
        time.sleep(0.01)

    SX = DISPLAY_W / DETECT_W
    SY = DISPLAY_H / DETECT_H

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

            # ── Animation des icônes 3D dans les menus ───────────────────────────
            _menu_sphere_ay  = (_menu_sphere_ay  + 0.04) % (2*math.pi)
            _menu_sphere_ax  = (_menu_sphere_ax  + 0.02) % (2*math.pi)
            _menu_pyramid_ay = (_menu_pyramid_ay + 0.05) % (2*math.pi)
            _menu_pyramid_ax = (_menu_pyramid_ax + 0.02) % (2*math.pi)

            # ── Rotation continue du menu (fond) ─────────────────────────────────
            menu_rotation        = (menu_rotation        + 0.4) % 360
            shape_menu_rotation  = (shape_menu_rotation  + 0.4) % 360

            # ── FSM Menu principal ───────────────────────────────────────────────
            if menu_phase == 0:
                if hand_data and hand_data[0]['gestes']['poing']:
                    pts = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]
                    menu_cx, menu_cy = pts[0]
                    menu_phase = 1
                    hovered_idx = -1
                    # FIX: geler la rotation à l'instant d'ouverture
                    pass  # rotation non gelée
                _draw_hint(display)

            elif menu_phase == 1:
                if hand_data:
                    poing = hand_data[0]['gestes']['poing']
                    pts   = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]
                    wx, wy = pts[0]
                    dist = _dist((wx, wy), (menu_cx, menu_cy))

                    if dist > 40:
                        hovered_idx = _hovered_option_dynamic(
                            wx, wy, menu_cx, menu_cy,
                            menu_rotation, MENU_OPTIONS)
                    else:
                        hovered_idx = -1

                    _draw_dynamic_menu(display, menu_cx, menu_cy,
                                       hovered_idx, wx, wy,
                                       menu_rotation,
                                       MENU_OPTIONS)

                    if not poing:
                        if hovered_idx != -1:
                            action = MENU_OPTIONS[hovered_idx][0]
                        menu_phase = 0
                else:
                    menu_phase = 0

            # ── Action dispatch ──────────────────────────────────────────────────
            if action == "Quit":
                break

            elif action == "Mouse":
                action = None
                virt_mouse.run_mouse_mode(cam, det)
                menu_phase  = 0
                hovered_idx = -1
                # FIX: ne pas vider les shapes au retour du mode souris
                # (shapes.clear() supprimé)

            elif action == "Shapes":
                action        = None
                shapes_active = True
                menu_phase    = -1

            # ── Shapes mode ──────────────────────────────────────────────────────
            if shapes_active and hand_data:   # FIX: guard hand_data vide
                # 1. Récupération des pincements et des poings
                active_pinches = []
                fists = []
                for hi, hand in enumerate(hand_data):
                    pts     = hand['points']
                    tx  = int(pts[4][0]*SX);  ty = int(pts[4][1]*SY)
                    ix  = int(pts[8][0]*SX);  iy = int(pts[8][1]*SY)
                    ccx = (tx+ix)//2;         ccy = (ty+iy)//2
                    is_fist  = hand['gestes']['poing']
                    is_pinch = hand['gestes']['pincement']
                    if is_pinch or is_fist:
                        active_pinches.append((ccx, ccy, is_fist, hi))
                        cv2.circle(display, (ccx, ccy), 8,
                                   (0,0,255) if is_fist else (0,255,0), -1)
                    if is_fist:
                        fists.append((ccx, ccy, hi))

                # 2. Gestion de l'état "saisie" des formes
                for shape in shapes:
                    shape.pinches = []
                    if shape.locked_hand_id is not None:
                        lid = shape.locked_hand_id
                        # FIX: vérifie que l'index est encore valide
                        if lid < len(hand_data):
                            if hand_data[lid]['gestes']['poing']:
                                pts = hand_data[lid]['points']
                                tx  = int(pts[4][0]*SX);  ty = int(pts[4][1]*SY)
                                ix  = int(pts[8][0]*SX);  iy = int(pts[8][1]*SY)
                                ccx = (tx+ix)//2;         ccy = (ty+iy)//2
                                shape.pinches.append((ccx, ccy, True))
                            else:
                                shape.locked_hand_id = None
                        else:
                            shape.locked_hand_id = None

                for (px, py, is_fist, hi) in active_pinches:
                    for shape in shapes:
                        if shape.locked_hand_id == hi:
                            continue
                        if _dist((px,py),(shape.x,shape.y)) < shape.radius * 1.5:
                            shape.pinches.append((px, py, is_fist))
                            if is_fist:
                                shape.locked_hand_id = hi
                            break

                # 3. Menu radial shapes
                grabbing_something = any(s.pinches for s in shapes)

                if shape_menu_phase == 0:
                    if fists and not grabbing_something and len(shapes) < MAX_SHAPES:
                        fx, fy, _ = fists[0]
                        too_close = any(_dist((fx,fy),(s.x,s.y)) < s.radius * 2
                                        for s in shapes)
                        if not too_close:
                            shape_menu_cx, shape_menu_cy = fx, fy
                            shape_menu_phase = 1
                            shape_hovered_idx = -1
                            # FIX: geler la rotation à l'ouverture
                            pass  # rotation non gelée

                elif shape_menu_phase == 1:
                    if fists:
                        fx, fy, _ = fists[0]
                        dist = _dist((fx,fy),(shape_menu_cx, shape_menu_cy))
                        if dist > 40:
                            shape_hovered_idx = _hovered_option_dynamic(
                                fx, fy,
                                shape_menu_cx, shape_menu_cy,
                                shape_menu_rotation,
                                SHAPE_OPTIONS)
                        else:
                            shape_hovered_idx = -1

                        _draw_dynamic_menu(
                            display,
                            shape_menu_cx, shape_menu_cy,
                            shape_hovered_idx, fx, fy,
                            shape_menu_rotation,
                            SHAPE_OPTIONS,
                            icon_drawers=_SHAPE_ICON_DRAWERS) # icônes formes

                    else:
                        if shape_hovered_idx != -1:
                            label, _, _, ShapeClass, default_radius = \
                                SHAPE_OPTIONS[shape_hovered_idx]
                            if ShapeClass is not None:
                                shapes.append(ShapeClass(
                                    shape_menu_cx, shape_menu_cy, default_radius))
                        shape_menu_phase  = 0
                        shape_hovered_idx = -1

                # 4. Suppression par triple pincement
                for shape in shapes:
                    pinched = len(shape.pinches) > 0
                    if pinched and not shape.was_pinched_last_frame:
                        shape.pinch_count = (shape.pinch_count + 1
                                             if now - shape.last_pinch_time < 0.75
                                             else 1)
                        shape.last_pinch_time = now
                        if shape.pinch_count >= 3:
                            shape.to_delete = True
                    shape.was_pinched_last_frame = pinched
                shapes = [s for s in shapes if not s.to_delete]

                # 5. Physique et dessin
                for shape in shapes:
                    _update_shape_physics(shape, LISSAGE)
                    shape.draw(display)

            elif shapes_active:
                # Pas de main détectée mais shapes_active : on dessine quand même
                for shape in shapes:
                    shape.pinches = []
                    _update_shape_physics(shape, LISSAGE)
                    shape.draw(display)

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
    main()