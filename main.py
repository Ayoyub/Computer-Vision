import subprocess
import sys
import os
import cv2
import threading
import time
import math
import json
import numpy as np
from hand_detect import hand_detect, release, est_poing_ferme, est_pincement
from config import CAM

DETECT_W,  DETECT_H  = CAM['detect_w'],  CAM['detect_h']
DISPLAY_W, DISPLAY_H = CAM['display_w'], CAM['display_h']

# ── Load calibration if available, otherwise use fallback defaults ───────────────
_CALIB_FILE = 'calibration.json'
if os.path.exists(_CALIB_FILE):
    with open(_CALIB_FILE) as f:
        _calib = json.load(f)
    SEUIL_PAUME     = _calib['seuil_paume']
    SEUIL_POING_MENU = _calib['seuil_poing']
    print(f"[calibration] palm={SEUIL_PAUME:.0f}px  fist={SEUIL_POING_MENU:.0f}px")
else:
    SEUIL_PAUME      = 80   # fallback — run calibration.py for better accuracy
    SEUIL_POING_MENU = 55
    print("[calibration] no calibration.json found — using defaults")

# ── Radial menu constants ────────────────────────────────────────────────────────
MENU_RADIUS      = 130   # distance from center to each option bubble (display px)
MENU_ICON_RADIUS = 38    # radius of each option bubble
SEUIL_SELECT     = 48    # max distance index→option center to highlight it
FRAMES_CONFIRM   = 22    # consecutive frames hovering an option to confirm it

# (label, angle_deg, color_BGR)
MENU_OPTIONS = [
    ("Shapes",  90,  (180, 100, 255)),   # top
    ("Mouse",  270,  (200, 200, 200)),   # bottom
    ("Quit",     0,  (100, 120, 255)),   # right
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3D SHAPES                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class _Shape3D:
    """Shared state for all 3D interactive shapes."""
    def __init__(self, x, y, radius, color):
        self.x, self.y       = float(x), float(y)
        self.radius          = float(radius)
        self.color           = color
        self.vx = self.vy    = 0.0
        self.angle_x = self.angle_y = 0.0
        self.pinches         = []
        self.locked_hand_id  = None
        self.is_two_handed   = False
        self.is_grabbed_single = False
        self.is_rotating_manually = False
        self.initial_radius  = float(radius)
        self.initial_pinch_dist = 0
        self.two_hand_offset = (0, 0)
        self.drag_offset     = (0, 0)
        self.last_pinch_pos  = (0, 0)
        self.pinch_count     = 0
        self.last_pinch_time = 0.0
        self.was_pinched_last_frame = False
        self.to_delete       = False


class MatrixSphere3D(_Shape3D):
    """Wireframe sphere drawn from latitude/longitude rings."""

    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, (106, 50, 159))
        self.nodes, self.edges = [], []
        self._build(rings=10, segments=14)

    def _build(self, rings, segments):
        for i in range(rings + 1):
            phi = (i / rings) * math.pi
            for j in range(segments):
                theta = (j / segments) * 2 * math.pi
                self.nodes.append((
                    math.sin(phi) * math.cos(theta),
                    math.cos(phi),
                    math.sin(phi) * math.sin(theta),
                ))
        for i in range(rings):
            for j in range(segments):
                c = i * segments + j
                self.edges.append((c, i * segments + (j+1) % segments))
                self.edges.append((c, (i+1) * segments + j))

    def draw(self, img):
        if not self.locked_hand_id:
            self.angle_y += 0.003
            self.angle_x += 0.003
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        proj = []
        for nx, ny, nz in self.nodes:
            rx = nx*cy - nz*sy;  rz = nx*sy + nz*cy
            ry = ny*cx - rz*sx
            proj.append((int(rx*self.radius + self.x), int(ry*self.radius + self.y)))
        for a, b in self.edges:
            cv2.line(img, proj[a], proj[b], self.color, 1, cv2.LINE_AA)
        for px, py in proj:
            cv2.circle(img, (px, py), 2, (255, 0, 255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), (0,255,0), 1)


class MatrixPyramid3D(_Shape3D):
    """Wireframe tetrahedron."""

    NODES = [(0,-1,0), (-0.866,0.5,-0.5), (0.866,0.5,-0.5), (0,0.5,1)]
    EDGES = [(0,1),(0,2),(0,3),(1,2),(2,3),(3,1)]

    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, (0, 150, 255))
        self.nodes = self.NODES
        self.edges = self.EDGES

    def draw(self, img):
        if not self.locked_hand_id:
            self.angle_y += 0.04
            self.angle_x += 0.02
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        proj = []
        for nx, ny, nz in self.nodes:
            rx = nx*cy - nz*sy;  rz = nx*sy + nz*cy
            ry = ny*cx - rz*sx
            proj.append((int(rx*self.radius + self.x), int(ry*self.radius + self.y)))
        for a, b in self.edges:
            cv2.line(img, proj[a], proj[b], self.color, 2, cv2.LINE_AA)
        for px, py in proj:
            cv2.circle(img, (px, py), 4, (200,200,255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.color, 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RADIAL MENU RENDERER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _draw_radial_menu(img, cx, cy, hovered: int, progress: float):
    """
    Draw the radial menu centered at (cx, cy).
    hovered  : index of the highlighted option (-1 = none)
    progress : float [0,1] — confirmation arc fill around the hovered option
    """
    # Anchor dot
    ov = img.copy()
    cv2.circle(ov, (cx, cy), 22, (240, 240, 240), -1)
    cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)
    cv2.circle(img, (cx, cy), 22, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.circle(img, (cx-4, cy-4), 5, (255, 255, 255), -1, cv2.LINE_AA)

    for i, (label, angle_deg, color) in enumerate(MENU_OPTIONS):
        rad = math.radians(angle_deg)
        ox  = int(cx + MENU_RADIUS * math.cos(rad))
        oy  = int(cy - MENU_RADIUS * math.sin(rad))
        hot = (i == hovered)
        r   = MENU_ICON_RADIUS + (6 if hot else 0)

        # Spoke line from center to bubble
        ov = img.copy()
        cv2.line(ov, (cx, cy), (ox, oy), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)

        # Bubble gradient fill
        ov = img.copy()
        for j in range(r, 0, -1):
            ratio = j / r
            c = (int(color[0]*ratio), int(color[1]*ratio), int(color[2]*ratio))
            cv2.circle(ov, (ox, oy), j, c, -1)
        alpha = 0.8 if hot else 0.3
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

        # Border
        cv2.circle(img, (ox, oy), r, color if hot else (160,160,160),
                   2 if hot else 1, cv2.LINE_AA)

        # Glass highlight
        ov = img.copy()
        cv2.ellipse(ov, (ox - r//3, oy - r//2), (r//2, r//3), 45, 0, 360, (255,255,255), -1)
        cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)

        # Confirmation arc
        if hot and progress > 0:
            span = int(360 * progress)
            axes = (r+8, r+8)
            cv2.ellipse(img, (ox, oy), axes, -90, 0, 360,  (80,80,80),  3, cv2.LINE_AA)
            cv2.ellipse(img, (ox, oy), axes, -90, 0, span,  color,      3, cv2.LINE_AA)

        # Label below bubble
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.40
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        lx = ox - tw//2
        ly = oy + th//2 + r + 10

        ov = img.copy()
        cv2.rectangle(ov, (lx-5, ly-th-5), (lx+tw+5, ly+5), (20,20,20), -1)
        cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)
        cv2.putText(img, label, (lx, ly), font, scale,
                    (240,240,240) if hot else (170,170,170), 1, cv2.LINE_AA)


def _draw_hint(img):
    """Subtle bottom hint shown while the menu is closed."""
    h, w = img.shape[:2]
    hint = "open palm -> close fist -> radial menu"
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
    """Dedicated thread that always holds the freshest camera frame."""

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
    """Runs hand_detect() in a background thread so the display loop never blocks."""

    def __init__(self):
        self._in   = None;  self._in_lock  = threading.Lock()
        self._out  = None;  self._data     = []
        self._out_lock = threading.Lock()
        self.running   = True
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        while self.running:
            with self._in_lock:
                f = self._in
            if f is None:
                time.sleep(0.001)
                continue
            f, data = hand_detect(f)
            with self._out_lock:
                self._out  = f
                self._data = data

    def submit(self, frame):
        with self._in_lock:
            self._in = frame.copy()

    def get(self):
        with self._out_lock:
            return (self._out.copy() if self._out is not None else None,
                    list(self._data))

    def stop(self):
        self.running = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def _palm_open(pts):
    """Thumb (4) ↔ pinky (20) distance above threshold → open palm."""
    return _dist(pts[4], pts[20]) > SEUIL_PAUME


def _fist_closed(pts):
    """Thumb (4) ↔ pinky (20) distance below threshold → closed fist."""
    return _dist(pts[4], pts[20]) < SEUIL_POING_MENU


def _hovered_option(index_pos, cx, cy):
    """Return the index of the closest menu option within SEUIL_SELECT, or -1."""
    best_i, best_d = -1, float('inf')
    for i, (_, angle_deg, _) in enumerate(MENU_OPTIONS):
        ox = cx + MENU_RADIUS * math.cos(math.radians(angle_deg))
        oy = cy - MENU_RADIUS * math.sin(math.radians(angle_deg))
        d  = _dist(index_pos, (ox, oy))
        if d < SEUIL_SELECT and d < best_d:
            best_i, best_d = i, d
    return best_i


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SHAPES PHYSICS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _update_shape_physics(shape, LISSAGE=3.0):
    """Apply drag, fling inertia, and two-hand scaling to one shape."""
    shape.is_rotating_manually = False

    if len(shape.pinches) == 1:
        px, py, is_fist = shape.pinches[0]
        if not shape.is_grabbed_single:
            shape.is_grabbed_single  = True
            shape.is_two_handed      = False
            shape.drag_offset        = (shape.x - px, shape.y - py)
            shape.last_pinch_pos     = (px, py)

        if is_fist:
            # Rotate with fist movement
            shape.is_rotating_manually = True
            dx = px - shape.last_pinch_pos[0]
            dy = py - shape.last_pinch_pos[1]
            shape.angle_y  += dx * 0.015
            shape.angle_x  -= dy * 0.015
            shape.drag_offset = (shape.x - px, shape.y - py)
            shape.vx = shape.vy = 0
        else:
            # Drag with open pinch
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
        # Released — coast with friction
        shape.is_grabbed_single = shape.is_two_handed = False
        shape.x  += shape.vx;  shape.y  += shape.vy
        shape.vx *= 0.85;      shape.vy *= 0.85
        if abs(shape.vx) < 0.1: shape.vx = 0
        if abs(shape.vy) < 0.1: shape.vy = 0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    cam = CameraStream(CAM['source'], DISPLAY_W, DISPLAY_H)
    det = DetectionThread()

    # ── Radial menu state machine ────────────────────────────────────────────────
    # phase 0 : idle  — waiting for closed fist to open menu
    # phase 1 : open  — menu displayed, waiting for option selection
    menu_phase     = 0
    menu_cx        = DISPLAY_W // 2
    menu_cy        = DISPLAY_H // 2
    hovered_idx    = -1
    frames_hovered = 0
    action         = None   # confirmed action label

    # ── Shapes mode state ────────────────────────────────────────────────────────
    shapes         = []
    drawing_path   = []
    spawn_cd       = 0
    shapes_active  = False
    LISSAGE        = 3.0

    print("Ready — close your fist to open the radial menu.")
    while cam.read() is None:
        time.sleep(0.01)

    SX = DISPLAY_W / DETECT_W
    SY = DISPLAY_H / DETECT_H

    while True:
        frame = cam.read()
        if frame is None:
            continue

        det.submit(cv2.resize(frame, (DETECT_W, DETECT_H)))
        res_frame, hand_data = det.get()

        display = cv2.resize(res_frame, (DISPLAY_W, DISPLAY_H)) \
                  if res_frame is not None else frame

        now = time.time()
        if spawn_cd > 0:
            spawn_cd -= 1

        # Index fingertip in display coordinates
        index_disp = None
        if hand_data:
            ix, iy = hand_data[0]['points'][8]
            index_disp = (int(ix*SX), int(iy*SY))

        # ── Radial menu FSM ──────────────────────────────────────────────────────
        if menu_phase == 0:
            # Idle: detect closed fist to open menu
            if hand_data:
                pts  = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]
                if _fist_closed(pts):
                    menu_cx, menu_cy = pts[0]
                    menu_phase     = 1
                    hovered_idx    = -1
                    frames_hovered = 0
            _draw_hint(display)

        elif menu_phase == 1:
            # Open: track index finger, confirm on dwell
            if hand_data:
                pts = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]

                # Menu center follows the wrist slowly
                wx, wy   = pts[0]
                menu_cx += int((wx - menu_cx) * 0.12)
                menu_cy += int((wy - menu_cy) * 0.12)

                new_hov = _hovered_option(index_disp, menu_cx, menu_cy) \
                          if index_disp else -1

                if new_hov == hovered_idx and hovered_idx != -1:
                    frames_hovered += 1
                else:
                    hovered_idx    = new_hov
                    frames_hovered = 0

                progress = frames_hovered / FRAMES_CONFIRM if hovered_idx != -1 else 0

                if frames_hovered >= FRAMES_CONFIRM and hovered_idx != -1:
                    action         = MENU_OPTIONS[hovered_idx][0]
                    menu_phase     = 0
                    hovered_idx    = -1

                _draw_radial_menu(display, menu_cx, menu_cy, hovered_idx, progress)
            else:
                # Hand lost — close menu
                _draw_radial_menu(display, menu_cx, menu_cy, -1, 0)
                menu_phase = 0

        # ── Action dispatch ──────────────────────────────────────────────────────
        if action == "Quit":
            break

        elif action == "Mouse":
            action = None
            det.stop(); cam.stop(); release(); cv2.destroyAllWindows()
            subprocess.run(
                [sys.executable, "virt_mouse.py"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            return

        elif action == "Shapes":
            action        = None
            shapes_active = True
            menu_phase    = -1  # keep menu closed while shapes mode is active

        # ── Shapes mode ──────────────────────────────────────────────────────────
        if shapes_active:

            # Two-hand pyramid gesture
            if len(hand_data) == 2:
                h1 = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]
                h2 = [(int(x*SX), int(y*SY)) for x,y in hand_data[1]['points']]
                if _dist(h1[0], h2[0]) > 100:
                    t1, i1 = h1[4], h1[8]
                    t2, i2 = h2[4], h2[8]
                    if (_dist(t1,i1) > 50 and _dist(t2,i2) > 50 and
                            _dist(t1,t2) < 60 and _dist(i1,i2) < 60 and spawn_cd == 0):
                        cx = (t1[0]+t2[0]+i1[0]+i2[0])//4
                        cy = (t1[1]+t2[1]+i1[1]+i2[1])//4
                        shapes.append(MatrixPyramid3D(cx, cy, 80))
                        spawn_cd = 180

            # Collect active pinches from all hands
            active_pinches = []
            for hi, hand in enumerate(hand_data):
                pts = hand['points']
                tx  = int(pts[4][0]*SX);  ty = int(pts[4][1]*SY)
                ix  = int(pts[8][0]*SX);  iy = int(pts[8][1]*SY)
                cx  = (tx+ix)//2;        cy = (ty+iy)//2
                is_fist  = hand['gestes']['poing']
                is_pinch = hand['gestes']['pincement']
                if is_pinch or is_fist:
                    active_pinches.append((cx, cy, is_fist, hi))
                    cv2.circle(display, (cx, cy), 8,
                               (0,0,255) if is_fist else (0,255,0), -1)

            # Reset per-shape pinch list, keep locked-hand state
            for shape in shapes:
                shape.pinches = []
                if shape.locked_hand_id is not None:
                    lid = shape.locked_hand_id
                    if lid < len(hand_data):
                        pts = hand_data[lid]['points']
                        tx  = int(pts[4][0]*SX);  ty = int(pts[4][1]*SY)
                        ix  = int(pts[8][0]*SX);  iy = int(pts[8][1]*SY)
                        cx  = (tx+ix)//2;        cy = (ty+iy)//2
                        # Re-check fist using knuckle distances (more reliable than gestes here)
                        w0 = pts[0]
                        fist = (
                            _dist(pts[12], w0) < _dist(pts[9],  w0) and
                            _dist(pts[16], w0) < _dist(pts[13], w0) and
                            _dist(pts[20], w0) < _dist(pts[17], w0)
                        )
                        if fist:
                            shape.pinches.append((cx, cy, fist))
                        else:
                            shape.locked_hand_id = None
                    else:
                        shape.locked_hand_id = None

            # Assign new pinches to shapes
            for (px, py, is_fist, hi) in active_pinches:
                for shape in shapes:
                    if shape.locked_hand_id == hi:
                        continue
                    if _dist((px,py), (shape.x, shape.y)) < shape.radius:
                        shape.pinches.append((px, py, is_fist))
                        if is_fist:
                            shape.locked_hand_id = hi
                        break

            # Triple-pinch to delete
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

            # Circle drawing (single open pinch not touching any shape)
            grabbing = any(s.pinches for s in shapes)
            if len(active_pinches) == 1 and not grabbing:
                px, py, is_fist, _ = active_pinches[0]
                if not is_fist:
                    drawing_path.append((px, py))
                    for i in range(1, len(drawing_path)):
                        cv2.line(display, drawing_path[i-1], drawing_path[i],
                                 (255, 200, 0), 4)
            else:
                if len(drawing_path) > 20:
                    sp, ep = drawing_path[0], drawing_path[-1]
                    if _dist(sp, ep) < 100:
                        contour = np.array(drawing_path, dtype=np.int32).reshape(-1,1,2)
                        (cx, cy), r = cv2.minEnclosingCircle(contour)
                        if r > 30:
                            errs = [abs(_dist(p, (cx,cy)) - r) for p in drawing_path]
                            if (sum(errs)/len(errs)) / r < 0.25:
                                shapes.append(MatrixSphere3D(int(cx), int(cy), int(r)))
                drawing_path.clear()

            # Physics update
            for shape in shapes:
                _update_shape_physics(shape, LISSAGE)

            for shape in shapes:
                shape.draw(display)

        cv2.imshow("Vision AI", display)
        if cv2.waitKey(1) == 27:
            break

    det.stop()
    cam.stop()
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()