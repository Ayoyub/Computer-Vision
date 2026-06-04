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
# ║  3D SHAPES (Vectorisées avec NumPy)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class _Shape3D:
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
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, (106, 50, 159))
        self.nodes, self.edges = [], []
        self._build(rings=10, segments=14)

    def _build(self, rings, segments):
        for i in range(rings + 1):
            phi = (i / rings) * math.pi
            for j in range(segments):
                theta = (j / segments) * 2 * math.pi
                self.nodes.append([
                    math.sin(phi) * math.cos(theta),
                    math.cos(phi),
                    math.sin(phi) * math.sin(theta),
                ])
        self.nodes = np.array(self.nodes)
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

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R = Ry @ Rx
        
        rotated = self.nodes @ R.T
        proj = [(int(nx * self.radius + self.x), int(ny * self.radius + self.y)) for nx, ny, _ in rotated]
        
        for a, b in self.edges:
            cv2.line(img, proj[a], proj[b], self.color, 1, cv2.LINE_AA)
        for px, py in proj:
            cv2.circle(img, (px, py), 2, (255, 0, 255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), (0,255,0), 1)


class MatrixPyramid3D(_Shape3D):
    NODES = np.array([(0,-1,0), (-0.866,0.5,-0.5), (0.866,0.5,-0.5), (0,0.5,1)])
    EDGES = [(0,1),(0,2),(0,3),(1,2),(2,3),(3,1)]

    def __init__(self, x, y, radius):
        super().__init__(x, y, radius, (0, 150, 255))
        self.nodes = self.NODES
        self.edges = self.EDGES

    def draw(self, img):
        if not self.locked_hand_id:
            self.angle_y += 0.04
            self.angle_x += 0.02
            
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R = Ry @ Rx
        
        rotated = self.nodes @ R.T
        proj = [(int(nx * self.radius + self.x), int(ny * self.radius + self.y)) for nx, ny, _ in rotated]
        
        for a, b in self.edges:
            cv2.line(img, proj[a], proj[b], self.color, 2, cv2.LINE_AA)
        for px, py in proj:
            cv2.circle(img, (px, py), 4, (200,200,255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.color, 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RADIAL MENU RENDERER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _draw_radial_menu(img, cx, cy, hovered, cursor_x=None, cursor_y=None, rotation_offset=0.0):
    # Base du point d'ancrage
    cv2.circle(img, (cx, cy), 15, (20, 20, 20), -1)
    cv2.circle(img, (cx, cy), 15, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    # Ligne élastique
    if cursor_x and cursor_y:
        ov = img.copy()
        cv2.line(ov, (cx, cy), (cursor_x, cursor_y), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.4, img, 0.6, 0, img)
        cv2.circle(img, (cursor_x, cursor_y), 8, (255, 255, 255), -1, cv2.LINE_AA)

    for i, (label, base_angle, color) in enumerate(MENU_OPTIONS):
        # --- APPLIQUER LA ROTATION ICI ---
        angle_deg = (base_angle + rotation_offset) % 360
        rad = math.radians(angle_deg)
        
        ox  = int(cx + MENU_RADIUS * math.cos(rad))
        oy  = int(cy - MENU_RADIUS * math.sin(rad))
        hot = (i == hovered)
        
        r = MENU_ICON_RADIUS + (8 if hot else 0)

        # Tracé des branches
        cv2.line(img, (cx, cy), (ox, oy), (80, 80, 80), 1, cv2.LINE_AA)

        if hot:
            for glow_r in [r+12, r+8, r+4]:
                ov = img.copy()
                cv2.circle(ov, (ox, oy), glow_r, color, -1)
                cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)

        ov = img.copy()
        cv2.circle(ov, (ox, oy), r, color if hot else (40, 40, 40), -1)
        cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)

        cv2.circle(img, (ox, oy), r, color if hot else (120, 120, 120), 2 if hot else 1, cv2.LINE_AA)

        ov = img.copy()
        cv2.ellipse(ov, (ox - r//3, oy - r//2), (r//2, r//3), 45, 0, 360, (255,255,255), -1)
        cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)

        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45 if hot else 0.40
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        lx, ly = ox - tw//2, oy + th//2 + r + 15

        ov = img.copy()
        cv2.rectangle(ov, (lx-6, ly-th-6), (lx+tw+6, ly+6), (10,10,10), -1)
        cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
        cv2.putText(img, label, (lx, ly), font, scale, (255,255,255) if hot else (170,170,170), 1 if hot else 1, cv2.LINE_AA)

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
# ║  HELPERS & PHYSICS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def _hovered_option(index_pos, cx, cy):
    best_i, best_d = -1, float('inf')
    for i, (_, angle_deg, _) in enumerate(MENU_OPTIONS):
        ox = cx + MENU_RADIUS * math.cos(math.radians(angle_deg))
        oy = cy - MENU_RADIUS * math.sin(math.radians(angle_deg))
        d  = _dist(index_pos, (ox, oy))
        if d < SEUIL_SELECT and d < best_d:
            best_i, best_d = i, d
    return best_i

def _update_shape_physics(shape, LISSAGE=3.0):
    shape.is_rotating_manually = False

    if len(shape.pinches) == 1:
        px, py, is_fist = shape.pinches[0]
        if not shape.is_grabbed_single:
            shape.is_grabbed_single  = True
            shape.is_two_handed      = False
            shape.drag_offset        = (shape.x - px, shape.y - py)
            shape.last_pinch_pos     = (px, py)

        if is_fist:
            shape.is_rotating_manually = True
            dx = px - shape.last_pinch_pos[0]
            dy = py - shape.last_pinch_pos[1]
            shape.angle_y  += dx * 0.015
            shape.angle_x  -= dy * 0.015
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



def _hovered_option_by_angle(x, y, cx, cy, rotation_offset):
    """Sélectionne l'option en prenant en compte la rotation dynamique du menu."""
    angle_rad = math.atan2(-(y - cy), x - cx) 
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    
    best_i = -1
    min_diff = float('inf')
    
    for i, (_, base_angle, _) in enumerate(MENU_OPTIONS):
        # On ajoute la rotation actuelle à l'angle de base de l'option
        opt_angle = (base_angle + rotation_offset) % 360
        
        diff = min(abs(angle_deg - opt_angle), 360 - abs(angle_deg - opt_angle))
        if diff < min_diff:
            min_diff = diff
            best_i = i
            
    return best_i if min_diff <= 45 else -1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    cam = CameraStream(CAM['source'], DISPLAY_W, DISPLAY_H)
    det = DetectionThread()

    menu_phase     = 0
    menu_cx        = DISPLAY_W // 2
    menu_cy        = DISPLAY_H // 2
    hovered_idx    = -1
    frames_hovered = 0
    action         = None
    menu_rotation = 0.0

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

    # Bloc Try/Finally pour assurer la fermeture propre de la caméra
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
            if spawn_cd > 0:
                spawn_cd -= 1

            index_disp = None
            if hand_data:
                ix, iy = hand_data[0]['points'][8]
                index_disp = (int(ix*SX), int(iy*SY))

            # ── Radial menu FSM ──────────────────────────────────────────────────────
            menu_rotation = (menu_rotation + 0.6) % 360  # Augmente 0.6 pour tourner plus vite
            if menu_phase == 0:
                # Idle : On attend un poing fermé pour INVOQUER le menu
                if hand_data:
                    if hand_data[0]['gestes']['poing']:
                        # On ancre le menu là où le poing a été fermé
                        pts = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]
                        menu_cx, menu_cy = pts[0]
                        menu_phase  = 1
                        hovered_idx = -1
                _draw_hint(display)

            elif menu_phase == 1:
                # Ouvert : On maintient le poing et on tire vers une option
                if hand_data:
                    poing = hand_data[0]['gestes']['poing']
                    pts = [(int(x*SX), int(y*SY)) for x,y in hand_data[0]['points']]
                    wx, wy = pts[0] # Position actuelle du poignet (le "joystick")

                    dist = _dist((wx, wy), (menu_cx, menu_cy))
                    
                    # Zone morte au centre pour annuler (40 pixels)
                    if dist > 40:
                        hovered_idx = _hovered_option_by_angle(wx, wy, menu_cx, menu_cy, menu_rotation)
                    else:
                        hovered_idx = -1

                    _draw_radial_menu(display, menu_cx, menu_cy, hovered_idx, wx, wy, menu_rotation)

                    # Validation : on OUVRE la main pour confirmer l'action visée
                    if not poing:
                        if hovered_idx != -1:
                            action = MENU_OPTIONS[hovered_idx][0]
                        menu_phase = 0
                else:
                    # Sécurité : la main sort du champ, on annule
                    menu_phase = 0

            # ── Action dispatch ──────────────────────────────────────────────────────
            if action == "Quit":
                break

            elif action == "Mouse":
                action = None
                # Transfert de la caméra et de la détection vers la souris
                virt_mouse.run_mouse_mode(cam, det)
                # Réinitialisation après le retour du mode souris
                menu_phase = 0
                hovered_idx = -1
                shapes.clear()

            elif action == "Shapes":
                action        = None
                shapes_active = True
                menu_phase    = -1  

            # ── Shapes mode ──────────────────────────────────────────────────────────
            if shapes_active:
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
                        cv2.circle(display, (cx, cy), 8, (0,0,255) if is_fist else (0,255,0), -1)

                for shape in shapes:
                    shape.pinches = []
                    if shape.locked_hand_id is not None:
                        lid = shape.locked_hand_id
                        if lid < len(hand_data):
                            pts = hand_data[lid]['points']
                            tx  = int(pts[4][0]*SX);  ty = int(pts[4][1]*SY)
                            ix  = int(pts[8][0]*SX);  iy = int(pts[8][1]*SY)
                            cx  = (tx+ix)//2;        cy = (ty+iy)//2
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

                for (px, py, is_fist, hi) in active_pinches:
                    for shape in shapes:
                        if shape.locked_hand_id == hi:
                            continue
                        if _dist((px,py), (shape.x, shape.y)) < shape.radius:
                            shape.pinches.append((px, py, is_fist))
                            if is_fist:
                                shape.locked_hand_id = hi
                            break

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

                grabbing = any(s.pinches for s in shapes)
                if len(active_pinches) == 1 and not grabbing:
                    px, py, is_fist, _ = active_pinches[0]
                    if not is_fist:
                        drawing_path.append((px, py))
                        for i in range(1, len(drawing_path)):
                            cv2.line(display, drawing_path[i-1], drawing_path[i], (255, 200, 0), 4)
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

                for shape in shapes:
                    _update_shape_physics(shape, LISSAGE)

                for shape in shapes:
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