"""
virt_mouse.py — virtual mouse mode (hand-controlled cursor)
═══════════════════════════════════════════════════════════
Cursor movement uses a One Euro filter for jitter-free, low-lag control.
Input injection strategy (in priority order):
  1. ydotool  — works on X11 AND Wayland (kernel uinput via ydotoold)
  2. pynput   — fallback (X11 only)
On Wayland, `systemd --user start ydotoold` must be running.
"""
import shutil
import subprocess
import time

import cv2
import sys
from config import CAM, MOUSE
from gestures import check_double_fist, draw_double_fist_hint
from kalman_filter import OneEuroFilter2D

CAM_W, CAM_H = CAM['detect_w'], CAM['detect_h']

# ── Zone active (fraction normalisée du frame cam mappé sur tout l'écran) ───────
ACTIVE_ZONE = {
    'x_min': MOUSE['zone_x_min'],
    'x_max': MOUSE['zone_x_max'],
    'y_min': MOUSE['zone_y_min'],
    'y_max': MOUSE['zone_y_max'],
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INPUT BACKEND (ydotool → pynput)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class _MouseBackend:
    """Moves the cursor + clicks, using ydotool if available, else pynput."""

    def __init__(self):
        self.kind = None
        if shutil.which('ydotool') and self._ydotoold_alive():
            self.kind = 'ydotool'
            print("  → input backend : ydotool (Wayland + X11)")
        else:
            try:
                from pynput.mouse import Controller
                self._ctl = Controller()
                self.kind = 'pynput'
                print("  → input backend : pynput (X11 only — pour Wayland, "
                      "installe ydotool et lance 'systemd --user start ydotoold')")
            except Exception as e:
                print(f"  → ERREUR : aucun backend de saisie disponible ({e})")
                self.kind = None

    @staticmethod
    def _ydotoold_alive() -> bool:
        """ydotoold is alive if ydotool can move the mouse without error."""
        try:
            subprocess.run(['ydotool', 'mousemove', '-x', '0', '-y', '0'],
                           check=True, capture_output=True, timeout=2)
            return True
        except Exception:
            return False

    def get_screen_size(self):
        """Return (width, height) in pixels using the best available backend."""
        if self.kind == 'ydotool':
            try:
                out = subprocess.check_output(
                    ['ydotool', 'getgeometry'], timeout=2).decode().strip()
                w, h = out.split()[:2]
                return int(w), int(h)
            except Exception:
                pass
        # Windows: native ctypes call
        if sys.platform == 'win32':
            try:
                import ctypes
                user32 = ctypes.windll.user32
                w = user32.GetSystemMetrics(78)  # SM_CXSCREEN
                h = user32.GetSystemMetrics(79)  # SM_CYSCREEN
                if w and h:
                    return w, h
            except Exception:
                pass
        # macOS: system_profiler
        if sys.platform == 'darwin':
            try:
                import subprocess as _sp
                out = _sp.check_output(
                    ['system_profiler', 'SPDisplaysDataType'], timeout=2).decode()
                import re as _re
                m = _re.search(r'Resolution\":\s*"([\d]+)\s*x\s*([\d]+)', out, re.IGNORECASE)
                if m:
                    return int(m.group(1)), int(m.group(2))
            except Exception:
                pass
        # Linux X11: xdpyinfo
        try:
            out = subprocess.check_output(['xdpyinfo'], timeout=2).decode()
            for line in out.splitlines():
                if 'dimensions:' in line:
                    parts = line.split('dimensions:')[1].strip().split()
                    return int(parts[0]), int(parts[1])
        except Exception:
            pass
        # Linux Wayland (no ydotool): xrandr to find the largest connected monitor
        try:
            out = subprocess.check_output(['xrandr'], timeout=2).decode()
            best = (0, 0)
            for line in out.splitlines():
                if '*' in line and 'connected' not in line:
                    for tok in line.split():
                        if 'x' in tok and 'Hz' not in tok:
                            w, h = tok.split('x')[:2]
                            if int(w) * int(h) > best[0] * best[1]:
                                best = (int(w), int(h))
                    break
            if best[0]:
                return best
        except Exception:
            pass
        return 1920, 1080


    def move(self, x: int, y: int):
        if self.kind == 'ydotool':
            try:
                subprocess.run(['ydotool', 'mousemove', '--absolute',
                                '-x', str(int(x)), '-y', str(int(y))],
                               capture_output=True, timeout=2)
            except Exception:
                pass
        elif self.kind == 'pynput':
            self._ctl.position = (x, y)

    def click(self):
        if self.kind == 'ydotool':
            try:
                subprocess.run(['ydotool', 'click', '1'],
                               capture_output=True, timeout=2)
            except Exception:
                pass
        elif self.kind == 'pynput':
            from pynput.mouse import Button
            self._ctl.click(Button.left)

    def release(self):
        pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RENDERING                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _remap(value, in_min, in_max, out_min, out_max):
    """Remapping linéaire clampé."""
    if in_max == in_min:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    ratio = max(0.0, min(1.0, ratio))
    return int(out_min + ratio * (out_max - out_min))


def _draw_active_zone(display, ax1, ay1, ax2, ay2):
    """Affiche la zone active sous forme d'un cadre néon semi-transparent."""
    ov = display.copy()
    cv2.rectangle(ov, (ax1, ay1), (ax2, ay2), (80, 200, 120), -1)
    cv2.addWeighted(ov, 0.06, display, 0.94, 0, display)
    cv2.rectangle(display, (ax1, ay1), (ax2, ay2), (80, 200, 120), 1, cv2.LINE_AA)

    corner_len = 16
    color = (120, 240, 160)
    for cx, cy, sx, sy in [
        (ax1, ay1, 1, 1), (ax2, ay1, -1, 1),
        (ax1, ay2, 1, -1), (ax2, ay2, -1, -1),
    ]:
        cv2.line(display, (cx, cy), (cx + sx * corner_len, cy), color, 2, cv2.LINE_AA)
        cv2.line(display, (cx, cy), (cx, cy + sy * corner_len), color, 2, cv2.LINE_AA)

    label = "active zone"
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.33
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    lx = ax1 + (ax2 - ax1) // 2 - tw // 2
    ly = ay1 - 6
    cv2.putText(display, label, (lx, ly), font, scale, (80, 200, 120), 1, cv2.LINE_AA)


def _draw_hud(display, pts, poing, pincement, ax1, ay1, ax2, ay2):
    """HUD minimaliste pour le mode souris."""
    h = display.shape[0]
    ix, iy = pts[8]

    if pincement:
        ov = display.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.35, display, 0.65, 0, display)
        cv2.circle(display, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        cv2.line(display, (ix - 8, iy - 8), (ix + 8, iy + 8), (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(display, (ix + 8, iy - 8), (ix - 8, iy + 8), (160, 160, 160), 1, cv2.LINE_AA)

    label = "fist — idle" if poing else ("click" if pincement else "move")
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    bx, by = 12, h - 20

    ov = display.copy()
    cv2.rectangle(ov, (bx - 6, by - th - 6), (bx + tw + 6, by + 6), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.50, display, 0.50, 0, display)
    cv2.putText(display, label, (bx, by), font, scale, (200, 200, 200), 1, cv2.LINE_AA)

    hint = "C = recalibrate zone  |  ESC = back to menu"
    (hw, hh), _ = cv2.getTextSize(hint, font, 0.33, 1)
    hx = display.shape[1] // 2 - hw // 2
    cv2.putText(display, hint, (hx, h - 8), font, 0.33, (90, 90, 90), 1, cv2.LINE_AA)

    _draw_active_zone(display, ax1, ay1, ax2, ay2)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_mouse_mode(cam, det):
    """Boucle principale de la souris, appelée par main.py"""
    mouse = _MouseBackend()
    if mouse.kind is None:
        print("Impossible de contrôler la souris — mode souris indisponible.")
        time.sleep(2)
        return

    SCREEN_W, SCREEN_H = mouse.get_screen_size()

    clic_locked = False
    double_fist_ctr = [0]

    # Filtre 1€ dédié au curseur (plus serré que le lissage des landmarks)
    cursor_filter = OneEuroFilter2D(
        fc_min=MOUSE['cursor_min_cutoff'],
        beta=MOUSE['cursor_beta'],
        fc_d=MOUSE['cursor_d_cutoff'],
    )
    cursor_x = cursor_y = None  # position du curseur filtrée

    print(f"Mouse mode: {SCREEN_W}x{SCREEN_H} | both fists = menu | C = recalibrate | ESC = menu")

    SX = CAM['display_w'] / CAM_W
    SY = CAM['display_h'] / CAM_H
    DW = CAM['display_w']
    DH = CAM['display_h']

    def _zone_px():
        return (
            int(ACTIVE_ZONE['x_min'] * DW),
            int(ACTIVE_ZONE['y_min'] * DH),
            int(ACTIVE_ZONE['x_max'] * DW),
            int(ACTIVE_ZONE['y_max'] * DH),
        )

    ax1, ay1, ax2, ay2 = _zone_px()

    # ── Mode calibration ────────────────────────────────────────────────────────
    calibrating = False
    calib_frames = 0
    CALIB_DURATION = 90
    calib_xmin, calib_xmax = DW, 0
    calib_ymin, calib_ymax = DH, 0

    t0 = time.monotonic()

    while True:
        frame = cam.read()
        if frame is None:
            continue

        det.submit(cv2.resize(frame, (CAM_W, CAM_H)))
        res_frame, hand_data = det.get()

        display = cv2.resize(res_frame, (DW, DH)) if res_frame is not None else frame
        t = time.monotonic() - t0

        if hand_data:
            pts = hand_data[0]['points']
            poing = hand_data[0]['gestes']['poing']
            pincement = hand_data[0]['gestes']['pincement']
            pts_disp = [(int(x * SX), int(y * SY)) for x, y in pts]

            ix_disp, iy_disp = pts_disp[8]

            if calibrating:
                calib_xmin = min(calib_xmin, ix_disp)
                calib_xmax = max(calib_xmax, ix_disp)
                calib_ymin = min(calib_ymin, iy_disp)
                calib_ymax = max(calib_ymax, iy_disp)
                calib_frames += 1

                margin = 10
                ax1_c = max(0, calib_xmin - margin)
                ay1_c = max(0, calib_ymin - margin)
                ax2_c = min(DW, calib_xmax + margin)
                ay2_c = min(DH, calib_ymax + margin)
                cv2.rectangle(display, (ax1_c, ay1_c), (ax2_c, ay2_c), (0, 200, 255), 1, cv2.LINE_AA)

                remaining = CALIB_DURATION - calib_frames
                msg = f"Move hand to all corners... {remaining}"
                cv2.putText(display, msg, (DW // 2 - 160, DH // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

                if calib_frames >= CALIB_DURATION:
                    pad_x = max(20, int((calib_xmax - calib_xmin) * 0.05))
                    pad_y = max(20, int((calib_ymax - calib_ymin) * 0.05))
                    ACTIVE_ZONE['x_min'] = max(0.0, (calib_xmin - pad_x) / DW)
                    ACTIVE_ZONE['x_max'] = min(1.0, (calib_xmax + pad_x) / DW)
                    ACTIVE_ZONE['y_min'] = max(0.0, (calib_ymin - pad_y) / DH)
                    ACTIVE_ZONE['y_max'] = min(1.0, (calib_ymax + pad_y) / DH)
                    ax1, ay1, ax2, ay2 = _zone_px()
                    calibrating = False
                    print(f"Zone calibrée: x=[{ACTIVE_ZONE['x_min']:.2f},{ACTIVE_ZONE['x_max']:.2f}] "
                          f"y=[{ACTIVE_ZONE['y_min']:.2f},{ACTIVE_ZONE['y_max']:.2f}]")

            else:
                if not poing:
                    # Remappe l'index depuis la zone active vers tout l'écran,
                    # puis lisse avec le filtre 1€ (jitter au repos éliminé,
                    # zéro lag sur les mouvements rapides).
                    target_x = _remap(ix_disp, ax1, ax2, 0, SCREEN_W - 1)
                    target_y = _remap(iy_disp, ay1, ay2, 0, SCREEN_H - 1)
                    fx, fy = cursor_filter.filter(target_x, target_y, t)
                    if cursor_x is None:
                        cursor_x, cursor_y = int(fx), int(fy)
                    else:
                        cursor_x = int(fx)
                        cursor_y = int(fy)
                    mouse.move(cursor_x, cursor_y)

                if pincement and not clic_locked:
                    mouse.click()
                    clic_locked = True
                elif not pincement:
                    clic_locked = False

            _draw_hud(display, pts_disp, poing, pincement, ax1, ay1, ax2, ay2)

        # ── Double-fist → back to menu ────────────────────────────────────────────
        if check_double_fist(hand_data if hand_data else [], double_fist_ctr):
            break
        draw_double_fist_hint(display, double_fist_ctr)

        cv2.imshow("Vision AI", display)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key in (ord('c'), ord('C')):
            calibrating = True
            calib_frames = 0
            calib_xmin, calib_xmax = DW, 0
            calib_ymin, calib_ymax = DH, 0
            print("Calibration lancée — bouge ta main dans tous les coins pendant ~2s")
