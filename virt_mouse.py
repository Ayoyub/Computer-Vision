import cv2
import tkinter as tk
from pynput.mouse import Controller, Button
from config import CAM

CAM_W, CAM_H = CAM['detect_w'], CAM['detect_h']

# ── Zone active (pourcentage du frame cam mappé sur tout l'écran) ─────────────────
# 0.0 = bord gauche/haut  |  1.0 = bord droit/bas (coordonnées caméra normalisées)
ACTIVE_ZONE = {
    'x_min': 0.15,   # commence à 15% depuis la gauche
    'x_max': 0.85,   # finit à 85% depuis la gauche
    'y_min': 0.10,   # commence à 10% depuis le haut
    'y_max': 0.90,   # finit à 90% depuis le haut
}


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

    # Coins accentués
    corner_len = 16
    color = (120, 240, 160)
    for cx, cy, sx, sy in [
        (ax1, ay1,  1,  1), (ax2, ay1, -1,  1),
        (ax1, ay2,  1, -1), (ax2, ay2, -1, -1),
    ]:
        cv2.line(display, (cx, cy), (cx + sx*corner_len, cy), color, 2, cv2.LINE_AA)
        cv2.line(display, (cx, cy), (cx, cy + sy*corner_len), color, 2, cv2.LINE_AA)

    # Label centré en haut de la zone
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
        cv2.line(display, (ix-8, iy-8), (ix+8, iy+8), (160,160,160), 1, cv2.LINE_AA)
        cv2.line(display, (ix+8, iy-8), (ix-8, iy+8), (160,160,160), 1, cv2.LINE_AA)

    label = "fist — idle" if poing else ("click" if pincement else "move")
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    bx, by = 12, h - 20

    ov = display.copy()
    cv2.rectangle(ov, (bx-6, by-th-6), (bx+tw+6, by+6), (15,15,15), -1)
    cv2.addWeighted(ov, 0.50, display, 0.50, 0, display)
    cv2.putText(display, label, (bx, by), font, scale, (200,200,200), 1, cv2.LINE_AA)

    # Hint calibration
    hint = "C = recalibrate zone  |  ESC = back to menu"
    (hw, hh), _ = cv2.getTextSize(hint, font, 0.33, 1)
    hx = display.shape[1] // 2 - hw // 2
    cv2.putText(display, hint, (hx, h - 8), font, 0.33, (90, 90, 90), 1, cv2.LINE_AA)

    _draw_active_zone(display, ax1, ay1, ax2, ay2)


def run_mouse_mode(cam, det):
    """Boucle principale de la souris, appelée par main.py"""
    root = tk.Tk()
    root.withdraw()
    SCREEN_W = root.winfo_screenwidth()
    SCREEN_H = root.winfo_screenheight()
    root.destroy()

    mouse       = Controller()
    clic_locked = False

    print(f"Mode Souris: {SCREEN_W}x{SCREEN_H} | ESC = retour menu | C = recalibrer zone")

    SX = CAM['display_w'] / CAM_W
    SY = CAM['display_h'] / CAM_H

    DW = CAM['display_w']
    DH = CAM['display_h']

    # Zone active en pixels affichage (recalculée depuis les ratios)
    def _zone_px():
        return (
            int(ACTIVE_ZONE['x_min'] * DW),
            int(ACTIVE_ZONE['y_min'] * DH),
            int(ACTIVE_ZONE['x_max'] * DW),
            int(ACTIVE_ZONE['y_max'] * DH),
        )

    ax1, ay1, ax2, ay2 = _zone_px()

    # ── Mode calibration ──────────────────────────────────────────────────────────
    # Appuie sur C : place ta main aux 4 extrêmes pendant ~2s chacune
    # (implémentation simple : on echantillonne les positions min/max sur 90 frames)
    calibrating      = False
    calib_frames     = 0
    CALIB_DURATION   = 90        # frames d'échantillonnage
    calib_xmin, calib_xmax = DW, 0
    calib_ymin, calib_ymax = DH, 0

    while True:
        frame = cam.read()
        if frame is None:
            continue

        det.submit(cv2.resize(frame, (CAM_W, CAM_H)))
        res_frame, hand_data = det.get()

        display = cv2.resize(res_frame, (DW, DH)) if res_frame is not None else frame

        if hand_data:
            pts       = hand_data[0]['points']
            poing     = hand_data[0]['gestes']['poing']
            pincement = hand_data[0]['gestes']['pincement']
            pts_disp  = [(int(x*SX), int(y*SY)) for x, y in pts]

            ix_disp, iy_disp = pts_disp[8]

            if calibrating:
                # Enregistre l'étendue du mouvement de l'index
                calib_xmin = min(calib_xmin, ix_disp)
                calib_xmax = max(calib_xmax, ix_disp)
                calib_ymin = min(calib_ymin, iy_disp)
                calib_ymax = max(calib_ymax, iy_disp)
                calib_frames += 1

                # Overlay calibration
                margin = 10
                ax1_c = max(0, calib_xmin - margin)
                ay1_c = max(0, calib_ymin - margin)
                ax2_c = min(DW, calib_xmax + margin)
                ay2_c = min(DH, calib_ymax + margin)
                cv2.rectangle(display, (ax1_c, ay1_c), (ax2_c, ay2_c), (0, 200, 255), 1, cv2.LINE_AA)

                remaining = CALIB_DURATION - calib_frames
                msg = f"Move hand to all corners... {remaining}"
                cv2.putText(display, msg, (DW//2 - 160, DH//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1, cv2.LINE_AA)

                if calib_frames >= CALIB_DURATION:
                    # Sauvegarde la zone avec un léger padding
                    pad_x = max(20, int((calib_xmax - calib_xmin) * 0.05))
                    pad_y = max(20, int((calib_ymax - calib_ymin) * 0.05))
                    ACTIVE_ZONE['x_min'] = max(0.0, (calib_xmin - pad_x) / DW)
                    ACTIVE_ZONE['x_max'] = min(1.0, (calib_xmax + pad_x) / DW)
                    ACTIVE_ZONE['y_min'] = max(0.0, (calib_ymin - pad_y) / DH)
                    ACTIVE_ZONE['y_max'] = min(1.0, (calib_ymax + pad_y) / DH)
                    ax1, ay1, ax2, ay2 = _zone_px()
                    calibrating = False
                    print(f"Zone calibrée: x=[{ACTIVE_ZONE['x_min']:.2f},{ACTIVE_ZONE['x_max']:.2f}] y=[{ACTIVE_ZONE['y_min']:.2f},{ACTIVE_ZONE['y_max']:.2f}]")

            else:
                if not poing:
                    # Remappe l'index depuis la zone active vers tout l'écran
                    target_x = _remap(ix_disp, ax1, ax2, 0, SCREEN_W - 1)
                    target_y = _remap(iy_disp, ay1, ay2, 0, SCREEN_H - 1)
                    mouse.position = (target_x, target_y)

                if pincement and not clic_locked:
                    mouse.click(Button.left)
                    clic_locked = True
                elif not pincement:
                    clic_locked = False

            _draw_hud(display, pts_disp, poing, pincement, ax1, ay1, ax2, ay2)

        cv2.imshow("Vision AI", display)
        key = cv2.waitKey(1)

        if key == 27:   # ESC → retour menu
            break
        elif key == ord('c') or key == ord('C'):
            # Lance la calibration
            calibrating    = True
            calib_frames   = 0
            calib_xmin, calib_xmax = DW, 0
            calib_ymin, calib_ymax = DH, 0
            print("Calibration lancée — bouge ta main dans tous les coins pendant 3s")