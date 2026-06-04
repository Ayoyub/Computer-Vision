import cv2
import tkinter as tk
from pynput.mouse import Controller, Button
from config import CAM

CAM_W, CAM_H = CAM['detect_w'], CAM['detect_h']

def _draw_hud(display, pts, poing, pincement):
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

def run_mouse_mode(cam, det):
    """Boucle principale de la souris, appelée par main.py"""
    # Initialisation de Tkinter ici pour éviter qu'elle s'exécute à l'import
    root = tk.Tk()
    root.withdraw()
    SCREEN_W = root.winfo_screenwidth()
    SCREEN_H = root.winfo_screenheight()
    root.destroy()
    
    mouse = Controller()
    clic_locked = False
    
    print(f"Mode Souris: {SCREEN_W}x{SCREEN_H} | Appuie sur ESC pour retourner au menu")
    
    SX = CAM['display_w'] / CAM_W
    SY = CAM['display_h'] / CAM_H

    while True:
        frame = cam.read()
        if frame is None:
            continue

        det.submit(cv2.resize(frame, (CAM_W, CAM_H)))
        res_frame, hand_data = det.get()
        
        display = cv2.resize(res_frame, (CAM['display_w'], CAM['display_h'])) \
                  if res_frame is not None else frame

        if hand_data:
            pts       = hand_data[0]['points']
            poing     = hand_data[0]['gestes']['poing']
            pincement = hand_data[0]['gestes']['pincement']
            pts_disp  = [(int(x*SX), int(y*SY)) for x, y in pts]

            if not poing:
                ix, iy  = pts[8]
                target_x = max(0, min(SCREEN_W-1, int((ix / CAM_W) * SCREEN_W)))
                target_y = max(0, min(SCREEN_H-1, int((iy / CAM_H) * SCREEN_H)))
                mouse.position = (target_x, target_y)

            if pincement and not clic_locked:
                mouse.click(Button.left)
                clic_locked = True
            elif not pincement:
                clic_locked = False

            _draw_hud(display, pts_disp, poing, pincement)

        cv2.imshow("Vision AI", display)

        if cv2.waitKey(1) == 27: # Esc
            break