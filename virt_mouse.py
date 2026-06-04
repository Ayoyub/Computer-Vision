import cv2
import pyautogui
from hand_detect import hand_detect, release
from config import CAM

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0          # remove pyautogui's default 0.1 s delay

SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W,    CAM_H    = CAM['detect_w'], CAM['detect_h']

clic_locked = False             # prevents holding a pinch from firing many clicks


def _draw_hud(display, pts, poing, pincement):
    """Minimal HUD: state label + index fingertip indicator."""
    h = display.shape[0]
    ix, iy = pts[8]

    if pincement:
        # Soft white ring = click active
        ov = display.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.35, display, 0.65, 0, display)
        cv2.circle(display, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        # Small cross = fist, no action
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


# ── Camera ───────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM['source'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM['display_w'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM['display_h'])
cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAM['buffer_size'])

print(f"Screen: {SCREEN_W}×{SCREEN_H}  |  Esc to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame   = cv2.flip(frame, 1)
    small   = cv2.resize(frame, (CAM_W, CAM_H))
    display, hand_data = hand_detect(small)

    if hand_data:
        pts       = hand_data[0]['points']
        poing     = hand_data[0]['gestes']['poing']
        pincement = hand_data[0]['gestes']['pincement']

        # ── Cursor movement — driven by Kalman-filtered index fingertip ──────────
        if not poing:
            ix, iy  = pts[8]
            target_x = max(0, min(SCREEN_W-1, int((ix / CAM_W) * SCREEN_W)))
            target_y = max(0, min(SCREEN_H-1, int((iy / CAM_H) * SCREEN_H)))
            pyautogui.moveTo(target_x, target_y)

        # ── Left click — fires only on the rising edge of the pinch ─────────────
        if pincement and not clic_locked:
            pyautogui.click()
            clic_locked = True
        elif not pincement:
            clic_locked = False

        _draw_hud(display, pts, poing, pincement)

    display = cv2.resize(display, (CAM['display_w'], CAM['display_h']))
    cv2.imshow("Virtual Mouse", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
release()
cv2.destroyAllWindows()