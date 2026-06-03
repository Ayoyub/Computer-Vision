import cv2
import pyautogui
from hand_detect import hand_detect, release
from config import CAM

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

ECRAN_W, ECRAN_H = pyautogui.size()
CAM_W, CAM_H     = CAM['detect_w'], CAM['detect_h']

clic_verrouille = False

def draw_hud(display, points, poing, pincement):
    h, w = display.shape[:2]
    ix, iy = points[8]
    
    if pincement:
        ov = display.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.35, display, 0.65, 0, display)
        cv2.circle(display, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        cv2.line(display, (ix - 8, iy - 8), (ix + 8, iy + 8), (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(display, (ix + 8, iy - 8), (ix - 8, iy + 8), (160, 160, 160), 1, cv2.LINE_AA)

    if poing: etat = "poing  — aucune action"
    elif pincement: etat = "clic"
    else: etat = "déplacement"

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    (tw, th), _ = cv2.getTextSize(etat, font, scale, 1)

    bx, by = 12, h - 20
    ov = display.copy()
    cv2.rectangle(ov, (bx - 6, by - th - 6), (bx + tw + 6, by + 6), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.50, display, 0.50, 0, display)
    cv2.putText(display, etat, (bx, by), font, scale, (200, 200, 200), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(CAM['source'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM['display_w'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM['display_h'])
cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAM['buffer_size'])

print(f"Écran : {ECRAN_W}×{ECRAN_H}  |  Échap pour quitter")

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (CAM_W, CAM_H))
    
    # Toute la magie de lissage et d'hystérésis se passe ici maintenant
    display, hand_data = hand_detect(small)

    if hand_data:
        points = hand_data[0]['points']
        poing = hand_data[0]['gestes']['poing']
        pincement = hand_data[0]['gestes']['pincement']

        # ── Déplacement (Directement avec les coordonnées lissées de MediaPipe) ──
        if not poing:
            ix, iy = points[8]
            cible_x = int((ix / CAM_W) * ECRAN_W)
            cible_y = int((iy / CAM_H) * ECRAN_H)
            
            # Sécurité bords d'écran
            cible_x = max(0, min(ECRAN_W - 1, cible_x))
            cible_y = max(0, min(ECRAN_H - 1, cible_y))
            
            pyautogui.moveTo(cible_x, cible_y)

        # ── Clic gauche (Déclenché sur le front montant du pincement stabilisé) ──
        if pincement and not clic_verrouille:
            pyautogui.click()
            clic_verrouille = True
        elif not pincement:
            clic_verrouille = False

        draw_hud(display, points, poing, pincement)

    display = cv2.resize(display, (CAM['display_w'], CAM['display_h']))
    cv2.imshow("Souris Virtuelle", display)

    if cv2.waitKey(1) == 27: break

cap.release()
release()
cv2.destroyAllWindows()