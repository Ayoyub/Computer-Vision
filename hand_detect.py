# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HAND DETECT — détection + rendu liquid glass                              ║
# ║                                                                            ║
# ║  LANDMARKS MÉDIAPIPE (21 points numérotés sur la main) :                  ║
# ║                                                                            ║
# ║   0 = poignet                                                              ║
# ║   1-4  = pouce       (4  = bout)                                          ║
# ║   5-8  = index       (8  = bout)  ← curseur souris                        ║
# ║   9-12 = majeur      (12 = bout)                                          ║
# ║   13-16= annulaire   (16 = bout)                                          ║
# ║   17-20= auriculaire (20 = bout)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝import cv2
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
from config import HAND, DETECTION
from kalman_filter import KalmanCursor

MODEL_PATH = DETECTION['model_path']
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

SEUIL_POING         = HAND['seuil_poing']
SEUIL_PINCEMENT_ON  = HAND['seuil_pincement_on']
SEUIL_PINCEMENT_OFF = HAND['seuil_pincement_off']

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé.")

_ensure_model()

_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
_options = vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=DETECTION['num_hands']
)
landmarker = vision.HandLandmarker.create_from_options(_options)

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
_FINGERTIP_IDS = {4, 8, 12, 16, 20}

# ── Variables d'état globales pour la mémoire inter-frames ──────────────────
_pinch_states = {}
_kalman_filters = {}

def est_poing_ferme(points):
    poignet = points[0]
    bouts   = [points[i] for i in [8, 12, 16, 20]]
    dist_moy = sum(math.hypot(b[0] - poignet[0], b[1] - poignet[1]) for b in bouts) / 4
    return dist_moy < SEUIL_POING

def est_pincement(points, hand_idx):
    if est_poing_ferme(points):
        _pinch_states[hand_idx] = False
        return False
        
    tx, ty = points[4]
    ix, iy = points[8]
    dist = math.hypot(tx - ix, ty - iy)

    etat_actuel = _pinch_states.get(hand_idx, False)

    # Logique d'hystérésis
    if not etat_actuel and dist < SEUIL_PINCEMENT_ON:
        _pinch_states[hand_idx] = True
    elif etat_actuel and dist > SEUIL_PINCEMENT_OFF:
        _pinch_states[hand_idx] = False

    return _pinch_states.get(hand_idx, False)

def rotation_3d(landmarks_raw):
    p0 = landmarks_raw[0]
    p5 = landmarks_raw[5]
    p17= landmarks_raw[17]

    dx, dy, dz = p5.x - p0.x, p5.y - p0.y, p5.z - p0.z
    lx, ly, lz = p17.x - p5.x, p17.y - p5.y, p17.z - p5.z

    inclinaison = math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    rotation    = math.degrees(math.atan2(dz, dx))
    roulis      = math.degrees(math.atan2(ly, lx))

    return {
        'inclinaison': round(inclinaison, 1),
        'rotation'   : round(rotation, 1),
        'roulis'     : round(roulis, 1),
    }

def _draw_liquid_hand(img, points, gestes):
    poing     = gestes['poing']
    pincement = gestes['pincement']

    overlay = img.copy()
    for s, e in _HAND_CONNECTIONS:
        couleur = (130, 130, 130) if poing else (210, 210, 210)
        cv2.line(overlay, points[s], points[e], couleur, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    for i, (px, py) in enumerate(points):
        tip = i in _FINGERTIP_IDS
        r   = 5 if tip else 3

        if not poing:
            for halo_r, alpha in [(r + 6, 0.06), (r + 3, 0.10), (r + 1, 0.14)]:
                ov = img.copy()
                cv2.circle(ov, (px, py), halo_r, (240, 240, 240), -1)
                cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        cv2.circle(img, (px, py), r, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, (180, 180, 180), 1, cv2.LINE_AA)
        if not poing:
            cv2.circle(img, (px - 1, py - 1), max(1, r - 2), (255, 255, 255), -1, cv2.LINE_AA)

    if pincement:
        ix, iy = points[8]
        ov = img.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
        cv2.circle(img, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        ix, iy = points[8]
        cv2.line(img, (ix - 8, iy - 8), (ix + 8, iy + 8), (140, 140, 140), 1, cv2.LINE_AA)
        cv2.line(img, (ix + 8, iy - 8), (ix - 8, iy + 8), (140, 140, 140), 1, cv2.LINE_AA)

def hand_detect(img):
    h, w = img.shape[:2]
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_data = []

    # Reset les filtres si aucune main n'est présente à l'écran
    if not result.hand_landmarks:
        _pinch_states.clear()
        _kalman_filters.clear()

    if result.hand_landmarks:
        for idx, raw_landmarks in enumerate(result.hand_landmarks):
            if idx not in _kalman_filters:
                _kalman_filters[idx] = KalmanCursor()

            points = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]

            # ── APPLICATION DU FILTRE DE KALMAN SUR L'INDEX ──
            kx, ky = _kalman_filters[idx].update(points[8][0], points[8][1])
            points[8] = (kx, ky) # Remplacement par la coordonnée lissée

            gestes = {
                'poing': est_poing_ferme(points),
                'pincement': est_pincement(points, idx),
            }
            rot = rotation_3d(raw_landmarks)

            hand_data.append({
                'points'  : points,
                'gestes'  : gestes,
                'rotation': rot,
            })

            _draw_liquid_hand(img, points, gestes)

    return img, hand_data

def release():
    landmarker.close()