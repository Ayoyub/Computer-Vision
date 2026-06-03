# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HAND DETECT — détection + Kalman 21 points + rendu liquid glass           ║
# ║                                                                            ║
# ║  LANDMARKS MÉDIAPIPE (21 points numérotés sur la main) :                  ║
# ║   0 = poignet                                                              ║
# ║   1-4  = pouce       (4  = bout)                                          ║
# ║   5-8  = index       (8  = bout)  ← curseur souris                        ║
# ║   9-12 = majeur      (12 = bout)                                          ║
# ║   13-16= annulaire   (16 = bout)                                          ║
# ║   17-20= auriculaire (20 = bout)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
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

# ── État inter-frames ────────────────────────────────────────────────────────────
_pinch_states   = {}   # hystérésis pincement par main

# Pool de filtres Kalman : dict[hand_idx] → liste de 21 KalmanCursor
# Un KalmanCursor par landmark, réutilisé frame après frame pour la continuité.
_kalman_pool    = {}


def _get_kalman(hand_idx):
    """Retourne (ou crée) les 21 filtres Kalman pour la main hand_idx."""
    if hand_idx not in _kalman_pool:
        # 21 filtres indépendants, un par landmark
        _kalman_pool[hand_idx] = [KalmanCursor() for _ in range(21)]
    return _kalman_pool[hand_idx]


def _apply_kalman(hand_idx, raw_points):
    """
    Filtre les 21 landmarks d'une main via leurs filtres Kalman individuels.

    Chaque point passe dans son propre KalmanCursor (position + vitesse 2D).
    Résultat : tremblements haute-fréquence atténués, mouvements réels conservés.

    Paramètres à ajuster dans kalman_filter.py :
      processNoiseCov    → ↑ plus réactif, ↓ plus lisse
      measurementNoiseCov → ↑ ignore plus MediaPipe, ↓ suit plus fidèlement
    """
    filters = _get_kalman(hand_idx)
    filtered = []
    for i, (x, y) in enumerate(raw_points):
        fx, fy = filters[i].update(x, y)
        filtered.append((fx, fy))
    return filtered


# ── Détection de gestes ──────────────────────────────────────────────────────────

def est_poing_ferme(points):
    """ Main fermée si tous les bouts de doigts (y compris le pouce) sont proches du poignet (distance < SEUIL_POING).
    La détection est plus robuste en vérifiant que chaque doigt est individuellement proche du poignet.
    """
    poignet = points[0]
    bouts = [points[i] for i in [4, 8, 12, 16, 20]]  # Inclure le pouce (4)
    # Vérifier que chaque bout de doigt est proche du poignet
    for b in bouts:
        dist = math.hypot(b[0] - poignet[0], b[1] - poignet[1])
        if dist > SEUIL_POING:
            return False
    return True


def est_pincement(points, hand_idx):
    """ Pincement pouce (4) + index (8) avec hystérésis :
    - Se déclenche quand dist < SEUIL_PINCEMENT_ON (22px)
    - Se relâche quand dist > SEUIL_PINCEMENT_OFF (32px)
    """
    tx, ty = points[4]
    ix, iy = points[8]
    dist = math.hypot(tx - ix, ty - iy)
    etat = _pinch_states.get(hand_idx, False)
    if not etat and dist < SEUIL_PINCEMENT_ON:
        _pinch_states[hand_idx] = True
    elif etat and dist > SEUIL_PINCEMENT_OFF:
        _pinch_states[hand_idx] = False
    return _pinch_states.get(hand_idx, False)


def rotation_3d(landmarks_raw):
    """
    Estime l'orientation 3D de la main depuis les coordonnées Z de MediaPipe.
    Retourne {'inclinaison', 'rotation', 'roulis'} en degrés.
    Utilisé par scene_3d.py pour piloter la rotation de l'objet.
    """
    p0  = landmarks_raw[0]
    p5  = landmarks_raw[5]
    p17 = landmarks_raw[17]

    dx, dy, dz = p5.x - p0.x, p5.y - p0.y, p5.z - p0.z
    lx, ly     = p17.x - p5.x, p17.y - p5.y

    return {
        'inclinaison': round(math.degrees(math.atan2(dy, math.hypot(dx, dz))), 1),
        'rotation'   : round(math.degrees(math.atan2(dz, dx)), 1),
        'roulis'     : round(math.degrees(math.atan2(ly, lx)), 1),
    }


# ── Rendu liquid glass ───────────────────────────────────────────────────────────

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

        cv2.circle(img, (px, py), r, (30, 30, 30),    -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, (180, 180, 180),  1, cv2.LINE_AA)
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


# ── API publique ─────────────────────────────────────────────────────────────────

def hand_detect(img):
    """
    Détecte les mains dans img (BGR).
    Applique le filtre de Kalman sur les 21 landmarks de chaque main.
    Dessine le rendu liquid glass.

    Retourne :
      img       : frame annotée
      hand_data : liste de dicts par main :
        {
          'points'   : 21 (x, y) filtrés par Kalman,
          'gestes'   : {'poing': bool, 'pincement': bool},
          'rotation' : {'inclinaison': float, 'rotation': float, 'roulis': float}
        }
    """
    h, w     = img.shape[:2]
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = int(time.time() * 1000)

    result    = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_data = []

    if not result.hand_landmarks:
        # Plus aucune main visible : on nettoie l'état inter-frames
        _pinch_states.clear()
        _kalman_pool.clear()
        return img, hand_data

    current_ids = set(range(len(result.hand_landmarks)))

    # Nettoie les filtres des mains qui ont disparu
    for stale in set(_kalman_pool.keys()) - current_ids:
        del _kalman_pool[stale]
        _pinch_states.pop(stale, None)

    for idx, raw_landmarks in enumerate(result.hand_landmarks):
        # Points bruts en pixels
        raw_points = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]

        # Kalman sur les 21 landmarks
        points = _apply_kalman(idx, raw_points)

        gestes = {
            'poing'    : est_poing_ferme(points),
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
    """Ferme proprement le landmarker MediaPipe."""
    landmarker.close()