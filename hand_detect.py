import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
from collections import deque
from config import HAND, DETECTION
from kalman_filter import KalmanCursor

# ── Model ────────────────────────────────────────────────────────────────────────
MODEL_PATH = DETECTION['model_path']
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

_ensure_model()

# ── MediaPipe landmarker ─────────────────────────────────────────────────────────
_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=DETECTION['num_hands'],
        min_hand_detection_confidence=0.6,   # filtre les détections douteuses
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
)

# ── Squelette (topologie MediaPipe) ─────────────────────────────────────────────
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_FINGERTIPS = {4, 8, 12, 16, 20}

# ── Seuils depuis config ─────────────────────────────────────────────────────────
_PINCH_ON  = HAND['seuil_pincement_on']
_PINCH_OFF = HAND['seuil_pincement_off']

# ── État inter-frames par main ───────────────────────────────────────────────────
_pinch_states  = {}   # {idx: bool}
_kalman_pool   = {}   # {idx: [KalmanCursor x 21]}

# ── Débouncing temporel des gestes ──────────────────────────────────────────────
# Un geste doit être détecté N frames consécutives pour être validé,
# et absent N frames pour être annulé.
# Valeurs asymétriques : on active rapidement, on désactive plus lentement
# (évite les scintillements lors des transitions).
_DEBOUNCE_ON  = 2   # frames pour activer  (réactivité)
_DEBOUNCE_OFF = 4   # frames pour désactiver (stabilité)

_poing_history   = {}   # {idx: deque de bools}
_pinch_history   = {}   # {idx: deque de bools}


def _get_history(store: dict, idx: int, maxlen: int) -> deque:
    if idx not in store:
        store[idx] = deque(maxlen=maxlen)
    return store[idx]


def _debounce(history: deque, raw: bool,
              on_thresh: int, off_thresh: int, current: bool) -> bool:
    """
    Applique un débounce asymétrique sur un signal booléen.

    - Si current=False et les `on_thresh` dernières valeurs sont True  → active
    - Si current=True  et les `off_thresh` dernières valeurs sont False → désactive
    - Sinon → inchangé
    """
    history.append(raw)
    if not current and len(history) >= on_thresh:
        if all(list(history)[-on_thresh:]):
            return True
    if current and len(history) >= off_thresh:
        if not any(list(history)[-off_thresh:]):
            return False
    return current


# ── Kalman helpers ───────────────────────────────────────────────────────────────

def _get_kalman(idx: int) -> list:
    if idx not in _kalman_pool:
        _kalman_pool[idx] = [KalmanCursor() for _ in range(21)]
    return _kalman_pool[idx]


def _apply_kalman(idx: int, raw_float: list, hand_scale: float) -> list:
    """
    Filtre les 21 landmarks en virgule flottante.
    raw_float : liste de (x_float, y_float) en pixels (pas encore int).
    Retourne des (int, int) pour la compatibilité avec le reste du code.
    """
    filters = _get_kalman(idx)
    result  = []
    for i, (x, y) in enumerate(raw_float):
        fx, fy = filters[i].update(x, y, hand_scale)
        result.append((int(round(fx)), int(round(fy))))
    return result


# ── Calcul de la taille de la main ──────────────────────────────────────────────

def _hand_scale(pts) -> float:
    """
    Distance poignet (0) → base majeur (9), en pixels.
    Sert de référence d'échelle pour normaliser tous les seuils.
    Accepte des tuples int ou float.
    """
    dx = pts[9][0] - pts[0][0]
    dy = pts[9][1] - pts[0][1]
    s  = math.hypot(dx, dy)
    return s if s > 1.0 else 1.0


# ── Détection de gestes ──────────────────────────────────────────────────────────

def _est_poing_brut(pts: list) -> bool:
    """
    Détection biomécanique stricte d'un poing fermé.

    Deux conditions par doigt (index/majeur/annulaire/auriculaire) :
    1. Le bout du doigt est plus proche du poignet que l'articulation médiane
       (le doigt est replié, pas simplement baissé).
    2. La distance bout→base est inférieure à 55% de la paume
       (le doigt est serré, pas juste fléchi à moitié).

    Validé si ≥ 3 doigts sur 4 satisfont les deux conditions.
    Le pouce est délibérément exclu : sa géométrie est trop variable
    selon l'orientation de la main.
    """
    wrist = pts[0]
    scale = _hand_scale(pts)

    fingers = [
        (8,  6,  5),   # index
        (12, 10, 9),   # majeur
        (16, 14, 13),  # annulaire
        (20, 18, 17),  # auriculaire
    ]

    folded = 0
    for tip, mid, base in fingers:
        d_tip_wrist = math.hypot(pts[tip][0] - wrist[0], pts[tip][1] - wrist[1])
        d_mid_wrist = math.hypot(pts[mid][0] - wrist[0], pts[mid][1] - wrist[1])
        d_tip_base  = math.hypot(pts[tip][0] - pts[base][0],
                                  pts[tip][1] - pts[base][1])

        is_curled = d_tip_wrist < d_mid_wrist * 0.90
        is_short  = d_tip_base  < scale * 0.55

        if is_curled and is_short:
            folded += 1

    return folded >= 3


def _est_pincement_brut(pts: list, hand_idx: int) -> bool:
    """
    Détection du pincement pouce-index avec hystérésis proportionnelle.
    Le ratio distance/taille_main normalise automatiquement selon la distance caméra.
    """
    scale = _hand_scale(pts)
    dist  = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1])
    ratio = dist / scale

    state = _pinch_states.get(hand_idx, False)
    if not state and ratio < 0.25:
        _pinch_states[hand_idx] = True
    elif state and ratio > 0.40:
        _pinch_states[hand_idx] = False

    return _pinch_states.get(hand_idx, False)


# ── États déboncés (persistants inter-frames) ────────────────────────────────────
_poing_debounced = {}   # {idx: bool}
_pinch_debounced = {}   # {idx: bool}


def est_poing_ferme(pts: list, idx: int) -> bool:
    """Poing avec débounce temporel — élimine les faux positifs ponctuels."""
    raw     = _est_poing_brut(pts)
    hist    = _get_history(_poing_history, idx, max(_DEBOUNCE_ON, _DEBOUNCE_OFF))
    current = _poing_debounced.get(idx, False)
    result  = _debounce(hist, raw, _DEBOUNCE_ON, _DEBOUNCE_OFF, current)
    _poing_debounced[idx] = result
    return result


def est_pincement(pts: list, idx: int) -> bool:
    """Pincement avec hystérésis proportionnelle + débounce temporel."""
    raw     = _est_pincement_brut(pts, idx)
    hist    = _get_history(_pinch_history, idx, max(_DEBOUNCE_ON, _DEBOUNCE_OFF))
    current = _pinch_debounced.get(idx, False)
    result  = _debounce(hist, raw, _DEBOUNCE_ON, _DEBOUNCE_OFF, current)
    _pinch_debounced[idx] = result
    return result


def rotation_3d(raw_landmarks) -> dict:
    """
    Estimation de l'orientation 3D depuis les coordonnées Z relatives de MediaPipe.
    Retourne les angles en degrés : inclinaison (X), rotation (Y), roulis (Z).
    """
    p0, p5, p17 = raw_landmarks[0], raw_landmarks[5], raw_landmarks[17]
    dx = p5.x - p0.x;  dy = p5.y - p0.y;  dz = p5.z - p0.z
    lx = p17.x - p5.x; ly = p17.y - p5.y

    return {
        'inclinaison': round(math.degrees(math.atan2(dy, math.hypot(dx, dz))), 1),
        'rotation'   : round(math.degrees(math.atan2(dz, dx)), 1),
        'roulis'     : round(math.degrees(math.atan2(ly, lx)), 1),
    }


# ── Rendu liquid-glass ───────────────────────────────────────────────────────────

def _draw_hand(img, pts: list, gestes: dict):
    """Dessine le squelette et les articulations dans un style liquid-glass minimal."""
    poing     = gestes['poing']
    pincement = gestes['pincement']

    # Lignes du squelette
    overlay = img.copy()
    color   = (130, 130, 130) if poing else (210, 210, 210)
    for s, e in _CONNECTIONS:
        cv2.line(overlay, pts[s], pts[e], color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # Articulations
    for i, (px, py) in enumerate(pts):
        r = 5 if i in _FINGERTIPS else 3

        if not poing:
            for hr, alpha in [(r+6, 0.06), (r+3, 0.10), (r+1, 0.14)]:
                ov = img.copy()
                cv2.circle(ov, (px, py), hr, (240, 240, 240), -1)
                cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

        cv2.circle(img, (px, py), r, (30, 30, 30),    -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, (180, 180, 180),   1, cv2.LINE_AA)

        if not poing:
            cv2.circle(img, (px-1, py-1), max(1, r-2), (255, 255, 255), -1, cv2.LINE_AA)

    # Indicateurs d'état sur le bout de l'index
    ix, iy = pts[8]
    if pincement:
        ov = img.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
        cv2.circle(img, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        cv2.line(img, (ix-8, iy-8), (ix+8, iy+8), (140, 140, 140), 1, cv2.LINE_AA)
        cv2.line(img, (ix+8, iy-8), (ix-8, iy+8), (140, 140, 140), 1, cv2.LINE_AA)


# ── API publique ─────────────────────────────────────────────────────────────────

def hand_detect(img):
    """
    Détecte les mains dans une frame BGR, applique le filtre de Kalman adaptatif,
    dessine l'overlay liquid-glass, et retourne la frame annotée + les données.

    Retourne :
        img       : frame BGR annotée
        hand_data : liste de dicts, un par main détectée :
            {
              'points'   : liste de 21 (int, int) — coords filtrées,
              'gestes'   : {'poing': bool, 'pincement': bool},
              'rotation' : {'inclinaison': float, 'rotation': float, 'roulis': float}
            }
    """
    h, w = img.shape[:2]
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    result = _landmarker.detect_for_video(mp_img, int(time.time() * 1000))

    if not result.hand_landmarks:
        # Nettoyage de tous les états quand plus aucune main n'est visible
        _pinch_states.clear()
        _kalman_pool.clear()
        _poing_debounced.clear()
        _pinch_debounced.clear()
        _poing_history.clear()
        _pinch_history.clear()
        return img, []

    # Supprime l'état des mains qui ont disparu cette frame
    live = set(range(len(result.hand_landmarks)))
    for stale in set(_kalman_pool) - live:
        del _kalman_pool[stale]
        _pinch_states.pop(stale, None)
        _poing_debounced.pop(stale, None)
        _pinch_debounced.pop(stale, None)
        _poing_history.pop(stale, None)
        _pinch_history.pop(stale, None)

    hand_data = []
    for idx, raw_lm in enumerate(result.hand_landmarks):
        # ── Coordonnées en virgule flottante AVANT conversion int ────────────────
        # FIX vs version précédente : on filtrait des int, perdant la sub-pixel
        # précision avant même que Kalman puisse travailler.
        raw_float = [(lm.x * w, lm.y * h) for lm in raw_lm]

        # Calcul de l'échelle sur les coords brutes (pas encore filtrées)
        scale = math.hypot(
            raw_float[9][0] - raw_float[0][0],
            raw_float[9][1] - raw_float[0][1]
        )
        scale = max(scale, 1.0)

        # Filtrage Kalman adaptatif avec échelle de la main
        pts = _apply_kalman(idx, raw_float, scale)

        gestes = {
            'poing'    : est_poing_ferme(pts, idx),
            'pincement': est_pincement(pts, idx),
        }
        hand_data.append({
            'points'  : pts,
            'gestes'  : gestes,
            'rotation': rotation_3d(raw_lm),
        })
        _draw_hand(img, pts, gestes)

    return img, hand_data


def release():
    """Ferme proprement le landmarker MediaPipe."""
    _landmarker.close()