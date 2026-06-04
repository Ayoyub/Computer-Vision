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

# ── MediaPipe landmarker (initialized once at module load) ───────────────────────
_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=DETECTION['num_hands'],
    )
)

# ── Skeleton connections (MediaPipe hand topology) ───────────────────────────────
# Landmark indices:
#   0=wrist  1-4=thumb  5-8=index  9-12=middle  13-16=ring  17-20=pinky
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),           # palm cross-connections
]
_FINGERTIPS = {4, 8, 12, 16, 20}

# ── Gesture thresholds (loaded from config) ──────────────────────────────────────
_SEUIL_POING = HAND['seuil_poing']
_PINCH_ON    = HAND['seuil_pincement_on']
_PINCH_OFF   = HAND['seuil_pincement_off']

# ── Per-hand inter-frame state ───────────────────────────────────────────────────
_pinch_states = {}          # {hand_idx: bool}  — pinch hysteresis
_kalman_pool  = {}          # {hand_idx: [KalmanCursor x 21]}


# ── Kalman helpers ───────────────────────────────────────────────────────────────

def _get_kalman(hand_idx: int) -> list:
    """Return (or create) the 21 Kalman filters for a given hand."""
    if hand_idx not in _kalman_pool:
        _kalman_pool[hand_idx] = [KalmanCursor() for _ in range(21)]
    return _kalman_pool[hand_idx]


def _apply_kalman(hand_idx: int, raw: list) -> list:
    """Filter all 21 landmarks of a hand through their individual Kalman filters."""
    return [_get_kalman(hand_idx)[i].update(x, y) for i, (x, y) in enumerate(raw)]


# ── Gesture detection ────────────────────────────────────────────────────────────

def _get_hand_scale(pts: list) -> float:
    """Calcule la taille de la main (poignet -> base du majeur) comme référence d'échelle."""
    scale = math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1])
    return scale if scale > 0 else 1.0

def est_poing_ferme(pts: list) -> bool:
    """
    Détection biomécanique : Vérifie si les bouts des doigts sont pliés 
    vers l'intérieur (plus proches du poignet que leurs jointures).
    """
    wrist = pts[0]
    # Couples : (Bout du doigt, Jointure du milieu)
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    
    folded_count = 0
    for tip, mid in fingers:
        d_tip = math.hypot(pts[tip][0] - wrist[0], pts[tip][1] - wrist[1])
        d_mid = math.hypot(pts[mid][0] - wrist[0], pts[mid][1] - wrist[1])
        
        # Si le bout du doigt s'est rapproché de la paume par rapport à l'articulation
        if d_tip < d_mid * 1.05: # Petite marge d'erreur de 5%
            folded_count += 1
            
    # On valide si au moins 3 des 4 doigts principaux sont pliés
    return folded_count >= 3



def est_pincement(pts: list, hand_idx: int) -> bool:
    """Détection du pincement avec hystérésis proportionnelle."""
    scale = _get_hand_scale(pts)
    dist = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1])
    ratio = dist / scale

    state = _pinch_states.get(hand_idx, False)
    
    # Ratios relatifs : < 0.25 = pincé, > 0.40 = relâché
    if not state and ratio < 0.25:
        _pinch_states[hand_idx] = True
    elif state and ratio > 0.40:
        _pinch_states[hand_idx] = False

    return _pinch_states.get(hand_idx, False)

def rotation_3d(raw_landmarks) -> dict:
    """
    Estimate 3D hand orientation from MediaPipe's relative Z coordinates.
    Returns angles in degrees: inclinaison (X), rotation (Y), roulis (Z).
    """
    p0, p5, p17 = raw_landmarks[0], raw_landmarks[5], raw_landmarks[17]
    dx = p5.x - p0.x;  dy = p5.y - p0.y;  dz = p5.z - p0.z
    lx = p17.x - p5.x; ly = p17.y - p5.y

    return {
        'inclinaison': round(math.degrees(math.atan2(dy, math.hypot(dx, dz))), 1),
        'rotation'   : round(math.degrees(math.atan2(dz, dx)), 1),
        'roulis'     : round(math.degrees(math.atan2(ly, lx)), 1),
    }


# ── Liquid glass renderer ────────────────────────────────────────────────────────

def _draw_hand(img, pts: list, gestes: dict):
    """Draw the hand skeleton and joints in a minimal liquid-glass style."""
    poing     = gestes['poing']
    pincement = gestes['pincement']

    # Skeleton lines — darker when fist is detected
    overlay = img.copy()
    color   = (130, 130, 130) if poing else (210, 210, 210)
    for s, e in _CONNECTIONS:
        cv2.line(overlay, pts[s], pts[e], color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # Joints
    for i, (px, py) in enumerate(pts):
        r = 5 if i in _FINGERTIPS else 3

        if not poing:
            # Diffuse halo (3 passes, decreasing radius)
            for hr, alpha in [(r+6, 0.06), (r+3, 0.10), (r+1, 0.14)]:
                ov = img.copy()
                cv2.circle(ov, (px, py), hr, (240, 240, 240), -1)
                cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

        cv2.circle(img, (px, py), r, (30, 30, 30),   -1, cv2.LINE_AA)  # dark base
        cv2.circle(img, (px, py), r, (180, 180, 180),  1, cv2.LINE_AA)  # pearlescent rim

        if not poing:
            # White glint offset — simulates glass refraction
            cv2.circle(img, (px-1, py-1), max(1, r-2), (255, 255, 255), -1, cv2.LINE_AA)

    # State indicators on index fingertip
    ix, iy = pts[8]
    if pincement:
        # Soft white halo = pinch active
        ov = img.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
        cv2.circle(img, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        # Small cross = fist locked, no action
        cv2.line(img, (ix-8, iy-8), (ix+8, iy+8), (140,140,140), 1, cv2.LINE_AA)
        cv2.line(img, (ix+8, iy-8), (ix-8, iy+8), (140,140,140), 1, cv2.LINE_AA)


# ── Public API ───────────────────────────────────────────────────────────────────

def hand_detect(img):
    """
    Detect hands in a BGR frame, apply per-landmark Kalman filtering,
    draw the liquid-glass overlay, and return annotated frame + hand data.

    Returns:
        img       : annotated BGR frame
        hand_data : list of dicts, one per detected hand:
            {
              'points'   : list of 21 (x, y) — Kalman-filtered pixel coords,
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
        _pinch_states.clear()
        _kalman_pool.clear()
        return img, []

    # Remove Kalman state for hands that disappeared this frame
    live = set(range(len(result.hand_landmarks)))
    for stale in set(_kalman_pool) - live:
        del _kalman_pool[stale]
        _pinch_states.pop(stale, None)

    hand_data = []
    for idx, raw_lm in enumerate(result.hand_landmarks):
        raw_pts = [(int(lm.x * w), int(lm.y * h)) for lm in raw_lm]
        pts     = _apply_kalman(idx, raw_pts)

        gestes = {
            'poing'    : est_poing_ferme(pts),
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
    """Cleanly close the MediaPipe landmarker."""
    _landmarker.close()