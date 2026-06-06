import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import numpy as np
from collections import deque
from config import HAND, DETECTION
from kalman_filter import KalmanCursor
import td_render as td

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

_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=DETECTION['num_hands'],
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
)

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_FINGERTIPS  = {4, 8, 12, 16, 20}
_PALM_JOINTS = {0, 1, 5, 9, 13, 17}

_PINCH_ON  = HAND['seuil_pincement_on']
_PINCH_OFF = HAND['seuil_pincement_off']

_pinch_states    = {}
_kalman_pool     = {}
_DEBOUNCE_ON     = 2
_DEBOUNCE_OFF    = 4
_poing_history   = {}
_pinch_history   = {}
_poing_debounced = {}
_pinch_debounced = {}


def _get_history(store, idx, maxlen):
    if idx not in store:
        store[idx] = deque(maxlen=maxlen)
    return store[idx]


def _debounce(history, raw, on_thresh, off_thresh, current):
    history.append(raw)
    if not current and len(history) >= on_thresh:
        if all(list(history)[-on_thresh:]):
            return True
    if current and len(history) >= off_thresh:
        if not any(list(history)[-off_thresh:]):
            return False
    return current


def _get_kalman(idx):
    if idx not in _kalman_pool:
        _kalman_pool[idx] = [KalmanCursor() for _ in range(21)]
    return _kalman_pool[idx]


def _apply_kalman(idx, raw_float, hand_scale):
    filters = _get_kalman(idx)
    result  = []
    for i, (x, y) in enumerate(raw_float):
        fx, fy = filters[i].update(x, y, hand_scale)
        result.append((int(round(fx)), int(round(fy))))
    return result


def _hand_scale(pts):
    dx = pts[9][0] - pts[0][0]
    dy = pts[9][1] - pts[0][1]
    s  = math.hypot(dx, dy)
    return s if s > 1.0 else 1.0


def _est_poing_brut(pts):
    wrist = pts[0]
    scale = _hand_scale(pts)
    fingers = [(8,6,5),(12,10,9),(16,14,13),(20,18,17)]
    folded = 0
    for tip, mid, base in fingers:
        d_tip_wrist = math.hypot(pts[tip][0]-wrist[0], pts[tip][1]-wrist[1])
        d_mid_wrist = math.hypot(pts[mid][0]-wrist[0], pts[mid][1]-wrist[1])
        d_tip_base  = math.hypot(pts[tip][0]-pts[base][0], pts[tip][1]-pts[base][1])
        if d_tip_wrist < d_mid_wrist * 0.90 and d_tip_base < scale * 0.55:
            folded += 1
    return folded >= 3


def _est_pincement_brut(pts, hand_idx):
    scale = _hand_scale(pts)
    dist  = math.hypot(pts[4][0]-pts[8][0], pts[4][1]-pts[8][1])
    ratio = dist / scale
    state = _pinch_states.get(hand_idx, False)
    if not state and ratio < 0.25:
        _pinch_states[hand_idx] = True
    elif state and ratio > 0.40:
        _pinch_states[hand_idx] = False
    return _pinch_states.get(hand_idx, False)


def est_poing_ferme(pts, idx):
    raw     = _est_poing_brut(pts)
    hist    = _get_history(_poing_history, idx, max(_DEBOUNCE_ON, _DEBOUNCE_OFF))
    current = _poing_debounced.get(idx, False)
    result  = _debounce(hist, raw, _DEBOUNCE_ON, _DEBOUNCE_OFF, current)
    _poing_debounced[idx] = result
    return result


def est_pincement(pts, idx):
    raw     = _est_pincement_brut(pts, idx)
    hist    = _get_history(_pinch_history, idx, max(_DEBOUNCE_ON, _DEBOUNCE_OFF))
    current = _pinch_debounced.get(idx, False)
    result  = _debounce(hist, raw, _DEBOUNCE_ON, _DEBOUNCE_OFF, current)
    _pinch_debounced[idx] = result
    return result


def rotation_3d(raw_landmarks):
    p0, p5, p17 = raw_landmarks[0], raw_landmarks[5], raw_landmarks[17]
    dx = p5.x-p0.x; dy = p5.y-p0.y; dz = p5.z-p0.z
    lx = p17.x-p5.x; ly = p17.y-p5.y
    return {
        'inclinaison': round(math.degrees(math.atan2(dy, math.hypot(dx,dz))), 1),
        'rotation'   : round(math.degrees(math.atan2(dz, dx)), 1),
        'roulis'     : round(math.degrees(math.atan2(ly, lx)), 1),
    }


# ── TouchDesigner-style hand renderer ────────────────────────────────────────────

def _draw_hand_td(img, pts, gestes):
    """
    Renders the hand skeleton in TouchDesigner aesthetic:
    - Hair-thin lines on a float32 emission layer
    - Gaussian bloom pass
    - Additive composite onto the (already-dimmed) frame
    """
    h, w   = img.shape[:2]
    poing  = gestes['poing']
    pinch  = gestes['pincement']

    em = td.make_emission(h, w)

    # ── Skeleton lines ────────────────────────────────────────────────────────────
    # Palm connections slightly dimmer than finger bones
    for s, e in _CONNECTIONS:
        is_palm = (s in _PALM_JOINTS and e in _PALM_JOINTS)
        color   = td.TD_BLUE if (is_palm or poing) else td.TD_CYAN
        # Scale color down slightly for a more ghostly feel
        td.draw_line_em(em, pts[s], pts[e],
                        color=(color * (0.5 if poing else 0.8)))

    # ── Joints ────────────────────────────────────────────────────────────────────
    for i, pt in enumerate(pts):
        if i in _FINGERTIPS:
            td.draw_node_em(em, pt, r=3,
                            core=td.TD_WHITE * (0.3 if poing else 1.0),
                            rim=td.TD_CYAN  * (0.3 if poing else 0.9))
        elif i in _PALM_JOINTS:
            td.draw_node_em(em, pt, r=2,
                            core=td.TD_WHITE * 0.4,
                            rim=td.TD_BLUE)
        else:
            td.draw_node_em(em, pt, r=1,
                            core=td.TD_WHITE * 0.3,
                            rim=td.TD_BLUE   * 0.6)

    # ── Fingertip state indicator ─────────────────────────────────────────────────
    td.draw_fingertip_em(em, pts[8], pinch=pinch, fist=poing)

    # ── Bloom & composite ─────────────────────────────────────────────────────────
    bloomed = td.bloom(em, kernel=15, sigma=6.0, strength=0.9)

    # Additive blend onto img (img is already float32 0-1 at this point,
    # OR we do the blend in uint8 space carefully)
    img_f   = img.astype(np.float32) / 255.0
    result  = np.clip(img_f + bloomed, 0, 1)
    img[:]  = (result * 255).astype(np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────────

def hand_detect(img):
    """
    Detect hands on the original bright frame, then apply TD dimming + skeleton
    only to the display copy.
    Returns (annotated_display_frame, hand_data).
    """
    h, w = img.shape[:2]

    # ── Detection on the ORIGINAL frame (full brightness) ────────────────────────
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    result = _landmarker.detect_for_video(mp_img, int(time.time() * 1000))

    # ── Dim the frame for display AFTER detection ─────────────────────────────────
    img_f  = img.astype(np.float32) / 255.0
    img_f  = img_f * td.CAM_DIM + td.BG_TINT
    img[:] = np.clip(img_f * 255, 0, 255).astype(np.uint8)

    if not result.hand_landmarks:
        _pinch_states.clear(); _kalman_pool.clear()
        _poing_debounced.clear(); _pinch_debounced.clear()
        _poing_history.clear();  _pinch_history.clear()
        return img, []

    live = set(range(len(result.hand_landmarks)))
    for stale in set(_kalman_pool) - live:
        del _kalman_pool[stale]
        for d in (_pinch_states, _poing_debounced, _pinch_debounced,
                  _poing_history, _pinch_history):
            d.pop(stale, None)

    hand_data = []
    for idx, raw_lm in enumerate(result.hand_landmarks):
        raw_float = [(lm.x * w, lm.y * h) for lm in raw_lm]
        scale = max(math.hypot(
            raw_float[9][0]-raw_float[0][0],
            raw_float[9][1]-raw_float[0][1]), 1.0)
        pts    = _apply_kalman(idx, raw_float, scale)
        gestes = {
            'poing'    : est_poing_ferme(pts, idx),
            'pincement': est_pincement(pts, idx),
        }
        hand_data.append({
            'points'  : pts,
            'gestes'  : gestes,
            'rotation': rotation_3d(raw_lm),
        })
        _draw_hand_td(img, pts, gestes)

    return img, hand_data


def release():
    _landmarker.close()