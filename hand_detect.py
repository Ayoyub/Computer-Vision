import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

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
    num_hands=2
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


def _draw_liquid_hand(img, points):
    # ── Connexions : traits fins gris semi-transparents ──
    overlay = img.copy()
    for s, e in _HAND_CONNECTIONS:
        cv2.line(overlay, points[s], points[e], (210, 210, 210), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # ── Joints ──
    for i, (px, py) in enumerate(points):
        tip = i in _FINGERTIP_IDS
        r   = 5 if tip else 3

        # Halo diffus (3 passes d'alpha décroissant)
        for halo_r, alpha in [(r + 6, 0.06), (r + 3, 0.10), (r + 1, 0.14)]:
            ov = img.copy()
            cv2.circle(ov, (px, py), halo_r, (240, 240, 240), -1)
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        # Fond sombre
        cv2.circle(img, (px, py), r, (30, 30, 30), -1, cv2.LINE_AA)
        # Contour nacré
        cv2.circle(img, (px, py), r, (180, 180, 180), 1, cv2.LINE_AA)
        # Éclat blanc décalé (effet verre)
        cv2.circle(img, (px - 1, py - 1), max(1, r - 2), (255, 255, 255), -1, cv2.LINE_AA)


def hand_detect(img):
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_data = []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            hand_data.append(points)
            _draw_liquid_hand(img, points)

    return img, hand_data


def release():
    landmarker.close()