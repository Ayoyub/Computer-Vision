import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

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
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Téléchargement automatique du modèle ────────────────────────────────────────
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé.")

_ensure_model()

# ── Initialisation MediaPipe (fait une seule fois au chargement du module) ───────
# running_mode=VIDEO : le landmarker attend un timestamp croissant à chaque frame
# num_hands : nombre maximum de mains détectées simultanément
_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
_options = vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
landmarker = vision.HandLandmarker.create_from_options(_options)

# ── Connexions entre les 21 landmarks ───────────────────────────────────────────
# Chaque tuple (a, b) = segment entre le point a et le point b
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # majeur
    (0, 13), (13, 14), (14, 15), (15, 16),    # annulaire
    (0, 17), (17, 18), (18, 19), (19, 20),    # auriculaire
    (5, 9), (9, 13), (13, 17),                # paume (traverse horizontale)
]

# Indices des bouts de doigts — utilisés pour l'effet visuel et la détection poing
_FINGERTIP_IDS = {4, 8, 12, 16, 20}

# ── Seuils de détection de gestes ───────────────────────────────────────────────
# Distance moyenne (bouts des 4 doigts → poignet) en dessous de laquelle
# on considère que la main est fermée (poing).
# Exprimé en pixels sur le frame passé à hand_detect().
# ↑ Augmente si le poing est trop souvent raté
# ↓ Diminue si la main ouverte est détectée comme poing
SEUIL_POING = 75

# Distance pouce (4) ↔ index (8) en dessous de laquelle on détecte un pincement.
# Exprimé en pixels sur le frame passé à hand_detect().
# ↑ Augmente si le pincement est trop difficile à déclencher
# ↓ Diminue si le pincement se déclenche trop facilement
SEUIL_PINCEMENT = 22


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DÉTECTION DE GESTES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def est_poing_ferme(points):
    """
    Retourne True si la main est fermée (poing).

    Méthode : distance moyenne entre les 4 bouts de doigts (8, 12, 16, 20)
    et le poignet (0). Le pouce (4) est exclu car il reste visible même poing fermé.

    Utilisé pour bloquer tout geste quand le poing est fermé.
    """
    poignet = points[0]
    bouts   = [points[i] for i in [8, 12, 16, 20]]
    dist_moy = sum(
        math.hypot(b[0] - poignet[0], b[1] - poignet[1]) for b in bouts
    ) / 4
    return dist_moy < SEUIL_POING


def est_pincement(points):
    """
    Retourne True si le pouce (4) et l'index (8) sont pincés.

    Bloqué si la main est fermée (poing) pour éviter les faux clics.
    """
    if est_poing_ferme(points):
        return False
    tx, ty = points[4]
    ix, iy = points[8]
    return math.hypot(tx - ix, ty - iy) < SEUIL_PINCEMENT


def rotation_3d(landmarks_raw):
    """
    Estime l'orientation 3D de la main à partir des coordonnées Z normalisées
    fournies par MediaPipe (avant remap en pixels).

    Retourne un dict avec :
      - 'inclinaison'  : angle en degrés autour de l'axe X (main penchée vers/loin)
      - 'rotation'     : angle en degrés autour de l'axe Y (paume face / dos)
      - 'roulis'       : angle en degrés autour de l'axe Z (main inclinée gauche/droite)

    landmarks_raw = liste de objets avec attributs .x .y .z (valeurs 0.0–1.0 + z relatif)
    """
    # Vecteur poignet → base de l'index (axe longitudinal de la main)
    p0 = landmarks_raw[0]   # poignet
    p5 = landmarks_raw[5]   # base index
    p17= landmarks_raw[17]  # base auriculaire

    # Vecteur principal (poignet → base index)
    dx = p5.x - p0.x
    dy = p5.y - p0.y
    dz = p5.z - p0.z

    # Vecteur latéral (base index → base auriculaire)
    lx = p17.x - p5.x
    ly = p17.y - p5.y
    lz = p17.z - p5.z

    # Angles (en degrés) — math.atan2 retourne des radians
    inclinaison = math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    rotation    = math.degrees(math.atan2(dz, dx))
    roulis      = math.degrees(math.atan2(ly, lx))

    return {
        'inclinaison': round(inclinaison, 1),
        'rotation'   : round(rotation,    1),
        'roulis'     : round(roulis,       1),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RENDU LIQUID GLASS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _draw_liquid_hand(img, points, gestes):
    """
    Dessine la main sur img en style liquid glass monochrome.

    gestes = dict retourné par detect_gestes()
    """
    poing     = gestes['poing']
    pincement = gestes['pincement']

    # ── Connexions : traits fins gris semi-transparents ──────────────────────────
    overlay = img.copy()
    for s, e in _HAND_CONNECTIONS:
        # Opacité réduite si poing fermé (feedback visuel)
        couleur = (130, 130, 130) if poing else (210, 210, 210)
        cv2.line(overlay, points[s], points[e], couleur, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # ── Joints ───────────────────────────────────────────────────────────────────
    for i, (px, py) in enumerate(points):
        tip = i in _FINGERTIP_IDS
        r   = 5 if tip else 3

        # Pas de halo si poing fermé (allège le rendu)
        if not poing:
            for halo_r, alpha in [(r + 6, 0.06), (r + 3, 0.10), (r + 1, 0.14)]:
                ov = img.copy()
                cv2.circle(ov, (px, py), halo_r, (240, 240, 240), -1)
                cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        # Fond sombre du joint
        cv2.circle(img, (px, py), r, (30, 30, 30), -1, cv2.LINE_AA)
        # Contour nacré
        cv2.circle(img, (px, py), r, (180, 180, 180), 1, cv2.LINE_AA)
        # Éclat blanc (simuler la réfraction du verre)
        if not poing:
            cv2.circle(img, (px - 1, py - 1), max(1, r - 2),
                       (255, 255, 255), -1, cv2.LINE_AA)

    # ── Indicateur pincement ─────────────────────────────────────────────────────
    if pincement:
        ix, iy = points[8]
        ov = img.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
        cv2.circle(img, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Indicateur poing ─────────────────────────────────────────────────────────
    elif poing:
        ix, iy = points[8]
        cv2.line(img, (ix - 8, iy - 8), (ix + 8, iy + 8),
                 (140, 140, 140), 1, cv2.LINE_AA)
        cv2.line(img, (ix + 8, iy - 8), (ix - 8, iy + 8),
                 (140, 140, 140), 1, cv2.LINE_AA)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  API PUBLIQUE                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def detect_gestes(points):
    """
    À partir d'une liste de 21 points (x, y) en pixels, retourne un dict :
      - 'poing'     : bool — main fermée
      - 'pincement' : bool — pouce + index serrés (et main ouverte)
    """
    return {
        'poing'    : est_poing_ferme(points),
        'pincement': est_pincement(points),
    }


def hand_detect(img):
    """
    Détecte les mains dans img (BGR), dessine le rendu liquid glass,
    et retourne :
      - img modifié
      - hand_data : liste de dicts, un par main détectée :
            {
              'points'   : liste de 21 (x, y) en pixels,
              'gestes'   : {'poing': bool, 'pincement': bool},
              'rotation' : {'inclinaison': float, 'rotation': float, 'roulis': float}
            }
    """
    h, w = img.shape[:2]
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = int(time.time() * 1000)

    result   = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_data = []

    if result.hand_landmarks:
        for raw_landmarks in result.hand_landmarks:
            # Points en pixels (pour le dessin et les calculs de distance)
            points = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]

            gestes = detect_gestes(points)
            rot    = rotation_3d(raw_landmarks)

            hand_data.append({
                'points'  : points,
                'gestes'  : gestes,
                'rotation': rot,
            })

            _draw_liquid_hand(img, points, gestes)

    return img, hand_data


def release():
    """Ferme proprement le landmarker MediaPipe. Appeler en fin de programme."""
    landmarker.close()