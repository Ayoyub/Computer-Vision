# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SCENE_3D.PY — Contrôle d'objet 3D par la main                            ║
# ║                                                                            ║
# ║  Gestes :                                                                  ║
# ║    Main ouverte  → rotation de l'objet (roulis/inclinaison de la main)    ║
# ║    Poing fermé   → pause (l'objet garde sa dernière orientation)           ║
# ║    Pincement     → cycle entre les objets (cube / octaèdre / icosaèdre)   ║
# ║    Touche Échap  → quitter                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import cv2
import numpy as np
import math
import time
import threading
from hand_detect import hand_detect, release
from config import CAM

# ── Géométrie des objets 3D ──────────────────────────────────────────────────────
# Chaque objet est défini par ses sommets et ses arêtes.
# Les coordonnées sont normalisées dans [-1, 1].

def _make_cube():
    verts = np.array([
        [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
        [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # face arrière
        (4,5),(5,6),(6,7),(7,4),   # face avant
        (0,4),(1,5),(2,6),(3,7),   # arêtes latérales
    ]
    return verts, edges

def _make_octahedron():
    verts = np.array([
        [ 0, 0, 1.4], [ 0, 0,-1.4],
        [ 1, 0, 0  ], [-1, 0, 0  ],
        [ 0, 1, 0  ], [ 0,-1, 0  ],
    ], dtype=float)
    edges = [
        (0,2),(0,3),(0,4),(0,5),
        (1,2),(1,3),(1,4),(1,5),
        (2,4),(4,3),(3,5),(5,2),
    ]
    return verts, edges

def _make_icosahedron():
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [ 0, 1, phi],[ 0,-1, phi],[ 0, 1,-phi],[ 0,-1,-phi],
        [ 1, phi, 0],[-1, phi, 0],[ 1,-phi, 0],[-1,-phi, 0],
        [ phi, 0, 1],[ phi, 0,-1],[-phi, 0, 1],[-phi, 0,-1],
    ], dtype=float)
    edges = [
        (0,1),(0,4),(0,5),(0,8),(0,10),
        (1,6),(1,7),(1,8),(1,10),
        (2,3),(2,4),(2,5),(2,9),(2,11),
        (3,6),(3,7),(3,9),(3,11),
        (4,5),(4,8),(4,9),
        (5,10),(5,11),
        (6,7),(6,8),(6,9),
        (7,10),(7,11),
        (8,9),(10,11),
    ]
    return verts, edges

OBJECTS = [
    ("Cube",         _make_cube()),
    ("Octaèdre",     _make_octahedron()),
    ("Icosaèdre",    _make_icosahedron()),
]


# ── Rotation 3D par matrices ──────────────────────────────────────────────────────

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def project(verts, angle_x, angle_y, angle_z, scale, cx, cy, fov=4.0):
    """
    Projection perspective des sommets 3D vers le plan 2D.

    angle_x/y/z : angles de rotation en radians
    scale       : taille de l'objet en pixels
    cx, cy      : centre de la scène
    fov         : facteur de perspective (plus grand = moins de perspective)
    """
    R = rot_z(angle_z) @ rot_y(angle_y) @ rot_x(angle_x)
    rotated = (R @ verts.T).T         # (N, 3)
    z_offset = fov
    projected = []
    for v in rotated:
        z = v[2] + z_offset
        if z <= 0:
            z = 0.001
        px = int(cx + (v[0] / z) * scale * z_offset)
        py = int(cy - (v[1] / z) * scale * z_offset)
        depth = v[2]                  # profondeur pour le z-sort
        projected.append((px, py, depth))
    return projected


# ── Dessin liquid glass de l'objet ───────────────────────────────────────────────

def draw_object(img, verts_2d, edges, label):
    """
    Dessine l'objet avec un style liquid glass :
    - Arêtes triées par profondeur (z-sort) → les arêtes lointaines sont plus transparentes
    - Sommets : halo + éclat blanc
    - Label en bas
    """
    h, w = img.shape[:2]

    # Trier les arêtes par profondeur moyenne (les plus lointaines d'abord)
    def edge_depth(e):
        return (verts_2d[e[0]][2] + verts_2d[e[1]][2]) / 2

    sorted_edges = sorted(edges, key=edge_depth)

    # Normaliser la profondeur pour l'opacité
    if verts_2d:
        depths = [v[2] for v in verts_2d]
        dmin, dmax = min(depths), max(depths)
        d_range = max(dmax - dmin, 0.001)
    else:
        dmin, d_range = 0, 1

    for s, e in sorted_edges:
        p1 = (verts_2d[s][0], verts_2d[s][1])
        p2 = (verts_2d[e][0], verts_2d[e][1])

        # Profondeur normalisée [0, 1] → opacité : lointain=0.25, proche=0.75
        depth_norm = ((verts_2d[s][2] + verts_2d[e][2]) / 2 - dmin) / d_range
        alpha = 0.25 + depth_norm * 0.50

        # Épaisseur : arêtes proches légèrement plus épaisses
        thickness = 2 if depth_norm > 0.6 else 1

        ov = img.copy()
        cv2.line(ov, p1, p2, (220, 220, 220), thickness, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

    # Sommets
    for px, py, depth in verts_2d:
        depth_norm = (depth - dmin) / d_range
        r = 4 if depth_norm > 0.6 else 2

        # Halo
        ov = img.copy()
        cv2.circle(ov, (px, py), r + 4, (200, 200, 200), -1)
        cv2.addWeighted(ov, 0.08, img, 0.92, 0, img)

        # Corps
        cv2.circle(img, (px, py), r, (25, 25, 25),  -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, (170,170,170),   1, cv2.LINE_AA)

        # Éclat blanc (effet verre)
        if depth_norm > 0.4:
            cv2.circle(img, (px - 1, py - 1), max(1, r - 1), (255,255,255), -1, cv2.LINE_AA)

    # Label de l'objet
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    lx, ly = w // 2 - tw // 2, h - 16

    ov = img.copy()
    cv2.rectangle(ov, (lx - 6, ly - th - 4), (lx + tw + 6, ly + 4), (15,15,15), -1)
    cv2.addWeighted(ov, 0.50, img, 0.50, 0, img)
    cv2.putText(img, label, (lx, ly), font, scale, (200,200,200), 1, cv2.LINE_AA)


# ── HUD angles ───────────────────────────────────────────────────────────────────

def draw_hud_angles(img, rot):
    """Affiche les 3 angles de rotation en haut à gauche."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.38
    lines = [
        f"inclinaison : {rot['inclinaison']:+.1f}°",
        f"rotation    : {rot['rotation']:+.1f}°",
        f"roulis      : {rot['roulis']:+.1f}°",
    ]
    for i, line in enumerate(lines):
        y = 20 + i * 18
        (tw, th), _ = cv2.getTextSize(line, font, scale, 1)
        ov = img.copy()
        cv2.rectangle(ov, (8, y - th - 2), (8 + tw + 6, y + 4), (15,15,15), -1)
        cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
        cv2.putText(img, line, (11, y), font, scale, (190,190,190), 1, cv2.LINE_AA)


def draw_hud_hint(img):
    """Affiche les contrôles en bas à droite."""
    h, w = img.shape[:2]
    hints = [
        "main ouverte → rotation",
        "poing        → pause",
        "pincement    → changer objet",
        "Échap        → quitter",
    ]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.34
    for i, hint in enumerate(reversed(hints)):
        y = h - 12 - i * 16
        (tw, th), _ = cv2.getTextSize(hint, font, scale, 1)
        cv2.putText(img, hint, (w - tw - 10, y), font, scale, (120,120,120), 1, cv2.LINE_AA)


# ── Capture caméra dans un thread ────────────────────────────────────────────────

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM['source'], cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM['display_w'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM['display_h'])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAM['buffer_size'])
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


# ── Boucle principale ─────────────────────────────────────────────────────────────

def run():
    cam = CameraStream()

    # Attendre la première frame
    while cam.get() is None:
        time.sleep(0.01)

    # État de l'objet
    obj_idx   = 0                           # index dans OBJECTS
    angle_x   = 0.0
    angle_y   = 0.0
    angle_z   = 0.0

    # Vitesse angulaire (inertie douce)
    vel_x = vel_y = vel_z = 0.0
    INERTIE  = 0.85    # 0=stop immédiat, 1=glisse indéfiniment
    SENSIB   = 0.022   # degré de rotation par degré de roulis/inclinaison

    # Anti-rebond du pincement pour changer d'objet
    pinch_locked = False

    CAM_W = CAM['detect_w']
    CAM_H = CAM['detect_h']
    DW    = CAM['display_w']
    DH    = CAM['display_h']
    CX    = DW // 2
    CY    = DH // 2
    SCALE = min(DW, DH) // 3   # taille de l'objet

    print("Scène 3D lancée — Échap pour quitter")

    while True:
        frame = cam.get()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (CAM_W, CAM_H))

        # Détection de la main
        annotated, hand_data = hand_detect(small)

        # Upscale le feed caméra annoté comme fond
        bg = cv2.resize(annotated, (DW, DH))

        rot_data = None

        if hand_data:
            h0      = hand_data[0]
            gestes  = h0['gestes']
            rot_data = h0['rotation']

            poing     = gestes['poing']
            pincement = gestes['pincement']

            if not poing:
                # ── Rotation pilotée par les angles de la main ─────────────────
                # rot_data vient de rotation_3d() dans hand_detect.py
                # roulis      → rotation autour de Z (main inclinée gauche/droite)
                # inclinaison → rotation autour de X (main vers/loin caméra)
                # rotation    → rotation autour de Y (paume face/dos)
                target_vz = math.radians(rot_data['roulis'])       * SENSIB * 60
                target_vx = math.radians(rot_data['inclinaison'])  * SENSIB * 60
                target_vy = math.radians(rot_data['rotation'])     * SENSIB * 20

                # Lissage exponentiel de la vitesse (évite les à-coups)
                vel_x = vel_x * 0.6 + target_vx * 0.4
                vel_y = vel_y * 0.6 + target_vy * 0.4
                vel_z = vel_z * 0.6 + target_vz * 0.4

            else:
                # Poing = pause : décélération avec inertie
                vel_x *= INERTIE
                vel_y *= INERTIE
                vel_z *= INERTIE

            # Changement d'objet au pincement
            if pincement and not pinch_locked:
                obj_idx    = (obj_idx + 1) % len(OBJECTS)
                pinch_locked = True
            elif not pincement:
                pinch_locked = False

        else:
            # Aucune main : inertie naturelle
            vel_x *= INERTIE
            vel_y *= INERTIE
            vel_z *= INERTIE

        # Arrêt si vitesse négligeable
        if abs(vel_x) < 0.0001: vel_x = 0.0
        if abs(vel_y) < 0.0001: vel_y = 0.0
        if abs(vel_z) < 0.0001: vel_z = 0.0

        # Mise à jour des angles
        angle_x += vel_x
        angle_y += vel_y
        angle_z += vel_z

        # ── Rendu de l'objet ────────────────────────────────────────────────────
        name, (verts, edges) = OBJECTS[obj_idx]
        verts_2d = project(verts, angle_x, angle_y, angle_z, SCALE, CX, CY)
        draw_object(bg, verts_2d, edges, name)

        # ── HUD ─────────────────────────────────────────────────────────────────
        if rot_data:
            draw_hud_angles(bg, rot_data)
        draw_hud_hint(bg)

        cv2.imshow("Scène 3D — Contrôle par la main", bg)

        if cv2.waitKey(1) == 27:
            break

    cam.stop()
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
