import subprocess
import sys
import cv2
import threading
import time
import math
import os
import numpy as np
from face_detect import face_detect
from hand_detect import hand_detect, release, est_poing_ferme, est_pincement
from config import CAM

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MENU RADIAL GESTUEL                                                       ║
# ║                                                                            ║
# ║  Déclenchement :                                                           ║
# ║    1. Paume ouverte face caméra (pouce+index écartés > SEUIL_PAUME)       ║
# ║    2. Fermer le poing → le menu radial apparaît                            ║
# ║    3. Rouvrir la main et pointer vers une option                           ║
# ║    4. Pincer pour confirmer la sélection                                   ║
# ║                                                                            ║
# ║  Options du menu :                                                         ║
# ║    🔵 Shapes Drag & Drop  (haut)                                           ║
# ║    🖱️  Souris Virtuelle    (bas)                                            ║
# ║    ❌  Quitter             (droite)                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DETECT_W,  DETECT_H  = CAM['detect_w'],  CAM['detect_h']
DISPLAY_W, DISPLAY_H = CAM['display_w'], CAM['display_h']

# ── Paramètres du menu radial ────────────────────────────────────────────────────
MENU_RAYON          = 130    # rayon du cercle des options (pixels sur display)
MENU_RAYON_ICONE    = 38     # rayon de chaque bulle d'option
SEUIL_PAUME         = 80     # distance pouce-auriculaire pour "paume ouverte"
SEUIL_POING_MENU    = 55     # distance pouce-auriculaire pour "poing fermé"
SEUIL_SELECT        = 48     # distance index-centre option pour la surbrillance
FRAMES_PAUME        = 18     # frames consécutives de paume ouverte requises
FRAMES_CONFIRM      = 22     # frames consécutives de survol pour confirmer

# Options : (label, angle_deg, emoji, couleur_BGR)
MENU_OPTIONS = [
    ("Shapes",   90,  "◉", (180, 100, 255)),   # haut
    ("Souris",  270,  "⊕", (200, 200, 200)),   # bas
    ("Quitter", 0,    "✕", (100, 120, 255)),   # droite
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FORMES 3D (inchangées)                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MatrixSphere3D:
    def __init__(self, x, y, radius):
        self.locked_hand_id = None
        self.x = float(x); self.y = float(y); self.radius = float(radius)
        self.color = (106, 50, 159)
        self.pinches = []
        self.is_two_handed = False; self.initial_radius = radius
        self.initial_pinch_dist = 0; self.two_hand_offset = (0, 0)
        self.is_grabbed_single = False; self.drag_offset = (0, 0)
        self.vx = 0.0; self.vy = 0.0
        self.pinch_count = 0; self.last_pinch_time = 0
        self.was_pinched_last_frame = False; self.to_delete = False
        self.is_rotating_manually = False; self.last_pinch_pos = (0, 0)
        self.nodes = []; self.edges = []
        self.angle_x = 0.0; self.angle_y = 0.0
        self._generate_geometry(rings=10, segments=14)

    def _generate_geometry(self, rings, segments):
        for i in range(rings + 1):
            phi = (i / rings) * math.pi
            for j in range(segments):
                theta = (j / segments) * 2 * math.pi
                self.nodes.append((math.sin(phi)*math.cos(theta),
                                   math.cos(phi),
                                   math.sin(phi)*math.sin(theta)))
        for i in range(rings):
            for j in range(segments):
                c = i * segments + j
                self.edges.append((c, i * segments + ((j+1) % segments)))
                self.edges.append((c, (i+1) * segments + j))

    def draw(self, img):
        if self.locked_hand_id is None:
            self.angle_y += 0.003; self.angle_x += 0.003
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        projected = []
        for nx, ny, nz in self.nodes:
            rx = nx*cos_y - nz*sin_y; rz = nx*sin_y + nz*cos_y
            ry = ny*cos_x - rz*sin_x
            projected.append((int(rx*self.radius+self.x), int(ry*self.radius+self.y)))
        for i1, i2 in self.edges:
            cv2.line(img, projected[i1], projected[i2], self.color, 1, cv2.LINE_AA)
        for px, py in projected:
            cv2.circle(img, (px, py), 2, (255, 0, 255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), (0,255,0), 1)


class MatrixPyramid3D:
    def __init__(self, x, y, radius):
        self.locked_hand_id = None
        self.x = float(x); self.y = float(y); self.radius = float(radius)
        self.color = (0, 150, 255)
        self.pinches = []
        self.is_two_handed = False; self.initial_radius = radius
        self.initial_pinch_dist = 0; self.two_hand_offset = (0, 0)
        self.is_grabbed_single = False; self.drag_offset = (0, 0)
        self.vx = 0.0; self.vy = 0.0
        self.pinch_count = 0; self.last_pinch_time = 0
        self.was_pinched_last_frame = False; self.to_delete = False
        self.is_rotating_manually = False; self.last_pinch_pos = (0, 0)
        self.nodes = [(0,-1,0),(-0.866,0.5,-0.5),(0.866,0.5,-0.5),(0,0.5,1)]
        self.edges = [(0,1),(0,2),(0,3),(1,2),(2,3),(3,1)]
        self.angle_x = 0.0; self.angle_y = 0.0

    def draw(self, img):
        if self.locked_hand_id is None:
            self.angle_y += 0.04; self.angle_x += 0.02
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        projected = []
        for nx, ny, nz in self.nodes:
            rx = nx*cos_y - nz*sin_y; rz = nx*sin_y + nz*cos_y
            ry = ny*cos_x - rz*sin_x
            projected.append((int(rx*self.radius+self.x), int(ry*self.radius+self.y)))
        for i1, i2 in self.edges:
            cv2.line(img, projected[i1], projected[i2], self.color, 2, cv2.LINE_AA)
        for px, py in projected:
            cv2.circle(img, (px, py), 4, (200,200,255), -1, cv2.LINE_AA)
        if self.pinches:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.color, 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RENDU DU MENU RADIAL                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def draw_radial_menu(img, cx, cy, hovered_idx, confirm_progress):
    """ Dessine le menu radial centré en (cx, cy).
    hovered_idx : index de l'option survolée (-1 = aucune)
    confirm_progress: float [0, 1] — arc de confirmation de la sélection
    """
    # Cercle central (point d'ancrage du menu)
    ov = img.copy()
    cv2.circle(ov, (cx, cy), 22, (240, 240, 240), -1)
    cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)
    cv2.circle(img, (cx, cy), 22, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.circle(img, (cx-4, cy-4), 5, (255, 255, 255), -1, cv2.LINE_AA)

    for i, (label, angle_deg, emoji, color) in enumerate(MENU_OPTIONS):
        angle_rad = math.radians(angle_deg)
        ox = int(cx + MENU_RAYON * math.cos(angle_rad))
        oy = int(cy - MENU_RAYON * math.sin(angle_rad))
        is_hovered = (i == hovered_idx)
        r = MENU_RAYON_ICONE + (6 if is_hovered else 0)
        alpha = 0.8 if is_hovered else 0.3  # Opacité réduite pour les icônes non survolées

        # Ligne de connexion centre → option
        ov = img.copy()
        cv2.line(ov, (cx, cy), (ox, oy), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)

        # Dégradé pour la bulle (effet moderne)
        ov = img.copy()
        for j in range(r, 0, -1):
            ratio = j / r
            current_color = (int(color[0] * ratio), int(color[1] * ratio), int(color[2] * ratio))
            cv2.circle(ov, (ox, oy), j, current_color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        # Contour nacré (plus lumineux si survolé)
        border_color = color if is_hovered else (160, 160, 160)
        cv2.circle(img, (ox, oy), r, border_color, 1 + is_hovered, cv2.LINE_AA)

        # Reflet amélioré (effet verre)
        reflet_ov = img.copy()
        cv2.ellipse(reflet_ov, (ox - r//3, oy - r//2), (r//2, r//3), 45, 0, 360, (255, 255, 255), -1)
        cv2.addWeighted(reflet_ov, 0.25, img, 0.75, 0, img)

        # Arc de confirmation dynamique (remplissage progressif)
        if is_hovered and confirm_progress > 0:
            angle_span = int(360 * confirm_progress)
            axes = (r + 8, r + 8)
            # Arc de fond (gris clair)
            cv2.ellipse(img, (ox, oy), axes, -90, 0, 360, (80, 80, 80), 3, cv2.LINE_AA)
            # Arc de progression (couleur vive)
            cv2.ellipse(img, (ox, oy), axes, -90, 0, angle_span, color, 3, cv2.LINE_AA)

        # Label texte avec fond semi-transparent
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.40
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        lx = ox - tw // 2
        ly = oy + th // 2 + r + 10  # Décalage sous l'icône
        
        # Fond pour le label
        label_ov = img.copy()
        cv2.rectangle(label_ov, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (20, 20, 20), -1)
        cv2.addWeighted(label_ov, 0.7, img, 0.3, 0, img)
        
        # Texte du label
        text_color = (240, 240, 240) if is_hovered else (170, 170, 170)
        cv2.putText(img, label, (lx, ly), font, scale, text_color, 1, cv2.LINE_AA)


def draw_menu_hint(img):
    """Petite instruction en bas de l'écran hors menu."""
    h, w = img.shape[:2]
    hint = "paume ouverte → fermer le poing → menu radial"
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.36
    (tw, th), _ = cv2.getTextSize(hint, font, scale, 1)
    lx, ly = w // 2 - tw // 2, h - 14

    ov = img.copy()
    cv2.rectangle(ov, (lx-6, ly-th-4), (lx+tw+6, ly+4), (12,12,12), -1)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    cv2.putText(img, hint, (lx, ly), font, scale, (110,110,110), 1, cv2.LINE_AA)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  THREADS CAMÉRA / DÉTECTION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CameraStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM['buffer_size'])
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class DetectionThread:
    def __init__(self):
        self.input_frame = None
        self.output_frame = None
        self.output_data = []
        self.lock_in  = threading.Lock()
        self.lock_out = threading.Lock()
        self.running  = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.running:
            with self.lock_in:
                frame = self.input_frame
            if frame is None:
                time.sleep(0.001)
                continue
            frame, hand_data = hand_detect(frame)
            with self.lock_out:
                self.output_frame = frame
                self.output_data  = hand_data

    def submit(self, frame):
        with self.lock_in:
            self.input_frame = frame.copy()

    def get(self):
        with self.lock_out:
            if self.output_frame is not None:
                return self.output_frame.copy(), list(self.output_data)
            return None, []

    def stop(self):
        self.running = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UTILITAIRES                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def paume_ouverte(pts):
    """
    Retourne True si la main est ouverte (paume face caméra).
    Critère : distance pouce (4) ↔ auriculaire (20) > SEUIL_PAUME.
    """
    return get_dist(pts[4], pts[20]) > SEUIL_PAUME


def poing_ferme(pts):
    """
    Retourne True si la main est fermée (poing).
    Critère : distance pouce (4) ↔ auriculaire (20) < SEUIL_POING_MENU.
    """
    return get_dist(pts[4], pts[20]) < SEUIL_POING_MENU


def option_survolee(index_pos, cx, cy):
    """
    Retourne l'index de l'option du menu la plus proche du doigt index,
    ou -1 si aucune n'est dans le rayon de sélection.
    """
    best_i, best_d = -1, float('inf')
    for i, (_, angle_deg, _, _) in enumerate(MENU_OPTIONS):
        angle_rad = math.radians(angle_deg)
        ox = cx + MENU_RAYON * math.cos(angle_rad)
        oy = cy - MENU_RAYON * math.sin(angle_rad)
        d  = get_dist(index_pos, (ox, oy))
        if d < SEUIL_SELECT and d < best_d:
            best_i, best_d = i, d
    return best_i


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BOUCLE PRINCIPALE                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    cam = CameraStream(src=CAM['source'], width=DISPLAY_W, height=DISPLAY_H)
    det = DetectionThread()

    # ── État du menu radial ──────────────────────────────────────────────────────
    # Phase 0 : menu fermé, attente poing fermé pour ouvrir
    # Phase 1 : menu affiché, attente sélection
    menu_phase = 0
    menu_cx = DISPLAY_W // 2  # centre du menu
    menu_cy = DISPLAY_H // 2
    hovered_idx = -1
    frames_hovered = 0  # compteur de frames sur la même option
    action_triggered = None  # option confirmée
    pinch_detected = False  # état du pincement pour confirmation

    # ── État des formes (Shapes mode) ───────────────────────────────────────────
    shapes = []
    drawing_path = []
    spawn_cooldown = 0
    LISSAGE = 3.0
    shapes_active = False  # True = mode shapes en cours

    print("Caméra lancée — fermez le poing pour ouvrir le menu.")
    while cam.read() is None:
        time.sleep(0.01)

    while True:
        frame = cam.read()
        if frame is None:
            continue
        small = cv2.resize(frame, (DETECT_W, DETECT_H))
        det.submit(small)
        result_frame, hand_data = det.get()
        if result_frame is not None:
            display = cv2.resize(result_frame, (DISPLAY_W, DISPLAY_H))
        else:
            display = frame
        hand_data = [] if not hand_data else hand_data  # S'assurer que hand_data est une liste
        scale_x = DISPLAY_W / DETECT_W
        scale_y = DISPLAY_H / DETECT_H
        current_time = time.time()
        if spawn_cooldown > 0:
            spawn_cooldown -= 1

        # ── Coordonnées du premier index en display ──────────────────────────────
        index_disp = None
        if hand_data:
            ix, iy = hand_data[0]['points'][8]
            index_disp = (int(ix * scale_x), int(iy * scale_y))

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║ MACHINE À ÉTATS DU MENU RADIAL ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        if menu_phase == 0:
            # ── Phase 0 : menu fermé, attente poing fermé pour ouvrir ─────────────
            if hand_data:
                pts_raw = hand_data[0]['points']
                pts_disp = [(int(x*scale_x), int(y*scale_y)) for x,y in pts_raw]
                if poing_ferme(pts_disp):
                    # Mémorise la position du poignet comme centre du menu
                    wx, wy = pts_disp[0]
                    menu_cx, menu_cy = wx, wy
                    menu_phase = 1
                    hovered_idx = -1
                    frames_hovered = 0
                    pinch_detected = False
            draw_menu_hint(display)
        elif menu_phase == 1:
            # ── Phase 1 : menu affiché, attente sélection ────────────────────────
            if hand_data:
                pts_raw = hand_data[0]['points']
                pts_disp = [(int(x*scale_x), int(y*scale_y)) for x,y in pts_raw]
                
                # Mise à jour du centre si la main bouge (suit le poignet)
                wx, wy = pts_disp[0]
                menu_cx = menu_cx + int((wx - menu_cx) * 0.12)
                menu_cy = menu_cy + int((wy - menu_cy) * 0.12)
                
                # Option survolée par l'index
                new_hovered = option_survolee(index_disp, menu_cx, menu_cy) if index_disp else -1
                if new_hovered == hovered_idx and hovered_idx != -1:
                    frames_hovered += 1
                else:
                    hovered_idx = new_hovered
                    frames_hovered = 0
                
                # Détection du pincement pour confirmation
                pinch_detected = est_pincement(pts_raw, 0)
                confirm_progress = frames_hovered / FRAMES_CONFIRM if hovered_idx != -1 and pinch_detected else 0
                
                # Confirmation après FRAMES_CONFIRM frames consécutives sur la même option ET pincement
                if frames_hovered >= FRAMES_CONFIRM and hovered_idx != -1 and pinch_detected:
                    action_triggered = MENU_OPTIONS[hovered_idx][0]
                    menu_phase = 0
                    hovered_idx = -1
                    shapes_active = False  # Désactive le mode Shapes si une action est déclenchée via le menu
                draw_radial_menu(display, menu_cx, menu_cy, hovered_idx, confirm_progress)
            else:
                # Main perdue → fermeture du menu
                draw_radial_menu(display, menu_cx, menu_cy, -1, 0)
                menu_phase = 0

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║  EXÉCUTION DE L'ACTION DÉCLENCHÉE                                    ║
        # ╚══════════════════════════════════════════════════════════════════════╝

        if action_triggered == "Quitter":
            print("\nArrêt du système.")
            break

        elif action_triggered == "Souris":
            print("\n🚀 Transition vers la Souris Virtuelle...")
            action_triggered = None
            
            # 1. DÉVERROUILLAGE TOTAL DES RESSOURCES MATÉRIELLES
            det.stop()                # Arrête le thread de détection
            cam.stop()                # Libère le flux /dev/video ou MediaFoundation
            release()                 # Vide MediaPipe de la RAM/VRAM
            cv2.destroyAllWindows()   # Ferme le menu radial proprement
            
            # 2. LANCEMENT DU MODULE INDÉPENDANT
            script_dir = os.path.dirname(os.path.abspath(__file__))
            subprocess.run([sys.executable, "virt_mouse.py"], cwd=script_dir)
            
            # 3. FIN DU HUB (virt_mouse.py prend le relais complet)
            return


        elif action_triggered == "Shapes":
            print("\n🔵 Mode 3D verrouillé sur ON !")
            action_triggered = None
            shapes_active = True
            menu_phase = -1  # Désactive le menu pour faire place nette

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║  MODE SHAPES (actif uniquement si shapes_active)                   ║
        # ╚══════════════════════════════════════════════════════════════════════╝

        if shapes_active:

            # Geste triangle (2 mains)
            if len(hand_data) == 2:
                h1 = [(int(x*scale_x), int(y*scale_y)) for x,y in hand_data[0]['points']]
                h2 = [(int(x*scale_x), int(y*scale_y)) for x,y in hand_data[1]['points']]
                w1x, w1y = h1[0]; w2x, w2y = h2[0]
                if get_dist((w1x,w1y),(w2x,w2y)) > 100:
                    t1 = h1[4]; t2 = h2[4]; i1 = h1[8]; i2 = h2[8]
                    if (get_dist(t1,i1) > 50 and get_dist(t2,i2) > 50 and
                            get_dist(t1,t2) < 60 and get_dist(i1,i2) < 60 and
                            spawn_cooldown == 0):
                        cx = (t1[0]+t2[0]+i1[0]+i2[0])//4
                        cy = (t1[1]+t2[1]+i1[1]+i2[1])//4
                        shapes.append(MatrixPyramid3D(cx, cy, 80))
                        spawn_cooldown = 180

            # --- RECHERCHE DES PINCEMENTS ET DU POING ---
            active_pinches = []
            for hand_idx, hand in enumerate(hand_data):
                pts = hand['points']
                tx, ty = pts[4] # Pouce
                ix, iy = pts[8] # Index
                
                # Mise à l'échelle pour l'écran
                tx, ty = int(tx * scale_x), int(ty * scale_y)
                ix, iy = int(ix * scale_x), int(iy * scale_y)
                
                # Le centre de l'action est entre le pouce et l'index
                cx, cy = (tx + ix) // 2, (ty + iy) // 2
                
                # Au lieu de faire des maths, on lit simplement l'état Hystérésis du détecteur !
                is_fist = hand['gestes']['poing']
                is_pinch = hand['gestes']['pincement']
                
                # Si la main pince ou ferme le poing
                if is_pinch or is_fist:
                    active_pinches.append((cx, cy, is_fist, hand_idx))
                    couleur = (0, 0, 255) if is_fist else (0, 255, 0)
                    cv2.circle(display, (cx, cy), 8, couleur, -1)

            # Reset pinches
            for shape in shapes:
                shape.pinches = []
                if shape.locked_hand_id is not None:
                    if shape.locked_hand_id < len(hand_data):
                        hand = hand_data[shape.locked_hand_id]
                        pts  = hand['points']
                        tx,ty = int(pts[4][0]*scale_x), int(pts[4][1]*scale_y)
                        ix,iy = int(pts[8][0]*scale_x), int(pts[8][1]*scale_y)
                        cx,cy = (tx+ix)//2, (ty+iy)//2
                        wrist = pts[0]
                        is_fist = (get_dist(pts[12],wrist) < get_dist(pts[9],wrist) and
                                   get_dist(pts[16],wrist) < get_dist(pts[13],wrist) and
                                   get_dist(pts[20],wrist) < get_dist(pts[17],wrist))
                        if is_fist:
                            shape.pinches.append((cx,cy,is_fist))
                        else:
                            shape.locked_hand_id = None
                    else:
                        shape.locked_hand_id = None

            for (px,py,is_fist,hand_idx) in active_pinches:
                for shape in shapes:
                    if shape.locked_hand_id == hand_idx:
                        continue
                    if get_dist((px,py),(shape.x,shape.y)) < shape.radius:
                        shape.pinches.append((px,py,is_fist))
                        if is_fist:
                            shape.locked_hand_id = hand_idx
                        break

            # Triple-clic suppression
            for shape in shapes:
                is_pinched_now = len(shape.pinches) > 0
                if is_pinched_now and not shape.was_pinched_last_frame:
                    if current_time - shape.last_pinch_time < 0.75:
                        shape.pinch_count += 1
                    else:
                        shape.pinch_count = 1
                    shape.last_pinch_time = current_time
                    if shape.pinch_count >= 3:
                        shape.to_delete = True
                shape.was_pinched_last_frame = is_pinched_now
            shapes = [s for s in shapes if not s.to_delete]

            # --- DESSIN DE CERCLE ---
            is_grabbing = any(len(s.pinches) > 0 for s in shapes)
            
            if len(active_pinches) == 1 and not is_grabbing:
                px, py, is_fist, _ = active_pinches[0]
                if not is_fist:
                    drawing_path.append((px, py))
                    for i in range(1, len(drawing_path)):
                        cv2.line(display, drawing_path[i-1], drawing_path[i], (255, 200, 0), 4)
            else:
                # Dès qu'on lâche le trait, on analyse le dessin
                if len(drawing_path) > 20:
                    sp, ep = drawing_path[0], drawing_path[-1]
                    
                    # Le trait se referme-t-il sur lui-même ?
                    if get_dist(sp, ep) < 100:
                        
                        # 1. Formatage blindé pour OpenCV
                        contour = np.array(drawing_path, dtype=np.int32).reshape((-1, 1, 2))
                        
                        # 2. OpenCV trouve le cercle parfait englobant ton dessin
                        (cx, cy), rayon_parfait = cv2.minEnclosingCircle(contour)
                        
                        if rayon_parfait > 30:
                            # 3. Vérification : est-ce que le dessin suit ce cercle parfait ?
                            erreurs = []
                            for p in drawing_path:
                                d = get_dist(p, (cx, cy))
                                erreurs.append(abs(d - rayon_parfait))
                                
                            erreur_moyenne = sum(erreurs) / len(erreurs)
                            
                            # Si l'écart moyen est inférieur à 25% du rayon idéal, c'est validé !
                            if (erreur_moyenne / rayon_parfait) < 0.25:
                                shapes.append(MatrixSphere3D(int(cx), int(cy), int(rayon_parfait)))
                                
                drawing_path.clear()


            # Physique
            for shape in shapes:
                shape.is_rotating_manually = False
                if len(shape.pinches) == 1:
                    px,py,is_fist = shape.pinches[0]
                    if not shape.is_grabbed_single:
                        shape.is_grabbed_single = True
                        shape.is_two_handed = False
                        shape.drag_offset = (shape.x-px, shape.y-py)
                        shape.last_pinch_pos = (px,py)
                    if is_fist:
                        shape.is_rotating_manually = True
                        dx = px - shape.last_pinch_pos[0]
                        dy = py - shape.last_pinch_pos[1]
                        shape.angle_y += dx*0.015; shape.angle_x -= dy*0.015
                        shape.drag_offset = (shape.x-px, shape.y-py)
                        shape.vx = 0; shape.vy = 0
                    else:
                        cx2 = px+shape.drag_offset[0]; cy2 = py+shape.drag_offset[1]
                        shape.vx = (cx2-shape.x)/LISSAGE; shape.vy = (cy2-shape.y)/LISSAGE
                        shape.x += shape.vx; shape.y += shape.vy
                    shape.last_pinch_pos = (px,py)
                elif len(shape.pinches) == 2:
                    shape.is_grabbed_single = False
                    p1x,p1y,_ = shape.pinches[0]; p2x,p2y,_ = shape.pinches[1]
                    cc_x = (p1x+p2x)//2; cc_y = (p1y+p2y)//2
                    cpd  = get_dist((p1x,p1y),(p2x,p2y))
                    if not shape.is_two_handed:
                        shape.is_two_handed = True
                        shape.initial_pinch_dist = cpd
                        shape.initial_radius = shape.radius
                        shape.two_hand_offset = (shape.x-cc_x, shape.y-cc_y)
                    else:
                        if shape.initial_pinch_dist > 0:
                            sf = cpd/shape.initial_pinch_dist
                            cr = max(30, min(400, int(shape.initial_radius*sf)))
                            shape.radius += (cr-shape.radius)/LISSAGE
                    cx2 = cc_x+shape.two_hand_offset[0]; cy2 = cc_y+shape.two_hand_offset[1]
                    shape.vx = (cx2-shape.x)/LISSAGE; shape.vy = (cy2-shape.y)/LISSAGE
                    shape.x += shape.vx; shape.y += shape.vy
                else:
                    shape.is_grabbed_single = False; shape.is_two_handed = False
                    shape.x += shape.vx; shape.y += shape.vy
                    shape.vx *= 0.85; shape.vy *= 0.85
                    if abs(shape.vx) < 0.1: shape.vx = 0
                    if abs(shape.vy) < 0.1: shape.vy = 0

            for shape in shapes:
                shape.draw(display)

        cv2.imshow("Vision IA", display)
        if cv2.waitKey(1) == 27:
            break

    det.stop()
    cam.stop()
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()