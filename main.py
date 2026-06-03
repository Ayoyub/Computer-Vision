import subprocess
import sys
import cv2
import threading
import time
import math
from face_detect import face_detect
from hand_detect import hand_detect, release
from config import CAM, SHAPES, PHYSICS

DETECT_W, DETECT_H   = CAM['detect_w'], CAM['detect_h']
DISPLAY_W, DISPLAY_H = CAM['display_w'], CAM['display_h']
LISSAGE              = PHYSICS['lissage']

# --- Formes Interactives 3D ---
class MatrixSphere3D:
    def __init__(self, x, y, radius):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.color = (106, 50, 159) # Violet
        self.pinches = []
        
        # --- Variables pour le Drag & Drop 2 mains ---
        self.is_two_handed = False
        self.initial_radius = radius
        self.initial_pinch_dist = 0
        self.two_hand_offset = (0, 0)
        self.is_grabbed_single = False
        self.drag_offset = (0, 0)
        
        # --- Variables pour l'Inertie (Lancer) ---
        self.vx = 0.0 # Vitesse horizontale
        self.vy = 0.0 # Vitesse verticale
        
        # --- Variables pour la suppression (Triple-clic) ---
        self.pinch_count = 0
        self.last_pinch_time = 0
        self.was_pinched_last_frame = False
        self.to_delete = False

        # --- Variables pour la rotation manuelle (Poing) ---
        self.is_rotating_manually = False
        self.last_pinch_pos = (0, 0)
        
        self.nodes = []
        self.edges = []
        self.angle_x = 0.0
        self.angle_y = 0.0
        self._generate_geometry(rings=10, segments=14)

    def _generate_geometry(self, rings, segments):
        # Génération mathématique d'une sphère en points 3D (Vertices)
        for i in range(rings + 1):
            phi = (i / rings) * math.pi
            for j in range(segments):
                theta = (j / segments) * 2 * math.pi
                nx = math.sin(phi) * math.cos(theta)
                ny = math.cos(phi)
                nz = math.sin(phi) * math.sin(theta)
                self.nodes.append((nx, ny, nz))
                
        # Connexion des points pour faire le maillage (Edges)
        for i in range(rings):
            for j in range(segments):
                current = i * segments + j
                next_j = i * segments + ((j + 1) % segments)
                next_i = (i + 1) * segments + j
                self.edges.append((current, next_j))
                self.edges.append((current, next_i))

    def draw(self, img):
        # La rotation auto s'arrête si on tourne la forme à la main avec le poing
        if not self.is_rotating_manually:
            self.angle_y += 0.003
            self.angle_x += 0.003
            
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        projected = []
        
        # Projection de la 3D vers l'écran 2D
        for nx, ny, nz in self.nodes:
            rx = nx * cos_y - nz * sin_y
            rz = nx * sin_y + nz * cos_y
            ry = ny * cos_x - rz * sin_x
            rz = ny * sin_x + rz * cos_x
            px = int(rx * self.radius + self.x)
            py = int(ry * self.radius + self.y)
            projected.append((px, py))

        # Dessin des lignes du maillage
        for idx1, idx2 in self.edges:
            p1, p2 = projected[idx1], projected[idx2]
            cv2.line(img, p1, p2, self.color, 1, cv2.LINE_AA)
            
        # Dessin des points lumineux (sommets)
        for px, py in projected:
            cv2.circle(img, (px, py), 2, (255, 0, 255), -1, cv2.LINE_AA)
            
        # Effet visuel : l'hologramme s'entoure d'un halo vert quand on le pince
        if len(self.pinches) > 0:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), (0, 255, 0), 1)

class MatrixPyramid3D:
    def __init__(self, x, y, radius):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.color = (0, 150, 255) # Orange/Or
        self.pinches = []
        
        self.is_two_handed = False
        self.initial_radius = radius
        self.initial_pinch_dist = 0
        self.two_hand_offset = (0, 0)
        self.is_grabbed_single = False
        self.drag_offset = (0, 0)
        
        # --- Variables pour l'Inertie (Lancer) ---
        self.vx = 0.0 
        self.vy = 0.0 
        
        self.pinch_count = 0
        self.last_pinch_time = 0
        self.was_pinched_last_frame = False
        self.to_delete = False

        self.is_rotating_manually = False
        self.last_pinch_pos = (0, 0)
        
        # 4 Sommets mathématiques d'un tétraèdre (Pyramide à base triangulaire)
        self.nodes = [
            (0, -1, 0),                 # Haut
            (-0.866, 0.5, -0.5),        # Bas Gauche
            (0.866, 0.5, -0.5),         # Bas Droite
            (0, 0.5, 1)                 # Avant
        ]
        self.edges = [(0,1), (0,2), (0,3), (1,2), (2,3), (3,1)]
        self.angle_x = 0.0
        self.angle_y = 0.0

    def draw(self, img):
        if not self.is_rotating_manually:
            self.angle_y += 0.04
            self.angle_x += 0.02
            
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        projected = []
        
        for nx, ny, nz in self.nodes:
            rx = nx * cos_y - nz * sin_y
            rz = nx * sin_y + nz * cos_y
            ry = ny * cos_x - rz * sin_x
            rz = ny * sin_x + rz * cos_x
            px = int(rx * self.radius + self.x)
            py = int(ry * self.radius + self.y)
            projected.append((px, py))

        for idx1, idx2 in self.edges:
            p1, p2 = projected[idx1], projected[idx2]
            cv2.line(img, p1, p2, self.color, 2, cv2.LINE_AA)
            
        for px, py in projected:
            cv2.circle(img, (px, py), 4, (200, 200, 255), -1, cv2.LINE_AA)
            
        if len(self.pinches) > 0:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.color, 1)


# --- Threads Caméra et Détection ---
class CameraStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # Effet miroir natif
                with self.lock:
                    self.frame = frame
                    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class DetectionThread:
    def __init__(self):
        self.input_frame, self.output_frame, self.output_data = None, None, []
        self.lock_in, self.lock_out = threading.Lock(), threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    def _worker(self):
        while self.running:
            with self.lock_in:
                frame = self.input_frame
            if frame is None:
                time.sleep(0.001)
                continue
            frame, _ = face_detect(frame)
            frame, hand_data = hand_detect(frame)
            with self.lock_out:
                self.output_frame, self.output_data = frame, hand_data
    def submit(self, frame):
        with self.lock_in:
            self.input_frame = frame.copy()
    def get(self):
        with self.lock_out:
            if self.output_frame is not None:
                return self.output_frame.copy(), self.output_data
            return None, []
    def stop(self):
        self.running = False
        self.thread.join()

DETECT_W, DETECT_H = 640, 360
DISPLAY_W, DISPLAY_H = 1280, 720

# Fonction utilitaire pour calculer les distances entre 2 points
def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    cam = CameraStream(width=DISPLAY_W, height=DISPLAY_H)
    det = DetectionThread()

    shapes = [] 
    drawing_path = []
    spawn_cooldown = 0 
    
    # PARAMÈTRE DE LISSAGE (Plus il est grand, plus la forme met du temps à suivre)
    LISSAGE = 3.0 

    print("Caméra lancée — Échap pour quitter.")
    while cam.read() is None: time.sleep(0.01)

    while True:
        frame = cam.read()
        if frame is None: continue

        small = cv2.resize(frame, (DETECT_W, DETECT_H))
        det.submit(small)
        result_frame, hand_data = det.get()
        
        if result_frame is not None: display = cv2.resize(result_frame, (DISPLAY_W, DISPLAY_H))
        else:
            display = frame
            hand_data = []

        scale_x, scale_y = DISPLAY_W / DETECT_W, DISPLAY_H / DETECT_H
        current_time = time.time()
        
        # Le cooldown empêche de faire apparaître 50 formes à la seconde
        if spawn_cooldown > 0:
            spawn_cooldown -= 1

        # --- 1. GESTE STATIQUE : LE TRIANGLE (ILLUMINATI) ---
        if len(hand_data) == 2:
            h1, h2 = hand_data[0], hand_data[1]
            w1x, w1y = int(h1[0][0]*scale_x), int(h1[0][1]*scale_y)
            w2x, w2y = int(h2[0][0]*scale_x), int(h2[0][1]*scale_y)
            
            # SÉCURITÉ : Les poignets doivent être éloignés (évite le bug de la main fantôme)
            if get_dist((w1x, w1y), (w2x, w2y)) > 100:
                thumb1, thumb2 = h1[4], h2[4]
                index1, index2 = h1[8], h2[8]
                
                t1x, t1y = int(thumb1[0]*scale_x), int(thumb1[1]*scale_y)
                t2x, t2y = int(thumb2[0]*scale_x), int(thumb2[1]*scale_y)
                i1x, i1y = int(index1[0]*scale_x), int(index1[1]*scale_y)
                i2x, i2y = int(index2[0]*scale_x), int(index2[1]*scale_y)

                # Écartement des doigts (vérifie que la main est ouverte et pas en pincement)
                spread1 = get_dist((t1x, t1y), (i1x, i1y))
                spread2 = get_dist((t2x, t2y), (i2x, i2y))

                if spread1 > 50 and spread2 > 50:
                    # Si Pouces gauche/droit se touchent ET Index gauche/droit se touchent
                    if get_dist((t1x, t1y), (t2x, t2y)) < 60 and get_dist((i1x, i1y), (i2x, i2y)) < 60:
                        if spawn_cooldown == 0:
                            cx = (t1x + t2x + i1x + i2x) // 4
                            cy = (t1y + t2y + i1y + i2y) // 4
                            shapes.append(MatrixPyramid3D(cx, cy, 80))
                            spawn_cooldown = 180 # Pause de 3 secondes avant la prochaine apparition

        # --- RECHERCHE DES PINCEMENTS ET DU POING ---
        active_pinches = []
        for hand in hand_data:
            tx, ty = hand[4] # Pouce
            ix, iy = hand[8] # Index
            tx, ty = int(tx * scale_x), int(ty * scale_y)
            ix, iy = int(ix * scale_x), int(iy * scale_y)
            cx, cy = (tx + ix) // 2, (ty + iy) // 2
            
            # Si le pouce et l'index sont proches (Pincement)
            if get_dist((tx, ty), (ix, iy)) < 35:
                # Analyse du Poing : Majeur, Annulaire et Auriculaire sont-ils repliés vers le poignet ?
                wrist = hand[0]
                folded_mid = get_dist(hand[12], wrist) < get_dist(hand[9], wrist)
                folded_ring = get_dist(hand[16], wrist) < get_dist(hand[13], wrist)
                folded_pinky = get_dist(hand[20], wrist) < get_dist(hand[17], wrist)
                
                is_fist = folded_mid and folded_ring and folded_pinky
                
                active_pinches.append((cx, cy, is_fist))
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)

        # Assigner les pincements aux formes proches
        for shape in shapes: shape.pinches = []
        for px, py, is_fist in active_pinches:
            for shape in shapes:
                if get_dist((px, py), (shape.x, shape.y)) < shape.radius:
                    shape.pinches.append((px, py, is_fist))
                    break # On attrape une seule forme par main

        # --- LOGIQUE DE SUPPRESSION (TRIPLE-CLIC) ---
        for shape in shapes:
            is_pinched_now = len(shape.pinches) > 0
            
            if is_pinched_now and not shape.was_pinched_last_frame:
                if current_time - shape.last_pinch_time < 0.75: # Si moins de 0.75s depuis dernier clic
                    shape.pinch_count += 1
                else:
                    shape.pinch_count = 1
                
                shape.last_pinch_time = current_time
                if shape.pinch_count >= 3:
                    shape.to_delete = True

            shape.was_pinched_last_frame = is_pinched_now

        # Nettoyage de la liste (Supprime les formes détruites)
        shapes = [s for s in shapes if not s.to_delete]

        # --- 2. GESTE DYNAMIQUE : DESSINER UN CERCLE INTELLIGENT ---
        is_grabbing_something = any(len(s.pinches) > 0 for s in shapes)
        
        # Si on pince dans le vide (sans poing)
        if len(active_pinches) == 1 and not is_grabbing_something:
            px, py, is_fist = active_pinches[0]
            if not is_fist:
                drawing_path.append((px, py))
                if len(drawing_path) > 1:
                    for i in range(1, len(drawing_path)):
                        cv2.line(display, drawing_path[i-1], drawing_path[i], (255, 200, 0), 4)
        else:
            # Dès qu'on lâche le trait, on analyse le dessin
            if len(drawing_path) > 20: 
                start_p = drawing_path[0]
                end_p = drawing_path[-1]
                
                # Étape 1 : Le trait se referme-t-il sur lui-même ?
                if get_dist(start_p, end_p) < 100:
                    xs = [p[0] for p in drawing_path]
                    ys = [p[1] for p in drawing_path]
                    
                    # Étape 2 : Calcul du Centre de la forme (Barycentre)
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    
                    # Étape 3 : Calcul de la distance entre le centre et CHAQUE point du tracé
                    distances = [get_dist((x, y), (cx, cy)) for x, y in zip(xs, ys)]
                    rayon_moyen = sum(distances) / len(distances)
                    
                    # Étape 4 : Le dessin doit être assez grand, et surtout... rond !
                    if rayon_moyen > 30:
                        # On calcule l'écart de chaque point par rapport au rayon parfait
                        erreurs = [abs(d - rayon_moyen) for d in distances]
                        erreur_moyenne = sum(erreurs) / len(erreurs)
                        
                        # Si l'erreur moyenne est très faible (< 25%), c'est un vrai cercle !
                        if erreur_moyenne / rayon_moyen < 0.25:
                            shapes.append(MatrixSphere3D(cx, cy, rayon_moyen))
            
            drawing_path.clear()

        # --- PHYSIQUE : DRAG & DROP, LISSAGE, ROTATION ET INERTIE ---
        for shape in shapes:
            shape.is_rotating_manually = False
            
            if len(shape.pinches) == 1:
                # --- MODE 1 MAIN ---
                px, py, is_fist = shape.pinches[0]
                
                # Snapshot du moment où on attrape la forme
                if not shape.is_grabbed_single:
                    shape.is_grabbed_single = True
                    shape.is_two_handed = False
                    shape.drag_offset = (shape.x - px, shape.y - py)
                    shape.last_pinch_pos = (px, py)
                
                if is_fist:
                    # ROTATION 3D : Gèle la position, tourne la forme
                    shape.is_rotating_manually = True
                    dx = px - shape.last_pinch_pos[0]
                    dy = py - shape.last_pinch_pos[1]
                    
                    shape.angle_y += dx * 0.015
                    shape.angle_x -= dy * 0.015 
                    
                    # On empêche la forme de sauter en mettant à jour l'offset
                    shape.drag_offset = (shape.x - px, shape.y - py)
                    # La vitesse (pour l'inertie) tombe à zéro
                    shape.vx = 0
                    shape.vy = 0
                else:
                    # DÉPLACEMENT AVEC VITESSE ET LISSAGE
                    cible_x = px + shape.drag_offset[0]
                    cible_y = py + shape.drag_offset[1]
                    
                    # On calcule le vecteur de vitesse actuel
                    shape.vx = (cible_x - shape.x) / LISSAGE
                    shape.vy = (cible_y - shape.y) / LISSAGE
                    
                    # On applique la vitesse à la position
                    shape.x += shape.vx
                    shape.y += shape.vy
                
                shape.last_pinch_pos = (px, py)
                
            elif len(shape.pinches) == 2:
                # --- MODE 2 MAINS (Redimensionnement) ---
                shape.is_grabbed_single = False
                p1x, p1y, fist1 = shape.pinches[0]
                p2x, p2y, fist2 = shape.pinches[1]
                
                current_center_x = (p1x + p2x) // 2
                current_center_y = (p1y + p2y) // 2
                current_pinch_dist = get_dist((p1x, p1y), (p2x, p2y))
                
                if not shape.is_two_handed:
                    shape.is_two_handed = True
                    shape.initial_pinch_dist = current_pinch_dist
                    shape.initial_radius = shape.radius
                    shape.two_hand_offset = (shape.x - current_center_x, shape.y - current_center_y)
                else:
                    if shape.initial_pinch_dist > 0:
                        scale_factor = current_pinch_dist / shape.initial_pinch_dist
                        cible_radius = int(shape.initial_radius * scale_factor)
                        cible_radius = max(30, min(400, cible_radius))
                        shape.radius += (cible_radius - shape.radius) / LISSAGE
                
                # DÉPLACEMENT À 2 MAINS
                cible_x = current_center_x + shape.two_hand_offset[0]
                cible_y = current_center_y + shape.two_hand_offset[1]
                
                shape.vx = (cible_x - shape.x) / LISSAGE
                shape.vy = (cible_y - shape.y) / LISSAGE
                
                shape.x += shape.vx
                shape.y += shape.vy
                
            else:
                # --- PERSONNE NE TOUCHE LA FORME : INERTIE (LANCER) ---
                shape.is_grabbed_single = False
                shape.is_two_handed = False
                
                # La forme continue de bouger grâce à la vitesse emmagasinée...
                shape.x += shape.vx
                shape.y += shape.vy
                
                # ... mais elle freine doucement grâce à la friction (0.85 = perd 15% de vitesse par frame)
                shape.vx *= 0.85
                shape.vy *= 0.85
                
                # Sécurité pour éviter les calculs infinis une fois arrêtée
                if abs(shape.vx) < 0.1: shape.vx = 0
                if abs(shape.vy) < 0.1: shape.vy = 0

        # --- DESSIN FINAL ---
        for shape in shapes:
            shape.draw(display)

        cv2.imshow("Vision - Visage & Mains", display)
        if cv2.waitKey(1) == 27: break

    det.stop()
    cam.stop()
    release()
    cv2.destroyAllWindows()

# --- HUB PRINCIPAL ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 🧠 SYSTÈME DE VISION IA - MENU PRINCIPAL")
    print("="*50)
    print("[1] 🔵 Shapes Drag & Drop")
    print("[2] 🖱️ Virtual Mouse")
    print("[3] ❌ Quitter")
    print("="*50)

    choix = input("\n👉 Choisis ton module (1, 2 ou 3) : ").strip()

    if choix == "1":
        print("\n🚀 Lancement du module Matrix 3D...\n")
        main()
        
    elif choix == "2":
        print("\n🚀 Lancement de la Souris Virtuelle...\n")
        try:
            # sys.executable garantit d'utiliser le bon environnement Python
            subprocess.run([sys.executable, "virt_mouse.py"])
        except Exception as e:
            print(f"❌ Erreur lors du lancement de la souris virtuelle : {e}")
            
    else:
        print("\nArrêt du système.")
        sys.exit()