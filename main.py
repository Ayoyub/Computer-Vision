import cv2
import threading
import time
import math
from face_detect import face_detect
from hand_detect import hand_detect, release

# --- Formes Interactives ---
# --- Formes Interactives 3D (Style Matrix) ---
class MatrixSphere3D:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = (106,50,159)
        self.pinches = []
        
        # Interactions
        self.is_two_handed = False
        self.initial_radius = radius
        self.initial_pinch_dist = 0
        self.two_hand_offset = (0, 0)
        self.is_grabbed_single = False
        self.drag_offset = (0, 0)

        # Génération de la géométrie 3D (Hologramme)
        self.nodes = []
        self.edges = []
        self.angle_x = 0.0
        self.angle_y = 0.0
        self._generate_geometry(rings=10, segments=14)

    def _generate_geometry(self, rings, segments):
        # 1. Création des points (Vertices) sur une sphère de rayon 1
        for i in range(rings + 1):
            phi = (i / rings) * math.pi
            for j in range(segments):
                theta = (j / segments) * 2 * math.pi
                nx = math.sin(phi) * math.cos(theta)
                ny = math.cos(phi)
                nz = math.sin(phi) * math.sin(theta)
                self.nodes.append((nx, ny, nz))

        # 2. Création des lignes (Edges) pour relier les points
        for i in range(rings):
            for j in range(segments):
                current = i * segments + j
                next_j = i * segments + ((j + 1) % segments)
                next_i = (i + 1) * segments + j
                
                # Lignes horizontales
                self.edges.append((current, next_j))
                # Lignes verticales
                self.edges.append((current, next_i))

    def draw(self, img):
        # Faire tourner la sphère doucement à chaque frame
        self.angle_y += 0.003
        self.angle_x += 0.003
        
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)

        projected = []
        
        # --- PROJECTION 3D vers 2D ---
        for nx, ny, nz in self.nodes:
            # Rotation sur l'axe Y
            rx = nx * cos_y - nz * sin_y
            rz = nx * sin_y + nz * cos_y
            # Rotation sur l'axe X
            ry = ny * cos_x - rz * sin_x
            rz = ny * sin_x + rz * cos_x
            
            # Mise à l'échelle (radius) et placement sur l'écran (x, y)
            px = int(rx * self.radius + self.x)
            py = int(ry * self.radius + self.y)
            projected.append((px, py))

        # --- DESSIN (Le rendu Matrix) ---
        # 1. On dessine les lignes du maillage
        for idx1, idx2 in self.edges:
            p1, p2 = projected[idx1], projected[idx2]
            cv2.line(img, p1, p2, self.color, 1, cv2.LINE_AA)

        # 2. On dessine les sommets (Vertices) brillants
        for px, py in projected:
            cv2.circle(img, (px, py), 2, (150, 255, 150), -1, cv2.LINE_AA)
            
        # Effet visuel si la sphère est pincée (elle devient plus brillante)
        if len(self.pinches) > 0:
            cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), (0, 255, 0), 1)

# --- Capture dans un thread séparé ---
class CameraStream:
    def __init__(self, src=1, width=1280, height=720):
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
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# --- Détection dans un thread séparé ---
class DetectionThread:
    def __init__(self):
        self.input_frame  = None
        self.output_frame = None
        self.output_data  = [] 
        self.lock_in  = threading.Lock()
        self.lock_out = threading.Lock()
        self.running  = True
        self.thread   = threading.Thread(target=self._worker, daemon=True)
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
                self.output_frame = frame
                self.output_data = hand_data

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


# --- Résolution ---
DETECT_W, DETECT_H = 640, 360
DISPLAY_W, DISPLAY_H = 1280, 720


def main():
    cam = CameraStream(width=DISPLAY_W, height=DISPLAY_H)
    det = DetectionThread()

    shapes = [
        MatrixSphere3D(400, 300, 80),
        MatrixSphere3D(800, 300, 60)
    ]

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    print("Caméra lancée — Échap pour quitter.")

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
            hand_data = []

        scale_x = DISPLAY_W / DETECT_W
        scale_y = DISPLAY_H / DETECT_H

        active_pinches = []
        for hand in hand_data:
            tx, ty = hand[4] # Pouce
            ix, iy = hand[8] # Index
            
            tx, ty = int(tx * scale_x), int(ty * scale_y)
            ix, iy = int(ix * scale_x), int(iy * scale_y)

            cx, cy = (tx + ix) // 2, (ty + iy) // 2
            distance = math.hypot(tx - ix, ty - iy)
            
            if distance < 25:
                active_pinches.append((cx, cy))
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)

        for shape in shapes:
            shape.pinches = []

        for px, py in active_pinches:
            for shape in shapes:
                dist_to_shape = math.hypot(px - shape.x, py - shape.y)
                if dist_to_shape < shape.radius:
                    shape.pinches.append((px, py))
                    break

        # --- PHYSIQUE AVEC OFFSET (Zéro Saut) ---
        for shape in shapes:
            if len(shape.pinches) == 1:
                # --- MODE 1 MAIN : Déplacement Relatif ---
                p = shape.pinches[0]
                
                # Instant précis où l'on attrape la forme
                if not shape.is_grabbed_single:
                    shape.is_grabbed_single = True
                    shape.is_two_handed = False
                    # On calcule l'écart (l'offset) entre le centre de la bulle et le pincement
                    shape.drag_offset = (shape.x - p[0], shape.y - p[1])
                
                # On applique le décalage mémorisé
                shape.x = p[0] + shape.drag_offset[0]
                shape.y = p[1] + shape.drag_offset[1]
                
            elif len(shape.pinches) == 2:
                # --- MODE 2 MAINS : Redimensionnement et Déplacement Relatif ---
                shape.is_grabbed_single = False
                p1, p2 = shape.pinches[0], shape.pinches[1]
                
                # Milieu exact entre tes deux mains
                current_center_x = (p1[0] + p2[0]) // 2
                current_center_y = (p1[1] + p2[1]) // 2
                current_pinch_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                
                # Instant précis où la 2ème main touche la bulle
                if not shape.is_two_handed:
                    shape.is_two_handed = True
                    shape.initial_pinch_dist = current_pinch_dist
                    shape.initial_radius = shape.radius
                    # On mémorise aussi l'offset avec les 2 mains pour ne pas que ça saute !
                    shape.two_hand_offset = (shape.x - current_center_x, shape.y - current_center_y)
                else:
                    if shape.initial_pinch_dist > 0:
                        scale_factor = current_pinch_dist / shape.initial_pinch_dist
                        new_radius = int(shape.initial_radius * scale_factor)
                        shape.radius = max(30, min(400, new_radius))
                
                # Déplacement doux avec l'offset des 2 mains
                shape.x = current_center_x + shape.two_hand_offset[0]
                shape.y = current_center_y + shape.two_hand_offset[1]
            
            else:
                # Personne ne touche
                shape.is_grabbed_single = False
                shape.is_two_handed = False

        for shape in shapes:
            shape.draw(display)

        # --- OVERLAY FPS ---
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(display, f"FPS: {fps_display:.0f}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2, cv2.LINE_AA)

        cv2.imshow("Vision - Visage & Mains", display)

        if cv2.waitKey(1) == 27:
            break

    det.stop()
    cam.stop()
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()