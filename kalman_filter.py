import cv2
import numpy as np
import math

class KalmanCursor:
    def __init__(self):
        # 4 états (x, y, vx, vy) et 2 mesures (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Matrice de mesure (On observe seulement X et Y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Matrice de transition (Mouvement linéaire à vitesse constante)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Bruit du modèle (Inertie). Diminuer ces valeurs rend le curseur plus "lourd"
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 5]
        ], np.float32) * 5e-2

        # Bruit de mesure. Augmenter si MediaPipe tremble trop
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-2
        
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
            return int(x), int(y)

        # Sécurité : Si la main "saute" (ex: réapparition), on reset le filtre 
        # pour éviter que le curseur ne traverse l'écran au ralenti
        last_x, last_y = self.kf.statePost[0, 0], self.kf.statePost[1, 0]
        if math.hypot(x - last_x, y - last_y) > 150:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)

        # Correction et Prédiction
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        
        return int(prediction[0, 0]), int(prediction[1, 0])