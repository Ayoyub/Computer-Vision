import cv2
import numpy as np
import math


class KalmanCursor:
    """
    Filtre de Kalman 2D adaptatif pour un point (x, y).

    Améliorations vs version précédente :
    ──────────────────────────────────────
    1. Ordre correct predict → correct (plus de décalage d'une frame).
    2. JUMP_THRESHOLD normalisé : passé en argument à update() pour s'adapter
       à la taille réelle de la main détectée (évite les faux resets selon
       la distance caméra).
    3. Bruit de processus adaptatif (α-β) : quand le point bouge vite,
       le filtre devient plus réactif ; quand il est quasi-statique, il lisse
       davantage. Évite le lag sur les grands mouvements intentionnels.
    4. Retour en float — la conversion int est faite par l'appelant, ce qui
       préserve la précision pour les calculs de gestes.

    État   : [x, y, vx, vy]
    Mesure : [x, y]
    """

    # ── Paramètres de base ────────────────────────────────────────────────────────
    # PROCESS_NOISE     ↑ = plus réactif,    ↓ = plus lisse
    # MEASUREMENT_NOISE ↑ = ignore le jitter, ↓ = suit MediaPipe de près
    PROCESS_NOISE     = 1e-4
    MEASUREMENT_NOISE = 3e-2

    # Bruit de processus adaptatif : multiplié par ce facteur quand la vitesse
    # dépasse ADAPTIVE_SPEED_THRESHOLD (pixels/frame normalisés).
    # Cela rend le filtre plus réactif sur les mouvements rapides.
    ADAPTIVE_FACTOR         = 8.0
    ADAPTIVE_SPEED_THRESHOLD = 0.04   # ratio vitesse/taille_main

    # Seuil de reset dur (ratio de la taille de la main).
    # Ex : 1.5 = le point peut sauter jusqu'à 1.5× la longueur paume→majeur
    # avant qu'on considère que c'est une nouvelle main.
    JUMP_RATIO = 1.5

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)

        # Modèle vitesse constante : x' = x + vx, etc.
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)

        self._base_Q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 5],
        ], np.float32) * self.PROCESS_NOISE

        self.kf.processNoiseCov    = self._base_Q.copy()
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.MEASUREMENT_NOISE
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)

        self._initialized = False
        self._last_speed  = 0.0   # vitesse normalisée (pour l'adaptatif)

    # ── Réinitialisation ──────────────────────────────────────────────────────────

    def _reset(self, x: float, y: float) -> None:
        state = np.array([[x], [y], [0.0], [0.0]], np.float32)
        self.kf.statePre  = state.copy()
        self.kf.statePost = state.copy()
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._last_speed  = 0.0
        self._initialized = True

    # ── Update principal ──────────────────────────────────────────────────────────

    def update(self, x: float, y: float,
               hand_scale: float = 100.0) -> tuple[float, float]:
        """
        Filtre la mesure (x, y) et retourne la position filtrée en float.

        hand_scale : taille de référence de la main en pixels (poignet→base majeur).
                     Utilisé pour normaliser le seuil de jump et la vitesse adaptative.
        """
        if not self._initialized:
            self._reset(x, y)
            return x, y

        last_x = float(self.kf.statePost[0, 0])
        last_y = float(self.kf.statePost[1, 0])
        jump   = math.hypot(x - last_x, y - last_y)

        # Reset dur si le point téléporte (main sortie/rentrée du cadre)
        if jump > hand_scale * self.JUMP_RATIO:
            self._reset(x, y)
            return x, y

        # ── Bruit adaptatif ───────────────────────────────────────────────────────
        # Normalise la vitesse par la taille de la main pour être indépendant
        # de la distance caméra.
        speed_ratio = jump / max(hand_scale, 1.0)
        self._last_speed = speed_ratio

        if speed_ratio > self.ADAPTIVE_SPEED_THRESHOLD:
            # Mouvement rapide : on fait plus confiance à la mesure
            self.kf.processNoiseCov = self._base_Q * self.ADAPTIVE_FACTOR
        else:
            # Quasi-statique : on lisse agressivement
            self.kf.processNoiseCov = self._base_Q

        # ── Cycle Kalman correct : predict d'abord, puis corriger ─────────────────
        # FIX vs version précédente : l'ordre était inversé (correct→predict),
        # ce qui introduisait un retard d'une frame.
        self.kf.predict()
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        corrected   = self.kf.correct(measurement)

        return float(corrected[0, 0]), float(corrected[1, 0])