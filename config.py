# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIG.PY — Python-CV                              ║
# ║                                                                            ║
# ║  Toutes les constantes du projet sont ici.                                 ║
# ║  Modifie ce fichier uniquement — ne touche pas aux autres.                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Imports dans les autres fichiers ────────────────────────────────────────────
#
#   from config import CAM, DETECTION, HAND, MOUSE, SHAPES, PHYSICS
#
# Chaque section est un dict, accès par clé :
#   ex : CAM['width']   HAND['seuil_poing']   MOUSE['alpha']


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CAMÉRA                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
CAM = {
    # ── Source de la caméra ──────────────────────────────────────────────────────
    # 0 = webcam intégrée (défaut)
    # 1, 2, ... = caméras externes (USB, capture card, etc.)
    # Tu peux aussi mettre un chemin de fichier vidéo : 'video.mp4'
    'source': 0,

    # Résolution d'affichage (fenêtre visible)
    'display_w': 1280,
    'display_h': 720,

    # Résolution de détection (frame envoyé à MediaPipe et Haar)
    # Plus petit = plus rapide, moins précis
    # Recommandé : 640×360 (moitié de 1280×720)
    'detect_w': 640,
    'detect_h': 360,

    # Taille du buffer interne de la webcam
    # 1 = toujours la frame la plus fraîche (recommandé)
    'buffer_size': 1,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DÉTECTION MÉDIAPIPE                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
DETECTION = {
    # Nombre maximum de mains détectées simultanément
    # 1 = plus rapide    |    2 = nécessaire pour le zoom et la pyramide
    'num_hands': 2,

    # Chemin local du modèle MediaPipe (téléchargé automatiquement si absent)
    'model_path': 'hand_landmarker.task',
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GESTES DE LA MAIN                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
HAND = {
    # ── Seuil poing ─────────────────────────────────────────────────────────────
    # Distance moyenne (pixels) entre les bouts des 4 doigts et le poignet.
    # En dessous = main considérée fermée → aucun pincement possible.
    # Mesuré sur le frame de DÉTECTION (640×360), pas sur l'affichage.
    # ↑ Augmente si le poing n'est pas détecté assez tôt
    # ↓ Diminue si la main ouverte est détectée à tort comme poing
    'seuil_poing': 70,

    # ── Hystérésis Pincement ────────────────────────────────────────────────────
    'seuil_pincement_on': 15,   # Distance pour DÉCLENCHER le clic
    'seuil_pincement_off': 20,  # Distance pour RELÂCHER le clic

    # ── Seuil pincement (main.py / shapes) ──────────────────────────────────────
    # Même concept mais utilisé dans main.py pour interagir avec les formes 3D.
    # Exprimé sur le frame d'AFFICHAGE (1280×720) après scale.
    # Légèrement plus grand car les coordonnées sont upscalées.
    'seuil_pincement_shapes': 80,

    # ── Seuil doigts écartés (geste pyramide) ───────────────────────────────────
    # Distance minimale pouce-index pour confirmer que la main est ouverte
    # (utilisé dans le geste d'invocation de la pyramide à 2 mains).
    'seuil_main_ouverte': 50,

    # ── Seuil jonction 2 mains (geste pyramide) ─────────────────────────────────
    # Distance maximale entre les pouces / index des 2 mains pour déclencher
    # la création d'une pyramide (triangle formé avec les 2 mains).
    'seuil_jonction_mains': 60,

    # ── Seuil séparation poignets (sécurité 2 mains) ────────────────────────────
    # Distance minimale entre les 2 poignets pour considérer qu'il y a bien
    # 2 mains différentes (évite le bug de la "main fantôme" de MediaPipe).
    'seuil_separation_poignets': 100,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SOURIS VIRTUELLE (virt_mouse.py)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
MOUSE = {
    # ── Lissage du curseur ───────────────────────────────────────────────────────
    # Coefficient du lissage exponentiel : curr += alpha * (cible - curr)
    # 0.05 = très fluide mais lent    |    0.50 = réactif mais saccadé
    # Recommandé : entre 0.15 et 0.30
    'alpha': 0.20,

    # ── Délai anti-rebond du clic ────────────────────────────────────────────────
    # Pas utilisé actuellement (verrou booléen),
    # mais prévu pour un futur cooldown par timer.
    # Valeur en secondes.
    'cooldown_clic': 0.0,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FORMES 3D INTERACTIVES (main.py)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
SHAPES = {
    # ── Sphère (cercle dessiné à la main) ───────────────────────────────────────
    # Nombre d'anneaux et de segments de la sphère (résolution du maillage)
    # ↑ Augmente pour plus de détails (plus lent)
    'sphere_rings'   : 10,
    'sphere_segments': 14,

    # Couleur de la sphère en BGR
    'sphere_color': (106, 50, 159),   # violet

    # ── Pyramide (geste 2 mains) ─────────────────────────────────────────────────
    # Couleur de la pyramide en BGR
    'pyramid_color': (0, 150, 255),   # orange

    # Rayon initial de la pyramide au spawn
    'pyramid_radius_initial': 80,

    # ── Dessin ───────────────────────────────────────────────────────────────────
    # Nombre minimum de points dans le tracé pour analyser la forme dessinée
    'drawing_min_points': 20,

    # Rayon minimum (pixels) pour qu'un cercle dessiné soit accepté
    'drawing_min_radius': 30,

    # Tolérance de circularité : erreur_moyenne / rayon_moyen < cette valeur
    # 0.25 = 25% de tolérance    |    0.15 = cercle presque parfait requis
    'drawing_circle_tolerance': 0.25,

    # Distance max (pixels) entre début et fin du tracé pour fermer la forme
    'drawing_close_threshold': 100,

    # Couleur du trait de dessin en BGR
    'drawing_color': (255, 200, 0),   # jaune

    # Spawn cooldown (frames) entre deux créations de forme
    # 180 frames ≈ 3 secondes à 60fps
    'spawn_cooldown_frames': 180,

    # ── Triple clic (suppression) ────────────────────────────────────────────────
    # Fenêtre de temps (secondes) entre 2 pincements pour compter comme suite
    'triple_clic_fenetre': 0.75,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHYSIQUE DES FORMES (main.py)                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
PHYSICS = {
    # ── Lissage du déplacement ───────────────────────────────────────────────────
    # Diviseur appliqué à la différence position cible / position actuelle.
    # Plus grand = plus fluide / plus d'inertie au lancer
    # vx = (cible_x - shape.x) / lissage
    # Recommandé : entre 2.0 et 6.0
    'lissage': 3.0,

    # ── Friction (inertie au lancer) ─────────────────────────────────────────────
    # Coefficient multiplicatif appliqué à la vitesse à chaque frame après relâchement.
    # 1.0 = glisse indéfiniment    |    0.0 = s'arrête immédiatement
    # Recommandé : entre 0.80 et 0.92
    'friction': 0.85,

    # ── Seuil d'arrêt ────────────────────────────────────────────────────────────
    # Vitesse en dessous de laquelle on met la vitesse à exactement 0
    # (évite les calculs infinis sur des mouvements imperceptibles)
    'vitesse_min': 0.1,

    # ── Rotation manuelle (poing) ────────────────────────────────────────────────
    # Sensibilité de la rotation 3D quand on tourne la forme avec le poing.
    # Coefficient appliqué au déplacement du poing en pixels.
    # ↑ Augmente pour une rotation plus réactive
    'rotation_sensibilite': 0.015,

    # ── Redimensionnement (2 mains) ──────────────────────────────────────────────
    # Rayon minimum et maximum d'une forme (pixels sur le frame d'affichage)
    'radius_min': 30,
    'radius_max': 400,
}