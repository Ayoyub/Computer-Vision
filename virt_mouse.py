import cv2
import math
import pyautogui
from hand_detect import hand_detect, release

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        PARAMÈTRES — MODIFIE ICI                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Lissage du curseur ──────────────────────────────────────────────────────────
# Valeur entre 0.0 et 1.0
# 0.05 = très fluide mais lent    |    0.5 = réactif mais saccadé
# Recommandé : entre 0.15 et 0.30
ALPHA_LISSAGE = 0.20

# ── Seuil de pincement (clic gauche) ────────────────────────────────────────────
# Distance en pixels entre le POUCE (4) et l'INDEX (8) sur le frame réduit (640×360)
# Plus petit = il faut pincer fort    |    Plus grand = clic trop facile
# Recommandé : entre 18 et 30
SEUIL_PINCEMENT = 22

# ── Seuil "main fermée" (poing) ─────────────────────────────────────────────────
# Quand le poing est fermé, les 4 bouts de doigts remontent vers la paume.
# On mesure la distance moyenne entre chaque bout de doigt et le poignet (point 0).
# En dessous de ce seuil = main fermée → aucune action.
# Recommandé : entre 60 et 90 (ajuste si faux positifs)
SEUIL_POING = 75

# ── Résolution de détection ──────────────────────────────────────────────────────
# Doit correspondre au resize dans la boucle principale
CAM_W, CAM_H = 640, 360

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        FIN DES PARAMÈTRES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Sécurité pyautogui : coin haut-gauche de l'écran = tout stoppe
pyautogui.FAILSAFE = True
# Supprime le délai de 0.1s que pyautogui ajoute par défaut à chaque action
pyautogui.PAUSE    = 0

ECRAN_W, ECRAN_H = pyautogui.size()

# Position courante du curseur (initialisée au centre de l'écran)
curr_x, curr_y = ECRAN_W // 2, ECRAN_H // 2

# Verrou de clic : empêche de déclencher 60 clics/sec quand on maintient le pincement
clic_verrouille = False

# ── Landmarks utiles (numéros définis par MediaPipe) ────────────────────────────
# 0  = Poignet
# 4  = Bout du pouce
# 8  = Bout de l'index      ← curseur de la souris
# 12 = Bout du majeur
# 16 = Bout de l'annulaire
# 20 = Bout de l'auriculaire
BOUT_DOIGTS = [4, 8, 12, 16, 20]   # les 5 bouts de doigts


def est_poing_ferme(points):
    """
    Détecte si la main est fermée (poing).

    Principe : quand la main est ouverte, les bouts de doigts sont loin du poignet.
    Quand elle est fermée, ils se rapprochent.
    On calcule la distance moyenne entre les 4 bouts de doigts (hors pouce)
    et le poignet (point 0). Si cette moyenne est sous SEUIL_POING → poing fermé.

    Le pouce est exclu car il reste souvent visible même poing fermé.
    """
    poignet = points[0]
    # Bouts des 4 doigts (index=8, majeur=12, annulaire=16, auriculaire=20)
    bouts = [points[i] for i in [8, 12, 16, 20]]
    dist_moy = sum(math.hypot(b[0] - poignet[0], b[1] - poignet[1]) for b in bouts) / 4
    return dist_moy < SEUIL_POING


def est_pincement(points):
    """
    Détecte le pincement Pouce (4) + Index (8).

    Retourne True uniquement si :
    - la distance pouce-index est sous SEUIL_PINCEMENT
    - ET la main n'est PAS fermée (pour éviter les faux clics sur un poing)
    """
    if est_poing_ferme(points):
        return False   # poing fermé → jamais de clic
    tx, ty = points[4]
    ix, iy = points[8]
    return math.hypot(tx - ix, ty - iy) < SEUIL_PINCEMENT


def draw_hud(display, points, poing, pincement):
    """
    Affiche un petit HUD liquid glass en bas de l'image :
    - État de la main (ouverte / poing / pincement)
    - Indicateur visuel de pincement sur le doigt
    """
    h, w = display.shape[:2]

    # ── Indicateur sur le doigt index ──
    ix, iy = points[8]
    if pincement:
        # Cercle blanc semi-transparent = pincement actif
        ov = display.copy()
        cv2.circle(ov, (ix, iy), 14, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.35, display, 0.65, 0, display)
        cv2.circle(display, (ix, iy), 14, (220, 220, 220), 1, cv2.LINE_AA)
    elif poing:
        # Croix discrète = poing détecté, aucune action
        cv2.line(display, (ix - 8, iy - 8), (ix + 8, iy + 8), (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(display, (ix + 8, iy - 8), (ix - 8, iy + 8), (160, 160, 160), 1, cv2.LINE_AA)

    # ── Barre de statut en bas ──
    if poing:
        etat = "poing  — aucune action"
    elif pincement:
        etat = "clic"
    else:
        etat = "déplacement"

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    (tw, th), _ = cv2.getTextSize(etat, font, scale, 1)

    # Fond semi-transparent
    bx, by = 12, h - 20
    ov = display.copy()
    cv2.rectangle(ov, (bx - 6, by - th - 6), (bx + tw + 6, by + 6), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.50, display, 0.50, 0, display)
    cv2.putText(display, etat, (bx, by), font, scale, (200, 200, 200), 1, cv2.LINE_AA)


# ── Démarrage caméra ────────────────────────────────────────────────────────────
camera = int(input("Numéro de caméra (Entrée = 0) : ") or 0)
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

print(f"Écran : {ECRAN_W}×{ECRAN_H}  |  Échap pour quitter")
print(f"Seuil pincement : {SEUIL_PINCEMENT}px  |  Seuil poing : {SEUIL_POING}px")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Miroir horizontal — indispensable pour que le geste soit intuitif
    frame = cv2.flip(frame, 1)

    # Réduction pour accélérer la détection MediaPipe
    small   = cv2.resize(frame, (CAM_W, CAM_H))
    display, hand_data = hand_detect(small)

    if hand_data:
        points = hand_data[0]   # on utilise la première main détectée

        poing     = est_poing_ferme(points)
        pincement = est_pincement(points)

        # ── Déplacement du curseur ──────────────────────────────────────────────
        # On n'utilise l'index (8) que si la main est ouverte (pas poing, pas clic)
        if not poing:
            ix, iy = points[8]

            # Conversion coordonnées caméra → coordonnées écran Windows
            cible_x = int((ix / CAM_W) * ECRAN_W)
            cible_y = int((iy / CAM_H) * ECRAN_H)

            # Lissage exponentiel : réduit les tremblements
            # curr = curr + ALPHA * (cible - curr)
            curr_x = curr_x + ALPHA_LISSAGE * (cible_x - curr_x)
            curr_y = curr_y + ALPHA_LISSAGE * (cible_y - curr_y)

            pyautogui.moveTo(int(curr_x), int(curr_y))

        # ── Clic gauche ─────────────────────────────────────────────────────────
        if pincement and not clic_verrouille:
            pyautogui.click()
            clic_verrouille = True    # un seul clic tant que les doigts restent serrés
        elif not pincement:
            clic_verrouille = False   # déverrouille quand on rouvre

        # ── HUD ─────────────────────────────────────────────────────────────────
        draw_hud(display, points, poing, pincement)

    # Upscale pour l'affichage plein écran
    display = cv2.resize(display, (1280, 720))
    cv2.imshow("Souris Virtuelle", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
release()
cv2.destroyAllWindows()