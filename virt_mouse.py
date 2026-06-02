import cv2
import time
import math
import pyautogui
from hand_detect import hand_detect, release

# --- Configuration de la Souris ---
# Sécurité : Si la souris devient folle, mets-la vite dans un coin de l'écran pour tout couper !
pyautogui.FAILSAFE = True 

# Récupération de la taille de ton vrai écran Windows (ex: 1920x1080)
ECRAN_W, ECRAN_H = pyautogui.size()

# Résolution de la caméra
CAM_W, CAM_H = 640, 360  

# Paramètres de lissage (Plus le chiffre est grand, plus c'est fluide mais lent)
LISSAGE = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
camera = int(input('Entrez num de la caméra: '))
cap = cv2.VideoCapture(camera)
cap.set(3, 1280)
cap.set(4, 720)

# Pour éviter de faire 50 clics par seconde quand on pince
clic_verrouille = False 

print(f"Écran détecté : {ECRAN_W}x{ECRAN_H}. Appuie sur Echap pour quitter.")

while True:
    succes, frame = cap.read()
    if not succes:
        break

    # Effet miroir (indispensable pour que ce soit intuitif)
    frame = cv2.flip(frame, 1)

    # Réduction pour la vitesse de détection
    small = cv2.resize(frame, (CAM_W, CAM_H))
    
    # Détection des mains via ton module
    display, hand_data = hand_detect(small)

    if len(hand_data) > 0:
        # On ne prend que la première main détectée
        main = hand_data[0]
        
        # Coordonnées (Pouce = 4, Index = 8, Majeur = 12)
        tx, ty = main[4]
        ix, iy = main[8]
        mx, my = main[12]

        # 1. DÉPLACEMENT (On se base sur l'Index)
        # On convertit la position de la caméra vers la taille de l'écran Windows
        cible_x = int((ix / CAM_W) * ECRAN_W)
        cible_y = int((iy / CAM_H) * ECRAN_H)

        # Application du Lissage (Formule mathématique d'interpolation)
        curr_x = prev_x + (cible_x - prev_x) / LISSAGE
        curr_y = prev_y + (cible_y - prev_y) / LISSAGE

        # On bouge la vraie souris Windows !
        pyautogui.moveTo(curr_x, curr_y)
        
        prev_x, prev_y = curr_x, curr_y

        # 2. LE CLIC GAUCHE (Pincement Pouce + Index)
        distance_clic = math.hypot(tx - ix, ty - iy)
        
        if distance_clic < 20: # Doigts fermés
            cv2.circle(display, (ix, iy), 10, (0, 255, 0), -1)
            
            if not clic_verrouille:
                pyautogui.click()
                clic_verrouille = True # On verrouille pour faire 1 seul clic
        else:
            clic_verrouille = False # On déverrouille quand on rouvre les doigts

    # Affichage
    cv2.imshow("Souris Virtuelle", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
release()
cv2.destroyAllWindows()