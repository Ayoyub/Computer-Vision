# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CALIBRATION.PY — Calibration des seuils gestuels                         ║
# ║                                                                            ║
# ║  À lancer une seule fois. Résultats sauvegardés dans calibration.json.    ║
# ║  main.py charge ce fichier au démarrage s'il existe.                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import cv2
import math
import json
import time
import os
from hand_detect import hand_detect, release
from config import CAM

CAM_W, CAM_H = CAM['detect_w'],  CAM['detect_h']
DW,   DH     = CAM['display_w'], CAM['display_h']
SX,   SY     = DW / CAM_W, DH / CAM_H

OUTPUT_FILE  = 'calibration.json'

# ── Marge de sécurité appliquée entre les deux seuils mesurés ───────────────────
# Ex : paume_moy=180, poing_moy=60 → midpoint=120
# MARGE_PAUME  = midpoint + X%  → seuil d'entrée "paume détectée"
# MARGE_POING  = midpoint - X%  → seuil d'entrée "poing détecté"
MARGE = 0.15   # 15% de marge de chaque côté du midpoint


def get_dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def dist_4_20(pts):
    """Distance pouce (4) ↔ auriculaire (20) — métrique principale paume/poing."""
    return get_dist(pts[4], pts[20])


def dist_bouts_poignet(pts):
    """Distance moyenne bouts des 4 doigts ↔ poignet — métrique secondaire."""
    poignet = pts[0]
    return sum(get_dist(pts[i], poignet) for i in [8, 12, 16, 20]) / 4


def draw_instructions(img, titre, instruction, progression, n_samples, target):
    """
    Affiche l'interface de calibration sur le frame.

    titre       : étape en cours
    instruction : texte d'aide
    progression : float [0, 1] — barre de progression
    n_samples   : nombre d'échantillons collectés
    target      : nombre d'échantillons cible
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Fond semi-transparent en haut
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 110), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)

    # Titre
    cv2.putText(img, titre, (20, 36), font, 0.80, (230, 230, 230), 1, cv2.LINE_AA)

    # Instruction
    cv2.putText(img, instruction, (20, 68), font, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # Barre de progression
    bar_x, bar_y, bar_w, bar_h = 20, 82, w - 40, 10
    # fond
    ov = img.copy()
    cv2.rectangle(ov, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)
    # remplissage
    fill = int(bar_w * min(progression, 1.0))
    if fill > 0:
        ov = img.copy()
        cv2.rectangle(ov, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (200, 200, 200), -1)
        cv2.addWeighted(ov, 0.80, img, 0.20, 0, img)
    # contour
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

    # Compteur en bas à droite
    compteur = f"{n_samples}/{target}"
    (tw, th), _ = cv2.getTextSize(compteur, font, 0.40, 1)
    cv2.putText(img, compteur, (w - tw - 20, bar_y + bar_h + 14),
                font, 0.40, (120, 120, 120), 1, cv2.LINE_AA)


def draw_feedback(img, pts, d_4_20_val, d_moy_val, label):
    """Affiche les métriques en temps réel sur le feed caméra."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]

    # Points clés
    colors = {4: (255, 200, 100), 8: (100, 255, 100),
              20: (100, 200, 255), 0: (200, 200, 200)}
    for idx, color in colors.items():
        px, py = pts[idx]
        cv2.circle(img, (px, py), 7, color, -1, cv2.LINE_AA)

    # Ligne pouce-auriculaire (métrique principale)
    cv2.line(img, pts[4], pts[20], (180, 180, 180), 1, cv2.LINE_AA)

    # Métriques
    lines = [
        f"pouce <-> auriculaire : {d_4_20_val:.0f} px",
        f"bouts -> poignet (moy): {d_moy_val:.0f} px",
        f"etat detecte          : {label}",
    ]
    for i, line in enumerate(lines):
        y = h - 60 + i * 18
        ov = img.copy()
        (tw, th), _ = cv2.getTextSize(line, font, 0.38, 1)
        cv2.rectangle(ov, (10, y - th - 2), (10 + tw + 6, y + 4), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        cv2.putText(img, line, (13, y), font, 0.38, (170, 170, 170), 1, cv2.LINE_AA)


def collect_samples(cap, label, instruction, n_target=60):
    """
    Collecte n_target mesures de dist_4_20 et dist_bouts_poignet
    pendant que l'utilisateur maintient la pose demandée.

    Retourne (mean_4_20, mean_bouts, std_4_20) ou None si annulé.
    """
    samples_4_20  = []
    samples_bouts = []

    titre = f"Étape : {label}"

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (CAM_W, CAM_H))
        display, hand_data = hand_detect(small)
        display = cv2.resize(display, (DW, DH))

        n = len(samples_4_20)
        progression = n / n_target

        if hand_data:
            pts_raw = hand_data[0]['points']
            pts     = [(int(x*SX), int(y*SY)) for x,y in pts_raw]

            d420  = dist_4_20(pts)
            dbout = dist_bouts_poignet(pts)

            # On collecte seulement si la main est stable (pas de saut brutal)
            if n == 0 or abs(d420 - samples_4_20[-1]) < 40:
                samples_4_20.append(d420)
                samples_bouts.append(dbout)

            etat = label
            draw_feedback(display, pts, d420, dbout, etat)

        else:
            draw_feedback(display, {0:(DW//2,DH//2),4:(DW//2,DH//2),
                                     8:(DW//2,DH//2),20:(DW//2,DH//2)},
                          0, 0, "aucune main détectée")

        draw_instructions(display, titre, instruction, progression, n, n_target)

        # Confirmation visuelle quand c'est bon
        if n >= n_target:
            mean_420  = sum(samples_4_20)  / len(samples_4_20)
            mean_bout = sum(samples_bouts) / len(samples_bouts)
            std_420   = (sum((x - mean_420)**2 for x in samples_4_20) / len(samples_4_20)) ** 0.5

            # Flash vert de confirmation
            ov = display.copy()
            cv2.rectangle(ov, (0,0), (DW, DH), (100,200,100), -1)
            cv2.addWeighted(ov, 0.15, display, 0.85, 0, display)

            msg = f"Mesure OK — moyenne: {mean_420:.0f}px  ecart-type: {std_420:.0f}px"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(msg, font, 0.48, 1)
            cv2.putText(display, msg, (DW//2 - tw//2, DH//2),
                        font, 0.48, (200, 240, 200), 1, cv2.LINE_AA)
            cv2.putText(display, "Appuie sur ESPACE pour continuer",
                        (DW//2 - 140, DH//2 + 30), font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(0)
            if key == 32:   # espace
                return mean_420, mean_bout, std_420
            elif key == 27: # échap
                return None

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1)
        if key == 27:
            return None


def run():
    cap = cv2.VideoCapture(CAM['source'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DH)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("═" * 56)
    print("  CALIBRATION DES GESTES — Python-CV")
    print("═" * 56)
    print("  Deux poses à maintenir ~2 secondes chacune.")
    print("  Résultat sauvegardé dans calibration.json")
    print("  Échap pour annuler à tout moment.")
    print("═" * 56)

    # ── Étape 1 : Paume ouverte ──────────────────────────────────────────────────
    print("\n[1/2] Présente ta paume ouverte face à la caméra...")
    result_paume = collect_samples(
        cap,
        label       = "PAUME OUVERTE",
        instruction = "Tend la main, doigts eccartes, paume face camera — maintiens la pose",
        n_target    = 60,
    )
    if result_paume is None:
        print("Calibration annulée.")
        cap.release(); release(); cv2.destroyAllWindows(); return

    mean_paume, _, std_paume = result_paume
    print(f"  Paume : moyenne={mean_paume:.0f}px  std={std_paume:.0f}px")

    # ── Étape 2 : Poing fermé ────────────────────────────────────────────────────
    print("\n[2/2] Ferme bien le poing...")
    result_poing = collect_samples(
        cap,
        label       = "POING FERME",
        instruction = "Ferme le poing completement, pouce sur le cote — maintiens la pose",
        n_target    = 60,
    )
    if result_poing is None:
        print("Calibration annulée.")
        cap.release(); release(); cv2.destroyAllWindows(); return

    mean_poing, _, std_poing = result_poing
    print(f"  Poing : moyenne={mean_poing:.0f}px  std={std_poing:.0f}px")

    # ── Calcul des seuils ────────────────────────────────────────────────────────
    # Midpoint entre les deux moyennes
    midpoint = (mean_paume + mean_poing) / 2

    # Seuil paume  = midpoint + marge  (on veut être sûr que c'est bien ouvert)
    # Seuil poing  = midpoint - marge  (on veut être sûr que c'est bien fermé)
    seuil_paume = midpoint * (1 + MARGE)
    seuil_poing = midpoint * (1 - MARGE)

    # Sécurité : seuil_paume toujours > seuil_poing
    if seuil_paume <= seuil_poing:
        seuil_paume = mean_paume * 0.75
        seuil_poing = mean_poing * 1.25

    data = {
        '_comment'   : "Généré automatiquement par calibration.py — ne pas éditer à la main",
        'mean_paume' : round(mean_paume, 1),
        'mean_poing' : round(mean_poing, 1),
        'midpoint'   : round(midpoint, 1),
        'seuil_paume': round(seuil_paume, 1),   # d_4_20 > ce seuil → paume ouverte
        'seuil_poing': round(seuil_poing, 1),   # d_4_20 < ce seuil → poing fermé
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    # ── Écran de confirmation ────────────────────────────────────────────────────
    ok, frame = cap.read()
    if ok:
        frame = cv2.flip(frame, 1)
        display = cv2.resize(frame, (DW, DH))
    else:
        display = cv2.Mat() if False else \
                  cv2.imencode('.jpg', cv2.Mat())[1]  # fallback
        display = cv2.zeros((DH, DW, 3), dtype='uint8') \
                  if not ok else cv2.resize(frame, (DW, DH))

    # Fond sombre
    ov = display.copy() if ok else display
    cv2.rectangle(ov, (0, 0), (DW, DH), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.75, display if ok else ov, 0.25, 0, display)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines_confirm = [
        "Calibration terminee !",
        "",
        f"Seuil paume  (d > {seuil_paume:.0f}px)",
        f"Seuil poing  (d < {seuil_poing:.0f}px)",
        f"Midpoint         {midpoint:.0f}px",
        "",
        f"Sauvegarde dans : {OUTPUT_FILE}",
        "",
        "Appuie sur une touche pour quitter",
    ]
    for i, line in enumerate(lines_confirm):
        y = DH//2 - 90 + i * 28
        scale = 0.65 if i == 0 else 0.44
        color = (220, 220, 220) if i == 0 else (140, 140, 140)
        (tw, _), _ = cv2.getTextSize(line, font, scale, 1)
        cv2.putText(display, line, (DW//2 - tw//2, y), font, scale, color, 1, cv2.LINE_AA)

    cv2.imshow("Calibration", display)
    cv2.waitKey(0)

    print("\n" + "═" * 56)
    print(f"  Seuil paume  : {seuil_paume:.0f}px  (d_4_20 > ce seuil)")
    print(f"  Seuil poing  : {seuil_poing:.0f}px  (d_4_20 < ce seuil)")
    print(f"  Sauvegardé   : {OUTPUT_FILE}")
    print("═" * 56)

    cap.release()
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()