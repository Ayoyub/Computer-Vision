"""
test_cam.py — enumerate available webcams (cross-platform)
═══════════════════════════════════════════════════════════
Run directly:  python test_cam.py
"""
import cv2

print("Test de toutes les caméras disponibles (0 à 5) :\n")
for i in range(6):
    # V4L2 = back-end Linux (webcams), DSHOW = Windows
    backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(i, backend)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  Index {i} : OK — résolution {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        else:
            print(f"  Index {i} : ouverte mais impossible de lire une frame")
        cap.release()
    else:
        print(f"  Index {i} : non disponible")

print("\nLance ce script directement ET depuis main.py pour comparer.")
