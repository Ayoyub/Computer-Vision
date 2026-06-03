import cv2

print("Test de toutes les caméras disponibles (0 à 5) :\n")
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"  Index {i} : OK — résolution {int(cap.get(3))}x{int(cap.get(4))}")
        cap.release()
    else:
        print(f"  Index {i} : non disponible")

print("\nLance ce script directement ET depuis main.py pour comparer.")