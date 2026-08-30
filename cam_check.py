import cv2
for i in range(6):
    c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, f = c.read()
    print(i, ok, f.shape if ok else "-")
    c.release()