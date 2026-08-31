import cv2
caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in (0, 1, 2)]
for c in caps:
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    for k, c in enumerate(caps):
        ok, f = c.read()
        if ok:
            print(k, f.shape, end="  ")
            cv2.imshow(f"cam{k}", f)
    print(end="\r")
    if cv2.waitKey(1) == 27:
        break
for c in caps: c.release()
cv2.destroyAllWindows()