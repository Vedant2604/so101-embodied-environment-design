"""Click the 4 corners of the tray interior (any order). Esc to quit.
   python calibrate_tray.py --cam 1
"""
import argparse, json
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1)
args = ap.parse_args()

cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(10):
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("no frame")

pts = []
def click(ev, x, y, flags, param):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])
        print(f"corner {len(pts)}: ({x}, {y})")

cv2.namedWindow("click 4 tray corners")
cv2.setMouseCallback("click 4 tray corners", click)

while True:
    disp = frame.copy()
    for p in pts:
        cv2.circle(disp, tuple(p), 5, (0, 255, 0), -1)
    if len(pts) == 4:
        cv2.polylines(disp, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
    cv2.imshow("click 4 tray corners", disp)
    k = cv2.waitKey(20)
    if k == 27 or (len(pts) == 4 and k == 13):   # Esc, or Enter when done
        break

cv2.destroyAllWindows()
if len(pts) == 4:
    json.dump({"tray_corners": pts}, open("tray_region.json", "w"), indent=2)
    print("saved tray_region.json")
else:
    print("need 4 corners, nothing saved")