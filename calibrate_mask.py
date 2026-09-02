"""Click 4 corners around the arm/servo region to exclude it. Enter to save.
   python calibrate_mask.py --cam 1
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

pts = []
def click(ev, x, y, flags, param):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y]); print(f"corner {len(pts)}: ({x}, {y})")

cv2.namedWindow("click 4 corners around the arm")
cv2.setMouseCallback("click 4 corners around the arm", click)
while True:
    d = frame.copy()
    for p in pts: cv2.circle(d, tuple(p), 5, (0,0,255), -1)
    if len(pts) == 4: cv2.polylines(d, [np.array(pts, np.int32)], True, (0,0,255), 2)
    cv2.imshow("click 4 corners around the arm", d)
    k = cv2.waitKey(20)
    if k == 27 or (len(pts) == 4 and k == 13): break
cv2.destroyAllWindows()

if len(pts) == 4:
    j = json.load(open("tray_region.json"))
    j["exclude"] = pts
    json.dump(j, open("tray_region.json", "w"), indent=2)
    print("saved exclusion zone")