"""Click the 4 inside corners of the taped work region. Enter to save.
   python calibrate_work.py --cam 1

Updates only `work_corners` in tray_region.json; tray_corners is preserved.
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1)
args = ap.parse_args()

cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ok = False
for _ in range(10):
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("no frame from camera")

cfg = {}
if os.path.exists("tray_region.json"):
    cfg = json.load(open("tray_region.json"))
tray = np.array(cfg["tray_corners"], np.int32) if "tray_corners" in cfg else None
old = np.array(cfg["work_corners"], np.int32) if "work_corners" in cfg else None

pts = []


def click(ev, x, y, flags, param):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])
        print(f"corner {len(pts)}: ({x}, {y})")


win = "click 4 work-region corners (Enter=save, r=reset, Esc=cancel)"
cv2.namedWindow(win)
cv2.setMouseCallback(win, click)

while True:
    disp = frame.copy()
    if tray is not None:
        cv2.polylines(disp, [tray], True, (255, 200, 0), 1)      # tray, cyan
    if old is not None:
        cv2.polylines(disp, [old], True, (120, 120, 120), 1)     # old, grey
    for p in pts:
        cv2.circle(disp, tuple(p), 5, (0, 255, 0), -1)
    if len(pts) == 4:
        cv2.polylines(disp, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(disp, "Enter to save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(win, disp)

    k = cv2.waitKey(20)
    if k == 27:
        pts = []
        break
    if k in (ord("r"), ord("R")):
        pts = []
    if len(pts) == 4 and k == 13:
        break

cv2.destroyAllWindows()

if len(pts) == 4:
    cfg["work_corners"] = pts
    json.dump(cfg, open("tray_region.json", "w"), indent=2)
    print("saved work_corners (tray_corners preserved)")
else:
    print("cancelled, nothing saved")
