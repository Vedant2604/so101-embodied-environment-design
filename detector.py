"""Cube detector + world-state estimator for the SO-101 rig.

Standalone test:  python detector.py --cam 1
Reads tray_region.json produced by calibrate_tray.py.
"""
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

DARK_THRESH = 70        # pixels darker than this are candidates
MIN_AREA    = 1500       # cube is roughly 60x60 px
MAX_AREA    = 12000
MAX_ASPECT  = 1.5       # cube is compact; tray border is elongated


@dataclass
class WorldState:
    region: str                      # "table" | "tray" | "lost"
    xy: tuple
    confident: bool


class CubeDetector:
    def __init__(self, tray_json="tray_region.json"):
        if not os.path.exists(tray_json):
            raise SystemExit(f"{tray_json} not found — run calibrate_tray.py first")
        j = json.load(open(tray_json))
        self.tray = np.array(j["tray_corners"], np.int32)
        self.exclude = np.array(j["exclude"], np.int32) if "exclude" in j else None

    def _in_tray(self, xy):
        return cv2.pointPolygonTest(self.tray, (float(xy[0]), float(xy[1])), False) >= 0

    def find(self, frame):
        """Return (WorldState, debug_frame)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        if self.exclude is not None:
            cv2.fillPoly(mask, [self.exclude], 0)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_area = None, 0
        for c in cnts:
            area = cv2.contourArea(c)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
                continue
            if area > best_area:
                best, best_area = c, area

        dbg = frame.copy()
        cv2.polylines(dbg, [self.tray], True, (255, 200, 0), 2)
        if self.exclude is not None:
            cv2.polylines(dbg, [self.exclude], True, (0, 0, 255), 1)

        if best is None:
            cv2.putText(dbg, "LOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            return WorldState("lost", (-1, -1), False), dbg

        M = cv2.moments(best)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        region = "tray" if self._in_tray((cx, cy)) else "table"

        colour = (0, 255, 0) if region == "tray" else (0, 165, 255)
        cv2.drawContours(dbg, [best], -1, colour, 2)
        cv2.circle(dbg, (cx, cy), 4, colour, -1)
        cv2.putText(dbg, f"{region}  ({cx},{cy})  area={int(best_area)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        return WorldState(region, (cx, cy), True), dbg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    args = ap.parse_args()

    det = CubeDetector()
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("move the cube around. Esc to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        ws, dbg = det.find(frame)
        cv2.imshow("detector", dbg)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()