"""Cube detector + world-state estimator for the SO-101 rig.

Scene camera decides where the cube is (table / tray). Wrist camera decides
whether it is currently in the gripper, so a carried cube is reported as
"held" rather than "lost".

  python detector.py --cams 0 1          # wrist, scene — live view
  python detector.py --cams 0 1 --tune   # print every candidate blob's area
"""
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

# --- scene camera (cube on the table / in the tray) -----------------------
DARK_THRESH = 70        # pixels darker than this are candidates
MIN_AREA    = 1500       # lower while tuning; raise once areas are measured
MAX_AREA    = 12000
MAX_ASPECT  = 1.5       # cube is compact; tray tape is elongated
TRAY_INSET  = 0.75      # shrink tray polygon so tape on the border isn't "in tray"

# --- wrist camera (is the cube in the gripper?) ---------------------------
WRIST_DARK_THRESH = 70
WRIST_HELD_FRAC   = 0.12   # dark pixels covering this fraction => holding cube


@dataclass
class WorldState:
    region: str          # "table" | "tray" | "held" | "lost"
    xy: tuple            # scene-camera centroid, (-1,-1) if unknown
    confident: bool


class CubeDetector:
    def __init__(self, tray_json="tray_region.json"):
        if not os.path.exists(tray_json):
            raise SystemExit(f"{tray_json} not found — run calibrate_tray.py first")
        j = json.load(open(tray_json))
        self.tray = np.array(j["tray_corners"], np.int32)
        self.exclude = np.array(j["exclude"], np.int32) if "exclude" in j else None

        c = self.tray.mean(axis=0)
        self.tray_inner = np.round(c + (self.tray - c) * TRAY_INSET).astype(np.int32)

    # ------------------------------------------------------------------
    def _in_tray(self, xy):
        return cv2.pointPolygonTest(
            self.tray_inner, (float(xy[0]), float(xy[1])), False) >= 0

    def holding(self, wrist_frame):
        """True if a large dark region dominates the wrist view."""
        if wrist_frame is None:
            return False
        gray = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, m = cv2.threshold(gray, WRIST_DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
        # look only at the lower-centre, where a gripped cube sits
        h, w = m.shape
        roi = m[int(h * 0.35):, int(w * 0.20):int(w * 0.80)]
        return (roi > 0).mean() > WRIST_HELD_FRAC

    def find(self, scene_frame, wrist_frame=None, tune=False):
        """Return (WorldState, debug_frame)."""
        gray = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        if self.exclude is not None:
            cv2.fillPoly(mask, [self.exclude], 0)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_area = None, 0
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if tune and area > 300:
                print(f"  blob area={area:7.0f} aspect={aspect:4.2f} at ({x},{y})")
            if not (MIN_AREA < area < MAX_AREA):
                continue
            if aspect > MAX_ASPECT:
                continue
            if area > best_area:
                best, best_area = c, area

        dbg = scene_frame.copy()
        cv2.polylines(dbg, [self.tray], True, (255, 200, 0), 1)
        cv2.polylines(dbg, [self.tray_inner], True, (255, 200, 0), 2)
        if self.exclude is not None:
            cv2.polylines(dbg, [self.exclude], True, (0, 0, 255), 1)

        held = self.holding(wrist_frame)

        if best is None:
            region = "held" if held else "lost"
            colour = (200, 200, 0) if held else (0, 0, 255)
            cv2.putText(dbg, region.upper(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            return WorldState(region, (-1, -1), held), dbg

        M = cv2.moments(best)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        region = "tray" if self._in_tray((cx, cy)) else "table"

        colour = (0, 255, 0) if region == "tray" else (0, 165, 255)
        cv2.drawContours(dbg, [best], -1, colour, 2)
        cv2.circle(dbg, (cx, cy), 4, colour, -1)
        label = f"{region}  ({cx},{cy})  area={int(best_area)}"
        if held:
            label += "  [gripper]"
        cv2.putText(dbg, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        return WorldState(region, (cx, cy), True), dbg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1],
                    help="wrist index, scene index")
    ap.add_argument("--tune", action="store_true",
                    help="print every candidate blob's area and aspect")
    args = ap.parse_args()

    det = CubeDetector()
    caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in args.cams]
    for c in caps:
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("move the cube: table / tray / in the gripper / hidden. Esc to quit.")
    n = 0
    while True:
        okw, wrist = caps[0].read()
        oks, scene = caps[1].read()
        if not oks:
            continue
        n += 1
        show = args.tune and n % 30 == 0
        if show:
            print("--- frame")
        ws, dbg = det.find(scene, wrist if okw else None, tune=show)
        cv2.imshow("scene", dbg)
        if okw:
            cv2.imshow("wrist", wrist)
        if cv2.waitKey(1) == 27:
            break
    for c in caps:
        c.release()
    cv2.destroyAllWindows()