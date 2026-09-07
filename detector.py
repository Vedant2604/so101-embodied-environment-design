"""Cube detector + world-state estimator for the SO-101 rig.

The cube is a 3D object seen at a shallow angle: its base sits on the surface,
but the blob in the image extends upward well past it. Classifying by the blob
centroid therefore puts a cube that is plainly inside the tray *outside* the
tray polygon. This detector classifies by the blob's **bottom-centre point** —
where the cube actually contacts the surface — which is the only point that
answers "where is the cube" correctly under perspective.

Regions reported:
  "tray"        contact point inside the tray interior   -> retrieve is feasible
  "table_in"    contact point inside the taped square    -> place is feasible
  "table_edge"  on the table but outside that square     -> neither is feasible
  "held"        no blob, but the wrist cam sees the cube in the gripper
  "lost"        no blob and nothing in the gripper

  python detector.py --cams 0 1            live view
  python detector.py --cams 0 1 --tune     print every candidate blob
"""
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

# --- cube blob ------------------------------------------------------------
DARK_THRESH = 70        # pixels darker than this are cube candidates
MIN_AREA    = 1200      # below this is clutter, tape corners, cable
MAX_AREA    = 14000     # above this is the arm or the dark background band
MAX_ASPECT  = 2.2       # generous: perspective stretches the cube

# --- region polygons ------------------------------------------------------
# The contact point is used for classification, so the polygons do NOT need to
# be shrunk much — a cube on the border is genuinely on the border.
TRAY_INSET  = 0.98
WORK_INSET  = 0.98

# --- wrist camera ---------------------------------------------------------
WRIST_DARK_THRESH = 70
WRIST_HELD_FRAC   = 0.12


@dataclass
class WorldState:
    region: str          # tray | table_in | table_edge | held | lost
    xy: tuple            # contact point (bottom-centre of the blob)
    confident: bool
    area: float = 0.0


def _inset(poly, frac):
    c = poly.mean(axis=0)
    return np.round(c + (poly - c) * frac).astype(np.int32)


class CubeDetector:
    def __init__(self, cfg="tray_region.json"):
        if not os.path.exists(cfg):
            raise SystemExit(f"{cfg} not found - run calibrate_tray.py first")
        j = json.load(open(cfg))

        if "tray_corners" not in j:
            raise SystemExit("no tray_corners in config - run calibrate_tray.py")
        self.tray = np.array(j["tray_corners"], np.int32)
        self.tray_in = _inset(self.tray, TRAY_INSET)

        self.work_in = None
        if "work_corners" in j:
            self.work_in = _inset(np.array(j["work_corners"], np.int32), WORK_INSET)
        else:
            print("detector: no work_corners in config - every table position "
                  "will read table_in (run calibrate_work.py to change that)")

    # ------------------------------------------------------------------
    @staticmethod
    def _inside(poly, pt):
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def _classify(self, contact):
        if self._inside(self.tray_in, contact):
            return "tray"
        if self.work_in is None or self._inside(self.work_in, contact):
            return "table_in"
        return "table_edge"

    def holding(self, wrist_frame):
        """True if a large dark region fills the lower-centre of the wrist view."""
        if wrist_frame is None:
            return False
        gray = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, m = cv2.threshold(gray, WRIST_DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
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

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # a candidate is the cube only if its CONTACT POINT falls in a known
        # region — this rejects background clutter without needing tight
        # area/aspect filters, and survives perspective distortion
        best, best_area, best_contact, best_region = None, 0, None, None
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            contact = (x + w // 2, y + h - 2)
            if tune and area > 600:
                inside = ("tray" if self._inside(self.tray_in, contact)
                          else "work" if (self.work_in is not None
                                          and self._inside(self.work_in, contact))
                          else "-")
                print(f"  area={area:7.0f} aspect={aspect:4.2f} "
                      f"contact={contact} in={inside}")
            if not (MIN_AREA < area < MAX_AREA) or aspect > MAX_ASPECT:
                continue
            in_tray = self._inside(self.tray_in, contact)
            in_work = self.work_in is not None and self._inside(self.work_in, contact)
            if not (in_tray or in_work):
                continue                       # not in any region we care about
            if area > best_area:
                best, best_area = c, area
                best_contact = contact
                best_region = "tray" if in_tray else "table_in"

        dbg = scene_frame.copy()
        cv2.polylines(dbg, [self.tray_in], True, (255, 200, 0), 2)
        if self.work_in is not None:
            cv2.polylines(dbg, [self.work_in], True, (200, 200, 200), 2)

        held = self.holding(wrist_frame)

        if best is None:
            region = "held" if held else "lost"
            colour = (200, 200, 0) if held else (0, 0, 255)
            cv2.putText(dbg, region.upper(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            return WorldState(region, (-1, -1), held), dbg

        colour = (0, 255, 0) if best_region == "tray" else (0, 165, 255)
        cv2.drawContours(dbg, [best], -1, colour, 1)
        cv2.circle(dbg, best_contact, 6, colour, -1)          # contact point
        cv2.line(dbg, (best_contact[0] - 12, best_contact[1]),
                 (best_contact[0] + 12, best_contact[1]), colour, 2)
        label = f"{best_region}  contact={best_contact}  area={int(best_area)}"
        if held:
            label += "  [gripper]"
        cv2.putText(dbg, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

        return WorldState(best_region, best_contact, True, best_area), dbg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1],
                    help="wrist index, scene index")
    ap.add_argument("--tune", action="store_true",
                    help="print every candidate blob and where its contact point falls")
    args = ap.parse_args()

    det = CubeDetector()
    caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in args.cams]
    for c in caps:
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("cyan = tray, grey = work region, filled dot = contact point")
    print("put the cube in the tray, then in the square, then in the gripper. Esc quits.")
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