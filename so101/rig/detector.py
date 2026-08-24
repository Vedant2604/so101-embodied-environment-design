"""HSV cube detector -> real-world success signal / reward for the SO-101 rig.

Calibrate once (sliders for colour + click the two zone centres):
    python detector.py --calibrate --cam 1

Live test (overlay + zone readout):
    python detector.py --cam 1

Use inside training code:
    from detector import CubeDetector
    det = CubeDetector.load("detector_cfg.json", cam_index=1)
    r, success = det.reward(target="B")
"""

import argparse
import json

import cv2
import numpy as np

CFG_PATH = "detector_cfg.json"


class CubeDetector:
    def __init__(self, cam_index, hsv_lo, hsv_hi, zone_a, zone_b,
                 zone_radius_px=40, min_area=150):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.hsv_lo = np.array(hsv_lo, np.uint8)
        self.hsv_hi = np.array(hsv_hi, np.uint8)
        self.zone_a = np.array(zone_a, float)
        self.zone_b = np.array(zone_b, float)
        self.r = float(zone_radius_px)
        self.min_area = min_area

    @classmethod
    def load(cls, path=CFG_PATH, cam_index=None):
        cfg = json.load(open(path))
        if cam_index is not None:
            cfg["cam_index"] = cam_index
        return cls(**cfg)

    def _mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.hsv_lo[0] > self.hsv_hi[0]:          # red wraps H=0
            m = cv2.inRange(hsv, np.array([0, self.hsv_lo[1], self.hsv_lo[2]]),
                            self.hsv_hi) | \
                cv2.inRange(hsv, self.hsv_lo,
                            np.array([179, self.hsv_hi[1], self.hsv_hi[2]]))
        else:
            m = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return {"found": False, "zone": None, "frame": None}
        mask = self._mask(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= self.min_area]
        if not cnts:
            return {"found": False, "zone": None, "frame": frame, "mask": mask}
        M = cv2.moments(max(cnts, key=cv2.contourArea))
        xy = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        dA = float(np.linalg.norm(xy - self.zone_a))
        dB = float(np.linalg.norm(xy - self.zone_b))
        zone = "A" if dA < self.r else ("B" if dB < self.r else None)
        return {"found": True, "xy_px": xy, "zone": zone, "dist_A": dA,
                "dist_B": dB, "frame": frame, "mask": mask}

    def reward(self, target: str, dense=True):
        s = self.read()
        if not s["found"]:
            return 0.0, False
        success = s["zone"] == target
        if not dense:
            return (1.0 if success else 0.0), success
        d = s["dist_A"] if target == "A" else s["dist_B"]
        return -d / 100.0 + (10.0 if success else 0.0), success

    def release(self):
        self.cap.release()


def calibrate(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("cal")
    for n, v, mx in (("H lo", 0, 179), ("S lo", 120, 255), ("V lo", 70, 255),
                     ("H hi", 10, 179), ("S hi", 255, 255), ("V hi", 255, 255)):
        cv2.createTrackbar(n, "cal", v, mx, lambda x: None)

    clicks = []
    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append([x, y])
            print(f"zone {'A' if len(clicks) == 1 else 'B'} centre = ({x}, {y})")
    cv2.setMouseCallback("cal", on_click)

    print("1) sliders until ONLY the cube is white in the mask\n"
          "2) click zone A centre, then zone B centre\n"
          "3) 's' saves, 'q' quits")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        lo = np.array([cv2.getTrackbarPos(n, "cal") for n in ("H lo", "S lo", "V lo")])
        hi = np.array([cv2.getTrackbarPos(n, "cal") for n in ("H hi", "S hi", "V hi")])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if lo[0] > hi[0]:
            mask = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), hi) | \
                   cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]]))
        else:
            mask = cv2.inRange(hsv, lo, hi)
        vis = frame.copy()
        for i, c in enumerate(clicks):
            cv2.circle(vis, tuple(c), 40, (255, 0, 0) if i == 0 else (0, 140, 255), 2)
        cv2.imshow("cal", np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s") and len(clicks) == 2:
            json.dump({"cam_index": cam_index, "hsv_lo": lo.tolist(),
                       "hsv_hi": hi.tolist(), "zone_a": clicks[0],
                       "zone_b": clicks[1], "zone_radius_px": 40},
                      open(CFG_PATH, "w"), indent=2)
            print("saved ->", CFG_PATH)
            break
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def live(cam_index):
    det = CubeDetector.load(CFG_PATH, cam_index)
    print("Live. Push the cube between zones. 'q' quits.")
    while True:
        s = det.read()
        if s.get("frame") is None:
            continue
        vis = s["frame"]
        txt = "NOT FOUND"
        if s["found"]:
            cv2.circle(vis, tuple(int(v) for v in s["xy_px"]), 10, (0, 255, 0), -1)
            txt = f"zone={s['zone']}  dA={s['dist_A']:.0f}  dB={s['dist_B']:.0f}"
        cv2.circle(vis, tuple(det.zone_a.astype(int)), int(det.r), (255, 0, 0), 2)
        cv2.circle(vis, tuple(det.zone_b.astype(int)), int(det.r), (0, 140, 255), 2)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.imshow("detector", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    det.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--cam", type=int, default=1)
    a = ap.parse_args()
    calibrate(a.cam) if a.calibrate else live(a.cam)