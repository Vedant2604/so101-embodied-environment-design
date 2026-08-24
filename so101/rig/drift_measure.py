"""AIRIM Experiment 1 — calibration drift measurement for a low-cost arm.

Measures how far the arm's ACTUAL gripper pose deviates from its COMMANDED
pose over hours of continuous operation, using an ArUco marker on the gripper
seen by the fixed top camera. No external metrology equipment.

Setup
-----
1. Print an ArUco marker (DICT_4X4_50, id 0), ~3 cm, tape it flat on the
   gripper's top face so the top camera always sees it at the check poses.
2. Fix the top camera. NEVER move it during a drift campaign.
3. Define K reference poses (joint targets) the arm can reach with the marker
   clearly visible. Defaults below are a starting point — edit for your arm.

Usage
-----
    # one measurement round (run this every N minutes / episodes)
    python drift_measure.py --round 0 --note "baseline before run"

    # continuous campaign: measure every 15 min for 8 hours
    python drift_measure.py --campaign --interval-min 15 --hours 8

Output
------
    drift_log.csv : round, timestamp, pose_id, u_px, v_px, du, dv, err_px,
                    temp_c (per motor), load, note
    Analysis: err_px vs elapsed hours = your drift curve (the paper's Fig. 1).
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM3"
ROBOT_ID = "follower_arm"
CAM_INDEX = 1
LOG = "drift_log.csv"

# K reference poses, joint targets in degrees. EDIT to match your workspace.
REFERENCE_POSES = {
    0: {"shoulder_pan": 0,   "shoulder_lift": -30, "elbow_flex": 60,
        "wrist_flex": 0,  "wrist_roll": 0, "gripper": 20},
    1: {"shoulder_pan": -25, "shoulder_lift": -30, "elbow_flex": 60,
        "wrist_flex": 0,  "wrist_roll": 0, "gripper": 20},
    2: {"shoulder_pan": 25,  "shoulder_lift": -30, "elbow_flex": 60,
        "wrist_flex": 0,  "wrist_roll": 0, "gripper": 20},
    3: {"shoulder_pan": 0,   "shoulder_lift": -45, "elbow_flex": 75,
        "wrist_flex": 0,  "wrist_roll": 0, "gripper": 20},
    4: {"shoulder_pan": 0,   "shoulder_lift": -15, "elbow_flex": 45,
        "wrist_flex": 0,  "wrist_roll": 0, "gripper": 20},
}

ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect_marker(cap, tries=10):
    """Return marker centre (u, v) in pixels, or None."""
    det = cv2.aruco.ArucoDetector(ARUCO, cv2.aruco.DetectorParameters())
    for _ in range(tries):
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = det.detectMarkers(frame)
        if ids is not None and len(corners) > 0:
            return corners[0][0].mean(axis=0)
        time.sleep(0.05)
    return None


def goto(robot, pose, settle_s=2.0, steps=40):
    """Move smoothly to a joint pose and let it settle."""
    obs = robot.get_observation()
    cur = {m: float(obs[f"{m}.pos"]) for m in pose if f"{m}.pos" in obs}
    for i in range(1, steps + 1):
        a = {f"{m}.pos": cur[m] + (pose[m] - cur[m]) * i / steps for m in cur}
        robot.send_action(a)
        time.sleep(0.03)
    time.sleep(settle_s)


def telemetry(robot):
    """Per-motor temperature / load if the bus exposes them; else empty."""
    out = {}
    for reg in ("Present_Temperature", "Present_Load"):
        try:
            out[reg] = robot.bus.sync_read(reg)
        except Exception:
            out[reg] = {}
    return out


def measure_round(robot, cap, rnd, baseline, note=""):
    rows = []
    for pid, pose in REFERENCE_POSES.items():
        goto(robot, pose)
        uv = detect_marker(cap)
        tel = telemetry(robot)
        if uv is None:
            print(f"  pose {pid}: MARKER NOT SEEN")
            rows.append([rnd, time.time(), pid, "", "", "", "", "", "", "",
                         note + "|marker_missing"])
            continue
        if pid not in baseline:
            baseline[pid] = uv
        du, dv = uv - baseline[pid]
        err = float(np.hypot(du, dv))
        temps = tel.get("Present_Temperature", {})
        loads = tel.get("Present_Load", {})
        print(f"  pose {pid}: u={uv[0]:7.1f} v={uv[1]:7.1f} "
              f"du={du:+6.1f} dv={dv:+6.1f} err={err:5.1f}px")
        rows.append([rnd, time.time(), pid, round(uv[0], 2), round(uv[1], 2),
                     round(du, 2), round(dv, 2), round(err, 2),
                     str(temps), str(loads), note])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--interval-min", type=float, default=15)
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--cam", type=int, default=CAM_INDEX)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID,
                                              use_degrees=True))
    robot.connect()

    new = not os.path.exists(LOG)
    f = open(LOG, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["round", "timestamp", "pose_id", "u_px", "v_px", "du", "dv",
                    "err_px", "temps", "loads", "note"])

    baseline = {}
    try:
        if args.campaign:
            t_end = time.time() + args.hours * 3600
            rnd = 0
            while time.time() < t_end:
                print(f"\n=== round {rnd} @ {time.strftime('%H:%M:%S')} ===")
                for row in measure_round(robot, cap, rnd, baseline, args.note):
                    w.writerow(row)
                f.flush()
                rnd += 1
                time.sleep(args.interval_min * 60)
        else:
            for row in measure_round(robot, cap, args.round, baseline, args.note):
                w.writerow(row)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        f.close()
        cap.release()
        robot.disconnect()
        print(f"log -> {LOG}")


if __name__ == "__main__":
    main()