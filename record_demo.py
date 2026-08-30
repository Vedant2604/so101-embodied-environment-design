"""Record teleop demos.
   X = start/stop episode   Circle = abort (while recording) / delete last   PS = quit
   Usage: python record_demos.py --task pick --cams 0 2
"""
import argparse, os, shutil, threading, time
import cv2, numpy as np, pygame
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT, FPS, SPEED, GRIP_SPEED, DEADZONE = "COM3", 30, 1.2, 1.5, 0.08
AXIS_LX, AXIS_LY, AXIS_RX, AXIS_RY, AXIS_L2, AXIS_R2 = 0, 1, 2, 3, 4, 5
BTN_L1, BTN_R1, BTN_PS, BTN_REC, BTN_DROP = 9, 10, 5, 0, 1
LIMITS = {"shoulder_pan": (-110, 110), "shoulder_lift": (-110, 110),
          "elbow_flex": (-110, 110), "wrist_flex": (-110, 110),
          "wrist_roll": (-160, 160), "gripper": (0, 100)}
JOINTS = list(LIMITS)

dz = lambda v: 0.0 if abs(v) < DEADZONE else v


class Cam:
    """Background grabber so camera reads never block the control loop."""
    def __init__(self, idx):
        self.c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, f = self.c.read()
            if ok:
                self.frame = f

    def release(self):
        self.c.release()


ap = argparse.ArgumentParser()
ap.add_argument("--task", default="pick")
ap.add_argument("--cams", type=int, nargs=2, default=[0, 2])
args = ap.parse_args()
root = os.path.join("demos", args.task)
os.makedirs(root, exist_ok=True)

pygame.init(); pygame.joystick.init()
pad = pygame.joystick.Joystick(0); pad.init()
caps = [Cam(i) for i in args.cams]
time.sleep(1.0)   # let the grabbers warm up

arm = SO101Follower(SO101FollowerConfig(port=PORT, id="follower_arm", use_degrees=True))
arm.connect()
target = {m: float(arm.get_observation()[f"{m}.pos"]) for m in JOINTS}
ep = len([d for d in os.listdir(root) if d.startswith("ep")])
print(f"Ready. {ep} episodes on disk. X to start recording.")

rec, buf, last_dir = False, [], None
try:
    while True:
        t0 = time.perf_counter()
        pygame.event.pump()
        if pad.get_button(BTN_PS):
            break

        l2 = (pad.get_axis(AXIS_L2) + 1) / 2
        r2 = (pad.get_axis(AXIS_R2) + 1) / 2
        delta = {"shoulder_pan": -dz(pad.get_axis(AXIS_LX)) * SPEED,
                 "shoulder_lift": -dz(pad.get_axis(AXIS_LY)) * SPEED,
                 "elbow_flex": dz(pad.get_axis(AXIS_RY)) * SPEED,
                 "wrist_roll": dz(pad.get_axis(AXIS_RX)) * SPEED,
                 "wrist_flex": (r2 - l2) * SPEED,
                 "gripper": GRIP_SPEED * (pad.get_button(BTN_R1) - pad.get_button(BTN_L1))}

        state = [float(arm.get_observation()[f"{m}.pos"]) for m in JOINTS]
        for m, d in delta.items():
            lo, hi = LIMITS[m]
            target[m] = max(lo, min(hi, target[m] + d))
        arm.send_action({f"{m}.pos": target[m] for m in JOINTS})

        # ---- X: start / stop+save ----
        if pad.get_button(BTN_REC):
            if not rec:
                rec, buf = True, []
                print(f"REC ep{ep}...                    ")
            else:
                rec = False
                cv2.destroyAllWindows()
                if buf:
                    d = os.path.join(root, f"ep{ep:03d}")
                    os.makedirs(d, exist_ok=True)
                    np.savez(os.path.join(d, "traj.npz"),
                             state=np.array([f[0] for f in buf], np.float32),
                             action=np.array([f[1] for f in buf], np.float32))
                    for i, f in enumerate(buf):
                        for k, img in enumerate(f[2]):
                            cv2.imwrite(os.path.join(d, f"cam{k}_{i:04d}.jpg"), img,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                    print(f"saved ep{ep} ({len(buf)} frames)          ")
                    last_dir = d
                    ep += 1
            time.sleep(0.3)

        # ---- Circle: abort current / delete last saved ----
        if pad.get_button(BTN_DROP):
            if rec:
                rec, buf = False, []
                cv2.destroyAllWindows()
                print(f"aborted ep{ep} (not saved)        ")
            elif last_dir:
                shutil.rmtree(last_dir)
                ep -= 1
                last_dir = None
                print("deleted last saved                ")
            time.sleep(0.3)

        # ---- capture ----
        if rec:
            frames = [c.frame for c in caps]
            if all(f is not None for f in frames):
                buf.append((state, [target[m] for m in JOINTS], [f.copy() for f in frames]))
                cv2.imshow("wrist", frames[0])
                cv2.imshow("scene", frames[1])
                cv2.waitKey(1)

        print(f"{'REC' if rec else '   '} ep{ep} frames={len(buf)}", end="   \r")
        time.sleep(max(1 / FPS - (time.perf_counter() - t0), 0))
finally:
    arm.disconnect()
    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("\nclosed")