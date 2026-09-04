"""Run a trained SmolVLA policy on the real SO-101.

  Dry run:   python eval_smolvla.py --ckpt <path> --cams 0 1 --task place
  Live:      python eval_smolvla.py --ckpt <path> --cams 0 1 --go --steps 800 --task place
  Retrieve:  python eval_smolvla.py --ckpt <path> --cams 0 1 --go --steps 800 --task retrieve

Camera keys must match the --rename_map used during training:
  wrist -> observation.images.camera1
  scene -> observation.images.camera2
"""
import argparse
import time

import cv2
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
LIMITS = {"shoulder_pan": (-110, 110), "shoulder_lift": (-110, 110),
          "elbow_flex": (-110, 110), "wrist_flex": (-110, 110),
          "wrist_roll": (-160, 160), "gripper": (0, 100)}
TASKS = {
    "place":    "pick up the cube and place it in the tray",
    "retrieve": "take the cube out of the tray and place it on the table",
}
W, H = 320, 240

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cams", type=int, nargs=2, default=[0, 1],
                help="wrist index, scene index")
ap.add_argument("--task", default="place", choices=list(TASKS))
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--go", action="store_true", help="actually move the arm")
ap.add_argument("--max-delta", type=float, default=8.0)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
policy = SmolVLAPolicy.from_pretrained(args.ckpt).to(dev).eval()
policy.reset()
print(f"loaded {args.ckpt} on {dev}")
print(f'task: {args.task} -> "{TASKS[args.task]}"')

pre = post = None
try:
    from lerobot.processor import PolicyProcessorPipeline
    pre = PolicyProcessorPipeline.from_pretrained(
        args.ckpt, config_filename="policy_preprocessor.json")
    post = PolicyProcessorPipeline.from_pretrained(
        args.ckpt, config_filename="policy_postprocessor.json")
    print("loaded processor pipeline")
except Exception as e:
    print("no external processors:", type(e).__name__)

caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in args.cams]
for c in caps:
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

arm = SO101Follower(SO101FollowerConfig(port="COM3", id="follower_arm",
                                        use_degrees=True))
arm.connect()
print("arm connected |", "LIVE — arm will move" if args.go else "DRY RUN — no motion")


def to_tensor(frame):
    img = cv2.resize(frame, (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(dev)


hz_t0, hz_n = time.perf_counter(), 0

try:
    for step in range(args.steps):
        t0 = time.perf_counter()
        obs = arm.get_observation()
        cur = np.array([float(obs[f"{m}.pos"]) for m in JOINTS], dtype=np.float32)

        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            continue

        batch = {
            "observation.state": torch.from_numpy(cur).unsqueeze(0).to(dev),
            "observation.images.camera1": to_tensor(frames[0]),   # wrist
            "observation.images.camera2": to_tensor(frames[1]),   # scene
            "task": TASKS[args.task],
        }

        if pre is not None:
            batch = pre(batch)
        with torch.inference_mode():
            out = policy.select_action(batch)
        if post is not None:
            out = post({"action": out})
            if isinstance(out, dict):
                out = out["action"]

        act = np.asarray(out.detach().cpu() if torch.is_tensor(out) else out,
                         dtype=np.float32).reshape(-1)[:6]
        act = np.clip(act, cur - args.max_delta, cur + args.max_delta)
        for i, m in enumerate(JOINTS):
            lo, hi = LIMITS[m]
            act[i] = float(np.clip(act[i], lo, hi))

        hz_n += 1
        if step % 20 == 0:
            hz = hz_n / max(time.perf_counter() - hz_t0, 1e-6)
            print(f"{step:4d} [{hz:4.1f} Hz] cur {np.round(cur,1)} -> act {np.round(act,1)}")
            hz_t0, hz_n = time.perf_counter(), 0

        if args.go:
            arm.send_action({f"{m}.pos": float(act[i]) for i, m in enumerate(JOINTS)})

        cv2.imshow("scene", frames[1])
        if cv2.waitKey(1) == 27:
            break
        time.sleep(max(1 / 30 - (time.perf_counter() - t0), 0))
except KeyboardInterrupt:
    pass
finally:
    arm.disconnect()
    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    print("closed")
