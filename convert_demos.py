"""Convert a demo folder into a video-encoded LeRobotDataset.

  python convert_demos.py retrieve
  python convert_demos.py pick3 --repo Ved4nt/so101_place
  python convert_demos.py pick3 retrieve --repo Ved4nt/so101_multitask   # multi-task
"""
import argparse
import glob
import os

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 30
W, H = 320, 240
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

# folder name -> the task string the policy is conditioned on
TASK_STRINGS = {
    "pick":     "pick up the cube and place it in the tray",
    "pick2":    "pick up the cube and place it in the tray",
    "pick3":    "pick up the cube and place it in the tray",
    "place4":     "pick up the cube and place it in the tray",
    "reposition": "pick up the cube and place it somewhere else on the table",
    "retrieve": "take the cube out of the tray and place it on the table",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+",
                    help="demo folder name(s) under demos/, e.g. retrieve")
    ap.add_argument("--repo", default=None,
                    help="HF repo id (default: Ved4nt/so101_<task>)")
    ap.add_argument("--user", default="Ved4nt")
    ap.add_argument("--src", default="demos")
    args = ap.parse_args()

    repo_id = args.repo or f"{args.user}/so101_{'_'.join(args.tasks)}"

    plan = []
    for t in args.tasks:
        d = os.path.join(args.src, t)
        if not os.path.isdir(d):
            raise SystemExit(f"no such folder: {d}")
        if t not in TASK_STRINGS:
            raise SystemExit(f"no task string defined for '{t}' — add it to TASK_STRINGS")
        eps = sorted(glob.glob(os.path.join(d, "ep*")))
        plan.append((t, eps))
        print(f"{d}: {len(eps)} episodes -> \"{TASK_STRINGS[t]}\"")

    print(f"\nrepo_id: {repo_id}")
    if input("proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("cancelled")

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action":            {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "observation.images.wrist": {"dtype": "video", "shape": (H, W, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.images.scene": {"dtype": "video", "shape": (H, W, 3),
                                     "names": ["height", "width", "channels"]},
    }

    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features,
                               robot_type="so101", use_videos=True)

    for task, eps in plan:
        task_str = TASK_STRINGS[task]
        for d in eps:
            t = np.load(os.path.join(d, "traj.npz"))
            state, action = t["state"], t["action"]
            n = 0
            for i in range(len(state)):
                w = cv2.imread(os.path.join(d, f"cam0_{i:04d}.jpg"))
                s = cv2.imread(os.path.join(d, f"cam1_{i:04d}.jpg"))
                if w is None or s is None:
                    break
                w = cv2.resize(w, (W, H))
                s = cv2.resize(s, (W, H))
                ds.add_frame({
                    "observation.state": state[i].astype(np.float32),
                    "action": action[i].astype(np.float32),
                    "observation.images.wrist": cv2.cvtColor(w, cv2.COLOR_BGR2RGB),
                    "observation.images.scene": cv2.cvtColor(s, cv2.COLOR_BGR2RGB),
                    "task": task_str,
                })
                n += 1
            ds.save_episode()
            print(f"  {task}/{os.path.basename(d)}: {n}")

    ds.finalize()
    print(f"\ndone — {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"local: $HF_LEROBOT_HOME/{repo_id}")


if __name__ == "__main__":
    main()