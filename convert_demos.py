"""Convert one or more demo folders into a single multi-task LeRobotDataset.

  python convert_demos.py
"""
import glob, os
import cv2, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "Ved4nt/so101_multitask"
FPS = 30
W, H = 320, 240
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

# folder -> task string the policy is conditioned on
TASKS = {
    "demos/pick3":    "pick up the cube and place it in the tray",
    "demos/retrieve": "take the cube out of the tray and place it on the table",
}


def main():
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action":            {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "observation.images.wrist": {"dtype": "video", "shape": (H, W, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.images.scene": {"dtype": "video", "shape": (H, W, 3),
                                     "names": ["height", "width", "channels"]},
    }

    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=features,
                               robot_type="so101", use_videos=True)

    for src, task in TASKS.items():
        eps = sorted(glob.glob(os.path.join(src, "ep*")))
        print(f"\n{src}: {len(eps)} episodes -> \"{task}\"")
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
                    "task": task,
                })
                n += 1
            ds.save_episode()
            print(f"  {os.path.basename(d)}: {n}")

    ds.finalize()
    print(f"\ndone — {ds.num_episodes} episodes, {ds.num_frames} frames")


if __name__ == "__main__":
    main()1