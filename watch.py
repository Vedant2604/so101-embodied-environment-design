"""Watch a trained policy run in the MuJoCo viewer.

    python watch.py                         # defaults to runs/p0_bidir/model
    python watch.py runs/p0_bidir_s1/model  # any saved model
"""

import os
import sys
import time

sys.path.insert(0, os.path.join("so101", "envs"))

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from push_env import SO101PushEnv

model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/p0_bidir/model"
SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
env = SO101PushEnv(mode="episodic", seed=0)
model = SAC.load(model_path, device="cpu")
print("loaded:", model_path)

obs, _ = env.reset()
ep, wins = 0, 0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        viewer.sync()
        time.sleep(0.02/SPEED)          # slow to watchable speed

        if term or trunc:
            ep += 1
            wins += int(info["success"])
            print(f"episode {ep}: target={info['target']} "
                  f"success={info['success']} | running {wins}/{ep}")
            obs, _ = env.reset()
            time.sleep(0.5)