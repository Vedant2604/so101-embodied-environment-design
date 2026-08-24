import sys, os
sys.path.insert(0, "so101/envs")
import numpy as np
from push_env import SO101PushEnv, HOME

env = SO101PushEnv(seed=0)
env.reset()
print("start   reach:", round(np.linalg.norm(env.grip_xy - env.cube_xy), 3))

# hold still for 100 steps — does reward stay flat?
for _ in range(100):
    obs, r, *_ , info = env.step(np.zeros(6))
print("after hold  reach:", round(info["reach"], 3), "reward:", round(r, 2))

# push toward the cube: which joint moves gripper in -x?
for j in range(6):
    env.reset()
    a = np.zeros(6); a[j] = 1.0
    for _ in range(50):
        obs, r, *_, info = env.step(a)
    print(f"joint {j} +1 for 50 steps -> reach {info['reach']:.3f}, grip {np.round(env.grip_xy,3)}")

env.reset()
print("HOME grip xyz:", np.round(env.data.xpos[env.grip_bid], 3))
a = np.zeros(6); a[0] = -1.0
for _ in range(50):
    obs, r, *_, info = env.step(a)
print("joint0 -1:", np.round(env.data.xpos[env.grip_bid], 3))    