import sys, os
sys.path.insert(0, os.path.join("so101", "envs"))
import numpy as np
from push_env import SO101PushEnv

env = SO101PushEnv(mode="reset_free", seed=0)
obs, _ = env.reset()
for i in range(5):
    for _ in range(300):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        if te or tr:
            break
    print(f"ep{i}: target={info['target']} cube={np.round(env.cube_xy,3)} "
          f"dist={info['dist']:.3f} interventions={info['interventions_total']}")
    obs, _ = env.reset()