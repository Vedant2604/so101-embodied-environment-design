import sys, os
sys.path.insert(0, os.path.join("so101", "envs"))
import mujoco, numpy as np
from push_env import SO101PushEnv
env = SO101PushEnv(seed=0); env.reset()
m = env.model
for i in range(m.nbody):
    print("body", i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i),
          np.round(env.data.xpos[i], 3))
print("---sites---")
for i in range(m.nsite):
    print("site", i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i),
          np.round(env.data.site_xpos[i], 3))