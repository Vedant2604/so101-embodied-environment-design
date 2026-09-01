"""Check demo integrity.  python check_demos.py retrieve"""
import glob, os, sys
import numpy as np

task = sys.argv[1] if len(sys.argv) > 1 else "pick3"
dirs = sorted(glob.glob(f"demos/{task}/ep*"))
if not dirs:
    print(f"no episodes found in demos/{task}/")
    sys.exit(1)

total = 0
for d in dirs:
    t = np.load(os.path.join(d, "traj.npz"))
    n0 = len(glob.glob(os.path.join(d, "cam0_*.jpg")))
    n1 = len(glob.glob(os.path.join(d, "cam1_*.jpg")))
    n = len(t["state"])
    total += n
    flag = "" if n == n0 == n1 else "  <-- MISMATCH"
    print(f"{os.path.basename(d)}: {n} steps | cam0 {n0} cam1 {n1} | "
          f"gripper {t['state'][:,5].min():.0f}-{t['state'][:,5].max():.0f}{flag}")

print(f"\n{len(dirs)} episodes, {total} frames, mean {total//len(dirs)} per episode")