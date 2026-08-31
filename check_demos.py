import glob, os, numpy as np, cv2
for d in sorted(glob.glob("demos/pick3/ep*")):
    t = np.load(os.path.join(d, "traj.npz"))
    n0 = len(glob.glob(os.path.join(d, "cam0_*.jpg")))
    n1 = len(glob.glob(os.path.join(d, "cam1_*.jpg")))
    print(f"{d}: {len(t['state'])} steps | cam0 {n0} cam1 {n1} | "
          f"gripper {t['state'][:,5].min():.0f}-{t['state'][:,5].max():.0f}")
mid = sorted(glob.glob("demos/pick/ep000/cam*_0050.jpg"))
for p in mid:
    cv2.imshow(p, cv2.imread(p))
cv2.waitKey(0)