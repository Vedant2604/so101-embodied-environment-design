"""Run a trained ACT policy on the real SO-101.

  Dry run (no motion):  python eval_policy.py --ckpt outputs/act_pick/checkpoints/010000/pretrained_model
  Live:                 python eval_policy.py --ckpt ... --go --steps 1200
  With gripper assist:  python eval_policy.py --ckpt ... --go --steps 1200 --grip-open 30
"""
import argparse, time
import cv2, numpy as np, torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
LIMITS = {"shoulder_pan": (-110, 110), "shoulder_lift": (-110, 110),
          "elbow_flex": (-110, 110), "wrist_flex": (-110, 110),
          "wrist_roll": (-160, 160), "gripper": (0, 100)}
W, H = 320, 240          # must match training resolution
MAX_DELTA = 4.0          # deg per step safety clamp

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cams", type=int, nargs=2, default=[0, 2])
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--n-action-steps", type=int, default=0,
                help="override execution horizon (0 = use trained value, 100)")
ap.add_argument("--go", action="store_true", help="actually move the arm")
ap.add_argument("--grip-open", type=float, default=0.0,
                help="minimum gripper opening (0 = off, policy unmodified)")
ap.add_argument("--max-delta", type=float, default=4.0,
                help="per-step joint motion clamp in degrees")
ap.add_argument("--wb", type=int, default=6000,
                help="white balance temperature")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
policy = ACTPolicy.from_pretrained(args.ckpt).to(dev).eval()
ap_n = getattr(policy.config, "n_action_steps", None)
if args.n_action_steps > 0:
    policy.config.n_action_steps = args.n_action_steps
    print(f"n_action_steps {ap_n} -> {policy.config.n_action_steps}")
policy.reset()
print("queue maxlen:", getattr(policy, "_action_queue", None) and policy._action_queue.maxlen)
print(f"loaded {args.ckpt} on {dev}")

# preprocessor / postprocessor (normalization lives outside the model in 0.6.x)
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
if args.grip_open > 0:
    print(f"gripper assist ON (min opening {args.grip_open})")


def to_tensor(frame):
    img = cv2.resize(frame, (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(dev)


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
            "observation.images.wrist": to_tensor(frames[0]),
            "observation.images.scene": to_tensor(frames[1]),
            "task": "pick up the cube and place it in the tray",
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

        if args.grip_open > 0 and act[5] < args.grip_open:
            act[5] = args.grip_open

        if step % 20 == 0:
            print(f"{step:4d} cur {np.round(cur,1)} -> act {np.round(act,1)}")

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