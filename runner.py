
"""Autonomous multi-task practice session for the SO-101.
 
Runs episodes back to back without human resets. A scheduler picks which task
to attempt from the current world state; the detector scores the outcome and
flags interventions. Every episode is logged with its cost.
 
  Dry run (no motion):  python runner.py --ckpt <path> --cams 0 1
  Live:                 python runner.py --ckpt <path> --cams 0 1 --go
  Time-boxed:           python runner.py --ckpt <path> --cams 0 1 --go --minutes 60
"""
import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
 
from detector import CubeDetector
from lerobot.policies.act.modeling_act import ACTPolicy
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
# task -> (required start region, goal region)
TASK_SPEC = {
    "place":    ("table", "tray"),
    "retrieve": ("tray", "table"),
}
W, H = 320, 240
 
 
# --------------------------------------------------------------------------
# schedulers
# --------------------------------------------------------------------------
class FeasibleScheduler:
    """Hand-designed: run whichever task's precondition currently holds.
 
    This is the Gupta et al. (ICRA 2021) pattern — tasks reset each other,
    and the sequencer is engineered rather than learned. It is the baseline
    every learned scheduler has to beat.
    """
    name = "feasible"
 
    def feasible(self, ws):
        return [t for t, (start, _) in TASK_SPEC.items() if ws.region == start]
 
    def select(self, ws, history):
        f = self.feasible(ws)
        return f[0] if f else None
 
    def update(self, *a, **kw):
        pass
 
 
class AlternateScheduler:
    """Strict alternation, ignoring world state. Will stall — included as a
    control showing why feasibility matters."""
    name = "alternate"
 
    def __init__(self):
        self.i = 0
 
    def feasible(self, ws):
        return [t for t, (start, _) in TASK_SPEC.items() if ws.region == start]
 
    def select(self, ws, history):
        t = list(TASKS)[self.i % len(TASKS)]
        self.i += 1
        return t if t in self.feasible(ws) else None
 
    def update(self, *a, **kw):
        pass
 
 
SCHEDULERS = {"feasible": FeasibleScheduler, "alternate": AlternateScheduler}
 
 
# --------------------------------------------------------------------------
# cost ledger
# --------------------------------------------------------------------------
class Ledger:
    def __init__(self):
        self.episodes = 0
        self.successes = 0
        self.interventions = 0
        self.seconds = 0.0
        self.joint_travel = 0.0
        self.t0 = time.time()
 
    def charge(self, outcome):
        self.episodes += 1
        self.successes += int(outcome["success"])
        self.interventions += int(outcome["intervention"])
        self.seconds += outcome["seconds"]
        self.joint_travel += outcome["joint_travel"]
 
    @property
    def elapsed_min(self):
        return (time.time() - self.t0) / 60.0
 
    def summary(self):
        rate = self.successes / max(self.episodes, 1)
        per_hr = self.interventions / max(self.elapsed_min / 60.0, 1e-6)
        return (f"{self.episodes} episodes | {self.successes} success "
                f"({rate:.0%}) | {self.interventions} interventions "
                f"({per_hr:.1f}/hr) | {self.elapsed_min:.1f} min")
 
 
# --------------------------------------------------------------------------
def to_tensor(frame, dev):
    img = cv2.resize(frame, (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(dev)
 
 
def run_episode(task, policy, pre, post, arm, caps, det, args, dev):
    """One attempt. Terminates on detected goal, step cap, or lost cube."""
    policy.reset()
    goal = TASK_SPEC[task][1]
    t_start = time.perf_counter()
    travel = 0.0
    lost_since = None
    settle_n, settle_xy = 0, None
    success = False
 
    for step in range(args.max_steps):
        t0 = time.perf_counter()
        obs = arm.get_observation()
        cur = np.array([float(obs[f"{m}.pos"]) for m in JOINTS], dtype=np.float32)
 
        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            continue
 
        ws, dbg = det.find(frames[1])
 
        # goal reached, and the cube has settled (not still being carried:
        # a side-on camera projects a cube held in the air into the tray region)
        if ws.region == goal and ws.confident:
            if settle_xy is not None:
                moved = abs(ws.xy[0] - settle_xy[0]) + abs(ws.xy[1] - settle_xy[1])
                settle_n = settle_n + 1 if moved < args.settle_px else 0
            settle_xy = ws.xy
            if settle_n >= args.settle_frames:
                success = True
                break
        else:
            settle_n, settle_xy = 0, None
 
        # cube gone for too long?
        if ws.region == "lost":
            lost_since = lost_since or time.perf_counter()
            if time.perf_counter() - lost_since > args.lost_timeout:
                break
        else:
            lost_since = None
 
        batch = {
            "observation.state": torch.from_numpy(cur).unsqueeze(0).to(dev),
            "observation.images.wrist": to_tensor(frames[0], dev),
            "observation.images.scene": to_tensor(frames[1], dev),
            "task": TASKS[task],
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
 
        travel += float(np.abs(act - cur).sum())
 
        if args.go:
            arm.send_action({f"{m}.pos": float(act[i]) for i, m in enumerate(JOINTS)})
 
        cv2.putText(dbg, f"{task}  step {step}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("session", dbg)
        if cv2.waitKey(1) == 27:
            raise KeyboardInterrupt
 
        time.sleep(max(1 / 30 - (time.perf_counter() - t0), 0))
 
    return {
        "task": task,
        "success": success,
        "steps": step + 1,
        "seconds": time.perf_counter() - t_start,
        "joint_travel": travel,
        "intervention": False,
    }
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--scheduler", default="feasible", choices=list(SCHEDULERS))
    ap.add_argument("--go", action="store_true", help="actually move the arm")
    ap.add_argument("--minutes", type=float, default=0, help="0 = until Ctrl+C")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--max-delta", type=float, default=8.0)
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between episodes")
    ap.add_argument("--lost-timeout", type=float, default=10.0)
    ap.add_argument("--settle-frames", type=int, default=15,
                    help="frames the cube must sit still in the goal region")
    ap.add_argument("--settle-px", type=float, default=6.0,
                    help="max pixel movement per frame to count as settled")
    ap.add_argument("--log", default="runs/session.csv")
    args = ap.parse_args()
 
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt).resolve()
    policy = ACTPolicy.from_pretrained(ckpt).to(dev).eval()
    print(f"loaded {ckpt} on {dev}")
 
    pre = post = None
    try:
        from lerobot.processor import PolicyProcessorPipeline
        pre = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_preprocessor.json")
        post = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_postprocessor.json")
        print("loaded processor pipeline")
    except Exception as e:
        print("no external processors:", type(e).__name__)
 
    det = CubeDetector()
    sched = SCHEDULERS[args.scheduler]()
 
    caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in args.cams]
    for c in caps:
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    arm = SO101Follower(SO101FollowerConfig(port="COM3", id="follower_arm",
                                            use_degrees=True))
    arm.connect()
    print(f"scheduler={sched.name} |",
          "LIVE — arm will move" if args.go else "DRY RUN — no motion")
 
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    new = not os.path.exists(args.log)
    logf = open(args.log, "a", newline="")
    log = csv.writer(logf)
    if new:
        log.writerow(["episode", "wallclock_min", "scheduler", "task",
                      "start_region", "success", "steps", "seconds",
                      "joint_travel", "intervention", "cum_successes",
                      "cum_interventions"])
 
    ledger = Ledger()
    history = []
 
    try:
        while True:
            if args.minutes and ledger.elapsed_min >= args.minutes:
                print("\ntime budget reached")
                break
 
            # release anything held, settle, then read the world
            time.sleep(args.pause)
            if args.go:
                o = arm.get_observation()
                a = {f"{m}.pos": float(o[f"{m}.pos"]) for m in JOINTS}
                a["gripper.pos"] = 40.0
                arm.send_action(a)
                time.sleep(1.0)
            ok, frame = caps[1].read()
            ws, dbg = det.find(frame) if ok else (None, None)
 
            if ws is None or not sched.feasible(ws):
                ledger.interventions += 1
                print(f"\n[INTERVENTION #{ledger.interventions}] "
                      f"cube region = {ws.region if ws else 'no frame'}. "
                      f"Place the cube on the table or in the tray.")
                log.writerow([ledger.episodes, round(ledger.elapsed_min, 2),
                              sched.name, "", ws.region if ws else "", "", "",
                              "", "", 1, ledger.successes, ledger.interventions])
                logf.flush()

                stable = 0
                while True:
                    ok2, f2 = caps[1].read()
                    if ok2:
                        ws2, d2 = det.find(f2)
                        cv2.putText(d2, "INTERVENTION - place the cube",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)
                        cv2.imshow("session", d2)
                        if ws2.confident and ws2.region in ("table", "tray"):
                            stable += 1
                            if stable > 45:      # ~1.5 s of stable detection
                                print("  scene recovered, resuming")
                                break
                        else:
                            stable = 0
                    k = cv2.waitKey(30)
                    if k == 27:
                        raise KeyboardInterrupt
                continue
 
            task = sched.select(ws, history)
            if task is None:
                continue
 
            print(f"\nep {ledger.episodes:3d} | cube {ws.region} -> task {task}")
            outcome = run_episode(task, policy, pre, post, arm, caps,
                                  det, args, dev)
            outcome["start_region"] = ws.region
            ledger.charge(outcome)
            history.append(outcome)
            sched.update(ws, task, outcome)
 
            print(f"  {'SUCCESS' if outcome['success'] else 'fail   '} "
                  f"in {outcome['steps']} steps, {outcome['seconds']:.0f}s "
                  f"| {ledger.summary()}")
 
            log.writerow([ledger.episodes, round(ledger.elapsed_min, 2),
                          sched.name, task, ws.region,
                          int(outcome["success"]), outcome["steps"],
                          round(outcome["seconds"], 1),
                          round(outcome["joint_travel"], 1), 0,
                          ledger.successes, ledger.interventions])
            logf.flush()
 
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        print("\nSESSION:", ledger.summary())
        logf.close()
        arm.disconnect()
        for c in caps:
            c.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
