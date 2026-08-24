"""Training for the SO-101 push task — Phase 0/1 and EED curricula.

  python so101/training/train.py --mode episodic   --steps 600000 --run p0_s0 --seed 0
  python so101/training/train.py --curriculum baseline --steps 600000 --run eed_base_s0 --seed 0
  python so101/training/train.py --curriculum progress --steps 600000 --run eed_h1_s0   --seed 0
  python so101/training/train.py --curriculum regret   --steps 600000 --run eed_h2_s0   --seed 0
  python so101/training/train.py --curriculum vaprl    --steps 600000 --run eed_vaprl_s0 --seed 0
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from push_env import ZONE_A, ZONE_B, ZONE_RADIUS, SO101PushEnv
from curriculum import make_env


def heldout_grid(zone_centre, n=5, spread=ZONE_RADIUS * 0.7):
    offs = np.linspace(-spread, spread, n)
    return [zone_centre + np.array([dx, dy]) for dx in offs for dy in offs]


GRID_A = heldout_grid(ZONE_A)
GRID_B = heldout_grid(ZONE_B)


def evaluate(model, seed=0, max_steps=300):
    env = SO101PushEnv(mode="episodic", seed=seed)
    results = {}
    for grid, target, tag in ((GRID_A, "B", "A2B"), (GRID_B, "A", "B2A")):
        wins = 0
        for pt in grid:
            env.reset()
            env.set_start_state(pt, target)
            obs = env._obs()
            for _ in range(max_steps):
                act, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                if info["success"]:
                    wins += 1
                    break
                if term or trunc:
                    break
        results[tag] = wins / len(grid)
    results["mean"] = (results["A2B"] + results["B2A"]) / 2
    return results


class LogCallback(BaseCallback):
    def __init__(self, run_dir, eval_every=25_000, verbose=0):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.run_dir = run_dir
        self.ep_rew, self.ep_len, self.ep_idx, self.ep_success = 0.0, 0, 0, 0
        self.t0 = time.time()
        os.makedirs(run_dir, exist_ok=True)
        self.ep_file = open(os.path.join(run_dir, "episodes.csv"), "w", newline="")
        self.ep_csv = csv.writer(self.ep_file)
        self.ep_csv.writerow(["episode", "steps", "return", "success", "target",
                              "start_bin", "delivered_bin", "designer_bonus",
                              "intervention_kind", "interventions_total",
                              "wallclock_s", "global_step"])
        self.ev_file = open(os.path.join(run_dir, "eval.csv"), "w", newline="")
        self.ev_csv = csv.writer(self.ev_file)
        self.ev_csv.writerow(["global_step", "A2B", "B2A", "mean"])
        self._next_eval = eval_every

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.ep_rew += float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
        self.ep_len += 1
        if info.get("success"):
            self.ep_success = 1

        if bool(np.asarray(self.locals["dones"]).reshape(-1)[0]):
            self.ep_idx += 1
            self.ep_csv.writerow([
                self.ep_idx, self.ep_len, round(self.ep_rew, 3), self.ep_success,
                info.get("target", ""), info.get("start_bin", ""),
                info.get("cube_bin", ""), round(info.get("designer_bonus", 0.0), 4),
                info.get("intervention_kind", ""), info.get("interventions_total", 0),
                round(time.time() - self.t0, 1), self.num_timesteps])
            self.ep_file.flush()
            self.logger.record("episode/success", self.ep_success)
            self.logger.record("episode/return", self.ep_rew)
            self.logger.record("episode/designer_bonus", info.get("designer_bonus", 0.0))
            self.logger.record("episode/interventions", info.get("interventions_total", 0))
            self.ep_rew, self.ep_len, self.ep_success = 0.0, 0, 0

        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            res = evaluate(self.model)
            self.ev_csv.writerow([self.num_timesteps, res["A2B"], res["B2A"], res["mean"]])
            self.ev_file.flush()
            for k, v in res.items():
                self.logger.record(f"heldout/{k}", v)
            print(f"[eval @ {self.num_timesteps}] A2B={res['A2B']:.2f} "
                f"B2A={res['B2A']:.2f} mean={res['mean']:.2f}")
            self.model.save(os.path.join(self.run_dir, f"ckpt_{self.num_timesteps}"))
        return True

    def _on_training_end(self):
        self.ep_file.close()
        self.ev_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default=None, choices=["episodic", "reset_free"])
    p.add_argument("--curriculum", default=None,
                   choices=["baseline", "progress", "regret", "vaprl"])
    p.add_argument("--steps", type=int, default=600_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default=None)
    p.add_argument("--reward", default="dense", choices=["dense", "sparse"])
    p.add_argument("--target-entropy", type=float, default=-2.0)
    args = p.parse_args()

    if (args.mode is None) == (args.curriculum is None):
        raise SystemExit("Pass exactly one of --mode or --curriculum")

    run = args.run or (args.curriculum or args.mode) + f"_s{args.seed}"
    run_dir = os.path.join("runs", run)
    os.makedirs(run_dir, exist_ok=True)

    if args.curriculum:
        env = make_env(args.curriculum, seed=args.seed)
    else:
        env = SO101PushEnv(mode=args.mode, reward_type=args.reward, seed=args.seed)

    model = SAC("MlpPolicy", env,
                learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                learning_starts=5_000, train_freq=1, gradient_steps=1,
                ent_coef="auto_0.5", target_entropy=args.target_entropy,
                seed=args.seed, verbose=1, device="cuda")

    if hasattr(env, "attach"):
        env.attach(model)

    model.learn(total_timesteps=args.steps, callback=LogCallback(run_dir))
    model.save(os.path.join(run_dir, "model"))
    print("FINAL held-out:", evaluate(model))


if __name__ == "__main__":
    main()