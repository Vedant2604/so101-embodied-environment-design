"""Embodied Environment Design — curriculum machinery for the SO-101 push task.

Implements the designer objectives as end-of-episode reward bonuses on a
single goal-conditioned SAC that plays both directions (shared-parameter
two-agent; separate networks is a later ablation).

The embodied principle: in `progress` and `regret` modes the cube is NEVER
teleported — every start state is wherever the previous episode physically
left the cube. The designer bonus pays the finishing episode for WHERE it
left the cube, scored by what that start does for the opposite direction.

Modes
-----
none      : plain reset-free forward-backward (SERL-style baseline)
progress  : H1 — bonus = lambda * learning-progress of the OPPOSITE direction
            at the delivered bin (per-bin EMA success, fast vs slow EMA gap)
regret    : H2 — bonus = lambda * max(snapshot_success - current_success, 0)
            at the delivered bin, snapshot frozen every FREEZE_EVERY episodes
            (success-table proxy for a frozen-checkpoint antagonist; stated
            as such in any writeup)
vaprl     : disembodied oracle baseline — cube TELEPORTED each episode to the
            frontier bin chosen by the critic (value closest to the median of
            reachable-bin values). Curriculum WITHOUT a learning designer.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs"))

from push_env import (BIN_NX, BIN_NY, SO101PushEnv, bin_centre, xy_to_bin)

N_BINS = BIN_NX * BIN_NY


class BinStats:
    """Per-(direction, bin) success statistics with fast & slow EMAs."""

    def __init__(self, fast=0.05, slow=0.005):
        self.fast_a, self.slow_a = fast, slow
        # index 0 = direction A2B (target B), 1 = B2A (target A)
        self.fast = np.full((2, N_BINS), 0.0)
        self.slow = np.full((2, N_BINS), 0.0)
        self.counts = np.zeros((2, N_BINS), dtype=int)

    @staticmethod
    def dir_idx(target: str) -> int:
        return 0 if target == "B" else 1

    def update(self, target: str, start_bin: int, success: bool):
        d = self.dir_idx(target)
        s = float(success)
        self.fast[d, start_bin] += self.fast_a * (s - self.fast[d, start_bin])
        self.slow[d, start_bin] += self.slow_a * (s - self.slow[d, start_bin])
        self.counts[d, start_bin] += 1

    def progress(self, target: str, b: int) -> float:
        """Learning progress = |fast - slow| EMA gap (Oudeyer-style)."""
        d = self.dir_idx(target)
        return float(abs(self.fast[d, b] - self.slow[d, b]))

    def success_table(self, target: str) -> np.ndarray:
        return self.fast[self.dir_idx(target)].copy()


class EEDWrapper(gym.Wrapper):
    """Adds the designer bonus to the final transition of each episode.

    Works ONLY with mode='reset_free' underneath: episode k's final cube
    position IS episode k+1's start, so the finishing policy is the designer
    of the next start state.
    """

    def __init__(self, env: SO101PushEnv, objective: str = "none",
                 lam: float = 3.0, freeze_every: int = 200):
        assert objective in ("none", "progress", "regret")
        assert env.mode == "reset_free", "EEDWrapper requires reset_free"
        super().__init__(env)
        self.objective = objective
        self.lam = lam
        self.freeze_every = freeze_every
        self.stats = BinStats()
        self.snapshot = np.zeros((2, N_BINS))     # frozen success tables
        self._episodes_seen = 0
        self._start_bin = None
        self._start_target = None
        self.last_bonus = 0.0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._start_bin = xy_to_bin(self.env.unwrapped.cube_xy)
        self._start_target = self.env.unwrapped.target_name
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        done = term or trunc
        if done:
            # 1. update stats for the episode that just finished
            self.stats.update(self._start_target, self._start_bin,
                              info["success"])
            self._episodes_seen += 1
            if self._episodes_seen % self.freeze_every == 0:
                self.snapshot = np.stack([self.stats.success_table("B"),
                                          self.stats.success_table("A")])

            # 2. designer bonus: pay THIS episode for where it left the cube,
            #    scored for the NEXT direction (post-flip target)
            bonus = 0.0
            if self.objective != "none" and not info["intervention"]:
                delivered = info["cube_bin"]
                next_target = self.env.unwrapped.target_name  # already flipped
                if self.objective == "progress":
                    bonus = self.lam * self.stats.progress(next_target,
                                                           delivered)
                else:  # regret
                    d = BinStats.dir_idx(next_target)
                    cur = self.stats.fast[d, delivered]
                    reg = max(self.snapshot[d, delivered] - cur, 0.0)
                    bonus = self.lam * reg
            r += bonus
            self.last_bonus = bonus
            info["designer_bonus"] = bonus
            info["start_bin"] = self._start_bin
        return obs, r, term, trunc, info


class VaPRLTeleportWrapper(gym.Wrapper):
    """Disembodied-oracle baseline: teleports the cube each episode to the
    critic-frontier bin (value closest to the median across bins) for the
    current direction. Curriculum WITHOUT a learning designer, and WITHOUT
    the embodiment constraint.

    Call .attach(model) after creating the SAC model.
    """

    def __init__(self, env: SO101PushEnv, explore_eps: float = 0.2):
        assert env.mode == "reset_free"
        super().__init__(env)
        self.model = None
        self.explore_eps = explore_eps
        self.rng = np.random.default_rng(0)

    def attach(self, model):
        self.model = model

    def _bin_value(self, b: int) -> float:
        e = self.env.unwrapped
        # synthesise the observation the policy would see starting from bin b
        cube = bin_centre(b)
        tgt = e.target_xy
        from push_env import HOME
        obs = np.concatenate([HOME, np.zeros(6), cube, np.zeros(2),
                              tgt, tgt - cube]).astype(np.float32)
        import torch
        with torch.no_grad():
            o = torch.as_tensor(obs, device=self.model.device).unsqueeze(0)
            a = self.model.actor(o, deterministic=True)
            q = torch.min(*self.model.critic(o, a)) if isinstance(
                self.model.critic(o, a), tuple) else self.model.critic(o, a)[0]
        return float(q.squeeze().cpu())

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        e = self.env.unwrapped
        if self.model is not None:
            if self.rng.random() < self.explore_eps:
                b = int(self.rng.integers(N_BINS))
            else:
                vals = np.array([self._bin_value(b) for b in range(N_BINS)])
                b = int(np.argmin(np.abs(vals - np.median(vals))))
            e.set_start_state(bin_centre(b), e.target_name)
            obs = e._obs()
        return obs, info


def make_env(mode_flag: str, seed: int):
    """Factory used by train.py. mode_flag in
    {baseline, progress, regret, vaprl}."""
    base = SO101PushEnv(mode="reset_free", seed=seed)
    if mode_flag == "baseline":
        return EEDWrapper(base, objective="none")
    if mode_flag in ("progress", "regret"):
        return EEDWrapper(base, objective=mode_flag)
    if mode_flag == "vaprl":
        return VaPRLTeleportWrapper(base)
    raise ValueError(mode_flag)


if __name__ == "__main__":
    # ---- self-test: designer bonus mechanics in all modes ----
    bs = BinStats()
    for _ in range(50):
        bs.update("B", 10, True)
    print("progress while improving:", round(bs.progress("B", 10), 4), "(expect > 0.5)")

    env = make_env("progress", 0)
    env.reset()
    bonuses = []
    for i in range(3):
        tgt = env.env.unwrapped.target_xy.copy()
        env.env.unwrapped.set_start_state(tgt, env.env.unwrapped.target_name)
        o, r, te, tr, info = env.step(np.zeros(6))
        bonuses.append(info["designer_bonus"])
        env.reset()
    print("progress-mode bonuses over 3 forced successes:",
          [round(b, 3) for b in bonuses], "(expect 0, 0, then > 0)")

    envr = make_env("regret", 0)
    envr.reset()
    tgt = envr.env.unwrapped.target_xy.copy()
    envr.env.unwrapped.set_start_state(tgt, envr.env.unwrapped.target_name)
    o, r, te, tr, info = envr.step(np.zeros(6))
    b = info["cube_bin"]
    print("regret bonus pre-snapshot:", round(info["designer_bonus"], 4), "(expect 0)")
    for _ in range(200):
        envr.stats.update("B", b, True)
    envr.snapshot = np.stack([envr.stats.success_table("B"),
                              envr.stats.success_table("A")])
    for _ in range(200):
        envr.stats.update("B", b, False)
    d = BinStats.dir_idx("B")
    print("regret after decline vs snapshot:",
          round(max(envr.snapshot[d, b] - envr.stats.fast[d, b], 0.0), 3),
          "(expect ~1.0)")
    print("CURRICULUM SELF-TEST DONE")