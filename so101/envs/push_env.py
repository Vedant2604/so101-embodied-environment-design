"""Two-zone cube-push environment for the SO-101 (MuJoCo)."""

from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

DEFAULT_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "mujoco_menagerie", "trs_so_arm100", "arena_so101.xml"
)

TRAY_CENTRE = np.array([0.0, -0.30])
ZONE_A = np.array([-0.075, -0.30])
ZONE_B = np.array([0.075, -0.30])
ZONE_RADIUS = 0.04
CUBE_Z = 0.03

HOME = np.array([-0.001, -1.092, 1.978, -0.635, -0.872, 1.226])

ACTION_SCALE = 0.15
REACH_WEIGHT = 0.5
SUCCESS_BONUS = 10.0
TIP_BONUS = 0.05

ESCAPE_X = 0.14
ESCAPE_Y = 0.030
STUCK_STEPS = 300
STUCK_DIST = 0.01

# --- start-state bin grid (8 x 4 = 32 bins over the tray interior) ---------
BIN_NX, BIN_NY = 8, 4
BIN_X_HALF, BIN_Y_HALF = 0.1125, 0.0375   # reachable cube area, rel. tray centre


def xy_to_bin(xy):
    """Absolute cube xy -> bin index in [0, BIN_NX*BIN_NY)."""
    rx = np.clip((xy[0] - TRAY_CENTRE[0] + BIN_X_HALF) / (2 * BIN_X_HALF), 0, 0.999)
    ry = np.clip((xy[1] - TRAY_CENTRE[1] + BIN_Y_HALF) / (2 * BIN_Y_HALF), 0, 0.999)
    return int(ry * BIN_NY) * BIN_NX + int(rx * BIN_NX)


def bin_centre(idx):
    """Bin index -> absolute xy of the bin centre."""
    iy, ix = divmod(int(idx), BIN_NX)
    x = TRAY_CENTRE[0] - BIN_X_HALF + (ix + 0.5) * (2 * BIN_X_HALF) / BIN_NX
    y = TRAY_CENTRE[1] - BIN_Y_HALF + (iy + 0.5) * (2 * BIN_Y_HALF) / BIN_NY
    return np.array([x, y])


JAW_GEOMS = [f"{s}_jaw_pad_{i}" for s in ("fixed", "moving") for i in range(1, 5)]


class SO101PushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, xml_path=DEFAULT_XML, mode="episodic", reward_type="dense",
                 max_episode_steps=300, frame_skip=10, start_zone="A",
                 randomize_start=True, seed=None):
        assert mode in ("episodic", "reset_free")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.mode = mode
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.randomize_start = randomize_start

        self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.grip_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.jaw_gids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in JAW_GEOMS
        }
        self.jaw_gids.discard(-1)

        self.cube_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.cube_dof_adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.nu = self.model.nu
        self.ctrlrange = self.model.actuator_ctrlrange.copy()

        self.target_name = "B" if start_zone == "A" else "A"
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)

        self._steps = 0
        self._renderer = None
        self._stuck_ref = None
        self._stuck_count = 0

        self.episode_count = 0
        self.intervention_count = 0

    @property
    def target_xy(self):
        return ZONE_A if self.target_name == "A" else ZONE_B

    @property
    def cube_xy(self):
        return self.data.xpos[self.cube_bid][:2].copy()

    @property
    def grip_xy(self):
        return self.data.xpos[self.grip_bid][:2].copy()

    def _tip_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == self.cube_gid and c.geom2 in self.jaw_gids) or \
               (c.geom2 == self.cube_gid and c.geom1 in self.jaw_gids):
                return True
        return False

    def _obs(self):
        qpos = self.data.qpos[: self.nu].copy()
        qvel = self.data.qvel[: self.nu].copy()
        cube = self.cube_xy
        cube_v = self.data.cvel[self.cube_bid][3:5].copy()
        tgt = self.target_xy
        return np.concatenate([qpos, qvel, cube, cube_v, tgt, tgt - cube]).astype(np.float32)

    def _success(self):
        return bool(np.linalg.norm(self.cube_xy - self.target_xy) < ZONE_RADIUS)

    def _go_home(self):
        self.data.qpos[: self.nu] = HOME
        self.data.qvel[: self.nu] = 0.0
        self.data.ctrl[: self.nu] = HOME

    def _place_cube(self, zone):
        centre = ZONE_A if zone == "A" else ZONE_B
        if self.randomize_start:
            r = ZONE_RADIUS * 0.6 * np.sqrt(self.rng.random())
            th = self.rng.random() * 2 * np.pi
            xy = centre + np.array([r * np.cos(th), r * np.sin(th)])
        else:
            xy = centre
        a = self.cube_qpos_adr
        self.data.qpos[a: a + 3] = [xy[0], xy[1], CUBE_Z]
        self.data.qpos[a + 3: a + 7] = [1, 0, 0, 0]
        self.data.qvel[self.cube_dof_adr: self.cube_dof_adr + 6] = 0.0

    def _cube_escaped(self):
        c = self.cube_xy
        z = self.data.xpos[self.cube_bid][2]
        return bool(abs(c[0] - TRAY_CENTRE[0]) > ESCAPE_X
                    or abs(c[1] - TRAY_CENTRE[1]) > ESCAPE_Y
                    or z < 0.005)

    def _cube_stuck(self):
        c = self.cube_xy
        if self._stuck_ref is None:
            self._stuck_ref = c
            self._stuck_count = 0
            return False
        if np.linalg.norm(c - self._stuck_ref) > STUCK_DIST:
            self._stuck_ref = c
            self._stuck_count = 0
            return False
        self._stuck_count += 1
        return self._stuck_count >= STUCK_STEPS

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.mode == "episodic" or self.episode_count == 0:
            mujoco.mj_resetData(self.model, self.data)
            if self.mode == "episodic" and self.episode_count > 0:
                self.target_name = "A" if self.target_name == "B" else "B"
            self._place_cube("A" if self.target_name == "B" else "B")
            self._stuck_ref, self._stuck_count = None, 0
        self._go_home()
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self.episode_count += 1
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.data.ctrl[: self.nu] + action * ACTION_SCALE
        self.data.ctrl[: self.nu] = np.clip(ctrl, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        success = self._success()
        dist = float(np.linalg.norm(self.cube_xy - self.target_xy))
        d_reach = float(np.linalg.norm(self.grip_xy - self.cube_xy))
        tip = self._tip_contact()

        if self.reward_type == "dense":
            reward = (-dist - REACH_WEIGHT * d_reach
                      + (TIP_BONUS if tip else 0.0)
                      + (SUCCESS_BONUS if success else 0.0))
        else:
            reward = 1.0 if success else 0.0

        escaped = self._cube_escaped()
        stuck = self._cube_stuck() and not success
        intervened = escaped or stuck
        if intervened:
            self.intervention_count += 1

        terminated = False
        truncated = self._steps >= self.max_episode_steps

        if success:
            if self.mode == "reset_free":
                self.target_name = "A" if self.target_name == "B" else "B"
                truncated = True
            else:
                terminated = True
            self._stuck_ref, self._stuck_count = None, 0

        if intervened:
            self._place_cube("A" if self.target_name == "B" else "B")
            mujoco.mj_forward(self.model, self.data)
            self._stuck_ref, self._stuck_count = None, 0
            truncated = True

        info = {"success": success, "dist": dist, "reach": d_reach, "tip": tip,
                "target": self.target_name, "intervention": intervened,
                "intervention_kind": ("escape" if escaped else "stuck" if stuck else ""),
                "cube_bin": xy_to_bin(self.cube_xy),
                "interventions_total": self.intervention_count}
        return self._obs(), float(reward), terminated, truncated, info

    def set_start_state(self, cube_xy, target):
        a = self.cube_qpos_adr
        self.data.qpos[a: a + 3] = [cube_xy[0], cube_xy[1], CUBE_Z]
        self.data.qpos[a + 3: a + 7] = [1, 0, 0, 0]
        self.data.qvel[self.cube_dof_adr: self.cube_dof_adr + 6] = 0.0
        self.target_name = target
        self._stuck_ref, self._stuck_count = None, 0
        self._go_home()
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera="top_view")
        return self._renderer.render()


if __name__ == "__main__":
    # ---- self-test ----
    ok = all(xy_to_bin(bin_centre(i)) == i for i in range(BIN_NX * BIN_NY))
    print("bin round-trip all 32:", ok,
          "| zone A bin:", xy_to_bin(ZONE_A), "| zone B bin:", xy_to_bin(ZONE_B))

    env = SO101PushEnv(mode="episodic", seed=0)
    obs, _ = env.reset()
    print("obs dim:", obs.shape, "act dim:", env.action_space.shape)

    total, min_reach, tips = 0.0, 9.9, 0
    for i in range(600):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        min_reach = min(min_reach, info["reach"])
        tips += int(info["tip"])
        if term or trunc:
            print(f"ep end at {i}: success={info['success']} dist={info['dist']:.3f} "
                  f"interventions={info['interventions_total']} kind={info['intervention_kind']}")
            env.reset()
    print("random return:", round(total, 2), "| closest reach:", round(min_reach, 3),
          "| tip-contact steps:", tips)