"""Joint-space DualSense teleop for the SO-101 follower. No IK, no placo.

Sticks/triggers nudge joint targets directly:
  Left stick X  -> shoulder_pan      Left stick Y  -> shoulder_lift
  Right stick Y -> elbow_flex        Right stick X -> wrist_roll
  L2 / R2       -> wrist_flex down/up
  L1 / R1       -> gripper close/open
  PS button     -> quit

Run:  python teleop_joints_so101.py          (drive mode)
      python teleop_joints_so101.py --debug  (print axis/button numbers, arm idle)
"""

import sys
import time

import pygame

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ----------------------------- config -----------------------------
PORT = "COM3"
ROBOT_ID = "follower_arm"
FPS = 30

SPEED = 1.2       # degrees per tick at full stick, arm joints
GRIPPER_SPEED = 1.5

DEADZONE = 0.08

# DualSense via pygame on Windows — verify with --debug and adjust.
AXIS_LX, AXIS_LY, AXIS_RX, AXIS_RY = 0, 1, 2, 3
AXIS_L2, AXIS_R2 = 4, 5            # rest at -1.0, pressed -> +1.0
BTN_L1, BTN_R1 = 9, 10
BTN_PS = 5

LIMITS = {
    "shoulder_pan":  (-110, 110),
    "shoulder_lift": (-110, 110),
    "elbow_flex":    (-110, 110),
    "wrist_flex":    (-110, 110),
    "wrist_roll":    (-160, 160),
    "gripper":       (0, 100),
}
# ------------------------------------------------------------------


def dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v


def main():
    debug = "--debug" in sys.argv

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad found. Plug the DualSense in via USB.")
        return
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"Gamepad: {pad.get_name()} | axes={pad.get_numaxes()} buttons={pad.get_numbuttons()}")

    if debug:
        print("DEBUG MODE — arm will NOT move. Wiggle controls; Ctrl+C to exit.")
        while True:
            pygame.event.pump()
            axes = [round(pad.get_axis(a), 2) for a in range(pad.get_numaxes())]
            btns = [b for b in range(pad.get_numbuttons()) if pad.get_button(b)]
            print(f"axes={axes} buttons={btns}", end="      \r")
            time.sleep(0.05)

    follower = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True)
    )
    follower.connect()
    print("Arm connected.")

    obs = follower.get_observation()
    target = {m: float(obs[f"{m}.pos"]) for m in LIMITS if f"{m}.pos" in obs}
    missing = [m for m in LIMITS if m not in target]
    if missing:
        print("NOTE: motors not found in observation (check names):", missing)
    print("Targets initialized from current pose:", {k: round(v, 1) for k, v in target.items()})
    print("Driving. GENTLE inputs. PS button or Ctrl+C to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            pygame.event.pump()

            if pad.get_numbuttons() > BTN_PS and pad.get_button(BTN_PS):
                print("\nPS button — quitting.")
                break

            l2 = (pad.get_axis(AXIS_L2) + 1.0) / 2.0 if pad.get_numaxes() > AXIS_L2 else 0.0
            r2 = (pad.get_axis(AXIS_R2) + 1.0) / 2.0 if pad.get_numaxes() > AXIS_R2 else 0.0

            delta = {
                "shoulder_pan":  dz(pad.get_axis(AXIS_LX)) * SPEED,
                "shoulder_lift": -dz(pad.get_axis(AXIS_LY)) * SPEED,
                "elbow_flex":    -dz(pad.get_axis(AXIS_RY)) * SPEED,
                "wrist_roll":    dz(pad.get_axis(AXIS_RX)) * SPEED,
                "wrist_flex":    (r2 - l2) * SPEED,
                "gripper": (
                    (GRIPPER_SPEED if pad.get_button(BTN_R1) else 0.0)
                    - (GRIPPER_SPEED if pad.get_button(BTN_L1) else 0.0)
                ),
            }

            action = {}
            for m, d in delta.items():
                if m not in target:
                    continue
                lo, hi = LIMITS[m]
                target[m] = max(lo, min(hi, target[m] + d))
                action[f"{m}.pos"] = target[m]

            if action:
                follower.send_action(action)

            time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nCtrl+C — quitting.")
    finally:
        follower.disconnect()
        pygame.quit()
        print("Disconnected cleanly.")


if __name__ == "__main__":
    main()