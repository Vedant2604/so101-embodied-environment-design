# Autonomous manipulation on a self-built $200 robot arm

Teaching a low-cost 6-DOF arm to pick up a cube and place it in a tray, from
human demonstrations — and working toward a robot that keeps improving on its
own after deployment.

> **Status:** the behaviour-cloning policy performs the full pick-and-place
> autonomously on hardware, succeeding in **8 of 20** trials at varied cube
> positions with no human assistance.

*(video / GIF to be added)*

---

## Why this project

Almost all real-world robot learning research runs on hardware costing tens of
thousands of dollars, in labs with people available to reset the scene between
attempts. Two things follow: the results are hard for anyone else to reproduce,
and the methods quietly assume a human is standing by.

This project asks what is achievable on a **self-built arm costing about $200**,
and uses it to work toward the question that motivated it: **how should a robot
generate its own practice conditions once nobody is there to reset it?**

## The platform

| | |
|---|---|
| Arm | SO-101 follower, 6 DOF, 3D-printed frame, 6× Feetech STS3215 servos |
| Teleoperation | PS5 DualSense → joint-space control |
| Cameras | 2× 720p webcams — one wrist-mounted, one fixed third-person |
| Compute | single consumer GPU (RTX 2060, 6 GB) |
| Simulation | MuJoCo, using the SO-ARM100 model from DeepMind Menagerie |

Total hardware cost: roughly ₹15,000–18,000 / ~$200.

## What is here

```
so101/envs/push_env.py        MuJoCo two-zone push environment (episodic + reset-free)
so101/training/curriculum.py  curriculum wrappers: learning-progress, regret, value-picked
so101/training/train.py       SAC training and held-out-grid evaluation
record_demo.py                teleoperated demonstration recorder (joints + 2 cameras)
convert_demos.py              raw demos → LeRobotDataset, video-encoded
eval_policy.py                run a trained ACT policy on the physical arm
LOG.md                        full chronological record, including what failed
```

Dataset: [`Ved4nt/so101_pick3`](https://huggingface.co/datasets/Ved4nt/so101_pick3)
— 51 demonstrations, 26,483 frames, two camera streams.

## Method

1. **Record** ~50 teleoperated demonstrations of the task, capturing joint
   positions, commanded actions, and both camera streams at 30 fps.
2. **Convert** to a LeRobotDataset with MP4 video encoding.
3. **Train** an ACT policy (52M parameters, ResNet18 backbone, action chunks of
   100 steps) on a single consumer GPU.
4. **Deploy** on the physical arm, closing the loop at 30 Hz from camera frames
   and joint feedback.

## Results

Trained on 51 demonstrations, tested on hardware at matched cube positions:

| checkpoint | behaviour on the real arm |
|---|---|
| 20k steps | approaches accurately, cannot complete the grasp |
| 40k steps | correct overhead approach, grasp unreliable |
| 60k steps | **completes the task autonomously** — approach, grasp, transport, release |
| **100k steps** | **best policy** — same task, smooth and unaided, in ~500 steps |

The determining factor was **training duration**: the same dataset produced a
policy that needed manual assistance at 20k and 40k steps, one that succeeded
unaided at 60k, and a faster and smoother one at 100k.

### Measured success rate

**8 / 20 (40%) fully autonomous successes**, evaluated on the 100k checkpoint at
varied cube positions, with no human assistance and the arm returned to a
consistent start pose between trials.

Failures were dominated by a single compounding mode:

1. The grasp closes 1–2 cm short of the cube and misses.
2. The gripper is now at cube height, so the retry *pushes the cube* out of
   position rather than clearing it.
3. The policy then re-approaches the cube's **original** location — it is acting
   on a target position it no longer occupies.

Each step of that chain is a consequence of the training data: the
demonstrations contain no missed grasps, so neither recovery nor re-localisation
after a disturbance was ever demonstrated.

### Findings worth recording

**Dataset encoding decided whether the necessary training budget was reachable
at all.** Stored as images, the dataset made training dataloader-bound — the GPU
sat at 0% utilisation while CPUs decoded JPEGs, and a 100k-step run would have
taken over two days. Video encoding did not make the policy better; it made the
training budget that produces a working policy practical on one consumer GPU.

| | images | video |
|---|---|---|
| dataset size | 4.1 GB | 212 MB |
| dataloader wait per step | 1.5 s | 0.003 s |
| throughput | 4 samples/s | 44 samples/s |
| projected time to 100k steps | ~50 h | ~12 h |

**Validation loss was anti-correlated with real-world success.** Evaluation loss
rose steadily — ~0.26 at 20k, 0.2979 at 70k, 0.3014 at 100k — while training
loss fell to 0.027, the standard overfitting signature. Over exactly that
interval, performance on the physical robot improved at every checkpoint.
Training was in fact halted early at 60k on the strength of that metric;
resuming to 100k produced the best policy of the run. With a small held-out set,
the offline metric would have selected the worst checkpoint available.

**Demonstration consistency mattered as much as dataset size.** An early policy
systematically pushed the cube out of position before grasping it, because the
demonstrations mixed shallow and overhead approaches and the policy learned the
average. Re-recording with a strict hover-then-descend protocol fixed the
behaviour.

## Reproducing

```bash
# 1. record demonstrations (gamepad + arm connected)
python record_demo.py --task pick --cams 0 1

# 2. convert to a video-encoded LeRobot dataset
python convert_demos.py

# 3. train ACT
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=Ved4nt/so101_pick3 --policy.type=act \
  --output_dir=outputs/act_pick --steps=100000 --batch_size=20

# 4. run the policy on the arm
python eval_policy.py --ckpt outputs/act_pick/checkpoints/060000/pretrained_model \
  --cams 0 1 --go --steps 800 --max-delta 8
```

`eval_policy.py` defaults to a dry run that prints predicted actions without
moving the arm — check those look sane before passing `--go`.

## Multi-task: two negative results

Extending to a second task (`retrieve`: tray -> table) so the tasks reset each
other produced two informative failures, both documented in full in `LOG.md`.

**Multi-task ACT hit gradient interference.** Task conditioning worked — the
same weights behaved differently for each task string — but `place` degraded
badly relative to its single-task performance. This is the known
"similar-input, different-output" dilemma: mid-episode, the two tasks look
nearly identical and demand opposite actions, so the network averages them.

**Multi-task SmolVLA hit a hardware ceiling.** A 450M pretrained VLA fine-tuned
on the same data reached the lowest evaluation loss of any policy here (0.2270,
in 4 epochs) and still failed on the robot — because a 6 GB GPU only fits batch
10, giving 0.2M samples against ACT's 4.8M. Matching that exposure would take
roughly 16 days of continuous training. The model fits; the batch size the card
permits does not reach the sample counts the method needs.

Both are constraints of the hardware tier rather than of the methods, and both
are the kind of finding that only surfaces when the platform is genuinely cheap.

## Known limitations

- **No recovery from failure.** If a grasp misses, the policy re-attempts at the
  same position instead of lifting and re-approaching — and because the gripper
  is already at cube height, the retry pushes the cube away. It then reaches for
  where the cube used to be. Every recorded demonstration succeeded, so neither
  recovery nor re-localisation after a disturbance was ever demonstrated. This
  compounding failure accounts for most of the 12 unsuccessful trials.
- **No termination condition.** Given surplus steps after a success, the policy
  keeps acting and re-picks the cube from the tray. Demonstrations always ended
  at the drop.
- **Hardware degradation.** Over a week of operation the wrist-roll servo
  produced three bus faults and the gripper servo thermally overloaded. This
  bounds how long unattended autonomous operation can realistically run on this
  class of hardware.

## Direction

Behaviour cloning gives a policy that mostly works. It does not improve with
practice, and it cannot recover from its own mistakes — both because the
demonstrations contain no mistakes to learn from. The next stage is **autonomous practice**: the robot attempting the
task without human resets, scoring its own attempts from vision, and improving
through reinforcement learning on top of the demonstration-trained policy — with
the number of human interventions logged as the measure of real autonomy.

The simulation work in `so101/` is the first step toward the question underneath
that: when a robot resets itself, the reset is not neutral — it decides what the
robot practises next. Whether that mechanism should have an objective of its own
is what this project is ultimately about.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot), the
[SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) open hardware
design, and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
The ACT policy is from *Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware* (Zhao et al., 2023).
