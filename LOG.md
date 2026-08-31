# Project log — SO-101 autonomous manipulation

A running record of what was tried, what broke, and what the evidence said.
Written as it happened, including the dead ends.

---

## Phase A — Simulation (MuJoCo)

**Goal.** Build a reset-free RL testbed: a two-zone push task where the robot
practises without human resets, as a platform for studying whether the *reset
mechanism itself* should be a learning agent.

**What was built.** A MuJoCo arena with the SO-101 arm, a walled tray, two goal
zones and a cube; a Gymnasium environment with episodic and reset-free modes;
an intervention counter; and curriculum wrappers implementing three designer
objectives (learning-progress, regret, and a VaPRL-style value-picked baseline).

**What happened.**

- Episodic mode learned the task: held-out grid success reached 0.86–0.96 (one seed).
- Several real bugs were found and fixed along the way — most notably a HOME
  pose whose gripper was intersecting the tray geometry, leaving the arm with
  20 active contacts and physically unable to move. That one cost days and was
  only found by probing joint-by-joint motion.
- Reset-free mode **never converged**. Zero successes at 300k and again at 480k
  steps, across four conditions.

**Diagnosis.** A CEM trajectory optimiser was used to measure the arm's actual
push envelope. The result: the maximum achievable cube displacement in the
required direction landed almost exactly on the success threshold. The task had
been specified at the edge of the platform's kinematic capability. Geometry was
resized to match the measured envelope.

Even after resizing, reset-free learning did not take off. The structural reason
is now clear: with zero successes, the learning-progress and regret signals are
both identically zero, so the three "different" conditions were running the same
algorithm. The designer objectives were never actually tested.

**Decision.** Park the simulation work. The bottleneck was exploration — the
forward policy never succeeded, so nothing downstream had a signal. This is
precisely the problem that demonstrations solve, and it is what the literature
(Gupta's demonstration-bootstrapped autonomous practicing, SERL) already
concluded.

---

## Phase B — Demonstrations on the real arm

**Approach.** Collect teleoperated demonstrations, train a behaviour-cloning
policy (ACT), get it working on hardware, and only then layer reinforcement
learning on top of a policy that already mostly works.

### Round 1 — 34 demos, images

- Recorded 34 pick-and-place demos with a DualSense gamepad: wrist camera,
  fixed diagonal scene camera, 6 joint positions and 6 commanded actions per
  frame at 30 fps.
- Stored as JPEG frames in parquet. 28,426 frames → **12.6 GB**.
- Trained ACT (52M params, ResNet18 backbone) to 10k steps.

**First hardware run.** The policy located the cube, drove to it, and opened the
gripper at the right moment. The grasp failed — the jaws did not open wide
enough — but after a manual assist it completed the transport and release.
First evidence the pipeline worked end to end.

**Failure mode.** Across further trials the policy consistently approached too
low and *pushed* the cube out of position before it could close on it.

**Diagnosis.** Demonstration quality, not model capacity. The demos varied
between shallow and overhead approaches, and the policy learned the average — a
skimming trajectory that clips the cube.

### Round 2 — 26 demos, hover-then-descend

Re-recorded with an explicit protocol: gripper wide open, move to a pose clearly
*above* the cube, descend straight down, close, lift, transport, release.

Result: eval loss essentially unchanged (0.2603 vs 0.2629), and the policy was
slower and no more accurate. Two confounds made this uninterpretable — fewer
episodes than round 1, and a checkpoint at 5k rather than 10k.

### The infrastructure detour

Training was attempted on a rented GPU. It ran at ~15 s/step with the **GPU at
0% utilisation** — the CPU could not decode 640×480 JPEGs fast enough to feed
it. Moving the dataset to an SSD changed nothing (`data_s` 1.58 → 1.61),
confirming the bottleneck was decode work, not disk I/O.

The correct fix was not more compute. It was the dataset format.

### Round 3 — 51 demos, video-encoded

- Recorded 51 demos with the hover-then-descend protocol, now much more
  consistent (episode lengths 400–600 frames, versus 429–1211 previously).
- Converted with **MP4/AV1 video encoding** instead of raw frames.
  26,483 frames → **212 MB** (from 4.1 GB for the same data as images).
  On Windows this needed an `if __name__ == "__main__"` guard, because video
  encoding spawns worker processes that re-import the script.
- Trained ACT at batch 20.

**The effect of video encoding:**

| | images | video |
|---|---|---|
| dataset size | 4.1 GB | 212 MB |
| `data_s` (dataloader wait) | 1.5 s | **0.003 s** |
| throughput | 4 samples/s | **44 samples/s** |
| 100k steps | ~50 h | ~12 h |

---

## Results

Checkpoints tested on hardware at identical cube positions:

| checkpoint | behaviour |
|---|---|
| 20k | approaches accurately, cannot close the grasp; needs a nudge |
| 40k | approaches from above correctly, grasp still unreliable |
| 60k | **completes the full task unaided** — approach, grasp, transport, release, in ~700 steps |
| **100k** | **best policy** — same task, smooth and unaided, in ~500 steps |

Real-robot performance improved monotonically with training duration across all
four checkpoints. Remaining failures cluster on a single mode: the grasp closes
1–2 cm short of the cube.

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

### A methodological note worth recording

Evaluation loss **rose** monotonically — ~0.26 at 20k, 0.2979 at 70k, 0.3014 at
100k — while training `l1_loss` fell to 0.027. That is the textbook overfitting
signature. Over exactly the same interval, real-robot performance improved at
every checkpoint, ending fastest and smoothest at 100k.

With ~6 held-out episodes, offline validation loss was not merely uninformative
here; it was *anti-correlated* with task success. Acting on it would have meant
selecting the worst checkpoint. Training was in fact stopped early at 60k on the
strength of that metric, and resuming to 100k produced a better policy. The
robot was the only reliable evaluator.

### Other findings

- **Per-step motion clamp.** A 4°/step safety limit was throttling the final
  approach. Raising it to 8° made the policy visibly faster and more decisive.
- **Execution horizon.** Reducing ACT's `n_action_steps` from the trained 100
  down to 20 or 5 at inference made the arm oscillate rather than improving
  reactivity — a policy trained on 100-step chunks is not free to be
  interrupted mid-chunk.
- **No termination condition.** Given surplus steps after a success, the policy
  continues acting and re-picks the cube from the tray. Demonstrations always
  ended at the drop, so "done" was never demonstrated.
- **No recovery behaviour.** When a grasp misses, the policy re-attempts at the
  *same* position rather than lifting and re-approaching. The demonstrations
  contain no failures — every recorded episode succeeded — so recovery was
  never demonstrated and cannot be imitated. This is not a tuning problem; the
  information is absent from the training data. It is also the clearest
  motivation for autonomous practice: a policy that practises on its own
  generates the failures it needs to learn from.
- **Hardware degradation is a real constraint.** Over roughly a week of
  operation the wrist-roll servo produced three separate bus faults and the
  gripper servo threw a thermal overload. Any claim about long unattended
  autonomous operation on this class of hardware has to account for this.

---

## What this establishes

A self-built ~$200 arm, 51 human demonstrations, and a consumer GPU are enough
to train a visuomotor policy that performs pick-and-place autonomously. The
binding constraints were, in order: demonstration consistency, dataset encoding
format, and training duration — not model capacity or compute.

## Next

1. Quantify: 20 trials at varied cube positions, unaided, logged by failure mode.
2. Continue training to 100k and compare against 60k on the same trial protocol.
3. Vision-based reward detection (colour segmentation → cube position → success).
4. Reset-free autonomous practice on top of the BC policy, logging interventions
   per hour — the point at which the simulation work's original question becomes
   testable on hardware.
