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

---

## Phase C — Toward autonomous practice

Behaviour cloning produced a policy that works 40% of the time and cannot
improve from its own attempts. The next stage is autonomous practice: the robot
running unattended, scoring itself from vision, and improving without a human
resetting the scene between attempts.

### The reset problem, and why a second task

Pick-and-place is not self-resetting. After a success the cube is in the tray,
which the policy has never seen as a starting condition — so practice halts
after one episode unless a human moves the cube back. That is precisely the
human intervention the whole approach is meant to eliminate.

The answer, following Gupta et al. (ICRA 2021), is a *task network* rather than
a single task: define a set of tasks where the terminal state of one is a legal
start state of another, and they reset each other. At minimum scale that is two
tasks:

| task | precondition | postcondition |
|---|---|---|
| `place` | cube on table | cube in tray |
| `retrieve` | cube in tray | cube on table |

### Data

- `demos/retrieve` — 51 demonstrations, 29,367 frames, mean 575 per episode.
  Cube starts in the tray; drop positions on the table deliberately varied
  across the workspace, since those become the start distribution of `place`.
- Combined with `demos/pick3`: **102 episodes, 55,850 frames** across both
  tasks, converted into a single multi-task LeRobotDataset conditioned on the
  task string.

### Design: practice scheduling under embodied cost

With two tasks that reset each other, something must decide which to attempt
next. In prior work that scheduler is engineered — a hand-designed task graph
(Gupta et al. 2021), a fixed sequencer (DBAP), a value threshold (VaPRL) — and
its decisions are treated as free.

They are not free on hardware. Every attempt spends wall-clock, actuator
travel, and occasionally a human's attention. In simulation the right answer is
to make the reset distribution as broad as possible and scale; that option is
not purchasable when each reset has a price. **Which** state to practise from
becomes a budgeted decision.

The design (`scheduler_design.md`) treats the scheduler as a small learned
policy over a low-dimensional summary of training progress, rewarded by
measured competence gain net of time, wear, and intervention cost. It runs at
one decision per episode — hundreds per day rather than millions per hour —
which is what makes it learnable from real robot time.

Notably, the reward signal depends on the policy already sometimes succeeding:
competence gain is identically zero when nothing works. That is exactly why the
earlier simulation attempt produced a null result, and why the
demonstration-bootstrapped platform had to come first.

### Build order

1. Multi-task policy over both directions ← **current step**
2. Vision-based world-state estimator, validated offline on recorded frames
3. Ledger + session runner + hand-designed feasible scheduler
   → first unattended multi-task session, with interventions logged
4. Periodic competence probe
5. Learned schedulers and the comparison against hand-designed baselines

Steps 1–4 produce reportable results independently of whether step 5's claim
holds.

---

## Phase D — Multi-task policy and its failure modes

### The run

51 `retrieve` demonstrations were combined with the 51 `place` demonstrations
into one multi-task LeRobotDataset (102 episodes, 55,850 frames), with the two
tasks distinguished by the task string ACT conditions on. Trained to 150k steps
at batch 32 on a single consumer GPU (~20 h), final train `l1_loss` 0.026,
eval loss 0.2729.

### Task conditioning works

At the 50k checkpoint the same weights produce clearly different behaviour for
the two task strings — `retrieve` executes correctly, `place` does not. The
conditioning mechanism is sound; the two tasks are simply learned to different
standards.

`retrieve` is the easier problem: the cube always starts inside the small tray
region, so 51 demonstrations cover its start distribution densely. `place`
starts from the cube anywhere on the table, so the same number of
demonstrations covers a far larger space much more thinly.

### Three failure modes, each traced to a cause

**1. Task interference.** Mid-episode — cube grasped, arm in the air — the two
tasks are visually near-identical but require opposite motions. This is a
documented failure of single-network imitation policies: nearly identical
visual scenes and gripper trajectories corresponding to different intents force
the network to average contradictory actions. ACT and Diffusion Policy are
known to perform well on single-task benchmarks and to degrade in multi-task
settings for exactly this reason. The field's answers are language-grounded
visual representations, mixture-of-experts routing, or per-task adapters on a
frozen backbone — not a retreat to single-task policies, which remain the
dominant paradigm.

**2. No grasp perception.** After closing on the cube the policy sometimes
reopens and re-attempts. The cube is occluded once gripped, so visually the
state resembles "no cube in the gripper" — which in the demonstrations means
*keep trying*. The observation space contains no signal for "I am holding
something." Adding gripper load from the servos would supply it, at the cost of
re-recording the dataset.

**3. No recovery behaviour.** Unchanged from Phase B, and the root of both the
above: every recorded demonstration succeeded, so no state that only arises
after a mistake appears anywhere in the training data.

### Autonomous practice infrastructure

Built and committed:

- `calibrate_tray.py` — click the tray corners once; the camera is fixed
- `calibrate_mask.py` — exclusion zone over the arm, to stop dark servos being
  detected as the cube
- `detector.py` — dark-blob cube detection against the light table, with a tray
  inset so tape on the border does not count as in-tray, plus a wrist-camera
  check that reports a carried cube as `held` rather than `lost`
- `runner.py` — session loop: read world state, select a feasible task, run an
  episode, score it, charge time and joint travel to a cost ledger, log a CSV
  row, repeat. Interventions are logged when no task is feasible.

Two instrumentation problems worth recording, because both would have produced
false data:

- The side-on scene camera projects a cube *held in the air* into the tray's
  pixel region. Success therefore requires the cube to also be stationary —
  within 6 px for 15 consecutive frames — not merely inside the region.
- Detector thresholds were set by measurement rather than guesswork: static
  clutter (tape, cabling, arm edges) tops out at ~1510 px² while the cube
  measures 2278–6850 px² depending on distance, giving a clean separation at
  `MIN_AREA = 1800`. An earlier `MAX_ASPECT = 1.5` was too tight — the cube
  reaches 1.62 at some angles — and was causing it to read as lost.

### Where this leads

The single-task ACT policy (40%) outperformed the multi-task one on `place`.
That comparison — single-task ACT vs multi-task ACT on identical data, with
task interference as the explanation — is retained as a result rather than
discarded.

The next step is a language-conditioned pretrained policy rather than a
from-scratch one. Training ACT from scratch on 51 demonstrations per task asks
the network to learn what a cube looks like from those demonstrations alone; a
pretrained VLA with a frozen vision encoder already has that prior and needs
only to learn what to do with it. This matters for the scheduler work too: the
scheduler sits above whatever policy is underneath, and its experiment only
becomes meaningful once that policy is competent at both tasks.

---

## Phase D — Multi-task policies, and two negative results

### Multi-task ACT: task interference

Both tasks were combined into a single LeRobotDataset (102 episodes, 55,850
frames) distinguished by the task string ACT conditions on, and a multi-task
ACT policy trained to 150k steps (batch 32, ~97 epochs).

Task conditioning demonstrably worked — the same weights produced different
behaviour for the `place` and `retrieve` strings. But performance split sharply:
`retrieve` worked well, `place` could not reliably complete the grasp, despite
`place` having worked at 40% as a single-task policy on the same demonstrations.

This is a documented failure mode. ACT and Diffusion Policy perform strongly on
single-task benchmarks but suffer gradient interference in multi-task settings,
specifically through the **"similar-input, different-output" dilemma**: nearly
identical visual scenes and gripper trajectories correspond to different
intents, and the network is forced to average contradictory actions.

That describes this setup exactly. Mid-episode — cube held in the air — `place`
and `retrieve` look nearly identical and require opposite motions.

A second limitation surfaced in the same runs: after closing on the cube the
policy frequently **reopens and re-attempts the grasp**. Once the gripper
closes, the cube is occluded in both camera views, so the observation resembles
"no cube grasped" — a state that in the demonstrations means *keep trying*. The
policy has no proprioceptive signal (e.g. gripper load) that would distinguish
holding from empty.

### SmolVLA: a hardware-tier constraint

The interference diagnosis pointed to a language-conditioned pretrained policy,
where the instruction is grounded in a real semantic space rather than learned
from 51 episodes. SmolVLA (450M parameters, frozen SigLIP vision encoder,
pretrained on LeRobot community SO-100/SO-101 data) was fine-tuned on the same
multi-task dataset, with cameras renamed to match its expected feature keys.

It trained cleanly — final eval loss 0.2270, the lowest of any policy in this
project, reached in 4 epochs versus ACT's 97. On hardware, neither task worked.

The reason is arithmetic, and it is the interesting part:

| | batch | steps | samples seen |
|---|---|---|---|
| ACT (multi-task) | 32 | 150k | 4.8M |
| SmolVLA | 10 | 20k | 0.2M |

SmolVLA's reference recipe assumes batch 64. A 6 GB GPU fits batch 10 — at 16
the run spilled into shared memory and slowed 5×. Reaching ACT's sample
exposure at batch 10 would require ~480k steps, roughly 16 days of continuous
training.

**A 450M-parameter VLA cannot be fine-tuned to convergence on a 6 GB GPU**, not
because the model does not fit, but because the batch size that fits is too
small to reach the sample counts the method needs. Inference is not the
obstacle either: the policy ran at ~14 Hz on the same card, against the 30 Hz it
was trained at.

### What these two results establish

Both are constraints of the platform tier rather than failures of method, and
both are the kind of thing that only appears when the hardware is actually
cheap:

- Multi-task ACT is limited by **gradient interference** between visually
  similar tasks.
- Multi-task SmolVLA is limited by **the batch size a 6 GB GPU permits**.

The literature's own note that per-task checkpoints avoid interference but are
"deployment-prohibitive" does not bind here: with one robot and two tasks,
maintaining two single-task policies is entirely practical, and the scheduler is
indifferent to whether selecting a task means switching a string or switching a
checkpoint.

---

## Phase E — Toward goal-conditioned practice

### Why two tasks is not enough

With K=2 tasks and a feasibility filter, the scheduler usually has exactly one
legal action — the cube is either on the table (so run `place`) or in the tray
(so run `retrieve`). There is almost nothing to decide. The scheduling question
only becomes substantive when the choice is rich.

Enumerating more discrete tasks is the wrong scaling path: each would need its
own demonstrations. The alternative is **goal-conditioning** — replace
`pi(a | s, task_label)` with `pi(a | s, goal)`, where the goal is a target
state rather than a label. The task space becomes continuous, and the
scheduler's action space becomes *the workspace itself*: it chooses where to
practise next, not which of two labels to fire.

This is also how the prior work this project builds on actually operates —
VaPRL selects start states via a goal-conditioned value function, DBAP uses
goal-conditioned imitation. The two-task setup is the discrete special case.

### Hindsight relabeling means the existing data may suffice

The standard technique for annotating trajectories with goals is hindsight
relabeling: for each timestep, the set of achievable goals is the set of states
actually reached later in that same trajectory. Every recorded episode
therefore already contains its own goal label — the final cube position.

In principle the existing 102 episodes convert to goal-conditioned form without
new demonstrations. Whether LeRobot's ACT supports goal conditioning natively is
**not yet verified**; if it does not, feeding a goal image as an additional
camera stream is the usual workaround and the dataset format supports it.

### Act2Goal (Zhou et al., arXiv 2512.23541, Dec 2025)

The closest recent work, and worth recording precisely because parts of it are
adoptable and parts are not.

**What it does.** Given a current observation and a target *visual* goal, a
goal-conditioned visual world model generates a plausible sequence of
intermediate visual states capturing long-horizon structure. **Multi-Scale
Temporal Hashing (MSTH)** then decomposes that imagined trajectory into dense
proximal frames for fine-grained closed-loop control and sparse distal frames
that anchor global task consistency; both are coupled to motor control through
end-to-end cross-attention. It reports reward-free online adaptation via
hindsight goal relabeling with LoRA-based finetuning, improving real-robot
success from 30% to 90% on out-of-distribution tasks within minutes of
autonomous interaction.

(Note: Act2Goal's mechanism is a world model plus MSTH — not a flow-matching
action head. Flow matching is the action decoder used by pi-0 and SmolVLA, a
separate line.)

Last author Jianlan Luo is also an author on SERL and HIL-SERL, so this sits
directly in the real-world-RL lineage this project follows.

**What is adoptable here:**
- Visual goals as task specification, replacing language labels.
- Hindsight relabeling of the robot's own rollouts — no reward function needed.
- LoRA fine-tuning for fast online adaptation.

**What is not:** the video world model. Training generative video prediction is
far outside a 6 GB budget — the same tier constraint that ended the SmolVLA
attempt.

**What it does not address, and this remains the open gap:** Act2Goal relabels
whatever rollouts it happens to collect. It does not decide *which* goals are
worth attempting, and it does not price the attempt. Choosing where to practise,
under a budget of time, wear, and human interventions, is still unclaimed.

### Two platform issues identified

- **Scene camera occlusion.** When the arm lifts the cube from the table it
  substantially occludes the tray in the third-person view. The policy still
  works, but the detector loses the cube during transport and the success test
  depends on the cube being visible where it lands. A higher or more overhead
  camera would fix it — at the cost of invalidating every policy trained on the
  current viewpoint, so it is deferred to the next dataset.
- **Asymmetric start-state coverage.** `retrieve` always begins with the cube in
  the small tray region; `place` begins anywhere on a large table. With equal
  demonstration budgets, `place` gets far sparser coverage — which matches the
  observed difficulty gap between the two. Restricting the reachable workspace
  so both tasks have comparable coverage is a cheaper fix than recording more
  demonstrations.
