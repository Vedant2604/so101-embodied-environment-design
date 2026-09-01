# Practice Scheduling Under Embodied Cost
### Architecture v2 — cost-first design

> In simulation a reset is free, so the right answer is to make the reset
> distribution as broad and diverse as possible and let scale do the rest.
> On hardware every reset is paid for in brass tacks — wall-clock, actuator
> wear, and occasionally a human walking over. Breadth is no longer
> purchasable. **Which** state the robot practises from becomes a budgeted
> decision, and that decision is the object of this design.

---

## 0. The reframing

Prior reset-free systems ask: *what sequence of tasks keeps the robot running
without a human?* All of them answer with engineering — a task graph, a fixed
sequencer, a value threshold.

This design asks a different question: **given a fixed budget of robot-hours
and a finite tolerance for interventions, which practice attempts buy the most
policy improvement per unit cost?**

That reframing is what the real-hardware setting adds. It does not exist in
simulation because the denominator is zero.

---

## 1. The cost ledger

Every scheduling decision draws on three accounts. All three are already
instrumented on the platform.

| account | unit | measured how |
|---|---|---|
| **time** | seconds | episode wall-clock |
| **wear** | degrees of joint travel | summed per-step joint deltas |
| **attention** | interventions | world-state estimator flags an unreachable state |

A decision's price is `c = λ_t·time + λ_w·wear + λ_i·intervention`.

The intervention term dominates by design — it is the only cost that consumes
a *human*, and reducing it is the field's stated goal (EARL's central metric).
Setting λ_i ≫ λ_t, λ_w encodes that.

**Why this is the contribution.** No prior scheduler prices its own decisions.
Once decisions are priced, "practise the hardest thing" and "practise the most
informative thing" stop being the same policy — because hard attempts fail,
and failures are what generate interventions.

---

## 2. The two-timescale architecture

```
   once per episode                      30 Hz
   (~30-60 s apart)                      (control loop)
   ┌──────────────────┐                  ┌────────────────────┐
   │  SCHEDULER       │  task k          │  MULTI-TASK POLICY │
   │  sigma(k | omega)│ ───────────────► │  pi(a | s, k)      │
   │  2-layer MLP     │                  │  ACT, 52M params   │
   └────────▲─────────┘                  └─────────┬──────────┘
            │                                      │ joint targets
            │ reward = competence gain             ▼
            │          minus cost            ┌───────────┐
            │                                │  ROBOT    │
            │                                └─────┬─────┘
   ┌────────┴──────────┐                           │ frames
   │  LEDGER + PROBE   │◄──────────────────────────┘
   │  competence,      │      world-state estimator
   │  time, wear,      │      (cube region + position)
   │  interventions    │
   └───────────────────┘
```

The separation is what makes this learnable on real hardware: the scheduler
faces **hundreds of decisions per day**, not millions per hour. A small model
over a low-dimensional summary is the correct capacity, and it is trainable
inside a single project rather than a single datacentre.

---

## 3. Task set — K = 2 to start

| k | task | precondition | postcondition |
|---|---|---|---|
| 0 | `place` | cube on table | cube in tray |
| 1 | `retrieve` | cube in tray | cube on table |

Each is the other's reset. This is Gupta et al. (ICRA 2021) at minimum viable
scale — a task network where the failure state of one is a legal start state
of another, so the loop closes without a human.

Extensions (`recenter`, `retrieve-from-edge`) enlarge K without changing the
architecture. Deliberately deferred: K = 2 is the smallest system in which the
scheduling question is well-posed, and the smallest that can be run to a clean
result before deadlines.

---

## 4. Scheduler state, action, reward

**State** — small enough to learn from hundreds of decisions:

```
omega = [ cube_region        one-hot: table / tray / unreachable
          cube_xy            normalised position in region
          attempts[k]        per-task attempt counts
          success_rate[k]    windowed, per task
          progress[k]        change in success_rate over the window
          budget_remaining   fraction of session time and wear left
          since_intervention seconds since last human touch ]
```

No pixels, no joint angles. The scheduler reasons about the *training process*,
not the physics — that is the policy's job.

**Action** — one of the K tasks whose precondition currently holds. Infeasible
actions are masked, not penalised; the scheduler never learns to command
`retrieve` with the cube on the table because it is never offered.

**Reward** — competence gain, net of what it cost to buy:

```
r  =  [ C(t) - C(t-1) ]  -  lambda_t*time  -  lambda_w*wear  -  lambda_i*interventions
```

`C` is measured by a **periodic held-out probe**: every N episodes, run a fixed
set of start states and record success rate. Consecutive probe differences are
credited to the batch of episodes between them.

### The dependency that makes this work

`C(t) - C(t-1)` is identically zero when the policy never succeeds. That is
precisely what happened in the earlier simulation attempt: with no successes,
learning-progress and regret signals were both zero, and three nominally
different designers collapsed to the same random scheduler.

It is nonzero here **only because the BC policy already succeeds 8/20**.
The demonstration-bootstrapped platform is not a preliminary to this design —
it is its precondition.

---

## 5. Comparison

| scheduler | rule | prices cost? |
|---|---|---|
| `none` | frozen BC policy, no practice | — (control) |
| `alternate` | strict 0,1,0,1 regardless of state | no |
| `feasible` | any task whose precondition holds — the ICRA 2021 pattern | no |
| `value` | VaPRL-style value threshold over start states | no |
| `learned` | sigma trained on competence gain alone | no |
| `learned+cost` | sigma trained on the full ledger | **yes** |

The pair that carries the claim is `learned` vs `learned+cost`. Everything
else is context. If pricing decisions changes nothing, the contribution is
falsified cleanly — and that is worth knowing.

**Metrics** (EARL-compatible): deployed success rate on a fixed held-out trial
protocol; continuing success rate during autonomous operation; and
**interventions per hour** — plus, new here, **competence gained per robot-hour**
and **per intervention**.

---

## 6. Modules

```
scheduler/
  detector.py      frame -> cube pixel location + confidence
  world_state.py   frame -> WorldState(region, xy, confident)
  ledger.py        time / wear / intervention accounting
  competence.py    periodic held-out probe -> success rate
  policies.py      Scheduler protocol + the six variants
  runner.py        the session loop
  logging.py       one CSV row per episode
```

`detector.py` and `world_state.py` are validated **offline against recorded
frames** — no arm required. Hardware time is the scarce resource; anything
that can be debugged without it, should be.

**Interfaces**

```python
@dataclass
class WorldState:
    region: str                 # "table" | "tray" | "unreachable"
    xy: tuple[float, float]
    confident: bool

@dataclass
class Outcome:
    task: int
    success: bool
    steps: int
    seconds: float
    joint_travel: float
    intervention: bool

class Scheduler(Protocol):
    def feasible(self, w: WorldState) -> list[int]: ...
    def select(self, w: WorldState, ledger: Ledger) -> int: ...
    def update(self, batch: list[Transition]) -> None: ...   # no-op for baselines
```

Baselines and the learned scheduler differ **only** in `select` and `update`.
Everything around them is shared, so any measured difference is attributable
to the decision rule and nothing else.

**Session loop**

```python
while ledger.budget_remaining() > 0:
    w = world_state.estimate(scene_cam.frame())

    if not scheduler.feasible(w):          # cube off-table, unreachable pose
        ledger.log_intervention()
        wait_for_human()
        continue

    k = scheduler.select(w, ledger)
    outcome = run_episode(policy, task=k, max_steps=800)
    ledger.charge(outcome)

    if ledger.episodes_since_probe >= N:
        C_new = competence.probe(policy)
        r = (C_new - C_old) - ledger.cost_since_probe()
        scheduler.update(batch, r)
        C_old = C_new
```

The reward arrives **late and batched** — competence is only observable
periodically. The design owns that explicitly: the scheduler trains on batched
returns, not per-decision rewards, and is warm-started by behaviour-cloning the
`feasible` baseline so early hardware hours are not spent on random exploration.

---

## 7. Build order

| # | deliverable | reportable on its own? |
|---|---|---|
| 1 | `retrieve` demos + multi-task policy | yes — "policy performs both directions" |
| 2 | `detector` + `world_state`, validated offline | no (infrastructure) |
| 3 | `ledger` + `runner` + `feasible` scheduler | **yes — first unattended multi-task session, N hours, X interventions** |
| 4 | `competence` probe | yes — the improvement curve itself |
| 5 | learned schedulers + comparison table | the research claim |

Steps 1–4 are engineering with guaranteed output. Step 5 is the claim. Built in
this order, the project produces a result even if the claim does not land.

**Prerequisite still unmet:** `retrieve` does not exist. With one task the
scheduler has one legal action and nothing to decide. Everything above is
blocked on ~50 reverse demonstrations.
