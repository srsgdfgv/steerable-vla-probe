# Method

## 1. Data

- Bridge trajectories aligned to `steering_features_bridge` annotations (`load_bridge.py`).
- Study set: **32** annotated episodes exported as `probe_episodes/{traj}.npz` (images, actions, states).
- Subtask boundaries checked against gripper open/close switches (`check_offset.py`): 45/62 switches within ±2 frames.

## 2. Instruction styles (`perturb.py`)

| Style | Rule | Semantic edits | Baseline |
| --- | --- | --- | --- |
| language | no coords, no direction words | `name`: swap to another object in the episode | another language command in the same subtask |
| atomic | contains left/right/up/down/forward/backward/open/close | `dir`: flip those words | another atomic command |
| point | 1–2 `[x,y]` | `coord` mirror x→256−x; `name`; `both` | another point command |
| trace | ≥3 coords | `coord` mirror (main probe); horizontal **shift** ±40/80/100 in follow-up | another trace command when available |

One original command is sampled per style per contiguous subtask segment (`rng` seeded by traj id).

## 3. Main metric: teacher-forced KL (`action_dist.py`, `run_probe.py`)

1. Resize frame with lanczos3 to 224 (`prep`), matching training.
2. Greedy-decode the **original** instruction → 7 action tokens.
3. Teacher-force those tokens under original vs edited instruction; read 256-bin softmax on the last 256 vocab tokens per action dim.
4. Sum KL(p_orig ‖ p_edit) over dims: xyz [0:3], rot [3:6], grip [6].
5. Also log entropy of p_orig (`ent_xyz`, `ent_rot`, `ent_grip`).

Argmax of teacher-forced probs is asserted to match `predict_action` tokens in the CLI.

**Interpretation:** compare semantic edits to the **same-frame, same-style baseline**. If P(edit KL > baseline KL) ≈ 0.5, the edit is no stronger than paraphrasing. Absolute KL is heavy-tailed; prefer medians and paired rates.

## 4. Follow-ups

### Grip free decode (`run_grip_greedy.py`)

- Atomic commands that contain open/close but **no** motion direction words.
- Both original and flipped instructions fully greedy-decoded (no fixed prefix).
- Record continuous grip values and xyz/rot L2 between the two actions.

### Trace large shift (`run_trace_shift.py`)

- Per subtask, pick the trace command with largest horizontal room; shift all x by ±40/80/100 when in-bounds.
- Same teacher-forced KL as the main probe, plus same-style baseline when a second trace exists.

## 5. What this does *not* claim

- Closed-loop task success.
- Out-of-distribution objects / pure anonymous `grasp at [x,y]` without names (many point/trace strings still mention objects).
- That the original paper is wrong; only that **under this offline counterfactual design**, coordinate numerals show little causal effect on the action distribution relative to paraphrases.
