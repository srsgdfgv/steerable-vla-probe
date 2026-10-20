# Steerable VLA Offline Controllability Probe

Offline counterfactual probe for [Steerable Policies](https://steerable-policies.github.io/) ([arXiv 2602.13193](https://arxiv.org/abs/2602.13193)) on Bridge with OpenVLA 7B.

**Question:** when object names, direction words, or coordinates in the instruction are changed to contradict the scene, does the policy’s action distribution move? Do different instruction granularities steer equally?

This repo contains **only the probe**. Model code comes from the upstream project:

- Upstream: [steerable-policies/steerable-policies-bridge](https://github.com/steerable-policies/steerable-policies-bridge)
- Checkpoint: [Embodied-CoT/steerable-policy-openvla-7b-bridge](https://huggingface.co/Embodied-CoT/steerable-policy-openvla-7b-bridge)
- Annotations: [Embodied-CoT/steering_features_bridge](https://huggingface.co/datasets/Embodied-CoT/steering_features_bridge)

## What / How / Results (short)

| | |
| --- | --- |
| **What** | Measure whether Steerable OpenVLA (Bridge) actually uses direction words, object names, and pixel coordinates in language. |
| **How** | Same-frame counterfactual edits + same-style paraphrase baseline; main metric = teacher-forced KL on action bins; follow-ups = free greedy grip flips and large horizontal trace shifts. |
| **Results (narrow)** | Direction ≫ object name ≫ point/trace coordinates (coord effect usually ≤ paraphrase noise). Free decode: grip flips on ~27% of frames. Limits: 32 in-distribution Bridge trajectories; KL-focused; many coord sentences still contain object names; grip not sliced to late frames only. |

See [probe/results/SUMMARY.md](probe/results/SUMMARY.md) for numbers and [docs/METHOD.md](docs/METHOD.md) for definitions.

## Layout

```
probe/
  config.py              # PROBE_ROOT paths
  load_bridge.py         # export annotated episodes → npz
  check_offset.py        # subtask vs gripper timing
  visualize_coords.py    # draw points/traces on frames
  perturb.py             # counterfactual instruction edits
  action_dist.py         # predict / teacher-forced probs / KL
  run_probe.py           # main probe → results/probe.csv
  run_grip_greedy.py     # open/close free decode
  run_trace_shift.py     # trace ±40/80/100 px shifts
  analyze.py             # recompute tables from CSVs
  results/               # probe.csv, grip_greedy.csv, trace_shift.csv, SUMMARY.md
docs/
  METHOD.md
```

## Setup

1. Install and load the upstream VLA (GPU). If Meta Llama-2 weights are gated, point the 7B pure LLM hub id to a public mirror such as `NousResearch/Llama-2-7b-hf` in the upstream `llama2.py` registry (inference-only workaround).
2. Download Bridge RLDS, steering annotations, and the Bridge checkpoint under `$PROBE_ROOT` (default `/root/autodl-tmp`):

```
$PROBE_ROOT/
├── data/
│   ├── bridge_dataset/1.0.0/
│   ├── steering_features_bridge/
│   └── probe_episodes/          # from load_bridge.py
└── steerable-policy-openvla-7b-bridge/checkpoints/*.pt
```

3. Run from a checkout that has `prismatic` importable (e.g. `pip install -e .` on the upstream repo) with `probe/` on `PYTHONPATH` or `cd probe`.

```bash
export PROBE_ROOT=/path/to/root
cd probe
python load_bridge.py
python check_offset.py
python run_probe.py              # → results/probe.csv
python run_grip_greedy.py
python run_trace_shift.py
python analyze.py                # prints summary tables
```

Bundled CSVs under `probe/results/` match the runs reported in `SUMMARY.md`. Re-running overwrites if you delete those files first (scripts append).

## License

Probe scripts in this repository are MIT. Upstream Prismatic / OpenVLA code and model weights remain under their original licenses and are **not** vendored here.
