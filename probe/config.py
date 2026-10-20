import os

_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROBE_ROOT", "/root/autodl-tmp")
BRIDGE_DIR = f"{ROOT}/data/bridge_dataset/1.0.0"
ANN_DIR = f"{ROOT}/data/steering_features_bridge"
EP_DIR = f"{ROOT}/data/probe_episodes"
VIS_DIR = f"{ROOT}/data/probe_vis"
CKPT_DIR = f"{ROOT}/steerable-policy-openvla-7b-bridge/checkpoints"
# Bundled CSVs live under probe/results/; override with PROBE_RESULT_DIR if needed.
RESULT_DIR = os.environ.get("PROBE_RESULT_DIR", os.path.join(_PROBE_DIR, "results"))
