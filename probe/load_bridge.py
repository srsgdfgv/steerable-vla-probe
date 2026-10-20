import json
import os
import numpy as np
import tensorflow_datasets as tfds
from config import ANN_DIR, BRIDGE_DIR, EP_DIR as OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)

key_to_traj_idx = json.load(open(f"{ANN_DIR}/traj_idx_key_map.json"))["key_to_traj_idx"]
step2sub = json.load(open(f"{ANN_DIR}/step_to_subtask_dict.json"))

builder = tfds.builder_from_directory(BRIDGE_DIR)
ds = builder.as_dataset(split="train[:52]")

matched = 0
for i, ep in enumerate(tfds.as_numpy(ds)):
    file_path = ep["episode_metadata"]["file_path"].decode()
    episode_id = int(ep["episode_metadata"]["episode_id"])
    key = f"{file_path}-{episode_id}"
    traj_idx = key_to_traj_idx.get(key)

    steps = list(ep["steps"])
    n_frames = len(steps)

    if traj_idx is None:
        print(f"[{i}] 未匹配  frames={n_frames}  key={key}")
        continue

    n_ann = len(step2sub.get(str(traj_idx), {}))
    print(f"[{i}] traj_idx={traj_idx}  bridge帧数={n_frames}  标注帧数={n_ann}")
    if n_ann == 0:
        continue
    matched += 1

    images = np.stack([s["observation"]["image_0"] for s in steps])
    actions = np.stack([s["action"] for s in steps])
    states = np.stack([s["observation"]["state"] for s in steps])
    np.savez_compressed(f"{OUT_DIR}/{traj_idx}.npz", images=images, actions=actions, states=states)

print(f"\n保存有标注的 {matched} / 52 条")