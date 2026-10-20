import glob
import json
import os
from collections import Counter
import numpy as np
from config import ANN_DIR, EP_DIR

step2sub = json.load(open(f"{ANN_DIR}/step_to_subtask_dict.json"))

offsets = Counter()
for path in sorted(glob.glob(f"{EP_DIR}/*.npz"), key=lambda p: int(os.path.basename(p)[:-4])):
    traj = os.path.basename(path)[:-4]
    grip = np.load(path)["actions"][:, 6] > 0.5
    subs = [step2sub[traj][str(t)] for t in range(len(grip))]

    grip_switch = [t for t in range(1, len(grip)) if grip[t] != grip[t - 1]]
    sub_switch = [t for t in range(1, len(subs)) if subs[t] != subs[t - 1]]
    print(f"traj {traj}: 夹爪切换帧={grip_switch}  子任务切换帧={sub_switch}")
    for g in grip_switch:
        if sub_switch:
            d = min((s - g for s in sub_switch), key=abs)
            offsets[d] += 1

print("\n夹爪切换帧 → 最近子任务切换帧 的偏移（子任务帧 - 夹爪帧）：")
for d, n in sorted(offsets.items()):
    print(f"  偏移 {d:+d}: {n} 次")
