"""Main counterfactual probe over all annotated frames.

Writes results/probe.csv
Usage: python probe/run_probe.py [traj ...]
"""
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from action_dist import action_probs, entropy, kl, load, predict, prep
from config import ANN_DIR, EP_DIR, RESULT_DIR
from perturb import build_probes, traj_context

OUT = f"{RESULT_DIR}/probe.csv"
FIELDS = [
    "traj", "frame", "subtask", "progress", "style", "variant", "orig", "new",
    "xyz", "rot", "grip", "ent_xyz", "ent_rot", "ent_grip",
]


def segments(s):
    segs = []
    for f in range(len(s)):
        if segs and segs[-1][0] == s[str(f)]:
            segs[-1][2] = f
        else:
            segs.append([s[str(f)], f, f])
    return segs


if __name__ == "__main__":
    step2sub = json.load(open(f"{ANN_DIR}/step_to_subtask_dict.json"))
    cmds = json.load(open(f"{ANN_DIR}/subtask_level_commands.json"))
    trajs = sys.argv[1:] or sorted((f[:-4] for f in os.listdir(EP_DIR)), key=int)

    vla = load()
    os.makedirs(RESULT_DIR, exist_ok=True)
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fp:
        w = csv.DictWriter(fp, FIELDS)
        if new_file:
            w.writeheader()
        for traj in trajs:
            images = np.load(f"{EP_DIR}/{traj}.npz")["images"]
            objs, task = traj_context(cmds[traj])
            rng = np.random.default_rng(int(traj))
            for sub, start, end in segments(step2sub[traj]):
                probes = build_probes(cmds[traj][sub], objs, task, rng)
                for f in range(start, end + 1):
                    image = prep(images[f])
                    progress = (f - start) / max(end - start, 1)
                    cache = {}
                    for style, variant, orig, new in probes:
                        if orig not in cache:
                            _, ids = predict(vla, image, orig)
                            cache[orig] = (ids, action_probs(vla, image, orig, ids))
                        ids, p = cache[orig]
                        d = kl(p, action_probs(vla, image, new, ids)) | entropy(p)
                        w.writerow(dict(
                            traj=traj, frame=f, subtask=sub, progress=round(progress, 3),
                            style=style, variant=variant, orig=orig, new=new,
                            **{k: round(v, 5) for k, v in d.items()},
                        ))
            fp.flush()
            print(f"traj {traj} done", flush=True)
