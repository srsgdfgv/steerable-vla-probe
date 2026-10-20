"""Grip word flip: free greedy decode, no teacher forcing.

Writes results/grip_greedy.csv
Usage: python probe/run_grip_greedy.py [traj ...]
"""
import csv
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from action_dist import load, predict, prep
from config import ANN_DIR, EP_DIR, RESULT_DIR
from perturb import DIR, FLIP

OUT = f"{RESULT_DIR}/grip_greedy.csv"
FIELDS = [
    "traj", "frame", "subtask", "progress", "orig", "new",
    "grip_orig", "grip_new", "grip_delta",
    "xyz_l2", "rot_l2",
    "ax0", "ay0", "az0", "ax1", "ay1", "az1",
]
GRIP_ONLY = re.compile(r"\b(open|close)\b")
MOTION = re.compile(r"\b(left|right|up|down|forward|backward)\b")


def segments(s):
    segs = []
    for f in range(len(s)):
        if segs and segs[-1][0] == s[str(f)]:
            segs[-1][2] = f
        else:
            segs.append([s[str(f)], f, f])
    return segs


def pick_gripword(commands):
    for raw in commands:
        if not isinstance(raw, str):
            continue
        c = raw.strip().lower().rstrip(".").strip()
        if GRIP_ONLY.search(c) and not MOTION.search(c):
            return c
    return None


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
            for sub, start, end in segments(step2sub[traj]):
                orig = pick_gripword(cmds[traj][sub])
                if orig is None:
                    continue
                new = DIR.sub(lambda m: FLIP[m.group(1)], orig)
                if new == orig:
                    continue
                for f in range(start, end + 1):
                    image = prep(images[f])
                    progress = (f - start) / max(end - start, 1)
                    a0, _ = predict(vla, image, orig)
                    a1, _ = predict(vla, image, new)
                    a0, a1 = np.asarray(a0), np.asarray(a1)
                    w.writerow(dict(
                        traj=traj, frame=f, subtask=sub, progress=round(progress, 3),
                        orig=orig, new=new,
                        grip_orig=round(float(a0[6]), 5),
                        grip_new=round(float(a1[6]), 5),
                        grip_delta=round(float(a1[6] - a0[6]), 5),
                        xyz_l2=round(float(np.linalg.norm(a0[:3] - a1[:3])), 5),
                        rot_l2=round(float(np.linalg.norm(a0[3:6] - a1[3:6])), 5),
                        ax0=round(float(a0[0]), 5), ay0=round(float(a0[1]), 5), az0=round(float(a0[2]), 5),
                        ax1=round(float(a1[0]), 5), ay1=round(float(a1[1]), 5), az1=round(float(a1[2]), 5),
                    ))
            fp.flush()
            print(f"traj {traj} done", flush=True)
