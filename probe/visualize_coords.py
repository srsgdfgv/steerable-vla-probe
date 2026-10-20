import json
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from config import ANN_DIR, EP_DIR, VIS_DIR as OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)

step2sub = json.load(open(f"{ANN_DIR}/step_to_subtask_dict.json"))
cmds = json.load(open(f"{ANN_DIR}/subtask_level_commands.json"))
COORD = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def segments(traj):
    s = step2sub[traj]
    segs, prev = [], None
    for f in sorted(int(k) for k in s):
        sub = s[str(f)]
        if sub != prev:
            segs.append([sub, f, f])
            prev = sub
        else:
            segs[-1][2] = f
    return segs

OBJ = re.compile(r"(?:the\s+)?([a-z][a-z ]{1,25}?)\s+at\s+\[\s*(\d+)\s*,\s*(\d+)\s*\]", re.I)


def split_coords(commands):
    traces, points = [], {}
    for c in commands:
        if not isinstance(c, str):
            continue
        pts = [(int(a), int(b)) for a, b in COORD.findall(c)]
        if len(pts) >= 3:
            traces.append(pts)
        else:
            for name, a, b in OBJ.findall(c):
                points[(int(a), int(b))] = name.split()[-1]
    return traces, points


def visualize(traj):
    images = np.load(f"{EP_DIR}/{traj}.npz")["images"]
    segs = segments(traj)
    fig, axes = plt.subplots(1, len(segs), figsize=(5 * len(segs), 5), squeeze=False)
    for col, (sub, start, end) in enumerate(segs):
        traces, points = split_coords(cmds[traj][sub])
        ax = axes[0][col]
        ax.imshow(images[end])
        for tr in traces:
            ax.plot(*zip(*tr), "-o", color="red", ms=3, lw=1)
        for (x, y), name in points.items():
            ax.scatter(x, y, s=150, facecolors="none", edgecolors="yellow", lw=2)
            ax.text(x + 6, y, name, color="yellow", fontsize=10)
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_title(f"{col}: {sub}\nlast frame {end}", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{traj}.png", dpi=80)
    plt.close()
    print(f"saved {traj}.png  ({len(segs)} subtasks)")

if __name__ == "__main__":
    for traj in (sys.argv[1:] or ["0", "1", "2", "3", "5"]):
        visualize(traj)