"""Trace coord large-shift probe: horizontal translation (not mirror).

Writes results/trace_shift.csv
Usage: python probe/run_trace_shift.py [traj ...]
"""
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from action_dist import action_probs, kl, load, predict, prep
from config import ANN_DIR, EP_DIR, RESULT_DIR
from perturb import shift_room, shift_x, style, traj_context

OUT = f"{RESULT_DIR}/trace_shift.csv"
FIELDS = ["traj", "frame", "subtask", "progress", "dx", "variant", "orig", "new",
          "xyz", "rot", "grip", "room"]
SHIFTS = (40, 80, 100)


def segments(s):
    segs = []
    for f in range(len(s)):
        if segs and segs[-1][0] == s[str(f)]:
            segs[-1][2] = f
        else:
            segs.append([s[str(f)], f, f])
    return segs


def pick_trace(commands, task):
    best, best_room = None, -1
    for c in dict.fromkeys(
        x.strip().lower().rstrip(".").strip() for x in commands if isinstance(x, str)
    ):
        if style(c, task) != "trace":
            continue
        r = shift_room(c)
        if r > best_room:
            best, best_room = c, r
    return best, best_room


def build_variants(orig, room, others, rng):
    out = []
    for dx in SHIFTS:
        if room < dx:
            continue
        for sign, name in ((+1, f"shift_p{dx}"), (-1, f"shift_m{dx}")):
            new = shift_x(orig, sign * dx)
            if new:
                out.append((dx, name, new))
    if others:
        out.append((0, "baseline", others[rng.integers(len(others))]))
    return out


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
            _, task = traj_context(cmds[traj])
            rng = np.random.default_rng(int(traj) + 1000)
            for sub, start, end in segments(step2sub[traj]):
                orig, room = pick_trace(cmds[traj][sub], task)
                if orig is None or room < min(SHIFTS):
                    continue
                traces = []
                for raw in cmds[traj][sub]:
                    if not isinstance(raw, str):
                        continue
                    c = raw.strip().lower().rstrip(".").strip()
                    if style(c, task) == "trace":
                        traces.append(c)
                others = [c for c in dict.fromkeys(traces) if c != orig]
                variants = build_variants(orig, room, others, rng)
                if not variants:
                    continue
                for f in range(start, end + 1):
                    image = prep(images[f])
                    progress = (f - start) / max(end - start, 1)
                    _, ids = predict(vla, image, orig)
                    p0 = action_probs(vla, image, orig, ids)
                    for dx, variant, new in variants:
                        d = kl(p0, action_probs(vla, image, new, ids))
                        w.writerow(dict(
                            traj=traj, frame=f, subtask=sub, progress=round(progress, 3),
                            dx=dx, variant=variant, orig=orig, new=new, room=room,
                            **{k: round(v, 5) for k, v in d.items()},
                        ))
            fp.flush()
            print(f"traj {traj} done", flush=True)
