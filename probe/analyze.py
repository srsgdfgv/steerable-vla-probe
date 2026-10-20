"""Print key tables from bundled CSVs (no GPU).

Usage: python probe/analyze.py
"""
import os
import numpy as np
import pandas as pd
from config import RESULT_DIR

PROBE = os.path.join(RESULT_DIR, "probe.csv")
GRIP = os.path.join(RESULT_DIR, "grip_greedy.csv")
TRACE = os.path.join(RESULT_DIR, "trace_shift.csv")


def paired_rates(df):
    key = ["traj", "frame", "style"]
    b = df[df.variant == "baseline"].set_index(key)["xyz"].rename("base")
    sem = df[df.variant != "baseline"].join(b, on=key).dropna(subset=["base"])
    rows = []
    for (s, v), g in sem.groupby(["style", "variant"]):
        rows.append({
            "style": s, "variant": v, "n": len(g),
            "P_gt_base": round(float((g.xyz > g.base).mean()), 3),
            "med_delta": round(float((g.xyz - g.base).median()), 4),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(PROBE)
    print(f"probe.csv rows={len(df)} trajs={df.traj.nunique()}")
    base = df[df.variant == "baseline"].groupby("style")["xyz"].median()
    sem = df[df.variant != "baseline"].groupby(["style", "variant"])["xyz"].median()
    print("\n=== median xyz KL / same-style baseline ===")
    for (s, v), m in sem.items():
        b = float(base.get(s, np.nan))
        ratio = m / b if b and b > 0 else float("nan")
        print(f"  {s:8s} {v:8s}  med={m:.4f}  base={b:.4f}  ratio={ratio:.1f}x")

    print("\n=== same-frame P(edit > baseline) ===")
    print(paired_rates(df).to_string(index=False))

    pt = df[df["style"] == "point"]
    have = pt.pivot_table(index=["traj", "frame"], columns="variant", values="xyz", aggfunc="first")
    if {"name", "coord"}.issubset(have.columns):
        hc = have.dropna(subset=["name", "coord"])
        print(f"\npoint name vs coord frames={len(hc)}  "
              f"med name={hc['name'].median():.4f} coord={hc['coord'].median():.4f}  "
              f"P(name>coord)={(hc['name'] > hc['coord']).mean():.3f}")

    if os.path.exists(GRIP):
        g = pd.read_csv(GRIP)
        changed = g.grip_orig != g.grip_new
        print(f"\n=== grip_greedy ===")
        print(f"  frames={len(g)}  grip_changed={changed.mean():.2f}  "
              f"trajs_any={(g.groupby('traj').apply(lambda x: (x.grip_orig != x.grip_new).any()).mean()):.2f}")

    if os.path.exists(TRACE):
        t = pd.read_csv(TRACE)
        bmed = t.loc[t.variant == "baseline", "xyz"].median()
        key = ["traj", "frame"]
        bb = t[t.variant == "baseline"].set_index(key)["xyz"].rename("base")
        print(f"\n=== trace_shift (baseline med={bmed:.4f}) ===")
        for dx in [40, 80, 100]:
            part = t[t.dx == dx].join(bb, on=key)
            paired = part.dropna(subset=["base"])
            p = float((paired.xyz > paired.base).mean()) if len(paired) else float("nan")
            print(f"  dx=±{dx}  n={len(part)}  med_kl={part.xyz.median():.4f}  "
                  f"paired_n={len(paired)}  P>base={p:.2f}")


if __name__ == "__main__":
    main()
