import json
import re
import sys
import numpy as np

COORD = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
OBJ = re.compile(r"([a-z]+)\s+at\s+\[", re.I)
FLIP = {"left": "right", "right": "left", "up": "down", "down": "up",
        "forward": "backward", "backward": "forward", "open": "close", "close": "open"}
DIR = re.compile(r"\b(" + "|".join(FLIP) + r")\b")


def norm(c):
    return c.strip().lower().rstrip(".").strip()


def style(c, task):
    n = len(COORD.findall(c))
    if c == task:
        return "task"
    if n >= 3:
        return "trace"
    if n >= 1:
        return "point"
    if DIR.search(c):
        return "atomic"
    return "language"


def mirror(c):
    return COORD.sub(lambda m: f"[{256 - int(m.group(1))}, {m.group(2)}]", c)


def shift_x(c, dx):
    """整条指令的所有 x 平移 dx 像素；越界则返回 None。"""
    pts = [(int(x), int(y)) for x, y in COORD.findall(c)]
    if not pts:
        return None
    if any(not (0 <= x + dx <= 255) for x, _ in pts):
        return None
    return COORD.sub(lambda m: f"[{int(m.group(1)) + dx}, {m.group(2)}]", c)


def shift_room(c):
    """水平方向还能平移多少像素而不越界。"""
    pts = [(int(x), int(y)) for x, y in COORD.findall(c)]
    if not pts:
        return 0
    return min(min(x, 255 - x) for x, _ in pts)


def swap_name(c, objs):
    for o in objs:
        if re.search(rf"\b{o}\b", c):
            other = [x for x in objs if x != o and not re.search(rf"\b{x}\b", c)]
            if other:
                return re.sub(rf"\b{o}\b", other[0], c, count=1)
    return None


def build_probes(commands, objs, task, rng):
    """返回 [(风格, 改法, 原指令, 新指令)]，每种风格随机取一条原指令。"""
    by_style = {}
    for c in dict.fromkeys(norm(c) for c in commands if isinstance(c, str)):
        by_style.setdefault(style(c, task), []).append(c)
    by_style.pop("task", None)

    probes = []
    for s, cs in sorted(by_style.items()):
        orig = cs[rng.integers(len(cs))]
        variants = {}
        if s == "language":
            variants["name"] = swap_name(orig, objs)
        elif s == "atomic":
            variants["dir"] = DIR.sub(lambda m: FLIP[m.group(1)], orig)
        else:
            variants["coord"] = mirror(orig)
            if s == "point":
                variants["name"] = swap_name(orig, objs)
                variants["both"] = variants["name"] and mirror(variants["name"])
        others = [c for c in cs if c != orig]
        if others:
            variants["baseline"] = others[rng.integers(len(others))]
        probes += [(s, v, orig, new) for v, new in variants.items() if new]
    return probes


def traj_context(cmds_traj):
    """从一条轨迹的全部指令中取出：物体名列表、任务原句（出现在每个子任务里的那句）。"""
    lists = [[norm(c) for c in v if isinstance(c, str)] for v in cmds_traj.values()]
    common = set(lists[0]).intersection(*lists[1:])
    task = next(iter(common)) if len(common) == 1 else None
    objs = sorted({m.lower() for cs in lists for c in cs for m in OBJ.findall(c)} - {"gripper"})
    return objs, task


if __name__ == "__main__":
    ann_dir, traj = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0"
    cmds = json.load(open(f"{ann_dir}/subtask_level_commands.json"))
    objs, task = traj_context(cmds[traj])
    print("物体:", objs, "\n任务原句:", task)
    rng = np.random.default_rng(0)
    for sub, cs in cmds[traj].items():
        print(f"\n== {sub}")
        for s, v, o, n in build_probes(cs, objs, task, rng):
            print(f"  [{s}/{v}]\n    原: {o}\n    改: {n}")
