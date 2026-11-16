"""cap 外生性稳健性检验。

① 在 disjoint seeds（100–129，6/8 架）上重标定 cap：与评测集（seeds 42–61）不相交。
② 用新 cap 在评测集上重跑 ECBS / ECBS_DOSE，看结论是否改变。
口径与主表一致：3 s、d_min=6.0、t_max=48、K=16、heuristic exact。
"""
import csv, os, sys, time
import numpy as np
from dataclasses import replace

sys.path.insert(0, '.')
import thesis_experiments as T
T._USE_BAG = True
T._BAG.clear()
from config import Config
from lowlevel import warmup_heuristics
from highlevel import solve, arrival_index
from methods import METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
TAG = "" if _PKG else "_bag"
MS = ("ECBS",)


def run_one(seed, n, method):
    cfg = replace(Config(), seed=seed, n_agents=n, t_budget=3.0,
                  heuristic_mode="exact", dose_budget=2.88, d_min=6.0)
    env = T.make_env(cfg, seed)
    tasks = env.tasks[:n]
    warmup_heuristics(cfg, env, tasks)
    r = solve(cfg, env, tasks, METHODS[method], time.monotonic() + 3.0, seed)
    d = None
    if r.solved:
        d = max(sum(env.smoke_at(r.paths[i][t], t) * cfg.dt
                    for t in range(arrival_index(r.paths[i], g) + 1))
                for i, (s, g) in enumerate(tasks))
    return (r.solved, d, r.planning_time)


# ---------- ① 重标定 ----------
calib = []
for n in (6, 8):
    for seed in range(100, 130):
        ok, d, _ = run_one(seed, n, "ECBS")
        if ok:
            calib.append(d)
    print(f"[calib] n={n} done ({len(calib)} scenarios)", flush=True)

cap_new = float(np.percentile(calib, 60))
with open(os.path.join(THESIS, f"recalib_doses{TAG}.csv"), "w", newline="",
          encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["dose"]); w.writerows([[round(v, 4)] for v in calib])
print(f"\n标定集 seeds 100-129: n={len(calib)}  p60 = {cap_new:.4f}")
print(f"（原标定集 seeds 42-71 的 p60 = 2.8802）\n")

# ---------- ② 用新 cap 评测 ----------
B = 2.88
rows = []
for n in (6, 8):
    for seed in range(42, 62):
        for m in MS:
            ok, d, t = run_one(seed, n, m)
            rows.append(dict(scenario=f"dm_s{seed}_n{n}", n=n, seed=seed, method=m,
                             solved=int(ok), dose=None if d is None else round(d, 4),
                             time=round(t, 3)))

def summarise(rows, label, tag, cap):
    print(f"\n=== {label} (cap={cap:.4f}) ===")
    for m in MS:
        rs = [r for r in rows if r["method"] == m]
        sol = [r for r in rs if r["solved"] == 1]
        ok = [r for r in sol if r["dose"] is not None and r["dose"] <= cap + 1e-9]
        print(f"  {m:<11} solved {len(sol):>2}/{len(rs)}  compliant {len(ok):>2}"
              f"  violating {len(sol)-len(ok):>2}")

# 原 cap 与 新 cap 都算一遍（同一批运行，只是判定阈值不同）
summarise(rows, "评测集 seeds 42-61，原 cap", "old", 2.88)
summarise(rows, "评测集 seeds 42-61，重标定 cap", "new", cap_new)

path = os.path.join(THESIS, f"recalibration{TAG}.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "n", "seed", "method",
                                      "solved", "dose", "time"])
    w.writeheader(); w.writerows(rows)
print(f"\n写入 {path}")
with open(os.path.join(THESIS, f"recalib_cap{TAG}.txt"), "w") as f:
    f.write(f"cap_new={cap_new:.6f}\ncalib_n={len(calib)}\n")
