"""用重标定的 cap（见 data/thesis/recalib_cap.txt）真正重跑剂量层：
搜索预算与判定阈值都用新 cap。写 recalibration_layer.csv（包内 data/thesis/）。
"""
import csv, os, sys, time
from dataclasses import replace
sys.path.insert(0, '.')
import thesis_experiments as T
T._USE_BAG = True; T._BAG.clear()
from config import Config
from lowlevel import warmup_heuristics
from highlevel import solve, arrival_index
from methods import METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
TAG = "" if _PKG else "_bag"

with open(os.path.join(THESIS, f"recalib_cap{TAG}.txt")) as fh:
    CAP = float(fh.read().split("=")[1].split("\n")[0])
print(f"新 cap = {CAP:.4f}")

rows = []
for n in (6, 8):
    for seed in range(42, 62):
        cfg = replace(Config(), seed=seed, n_agents=n, t_budget=3.0,
                      heuristic_mode="exact", dose_budget=CAP, d_min=6.0)
        env = T.make_env(cfg, seed); tasks = env.tasks[:n]
        warmup_heuristics(cfg, env, tasks)
        r = solve(cfg, env, tasks, METHODS["ECBS_DOSE"], time.monotonic() + 3.0, seed)
        d = None
        if r.solved:
            d = max(sum(env.smoke_at(r.paths[i][t], t) * cfg.dt
                        for t in range(arrival_index(r.paths[i], g) + 1))
                    for i, (s, g) in enumerate(tasks))
        rows.append(dict(scenario=f"dm_s{seed}_n{n}", n=n, seed=seed,
                         solved=int(r.solved), dose=None if d is None else round(d, 4),
                         time=round(r.planning_time, 3)))
    print(f"  n={n} done", flush=True)

sol = [r for r in rows if r["solved"] == 1]
ok = [r for r in sol if r["dose"] <= CAP + 1e-9]
print(f"\nECBS_DOSE @ cap={CAP:.4f}: solved {len(sol)}/40, compliant {len(ok)}, "
      f"violating {len(sol)-len(ok)}")
with open(os.path.join(THESIS, f"recalibration_layer{TAG}.csv"), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "n", "seed", "solved", "dose", "time"])
    w.writeheader(); w.writerows(rows)
