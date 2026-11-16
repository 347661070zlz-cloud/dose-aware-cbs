"""EECBS 对照：与 ECBS 在**同一次运行、同一机器负载**下逐场景对比。
只跑三个方法，保证比较公平；结果写 eecbs.csv。
"""
import csv, os, sys, time
from dataclasses import replace
sys.path.insert(0, '.')
from config import Config
import thesis_experiments as T

# 必须与主表同底图：模块默认 _USE_BAG=False（程序化楼块），此处显式切到 3D BAG。
T._USE_BAG = True
T._BAG.clear()
from lowlevel import warmup_heuristics
from highlevel import solve, arrival_index
from methods import METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
TAG = "" if _PKG else "_bag"
MS = ("ECBS", "EECBS", "EECBS_DOSE")

rows = []
for n in (6, 8, 10, 12):
    for seed in range(42, 62):
        cfg = replace(Config(), seed=seed, n_agents=n, t_budget=3.0,
                      heuristic_mode="exact", dose_budget=2.88, d_min=6.0)
        env = T.make_env(cfg, seed); tasks = env.tasks[:n]
        warmup_heuristics(cfg, env, tasks)
        for m in MS:
            r = solve(cfg, env, tasks, METHODS[m], time.monotonic() + 3.0, seed)
            d = ""
            if r.solved:
                d = round(max(sum(env.smoke_at(r.paths[i][t], t) * cfg.dt
                                  for t in range(arrival_index(r.paths[i], g) + 1))
                              for i, (s, g) in enumerate(tasks)), 4)
            rows.append(dict(scenario=f"dm_s{seed}_n{n}", n_agents=n, seed=seed,
                             method=m, solved=int(r.solved), dose_to_arrival=d,
                             planning_time=round(r.planning_time, 3),
                             ct_nodes=r.ct_nodes, cost=round(r.cost, 2)))
    print(f"n={n} done", flush=True)

path = os.path.join(THESIS, f"eecbs{TAG}.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

B = 2.88
for m in MS:
    rs = [r for r in rows if r["method"] == m]
    sol = [r for r in rs if r["solved"] == 1]
    viol = [r for r in sol if r["dose_to_arrival"] != "" and float(r["dose_to_arrival"]) > B + 1e-9]
    print(f"{m:<11} solved {len(sol):>2}  violations {len(viol):>2}  compliant {len(sol)-len(viol)}")
print("写入", path)
