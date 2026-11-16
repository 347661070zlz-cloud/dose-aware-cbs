"""窗口鲁棒性：同一套协议（6 m、3 s、各窗口自身 p60 标定）在
原西南角窗口与新的东北角窗口上各跑一遍，检验结论是否依赖窗口选择。

这个实验是为回答"换场景算不算挑数据"：若两个窗口都为正、且新窗口更弱，
则窗口选择无法解释结果。
"""
import csv, json, sys, time

import os

HERE = os.path.dirname(os.path.abspath(__file__))
# 兼容两种布局：投稿包（脚本在 code/，数据在 ../data/）与开发工程（数据在 ./results/、./data/）。
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
BAG = os.path.abspath(os.path.join(HERE, "..", "data", "bag") if _PKG
                      else os.path.join(HERE, "data"))
TAG = "" if _PKG else "_bag"
from dataclasses import replace
import numpy as np

sys.path.insert(0, '.')
import bag_data
from config import Config
from environment import Environment
from highlevel import solve, arrival_index
from lowlevel import warmup_heuristics
from methods import METHODS

SW = (84300.0, 446800.0, 84400.0, 446900.0)   # 原窗口
NE = (84300.0, 446900.0, 84400.0, 447000.0)   # 现窗口
PAYLOAD = json.load(open(os.path.join(BAG, '3dbag_overview.geojson'), encoding='utf-8'))
METHODS_USED = ("A_BASE", "ECBS", "CBS_DOSE", "ECBS_DOSE")


def run_window(bbox, budget, seeds=range(42, 62), ns=(6, 8, 10, 12)):
    blocked, _ = bag_data.rasterize(PAYLOAD, bbox)
    rows = []
    for n in ns:
        for seed in seeds:
            cfg = replace(Config(), seed=seed, n_agents=n, t_budget=3.0,
                          heuristic_mode="exact", dose_budget=budget, d_min=6.0)
            env = Environment(cfg, seed, bag_layout=blocked, bag_meta={},
                              bottleneck=True)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg, env, tasks)
            for m in METHODS_USED:
                r = solve(cfg, env, tasks, METHODS[m],
                          time.monotonic() + 3.0, seed)
                d = ""
                if r.solved:
                    d = round(max(sum(env.smoke_at(r.paths[i][t], t) * cfg.dt
                                      for t in range(arrival_index(r.paths[i], g) + 1))
                                  for i, (s, g) in enumerate(tasks)), 4)
                rows.append(dict(bbox=str(bbox), n_agents=n, seed=seed,
                                 method=m, solved=int(r.solved),
                                 dose_to_arrival=d,
                                 planning_time=round(r.planning_time, 3)))
        print(f"  [{bbox[1]:.0f}] n={n} done", flush=True)
    return rows


if __name__ == "__main__":
    out = []
    for bbox, cap, tag in ((SW, 1.72, "sw"), (NE, 2.88, "ne")):
        print(f"=== {tag} bbox={bbox} cap={cap} ===", flush=True)
        rows = run_window(bbox, cap)
        with open(os.path.join(THESIS, f"window_{tag}{TAG}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        out.append((tag, cap, rows))
        print(f"  -> window_{tag}.csv", flush=True)

    print("\n窗口 | 方法 | 解出 | 违规 | 合规")
    for tag, cap, rows in out:
        for m in METHODS_USED:
            rs = [r for r in rows if r["method"] == m]
            sol = [r for r in rs if r["solved"] == 1]
            viol = [r for r in sol if r["dose_to_arrival"] != ""
                    and float(r["dose_to_arrival"]) > cap + 1e-6]
            print(f"{tag:>3} | {m:<10} | {len(sol):>2} | {len(viol):>2} | {len(sol)-len(viol):>2}")
