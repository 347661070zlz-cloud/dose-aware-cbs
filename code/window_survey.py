"""窗口普查：对 400 m 图内**所有满足几何判据**的 100 m 窗口，各跑一遍完整协议。

判据（与方法无关，只看几何）：
  1) 瓶颈墙缺口在 d_min=6 m 下存在贯通行（x=9..12 同行同层均达标）；
  2) 该窗口下 6 架任务中至少 4 个跨墙（起终点分列墙两侧）。

每个窗口：先用基线跑 40 场景标定 p60 作为该窗口的 cap，再跑 4 种方法。
目的：检验"约束层带来正增益"这一结论是否只在特定窗口成立。
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

X0, Y0 = 84300.0, 446800.0
PAYLOAD = json.load(open(os.path.join(BAG, '3dbag_overview.geojson'), encoding='utf-8'))
QUAL = [(0, 50), (0, 100), (50, 0), (50, 250), (100, 0),
        (150, 100), (150, 200), (200, 100), (250, 300), (300, 50)]
SEEDS = range(42, 52)          # 10 个种子
NS = (6, 8, 10, 12)            # 4 个规模 → 40 场景/窗口


def build(bbox_seed, n):
    dx, dy = bbox_seed
    bbox = (X0 + dx, Y0 + dy, X0 + dx + 100, Y0 + dy + 100)
    blocked, _ = bag_data.rasterize(PAYLOAD, bbox)
    return blocked, bbox


def main():
    summary = []
    for dx, dy in QUAL:
        blocked, bbox = build((dx, dy), 6)
        tag = f"x{dx}_y{dy}"
        print(f"\n=== window {tag} (bbox {bbox}) ===", flush=True)

        # 1) 标定：基线跑全部场景，取 p60
        doses = []
        envs = {}
        for n in NS:
            for seed in SEEDS:
                cfg = replace(Config(), seed=seed, n_agents=n, t_budget=3.0,
                              heuristic_mode="exact", dose_budget=1e9, d_min=6.0)
                env = Environment(cfg, seed, bag_layout=blocked, bag_meta={},
                                  bottleneck=True)
                tasks = env.tasks[:n]
                warmup_heuristics(cfg, env, tasks)
                envs[(n, seed)] = (cfg, env, tasks)
                r = solve(cfg, env, tasks, METHODS["A_BASE"],
                          time.monotonic() + 3.0, seed)
                if r.solved:
                    doses.append(max(sum(env.smoke_at(r.paths[i][t], t) * cfg.dt
                                         for t in range(arrival_index(r.paths[i], g) + 1))
                                     for i, (s, g) in enumerate(tasks)))
        cap = float(np.percentile(doses, 60))
        print(f"  baseline solved {len(doses)}/40, cap(p60) = {cap:.3f}", flush=True)

        # 2) 四种方法在该 cap 下跑
        rows = []
        for (n, seed), (cfg0, env, tasks) in envs.items():
            cfg = replace(cfg0, dose_budget=cap)
            for m in ("A_BASE", "ECBS", "CBS_DOSE", "ECBS_DOSE"):
                r = solve(cfg, env, tasks, METHODS[m],
                          time.monotonic() + 3.0, seed)
                d = ""
                if r.solved:
                    d = round(max(sum(env.smoke_at(r.paths[i][t], t) * cfg.dt
                                      for t in range(arrival_index(r.paths[i], g) + 1))
                                  for i, (s, g) in enumerate(tasks)), 4)
                rows.append(dict(window=tag, bbox=list(bbox), cap=round(cap, 4),
                                 n_agents=n, seed=seed, method=m,
                                 solved=int(r.solved), dose_to_arrival=d,
                                 planning_time=round(r.planning_time, 3)))

        with open(os.path.join(THESIS, f"window_{tag}{TAG}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        def compl(m):
            rs = [r for r in rows if r["method"] == m]
            sol = [r for r in rs if r["solved"] == 1]
            viol = [r for r in sol if r["dose_to_arrival"] != ""
                    and float(r["dose_to_arrival"]) > cap + 1e-6]
            return len(sol) - len(viol), len(sol), len(viol)

        c_cbs, s_cbs, _ = compl("A_BASE")
        c_ecbs, s_ecbs, _ = compl("ECBS")
        c_cdl, _, _ = compl("CBS_DOSE")
        c_edl, _, _ = compl("ECBS_DOSE")
        summary.append(dict(window=tag, dx=dx, dy=dy, cap=round(cap, 4),
                            cbs_compl=c_cbs, ecbs_compl=c_ecbs,
                            cbs_layer=c_cdl, ecbs_layer=c_edl,
                            gain_ecbs=c_edl - c_ecbs, gain_cbs=c_cdl - c_cbs))
        print(f"  ECBS {c_ecbs} -> {c_edl} (gain {c_edl-c_ecbs:+d}); "
              f"CBS {c_cbs} -> {c_cdl} (gain {c_cdl-c_cbs:+d})", flush=True)

    with open(os.path.join(THESIS, f"window_survey{TAG}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    g = [s["gain_ecbs"] for s in summary]
    print(f"\n=== 汇总：{len(g)} 个窗口 ===")
    print(f"  ECBS 增益: 均值 {np.mean(g):+.2f}, 范围 [{min(g):+d}, {max(g):+d}], "
          f"为正 {sum(1 for x in g if x > 0)}/{len(g)}")


if __name__ == "__main__":
    main()
