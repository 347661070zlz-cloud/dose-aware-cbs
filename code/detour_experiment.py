"""绕路可行性实验：对剂量盲基线"违规"的场景，用放宽的规划预算重解，看是否存在合规路径。

这一版修正了三个问题（均由审稿核查指出）：
  1. **时间口径**：`makespan = max(arrival_index) * dt`（与 metrics.py 一致）。
     旧版用了 `arrival_index + 1`，给每个样本凭空加了 1 秒，把"零延迟"算成了 +9%~30%。
  2. **必须真的重解**：以 15 s 预算重新调用求解器，而不是从主表转抄。
  3. **脚本入库**：写入本文件，可复现、可审。

输出：results/thesis/detour_bag.csv
     列 = scenario, n, base_dose, base_T, feasible, new_dose, new_T, delta
"""
import argparse
import csv
import time

from dataclasses import replace

import thesis_experiments as T
from config import Config
from highlevel import solve, arrival_index
from lowlevel import warmup_heuristics
from methods import METHODS


def run(budget=1.72, solve_budget=15.0, methods=("ECBS", "ECBS_DOSE")):
    T._USE_BAG = True
    base_rows = list(csv.DictReader(
        open("results/thesis/dose_main_bag.csv", encoding="utf-8-sig")))
    viol = [r for r in base_rows
            if r["method"] == methods[0] and r["solved"] == "1"
            and r["dose_to_arrival"] != ""
            and float(r["dose_to_arrival"]) > budget]
    print(f"baseline-violating scenarios: {len(viol)}", flush=True)

    out = []
    for k, r in enumerate(viol, 1):
        seed, n = int(r["seed"]), int(r["n_agents"])
        cfg = replace(Config(), seed=seed, n_agents=n, dose_budget=budget,
                      t_budget=solve_budget, heuristic_mode="exact")
        env = T.make_env(cfg, seed)
        tasks = env.tasks[:n]
        warmup_heuristics(cfg, env, tasks)
        res = solve(cfg, env, tasks, METHODS[methods[1]],
                    time.monotonic() + solve_budget, seed)
        row = dict(scenario=r["scenario"], n=n,
                   base_dose=float(r["dose_to_arrival"]),
                   base_T=float(r["makespan_s"]),
                   feasible=0, new_dose="", new_T="", delta="",
                   planning_time="")
        row["planning_time"] = round(res.planning_time, 3)
        if res.solved:
            # 与 metrics.py 完全一致：到达步数 × dt，不加 1
            arr = [arrival_index(res.paths[i], g) * cfg.dt
                   for i, (s_, g) in enumerate(tasks)]
            per = [sum(env.smoke_at(res.paths[i][t], t) * cfg.dt
                       for t in range(arrival_index(res.paths[i], g) + 1))
                   for i, (s_, g) in enumerate(tasks)]
            t1 = max(arr)
            if max(per) <= budget + 1e-9:
                row.update(feasible=1, new_dose=round(max(per), 4),
                           new_T=round(t1, 3),
                           delta=round(t1 / row["base_T"] - 1, 4))
        out.append(row)
        if k % 10 == 0:
            print(f"  {k}/{len(viol)}", flush=True)

    with open("results/thesis/detour_bag.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    feas = [o for o in out if o["feasible"] == 1]
    import statistics as st
    print(f"\ndone: {len(out)} violating, {len(feas)} admite a compliant path")
    if feas:
        ds = sorted(o["delta"] for o in feas)
        z = sum(1 for d in ds if abs(d) < 1e-9)
        print(f"  delay median {st.median(ds):+.3f}, max {max(ds):+.3f}, "
              f"zero-delay {z}/{len(ds)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.72)
    ap.add_argument("--solve-budget", type=float, default=15.0)
    a = ap.parse_args()
    run(a.budget, a.solve_budget)
