"""事后过滤基线：回答"把约束嵌进搜索"是否优于"先规划再过滤"。

三种对照：
  F0  filter-only：跑 cap-blind ECBS，合规就交付，超标就宣告无解（零成本）；
  F3  filter + retry(w_smoke×3)：被拒后加重烟雾软代价重解一次，再做同样的过滤；
  F10 filter + retry(w_smoke×10)：同上，权重更重。
与本文的约束层（ECBS_DOSE）比较合规交付数。

F0 不需要重算——它就是基线自己的合规子集；F3/F10 只对被拒场景重解，故很快。
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


def dose_of(env, res, tasks, cfg):
    return max(sum(env.smoke_at(res.paths[i][t], t) * cfg.dt
                   for t in range(arrival_index(res.paths[i], g) + 1))
               for i, (s_, g) in enumerate(tasks))


def run(budget=1.72, solve_budget=3.0, factors=(3.0, 10.0)):
    T._USE_BAG = True
    rows = list(csv.DictReader(open("results/thesis/dose_main_bag.csv",
                                   encoding="utf-8-sig")))
    by = {(r["scenario"], r["method"]): r for r in rows}
    scen = sorted({r["scenario"] for r in rows})

    out = []
    for k, sc in enumerate(scen, 1):
        base = by[(sc, "ECBS")]
        seed = int(base["seed"]); n = int(base["n_agents"])
        rec = dict(scenario=sc, n=n, ecbs_solved=int(base["solved"]),
                   ecbs_dose=base["dose_to_arrival"],
                   ecbs_time=float(base["planning_time"]))
        ok = (base["solved"] == "1" and base["dose_to_arrival"] != ""
              and float(base["dose_to_arrival"]) <= budget + 1e-9)
        rec["filter_only_ok"] = int(ok)
        for f in factors:
            rec[f"retry{int(f)}_ok"] = ""
            rec[f"retry{int(f)}_dose"] = ""
            rec[f"retry{int(f)}_time"] = ""
        if not ok:                      # 只有被拒的场景才需要重解
            for f in factors:
                cfg = replace(Config(), seed=seed, n_agents=n,
                              dose_budget=budget, t_budget=solve_budget,
                              heuristic_mode="exact",
                              w_smoke=Config().w_smoke * f)
                env = T.make_env(cfg, seed)
                tasks = env.tasks[:n]
                warmup_heuristics(cfg, env, tasks)
                res = solve(cfg, env, tasks, METHODS["ECBS"],
                            time.monotonic() + solve_budget, seed)
                # 记录重解本身的耗时（与主表同口径：求解器自报 planning_time）
                rec[f"retry{int(f)}_time"] = round(res.planning_time, 3)
                if res.solved:
                    d = dose_of(env, res, tasks, cfg)
                    rec[f"retry{int(f)}_ok"] = int(d <= budget + 1e-9)
                    rec[f"retry{int(f)}_dose"] = round(d, 4)
                else:
                    rec[f"retry{int(f)}_ok"] = 0
        out.append(rec)
        if k % 20 == 0:
            print(f"  {k}/{len(scen)}", flush=True)

    with open("results/thesis/filter_baseline_bag.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    n = len(out)
    f0 = sum(int(r["filter_only_ok"] or 0) for r in out)
    base_sum = sum(float(r["ecbs_time"]) for r in out)
    print(f"\nscenarios {n}")
    print(f"  ECBS + filter-only        : {f0} compliant, "
          f"total {base_sum:.1f} s (mean {base_sum/n:.2f} s)")
    for f in factors:
        add = sum(int(r[f"retry{int(f)}_ok"] or 0) for r in out)
        retry_sum = sum(float(r[f"retry{int(f)}_time"] or 0) for r in out)
        tot = base_sum + retry_sum
        print(f"  ECBS + filter + retry x{f:<4}: {f0 + add} compliant "
              f"(retry rescued {add}), total {tot:.1f} s "
              f"(mean {tot/n:.2f} s; retry pass {retry_sum:.1f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.72)
    ap.add_argument("--solve-budget", type=float, default=3.0)
    a = ap.parse_args()
    run(a.budget, a.solve_budget)
