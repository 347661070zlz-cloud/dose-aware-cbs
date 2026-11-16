"""A2 的可行替代：单机松弛证书。

逻辑：若某架飞机**单独**都不存在满足 D ≤ D̄ 的路径，则整个多机场景必然不可行。
这是 sound 的下界证书（不需要完备的联合搜索）。
对每个违规场景，用精确剂量支配逐架搜索（单机规模可算），统计被卡住的架数。
"""
import csv, os, sys, time
from collections import Counter
from dataclasses import replace

sys.path.insert(0, '.')
from config import Config
import thesis_experiments as T

# 必须与主表同底图：模块默认 _USE_BAG=False（程序化楼块），此处显式切到 3D BAG。
T._USE_BAG = True
T._BAG.clear()
from lowlevel import astar_dose, Constraints

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
TAG = "" if _PKG else "_bag"
AGENT_BUDGET = float(os.environ.get("AGENT_BUDGET", "12"))


def main():
    rows = list(csv.DictReader(open(os.path.join(THESIS, f"detour{TAG}.csv"),
                                   encoding="utf-8-sig")))
    out = []
    for r in rows:
        p = r["scenario"].split("_"); seed, n = int(p[1][1:]), int(p[2][1:])
        cfg = replace(Config(), seed=seed, n_agents=n, t_budget=AGENT_BUDGET,
                      heuristic_mode="exact", dose_budget=2.88, d_min=6.0,
                      t_max=48, exact_dominance=True, max_low_expanded=200000)
        env = T.make_env(cfg, seed); tasks = env.tasks[:n]
        blocked = []
        for i, (s, g) in enumerate(tasks):
            t0 = time.monotonic()
            path = astar_dose(cfg, env, s, g, Constraints(), t0 + AGENT_BUDGET, 0.0)
            if path is None:
                blocked.append(i)
        cert = int(bool(blocked))
        out.append(dict(scenario=r["scenario"], n=n,
                        bucketed_feasible=r["feasible"],
                        blocked_agents=len(blocked), certificate=cert))
        print(f"{r['scenario']:<12} n={n:<3} bucketed={r['feasible']} "
              f"blocked={len(blocked)}/{n}  cert={'YES' if cert else '-'}", flush=True)

    path = os.path.join(THESIS, f"certificate{TAG}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\n写入 {path}")
    c = Counter((x["bucketed_feasible"], x["certificate"]) for x in out)
    print("(bucketed_feasible, certificate) ->", dict(c))


if __name__ == "__main__":
    main()
