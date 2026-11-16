"""P1：把「根节点单机搜索失败」按支配规则拆开。

对 28 个 refusal 场景，逐架跑 astar_dose（无约束 = 约束树的根搜索），
只切换 exact_dominance（True = 精确剂量支配；False = 部署所用的 K=16 分桶）。
若某场景在分桶下有一架搜不到路、在精确支配下全都能搜到，
则该 refusal 至少部分是桶化不完备造成的，而不是剂量不可行。
"""
import csv, os, sys, time
from dataclasses import replace
sys.path.insert(0, '.')
import thesis_experiments as T
T._USE_BAG = True; T._BAG.clear()
from config import Config
from lowlevel import astar_dose, Constraints, warmup_heuristics

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
TAG = "" if _PKG else "_bag"
BUDGET = 6.0

rows = [r for r in csv.DictReader(open(os.path.join(THESIS, f"detour{TAG}.csv"),
                                      encoding="utf-8-sig")) if r["feasible"] == "0"]
print(f"场景数 {len(rows)}", flush=True)
out = []
for r in rows:
    p = r["scenario"].split("_"); seed, n = int(p[1][1:]), int(p[2][1:])
    rec = {"scenario": r["scenario"], "n": n}
    for rule, flag in (("exact", True), ("bucketed", False)):
        cfg = replace(Config(), seed=seed, n_agents=n, t_budget=BUDGET,
                      heuristic_mode="exact", dose_budget=2.88, d_min=6.0,
                      t_max=48, exact_dominance=flag, max_low_expanded=200000)
        env = T.make_env(cfg, seed); tasks = env.tasks[:n]
        warmup_heuristics(cfg, env, tasks)
        failed = 0; t0 = time.monotonic()
        for (s, g) in tasks:
            path = astar_dose(cfg, env, s, g, Constraints(),
                              time.monotonic() + BUDGET, 0.0)
            if path is None:
                failed += 1
        rec[f"{rule}_failed_agents"] = failed
        rec[f"{rule}_time"] = round(time.monotonic() - t0, 2)
    out.append(rec)
    print(f"{rec['scenario']:<11} exact_fail={rec['exact_failed_agents']} "
          f"bucket_fail={rec['bucketed_failed_agents']}", flush=True)

ex_all = sum(1 for r in out if r["exact_failed_agents"] == 0)
bk_all = sum(1 for r in out if r["bucketed_failed_agents"] == 0)
rescued = [r["scenario"] for r in out
           if r["exact_failed_agents"] == 0 and r["bucketed_failed_agents"] > 0]
print(f"\n全架可解：精确 {ex_all}/{len(out)}，分桶 {bk_all}/{len(out)}")
print(f"分桶失败但精确全通：{len(rescued)} 个 {rescued[:6]}")
path = os.path.join(THESIS, f"dominance_probe{TAG}.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("写入", path)
