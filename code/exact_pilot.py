"""精确剂量支配下的联合搜索 pilot（3 个场景）。

目的：判定每个 pilot 是「穷尽搜索空间后无解」还是「撞预算/上限中止」——
前者对离散化问题是不可行的证明，后者不是。
记录 ct_nodes / low_expanded / 是否耗尽（heap 空且未超时未触上限）。
"""
import csv, os, sys, time
from dataclasses import replace
sys.path.insert(0, '.')
import thesis_experiments as T
T._USE_BAG = True; T._BAG.clear()
from config import Config
from lowlevel import warmup_heuristics
from highlevel import solve
from methods import METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.isdir(os.path.join(HERE, "..", "data", "thesis"))
THESIS = os.path.abspath(os.path.join(HERE, "..", "data", "thesis") if _PKG
                         else os.path.join(HERE, "results", "thesis"))
TAG = "" if _PKG else "_bag"

PILOTS = [("dm_s42_n6", 42, 6), ("dm_s44_n6", 44, 6), ("dm_s46_n8", 46, 8)]
BUDGET = 25.0
out = []
for scn, seed, n in PILOTS:
    cfg = replace(Config(), seed=seed, n_agents=n, t_budget=BUDGET,
                  heuristic_mode="exact", dose_budget=2.88, d_min=6.0, t_max=48,
                  exact_dominance=True, max_low_expanded=200000, max_ct_nodes=20000)
    env = T.make_env(cfg, seed); tasks = env.tasks[:n]
    warmup_heuristics(cfg, env, tasks)
    t0 = time.monotonic()
    r = solve(cfg, env, tasks, METHODS["ECBS_DOSE"], t0 + BUDGET, seed)
    dt = time.monotonic() - t0
    out.append(dict(scenario=scn, n=n, solved=int(r.solved),
                    ct_nodes=r.ct_nodes, low_expanded=r.low_expanded,
                    time_s=round(dt, 2), reason=r.fallback_reason))
    print(f"{scn:<11} solved={int(r.solved)} ct={r.ct_nodes:<6} low={r.low_expanded:<7} "
          f"{dt:6.2f}s reason={r.fallback_reason}", flush=True)

path = os.path.join(THESIS, f"exact_feasibility{TAG}.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("写入", path)
