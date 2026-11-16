"""P1-2：对 28 个「无合规方案」实例放宽时间视界，看是否为 horizon 假阴性。

对 detour.csv 中 feasible==0 的 28 个场景，在 t_max ∈ {60, 72}（主表为 48）下重跑
剂量层，cap=2.88、预算 15 s。若某实例出现合规方案，则原先的「无解」是视界造成的。
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
BUDGET = 15.0

rows = [r for r in csv.DictReader(open(os.path.join(THESIS, f"detour{TAG}.csv"),
                                      encoding="utf-8-sig")) if r["feasible"] == "0"]
print(f"待诊断 {len(rows)} 个实例", flush=True)
out = []
for horizon in (60, 72):
    rescued = 0
    for r in rows:
        p = r["scenario"].split("_"); seed, n = int(p[1][1:]), int(p[2][1:])
        cfg = replace(Config(), seed=seed, n_agents=n, t_budget=BUDGET,
                      heuristic_mode="exact", dose_budget=2.88, d_min=6.0,
                      t_max=horizon)
        env = T.make_env(cfg, seed); tasks = env.tasks[:n]
        warmup_heuristics(cfg, env, tasks)
        res = solve(cfg, env, tasks, METHODS["ECBS_DOSE"],
                    time.monotonic() + BUDGET, seed)
        d = None
        if res.solved:
            d = max(sum(env.smoke_at(res.paths[i][t], t) * cfg.dt
                        for t in range(arrival_index(res.paths[i], g) + 1))
                    for i, (s, g) in enumerate(tasks))
        ok = res.solved and d is not None and d <= 2.88 + 1e-9
        rescued += int(ok)
        out.append(dict(scenario=r["scenario"], horizon=horizon, solved=int(res.solved),
                        compliant=int(ok), dose=None if d is None else round(d, 4),
                        time=round(res.planning_time, 2),
                        reason=res.fallback_reason))
    print(f"t_max={horizon}: 出现合规方案 {rescued}/{len(rows)}", flush=True)

path = os.path.join(THESIS, f"horizon_probe{TAG}.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("写入", path)
