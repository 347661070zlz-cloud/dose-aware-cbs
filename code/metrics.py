"""指标与汇总。

主指标（删失数据稳健）：T_budget 内解出率、time-to-first-feasible、重规划延迟、
任务截止前到达率。绝不只用"成功样本的平均运行时间"。
次指标：总代价 C、gap、omega 实测、约束树节点数、低层展开数、修复开销，
以及外部独立风险量（最小离障距离 / 烟雾剂量 / 能耗代理 / 曲率峰值）。
"""
import random
import statistics

from config import Config
from cost import external_risk
from highlevel import arrival_index


def aggregate_risk(cfg: Config, env, paths, cont_paths=None,
                   pre_smoothed=False):
    out = {"min_clearance": float("inf"), "p05_clearance": float("inf"),
           "avg_clearance": 0.0, "below_safe_ratio": 0.0,
           "curv_viol_ratio": 0.0, "smoke_dose": 0.0,
           "energy_proxy": 0.0, "curvature_max": 0.0}
    if not paths:
        return {k: (0.0 if k != "min_clearance" and k != "p05_clearance" else 0.0)
                for k in out}
    cont_paths = cont_paths or {}
    acc_avg = 0.0
    for i, p in paths.items():
        r = external_risk(cfg, env, p, cont_paths.get(i),
                          pre_smoothed=pre_smoothed)
        out["min_clearance"] = min(out["min_clearance"], r["min_clearance"])
        out["p05_clearance"] = min(out["p05_clearance"], r["p05_clearance"])
        out["below_safe_ratio"] = max(out["below_safe_ratio"],
                                      r["below_safe_ratio"])
        out["curv_viol_ratio"] = max(out["curv_viol_ratio"],
                                     r["curv_viol_ratio"])
        out["smoke_dose"] += r["smoke_dose"]
        out["energy_proxy"] += r["energy_proxy"]
        out["curvature_max"] = max(out["curvature_max"], r["curvature_max"])
        acc_avg += r["avg_clearance"]
    out["avg_clearance"] = acc_avg / len(paths)
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in out.items()}


def deadline_reached(cfg: Config, paths, goals):
    if not paths:
        return False
    for i, p in paths.items():
        if arrival_index(p, goals[i][1]) * cfg.dt > cfg.deadline:
            return False
    return True


def evaluate(cfg, env, tasks, result, spec, scenario, c_star, dynamic=False,
             replan_latency=None):
    solved = bool(result.solved)
    cost = result.cost if solved else None
    risk = (aggregate_risk(cfg, env, result.paths, result.cont_paths,
                           pre_smoothed=getattr(result, "pre_smoothed", False))
            if solved else {})
    ttff = result.ttff if solved else None
    arrivals = ([arrival_index(p, tasks[i][1]) * cfg.dt
                 for i, p in result.paths.items()] if solved else [])
    sum_arrival = sum(arrivals) if arrivals else 0.0
    makespan = max(arrivals) if arrivals else 0.0
    gap = None
    omega_real = None
    if solved and c_star and c_star > 0:
        gap = (cost - c_star) / c_star
        omega_real = cost / c_star
    return {
        "scenario": scenario,
        "seed": cfg.seed,
        "n_agents": len(tasks),
        "dynamic": int(dynamic),
        "method": spec.name,
        "group": spec.group,
        "solved": int(solved),
        # 失败样本的 ttff 记为 None；汇总时按删失数据口径处理（另列 censored 均值）
        "ttff": round(ttff, 4) if ttff is not None else "",
        "planning_time": round(result.planning_time, 4),
        "cost": round(cost, 4) if cost is not None else "",
        "gap_vs_cstar": round(gap, 5) if gap is not None else "",
        # C* 缺失时（标准 CBS 在参照时限内没解出）仍可做同预算相对比较
        "gap_vs_best": "",
        "omega_realized": round(omega_real, 5) if omega_real is not None else "",
        "ct_nodes": result.ct_nodes,
        "low_expanded": result.low_expanded,
        "splits": result.splits,
        "repair_attempts": result.repair_attempts,
        "repair_success": result.repair_success,
        "repair_overhead": round(result.repair_overhead, 4),
        "min_clearance": risk.get("min_clearance", ""),
        "p05_clearance": risk.get("p05_clearance", ""),
        "avg_clearance": risk.get("avg_clearance", ""),
        "below_safe_ratio": risk.get("below_safe_ratio", ""),
        "curv_viol_ratio": risk.get("curv_viol_ratio", ""),
        "smoke_dose": risk.get("smoke_dose", ""),
        "energy_proxy": risk.get("energy_proxy", ""),
        "curvature_max": risk.get("curvature_max", ""),
        "sum_arrival_s": round(sum_arrival, 3),
        "makespan_s": round(makespan, 3),
        "deadline_reached": int(deadline_reached(cfg, result.paths, tasks)) if solved else 0,
        # 灾害场景交付验收：p05 离障裕度 + 剂量上限。
        # 阈值取 cfg.clearance_required = 4.0 m。旧版 3.5 m 低于离散体素 SDF 的
        # 次小值 3.536 m，使本列恒等于 solved（240/240 案例），已修正。
        "compliant": int(solved
                         and risk.get("p05_clearance", 0.0) >= cfg.clearance_required
                         and risk.get("smoke_dose", 0.0) <= cfg.smoke_dose_cap),
        # 多档安全率（只报告，不参与主判据）：让阈值依赖显式可见
        "safe_3p5": int(solved and risk.get("p05_clearance", -1.0) >= 3.5),
        "safe_4p5": int(solved and risk.get("p05_clearance", -1.0) >= 4.5),
        "safe_5p0": int(solved and risk.get("p05_clearance", -1.0) >= 5.0),
        "replan_latency": round(replan_latency, 4) if replan_latency is not None else "",
        "fallback_reason": result.fallback_reason,
    }


def _mean(vals):
    vals = [v for v in vals if v != "" and v is not None]
    return round(statistics.mean(vals), 4) if vals else ""


def summarize(rows, cfg: Config):
    """按方法聚合。solved_rate 是主指标；其余均值明确区分"仅成功样本"与删失口径。"""
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    out = []
    for name, rs in sorted(by.items()):
        n = len(rs)
        solved = [r for r in rs if r["solved"] == 1]
        # 删失口径：失败样本的首次可行时间按预算上限计入，避免选择性报告
        censored_ttff = [r["ttff"] if r["ttff"] != "" else cfg.t_budget for r in rs]
        out.append({
            "method": name,
            "group": rs[0]["group"],
            "cases": n,
            "solved_rate": round(len(solved) / n, 4),
            "compliant_rate": round(sum(r.get("compliant", 0) for r in rs) / n, 4),
            "safety_rate_3p5": round(sum(r.get("safe_3p5", 0) for r in rs) / n, 4),
            "safety_rate_4p5": round(sum(r.get("safe_4p5", 0) for r in rs) / n, 4),
            "safety_rate_5p0": round(sum(r.get("safe_5p0", 0) for r in rs) / n, 4),
            "binding_rate": round(
                sum(1 for r in rs if r["solved"] == 1
                    and r.get("p05_clearance", 0) != ""
                    and float(r["p05_clearance"]) < cfg.clearance_required) / n, 4),
            "ttff_mean_solved": _mean([r["ttff"] for r in solved]),
            "ttff_censored_mean": round(statistics.mean(censored_ttff), 4),
            "cost_mean_solved": _mean([r["cost"] for r in solved]),
            "gap_mean_solved": _mean([r["gap_vs_cstar"] for r in solved]),
            "gap_vs_best_mean": _mean([r.get("gap_vs_best", "") for r in solved]),
            "omega_max_solved": _mean([r["omega_realized"] for r in solved]),
            "ct_nodes_mean": _mean([r["ct_nodes"] for r in rs]),
            "low_expanded_mean": _mean([r["low_expanded"] for r in rs]),
            "splits_mean": _mean([r["splits"] for r in rs]),
            "repair_success_rate": round(
                sum(r["repair_success"] for r in rs) /
                max(1, sum(r["repair_attempts"] for r in rs)), 4),
            "min_clearance_mean": _mean([r["min_clearance"] for r in solved]),
            "p05_clearance_mean": _mean([r["p05_clearance"] for r in solved]),
            "curv_viol_mean": _mean([r["curv_viol_ratio"] for r in solved]),
            "sum_arrival_mean": _mean([r["sum_arrival_s"] for r in solved]),
            "smoke_dose_mean": _mean([r["smoke_dose"] for r in solved]),
            "curvature_max_mean": _mean([r["curvature_max"] for r in solved]),
            "deadline_rate": round(
                sum(r["deadline_reached"] for r in rs) / n, 4),
        })
    return out


def mcnemar_exact(rows, method_a, method_b, metric="solved"):
    """配对二值指标的 McNemar 精确检验（纯标准库，二项分布精确双侧 p）。

    解出率/合规率是"同一批场景上的 0/1 结果"，属于配对二值数据；此时基准做法是
    McNemar 而不是把两组比例当独立样本做 Fisher。统计量只看不一致对
    b01（A 失败 B 成功）与 b10（A 成功 B 失败），b01==b10 时 p 恒为 1。

    返回 (n_pairs, b01, b10, 双侧精确 p)。
    """
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["method"]] = r
    b01 = b10 = n = 0
    for d in by.values():
        if method_a not in d or method_b not in d:
            continue
        va = d[method_a].get(metric, "")
        vb = d[method_b].get(metric, "")
        va = int(float(va)) if va != "" else 0
        vb = int(float(vb)) if vb != "" else 0
        n += 1
        if va == 0 and vb == 1:
            b01 += 1
        elif va == 1 and vb == 0:
            b10 += 1
    m = b01 + b10
    if m == 0:
        return n, b01, b10, 1.0
    k = min(b01, b10)
    # 双侧精确 p = 2 * P(X <= k | n=m, p=0.5)，截断到 1
    tail = sum(_binom_pmf(i, m) for i in range(0, k + 1))
    return n, b01, b10, round(min(1.0, 2.0 * tail), 4)


def _binom_pmf(k, n):
    from math import comb
    return comb(n, k) / (2.0 ** n)


def paired_permutation_test(rows, method_a, method_b, metric="compliant",
                            n_perm=20000, seed=7, only_solved=False):
    """按 scenario 配对的符号翻转置换检验（无分布假设，纯标准库）。

    返回 (平均差值, 双侧 p 值, 配对数)。用于判断"联合方案是否显著优于最强基线"，
    避免用 4 个样本的百分比差异下结论。
    """
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["method"]] = r
    diffs = []
    for d in by.values():
        if method_a in d and method_b in d:
            if only_solved and (d[method_a]["solved"] != 1
                                or d[method_b]["solved"] != 1):
                continue
            va = d[method_a].get(metric, "")
            vb = d[method_b].get(metric, "")
            va = float(va) if va != "" else 0.0
            vb = float(vb) if vb != "" else 0.0
            diffs.append(va - vb)
    if not diffs:
        return 0.0, 1.0, 0
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        s = sum(x if rng.random() < 0.5 else -x for x in diffs) / len(diffs)
        if abs(s) >= abs(obs) - 1e-12:
            count += 1
    return round(obs, 4), round((count + 1) / (n_perm + 1), 4), len(diffs)


def clearance_sensitivity(rows, thresholds, smoke_cap=1e9):
    """合规率随安全裕度阈值的变化，避免"照着结果挑一个阈值"。"""
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    out = []
    for thr in thresholds:
        for name, rs in sorted(by.items()):
            ok = sum(1 for r in rs
                     if r["solved"] == 1
                     and r["p05_clearance"] != ""
                     and float(r["p05_clearance"]) >= thr
                     and (r["smoke_dose"] == ""
                          or float(r["smoke_dose"]) <= smoke_cap))
            out.append({"threshold_m": thr, "method": name,
                        "cases": len(rs),
                        "compliant_rate": round(ok / len(rs), 4)})
    return out
