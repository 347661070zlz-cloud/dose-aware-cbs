"""学位论文第 7 章补充实验（对应开题表"验证方法"6 条要求）。

补齐 `run.py` 主实验之外的四组 + 一组对标基线量化表：
  dynamic   动态障碍物 / 环境时变重规划（开题表 1、5）
  scale     机数规模 6/10/16/20（开题表 1 工程价值）
  factorial 机数规模 × 2×2 四格：判定"两层都改是否必需"的决定性实验
  weights   等权重 / AHP 经验权重 / CRITIC 客观权重 + 敏感性（开题表 3）
  params    delta / omega / T_budget 三参数敏感性
  baselines 开题表"对比算法"的量化对照：标准三维A*+CBS、文献改进A*、
            文献改进CBS、ECBS、本文完整方法（开题表 2、4）

零第三方依赖。用法：
    python thesis_experiments.py
    python thesis_experiments.py --group dynamic
"""
import argparse
import csv
import os
import time
from dataclasses import replace

from config import Config
from environment import Environment
from highlevel import solve
from lowlevel import warmup_heuristics
from methods import METHODS
from metrics import evaluate, deadline_reached

# ---- 底图开关：--bag 时用公开 3D BAG 真实建筑，否则用程序化随机楼块 ----
_USE_BAG = False
_BAG = {}


def make_env(cfg, seed):
    """按全局开关构造环境。bag 底图只加载一次并缓存。"""
    if not _USE_BAG:
        return Environment(cfg, seed)
    if not _BAG:
        from bag_data import load_layout
        layout, meta = load_layout()
        _BAG["layout"], _BAG["meta"] = layout, meta
    return Environment(cfg, seed, bag_layout=_BAG["layout"],
                       bag_meta=_BAG["meta"], bottleneck=True)

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _out_dir():
    """结果目录：投稿包内写回 ../data/thesis（与随包 CSV 同处），
    工程内仍写 results/thesis。这样复现命令产出的文件与论文表格一一对应。"""
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("PAPER_DATA")
    if env:
        return os.path.abspath(os.path.join(env, "thesis"))
    cand = os.path.abspath(os.path.join(here, "..", "data", "thesis"))
    if os.path.isdir(cand):
        return cand
    return os.path.join(here, "results", "thesis")


OUT = _out_dir()
# 在投稿包内（../data/thesis 存在）一律写规范文件名，便于按 README 复现；
# 工程目录下仍保留 _bag 后缀，避免覆盖程序化楼块的旧结果。
IN_PKG = os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data", "thesis"))

# 开题表"验证方法 3"：三组代价权重方案
WEIGHT_SCHEMES = {
    "equal": dict(w_dist=1.0, w_sdf=1.0, w_smoke=1.0, w_wind=1.0, w_z=1.0),
    "ahp": dict(w_dist=1.0, w_sdf=1.2, w_smoke=1.5, w_wind=0.8, w_z=0.6),
    "critic": dict(w_dist=1.0, w_sdf=0.9, w_smoke=0.7, w_wind=1.3, w_z=0.5),
}

# 主对比表配置。刻意把"低层配置 / 高层配置 / 是否联合"写成显式列，
# 避免读者分不清哪一行是联合方案、哪一行是 2x2 消融格。
# 命名约定（2026-11-06 二次修订）：
#   **只有一个联合框架** = 创新点1(低层) + 创新点2(高层) + 协同机制。
#   早期文档里的「A 版 / B 版」是同一联合框架的两种**低层实现**，不是两个方案：
#     B 版 = 开题表原样：安全走廊约束 PSO（空间轨迹优化）
#     A 版 = 后续重构：时间维速度剖面（时间重参数化）
#
#   ★ 2×2 析因的两处修正（本次修订的重点）：
#   (1) 旧表把 B_LOWONLY（＝本文低层：多尺度 A* + 走廊 PSO）标成"文献改进 A*"，
#       属于张冠李戴；现已补上真正的 REF_ASTAR（自适应权重 A*）作对标基线 1。
#   (2) 旧表 2×2 的三个格子来自两套不同低层实现（B/D 是 PSO 支、C 是速度剖面支），
#       且 D 格额外打开了协同开关，导致 D−C 无法归因。现拆成两个各自闭合的 2×2：
#         PSO 支     : A_BASE / B_LOWONLY   / C_HIGHONLY   / D_FULL_NAIVE
#         速度剖面支 : A_BASE / B_LOWONLY_A / C_HIGHONLY_A / D_FULL_A_NAIVE
#       协同作为第三个因子，分别由 D_FULL − D_FULL_NAIVE、D_FULL_A − D_FULL_A_NAIVE
#       单独衡量；两个 ★ 行是加上协同后的完整方法。
CONFIGS = [
    dict(label="对标基线_标准Astar加经典CBS", role="对标基线1+2 / 2x2的基线格",
         low="标准三维A*(固定步长)", high="经典CBS(完备分裂)",
         joint="否", note="非联合；2x2 的基线格 A", method="A_BASE"),
    dict(label="对标基线_文献改进Astar", role="对标基线1(改进A*,自适应权重)",
         low="文献改进A*(f=g+ω·h, ω随距离衰减)", high="经典CBS",
         joint="否", note="非联合；复现邵新峰等2025 的权重部分", method="REF_ASTAR"),
    dict(label="对标基线_文献改进CBS", role="对标基线2(改进CBS,简化复现)",
         low="标准三维A*", high="文献改进CBS(加权队列+剪枝)",
         joint="否", note="非联合；简化复现，见提纲6.5", method="WPP_BASELINE"),
    dict(label="压力测试_ECBS", role="自选强基线(压力测试)",
         low="标准三维A*", high="ECBS(焦点搜索)",
         joint="否", note="非联合；仅用于适用边界分析", method="ECBS"),
    # ---- PSO 支的 2×2（低层因子 = 多尺度A*+走廊PSO；高层因子 = 评估-修复-分裂）----
    dict(label="消融格_仅改低层_走廊PSO版", role="2x2-A 消融B格(仅改低层)",
         low="本文低层:多尺度A*+安全走廊PSO", high="经典CBS",
         joint="否", note="非联合；PSO 支 2x2 的 B 格", method="B_LOWONLY"),
    dict(label="消融格_仅改高层_走廊PSO版", role="2x2-A 消融C格(仅改高层)",
         low="标准三维A*(固定步长)", high="创新点2:评估-修复-分裂",
         joint="否", note="非联合；PSO 支 2x2 的 C 格", method="C_HIGHONLY"),
    dict(label="消融格_两层都改_无协同_走廊PSO版", role="2x2-A 消融D格(无协同)",
         low="本文低层:多尺度A*+安全走廊PSO", high="创新点2:评估-修复-分裂",
         joint="否", note="非联合；PSO 支 2x2 的 D 格（无协同对照）",
         method="D_FULL_NAIVE"),
    dict(label="联合方案_开题版低层_走廊PSO", role="★完整方法 / 开题表原样低层",
         low="开题版低层:多尺度A*+安全走廊PSO", high="创新点2:评估-修复-分裂",
         joint="是", note="同一联合框架；D格+预算感知协同",
         method="D_FULL"),
    # ---- 速度剖面支的 2×2（低层因子 = 多尺度A*+时间维速度剖面）----
    dict(label="消融格_仅改低层_速度剖面版", role="2x2-B 消融B格(仅改低层)",
         low="重构版低层:多尺度A*+速度剖面", high="经典CBS",
         joint="否", note="非联合；速度剖面支 2x2 的 B 格", method="B_LOWONLY_A"),
    dict(label="消融格_仅改高层_速度剖面版", role="2x2-B 消融C格(仅改高层)",
         low="标准三维A*(固定步长)", high="创新点2:评估-修复-分裂",
         joint="否", note="非联合；速度剖面支 2x2 的 C 格",
         method="C_HIGHONLY_A"),
    dict(label="消融格_两层都改_无协同_速度剖面版", role="2x2-B 消融D格(无协同)",
         low="重构版低层:多尺度A*+速度剖面", high="创新点2:评估-修复-分裂",
         joint="否", note="非联合；速度剖面支 2x2 的 D 格（无协同对照）",
         method="D_FULL_A_NAIVE"),
    dict(label="联合方案_重构版低层_速度剖面", role="★完整方法 / 重构版低层",
         low="重构版低层:多尺度A*+时间维速度剖面", high="创新点2:评估-修复-分裂",
         joint="是", note="同一联合框架；D格+预算感知协同",
         method="D_FULL_A"),
]


def write_csv(name, rows):
    if not rows:
        return
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path} ({len(rows)} rows)", flush=True)


def solve_once(cfg, env, tasks, spec, seed, t_budget=None):
    budget = t_budget if t_budget is not None else cfg.t_budget
    return solve(cfg, env, tasks, spec, time.monotonic() + budget, seed)


# ---------------------------------------------------------------- 1 动态重规划
def group_dynamic(base):
    rows = []
    methods = ("A_BASE", "WPP_BASELINE", "D_FULL", "D_FULL_A")
    for seed in (42, 43, 44, 45):
        cfg = replace(base, seed=seed, n_agents=6)
        env = Environment(cfg, seed)
        tasks = env.tasks[:6]
        warmup_heuristics(cfg, env, tasks)
        for name in methods:
            spec = METHODS[name]
            init = solve_once(cfg, env, tasks, spec, seed)
            rec = dict(seed=seed, method=name,
                       initial_solved=int(init.solved),
                       replan_solved=0, replan_latency="",
                       deadline_reached=0,
                       initial_ttff=round(init.ttff, 4) if init.solved else "")
            if init.solved:
                # 环境时变 + 任务点变更：让重规划有真实工作量
                env2 = Environment(cfg, seed)
                env2.apply_dynamic_change(cfg.dynamic_change_at)
                tc = cfg.dynamic_change_at
                new_tasks = []
                for i, (s, g) in enumerate(tasks):
                    pos = tuple(init.paths[i][min(tc, len(init.paths[i]) - 1)])
                    ng = env.tasks[(i + 6) % len(env.tasks)][1]
                    new_tasks.append((pos, ng))
                t0 = time.monotonic()
                repl = solve_once(cfg, env2, new_tasks, spec, seed + 7)
                rec.update(replan_solved=int(repl.solved),
                           replan_latency=round(time.monotonic() - t0, 4),
                           deadline_reached=int(deadline_reached(
                               cfg, repl.paths, new_tasks) if repl.solved else 0))
            rows.append(rec)
        print(f"[dynamic] seed={seed} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2 机数规模
def group_scale(base):
    rows = []
    methods = ("A_BASE", "WPP_BASELINE", "D_FULL_A")
    for n in (6, 10, 16, 20):
        cfg_n = replace(base, n_agents=n)
        for seed in (42, 43):
            env = Environment(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                r = evaluate(cfg_n, env, tasks, res, METHODS[name],
                             f"s{seed}_n{n}", None)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 solved=r["solved"], ttff=r["ttff"],
                                 planning_time=r["planning_time"],
                                 cost=r["cost"], ct_nodes=r["ct_nodes"],
                                 low_expanded=r["low_expanded"],
                                 p05_clearance=r["p05_clearance"],
                                 curv_viol_ratio=r["curv_viol_ratio"],
                                 fallback=r["fallback_reason"]))
        print(f"[scale] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2b 规模×析因
def group_factorial(base, seeds=(42, 43, 44)):
    """**决定性实验**：机数规模 × 2×2 四格，回答"两层都改是不是必须的"。

    动机：20 场景主实验里 D≈B≈C（单层改进已足够），联合看起来"没有增量"。
    但这可能只是 6 架 / 1.5 s 太容易造成的**收益饱和**，而不是联合无效。
    唯一能把两者区分开的实验是：随规模上升，单层改进先掉、联合后掉。

    方法取 A（基线）/ B'（仅低层·剖面）/ C（仅高层·PSO 支）/ D'（两层都改·剖面）。
    C 取 PSO 支是因为 C_HIGHONLY 与 C_HIGHONLY_A 的高层实现只在
    `repair_use_speed` 上不同，而该开关在低层为 fixed 时不生效，两者等价。
    """
    rows = []
    methods = ("A_BASE", "B_LOWONLY_A", "C_HIGHONLY", "D_FULL_A")
    for n in (6, 8, 10, 12, 16):
        cfg_n = replace(base, n_agents=n)
        for seed in seeds:
            env = Environment(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                r = evaluate(cfg_n, env, tasks, res, METHODS[name],
                             f"fact_s{seed}_n{n}", None)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 solved=r["solved"], ttff=r["ttff"],
                                 planning_time=r["planning_time"],
                                 cost=r["cost"], ct_nodes=r["ct_nodes"],
                                 low_expanded=r["low_expanded"],
                                 curv_viol_ratio=r["curv_viol_ratio"],
                                 budget_stop=int(res.fallback_reason
                                                 == "budget_stop"),
                                 fallback=r["fallback_reason"]))
        print(f"[factorial] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2c 可飞性对照
def group_fly(base):
    """**B 方案专用**：把"轨迹可飞性（曲率超标率）"做成可检验的主指标。

    为什么单开一组：主实验 20 个场景里，曲率只能在"双方都解出"的配对上比较，
    有效配对数只有 12，配对置换检验 p=0.077 —— 功效不足、结论悬空。
    本组把场景扩到 30 个（seed 42–71）、方法精简到 7 个，专测曲率。

    物理含义：曲率超标的直接后果是转弯半径 R = v²/(g·tanφ) 不够，旋翼机在
    该点必须减速或失稳。CBS / ECBS 系完全不优化这一维，因此这是灾害低空
    场景里与"时限内可行"并列的硬需求。
    """
    rows = []
    methods = ("A_BASE", "REF_ASTAR", "C_HIGHONLY", "ECBS",
               "B_LOWONLY_A", "C_HIGHONLY_A", "D_FULL_A")
    for seed in range(42, 72):
        cfg = replace(base, seed=seed, n_agents=6)
        env = Environment(cfg, seed)
        tasks = env.tasks[:6]
        warmup_heuristics(cfg, env, tasks)
        for name in methods:
            res = solve_once(cfg, env, tasks, METHODS[name], seed)
            r = evaluate(cfg, env, tasks, res, METHODS[name],
                         f"fly_s{seed}", None)
            rows.append(dict(seed=seed, method=name, scenario=f"fly_s{seed}",
                             solved=r["solved"],
                             planning_time=r["planning_time"],
                             cost=r["cost"],
                             curv_viol_ratio=r["curv_viol_ratio"],
                             p05_clearance=r["p05_clearance"],
                             avg_clearance=r["avg_clearance"],
                             curvature_max=r["curvature_max"],
                             ct_nodes=r["ct_nodes"],
                             fallback=r["fallback_reason"]))
        if seed % 10 == 1:
            print(f"[fly] seed={seed} done", flush=True)
    return rows


def flyability_stats(rows):
    """对 30 场景的可飞性做配对检验，并给出"解出率守门"的读数。

    曲率是连续量，用配对符号翻转置换检验；同时报告**只在双方都解出的配对上**
    的差值，避免用"谁解得多"污染"谁更可飞"。
    """
    from metrics import paired_permutation_test
    base = "A_BASE"
    outs = []
    for m in ("D_FULL_A", "B_LOWONLY_A", "C_HIGHONLY_A", "C_HIGHONLY",
              "ECBS", "REF_ASTAR"):
        for metric in ("curv_viol_ratio", "p05_clearance", "avg_clearance"):
            d, p, n = paired_permutation_test(rows, m, base, metric,
                                              only_solved=True)
            outs.append(dict(method=m, vs=base, metric=metric,
                             mean_diff=d, p_two_sided=p, pairs=n))
        for metric in ("curv_viol_ratio", "avg_clearance"):
            d, p, n = paired_permutation_test(rows, m, "ECBS", metric,
                                              only_solved=True)
            outs.append(dict(method=m, vs="ECBS", metric=metric,
                             mean_diff=d, p_two_sided=p, pairs=n))
    # 解出率守门（不配对，直接报比例）
    gate = {}
    for m in {r["method"] for r in rows}:
        rs = [r for r in rows if r["method"] == m]
        gate[m] = (round(sum(int(r["solved"]) for r in rs) / len(rs), 4),
                   len(rs))
    return outs, gate


# ---------------------------------------------------------------- 2d 低层归因
def group_decomp(base, seeds=(45, 46, 47, 48, 49, 50, 51, 52)):
    """**论文 1 的归因实验**：把"仅改低层"这一格拆成多尺度 / 速度剖面两个因子。

    5 个配置构成低层因子的 2×2 + 一个事后平滑对照：
      A_BASE       : 固定步长、无时间维            （基线，00）
      MULTI_ONLY   : 多尺度、无时间维              （空间维，10）
      SPEED_ONLY   : 固定步长 + 时间维速度剖面     （时间维，01）
      B_LOWONLY_A  : 多尺度 + 时间维速度剖面       （两者，11）
      POST_SMOOTH  : 搜索与基线完全相同 + 几何平滑 （事后处理对照）

    用途：
      1) 回答"3 倍解出率到底来自哪个因子"；
      2) 回答"时间维速度剖面是不是就等于事后 B 样条平滑"——若等，
         POST_SMOOTH 应当与它同效；若不等，二者在解出率上必然分岔。
    """
    rows = []
    methods = ("A_BASE", "MULTI_ONLY", "SPEED_ONLY", "B_LOWONLY_A",
               "POST_SMOOTH")
    for n in (6, 8, 10, 12, 16):
        cfg_n = replace(base, n_agents=n)
        for seed in seeds:
            env = Environment(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                r = evaluate(cfg_n, env, tasks, res, METHODS[name],
                             f"dec_s{seed}_n{n}", None)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 scenario=f"dec_s{seed}_n{n}",
                                 solved=r["solved"],
                                 planning_time=r["planning_time"],
                                 cost=r["cost"],
                                 curv_viol_ratio=r["curv_viol_ratio"],
                                 p05_clearance=r["p05_clearance"],
                                 avg_clearance=r["avg_clearance"],
                                 ct_nodes=r["ct_nodes"],
                                 fallback=r["fallback_reason"]))
        print(f"[decomp] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2e 累积暴露约束
def group_dose(base, budget=None, seeds=tuple(range(42, 72))):
    """累积烟雾暴露作为硬约束：剂量盲 vs 剂量感知。

    关键指标不是速度，而是**可行性**：
      剂量盲的 CBS 优化的是代价，完全不知道剂量预算，会交付超预算的解；
      剂量感知的 CBS 在低层用标签设置 + 帕累托支配搜索，保证交付解不超预算，
      代价是搜索开销与（可能的）代价上升。

    budget=None 时只跑剂量盲基线并打印剂量分布，用于校准预算；
    给定时同时跑两者，并逐场景判定"解是否满足剂量预算"。
    """
    from highlevel import arrival_index
    methods = ("A_BASE", "CBS_DOSE") if budget else ("A_BASE",)
    rows = []
    # 只用 6/8 架：更大规模连基线都解不出，剂量约束无从比较
    for n in (6, 8):
        cfg_n = replace(base, n_agents=n,
                        dose_budget=(budget if budget else 1e9))
        for seed in seeds:
            env = make_env(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                ev = evaluate(cfg_n, env, tasks, res, METHODS[name],
                              f"dose_s{seed}_n{n}", None)
                # 剂量是**单机**约束：每架机各有暴露上限，取"各机最大剂量"
                # 作为该场景的判据（旧版误用机队合计，与低层口径不一致）
                dmax, dmean, viol = "", "", ""
                if res.solved:
                    per = [sum(env.smoke_at(res.paths[i][t], t) * cfg_n.dt
                               for t in range(arrival_index(res.paths[i], g_) + 1))
                           for i, (s_, g_) in enumerate(tasks)]
                    dmax = round(max(per), 4)
                    dmean = round(sum(per) / len(per), 4)
                    if budget:
                        viol = sum(1 for x in per
                                   if x > cfg_n.dose_budget + 1e-9)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 scenario=f"dose_s{seed}_n{n}",
                                 solved=ev["solved"],
                                 dose_to_arrival=dmax,
                                 dose_mean=dmean,
                                 viol_agents=viol,
                                 dose_ok=(int(dmax != "" and
                                              float(dmax) <= cfg_n.dose_budget
                                              + 1e-9)
                                          if budget else ""),
                                 planning_time=ev["planning_time"],
                                 cost=ev["cost"],
                                 makespan_s=ev["makespan_s"],
                                 sum_arrival_s=ev["sum_arrival_s"],
                                 ttff=ev["ttff"],
                                 ct_nodes=ev["ct_nodes"],
                                 low_expanded=ev["low_expanded"]))
        print(f"[dose] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2f TAES 主实验
def group_dose_main(base, budget=4.45, seeds=tuple(range(42, 62)),
                    ns=(6, 8, 10, 12)):
    """TAES 主实验：剂量约束 × 两个高层算法 × 四个规模。

    设计意图（回应审稿人最可能的两问）：
      ① "只跟弱基线比" → 加入 ECBS、ECBS_DOSE 两个强对照；
      ② "规模太小" → 扩到 6/8/10/12 架，并报告大规模下的不可解率。
    关键论点：剂量约束的处理与高层算法**正交**——若 ECBS_DOSE 同样零违规，
    就证明该资源约束层可以加在任何 CBS 族高层之上，而不是本文 CBS 实现的补丁。
    """
    from highlevel import arrival_index
    methods = ("A_BASE", "ECBS", "CBS_DOSE", "ECBS_DOSE")
    rows = []
    for n in ns:
        cfg_n = replace(base, n_agents=n, dose_budget=budget)
        for seed in seeds:
            env = make_env(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                ev = evaluate(cfg_n, env, tasks, res, METHODS[name],
                              f"dm_s{seed}_n{n}", None)
                dmax, viol = "", ""
                if res.solved:
                    per = [sum(env.smoke_at(res.paths[i][t], t) * cfg_n.dt
                               for t in range(arrival_index(res.paths[i], g_) + 1))
                           for i, (s_, g_) in enumerate(tasks)]
                    dmax = round(max(per), 4)
                    viol = sum(1 for x in per if x > budget + 1e-9)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 scenario=f"dm_s{seed}_n{n}",
                                 solved=ev["solved"],
                                 dose_to_arrival=dmax,
                                 viol_agents=viol,
                                 planning_time=ev["planning_time"],
                                 cost=ev["cost"],
                                 makespan_s=ev["makespan_s"],
                                 sum_arrival_s=ev["sum_arrival_s"],
                                 ct_nodes=ev["ct_nodes"],
                                 low_expanded=ev["low_expanded"]))
        print(f"[dose_main] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2g ECBS 前沿
def group_dose_ecbs(base, budget=None, seeds=tuple(range(42, 72))):
    """暴露—时效前沿（ECBS 版本）：与主表口径一致。

    主表用 ECBS 作强基线，前沿若仍用 CBS 会造成"两套口径"。本组把同一实验
    换成 (ECBS, ECBS+Dose) 重新扫预算，保证图 4 与表 1 可比。
    """
    from highlevel import arrival_index
    methods = ("ECBS", "ECBS_DOSE") if budget else ("ECBS",)
    rows = []
    for n in (6, 8):
        cfg_n = replace(base, n_agents=n,
                        dose_budget=(budget if budget else 1e9))
        for seed in seeds:
            env = make_env(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                ev = evaluate(cfg_n, env, tasks, res, METHODS[name],
                              f"de_s{seed}_n{n}", None)
                dmax, viol = "", ""
                if res.solved:
                    per = [sum(env.smoke_at(res.paths[i][t], t) * cfg_n.dt
                               for t in range(arrival_index(res.paths[i], g_) + 1))
                           for i, (s_, g_) in enumerate(tasks)]
                    dmax = round(max(per), 4)
                    viol = sum(1 for x in per if x > budget + 1e-9)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 scenario=f"de_s{seed}_n{n}",
                                 solved=ev["solved"],
                                 dose_to_arrival=dmax,
                                 viol_agents=viol,
                                 planning_time=ev["planning_time"],
                                 makespan_s=ev["makespan_s"],
                                 cost=ev["cost"],
                                 ct_nodes=ev["ct_nodes"]))
        print(f"[dose_ecbs] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 2h 强基线扫描
def group_strong(base, budget=4.45, seeds=tuple(range(42, 62)),
                 ns=(6, 8, 10, 12)):
    """"方法越强、剂量盲违规越多"的单调性检验。

    没有自己实现 EECBS（它需要在高层维护可学习的 cost-to-go 估计，硬做一版
    反而不可信），改用同一框架内**加强搜索力度**来验证同一命题：
    放宽有界次优界 ω，ECBS 会解出更多、解质量更松。若违规率随 ω 单调上升，
    则更强的 EECBS 只会更严重，结论不必依赖某个具体实现。
    """
    from highlevel import arrival_index
    methods = ("ECBS", "ECBS_W30", "ECBS_W50")
    rows = []
    for n in ns:
        cfg_n = replace(base, n_agents=n, dose_budget=budget)
        for seed in seeds:
            env = make_env(cfg_n, seed)
            tasks = env.tasks[:n]
            warmup_heuristics(cfg_n, env, tasks)
            for name in methods:
                res = solve_once(cfg_n, env, tasks, METHODS[name], seed)
                ev = evaluate(cfg_n, env, tasks, res, METHODS[name],
                              f"st_s{seed}_n{n}", None)
                dmax, viol = "", ""
                if res.solved:
                    per = [sum(env.smoke_at(res.paths[i][t], t) * cfg_n.dt
                               for t in range(arrival_index(res.paths[i], g_) + 1))
                           for i, (s_, g_) in enumerate(tasks)]
                    dmax = round(max(per), 4)
                    viol = sum(1 for x in per if x > budget + 1e-9)
                rows.append(dict(n_agents=n, seed=seed, method=name,
                                 scenario=f"st_s{seed}_n{n}",
                                 solved=ev["solved"],
                                 dose_to_arrival=dmax,
                                 viol_agents=viol,
                                 planning_time=ev["planning_time"],
                                 cost=ev["cost"],
                                 makespan_s=ev["makespan_s"]))
        print(f"[strong] n={n} done", flush=True)
    return rows


# ---------------------------------------------------------------- 3 权重方案
def group_weights(base):
    rows = []
    methods = ("A_BASE", "D_FULL", "D_FULL_A")
    for tag, w in WEIGHT_SCHEMES.items():
        cfg_w = replace(base, seed=42, n_agents=6, **w)
        for seed in (42, 43, 44, 45):
            cfg = replace(cfg_w, seed=seed)
            env = Environment(cfg, seed)
            tasks = env.tasks[:6]
            warmup_heuristics(cfg, env, tasks)
            for name in methods:
                res = solve_once(cfg, env, tasks, METHODS[name], seed)
                r = evaluate(cfg, env, tasks, res, METHODS[name],
                             f"{tag}_s{seed}", None)
                rows.append(dict(scheme=tag, seed=seed, method=name,
                                 solved=r["solved"], cost=r["cost"],
                                 planning_time=r["planning_time"],
                                 smoke_dose=r["smoke_dose"],
                                 p05_clearance=r["p05_clearance"],
                                 curv_viol_ratio=r["curv_viol_ratio"],
                                 sum_arrival_s=r["sum_arrival_s"]))
        print(f"[weights] {tag} done", flush=True)
    return rows


# ---------------------------------------------------------------- 4 参数敏感性
def group_params(base):
    rows = []
    methods = ("D_FULL_A", "D_FULL", "ECBS")
    plans = (("delta", (0.05, 0.15, 0.30)),
             ("omega", (1.05, 1.20, 1.50)),
             ("t_budget", (0.8, 1.5, 2.5, 4.0)))
    for param, values in plans:
        for val in values:
            for seed in (42, 43, 44):
                cfg = replace(base, seed=seed, n_agents=6, **{param: val})
                env = Environment(cfg, seed)
                tasks = env.tasks[:6]
                warmup_heuristics(cfg, env, tasks)
                for name in methods:
                    res = solve_once(cfg, env, tasks, METHODS[name], seed)
                    r = evaluate(cfg, env, tasks, res, METHODS[name],
                                 f"{param}{val}_s{seed}", None)
                    rows.append(dict(param=param, value=val, seed=seed,
                                     method=name, solved=r["solved"],
                                     ttff=r["ttff"], cost=r["cost"],
                                     ct_nodes=r["ct_nodes"],
                                     repair_success=r["repair_success"],
                                     sum_arrival_s=r["sum_arrival_s"],
                                     p05_clearance=r["p05_clearance"]))
            print(f"[params] {param}={val} done", flush=True)
    return rows


# ---------------------------------------------------------------- 5 对标基线量化
def group_baselines(base):
    rows = []
    for seed in (42, 43, 44, 45, 46, 47, 48):
        cfg = replace(base, seed=seed, n_agents=6)
        env = Environment(cfg, seed)
        tasks = env.tasks[:6]
        warmup_heuristics(cfg, env, tasks)
        for conf in CONFIGS:
            name = conf["method"]
            res = solve_once(cfg, env, tasks, METHODS[name], seed)
            r = evaluate(cfg, env, tasks, res, METHODS[name],
                         f"s{seed}", None)
            rows.append(dict(seed=seed, label=conf["label"],
                             role=conf["role"], low=conf["low"],
                             high=conf["high"], joint=conf["joint"],
                             note=conf["note"], method=name,
                             solved=r["solved"],
                             path_cost=r["cost"],
                             planning_time=r["planning_time"],
                             total_risk_proxy=r["smoke_dose"],
                             p05_clearance=r["p05_clearance"],
                             avg_clearance=r["avg_clearance"],
                             curv_viol_ratio=r["curv_viol_ratio"],
                             compliant=r["compliant"],
                             safe_4p5=r["safe_4p5"],
                             high_level_nodes=r["ct_nodes"],
                             low_level_calls=r["low_expanded"],
                             makespan_s=r["makespan_s"]))
        print(f"[baselines] seed={seed} done", flush=True)
    return rows


def write_main_table(rows):
    """按配置聚合成一张可直接放进论文第 7 章的主对比表。"""
    import statistics
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r["label"]].append(r)
    fields = ("solved", "path_cost", "planning_time", "total_risk_proxy",
              "p05_clearance", "avg_clearance", "curv_viol_ratio", "compliant",
              "safe_4p5",
              "high_level_nodes", "low_level_calls", "makespan_s")
    out = []
    for conf in CONFIGS:
        rs = g.get(conf["label"], [])
        rec = {k: conf[k] for k in ("label", "role", "low", "high",
                                    "joint", "note", "method")}
        rec["cases"] = len(rs)
        for f in fields:
            vals = []
            for r in rs:
                v = r.get(f, "")
                if v != "":
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            rec[f] = round(statistics.mean(vals), 4) if vals else ""
        out.append(rec)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "thesis_main_table.csv")
    with open(path, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"  -> {path} ({len(out)} rows)", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=("dynamic", "scale", "factorial",
                                       "factor_holdout", "fly", "decomp",
                                       "dose", "dose_main", "dose_ecbs",
                                       "strong", "weights",
                                       "params", "baselines", "all"),
                   default="all")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--bag", action="store_true",
                   help="用公开 3D BAG 真实建筑底图（默认程序化楼块）")
    p.add_argument("--t-budget", type=float, default=None,
                   help="覆盖规划预算（剂量约束实验放宽预算以增加可解样本）")
    p.add_argument("--dose-buckets", type=int, default=None,
                   help="剂量桶数（消融：8/16/32）")
    p.add_argument("--no-dose-prune", action="store_true",
                   help="消融：关闭暴露下界剪枝")
    p.add_argument("--dose-budget", type=float, default=None,
                   help="累积剂量预算；不传则只跑剂量盲基线并打印剂量分布（校准用）")
    p.add_argument("--static-risk", action="store_true",
                   help="判决实验：冻结烟羽与风场（t=0），检验收益是否依赖时变性")
    args = p.parse_args()
    base = Config(t_budget=3.0, heuristic_mode="exact")
    global _USE_BAG
    if args.bag:
        _USE_BAG = True
        print("[bag] 使用 3D BAG 真实建筑底图", flush=True)
    if args.t_budget:
        base = replace(base, t_budget=args.t_budget)
    if args.dose_buckets:
        base = replace(base, dose_buckets=args.dose_buckets)
    if args.no_dose_prune:
        base = replace(base, dose_prune=False)
    if args.static_risk:
        base = replace(base, static_risk=True)
        print("[static_risk] 烟羽与风场已冻结在 t=0", flush=True)
    os.makedirs(OUT, exist_ok=True)
    groups = {
        "dynamic": ("thesis_dynamic.csv", group_dynamic),
        "scale": ("thesis_scale.csv", group_scale),
        "factorial": ("scale_factorial.csv", group_factorial),
        "factor_holdout": ("scale_factorial_holdout.csv", group_factorial),
        "fly": ("flyability.csv", group_fly),
        "decomp": ("paper1_decomp.csv", group_decomp),
        "dose": ("dose_constraint.csv", group_dose),
        "dose_main": ("dose_main.csv", group_dose_main),
        "dose_ecbs": ("dose_ecbs.csv", group_dose_ecbs),
        "strong": ("strong_baselines.csv", group_strong),
        "weights": ("thesis_weights.csv", group_weights),
        "params": ("thesis_params.csv", group_params),
        "baselines": ("thesis_baselines.csv", group_baselines),
    }
    todo = groups.keys() if args.group == "all" else (args.group,)
    for key in todo:
        name, fn = groups[key]
        print(f"==== {key} ====", flush=True)
        t0 = time.monotonic()
        if key == "strong":
            produced = fn(base, args.dose_budget or 4.45,
                          tuple(args.seeds) if args.seeds
                          else tuple(range(42, 62)))
        elif key == "dose_ecbs":
            produced = fn(base, args.dose_budget,
                          tuple(args.seeds) if args.seeds
                          else tuple(range(42, 72)))
        elif key == "dose_main":
            produced = fn(base, args.dose_budget or 4.45,
                          tuple(args.seeds) if args.seeds
                          else tuple(range(42, 62)))
        elif key == "dose":
            produced = fn(base, args.dose_budget,
                          tuple(args.seeds) if args.seeds
                          else tuple(range(42, 72)))
        elif key == "decomp" and args.seeds:
            produced = fn(base, tuple(args.seeds))
        elif key in ("factorial", "factor_holdout") and args.seeds:
            produced = fn(base, tuple(args.seeds))
        else:
            produced = fn(base)
        if args.static_risk and name.endswith(".csv"):
            name = name[:-4] + "_static.csv"
        if args.bag and not IN_PKG and name.endswith(".csv"):
            name = name[:-4] + "_bag.csv"
        write_csv(name, produced)
        if key == "baselines":
            write_main_table(produced)
        if key == "fly":
            stats, gate = flyability_stats(produced)
            write_csv("flyability_significance.csv", stats)
            print("\n==== 可飞性配对检验（仅双方都解出的配对）====", flush=True)
            for r in stats:
                print(f"  {r['method']:<14} vs {r['vs']:<8} "
                      f"[{r['metric']:<16}] diff={r['mean_diff']:>9} "
                      f"p={r['p_two_sided']:>7} pairs={r['pairs']}", flush=True)
            print("\n==== 解出率守门（30 场景）====", flush=True)
            for m, (v, n) in sorted(gate.items(), key=lambda x: -x[1][0]):
                print(f"  {m:<14} {v:.4f}  (n={n})", flush=True)
        print(f"  elapsed {time.monotonic()-t0:.1f}s", flush=True)
    print(f"结果目录: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
