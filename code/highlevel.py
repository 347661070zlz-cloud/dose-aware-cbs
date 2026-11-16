"""高层多机冲突消解：基线 CBS / ECBS / ICBS / 优先规划 / WPP(近年改进) 与创新点 2。

创新点 2 = 影响评估驱动的「评估—修复—分裂」有界次优冲突消解：
  - 影响维度对齐灾害语义（持续时间、频次、局部危险度、绕行代价、耦合度）；
  - 低影响冲突先尝试局部修复（等待 / 短时绕行），失败或代价增量超 δ 才回退完备分裂；
  - 修复结果作为约束持久化写入节点，代价上升后重插 OPEN，维持 best-first 排序正确；
  - 明确有界次优：候选节点需满足 cost <= omega * lower_bound，并记录 omega 实测；
  - 预算将尽立即接受当前最好可行解（anytime）。

双向快照：每次高层迭代开始冻结冲突密度 D_k 下传给该次全部低层调用；
低层返回的风险/代价信息由 env.local_risk 与统一 C 上送给影响评估。
同时检测顶点冲突与边冲突（只检顶点会漏对向交换，破坏分支完备性）。
"""
import heapq
import itertools
import time
from dataclasses import dataclass, field

from config import Config
from cost import path_cost
from lowlevel import Constraints, plan, corridor_pso, LowLevelResult, BudgetSignal
from speedprofile import shift_from


@dataclass
class Conflict:
    kind: str          # "vertex" | "edge"
    a: int
    b: int
    t: int
    v: tuple           # 顶点位置 / 边起点
    v2: tuple = None   # 边终点

    def key(self):
        return (self.kind, min(self.a, self.b), max(self.a, self.b), self.t)


@dataclass
class CTNode:
    constraints: dict
    paths: dict
    cost: float
    nid: int
    n_conf: int = 0
    repairs: list = field(default_factory=list)


@dataclass
class HighLevelResult:
    method: str
    solved: bool
    paths: dict
    cost: float
    ttff: float
    planning_time: float
    ct_nodes: int
    low_expanded: int
    splits: int
    repair_attempts: int
    repair_success: int
    repair_overhead: float
    fallback_reason: str
    cont_paths: dict = field(default_factory=dict)
    pre_smoothed: bool = False


# ---------------------------------------------------------------------- 冲突
def detect_conflicts(cfg: Config, paths):
    out = []
    ids = sorted(paths)
    L = min(cfg.t_max, min(len(p) for p in paths.values()))
    for t in range(L):
        occ = {}
        for i in ids:
            occ.setdefault(tuple(paths[i][t]), []).append(i)
        for v, group in occ.items():
            if len(group) > 1:
                for a, b in itertools.combinations(group, 2):
                    out.append(Conflict("vertex", a, b, t, v))
        if t < L - 1:
            for a, b in itertools.combinations(ids, 2):
                pa, pb = paths[a], paths[b]
                if pa[t] == pb[t + 1] and pa[t + 1] == pb[t]:
                    out.append(Conflict("edge", a, b, t,
                                        tuple(pa[t]), tuple(pa[t + 1])))
    out.sort(key=lambda c: (c.t, c.a, c.b))
    return out


def conflict_density(cfg: Config, paths):
    """冻结的冲突密度快照 D_k，下传给本次高层迭代的所有低层调用。"""
    d = {}
    for p in paths.values():
        for v in p:
            d[tuple(v)] = d.get(tuple(v), 0) + 1
    mx = max(d.values()) if d else 1
    return {v: c / mx for v, c in d.items()}


def arrival_index(path, goal):
    for i, v in enumerate(path):
        if tuple(v) == tuple(goal):
            return i
    return len(path) - 1


def agent_cost(cfg, env, path, goal):
    return path_cost(cfg, env, path[:arrival_index(path, goal) + 1])


def node_cost(cfg, env, paths, goals):
    return sum(agent_cost(cfg, env, paths[i], goals[i][1]) for i in paths)


# ---------------------------------------------------------------------- 约束
def split_options(c):
    if c.kind == "vertex":
        return [(c.a, ("vertex", c.t, c.v)),
                (c.b, ("vertex", c.t, c.v))]
    return [(c.a, ("edge", c.t, c.v, c.v2)),
            (c.b, ("edge", c.t, c.v2, c.v))]


def add_constraint(cons, agent, item):
    c = cons[agent].copy()
    if item[0] == "vertex":
        c.vertex.add((item[1], item[2]))
    else:
        c.edge.add((item[1], item[2], item[3]))
    return c


def violates_constraints(path, cons, cfg):
    L = min(cfg.t_max, len(path))
    for t in range(L):
        if (t, tuple(path[t])) in cons.vertex:
            return True
        if t >= 1 and (t - 1, tuple(path[t - 1]), tuple(path[t])) in cons.edge:
            return True
    return False


# ---------------------------------------------------------------------- 影响评估
def impact_score(cfg: Config, env, c, conflicts, paths, goals, risk_maps=None):
    """I ∈ [0,1]：持续时间、发生频次、局部危险度、绕行代价、耦合程度。"""
    pair_keys = [x for x in conflicts
                 if min(x.a, x.b) == min(c.a, c.b) and max(x.a, x.b) == max(c.a, c.b)]
    duration = min(1.0, len(pair_keys) / max(1.0, 0.2 * cfg.t_max))
    frequency = min(1.0, len(pair_keys) / max(1.0, 0.25 * len(conflicts)))
    risk = env.local_risk(c.v, c.t)
    sdf = env.sdf_at(c.v)
    base_detour = min(1.0, max(0.0, 1.0 - sdf / (3.0 * cfg.d_min))
                      + 0.3 * env.obstacle_density(c.v))
    # 低层上送的绕行代价（实际路径代价 / 直线距离下界）参与影响评估
    low_detour = 0.0
    if risk_maps:
        ds = [risk_maps.get(x, {}).get("detour", 1.0) for x in (c.a, c.b)]
        low_detour = min(1.0, max(0.0, (max(ds) - 1.0) / 3.0))
    detour = min(1.0, 0.7 * base_detour + 0.3 * low_detour)
    near = 0
    for i, p in paths.items():
        if i in (c.a, c.b):
            continue
        if tuple(p[min(c.t, len(p) - 1)]) == tuple(c.v):
            near += 1
    coupling = min(1.0, near / max(1.0, 0.4 * len(paths)))
    return (0.20 * duration + 0.20 * frequency + 0.25 * risk
            + 0.20 * detour + 0.15 * coupling)


def is_cardinal_approx(cfg, env, c, paths, goals):
    """ICBS 的 cardinal 判定近似：无绕行空间或发生在到达段末端。"""
    ta = arrival_index(paths[c.a], goals[c.a][1])
    tb = arrival_index(paths[c.b], goals[c.b][1])
    return (c.t >= min(ta, tb) - 2) or (env.sdf_at(c.v) < 1.5 * cfg.d_min)


# ---------------------------------------------------------------------- 修复
def insert_wait(path, t, t_max):
    if t < 1 or t >= t_max:
        return None
    return path[:t] + [path[t - 1]] + path[t:t_max - 1]


def repair_many(cfg, env, node, conflicts, goals, spec, stats, root_lb,
                risk_maps=None, use_impact=True):
    """在一个约束树节点内批量修复多个低影响冲突，返回修复后的节点（或 None）。

    关键设计：修复结果"就地"继承到当前节点，随后在该节点上继续分裂，
    **不再把修复节点额外入堆**。若每修一个冲突就重新入堆，会产生大量近似
    重复节点，约束树反而膨胀（实测 ct_nodes 从 ~30 涨到 500+）。

    每个修复候选都要满足：不违反已有约束、目标冲突消失且不新增冲突、
    代价增量 <= delta*C_root、且不超过 omega*C_root 的次优界。
    CBS 约束只能"禁止"不能"强制停留"，所以修复结果靠路径继承而非写正向约束。
    """
    t0 = time.monotonic()
    paths = {i: list(p) for i, p in node.paths.items()}
    cons = {i: node.constraints[i].copy() for i in node.constraints}
    cur = list(conflicts)
    done = 0
    for _ in range(cfg.max_repairs_per_node):
        if not cur:
            break
        if use_impact:
            c = min(cur, key=lambda x: impact_score(cfg, env, x, cur, paths,
                                                    goals, risk_maps))
            if impact_score(cfg, env, c, cur, paths, goals,
                            risk_maps) >= cfg.impact_threshold:
                break
        else:   # 消融：去掉影响评估，按最早冲突尝试修复且不设影响阈值
            c = cur[0]
        stats["repair_attempts"] += 1
        ok = False
        for agent in (c.a, c.b):
            # 候选 1：单点插入等待（基线修复）
            cands = []
            w = insert_wait(paths[agent], c.t, cfg.t_max)
            if w is not None:
                cands.append(w)
            # 候选 2（A 方案）：速度剖面的"整体后移 / 减速"自由度，
            # 保持几何序列与相对时序，只改变到达时刻，比插一步等待更自然。
            if getattr(spec, "repair_use_speed", False):
                for delta in (1, 2):
                    for at in (max(1, c.t - 1), c.t):
                        s = shift_from(paths[agent], at, delta, cfg.t_max)
                        if s is not None:
                            cands.append(s)
            for cand in cands:
                if violates_constraints(cand, cons[agent], cfg):
                    continue
                trial = dict(paths)
                trial[agent] = cand
                newconf = detect_conflicts(cfg, trial)
                newkeys = {x.key() for x in newconf}
                if c.key() in newkeys or len(newkeys) > len(cur):
                    continue
                newcost = node_cost(cfg, env, trial, goals)
                # δ：修复的代价增量阈值，基准必须是**当前节点代价**（config.py 的
                # 注释即如此定义）。旧实现误用 root_lb，等于要求"修复后总代价仍
                # 不超过无约束最优和的 1.15 倍"——树越往下越不可能满足，深层节点
                # 的修复被系统性拒绝，创新点2 的修复能力被自己的实现压死了。
                if newcost > node.cost * (1.0 + cfg.delta):
                    continue
                # ω：有界次优界，基准是全树下界 root_lb（这部分旧实现是对的）
                if newcost > cfg.omega * root_lb:
                    continue
                paths = trial
                ok = True
                stats["repair_success"] += 1
                break
            if ok:
                break
        if not ok:
            break
        cur = detect_conflicts(cfg, paths)
        done += 1
    stats["repair_overhead"] += time.monotonic() - t0
    if done == 0:
        return None
    cost = node_cost(cfg, env, paths, goals)
    return CTNode(cons, paths, cost, -1, len(cur), node.repairs + [done])


# ---------------------------------------------------------------------- 规划
def plan_all(cfg, env, tasks, spec, cons, deadline_ts, density, seed,
             make_signal=None, risk_maps=None):
    paths, expanded, cont = {}, 0, {}
    for i, (s, g) in enumerate(tasks):
        sig = make_signal(0.0) if callable(make_signal) else None
        res = plan(cfg, env, s, g, cons.get(i, Constraints()),
                   spec.low_planner, False, deadline_ts,
                   density, seed * 1000 + i, budget_signal=sig,
                   speed_profile=spec.speed_profile,
                   dose_aware=getattr(spec, "dose_aware", False))
        if res is None:
            return None, expanded, {}
        paths[i] = res.grid_path
        expanded += res.expanded
        if risk_maps is not None:
            risk_maps[i] = res.risk_summary
        if res.cont_path is not None:
            cont[i] = res.cont_path
    return paths, expanded, cont


def optimize_final_paths(cfg, env, paths, goals, deadline_ts, seed):
    """阶段 B 只在最终解上执行一次。

    中间约束树节点只需要离散占用 grid_path 来检测冲突，没必要每次重规划都跑 PSO；
    否则单层成本被放大 β 倍、高层还没展开几个节点预算就耗尽（αβ 失衡）。
    这也是 anytime 的正确语义：用剩余预算改进最终交付的轨迹。
    """
    cont = {}
    for i, p in sorted(paths.items()):
        if deadline_ts is not None and time.monotonic() > deadline_ts:
            break
        arr = arrival_index(p, goals[i][1])
        seg = p[:arr + 1]
        skel = LowLevelResult(grid_path=seg, cost=path_cost(cfg, env, seg))
        traj, ok, _ = corridor_pso(cfg, env, skel, skel.cost, deadline_ts,
                                   seed * 1000 + i, length=arr + 1)
        if ok:
            cont[i] = traj
    return cont


def select_conflict(cfg, env, conflicts, strategy, paths, goals,
                    risk_maps=None, use_impact=True):
    if strategy == "icbs":
        cardinal = [c for c in conflicts if is_cardinal_approx(cfg, env, c, paths, goals)]
        return (cardinal or conflicts)[0]
    if strategy == "wpp_baseline":
        # 文献改进 CBS 的加权冲突队列（简化复现）：
        #   omega_i = alpha*c_i + beta*d_i + gamma*g_i
        #   c_i = 该时刻并发冲突数, d_i = 涉及智能体路径代价, g_i = 绕行/展开代理量
        # 早期版本把 -len(paths) 当权重项，在同一节点内恒为常数，等于没加权，
        # 会退化成"最早冲突优先"并造成额外开销——这里修正为真正的加权。
        # 注意：并行碰撞检测未实现（6 机规模下分组并行的收益不足以抵开销），
        # 故该配置只是"加权队列 + 代价剪枝"的简化复现，不能代表文献完整性能，
        # 论文中须标注，且不得据此断言"文献方法更差"。
        def _weight(c):
            concurrent = sum(1 for x in conflicts if x.t == c.t)
            dcost = sum(len(paths.get(a, ())) for a in (c.a, c.b)) / 100.0
            g = 0.0
            if risk_maps:
                g = sum(float(risk_maps.get(a, {}).get("detour", 1.0))
                        for a in (c.a, c.b))
            return 1.0 * concurrent + 0.5 * dcost + 0.3 * g
        return max(conflicts, key=_weight)
    if strategy == "ours":
        # 影响评估排序只作辅助；节点数的减少来自"修复优先于分裂"
        if not use_impact:      # 7 模块消融：去掉影响评估，退回最早冲突优先
            return conflicts[0]
        return max(conflicts, key=lambda c: impact_score(
            cfg, env, c, conflicts, paths, goals, risk_maps))
    return conflicts[0]


def solve(cfg: Config, env, tasks, spec, deadline_ts=None, seed=0):
    start_ts = time.monotonic()
    goals = tasks
    stats = {"ct_nodes": 0, "low_expanded": 0, "splits": 0,
             "repair_attempts": 0, "repair_success": 0, "repair_overhead": 0.0}
    cont_paths = {}
    risk_maps = {}                       # 低层上送的风险/绕行代价摘要
    total_budget = (deadline_ts - start_ts) if deadline_ts else 1e9
    adaptive = getattr(spec, "adaptive", False)
    ablate = getattr(spec, "ablate", "")
    use_impact = (ablate != "impact_eval")      # 7 模块消融：影响评估
    do_repair = (ablate != "local_repair")      # 7 模块消融：局部修复

    def remaining_ratio():
        if deadline_ts is None:
            return 1.0
        return max(0.0, deadline_ts - time.monotonic()) / max(1e-6, total_budget)

    def make_signal(priority=0.0):
        """高层 → 低层：冻结的剩余预算比例 + 当前冲突优先级。"""
        if not adaptive or deadline_ts is None:
            return None
        return BudgetSignal(remaining_ratio(), priority)

    # ---- 阶段 B：只在最终交付解上执行一次 ----
    # 曾经的尝试与结论（勿重犯）：
    #   把 optimize 搬进主循环、每找到一个更好的可行解就优化一次（anytime 语义），
    #   本意是让"协同"能改变解。实测结果：程序化场景曲率改善 10~15%，
    #   但真实 3D BAG 底图上曲率反而恶化 34%（0.3483 vs 0.2606）——
    #   因为门控按"剩余预算比例"判断，而真正交付的那个解通常出现在预算将尽时，
    #   于是它恰好被跳过，优化全花在了会被淘汰的中间解上。
    #   结论：该改动弊大于利，已回退。协同目前只有下面这一条通道：
    #   预算紧张时跳过阶段 B（β→1），把时间留给搜索、用可飞性换时限内可行。
    def stage_b_enabled():
        if not (spec.optimize and not getattr(spec, "speed_profile", False)):
            return False
        if getattr(spec, "force_stage_b", False):
            return True
        return (not adaptive) or remaining_ratio() > cfg.tight_budget_ratio

    pre_smoothed = False

    def finish(solved, paths, cost, reason, ttff=0.0):
        nonlocal pre_smoothed
        if solved and paths and getattr(spec, "post_smooth", False):
            # 事后平滑对照：搜索与基线完全相同，只把交付轨迹的折角拉圆。
            # 用它回答"速度剖面是不是等于事后平滑"——解出率必然与基线一致。
            from speedprofile import smooth_geometric
            for i, p in paths.items():
                cont_paths[i] = smooth_geometric(p, cfg.cell_xy, cfg.cell_z)
            pre_smoothed = True
        elif solved and paths and stage_b_enabled():
            cont_paths.update(
                optimize_final_paths(cfg, env, paths, goals, deadline_ts, seed))
        return HighLevelResult(
            method=spec.name, solved=solved, paths=paths or {}, cost=cost,
            ttff=ttff, planning_time=time.monotonic() - start_ts,
            ct_nodes=stats["ct_nodes"], low_expanded=stats["low_expanded"],
            splits=stats["splits"], repair_attempts=stats["repair_attempts"],
            repair_success=stats["repair_success"],
            repair_overhead=stats["repair_overhead"],
            fallback_reason=reason, cont_paths=cont_paths,
            pre_smoothed=pre_smoothed)

    def timed_out():
        return deadline_ts is not None and time.monotonic() > deadline_ts

    # ---- 优先规划（L1' 快速朴素基线）----
    if spec.high_strategy == "pp":
        cons = {}
        paths = {}
        for i, (s, g) in enumerate(tasks):
            if timed_out():
                return finish(False, None, 0.0, "budget")
            cset = Constraints()
            for j in range(i):
                pj = paths[j]
                for t in range(min(cfg.t_max, len(pj))):
                    cset.vertex.add((t, tuple(pj[t])))
                    if t >= 1:
                        cset.edge.add((t - 1, tuple(pj[t - 1]), tuple(pj[t])))
            res = plan(cfg, env, s, g, cset, spec.low_planner, False,
                       deadline_ts, {}, seed * 1000 + i,
                       budget_signal=make_signal(0.0),
                       speed_profile=spec.speed_profile,
                       dose_aware=getattr(spec, "dose_aware", False))
            if res is None:
                return finish(False, None, 0.0, "planner_failed")
            risk_maps[i] = res.risk_summary
            stats["low_expanded"] += res.expanded
            paths[i] = res.grid_path
            if res.cont_path is not None:
                cont_paths[i] = res.cont_path
        stats["ct_nodes"] = 1
        cost = node_cost(cfg, env, paths, goals)
        remaining = detect_conflicts(cfg, paths)
        ttff = time.monotonic() - start_ts
        return finish(not remaining, paths, cost,
                      "" if not remaining else "pp_conflict", ttff)

    # ---- CBS 系（含本文创新点 2）----
    root_cons = {i: Constraints() for i in range(len(tasks))}
    root_paths, exp, cont = plan_all(cfg, env, tasks, spec, root_cons,
                                     deadline_ts, {}, seed,
                                     make_signal=make_signal,
                                     risk_maps=risk_maps)
    stats["low_expanded"] += exp
    cont_paths.update(cont)
    if root_paths is None:
        return finish(False, None, 0.0, "root_failed")
    root_conf = detect_conflicts(cfg, root_paths)
    root = CTNode(root_cons, root_paths, node_cost(cfg, env, root_paths, goals),
                  0, len(root_conf))
    # 根节点 = 各机无约束最优路径之和，是 sum-of-costs 的全局下界。
    # δ（修复代价上限）与 ω（次优界）一律以它为基准，保证 C_found <= omega * C*。
    root_lb = root.cost

    # 每个方法可有自己的次优界（0 表示沿用 cfg.omega）
    w_hi = getattr(spec, "high_omega", 0.0) or cfg.omega
    counter = 1
    heap = [(root.cost, root.nid, root)]
    best = None
    best_ttff = 0.0
    first_feasible = None
    if not root_conf:
        first_feasible = time.monotonic() - start_ts
        best = root

    while heap and stats["ct_nodes"] < cfg.max_ct_nodes:
        if timed_out():
            break
        # EECBS：incumbent 上限剪枝——代价不低于当前最好解的节点不可能改进，
        # 直接丢弃（这是 EECBS 相对 ECBS 的核心剪枝；h_hat 用 incumbent 上界近似）
        if spec.high_strategy == "eecbs" and best is not None:
            heap = [(c, i, n) for c, i, n in heap if n.cost < best.cost - 1e-9]
            heapq.heapify(heap)
            if not heap:
                break
        cost_lb = heap[0][0]
        # ECBS 焦点列表：在 f <= omega * lb 的候选里取冲突最少者
        if spec.high_strategy in ("ecbs", "eecbs"):
            focal = [n for _, _, n in heap if n.cost <= w_hi * cost_lb]
            if not focal:
                _, _, node = heapq.heappop(heap)
            else:
                node = min(focal, key=lambda n: (n.n_conf, n.cost))
                heap = [(n.cost, n.nid, n) for _, _, n in heap if n is not node]
                heapq.heapify(heap)
        else:
            _, _, node = heapq.heappop(heap)
        stats["ct_nodes"] += 1
        conflicts = detect_conflicts(cfg, node.paths)
        if not conflicts:
            if first_feasible is None:
                first_feasible = time.monotonic() - start_ts
            if best is None or node.cost < best.cost:
                best, best_ttff = node, first_feasible
            # 最优 CBS 找到即返回；次优/anytime 方法在预算内继续改进
            if spec.high_strategy == "cbs":
                break
            continue

        # ---- 创新点 2：先在当前节点内批量修复低影响冲突，再决定是否分裂 ----
        if spec.high_strategy == "ours" and do_repair and not timed_out():
            rep = repair_many(cfg, env, node, conflicts, goals, spec,
                              stats, root_lb, risk_maps, use_impact=use_impact)
            if rep is not None:
                node = rep
                conflicts = detect_conflicts(cfg, node.paths)
                if not conflicts:
                    # 修复直接得到可行解，本节点无需分裂
                    if first_feasible is None:
                        first_feasible = time.monotonic() - start_ts
                    if best is None or node.cost < best.cost:
                        best, best_ttff = node, first_feasible
                    continue

        c = select_conflict(cfg, env, conflicts, spec.high_strategy,
                            node.paths, goals, risk_maps,
                            use_impact=use_impact)
        # 高层 → 低层：把当前冲突影响度作为优先级下传给本次分裂重规划
        priority = (impact_score(cfg, env, c, conflicts, node.paths, goals,
                                risk_maps) if adaptive else 0.0)

        # ---- 近年改进 CBS 的代价阈值剪枝 ----
        if spec.high_strategy == "wpp_baseline" \
                and node.cost > cfg.omega * cost_lb * 1.5:
            continue

        # ---- 完备分裂（修复失败 / 高影响冲突 / 全部基线）----
        stats["splits"] += 1
        density = conflict_density(cfg, node.paths)   # 冻结快照 D_k
        for agent, item in split_options(c):
            if timed_out():
                break
            child_cons = {i: node.constraints[i].copy() for i in node.constraints}
            child_cons[agent] = add_constraint(child_cons, agent, item)
            child_paths = dict(node.paths)
            res = plan(cfg, env, goals[agent][0], goals[agent][1],
                       child_cons[agent], spec.low_planner, False,
                       deadline_ts, density, seed * 1000 + agent,
                       budget_signal=make_signal(priority),
                       speed_profile=spec.speed_profile,
                       dose_aware=getattr(spec, "dose_aware", False))
            if res is None:
                continue
            risk_maps[agent] = res.risk_summary
            stats["low_expanded"] += res.expanded
            child_paths[agent] = res.grid_path
            if res.cont_path is not None:
                cont_paths[agent] = res.cont_path
            ccost = node_cost(cfg, env, child_paths, goals)
            # 有界次优过滤：本文方法以 C_root 为界（可证）；其余基线沿用堆顶下界
            if spec.high_strategy == "ours":
                if ccost > cfg.omega * root_lb:
                    continue
            elif spec.high_strategy in ("ecbs", "wpp_baseline"):
                if ccost > w_hi * cost_lb:
                    continue
            child = CTNode(child_cons, child_paths, ccost, counter,
                           len(detect_conflicts(cfg, child_paths)), list(node.repairs))
            counter += 1
            heapq.heappush(heap, (child.cost, child.nid, child))

    if best is None:
        return finish(False, None, 0.0,
                      "budget" if timed_out() else "exhausted")
    return finish(True, best.paths, best.cost,
                  "budget_stop" if timed_out() else "", best_ttff)
