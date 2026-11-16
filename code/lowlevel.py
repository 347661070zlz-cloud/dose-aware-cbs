"""创新点 1（低层）：风险感知自适应多尺度时空 A* + 安全走廊约束 PSO 两阶段 anytime 规划。

同时提供用于对照的低层基线：
  - fixed   : 固定单步长风险时空 A*（隔离"多尺度"贡献）
  - sipp    : SIPP 风格安全间隔跳跃（保最优提速的经典低层替代）
  - global_pso / gradient : 走廊 PSO 的选型对照

关键实现约束（对应前面的接口分析）：
1. Δt 固定不变；大空间步长被拆成多个"一个体素 = 一个时间步"的子步，中间体素同样
   检查硬约束 H 与高层下发的约束，因此不会跨过被禁止的 (t, voxel)；
2. 逆风使边权非对称 => 允许重开 CLOSED 节点（g 变小时重新入堆）；
3. 阶段 B 的连续轨迹不改变离散占用，高层仍以骨架 grid_path 做冲突检测；
   连续轨迹按统一 Δt 重采样后复核 H，不过则回退骨架；
4. anytime：预算耗尽 / 迭代上限 / 优化失败 / 复核不过，任一即返回骨架。
"""
import heapq
import math
import random
import time
from dataclasses import dataclass, field

from config import Config, chebyshev
from cost import (edge_metrics, hard_ok, heuristic, path_cost,
                  external_risk, c_min)
from speedprofile import optimize as speed_optimize

ZERO = (0, 0, 0)
UNIT_DIRS = [(dx, dy, dz)
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             for dz in (-1, 0, 1)
             if not (dx == 0 and dy == 0 and dz == 0)]

_EXACT_TABLES = {}


def _static_lower_cost(cfg, env, u, v):
    """忽略非负的 smoke/wind 项后的严格下界边权（用于反向 Dijkstra）。"""
    dx = (v[0] - u[0]) * cfg.cell_xy
    dy = (v[1] - u[1]) * cfg.cell_xy
    dz = (v[2] - u[2]) * cfg.cell_z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    base = cfg.w_dist * dist if dist > 1e-9 else c_min(cfg)
    soft = cfg.w_sdf * math.exp(
        -max(0.0, env.sdf_at(v) - cfg.d_min) / cfg.sigma_sdf)
    return base + soft + cfg.w_z * abs(v[2] - u[2])


def exact_table(cfg, env, goal):
    """从 goal 反向 Dijkstra，得到松弛图上的精确 h*（可采纳且远紧于欧氏下界）。

    借鉴 600m 项目 astar_planner.py 的 heuristic_scale='exact'：把整条静态
    step_cost 作边权预计算，能大幅降低低层展开节点数，即直接压低 β。
    """
    goal = tuple(goal)
    # 缓存键必须包含环境指纹：仅用 seed + 栅格尺寸不足以区分"同一 seed 下换了底图"
    # （多 tile 外部效度实验在同一进程里连续跑 3 张真实底图，只靠 seed 会串用 h* 表）。
    key = (getattr(env, "fingerprint", None), env.seed,
           cfg.nx, cfg.ny, cfg.nz, cfg.d_min,
           cfg.w_dist, cfg.w_sdf, cfg.w_z, goal)
    if key in _EXACT_TABLES:
        return _EXACT_TABLES[key]
    dist = {goal: 0.0}
    heap = [(0.0, goal)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")) + 1e-9:
            continue
        for dv in UNIT_DIRS:
            v = (u[0] + dv[0], u[1] + dv[1], u[2] + dv[2])
            if not env.is_free(v) or env.sdf_at(v) < cfg.d_min:
                continue
            nd = d + _static_lower_cost(cfg, env, v, u)
            if nd < dist.get(v, float("inf")) - 1e-9:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    _EXACT_TABLES[key] = dist
    return dist


def _h(cfg, v, goal, htable):
    if htable is not None:
        val = htable.get(tuple(v))
        if val is not None:
            return max(0.0, val)
    return heuristic(cfg, v, goal)


@dataclass
class BudgetSignal:
    """高层下传给低层的协同信号（第 k 次高层迭代冻结）。

    remaining_ratio: 剩余预算 / 总预算。低层据此决定本次调用是否启用多尺度与阶段 B。
    conflict_priority: 当前待消解冲突的影响度，融入冲突密度快照 D_k。
    """
    remaining_ratio: float = 1.0
    conflict_priority: float = 0.0


@dataclass
class LowLevelResult:
    grid_path: list
    cont_path: list = None
    cost: float = 0.0
    expanded: int = 0
    ttff: float = 0.0            # 首次可行解时间
    opt_success: bool = False
    opt_feasible_ratio: float = 0.0
    fallback_reason: str = ""
    planner: str = ""
    risk_summary: dict = field(default_factory=dict)  # 上送给高层影响评估
    speed_info: dict = field(default_factory=dict)    # 速度剖面认证信息


@dataclass
class Constraints:
    """单机约束：高层在第 k 次迭代下发的顶点/边约束（统一 Δt 栅格）。"""
    vertex: set = field(default_factory=set)   # (t, voxel)
    edge: set = field(default_factory=set)     # (t, from_voxel, to_voxel)

    def violates(self, prev, q, t):
        return (t, tuple(q)) in self.vertex or (t - 1, tuple(prev), tuple(q)) in self.edge

    def copy(self):
        return Constraints(set(self.vertex), set(self.edge))


# ---------------------------------------------------------------------- 多尺度
def choose_radius(cfg: Config, env, v, goal, conflict_density):
    """依据目标距离、局部净空、障碍密度、冻结的冲突密度 D_k 选择空间扩展半径。"""
    d = chebyshev(v, goal)
    sdf = env.sdf_at(v)
    density = env.obstacle_density(v)
    cd = conflict_density.get(tuple(v), 0.0) if conflict_density else 0.0
    if d > 5 and sdf > 2.0 * cfg.d_min and density < 0.18 and cd < 0.25:
        return cfg.r_max
    return 1


def _substeps(cfg, env, v, d, r, planner, constraints, t):
    """生成一个动作的子步体素序列（每子步恰占一个 Δt）。

    返回 (steps, final_t) 或 None（越界/约束违反）。
    """
    if d == ZERO:
        # 等待：SIPP 风格安全间隔跳跃，跳过连续被禁止的时刻
        if planner == "sipp" and constraints is not None:
            jump = 1
            while jump <= 4 and (t + jump, tuple(v)) in constraints.vertex:
                jump += 1
            if jump > 4 or t + jump >= cfg.t_max:
                return None
            return ([tuple(v)] * jump, t + jump)
        return ([tuple(v)], t + 1)

    unit = tuple(d)
    if r >= 2:
        mid = (v[0] + unit[0], v[1] + unit[1], v[2] + unit[2])
        end = (v[0] + 2 * unit[0], v[1] + 2 * unit[1], v[2] + 2 * unit[2])
        steps = [mid, end]
    else:
        steps = [(v[0] + unit[0], v[1] + unit[1], v[2] + unit[2])]
    return (steps, t + len(steps))


# ---------------------------------------------------------------------- A* 骨架
def astar_skeleton(cfg: Config, env, start, goal, constraints, planner,
                   deadline_ts, conflict_density, start_ts, htable=None):
    start = tuple(start)
    goal = tuple(goal)
    if constraints is None:
        constraints = Constraints()
    if deadline_ts is None:
        deadline_ts = float("inf")

    s0 = (start[0], start[1], start[2], 0)
    gbest = {s0: 0.0}
    parent = {}
    counter = 0
    # 自适应权重基线（REF_ASTAR）以起点的 h 为量程基准，ω 随 h/h_start 线性衰减
    h_start = max(1e-6, _h(cfg, start, goal, htable))
    heap = [(_h(cfg, start, goal, htable), 0.0, counter, s0)]
    expanded = 0

    while heap and expanded < cfg.max_low_expanded:
        if time.monotonic() > deadline_ts:
            return None, expanded, "budget"
        f, gval, _, state = heapq.heappop(heap)
        if gval > gbest.get(state, float("inf")) + 1e-9:
            continue
        expanded += 1
        v = (state[0], state[1], state[2])
        t = state[3]

        if v == goal:
            seq = []
            cur = state
            while cur in parent:
                prev, steps = parent[cur]
                seq.extend(reversed(steps))
                cur = prev
            seq.reverse()
            path = [start] + seq
            # 补齐到统一长度（到达后停留；评价只到到达时刻）
            full = list(path) + [goal] * max(0, cfg.t_max - len(path))
            full = full[:cfg.t_max]
            cost = path_cost(cfg, env, path)
            ttff = time.monotonic() - start_ts
            return LowLevelResult(grid_path=full, cost=cost, expanded=expanded,
                                  ttff=ttff, planner=planner), expanded, ""

        if t >= cfg.t_max - 1:
            continue

        # 多尺度只在"开阔且无约束"的区域加速：带高层约束重规划时，
        # 大步长要逐子步复核约束，失败率高、反而放大单次成本（αβ 失衡）。
        has_cons = bool(constraints.vertex or constraints.edge)
        r = 1 if (planner in ("fixed", "sipp", "refastar") or has_cons) else choose_radius(
            cfg, env, v, goal, conflict_density)
        for d in [ZERO] + UNIT_DIRS:
            built = _substeps(cfg, env, v, d, r, planner, constraints, t)
            if built is None:
                continue
            steps, t_end = built
            if t_end >= cfg.t_max:
                continue
            prev_v = v
            csum = 0.0
            ok = True
            for k, q in enumerate(steps):
                t_k = t + 1 + k
                if not env.in_bounds(q) or constraints.violates(prev_v, q, t_k):
                    ok = False
                    break
                c, power, smoke, _ = edge_metrics(cfg, env, prev_v, q, t_k - 1)
                if not hard_ok(cfg, env, q, t_k - 1, power, smoke):
                    ok = False
                    break
                csum += c
                prev_v = q
            if not ok:
                continue
            ns = (steps[-1][0], steps[-1][1], steps[-1][2], t_end)
            ng = gval + csum
            # 允许重开 CLOSED：边权非对称时 h 不满足一致性
            if ng < gbest.get(ns, float("inf")) - 1e-9:
                gbest[ns] = ng
                parent[ns] = (state, steps)
                counter += 1
                h = _h(cfg, steps[-1], goal, htable)
                if planner == "refastar":
                    # 对标基线 1：f = g + ω·h，ω 由 5.0 递减到 0.2
                    w = (cfg.ref_omega_min
                         + (cfg.ref_omega_max - cfg.ref_omega_min)
                         * min(1.0, h / h_start))
                else:
                    w = 1.0
                heapq.heappush(heap, (ng + w * h, ng, counter, ns))
    return None, expanded, "exhausted"


# ---------------------------------------------------------------------- 走廊 + PSO
def _waypoints(path):
    """提取骨架转折点，保留其时间索引：[(t, voxel), ...]。

    必须保留时间索引并做"按时间分段"插值，零偏移粒子才能精确还原骨架，
    否则"优化后风险不劣于骨架"的硬约束会对初始粒子也判违规，阶段 B 永不生效。
    """
    pts = [(0, path[0])]
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1], path[i], path[i + 1]
        if (b[0] - a[0], b[1] - a[1], b[2] - a[2]) != \
           (c[0] - b[0], c[1] - b[1], c[2] - b[2]):
            pts.append((i, b))
    pts.append((len(path) - 1, path[-1]))
    # 去重（连续转折点索引不能相同）
    clean = [pts[0]]
    for t, v in pts[1:]:
        if t > clean[-1][0]:
            clean.append((t, v))
    return clean


def _corridor_radii(cfg, env, wps):
    radii = []
    for _, v in wps:
        r = min(cfg.corridor_radius, max(0.0, env.sdf_at(v) - cfg.d_min))
        radii.append(r)
    return radii


def _interpolate(wps, offsets, cell_xy, cell_z, length):
    """按 waypoint 的真实时间索引分段插值，零偏移时精确还原骨架。"""
    out = []
    j = 0
    for t in range(length):
        while j < len(wps) - 2 and wps[j + 1][0] <= t:
            j += 1
        t0, a = wps[j]
        t1, b = wps[j + 1] if j + 1 < len(wps) else (max(t0 + 1, length - 1), a)
        frac = (t - t0) / max(1, t1 - t0)
        frac = max(0.0, min(1.0, frac))
        oa = offsets[j]
        ob = offsets[j + 1] if j + 1 < len(offsets) else oa
        out.append((
            (a[0] * (1 - frac) + b[0] * frac) * cell_xy + oa[0] * (1 - frac) + ob[0] * frac,
            (a[1] * (1 - frac) + b[1] * frac) * cell_xy + oa[1] * (1 - frac) + ob[1] * frac,
            (a[2] * (1 - frac) + b[2] * frac) * cell_z + oa[2] * (1 - frac) + ob[2] * frac,
        ))
    return out


def _voxel_of(p, cfg):
    return (int(round(p[0] / cfg.cell_xy)),
            int(round(p[1] / cfg.cell_xy)),
            int(round(p[2] / cfg.cell_z)))


def _fitness(cfg, env, wps, radii, offsets, skel_cost, length):
    traj = _interpolate(wps, offsets, cfg.cell_xy, cfg.cell_z, length)
    voxels = [_voxel_of(p, cfg) for p in traj]
    penalty = 0.0
    cost = 0.0
    # 走廊硬约束：粒子偏移必须落在以骨架 waypoint 为心的球内（球约束，非盒约束）
    for j, o in enumerate(offsets):
        mag = math.sqrt(sum(u * u for u in o))
        if mag > radii[j] + 1e-6:
            penalty += 1000.0 * (mag - radii[j])
    for t, (p, v) in enumerate(zip(traj, voxels)):
        if not env.is_free(v) or env.sdf_at(v) < cfg.d_min:
            penalty += 1000.0
        if env.smoke_at(v, t) > cfg.smoke_rate_max:
            penalty += 1000.0
        if t >= 1:
            c, power, smoke, _ = edge_metrics(cfg, env, voxels[t - 1], v, t - 1)
            cost += c
            if power > cfg.power_rate_max:
                penalty += 1000.0
    # 曲率惩罚
    curv = 0.0
    for t in range(1, len(traj) - 1):
        acc = (traj[t - 1][0] - 2 * traj[t][0] + traj[t + 1][0],
               traj[t - 1][1] - 2 * traj[t][1] + traj[t + 1][1],
               traj[t - 1][2] - 2 * traj[t][2] + traj[t + 1][2])
        curv += math.sqrt(sum(u * u for u in acc))
    # 灾害场景：在"代价不劣于骨架"的硬约束下，主动把轨迹推离障碍，
    # 把安全裕度从"碰巧更大"变成"被优化目标逼出来"的可复现收益。
    min_clear = min((env.continuous_sdf(p) for p in traj), default=0.0)
    # 硬约束：优化后风险不劣于骨架
    if cost > skel_cost * (1.0 + cfg.risk_not_worse_eps):
        penalty += 1000.0 * (cost - skel_cost)
    return (cost + 0.3 * curv - cfg.w_clear * min_clear + penalty,
            cost, penalty)


def corridor_pso(cfg: Config, env, skel, skel_cost, deadline_ts, seed,
                 length=None):
    """阶段 B：走廊内局部连续优化。返回 (cont_path, success, feasible_ratio)。

    length 可指定为"到达段长度"，避免对到达后停留的整段 t_max 做无意义评估。
    """
    length = length or len(skel.grid_path)
    wps = _waypoints(skel.grid_path[:length])
    radii = (_corridor_radii(cfg, env, wps) if cfg.corridor_enabled
             else [cfg.corridor_radius] * len(wps))
    rng = random.Random(seed)
    dim = len(wps)
    P = cfg.pso_particles

    def rand_particle():
        return [tuple(rng.uniform(-r * 0.35, r * 0.35) for _ in range(3))
                for r in radii]

    particles = [[(0.0, 0.0, 0.0)] * dim] + [rand_particle() for _ in range(P - 1)]
    velocities = [[(0.0, 0.0, 0.0)] * dim for _ in range(P)]
    pbest = [list(p) for p in particles]
    pbest_val = [float("inf")] * P
    gbest = [list(particles[0]), float("inf"), 0.0]
    feasible = 0
    tested = 0
    w, c1, c2 = 0.7, 1.4, 1.4

    for it in range(cfg.pso_iters):
        if deadline_ts is not None and time.monotonic() > deadline_ts:
            break
        for i, part in enumerate(particles):
            val, c, pen = _fitness(cfg, env, wps, radii, part, skel_cost, length)
            tested += 1
            if pen <= 1e-6:
                feasible += 1
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = [tuple(o) for o in part]
            if val < gbest[1]:
                gbest = [[tuple(o) for o in part], val, c]
        for i in range(P):
            newp, newv = [], []
            for j in range(dim):
                r1, r2 = rng.random(), rng.random()
                pv = pbest[i][j]
                gv = gbest[0][j]
                cur = particles[i][j]
                vel = velocities[i][j]
                nv = tuple(w * vel[k] + c1 * r1 * (pv[k] - cur[k])
                           + c2 * r2 * (gv[k] - cur[k]) for k in range(3))
                no = tuple(cur[k] + nv[k] for k in range(3))
                r = radii[j]
                mag = math.sqrt(sum(x * x for x in no))
                if mag > r and mag > 1e-9:
                    no = tuple(x * r / mag for x in no)   # 投影回球内
                no = tuple(max(-r, min(r, x)) for x in no)
                nv = tuple(max(-r, min(r, x)) for x in nv)
                newp.append(no)
                newv.append(nv)
            particles[i] = newp
            velocities[i] = newv

    ratio = feasible / max(1, tested)
    offs = gbest[0]
    val, c, pen = _fitness(cfg, env, wps, radii, offs, skel_cost, length)
    if pen > 1e-6 or c > skel_cost * (1.0 + cfg.risk_not_worse_eps):
        return None, False, ratio
    traj = _interpolate(wps, offs, cfg.cell_xy, cfg.cell_z, length)
    # 按统一 Δt 重采样后复核硬约束 H
    for t, p in enumerate(traj):
        v = _voxel_of(p, cfg)
        _, power, smoke, _ = edge_metrics(cfg, env, v, v, t)
        if not env.is_free(v) or env.sdf_at(v) < cfg.d_min \
                or smoke > cfg.smoke_rate_max:
            return None, False, ratio
    # 硬约束：优化后安全裕度不得劣于骨架。
    # 口径修正（关键）：骨架必须用**同一条参数化插值曲线**（零偏移粒子）来取基线，
    # 而不是体素中心折线。插值点位于两个体素中心之间，天然比中心更靠近障碍表面，
    # 旧版拿"体素中心的 min 连续 SDF"去比"插值点的 min 连续 SDF"，等于拿苹果比橘子，
    # 该判定恒不成立 → 阶段 B 认证通过数恒为 0 → 创新点1 的连续优化通道全程关闭。
    zero_offs = [(0.0, 0.0, 0.0)] * len(wps)
    base_traj = _interpolate(wps, zero_offs, cfg.cell_xy, cfg.cell_z, length)
    skel_clear = min((env.continuous_sdf(p) for p in base_traj), default=999.0)
    best_clear = min((env.continuous_sdf(p) for p in traj), default=0.0)
    if best_clear < skel_clear - 1e-6:
        return None, False, ratio
    return traj, True, ratio


def global_pso_ratio(cfg: Config, env, skel, seed):
    """L2 对照：无走廊的全局随机 PSO，统计粒子可行比例（验证最初动机）。"""
    rng = random.Random(seed)
    ok = 0
    trials = 120
    for _ in range(trials):
        p = (rng.randint(0, cfg.nx - 1), rng.randint(0, cfg.ny - 1),
             rng.randint(0, cfg.nz - 1))
        if env.is_free(p) and env.sdf_at(p) >= cfg.d_min \
                and env.smoke_at(p, 0) <= cfg.smoke_rate_max:
            ok += 1
    return ok / trials


_DOSE_LB = {}


def dose_lb_table(cfg: Config, env, goal):
    """每个体素到目标的**最小可能累积暴露**下界（反向 Dijkstra）。

    为什么需要它：剂量受限搜索里，费用启发 h 只说"还差多少代价"，对"还差多少
    暴露"一无所知，于是搜索会大量扩展注定超预算的标签，预算耗尽而误判无解。
    这里对每个格取**全时段暴露的最小值**（min_t smoke(v,t)）作为静态边权，
    反向 Dijkstra 得到 h_dose(v)；任何真实路径的剩余暴露都 ≥ h_dose(v)，
    因此 `已用剂量 + h_dose(v) > 预算` 即可安全剪枝（可采纳、不丢解）。
    """
    # key 必须包含 seed 与 t_max：floor[v]=min_t smoke_at(v,t) 同时依赖烟源（由 seed 决定）
    # 与时域长度。exact_table 早已这样写，本表原先漏了，会造成跨场景缓存污染。
    key = (getattr(env, "fingerprint", None), env.seed, cfg.nx, cfg.ny,
           cfg.nz, cfg.dt, cfg.t_max, goal)
    if key in _DOSE_LB:
        return _DOSE_LB[key]
    # 每格的"全时段最小暴露速率"
    floor = {}
    for x in range(cfg.nx):
        for y in range(cfg.ny):
            for z in range(cfg.nz):
                v = (x, y, z)
                if not env.is_free(v):
                    floor[v] = float("inf")
                    continue
                floor[v] = min(env.smoke_at(v, t) for t in range(cfg.t_max))
    dist = {tuple(goal): 0.0}
    heap = [(0.0, tuple(goal))]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist.get(v, float("inf")) + 1e-12:
            continue
        for dd in UNIT_DIRS:
            u = (v[0] + dd[0], v[1] + dd[1], v[2] + dd[2])
            if not env.in_bounds(u):
                continue
            w = floor.get(u, float("inf"))
            if w == float("inf"):
                continue
            nd = d + w * cfg.dt
            if nd < dist.get(u, float("inf")) - 1e-12:
                dist[u] = nd
                heapq.heappush(heap, (nd, u))
    _DOSE_LB[key] = dist
    return dist


def astar_dose(cfg: Config, env, start, goal, constraints, deadline_ts,
               start_ts, htable=None):
    """剂量受限的时空 A*：把累积烟雾暴露作为硬约束。

    为什么必须单开一个搜索：标准时空 A* 的状态是 (v, t)，同一 (v, t) 只保留
    最小代价即可——这依赖"最优子结构"。一旦加入累积剂量约束 D ≤ D_max，
    同一 (v, t) 上"代价低但剂量高"的标签可能通向死路，而"代价高但剂量低"
    的标签才可行，**子结构被破坏**。SIPP 的安全区间也随之失效（不是时间区间
    不可用，而是可行域还取决于剂量余量）。

    处理方式是资源受限最短路的标准做法——**标签设置 + 帕累托支配**：
        同 (v, t) 下，标签 (g1, k1) 支配 (g2, k2) ⟺ g1 ≤ g2 且 k1 ≤ k2
    剂量按 cfg.dose_buckets 离散为桶，桶数决定状态规模与近似的保守性。
    """
    start, goal = tuple(start), tuple(goal)
    if constraints is None:
        constraints = Constraints()
    if deadline_ts is None:
        deadline_ts = float("inf")
    budget = max(1e-9, cfg.dose_budget)
    K = max(2, cfg.dose_buckets)
    EXACT = bool(getattr(cfg, "exact_dominance", False))
    lb = (dose_lb_table(cfg, env, goal)
          if getattr(cfg, "dose_prune", True) else {})

    def bucket(d):
        return min(K - 1, int(d * (K - 1) / budget + 1e-12))

    s0 = (start[0], start[1], start[2], 0, 0)
    gbest = {s0: 0.0}
    parent = {}
    # (v,t) -> 该格该时刻已保留的 (k, g) 标签，用于支配判定
    labels = {(start[0], start[1], start[2], 0): [(0, 0.0)]}
    # 每个状态上存**与该状态父节点成对**的精确累积剂量。
    # 关键：剂量必须与 parent 一起写，否则桶塌缩后会出现"记的是低剂量、
    # 回溯却走了高剂量父节点"的不一致（实测会导致返回超预算的解）。
    # 起点在 t=0 也占一拍、也暴露，必须计入初始剂量，否则与 external_risk
    # 的口径（t 从 0 起算）差一拍，边界上会放过超预算的解。
    dose_of = {s0: env.smoke_at(start, 0) * cfg.dt}
    counter = 0
    heap = [(_h(cfg, start, goal, htable), 0.0, counter, s0)]
    expanded = 0

    while heap and expanded < cfg.max_low_expanded:
        if time.monotonic() > deadline_ts:
            return None, expanded, "budget"
        f, gval, _, state = heapq.heappop(heap)
        if gval > gbest.get(state, float("inf")) + 1e-9:
            continue
        expanded += 1
        v = (state[0], state[1], state[2])
        t, k = state[3], state[4]

        if v == goal:
            seq, cur = [], state
            while cur in parent:
                prev, steps = parent[cur]
                seq.extend(reversed(steps))
                cur = prev
            seq.reverse()
            path = [start] + seq
            full = list(path) + [goal] * max(0, cfg.t_max - len(path))
            full = full[:cfg.t_max]
            cost = path_cost(cfg, env, path)
            return LowLevelResult(grid_path=full, cost=cost, expanded=expanded,
                                  ttff=time.monotonic() - start_ts,
                                  planner="dose"), expanded, ""
        if t >= cfg.t_max - 1:
            continue

        for d in [ZERO] + UNIT_DIRS:
            built = _substeps(cfg, env, v, d, 1, "fixed", constraints, t)
            if built is None:
                continue
            steps, t_end = built
            if t_end >= cfg.t_max:
                continue
            prev_v, csum, dsum, ok = v, 0.0, 0.0, True
            for i, q in enumerate(steps):
                t_k = t + 1 + i
                if not env.in_bounds(q) or constraints.violates(prev_v, q, t_k):
                    ok = False
                    break
                c, power, smoke, _ = edge_metrics(cfg, env, prev_v, q, t_k - 1)
                if not hard_ok(cfg, env, q, t_k - 1, power, smoke):
                    ok = False
                    break
                csum += c
                # 剂量按"到达该格的时刻"计：dose += smoke(q, t_k)·dt，
                # 与 external_risk 的口径（smoke_at(path[t], t)）逐拍对齐；
                # 若沿用代价里的 t-1 会错一拍，边界附近会放过超预算的解。
                dsum += env.smoke_at(q, t_k) * cfg.dt
                prev_v = q
            if not ok:
                continue
            # 剂量硬约束：精确累计；超预算的分支直接剪掉
            nd = dose_of.get(state, 0.0) + dsum
            if nd > budget + 1e-9:
                continue
            # 暴露下界剪枝：剩余必经暴露已超预算 → 该标签注定无解，立即丢弃
            if nd + lb.get(tuple(steps[-1]), 0.0) > budget + 1e-9:
                continue
            # 精确支配实验：状态键仍用高精度离散（保证每个标签有独立父指针），
            # 但支配判定改用精确剂量 nd，从而定理 2 的前提（exact-dose）成立。
            nk = (round(nd, 6) if EXACT else bucket(nd))
            vt = (steps[-1][0], steps[-1][1], steps[-1][2], t_end)
            ng = gval + csum
            # 帕累托支配：同 (v,t) 下若有 (k'≤k, g'≤g) 则丢弃
            lst = labels.setdefault(vt, [])
            ck = (nd if EXACT else nk)
            if any(kk <= ck + 1e-9 and gg <= ng + 1e-9 for kk, gg in lst):
                continue
            labels[vt] = [(kk, gg) for kk, gg in lst
                          if not (ck <= kk + 1e-9 and ng <= gg + 1e-9)]
            labels[vt].append((ck, ng))
            ns = (steps[-1][0], steps[-1][1], steps[-1][2], t_end, nk)
            if ng < gbest.get(ns, float("inf")) - 1e-9:
                gbest[ns] = ng
                dose_of[ns] = nd
                parent[ns] = (state, steps)
                counter += 1
                heapq.heappush(heap, (ng + _h(cfg, steps[-1], goal, htable),
                                      ng, counter, ns))
    return None, expanded, "exhausted"


def warmup_heuristics(cfg: Config, env, goals):
    """在建表阶段一次性预计算所有目标的精确 h* 表，再开始计时。

    为什么必须预热：exact 启发表是从目标反向跑一遍 Dijkstra，单表开销可达
    0.05~0.5 s，远大于一次低层搜索本身。若不预热，这笔一次性开销会全部记在
    **本场景第一个被跑到的方法**头上（实验里恰好是基线 A_BASE），造成"基线更慢"
    的计时伪影——这是口径不公平，不是算法差异。预热后各方法的 planning_time
    只反映搜索本身。
    """
    if getattr(cfg, "heuristic_mode", "") != "exact":
        return 0.0
    t0 = time.monotonic()
    for _, g in goals:
        exact_table(cfg, env, g)
    return time.monotonic() - t0


# ---------------------------------------------------------------------- 顶层入口
def plan(cfg: Config, env, start, goal, constraints, planner, optimize,
         deadline_ts, conflict_density, seed=0, budget_signal=None,
         speed_profile=False, dose_aware=False):
    """低层统一入口。返回 LowLevelResult 或 None（预算内未找到可行骨架）。

    预算感知协同：高层下传 BudgetSignal。若剩余预算比例低于 tight_budget_ratio，
    本次调用自动关闭多尺度与阶段 B（β→1），退化为基线低层以保证时限内出解；
    预算充裕时才启用完整低层换取更好的安全裕度。这就是 αβ 的动态调节。
    """
    start_ts = time.monotonic()
    tight = (budget_signal is not None
             and budget_signal.remaining_ratio < cfg.tight_budget_ratio)
    eff_planner = "fixed" if tight else planner
    eff_optimize = False if tight else optimize

    density = dict(conflict_density or {})
    if budget_signal is not None and budget_signal.conflict_priority > 0:
        boost = 0.3 * budget_signal.conflict_priority
        density = {k: min(1.0, v + boost) for k, v in density.items()}

    htable = (exact_table(cfg, env, goal)
              if cfg.heuristic_mode == "exact" else None)
    if dose_aware:
        # 剂量受限搜索：不做多尺度、不做阶段 B（保持与基线同构，只差约束）
        res, expanded, reason = astar_dose(
            cfg, env, start, goal, constraints, deadline_ts, start_ts, htable)
    else:
        res, expanded, reason = astar_skeleton(
            cfg, env, start, goal, constraints, eff_planner,
            deadline_ts, density, start_ts, htable)
    if res is None:
        return None

    arr = next((i for i, v in enumerate(res.grid_path)
                if tuple(v) == tuple(goal)), len(res.grid_path) - 1)

    # A 方案：时间维速度剖面替代空间 PSO。几何不变，只重分配时间占用；
    # 展开序列已通过 H 与高层约束复核，高层仍会重新做冲突检测。
    #
    # 尝试过的"协同"改动（已回退，勿重犯）：预算紧张时连速度剖面一起关掉。
    # 动机是让 β 真正随预算下降。实测结果相反：tight 的判定用的是固定比例 0.55，
    # 而单次调用就要 0.2 s 左右，于是搜索的大部分时间都跑在"降级态"上，
    # 创新点 1 等于被关掉——真实 3D BAG 底图上 D_FULL_A 解出率从 0.571 掉到 0.429。
    # 结论：以固定剩余比例做降级太粗，收益也不成立，已回退。
    # 剂量模式下必须禁止后处理：速度剖面/阶段 B 会改写时间占用，
    # 从而改变累积暴露，却不重新过一次预算校验——soundness 会被静默破坏。
    if speed_profile and not dose_aware:
        occ, info = speed_optimize(cfg, env, res.grid_path, goal,
                                   constraints, deadline_ts)
        if occ is not None:
            res.grid_path = occ
            arr = next((i for i, v in enumerate(occ)
                        if tuple(v) == tuple(goal)), len(occ) - 1)
            res.cost = path_cost(cfg, env, occ[:arr + 1])
            res.speed_info = info
    # 低层风险/绕行代价摘要上送高层，用于影响评估（双向信息流的下→上一支）
    risk = external_risk(cfg, env, res.grid_path[:arr + 1])
    risk["detour"] = res.cost / max(1e-6, heuristic(cfg, start, goal))
    res.risk_summary = risk
    res.planner = eff_planner
    if tight:
        res.fallback_reason = "tight_budget_degraded"

    if eff_optimize and not dose_aware:
        # 阶段 B 只使用剩余预算的一部分；预算耗尽即返回骨架（anytime）
        if deadline_ts is None:
            opt_deadline = None
        else:
            remaining = max(0.0, deadline_ts - time.monotonic())
            opt_deadline = time.monotonic() + remaining * cfg.optimize_budget_frac
        cont, success, ratio = corridor_pso(
            cfg, env, res, res.cost, opt_deadline, seed, length=arr + 1)
        res.cont_path = cont
        res.opt_success = success
        res.opt_feasible_ratio = ratio
        res.fallback_reason = "" if success else (
            "budget" if (opt_deadline and time.monotonic() > opt_deadline)
            else "optimization_infeasible")
    return res
