"""时间维速度剖面：把创新点 1 的优化对象从"空间轨迹"搬到"何时到达何处"。

依据 600m_MAPF_project_20261103/BOTTLENECK_GUIDE.md:474-492 的诊断：
  - A* 在离散图上 Σstep 最优 → PSO 在代价维度数学上不可能赢；
  - 可飞性/曲率由 B 样条一次性平滑解决；
  - 唯一真空白是时间维 / 速度剖面：v(s) <= sqrt(g·tan(phi)·R(s))。

本模块在**不改变几何路径**的前提下重新分配每个体素的时间占用：
急弯处减速（dwell 增大）、直线段加速（dwell=1），从而
  ① 得到物理可执行的到达时刻与风险暴露（连续量，不受离散 SDF 钉死）；
  ② 给高层"局部修复"提供"整体后移 / 减速"这一真实自由度（不只是插一步等待）。

零第三方依赖。
"""
import math
import time

from cost import edge_metrics, hard_ok

G = 9.81


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def menger_curvature(a, b, c):
    """三点外接圆曲率 κ = 4·Area/(a·b·c)（输入为米制坐标）。"""
    ab, bc, ca = _sub(b, a), _sub(c, b), _sub(a, c)
    la, lb, lc = _norm(ab), _norm(bc), _norm(ca)
    denom = la * lb * lc
    if denom < 1e-9:
        return 0.0
    return 4.0 * (0.5 * _norm(_cross(ab, bc))) / denom


def _to_meters(cfg, v):
    return (v[0] * cfg.cell_xy, v[1] * cfg.cell_xy, v[2] * cfg.cell_z)


def geometry_of(grid_path, goal):
    """去掉连续停留，得到几何骨架；返回到达 goal 之前的几何点列。"""
    geom = []
    arr = next((i for i, v in enumerate(grid_path)
                if tuple(v) == tuple(goal)), len(grid_path) - 1)
    for v in grid_path[:arr + 1]:
        if not geom or tuple(v) != tuple(geom[-1]):
            geom.append(tuple(v))
    return geom, arr


def curvature_dwell(cfg, geom):
    """按曲率限速给出每个几何点的最小停留时间步。"""
    pts = [_to_meters(cfg, v) for v in geom]
    n = len(geom)
    dwell = [1] * n
    tan_phi = math.tan(math.radians(cfg.bank_deg))
    r_req = cfg.cruise_mps ** 2 / (G * tan_phi)
    for i in range(1, n - 1):
        kappa = menger_curvature(pts[i - 1], pts[i], pts[i + 1])
        if kappa > 1e-9:
            r = 1.0 / kappa
            v_allow = min(cfg.cruise_mps,
                          math.sqrt(G * tan_phi * min(r, 1e6)))
        else:
            v_allow = cfg.cruise_mps
        seg_len = _norm(_sub(pts[i], pts[i - 1]))
        need = max(1, math.ceil(seg_len / max(0.5, v_allow * cfg.dt)))
        # 曲率超标（R < R_req）时强制减速一拍
        if kappa > 1e-9 and (1.0 / kappa) < r_req:
            need = max(2, need)
        dwell[i] = min(4, need)
    return dwell


def build_occupancy(cfg, geom, dwell, goal, t_max):
    """把 (几何点, 停留步数) 展开成统一 Δt 的离散占用序列。"""
    out = []
    for v, d in zip(geom, dwell):
        out.extend([tuple(v)] * max(1, int(d)))
    out = out[:t_max]
    if len(out) < t_max:
        out.extend([tuple(goal)] * (t_max - len(out)))
    return out


def profile_ok(cfg, env, occ, constraints):
    """展开后的占用序列必须满足硬约束 H 与高层下发的顶点/边约束。"""
    prev = None
    for t, v in enumerate(occ):
        if not env.is_free(v) or env.sdf_at(v) < cfg.d_min:
            return False
        if env.smoke_at(v, t) > cfg.smoke_rate_max:
            return False
        if constraints is not None:
            if (t, tuple(v)) in constraints.vertex:
                return False
            if prev is not None and (t - 1, tuple(prev), tuple(v)) in constraints.edge:
                return False
        prev = v
    return True


def profile_score(cfg, env, occ, goal):
    """目标：到达时刻 + 累计烟雾暴露（时间维真正可优化的两个量）。"""
    arr = next((i for i, v in enumerate(occ)
                if tuple(v) == tuple(goal)), len(occ) - 1)
    exposure = sum(env.smoke_at(v, t) * cfg.dt
                   for t, v in enumerate(occ[:arr + 1]))
    return arr + cfg.w_smoke * exposure, arr, exposure


def optimize(cfg, env, grid_path, goal, constraints, deadline_ts):
    """速度剖面局部优化：在不改变几何的前提下压缩到达时刻 / 暴露。

    失败、预算耗尽或约束不满足一律返回 None（anytime 回退骨架）。
    返回 (occupancy, info_dict) 或 (None, info_dict)。
    """
    info = {"attempted": True, "certified": False,
            "arrival_before": 0, "arrival_after": 0}
    geom, arr0 = geometry_of(grid_path, goal)
    info["arrival_before"] = arr0
    if len(geom) < 3:
        return None, info
    dwell = curvature_dwell(cfg, geom)
    occ = build_occupancy(cfg, geom, dwell, goal, cfg.t_max)
    if not profile_ok(cfg, env, occ, constraints):
        return None, info
    base, arr, _ = profile_score(cfg, env, occ, goal)

    # 局部搜索：尝试给每段"加速"（dwell-1），只要不违反限速/约束/目标
    for _ in range(3):
        if deadline_ts is not None and time.monotonic() > deadline_ts:
            break
        improved = False
        for i in range(1, len(dwell) - 1):
            if dwell[i] <= 1:
                continue
            trial = list(dwell)
            trial[i] -= 1
            cand = build_occupancy(cfg, geom, trial, goal, cfg.t_max)
            if not profile_ok(cfg, env, cand, constraints):
                continue
            score, _, _ = profile_score(cfg, env, cand, goal)
            min_dwell = curvature_dwell(cfg, geom)
            if trial[i] < min_dwell[i]:
                continue
            if score < base - 1e-9:
                dwell, occ, base = trial, cand, score
                improved = True
        if not improved:
            break
    info["certified"] = True
    info["arrival_after"] = profile_score(cfg, env, occ, goal)[1]
    info["dwell_total"] = sum(dwell)
    return occ, info


def shift_from(path, t0, delta, t_max):
    """从 t0 起把后续占用整体后移 delta 拍（速度剖面的"整体减速"自由度）。

    与单点 insert_wait 不同：它保持几何序列与相对时序，只改变到达时刻，
    是高层"局部修复"消解同刻同点冲突时更自然的手段。
    """
    if t0 < 1 or delta < 1 or t0 + delta >= t_max:
        return None
    return path[:t0] + [path[t0 - 1]] * delta + path[t0:t_max - delta]


def smooth_geometric(path, cell_xy, cell_z, rounds=3):
    """纯几何平滑（多次 Chaikin），**不改变时间映射与占用序列**。

    作为"事后 B 样条平滑"的对照实现：只把折线拉圆，不参与冲突消解，
    因此解出率必然与基线完全一致——这正是要证明的点。

    ⚠️ 采样密度必须与基线口径一致：指标里的 `_chaikin` 会把 n 个点变成 2n 个点，
    若这里把平滑结果重采样回 n 点，点更稀会让 Menger 曲率估计系统性偏大，
    造成"平滑后反而更差"的假象（实测 0.1201 vs 0.1034）。故统一输出 2n 点。
    """
    n = len(path)
    pts = [(v[0] * cell_xy, v[1] * cell_xy, v[2] * cell_z) for v in path]
    if n < 3:
        return pts
    for _ in range(max(1, rounds)):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(tuple(0.75 * a[k] + 0.25 * b[k] for k in range(3)))
            out.append(tuple(0.25 * a[k] + 0.75 * b[k] for k in range(3)))
        out.append(pts[-1])
        pts = out
    # 重采样到 2n（与 _chaikin 的输出点数一致），保证曲率估计同口径
    target = 2 * n
    m = len(pts)
    res = []
    for i in range(target):
        x = i * (m - 1) / max(1, target - 1)
        j = min(m - 2, int(x))
        f = x - j
        res.append(tuple(pts[j][k] * (1 - f) + pts[j + 1][k] * f
                         for k in range(3)))
    return res
