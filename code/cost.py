"""统一代价 C 与硬约束 H。

公平对比的关键：低层 g/h、高层节点排序、sum of costs、以及所有外部基线的评分
一律使用本模块定义的同一个 C；H 不满足即解无效，绝不作为可交易的软权重。

可采纳性说明：
- 每个动作（含等待）的代价都 >= c_min，且软项（SDF/烟雾/逆风/垂直）非负；
- 一个时间步最多推进 r_max 格，因此 h = c_min * ceil(cheb(v,goal)/r_max) 低估真实代价，
  是对 C 可采纳的启发函数；
- 逆风代价使边权非对称 => h 一般不满足一致性，低层 A* 必须允许重开 CLOSED 节点。
"""
import math

from config import Config, chebyshev


def c_min(cfg: Config):
    """所有单边代价的保守下界（等待动作也带基准占位代价）。"""
    return cfg.w_dist * cfg.cell_xy * 0.5


def edge_metrics(cfg: Config, env, p, q, t):
    """返回 (cost, power_rate, smoke_rate, distance) —— 全部按统一 Δt 的一个子步计。"""
    dx = (q[0] - p[0]) * cfg.cell_xy
    dy = (q[1] - p[1]) * cfg.cell_xy
    dz = (q[2] - p[2]) * cfg.cell_z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    # 等待动作也计入基准占位代价，保证 c_min 可采纳
    base = cfg.w_dist * dist if dist > 1e-9 else c_min(cfg)

    sdf_q = env.sdf_at(q)
    soft_sdf = cfg.w_sdf * math.exp(
        -max(0.0, sdf_q - cfg.d_min) / cfg.sigma_sdf
    )

    smoke = env.smoke_at(q, t)
    wx, wy, wz = env.wind_at(p, t)
    if dist > 1e-9:
        head = -(wx * dx + wy * dy + wz * dz) / dist   # 逆风为正
    else:
        head = 0.0
    wind_pen = cfg.w_wind * max(0.0, head)
    vertical = cfg.w_z * abs(q[2] - p[2])

    # 能耗/机动代理量：距离 + 逆风惩罚 + 垂直机动
    power = dist + wind_pen + vertical
    c = base + soft_sdf + cfg.w_smoke * smoke * cfg.dt + wind_pen + vertical
    return c, power, smoke, dist


def hard_ok(cfg: Config, env, q, t, power, smoke):
    """硬约束 H。注意：累计暴露/能耗通过"逐步速率上限 + 有限 t_max"自动有界。"""
    if not env.is_free(q):
        return False
    if env.sdf_at(q) < cfg.d_min:
        return False
    if smoke > cfg.smoke_rate_max:
        return False
    if power > cfg.power_rate_max:
        return False
    # 动态障碍预测：若已预测到 t >= dynamic_change_at 时该体素会坍塌，
    # 当前规划即提前避开（消融实验中关闭此项以度量该模块贡献）。
    if getattr(cfg, "predict_dynamic", False) and t >= cfg.dynamic_change_at:
        future = getattr(env, "future_no_fly", ())
        if tuple(q) in future:
            return False
    return True


def heuristic(cfg: Config, v, goal):
    """对统一代价 C 可采纳且一致的欧氏距离下界。

    每条边的 base = w_dist * 该步三维距离，软项（SDF/烟雾/逆风/垂直）非负，
    因此任意路径代价 >= w_dist * 路径长度 >= w_dist * 起终直线距离。
    欧氏距离满足三角不等式，故 |h(p)-h(q)| <= w_dist * dist(p,q) <= c(p,q)，
    即使逆风使边权非对称，h 仍是一致的（重开 CLOSED 的逻辑保留作为兜底）。

    注意：不要用 c_min * ceil(cheb/r_max) 这种过松下界——它会让 A* 退化成
    Dijkstra 式膨胀，远距离任务扩展上万节点，直接冲爆时间预算。
    """
    dx = (goal[0] - v[0]) * cfg.cell_xy
    dy = (goal[1] - v[1]) * cfg.cell_xy
    dz = (goal[2] - v[2]) * cfg.cell_z
    return cfg.w_dist * math.sqrt(dx * dx + dy * dy + dz * dz)


def path_cost(cfg: Config, env, path):
    """按统一 C 重算一条离散路径的代价（用于节点代价更新与修复代价增量判定）。"""
    total = 0.0
    for t in range(1, len(path)):
        c, _, _, _ = edge_metrics(cfg, env, path[t - 1], path[t], t - 1)
        total += c
    return total


def _menger(a, b, c):
    """三点外接圆曲率 κ=4·Area/(a·b·c)，输入米制坐标。"""
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    ca = (a[0] - c[0], a[1] - c[1], a[2] - c[2])
    cr = (ab[1] * bc[2] - ab[2] * bc[1],
          ab[2] * bc[0] - ab[0] * bc[2],
          ab[0] * bc[1] - ab[1] * bc[0])
    n = lambda v: math.sqrt(sum(x * x for x in v))
    denom = n(ab) * n(bc) * n(ca)
    if denom < 1e-9:
        return 0.0
    return 4.0 * (0.5 * n(cr)) / denom


def _chaikin(pts):
    """一次性角切割平滑（0 迭代，几毫秒）：量曲率前统一口径。

    600m 项目 BOTTLENECK_GUIDE.md:444-467 指出：折线转角存在网格几何天花板，
    必须先在平滑后的连续曲线上量 Menger 曲率，并去掉首尾 5% 端点伪影。
    """
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        out.append(tuple(0.75 * a[k] + 0.25 * b[k] for k in range(3)))
        out.append(tuple(0.25 * a[k] + 0.75 * b[k] for k in range(3)))
    out.append(pts[-1])
    return out


def external_risk(cfg: Config, env, path, cont_path=None, pre_smoothed=False):
    """外部独立风险指标：不复用加权代价 C。

    离障裕度一律用"连续轨迹到障碍表面"的距离，并以 p05 分位数为主
    （min 会被单个离散值钉死）；无连续轨迹的基线用体素中心折线等价构造。
    曲率在所有方法上先做一次性平滑再量，避免网格几何天花板造成测法伪影。
    """
    if cont_path is None:
        cont_path = [(v[0] * cfg.cell_xy, v[1] * cfg.cell_xy,
                      v[2] * cfg.cell_z) for v in path]
    clears = [env.continuous_sdf(p) for p in cont_path]
    min_clear = min(clears) if clears else 999.0
    if clears:
        sc = sorted(clears)
        p05_clear = sc[int(0.05 * (len(sc) - 1))]
        avg_clear = sum(sc) / len(sc)
        below_ratio = sum(1 for x in sc if x < cfg.clearance_required) / len(sc)
    else:
        p05_clear, avg_clear, below_ratio = 999.0, 999.0, 0.0

    # pre_smoothed=True 时轨迹已由方案自身平滑过（如"事后平滑"对照），
    # 不能再套一次 Chaikin，否则等于给对照组多平滑一轮，口径不公平。
    sm = cont_path if pre_smoothed else _chaikin(cont_path)
    n = len(sm)
    lo, hi = max(1, int(0.05 * n)), min(n - 1, max(2, int(0.95 * n)))
    ks = [_menger(sm[i - 1], sm[i], sm[i + 1]) for i in range(lo, hi)]
    r_req = cfg.cruise_mps ** 2 / (9.81 * math.tan(math.radians(cfg.bank_deg)))
    curv_viol = (sum(1 for k in ks if k > 1.0 / r_req) / max(1, len(ks))
                 if ks else 0.0)

    smoke_dose = 0.0
    energy = 0.0
    curv_max = 0.0
    for t, v in enumerate(path):
        smoke_dose += env.smoke_at(v, t) * cfg.dt
        if t >= 1:
            _, p, _, d = edge_metrics(cfg, env, path[t - 1], v, t - 1)
            energy += p
        if 1 <= t < len(path) - 1:
            a, b, c = path[t - 1], path[t], path[t + 1]
            acc = ((a[0] - 2 * b[0] + c[0]) * cfg.cell_xy,
                   (a[1] - 2 * b[1] + c[1]) * cfg.cell_xy,
                   (a[2] - 2 * b[2] + c[2]) * cfg.cell_z)
            curv_max = max(curv_max, math.sqrt(sum(u * u for u in acc)))
    return {
        "min_clearance": round(min_clear, 3),
        "p05_clearance": round(p05_clear, 3),
        "avg_clearance": round(avg_clear, 3),
        "below_safe_ratio": round(below_ratio, 4),
        "curv_viol_ratio": round(curv_viol, 4),
        "smoke_dose": round(smoke_dose, 4),
        "energy_proxy": round(energy, 3),
        "curvature_max": round(curv_max, 3),
    }
