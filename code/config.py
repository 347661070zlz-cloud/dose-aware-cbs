"""统一配置。

所有基线共享同一栅格、同一时间步、同一代价函数 C、同一硬约束 H 与同一时间预算
T_budget。修改这里即可改变整个对比实验的口径，避免"基线用纯距离、本文用风险代价"
这类不公平比较。

零第三方依赖，仅使用 Python 标准库。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- 三维体素栅格与统一时间栅格 ----
    nx: int = 20
    ny: int = 20
    nz: int = 4
    cell_xy: float = 5.0          # m，水平分辨率（细化到 5 m，使近失距离有连续区间）
    cell_z: float = 2.5           # m，垂直分辨率
    dt: float = 1.0               # s，固定时间步；所有约束/冲突检测/重采样都在这套栅格上
    t_max: int = 48               # 最大离散时间步
    t_budget: float = 3.0         # s，规划时间预算（首要目标：预算内给出安全可行解）
    tau_env: float = 30.0         # s，环境变化周期；要求 T_budget < tau_env
    deadline: float = 60.0        # s，任务截止时刻

    # ---- 统一代价 C 的权重（高层排序、sum of costs、低层 g/h 全部使用同一个 C）----
    w_dist: float = 1.0
    w_sdf: float = 0.8
    w_smoke: float = 1.2
    w_wind: float = 0.6
    w_z: float = 0.5
    sigma_sdf: float = 6.0

    # ---- 硬约束 H：不满足即解无效，不参与加权交易 ----
    # m，规划期最小离障（硬约束，逐体素检查）。
    d_min: float = 6.0
    smoke_rate_max: float = 0.70  # 单步烟雾暴露速率上限（逐步约束 => 全局累计有界）
    power_rate_max: float = 12.0  # 单步能耗/机动上限（1 格水平=5，2 格对角约 10.3）

    # ---- 创新点 1：低层 ----
    r_max: int = 2                # 空间多尺度最大半径；Δt 固定，大步长拆成多个时间子步
    corridor_radius: float = 9.0  # 安全走廊半径上限
    # 对标基线 1（邵新峰等 2025 改进 A*）的自适应权重量程：ω∈[0.2, 5]，按 h/h_start 线性插值
    ref_omega_min: float = 0.2
    ref_omega_max: float = 5.0
    pso_particles: int = 10
    pso_iters: int = 15
    optimize_budget_frac: float = 0.35   # 阶段 B 最多占用剩余预算的比例
    risk_not_worse_eps: float = 0.02     # 优化后风险不得劣于骨架的容差
    w_clear: float = 0.8                 # 阶段 B 的安全裕度奖励：主动把轨迹推离障碍

    # ---- 创新点 2：高层 ----
    omega: float = 1.2            # 有界次优界：C_found <= omega * C*
    delta: float = 0.15           # 修复代价增量阈值（相对当前节点总代价）
    impact_threshold: float = 0.50
    max_repairs_per_node: int = 6   # 单个约束树节点内最多批量修复的冲突数
    max_ct_nodes: int = 1200
    max_low_expanded: int = 20000

    # ---- 高低层协同：预算感知调度（真正的"协同"，而非两模块拼接）----
    tight_budget_ratio: float = 0.55  # 剩余预算比例低于此值 → 关闭低层昂贵机制(β)
    adaptive_lowlevel: bool = True    # False = 非协同拼接对照(D_FULL_NAIVE)

    # ---- 灾害场景交付验收阈值（比规划期 H 更严；不满足即"不合规解"）----
    # 口径说明（重要）：
    #   旧值 3.5 m 低于全部可行解的 p05 离障下界（离散体素 SDF 的次小值 3.536 m），
    #   导致 compliant_rate 恒等于 solved_rate —— 12 个方法、240 个案例无一例外，
    #   该列不含任何独立信息，属"事后挑阈值"。故改为 4.0 m（越过 3.536 这一档），
    #   并强制在 summary.csv 中同时输出 3.5/4.0/4.5/5.0 四档安全率，
    #   让阈值依赖显式可见，而不是只报一个好看的数。
    clearance_required: float = 4.0   # m，p05 连续离障裕度下限（分位数，避免被单点钉死）
    clearance_thresholds = (3.5, 4.0, 4.5, 5.0)   # 敏感性分析用的阈值网格（只报告，不替换主判据）
    smoke_dose_cap: float = 1e9       # 累计烟雾剂量上限，默认不启用
    # ---- 累积暴露硬约束（新增机制的开关）----
    # dose_budget 为 None/1e9 时关闭；启用后低层切换到 astar_dose（标签设置 + 支配剪枝）,
    # 状态从 (v,t) 扩为 (v,t,dose)，最优子结构失效。
    dose_budget: float = 2.88
    dose_buckets: int = 16            # 剂量离散桶数（越大越精确、状态越多）
    exact_dominance: bool = False     # True 时用精确剂量做帕累托支配（实验用，极慢）            # 剂量离散桶数（越大越精确、状态越多）
    dose_prune: bool = True           # 是否启用暴露下界剪枝（消融开关）

    # ---- 连续轨迹可飞性 / 时间维（借鉴 600m 项目的 measure_spline_flyability）----
    cruise_mps: float = 10.0          # 巡航速度
    bank_deg: float = 30.0            # 最大坡度 → R_req = v²/(g·tanφ) ≈ 17.66 m
    curv_viol_cap: float = 1.0        # 曲率超标率仅作连续指标报告，不进二值硬判定
                                      # （缺乏物理阈值依据，设阈值会变成"照结果挑数"）
    heuristic_mode: str = "exact"     # exact = 反向 Dijkstra 精确 h*；euclidean = 旧欧氏下界
    # ---- 7 模块消融开关（开题表验证方法 5）----
    predict_dynamic: bool = True      # 动态障碍预测：t>=dynamic_change_at 避开预测坍塌区
    corridor_enabled: bool = True     # 阶段 B 安全走廊半径约束（False = 全局自由优化）

    # ---- 判决实验：把"环境自身时变"这一维关掉 ----
    # static_risk=True 时，烟羽与风场冻结在 t=0，环境退化为静态风险场。
    # 用途：检验"时间维速度剖面"的收益是否依赖于时变性——
    #   收益消失 → 机制是场景特异的（创新成立）；收益不变 → 换皮。
    static_risk: bool = False

    # 瓶颈墙位置：gate_x=0 表示取 nx//2；缺口按 ny 的比例给出。
    # 之所以可配：默认的中间列在真实 BAG 底图上被建筑夹住，缺口宽度不足以满足
    # d_min，左右自由区互不连通、任务全落一侧（详见 environment.py 注释）。
    gate_x: int = 0
    gate_lo: float = 0.38
    gate_hi: float = 0.58

    # ---- 实验设置 ----
    seed: int = 42
    n_agents: int = 6
    dynamic_change_at: int = 12
    enable_dynamic: bool = False


def chebyshev(a, b):
    """切比雪夫距离（格数），用于多尺度下界与邻域选择。"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def world_delta(cfg, a, b):
    """两个体素之间的实际三维位移（米）。"""
    return ((b[0] - a[0]) * cfg.cell_xy,
            (b[1] - a[1]) * cfg.cell_xy,
            (b[2] - a[2]) * cfg.cell_z)
