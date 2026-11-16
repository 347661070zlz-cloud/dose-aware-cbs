"""方法配置表：把低层规划器与高层策略组合成外部基线与 2×2 内部消融。

L0  质量参照：CBS 离线最优（不参与速度比较，只提供 C* 与 gap / omega 实测）
L1  同类主对照：ECBS / ICBS / 近年改进 CBS
L1' 快速朴素对照：优先规划 + SIPP
L3  强组合基线：已有最优低层 SIPP + 已有最优高层 ECBS（最容易被漏掉的竞争者）
L4  内部 2×2 消融：A 基线低层+基线高层 / B 仅新低层 / C 仅新高层 / D 完整方案
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    low_planner: str       # fixed | sipp | multiscale
    optimize: bool         # 是否启用阶段 B 安全走廊 PSO
    high_strategy: str     # cbs | ecbs | icbs | pp | wpp_baseline | ours
    budgeted: bool         # 是否受 T_budget 约束
    group: str
    adaptive: bool = False  # True = 启用预算感知协同；False = 非协同拼接对照
    speed_profile: bool = False  # 低层：True = 时间维速度剖面（A 方案），替代空间 PSO
    repair_use_speed: bool = False  # 高层：True = 修复可使用"整体后移/减速"自由度
    ablate: str = ""                # 7 模块消融：要去掉的模块名，空 = 完整方法
    force_stage_b: bool = False     # 消融用：强制在最终解执行阶段 B（不受协同门控影响）
    post_smooth: bool = False       # 对照用：仅对交付轨迹做几何平滑，不改变搜索
    dose_aware: bool = False        # 低层是否把累积烟雾剂量作为硬约束
    high_omega: float = 0.0         # 覆盖率 cfg.omega（0=用默认）；用于"更强的有界次优基线"
    # 注：低层机制与高层修复自由度必须分开配置，否则 2×2 消融的 C 格
    # （仅改高层）无法构造——早期版本把两者绑在同一个 speed_profile 上。


METHODS = {
    # ---- L0：离线最优参照（在本实现中给一个远大于 T_budget 的时间上限以防死循环）----
    "CBS_OFFLINE": MethodSpec("CBS_OFFLINE", "fixed", False, "cbs", False, "L0"),

    # ---- L1：同类有界次优 / 近年改进 ----
    "ECBS": MethodSpec("ECBS", "fixed", False, "ecbs", True, "L1"),
    "ICBS": MethodSpec("ICBS", "fixed", False, "icbs", True, "L1"),
    # EECBS：ECBS 的焦点搜索 + incumbent 上限剪枝 + anytime 改进。
    # 说明：未实现 Li et al. 的在线学习 cost-to-go（h_hat），此处用 incumbent
    # 上界与有界次优低层近似其"更强的高层搜索"，论文中按此如实标注。
    "EECBS": MethodSpec("EECBS", "fixed", False, "eecbs", True, "L1e"),
    "WPP_BASELINE": MethodSpec("WPP_BASELINE", "fixed", False,
                               "wpp_baseline", True, "L1"),

    # ---- L1r：开题表「对标基线 1」的文献改进 A* ----
    # 复现邵新峰等(2025)改进 A* 的核心机制：f = g + ω·h，ω 随"离终点远近"自适应
    # （远则 ω→5 偏贪心、近则 ω→0.2 保最优）。其 B 样条平滑部分在本实现中由
    # cost.external_risk 对**所有方法**统一施加的一次性 Chaikin 平滑体现，
    # 因此这里只复现权重部分，避免只给基线额外平滑而造成口径偏袒。
    "REF_ASTAR": MethodSpec("REF_ASTAR", "refastar", False, "cbs", True, "L1r"),

    # ---- L1'：快速朴素 ----
    "PP_SIPP": MethodSpec("PP_SIPP", "sipp", False, "pp", True, "L1p"),

    # ---- L3：已有最优组件的强拼接 ----
    "STRONG_SIPP_ECBS": MethodSpec("STRONG_SIPP_ECBS", "sipp", False,
                                   "ecbs", True, "L3"),

    # ---- L4：2×2 析因消融 ----
    "A_BASE": MethodSpec("A_BASE", "fixed", False, "cbs", True, "L4"),
    "B_LOWONLY": MethodSpec("B_LOWONLY", "multiscale", True, "cbs", True, "L4"),
    "C_HIGHONLY": MethodSpec("C_HIGHONLY", "fixed", False, "ours", True, "L4"),
    # D_FULL：预算感知协同版（真正的"高低层协同"）
    "D_FULL": MethodSpec("D_FULL", "multiscale", True, "ours", True, "L4",
                         adaptive=True),
    # 非协同拼接对照：与 D_FULL 完全相同，只是不下传预算信号，用于隔离协同的价值
    "D_FULL_NAIVE": MethodSpec("D_FULL_NAIVE", "multiscale", True, "ours", True,
                               "L4b"),
    # ---- A 方案的完整 2×2 析因消融 ----
    # A 格（基线/基线）复用 A_BASE；B 格、C 格、D 格如下。
    # B_LOWONLY_A : 仅改低层（速度剖面），高层保持经典 CBS
    "B_LOWONLY_A": MethodSpec("B_LOWONLY_A", "multiscale", False, "cbs", True,
                              "LA", speed_profile=True),
    # C_HIGHONLY_A: 仅改高层（评估-修复-分裂 + 速度剖面修复自由度），低层保持基线
    "C_HIGHONLY_A": MethodSpec("C_HIGHONLY_A", "fixed", False, "ours", True,
                               "LA", repair_use_speed=True),
    # D_FULL_A    : 两层都改 + 预算感知协同（完整联合方案 A）
    "D_FULL_A": MethodSpec("D_FULL_A", "multiscale", False, "ours", True,
                           "LA", adaptive=True, speed_profile=True,
                           repair_use_speed=True),
    # ---- 低层因子 2×2（论文 1 的归因实验）----
    # 目的：把"仅改低层"这一格（B_LOWONLY_A = 多尺度 + 速度剖面）拆开，
    # 否则无法回答"3 倍解出率到底是谁带来的"。
    #   MULTI_ONLY : 只开多尺度（空间维），不碰时间维
    #   SPEED_ONLY : 只开时间维速度剖面，步长固定（= 纯时间重参数化）
    "MULTI_ONLY": MethodSpec("MULTI_ONLY", "multiscale", False, "cbs", True,
                             "L4l"),
    # CBS_DOSE：与基线同构，但低层是剂量受限搜索（累积烟雾暴露为硬约束）。
    # 用途：与"剂量盲"的基线对比——看的是**可行性**（是否违反剂量预算），
    # 而不是速度。剂量盲方法在预算收紧后会返回不可行解，这才是差异所在。
    "CBS_DOSE": MethodSpec("CBS_DOSE", "fixed", False, "cbs", True, "L5",
                           dose_aware=True),
    # 强基线的剂量感知版本：把"资源约束层"接到 ECBS 上。
    # 用途有二：① 回答"你只跟弱基线比"；② 证明剂量约束的处理与高层算法正交
    # ——若 ECBS_DOSE 也零违规，则说明该层可以加在任何 CBS 族高层之上。
    "ECBS_DOSE": MethodSpec("ECBS_DOSE", "fixed", False, "ecbs", True, "L5",
                            dose_aware=True),
    # EECBS 的剂量感知版本：检验该层在更强的高层上同样成立。
    "EECBS_DOSE": MethodSpec("EECBS_DOSE", "fixed", False, "eecbs", True, "L5",
                             dose_aware=True),
    # 更强的有界次优基线：放宽 ω，搜索更激进（解出更多、解质量更松）。
    # 用途：检验"方法越强，剂量盲的违规是否越多"这一单调性论点——
    # 若成立，则更强的 EECBS 只会让违规更严重，无需另行实现也能支撑结论。
    "ECBS_W30": MethodSpec("ECBS_W30", "fixed", False, "ecbs", True, "L1",
                           high_omega=3.0),
    "ECBS_W50": MethodSpec("ECBS_W50", "fixed", False, "ecbs", True, "L1",
                           high_omega=5.0),
    "SPEED_ONLY": MethodSpec("SPEED_ONLY", "fixed", False, "cbs", True,
                             "L4l", speed_profile=True),
    # POST_SMOOTH：几何后处理对照——搜索完全不变（与基线一致），
    # 只在交付轨迹上做多次平滑，用于回答"你的速度剖面是不是就等于事后平滑"。
    "POST_SMOOTH": MethodSpec("POST_SMOOTH", "fixed", False, "cbs", True,
                              "L4l", post_smooth=True),

    # D_FULL_A_NAIVE : 与 D_FULL_A 完全相同，但不下传预算信号。
    # 用途：让速度剖面这一支的 2×2 也闭合——(A_BASE, B_LOWONLY_A,
    # C_HIGHONLY_A, D_FULL_A_NAIVE) 四格只差「低层/高层」两个因子，
    # 协同作为第三个因子由 D_FULL_A − D_FULL_A_NAIVE 单独衡量。
    # PSO 支同理用 D_FULL_NAIVE。早期版本把协同折进 D 格，导致 D−C 无法归因。
    "D_FULL_A_NAIVE": MethodSpec("D_FULL_A_NAIVE", "multiscale", False, "ours",
                                 True, "LA", speed_profile=True,
                                 repair_use_speed=True),
}

# 论文主结论取 D_FULL - A_BASE；B-A、C-A 归因；(D-B)-(C-A) 反映协同。
# D_FULL - D_FULL_NAIVE 单独衡量"预算感知协同"这一机制本身的贡献。
ABLATION = ("A_BASE", "B_LOWONLY", "C_HIGHONLY", "D_FULL")
MAIN_BASELINES = ("REF_ASTAR", "ECBS", "ICBS", "WPP_BASELINE", "PP_SIPP",
                  "STRONG_SIPP_ECBS")
EXTRA = ("D_FULL_NAIVE", "D_FULL_A_NAIVE")

# A 方案 2×2：A_BASE / B_LOWONLY_A / C_HIGHONLY_A / D_FULL_A
EXTRA_A = ("B_LOWONLY_A", "C_HIGHONLY_A", "D_FULL_A")
ABLATION_A = ("A_BASE", "B_LOWONLY_A", "C_HIGHONLY_A", "D_FULL_A")

# 开题表「验证方法 5」列出的 7 个模块。每项给出从完整联合方案中去掉该模块的方式。
# 完整方法以 D_FULL（开题版低层：多尺度 A* + 走廊 PSO + 修复优先高层）为基准。
MODULE_ABLATIONS = {
    "full": dict(suffix="完整方法", ablate="", low="multiscale",
                 optimize=True, overrides={}),
    "impact_eval": dict(suffix="去高层冲突影响评估", ablate="impact_eval",
                        low="multiscale", optimize=True, overrides={}),
    "local_repair": dict(suffix="去局部修复", ablate="local_repair",
                         low="multiscale", optimize=True, overrides={}),
    "adaptive_step": dict(suffix="去自适应步长", ablate="", low="fixed",
                          optimize=True, overrides={}),
    "sdf_penalty": dict(suffix="去SDF安全距离惩罚", ablate="", low="multiscale",
                        optimize=True, overrides={"w_sdf": 0.0}),
    "dynamic_prediction": dict(suffix="去动态障碍预测", ablate="",
                               low="multiscale", optimize=True,
                               overrides={"predict_dynamic": False}),
    "corridor": dict(suffix="去安全走廊约束", ablate="", low="multiscale",
                     optimize=True, overrides={"corridor_enabled": False}),
    "pso": dict(suffix="去PSO连续优化", ablate="", low="multiscale",
                optimize=False, overrides={}),
}
