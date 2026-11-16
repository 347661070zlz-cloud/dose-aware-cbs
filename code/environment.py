"""城市灾害三维环境：建筑体素、坍塌/禁飞区、SDF 安全距离场、时变烟雾场与风场。

设计要点：
1. 不同 seed 生成真正不同的建筑布局与任务（避免"多个场景其实是同一个实例"）；
2. 水平分辨率 5 m、垂直 2.5 m，使最小离障距离存在有意义的取值区间；
3. 所有基线读取同一环境对象，保证同一 H、同一风险场、同一 Δt。
"""
import math
import random

from config import Config, chebyshev


class Environment:
    def __init__(self, cfg: Config, seed: int, bag_layout=None, bag_meta=None,
                 bottleneck=False):
        """bag_layout: 可选，来自 bag_data.load_layout() 的真实 3D BAG 建筑体素集合。
        传入后建筑布局改用公开真实数据，禁飞区/烟源/风场/任务仍由 seed 生成，
        保证同一真实底图上可做多种子重复实验。不传则沿用程序化随机楼块。
        """
        self.cfg = cfg
        self.seed = seed
        self.blocked = set()          # 建筑体素
        self.no_fly = set()           # 坍塌/禁飞区
        self.future_no_fly = set()    # 预测在 dynamic_change_at 后坍塌的体素
        self.smoke_sources = []       # (x, y, z, amp, sigma, phase)
        self.wind_speed = 1.0
        self.sdf = {}                 # voxel -> 到最近障碍的距离(m)
        self.tasks = []               # [(start, goal), ...]
        self.bag_layout = bag_layout
        self.bag_meta = bag_meta or {}
        self.bottleneck = bottleneck
        self._rng = random.Random(seed)
        self._build()
        self._compute_sdf()
        self._build_tasks()
        # 环境指纹：布局（含真实底图/程序化楼块）+ 禁飞区 + 种子 的唯一标识。
        # 低层精确启发表(_EXACT_TABLES)以它为缓存键的一部分，避免"同进程内换了一张
        # 底图但 seed/栅格/目标相同"时误用上一张图的 h* 表。
        self.fingerprint = hash((frozenset(self.blocked), frozenset(self.no_fly),
                                 seed, cfg.nx, cfg.ny, cfg.nz))

    # ------------------------------------------------------------------ 建模
    def _build(self):
        c, rng = self.cfg, self._rng
        free_cells = [(x, y, z) for x in range(c.nx) for y in range(c.ny)
                      for z in range(c.nz)]

        if self.bag_layout:
            # 公开 3D BAG 真实建筑（footprint + b3_h_70p 高度栅格化结果）
            self.blocked = {v for v in self.bag_layout
                            if 0 <= v[0] < c.nx and 0 <= v[1] < c.ny
                            and 0 <= v[2] < c.nz}
        else:
            # 程序化随机矩形楼块：不同 seed 布局不同（对照用）
            n_blocks = 7 + rng.randint(0, 5)
            for _ in range(n_blocks):
                x0 = rng.randint(1, max(1, c.nx - 5))
                y0 = rng.randint(1, max(1, c.ny - 5))
                w = rng.randint(2, 4)
                h = rng.randint(2, 4)
                top = rng.randint(1, c.nz)
                for x in range(x0, min(c.nx, x0 + w)):
                    for y in range(y0, min(c.ny, y0 + h)):
                        for z in range(top):
                            self.blocked.add((x, y, z))

        # 坍塌/禁飞区：球形，位于中低空
        n_nofly = 2 + rng.randint(0, 2)
        for _ in range(n_nofly):
            cx = rng.randint(2, c.nx - 3)
            cy = rng.randint(2, c.ny - 3)
            cz = rng.randint(0, c.nz - 1)
            rad = rng.randint(2, 3)
            for v in free_cells:
                if (v[0] - cx) ** 2 + (v[1] - cy) ** 2 + (v[2] - cz) ** 2 <= rad ** 2:
                    self.no_fly.add(v)

        # 预测的未来坍塌区：动态障碍物预测模块据此在 t >= dynamic_change_at
        # 时提前避让。范围较小，避免把可行域整体堵死。
        # 使用独立随机流：避免消耗主 rng 而改变禁飞区/烟源/任务的既有布局，
        # 保证加入该模块前后所有场景可复现、可与既有结果对比。
        frng = random.Random(self.seed * 31 + 7919)
        fcx = frng.randint(3, max(3, c.nx - 4))
        fcy = frng.randint(3, max(3, c.ny - 4))
        fcz = frng.randint(0, max(0, c.nz - 2))
        for v in free_cells:
            if ((v[0] - fcx) ** 2 + (v[1] - fcy) ** 2
                    + (v[2] - fcz) ** 2) <= 4:
                self.future_no_fly.add(v)

        # 灾害场景「窄通道区」：坍塌封路形成应急走廊瓶颈（借鉴 600m 项目的
        # bottleneck gate）。在真实 3D BAG 底图上叠加一道封墙，只留 4 格(20 m)
        # 缺口，使多机必须争用同一通道，从而产生可测的高层冲突。
        if self.bottleneck:
            wx = c.gate_x if getattr(c, "gate_x", 0) else c.nx // 2
            gap_lo = int(getattr(c, "gate_lo", 0.38) * c.ny)
            gap_hi = int(getattr(c, "gate_hi", 0.58) * c.ny)
            for y in range(c.ny):
                if gap_lo <= y <= gap_hi:
                    continue
                for z in range(c.nz):
                    self.no_fly.add((wx, y, z))

        # 烟雾源（高斯团），随时间缓慢漂移；周期 = tau_env
        n_smoke = 2 + rng.randint(0, 2)
        for _ in range(n_smoke):
            self.smoke_sources.append((
                rng.randint(2, c.nx - 3) * c.cell_xy,
                rng.randint(2, c.ny - 3) * c.cell_xy,
                rng.randint(0, c.nz - 1) * c.cell_z,
                rng.uniform(0.35, 0.65),
                rng.uniform(12.0, 22.0),
                rng.uniform(0.0, 2 * math.pi),
            ))
        self.wind_speed = rng.uniform(0.6, 1.4)

    def _compute_sdf(self):
        c = self.cfg
        obstacles = [(x * c.cell_xy, y * c.cell_xy, z * c.cell_z)
                     for (x, y, z) in self.blocked]
        self.obstacles_world = obstacles
        # 按 x 分桶，供连续离障距离查询（避免每次遍历全部障碍）
        self._obs_buckets = {}
        for o in obstacles:
            k = int(o[0] // c.cell_xy)
            self._obs_buckets.setdefault(k, []).append(o)
        # 精确欧氏距离变换：真实建筑体素较多时比 O(free×obstacles) 暴力快得多。
        # 各向异性采样 (z, y, x) 与 cell_z / cell_xy 对应，语义与原暴力版一致
        # （到最近障碍体素中心的米制距离）。scipy 不可用时自动回退暴力实现。
        try:
            import numpy as np
            from scipy.ndimage import distance_transform_edt
            arr = np.ones((c.nz, c.ny, c.nx), dtype=bool)
            for (x, y, z) in self.blocked:
                arr[z, y, x] = False
            edt = distance_transform_edt(
                arr, sampling=(c.cell_z, c.cell_xy, c.cell_xy))
            for x in range(c.nx):
                for y in range(c.ny):
                    for z in range(c.nz):
                        self.sdf[(x, y, z)] = float(edt[z, y, x])
            return
        except ImportError:
            pass
        big = 1e6
        for x in range(c.nx):
            for y in range(c.ny):
                for z in range(c.nz):
                    v = (x, y, z)
                    if v in self.blocked:
                        self.sdf[v] = 0.0
                        continue
                    px, py, pz = x * c.cell_xy, y * c.cell_xy, z * c.cell_z
                    best = big
                    for ox, oy, oz in obstacles:
                        d2 = (px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2
                        if d2 < best:
                            best = d2
                    self.sdf[v] = math.sqrt(best) if best < big else 999.0

    def _build_tasks(self):
        """生成互不重合的起终点，并强制不同 seed 得到不同任务。"""
        c, rng = self.cfg, self._rng
        # 起终点自身也必须在 t=0 满足硬约束 H（含烟雾），否则 A* 只检查扩展节点、
        # 连续轨迹复核却检查起点，会出现"骨架合法但优化永不生效"的口径不一致。
        candidates = [(x, y, z) for x in range(c.nx) for y in range(c.ny)
                      for z in range(c.nz)
                      if self.is_free((x, y, z))
                      and self.sdf[(x, y, z)] >= c.d_min
                      and self.smoke_at((x, y, z), 0) <= c.smoke_rate_max]
        # 静态可行域连通分量：只在同一主连通区内配对，避免"起终点在硬约束下
        # 根本不可达"导致低层穷尽搜索、预算耗尽（这是灾害废墟里真实存在的问题）。
        static_ok = {v for v in self.sdf
                     if self.is_free(v) and self.sdf[v] >= c.d_min}
        comp = {}
        cid = 0
        for v0 in static_ok:
            if v0 in comp:
                continue
            stack = [v0]
            comp[v0] = cid
            while stack:
                v = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            w = (v[0] + dx, v[1] + dy, v[2] + dz)
                            if w in static_ok and w not in comp:
                                comp[w] = cid
                                stack.append(w)
            cid += 1
        if comp:
            sizes = {}
            for k in comp.values():
                sizes[k] = sizes.get(k, 0) + 1
            main = max(sizes, key=sizes.get)
            candidates = [v for v in candidates if comp.get(v) == main]
            rng.shuffle(candidates)
        need = 2 * max(c.n_agents, 24)
        if len(candidates) < need:
            # 放宽离障约束，保证有足够候选点
            candidates = [v for v in candidates if True]
            more = [(x, y, z) for x in range(c.nx) for y in range(c.ny)
                    for z in range(c.nz)
                    if self.is_free((x, y, z))
                    and self.sdf[(x, y, z)] >= c.d_min
                    and self.smoke_at((x, y, z), 0) <= c.smoke_rate_max
                    and (x, y, z) not in candidates]
            rng.shuffle(more)
            candidates += more
        # "对冲穿越"任务：一半 agent 由左向右、一半由右向左，且集中在
        # 中间高度层与中间 y 带，制造真实的多机冲突以检验高层冲突消解。
        # 距离限制在 4~10 格，避免单机规划本身吃掉整个 T_budget。
        used = set()
        tasks = []
        want = need // 2
        band = [v for v in candidates
                if c.ny // 3 <= v[1] <= 2 * c.ny // 3 and v[2] in (1, 2)]
        left = [v for v in band if v[0] <= c.nx // 2 - 2]
        right = [v for v in band if v[0] >= c.nx // 2 + 1]
        rng.shuffle(left)
        rng.shuffle(right)
        for k in range(want):
            spool, gpool = (left, right) if k % 2 == 0 else (right, left)
            s = g = None
            for ss in spool:
                if ss in used:
                    continue
                for gg in gpool:
                    if gg in used or gg == ss:
                        continue
                    if 4 <= chebyshev(ss, gg) <= 10:
                        s, g = ss, gg
                        break
                if s is not None:
                    break
            if s is None:
                break
            tasks.append((s, g))
            used.add(s)
            used.add(g)
        # 兜底：候选带不足时退回随机配对，保证任务数量
        if len(tasks) < want:
            for s in candidates:
                if len(tasks) >= want:
                    break
                if s in used:
                    continue
                for g in candidates:
                    if g in used or g == s:
                        continue
                    tasks.append((s, g))
                    used.add(s)
                    used.add(g)
                    break
        self.tasks = tasks

    # ------------------------------------------------------------------ 查询
    def in_bounds(self, v):
        c = self.cfg
        return 0 <= v[0] < c.nx and 0 <= v[1] < c.ny and 0 <= v[2] < c.nz

    def is_free(self, v):
        return self.in_bounds(v) and v not in self.blocked and v not in self.no_fly

    def sdf_at(self, v):
        return self.sdf.get(tuple(v), 0.0)

    def continuous_sdf(self, p):
        """连续三维点到最近障碍体素"表面"的距离（米）。

        离散 sdf_at 只取 {5, 7.07, 10...} 等离散值，无法度量 PSO 连续轨迹
        被推离障碍的收益；这里用点到轴对齐盒的距离，得到真正连续的裕度量。
        """
        c = self.cfg
        hx, hy, hz = c.cell_xy / 2, c.cell_xy / 2, c.cell_z / 2
        k = int(p[0] // c.cell_xy)
        best = 1e18
        for kk in range(k - 3, k + 4):
            for ox, oy, oz in self._obs_buckets.get(kk, ()):
                dx = max(0.0, abs(p[0] - ox) - hx)
                dy = max(0.0, abs(p[1] - oy) - hy)
                dz = max(0.0, abs(p[2] - oz) - hz)
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best:
                    best = d2
        return math.sqrt(best) if best < 1e17 else 999.0

    def smoke_at(self, v, t):
        c = self.cfg
        if c.static_risk:
            t = 0
        px, py, pz = v[0] * c.cell_xy, v[1] * c.cell_xy, v[2] * c.cell_z
        total = 0.0
        for sx, sy, sz, amp, sigma, phase in self.smoke_sources:
            d2 = (px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2
            drift = 0.8 + 0.2 * math.sin(2 * math.pi * t * c.dt / c.tau_env + phase)
            total += amp * math.exp(-d2 / (2.0 * sigma * sigma)) * drift
        return min(1.0, total)

    def wind_at(self, v, t):
        """缓慢旋转的水平风场；逆风代价在 cost.py 中按运动方向计算。"""
        if self.cfg.static_risk:
            t = 0
        ang = 2 * math.pi * t * self.cfg.dt / self.cfg.tau_env
        return (self.wind_speed * math.cos(ang),
                self.wind_speed * math.sin(ang),
                0.0)

    def local_risk(self, v, t):
        """[0,1] 局部危险度：SDF 越低、烟雾越浓越危险。用于高层影响评估。"""
        sdf = self.sdf_at(v)
        sdf_term = max(0.0, 1.0 - sdf / (4.0 * self.cfg.d_min))
        return min(1.0, 0.5 * sdf_term + 0.5 * self.smoke_at(v, t))

    def obstacle_density(self, v, radius=1):
        cnt = tot = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == dy == dz == 0:
                        continue
                    q = (v[0] + dx, v[1] + dy, v[2] + dz)
                    if not self.in_bounds(q):
                        continue
                    tot += 1
                    if q in self.blocked or q in self.no_fly:
                        cnt += 1
        return cnt / tot if tot else 0.0

    # ------------------------------------------------------------------ 动态
    def apply_dynamic_change(self, t_change):
        """环境中途变化：新增坍塌区并让烟源移动。用于动态重规划实验。"""
        c = self.cfg
        rng = random.Random(self.seed * 1000 + int(t_change))
        cx, cy = rng.randint(3, c.nx - 4), rng.randint(3, c.ny - 4)
        for x in range(c.nx):
            for y in range(c.ny):
                for z in range(c.nz):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - 1) ** 2 <= 4:
                        self.no_fly.add((x, y, z))
        self.smoke_sources = [
            (sx + 8.0, sy - 6.0, sz, min(1.0, amp * 1.15), sigma, phase + 0.7)
            for (sx, sy, sz, amp, sigma, phase) in self.smoke_sources
        ]
        self._compute_sdf()


def scenario_digest(env, n_agents):
    """用于断言不同 seed 的场景确实不同，防止有效样本量退化为 1。"""
    tasks = tuple((s, g) for s, g in env.tasks[:n_agents])
    return hash((env.seed, tasks, tuple(sorted(env.blocked))[::17]))
