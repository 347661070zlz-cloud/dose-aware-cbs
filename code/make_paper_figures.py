"""生成**英文版**论文插图（投 IEEE TAES 用），输出到 ../paper/figures。

为什么要单独一个脚本：`make_figures.py` 服务中文稿与图册，标题/图例都是中文；
英文稿的图不能出现中文。这里复用同一批数据与配色，只把文字换成英文，
并且**不改动**中文图册的生成流程（避免把中文稿的图改坏）。

生成 4 张（与 main.tex 的 Fig.1–Fig.4 一一对应）：
    fig02_scene_3d.png    3D 城区鸟瞰 + 研究子区
    fig01_scene_2d.png    实验子区俯视
    fig03_trajectories.png 六机轨迹对比
    fig12_frontier.png    暴露—时效前沿
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# 数据根目录：投稿包内为 ../data（内含 thesis/ 与 bag/）；原始工程内为 results/
_DATA_ENV = os.environ.get("PAPER_DATA")
if _DATA_ENV:
    DATA = os.path.abspath(_DATA_ENV)
elif os.path.isdir(os.path.join(HERE, "..", "data", "thesis")):
    DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
else:
    DATA = os.path.join(HERE, "results")
THESIS = os.path.join(DATA, "thesis")
BAG = os.path.join(DATA, "bag")
OUT = os.path.abspath(os.path.join(HERE, "..", "figures"))
if os.path.basename(HERE) != "code":
    OUT = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))


def read_csv(path):
    import csv as _csv
    with open(path, encoding="utf-8-sig") as f:
        return list(_csv.DictReader(f))


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _grid(s):
    g = s["grid"]
    h = np.zeros((g["ny"], g["nx"]))
    for x, y, z in s["blocked"]:
        h[y, x] = max(h[y, x], z + 1)
    return g, h


def fig_scene_2d(cfg_seed=48, n_agents=6, budget=2.88, d_min=6.0):
    """实验区俯视图：直接从 Environment 生成，避免读取过期的 scene json。

    旧版读 results/bag/scene_and_paths.json，那份 json 是用旧 bbox 生成的；
    实验区北移后必须重建，否则图上画的不是实验用的那块地。
    """
    from dataclasses import replace
    from matplotlib.colors import ListedColormap
    from config import Config
    import thesis_experiments as T

    T._USE_BAG = True
    cfg = replace(Config(), seed=cfg_seed, n_agents=n_agents,
                  t_budget=3.0, heuristic_mode="exact",
                  dose_budget=budget, d_min=d_min)
    env = T.make_env(cfg, cfg_seed)
    nx, ny = cfg.nx, cfg.ny

    H = np.zeros((ny, nx))
    for (x, y, z) in env.blocked:
        H[y, x] = max(H[y, x], z + 1)
    NF = np.zeros((ny, nx))
    for (x, y, z) in env.no_fly:
        NF[y, x] = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(H, origin="lower", cmap="Blues", vmin=0, vmax=max(1, H.max()),
                   extent=[0, nx, 0, ny], interpolation="nearest", alpha=0.9)
    ax.imshow(np.where(NF > 0, 1, np.nan), origin="lower",
              cmap=ListedColormap(["#c0504d"]), vmin=0, vmax=1,
              extent=[0, nx, 0, ny], alpha=0.45, interpolation="nearest")

    for src in env.smoke_sources:
        sx, sy = src[0] / cfg.cell_xy, src[1] / cfg.cell_xy
        ax.add_patch(plt.Circle((sx + 0.5, sy + 0.5), radius=src[4] / cfg.cell_xy,
                                fill=False, ls="--", lw=1.1, ec="#d98b2b"))
        ax.plot(sx + 0.5, sy + 0.5, marker="^", ms=7, color="#d98b2b")

    for i, (st_, gl) in enumerate(env.tasks[:n_agents]):
        ax.plot(st_[0] + 0.5, st_[1] + 0.5, "o", ms=7, color="#1f7a5a",
                label="Task start" if i == 0 else None)
        ax.plot(gl[0] + 0.5, gl[1] + 0.5, "*", ms=11, color="#c0504d",
                label="Task goal" if i == 0 else None)
        ax.annotate(str(i), (st_[0] + 0.5, st_[1] + 0.5), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")

    ax.annotate("wind", xy=(3.0, ny - 1.2), xytext=(1.4, ny - 1.2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#4a6fa5"),
                color="#4a6fa5")
    plt.colorbar(im, ax=ax, label="Building height (voxel layers)",
                 fraction=0.046, pad=0.03)
    ax.set_xlabel("X (cells, 5 m)")
    ax.set_ylabel("Y (cells, 5 m)")
    ax.set_title("Study area of the public 3D BAG scene: plan view "
                 f"({nx*cfg.cell_xy:.0f} m $\\times$ {ny*cfg.cell_xy:.0f} m)",
                 fontsize=10, color="#22304a")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4,
              fontsize=6.8, frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT, "fig01_scene_2d.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _pick_violating_case(budget):
    """挑一个"基线违规、加层后可合规"的场景，否则两幅图会完全相同。"""
    import csv
    fp = os.path.join(THESIS, "detour.csv")
    if not os.path.exists(fp):
        return PREFERRED_CASE
    cases = []
    with open(fp, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("feasible") == "1":
                seed = int(str(r["scenario"]).split("_s")[1].split("_")[0])
                cases.append((seed, int(r["n"])))
    if PREFERRED_CASE in cases:
        return PREFERRED_CASE
    return cases[0] if cases else PREFERRED_CASE


# 示例场景与配色（供 fig_traj / fig_scene_2d 使用）
PREFERRED_CASE = (48, 6)
DEMO_D_MIN = 6.0
DEMO_CAP = 2.88
COOL_GREY = "#9aa5b1"
COOL_DARK = "#1f4e79"
COOL_TEAL = "#2c7a7b"
COOL_ZONE = "#5b6fa8"


def _scene_arrays(env, cfg):
    """把场景压成三张图：建筑高度、禁区掩膜、烟雾浓度。"""
    nx, ny = cfg.nx, cfg.ny
    H = np.zeros((ny, nx))
    for (x, y, z) in env.blocked:
        H[y, x] = max(H[y, x], z + 1)
    NF = np.zeros((ny, nx))
    for (x, y, z) in env.no_fly:
        NF[y, x] = 1.0
    S = np.array([[env.smoke_at((x, y, 1), 0) for x in range(nx)]
                  for y in range(ny)])
    return H, NF, S


def _per_agent_dose(env, res, tasks, cfg):
    from highlevel import arrival_index
    out = []
    for i, (s_, g_) in enumerate(tasks):
        if i not in res.paths:
            out.append(float("nan"))
            continue
        arr = arrival_index(res.paths[i], g_)
        out.append(sum(env.smoke_at(res.paths[i][t], t) * cfg.dt
                       for t in range(arr + 1)))
    return out


COOL_GREY = "#9aa5b1"
COOL_DARK = "#1f4e79"
COOL_TEAL = "#2c7a7b"
COOL_ZONE = "#5b6fa8"


def fig_traj(cfg_seed=None, n_agents=None, budget=DEMO_CAP, d_min=DEMO_D_MIN):
    """轨迹对比图（冷色调、低信息密度）。

    上一版糊成一团的原因：六条彩色轨迹 + 暖色烟羽色场 + 红色禁区块叠在一起。
    这一版的做法：
      · 背景全部冷色：建筑浅灰、烟羽冷蓝、禁区淡靛蓝；
      · 只有"路线发生变化"的那架带颜色（深蓝实线 / 虚线＝另一方案），
        其余五架统一浅灰细线，不再抢视觉；
      · 底下加一张逐机剂量柱状图（两方案分组），把"哪架超标、降到多少"
        直接画出来，不再需要从轨迹里数。
    """
    import time
    from dataclasses import replace
    from matplotlib.colors import ListedColormap
    from config import Config
    import thesis_experiments as T
    from highlevel import solve, arrival_index
    from lowlevel import warmup_heuristics
    from methods import METHODS

    if cfg_seed is None or n_agents is None:
        cfg_seed, n_agents = PREFERRED_CASE
    T._USE_BAG = True
    cfg = replace(Config(), seed=cfg_seed, n_agents=n_agents,
                  t_budget=3.0, heuristic_mode="exact", dose_budget=budget,
                  d_min=d_min)
    env = T.make_env(cfg, cfg_seed)
    tasks = env.tasks[:n_agents]
    warmup_heuristics(cfg, env, tasks)
    H, NF, S = _scene_arrays(env, cfg)

    runs = []
    for name, lab in (("ECBS", "(a) cap-blind ECBS"),
                      ("ECBS_DOSE", "(b) ECBS + constraint layer")):
        res = solve(cfg, env, tasks, METHODS[name],
                    time.monotonic() + 3.0, cfg_seed)
        runs.append((lab, res, _per_agent_dose(env, res, tasks, cfg)))

    colmap = plt.get_cmap("tab10")
    d0, d1 = runs[0][2], runs[1][2]
    diffs = [abs((x - y)) if not (np.isnan(x) or np.isnan(y)) else -1
             for x, y in zip(d0, d1)]
    hot = int(np.argmax(diffs)) if max(diffs) > 1e-6 else -1

    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.0], hspace=0.30,
                          wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], sharey=None)]

    for ax, (lab, res, doses) in zip(axes, runs):
        # 建筑：带描边的格子，使"建筑之间的巷道"可见，而不是一团灰
        for (bx, by, bz) in {(x, y, 0) for (x, y, z) in env.blocked if z == 0}:
            ax.add_patch(plt.Rectangle((bx, by), 1, 1, facecolor="#c9c9c9",
                                       edgecolor="#8a8a8a", linewidth=0.35,
                                       alpha=0.75, zorder=2))
        ax.imshow(S, origin="lower", cmap="Blues", vmin=0.05, vmax=0.75,
                  extent=[0, cfg.nx, 0, cfg.ny], alpha=0.30,
                  interpolation="bilinear")
        ax.imshow(np.where(NF > 0, 1, np.nan), origin="lower",
                  cmap=ListedColormap([COOL_ZONE]), vmin=0, vmax=1,
                  extent=[0, cfg.nx, 0, cfg.ny], alpha=0.16,
                  interpolation="nearest")

        for i, (s_, g_) in enumerate(tasks):
            if i not in res.paths:
                continue
            p_ = res.paths[i]
            arr = arrival_index(p_, g_)
            if i == hot:
                continue
            col = colmap(i % 10)
            ax.plot([v[0] + 0.5 for v in p_[:arr + 1]],
                    [v[1] + 0.5 for v in p_[:arr + 1]],
                    lw=1.3, color=col, alpha=0.55, zorder=3)
            ax.plot(s_[0] + 0.5, s_[1] + 0.5, "o", ms=3.4, color=col,
                    mec="white", mew=0.4, alpha=0.8, zorder=4)
            ax.plot(g_[0] + 0.5, g_[1] + 0.5, "s", ms=3.4, color=col,
                    mec="white", mew=0.4, alpha=0.8, zorder=4)
            ax.annotate(str(i), (s_[0] + 0.5, s_[1] + 0.5), fontsize=6.4,
                        color=col, xytext=(3, 3), textcoords="offset points",
                        zorder=8)

        if hot >= 0 and hot in res.paths:
            other = runs[1] if lab.startswith("(a)") else runs[0]
            p_ = res.paths[hot]
            hc = colmap(hot % 10)
            ax.plot([v[0] + 0.5 for v in p_[:arrival_index(p_, tasks[hot][1]) + 1]],
                    [v[1] + 0.5 for v in p_[:arrival_index(p_, tasks[hot][1]) + 1]],
                    lw=2.6, color=hc, zorder=6)
            if hot in other[1].paths:
                po = other[1].paths[hot]
                ax.plot([v[0] + 0.5 for v in po[:arrival_index(po, tasks[hot][1]) + 1]],
                        [v[1] + 0.5 for v in po[:arrival_index(po, tasks[hot][1]) + 1]],
                        lw=1.1, ls=(0, (3, 2)), color=hc, alpha=0.55,
                        zorder=5)
            ax.plot(tasks[hot][0][0] + 0.5, tasks[hot][0][1] + 0.5, "o", ms=5,
                    color=hc, mec="white", mew=0.7, zorder=7)
            ax.plot(tasks[hot][1][0] + 0.5, tasks[hot][1][1] + 0.5, "s", ms=5,
                    color=hc, mec="white", mew=0.7, zorder=7)
            ax.annotate(f"UAV {hot}", xy=(tasks[hot][1][0] + 0.5,
                                          tasks[hot][1][1] + 0.5),
                        xytext=(5, -11), textcoords="offset points",
                        fontsize=7.2, color=hc, fontweight="bold")

        # 体素口径的最小离障（硬约束检查量）：与 external_risk 的连续轨迹口径不同，
        # 后者会把平滑后的航迹量到 3.5 m，与论文 6 m 硬约束写成一句话就矛盾了。
        mc = min(env.sdf[v] for i in range(len(tasks))
                 for v in res.paths[i][:arrival_index(res.paths[i], tasks[i][1]) + 1])
        ax.text(0.02, 0.03, f"min voxel clearance {mc:.1f} m (constraint {d_min:.1f} m)",
                transform=ax.transAxes, fontsize=6.6, color="#3a3a3a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85), zorder=9)
        ok = res.solved and max(doses) <= budget + 1e-9
        mk = max(arrival_index(res.paths[i], g_) * cfg.dt
                 for i, (s_, g_) in enumerate(tasks))
        ax.set_title(f"{lab}\nmax dose {max(doses):.2f} / cap {budget:.2f}, "
                     f"makespan {mk:.0f} s" + ("" if ok else "   (violates)"),
                     fontsize=8.4, color="#1b1b1b" if ok else "#8c2d24")
        ax.set_xlim(0, cfg.nx); ax.set_ylim(0, cfg.ny)
        ax.set_xlabel("X (cells, 5 m)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Y (cells, 5 m)", fontsize=8)

    # ---- 下方：逐机剂量柱状图 ----
    axb = fig.add_subplot(gs[1, :])
    xs = np.arange(n_agents)
    w = 0.38
    axb.bar(xs - w / 2, d0, w, color=COOL_GREY, label="cap-blind ECBS")
    axb.bar(xs + w / 2, d1, w, color=COOL_TEAL, label="+ constraint layer")
    axb.axhline(budget, color="#8c2d24", lw=0.9, ls="--")
    axb.text(-0.45, budget + 0.03, f"cap {budget:.2f}", va="bottom", ha="left",
             fontsize=7, color="#8c2d24")
    axb.set_xticks(xs)
    axb.set_xticklabels([f"UAV {i}" for i in xs], fontsize=7)
    axb.set_ylabel("cumulative dose", fontsize=8)
    axb.set_ylim(0, max(np.nanmax(d0), np.nanmax(d1), budget) * 1.18)
    axb.set_xlim(-0.6, n_agents - 0.4)
    axb.tick_params(labelsize=7)
    axb.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(fc="#bdbdbd", alpha=0.55, label="buildings"),
               Patch(fc="#7fa8d4", alpha=0.55, label="smoke (t = 0)"),
               Patch(fc=COOL_ZONE, alpha=0.20, label="no-fly / hazard"),
               Line2D([], [], color=colmap(hot % 10) if hot >= 0 else COOL_DARK,
                      lw=2.6, label=f"UAV {hot}: route in this plan"),
               Line2D([], [], color=colmap(hot % 10) if hot >= 0 else COOL_DARK,
                      lw=1.1, ls=(0, (3, 2)), label="same UAV, other plan"),
               Line2D([], [], color="#8a8a8a", lw=1.3, alpha=0.6,
                      label="other UAVs (numbered at start)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.13)
    path = os.path.join(OUT, "fig03_trajectories.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_frontier():
    """暴露—时效前沿（ECBS 口径，英文）。

    设计说明（重要）：**不要画"层的违规率"**。低层在搜索中就把超预算的分支剪掉，
    而评价又用同一个预算去判违规，因此该曲线恒为 0 —— 那是约束的定义，不是实验
    结果，画出来只会让人以为指标失效。真正有信息量的是两件事：
        (a) 合规交付的**绝对数量**随预算的变化；
        (b) 基线"违规交付率"与本文"拒绝率"的对照 —— 本文把"静默交付不可行解"
            换成了"显式宣告无解"，两者数量应大致相当。
    """
    files = []
    pats = [("ecbs_B*_bag.csv", 6, -8), ("ecbs_B*.csv", 6, -4)]
    for pat, i0, i1 in pats:
        for fp in glob.glob(os.path.join(THESIS, pat)):
            stem = os.path.basename(fp)
            try:
                files.append((float(stem[i0:i1]), fp))
            except ValueError:
                continue
        if files:
            break
    files.sort()
    if not files:
        raise SystemExit("缺少 ecbs_B*.csv，请先跑 dose_ecbs 组")

    data = {}
    for B, fp in files:
        rows = read_csv(fp)
        d = {}
        for m in ("ECBS", "ECBS_DOSE"):
            rs = [r for r in rows if r["method"] == m]
            n = len(rs)
            sol = [r for r in rs if str(r["solved"]) in ("1", "1.0")]
            viol = [r for r in sol if fnum(r["dose_to_arrival"]) > B + 1e-6]
            d[m] = dict(n=n, solved=len(sol), viol=len(viol),
                        compl=len(sol) - len(viol),
                        refuse=n - len(sol))
        data[B] = d

    Bs = sorted(data)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # (a) 合规交付绝对数量
    ax = axes[0]
    for m, lab, col in (("ECBS", "ECBS (exposure-blind)", "#c0504d"),
                        ("ECBS_DOSE", "ECBS + constraint layer", "#1f7a5a")):
        ax.plot(Bs, [data[b][m]["compl"] for b in Bs], "-o", ms=5, color=col,
                lw=1.9, label=lab)
    ax.set_xlabel("Per-agent exposure budget (dose)")
    ax.set_ylabel("Compliant plans delivered (of 60)")
    ax.set_title("(a) Compliant delivery", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) 不可用结果：基线违规 vs 本文拒绝
    ax = axes[1]
    ax.plot(Bs, [data[b]["ECBS"]["viol"] / data[b]["ECBS"]["n"] for b in Bs],
            "-s", ms=5, color="#c0504d", lw=1.9,
            label="Baseline: plan violates the cap")
    ax.plot(Bs, [data[b]["ECBS_DOSE"]["refuse"] / data[b]["ECBS_DOSE"]["n"]
                 for b in Bs],
            "-o", ms=5, color="#1f7a5a", lw=1.9,
            label="Layer: declared infeasible")
    ax.set_xlabel("Per-agent exposure budget (dose)")
    ax.set_ylabel("Fraction of scenarios")
    ax.set_title("(b) Unusable outcome: silently wrong vs explicitly infeasible",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, "fig12_frontier.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(BAG, "scene_and_paths.json")
    s = json.load(open(sp, encoding="utf-8"))

    # 3D 图要走 make_scene3d 的语言开关
    os.environ["FIG_LANG"] = "en"
    import importlib
    import make_scene3d
    importlib.reload(make_scene3d)
    # 论文里图题交给 LaTeX 的 \caption，故这里传空串，避免图文重复
    p3d = make_scene3d.render_bag(s, os.path.join(OUT, "fig02_scene_3d.png"),
                                  title="", caption="")

    for p in (p3d, fig_scene_2d(), fig_traj(), fig_frontier()):
        print("written:", p)


if __name__ == "__main__":
    main()
