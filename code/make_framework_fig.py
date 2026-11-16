"""框架示意图（学术风格）：细线直角框、白底、短标签、以记号代替散文。

排版约束（上一版的问题）：文字必须短到框内放得下；连线一律走框外，
不允许穿过方框或文字；(b) 的注记要留空位、用引线指认，不压曲线。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 7.0,
})

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "paper", "figures"))
LW, GRAY, DASH = 0.8, "#f2f2f2", (0, (4, 2))


def box(ax, x, y, w, h, fc="none"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                           linewidth=LW, zorder=2))


def arr(ax, p0, p1, ls="-", lw=LW, color="black", z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=z))


def architecture(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.03, 0.76, 0.58, 0.19)
    ax.text(0.32, 0.885, r"constraint tree (CBS / ECBS)", ha="center")
    ax.text(0.32, 0.815, r"node $n$: $C_n$, $\{\pi_i\}$", ha="center")

    box(ax, 0.03, 0.42, 0.58, 0.19, fc=GRAY)
    ax.text(0.32, 0.545, r"low-level search", ha="center")
    ax.text(0.32, 0.475, r"state $(v,t,d)$; labels $\mathcal{L}(v,t)$", ha="center")

    box(ax, 0.03, 0.08, 0.58, 0.19)
    ax.text(0.32, 0.205, r"conflict check", ha="center")
    ax.text(0.32, 0.135, r"vertex / edge at $(v,t)$", ha="center")

    arr(ax, (0.15, 0.76), (0.15, 0.61))
    ax.text(0.165, 0.685, r"plan($i$, $C_i$, $\bar{D}$)", fontsize=6.4, va="center")
    arr(ax, (0.40, 0.61), (0.40, 0.76), ls=DASH)
    ax.text(0.412, 0.685, r"$\pi_i$ / $\bot$", fontsize=6.4, va="center")
    arr(ax, (0.32, 0.42), (0.32, 0.27))
    ax.text(0.335, 0.345, r"$\{\pi_i\}$", fontsize=6.4, va="center")

    # 冲突 → 分裂：走框外右侧，避免穿过方框
    ax.plot([0.61, 0.72], [0.175, 0.175], color="black", lw=LW, ls=DASH)
    ax.plot([0.72, 0.72], [0.175, 0.855], color="black", lw=LW, ls=DASH)
    arr(ax, (0.72, 0.855), (0.61, 0.855), ls=DASH)
    ax.text(0.735, 0.515, "split", fontsize=6.4, rotation=90, va="center")


def dominance(ax):
    ax.set_xlim(0, 1.18); ax.set_ylim(0, 1.26)
    ax.set_xlabel(r"dose $d$", fontsize=7)
    ax.set_ylabel(r"cost $g$", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    front = [(0.26, 0.60), (0.50, 0.42), (0.74, 0.26)]
    dom = [(0.14, 0.84), (0.32, 0.94), (0.62, 0.66), (0.86, 0.50)]
    for x, y in dom:
        ax.plot(x, y, "o", ms=3.2, mfc="white", mec="black", mew=0.7, zorder=3)
    for x, y in front:
        ax.plot(x, y, "o", ms=3.4, mfc="black", mec="black", mew=0.7, zorder=4)
    ax.step([p[0] for p in front] + [1.10], [p[1] for p in front] + [0.26],
            where="post", color="black", lw=0.9, ls=DASH, zorder=2)
    ax.axvline(0.98, color="black", lw=0.9, ls=":")
    ax.text(0.98, 1.17, r"budget $\bar{D}$", ha="center", fontsize=6.6)

    ax.annotate("dominated", xy=(0.32, 0.94), xytext=(0.44, 1.00),
                fontsize=6.4, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444",
                                shrinkA=1, shrinkB=3))
    ax.annotate("kept: Pareto labels", xy=(0.50, 0.42), xytext=(0.30, 0.14),
                fontsize=6.4, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444",
                                shrinkA=1, shrinkB=3))


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    architecture(a1)
    dominance(a2)
    a1.text(0.0, 1.03, "(a)", fontsize=7.6, fontweight="bold", va="bottom")
    a2.text(-0.10, 1.03, "(b)", fontsize=7.6, fontweight="bold", va="bottom",
            transform=a2.transAxes)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.15, wspace=0.22)
    p = os.path.join(OUT, "fig00_framework.png")
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("written:", p)


if __name__ == "__main__":
    main()
