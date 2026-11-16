"""按真实建筑轮廓绘制等距 3D 城区场景（论文图 1 风格）。

与 make_figures.fig02_3d 的旧实现区别：不再逐体素画小立方体（会看起来像
一堵蓝色阶梯墙），而是把每个建筑的 footprint 拉成一个棱柱，按高度着色，
并叠加地面网格、禁飞墙、应急走廊、任务起终点、烟源、风向、图例与图题。

两个底图：
  - 3D BAG v2025.09.03（官方 WFS，荷兰代尔夫特，主实验底图）
  - 北京 600m 子区 shapefile（工作区已有，600m×600m / 154 栋真实建筑）

输出：figures/fig02_scene_3d.png、figures/fig02b_beijing_3d.png
"""
from __future__ import annotations

import json
import math
import os

# 语言开关：FIG_LANG=en 时输出英文图（投英文期刊用）；默认 zh 保持原图册不变
LANG = os.environ.get("FIG_LANG", "zh")


def T(zh, en):
    return en if LANG == "en" else zh
import ssl
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
BAG_RAW = os.path.join(HERE, "data", "3dbag_raw.geojson")
def _bag_overview_path():
    """3D BAG 鸟瞰底图缓存。优先用投稿包内的 ../data/bag，其次工程内 data/。

    投稿包把数据统一放在 paper/data/ 下，脚本在 paper/code/ 里运行；
    若不这样解析，脚本会在 code/ 下再生成一份重复缓存。
    """
    env = os.environ.get("PAPER_DATA")
    cands = []
    if env:
        cands.append(os.path.join(env, "bag", "3dbag_overview.geojson"))
    cands.append(os.path.join(HERE, "..", "data", "bag",
                              "3dbag_overview.geojson"))
    cands.append(os.path.join(HERE, "data", "3dbag_overview.geojson"))
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(cands[-1])


BAG_OVERVIEW = _bag_overview_path()
BAG_WFS = "https://data.3dbag.nl/api/BAG3D/wfs"
# 400m×400m 概览区：只用于三维鸟瞰示意，主实验仍取其中 100m×100m 子区
OVERVIEW_BBOX = (84300, 446800, 84700, 447200)
BEIJING_SHP = os.path.join(ROOT, "600m_MAPF_project_20261103",
                           "beijing_selected_600m.shp")


def _ensure_overview():
    """取更大范围的 3D BAG 建筑用于鸟瞰图（带离线缓存，可复现）。"""
    if os.path.exists(BAG_OVERVIEW):
        return
    url = (f"{BAG_WFS}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeNames=BAG3D:lod12&outputFormat=application/json&count=3000"
           f"&bbox={','.join(map(str, OVERVIEW_BBOX))},"
           f"urn:ogc:def:crs:EPSG::28992")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    os.makedirs(os.path.dirname(BAG_OVERVIEW), exist_ok=True)
    with open(BAG_OVERVIEW, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def prism_faces(poly, h):
    """把平面多边形拉成棱柱，返回 6 面顶点列表。"""
    n = len(poly)
    if n < 3:
        return []
    bot = [(x, y, 0.0) for x, y in poly]
    top = [(x, y, h) for x, y in poly]
    faces = [bot, top]
    for i in range(n):
        j = (i + 1) % n
        faces.append([bot[i], bot[j], top[j], top[i]])
    return faces


def box_faces(x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return [[v[k] for k in q] for q in idx]


def _polygons_from_geojson(path, x0, y0, scale, height_key, max_layers):
    j = json.load(open(path, encoding="utf-8"))
    out = []
    for f in j.get("features", []):
        p = f.get("properties", {}) or {}
        h = p.get(height_key)
        if not isinstance(h, (int, float)) or not math.isfinite(h):
            continue
        g = f.get("geometry", {})
        c = g.get("coordinates", [])
        rings = [c[0]] if g.get("type") == "Polygon" and c else (
            [poly[0] for poly in c if poly]
            if g.get("type") == "MultiPolygon" else [])
        for ring in rings:
            poly = [((px - x0) / scale, (py - y0) / scale) for px, py in ring]
            if len(poly) >= 3:
                out.append((poly, min(max_layers, max(1, h / 2.5))))
    return out


def _polygons_from_shp(path, scale=10.0, max_layers=14):
    import geopandas as gpd
    g = gpd.read_file(path).to_crs(3857)
    minx, miny, _mx, _my = g.total_bounds
    out = []
    for row in g.itertuples():
        geom = getattr(row, "geometry", None)
        h = getattr(row, "Height", None)
        if geom is None or geom.is_empty or h is None:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in polys:
            ring = list(p.exterior.coords)
            poly = [((px - minx) / scale, (py - miny) / scale)
                    for px, py in ring]
            if len(poly) >= 3:
                out.append((poly, min(max_layers, max(1, float(h) / 2.5))))
    return out


def _draw_buildings(ax, prisms):
    if not prisms:
        return
    hmax = max(h for _, h in prisms)
    for poly, h in prisms:
        # 浅蓝灰（接近常见论文场景图），避免深蓝把建筑压成一坨
        col = plt.cm.Blues(0.16 + 0.30 * (h / max(1.0, hmax)))
        pc = Poly3DCollection(prism_faces(poly, h),
                              facecolors=[col] * (len(poly) + 2),
                              edgecolors="#8395ab", linewidths=0.18,
                              shade=True, alpha=0.96)
        ax.add_collection3d(pc)


def _draw_ground(ax, W, H, step):
    for x in range(0, int(W) + 1, step):
        ax.plot([x, x], [0, H], [0, 0], color="#dfe5ee", lw=0.4, zorder=0)
    for y in range(0, int(H) + 1, step):
        ax.plot([0, W], [y, y], [0, 0], color="#dfe5ee", lw=0.4, zorder=0)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], [0, 0, 0, 0, 0],
            color="#b9c4d2", lw=0.9)


def _draw_overlays(ax, W, H, zmax, wall_x, gap, corridor_y,
                   starts, goals, smokes, wind_note):
    # 应急走廊带（地面高亮）
    y0, y1 = corridor_y
    ax.add_collection3d(Poly3DCollection(
        [[(0, y0, 0.02), (W, y0, 0.02), (W, y1, 0.02), (0, y1, 0.02)]],
        facecolors="#ffe9a8", edgecolors="none", alpha=0.5))
    # 坍塌封路禁飞墙（中间留闸门缺口）
    gl, gh = gap
    for yy0, yy1 in ((0, gl), (gh, H)):
        if yy1 - yy0 < 0.3:
            continue
        ax.add_collection3d(Poly3DCollection(
            box_faces(wall_x - 0.5, wall_x + 0.5, yy0, yy1, 0, zmax * 0.92),
            facecolors="#d84a4a", edgecolors="#b03636", linewidths=0.2,
            alpha=0.88, shade=True))
    ax.text(wall_x, (gl + gh) / 2, zmax * 0.98,
            T("闸门", "Gate"), color="#8d2b2b", fontsize=8, ha="center")
    # 任务起终点
    if starts:
        ax.scatter([p[0] for p in starts], [p[1] for p in starts],
                   [0.9] * len(starts), marker="s", s=75,
                   color="#2f6fd0", edgecolors="white", linewidths=0.6,
                   depthshade=False, zorder=8)
    if goals:
        ax.scatter([p[0] for p in goals], [p[1] for p in goals],
                   [0.9] * len(goals), marker="s", s=75,
                   color="#2fa15e", edgecolors="white", linewidths=0.6,
                   depthshade=False, zorder=8)
    # 烟源
    if smokes:
        ax.scatter([p[0] for p in smokes], [p[1] for p in smokes],
                   [0.9] * len(smokes), marker="o", s=60,
                   color="#c8322f", edgecolors="white", linewidths=0.6,
                   depthshade=False, zorder=9)
    # 风向
    ax.quiver(W * 0.12, H * 0.86, zmax * 1.05, W * 0.16, -H * 0.06, 0,
              color="#4a6fa5", linewidth=1.6, arrow_length_ratio=0.35)
    ax.text(W * 0.12, H * 0.86, zmax * 1.12, wind_note,
            color="#3c5f92", fontsize=8)


def _legend(fig, clean=False):
    if clean:
        handles = [
            Patch(facecolor="#9db4d0", edgecolor="#8395ab",
                  label=T("建筑（按高度着色）", "Buildings (height-shaded)")),
            plt.Line2D([], [], color="#c0392b", lw=1.8, ls="--",
                       label=T("论文主实验取样子区 100 m×100 m", "Study area, 100 m × 100 m")),
        ]
        ncol = 1
    else:
        handles = [
            Patch(facecolor="#9db4d0", edgecolor="#8395ab",
                  label=T("建筑（按高度着色）", "Buildings (height-shaded)")),
            Patch(facecolor="#d84a4a", edgecolor="#b03636",
                  label=T("禁飞墙（瓶颈）", "No-fly wall (bottleneck)")),
            Patch(facecolor="#ffe9a8", edgecolor="none", label=T("应急走廊带", "Emergency corridor")),
            Patch(facecolor="#2f6fd0", label=T("任务起点", "Task start")),
            Patch(facecolor="#2fa15e", label=T("任务终点", "Task goal")),
            plt.Line2D([], [], marker="o", color="w",
                       markerfacecolor="#c8322f", markersize=6,
                       label=T("烟源（离地）", "Smoke source (airborne)")),
        ]
        ncol = 2
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.94, 0.90), fontsize=8, frameon=True,
               framealpha=0.95, ncol=ncol)


def _render(prisms, W, H, zmax, wall_x, gap, corridor_y,
            starts, goals, smokes, title, caption, out_path, step=5,
            overlays=True, sample_box=None, elev=32, zaspect=0.42):
    fig = plt.figure(figsize=(10.5, 8.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    _draw_ground(ax, W, H, step)
    _draw_buildings(ax, prisms)
    if overlays:
        _draw_overlays(ax, W, H, zmax, wall_x, gap, corridor_y,
                       starts, goals, smokes, T("风向 +x, 4 m/s", "Wind +x, 4 m/s"))
    if sample_box:
        x0, y0, x1, y1 = sample_box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                [0.06] * 5, color="#c0392b", lw=2.0, ls="--")
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, 0.06,
                T("主实验子区", "Study area"), color="#c0392b", fontsize=8, ha="center")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, zmax * 1.2)
    ax.set_box_aspect((1.0, 1.0, zaspect))
    ax.view_init(elev=elev, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, loc="left", pad=2, color="#22304a")
    _legend(fig, clean=not overlays)
    fig.text(0.5, 0.045, caption, ha="center", fontsize=15, color="#1b1b1b")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


def render_bag(scene=None, out_path=None, title=None, caption=None):
    """公开 3D BAG 真实建筑**鸟瞰图**（图 2 附图）。

    旧版把 100m×100m / 18 栋楼的子区硬套「论文图 1」模板，导致棱柱连成
    几大块、两块红色禁飞墙挡住视线、黄走廊与散点叠在一起，构图混乱。
    这里改为 400m×400m 的真实城区鸟瞰，只标出主实验取样的 100m 子区，
    **不叠加任何灾害要素**；灾害与任务要素由 fig01（俯视主实验场景）承担。
    """
    _ensure_overview()
    prisms = _polygons_from_geojson(BAG_OVERVIEW, 84300.0, 446800.0, 10.0,
                                    "b3_h_70p", 12)
    W = H = 40
    pad = 2.0
    prisms = [(p, h) for p, h in prisms
              if all(-pad <= x <= W + pad and -pad <= y <= H + pad
                     for x, y in p)]
    _title = title if title is not None else T("公开 3D BAG 真实建筑三维鸟瞰"
                     "（代尔夫特 400 m×400 m；红虚线框＝图 1 俯视图的 100 m 实验子区）",
                     "3D bird's-eye view of public 3D BAG buildings "
                     "(Delft, 400 m x 400 m; dashed red box = the 100 m study area of Fig. 1)")
    _cap = caption if caption is not None else T("图 2：公开 3D BAG 建筑数据与论文取样子区（红框即图 1 场景）",
                     "Fig. 2. Public 3D BAG building stock and the study area "
                     "(red box = scene of Fig. 1)")
    return _render(prisms, W, H, 12, 0, (0, 0), (0, 0),
                   [], [], [], _title, _cap,
                   out_path, step=5, overlays=False,
                   sample_box=(0, 10, 10, 20), elev=38, zaspect=0.30)


def render_beijing(out_path):
    """工作区已有的北京 600m 子区真实建筑 shapefile。"""
    prisms = _polygons_from_shp(BEIJING_SHP, scale=10.0, max_layers=14)
    W = H = 60
    # 应急走廊双向任务：左侧 18 个起点、右侧 18 个终点（示意）
    starts = [(4 + 0.7 * (i % 6), 24 + 1.6 * (i // 6)) for i in range(18)]
    goals = [(54 - 0.7 * (i % 6), 27 + 1.6 * (i // 6)) for i in range(18)]
    smokes = [(18, 30), (30, 34), (46, 30)]
    return _render(prisms, W, H, 14, W // 2, (26, 34), (22, 38),
                   starts, goals, smokes,
                   "另一组 GIS 数据的场景示意（北京 600 m 子区实测建筑，非本文实验底图）",
                   "附图 A：北京 600 m 子区实测建筑（另一组数据，不参与本文实验）",
                   out_path, step=5)


if __name__ == "__main__":
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    sp = os.path.join(HERE, "results", "bag", "scene_and_paths.json")
    s = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    print(render_bag(s, os.path.join(out, "fig02_scene_3d.png")))
    print(render_beijing(os.path.join(out, "fig02b_beijing_3d.png")))
