"""公开 3D BAG 建筑数据接入（学位论文第 7 章「数据与场景」）。

数据源：3D BAG v2025.09.03，荷兰全国 LoD1.2/1.3/2.2 建筑模型，
由 TU Delft 3D geoinformation 与 3DGI 发布，许可 CC BY 4.0。
  - 官网 https://3dbag.nl/
  - WFS  https://data.3dbag.nl/api/BAG3D/wfs  (图层 BAG3D:lod12)
  - CRS  EPSG:28992 (RD New)，属性 b3_h_70p 为 70 分位屋顶高度(m)

本模块把官方 WFS 返回的 2D footprint + 高度栅格化为体素障碍，
输出 data/bag_voxels.json，运行时只读该 JSON（零第三方依赖、完全可复现）。
论文中须引用上述来源与 CC BY 4.0 许可，并记录 bbox 与查询 URL。

用法：
    python bag_data.py                # 用已缓存的 raw geojson 生成
    python bag_data.py --download     # 无缓存时先从 WFS 下载
    python bag_data.py --info         # 打印场景统计
"""
from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir():
    """体素缓存的查找顺序：环境变量 → 投稿包 ../data/bag → 工程内 data/。

    投稿包把数据统一放在 paper/data/ 下、脚本放在 paper/code/ 下，
    若不这样解析，脚本会去 code/data 找一个不存在的文件而触发联网重建。
    """
    env = os.environ.get("PAPER_DATA")
    cands = []
    if env:
        cands.append(os.path.join(env, "bag"))
    cands.append(os.path.join(HERE, "..", "data", "bag"))
    cands.append(os.path.join(HERE, "data"))
    for c in cands:
        if os.path.exists(os.path.join(c, "bag_voxels.json")):
            return os.path.abspath(c)
    return os.path.abspath(cands[-1])


DATA = _data_dir()
RAW = os.path.join(DATA, "3dbag_raw.geojson")
VOXELS = os.path.join(DATA, "bag_voxels.json")

WFS = "https://data.3dbag.nl/api/BAG3D/wfs"
LAYER = "BAG3D:lod12"
SOURCE = dict(
    dataset="3D BAG", version="v2025.09.03",
    publisher="TU Delft 3D geoinformation / 3DGI",
    license="CC BY 4.0", url="https://3dbag.nl/",
    wfs=WFS, layer=LAYER, crs="EPSG:28992",
    height_attribute="b3_h_70p",
    note="WFS serves 2D footprints with height attributes; 3D models available as CityJSON tiles.",
)

# 从下载到的 150m×150m 子区中取 100m×100m，与既有实验口径 20×20×5m 对齐
# 实验区（100 m × 100 m）。原为 (84300, 446800)-(84400, 446900)，位于 400 m 概览的
# 西南角，建筑稀疏：6 m 离障下瓶颈墙缺口被夹死，任务全落一侧。
# 北移 100 m 后建筑密度从 696 → 830 体素，6 m 下通道天然贯通（6/6 任务跨墙），
# 且 n=6..12 均可在 3 s 内解出，故改用此位置。
BBOX_RD = (84300.0, 446900.0, 84400.0, 447000.0)
GRID = dict(nx=20, ny=20, nz=6, cell_xy=5.0, cell_z=2.5)


def download_raw():
    """从 3D BAG 官方 WFS 取目标窗口外扩 50 m 的子区（含边界外建筑，避免边缘截断）。"""
    os.makedirs(DATA, exist_ok=True)
    # 必须由 BBOX_RD 推导：窗口曾整体北移 100 m，硬编码的旧坐标会漏掉北侧 50 m。
    pad = 50.0
    bbox = (BBOX_RD[0] - pad, BBOX_RD[1] - pad,
            BBOX_RD[2] + pad, BBOX_RD[3] + pad)
    url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeNames={LAYER}&outputFormat=application/json&count=1000"
           f"&bbox={','.join(map(str, bbox))},urn:ogc:def:crs:EPSG::28992")
    ctx = ssl._create_unverified_context()   # 官方站点证书链在部分环境不完整
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    with open(RAW, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    payload["_source"] = {**SOURCE, "query_url": url, "bbox_rd": list(bbox)}
    with open(os.path.join(DATA, "3dbag_source.json"), "w", encoding="utf-8") as f:
        json.dump(payload["_source"], f, ensure_ascii=False, indent=2)
    return payload, url


def _polylist(geom):
    t = geom.get("type")
    c = geom.get("coordinates", [])
    if t == "Polygon":
        return [c[0]] if c else []
    if t == "MultiPolygon":
        return [poly[0] for poly in c if poly]
    return []


def _pip(px, py, ring):
    """射线法判断点是否在多边形内（纯标准库）。"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > py) != (yj > py):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < xint:
                inside = not inside
        j = i
    return inside


def rasterize(payload, bbox=None):
    g = GRID
    x0, y0, _x1, _y1 = bbox or BBOX_RD
    blocked = set()
    buildings = []
    for feat in payload.get("features", []):
        props = feat.get("properties", {}) or {}
        h = props.get("b3_h_70p")
        if not isinstance(h, (int, float)) or not math.isfinite(h):
            continue
        rings = _polylist(feat.get("geometry", {}))
        if not rings:
            continue
        # 世界坐标(RD米) -> 栅格坐标(格)
        rings_grid = [[((px - x0) / g["cell_xy"], (py - y0) / g["cell_xy"])
                       for px, py in ring] for ring in rings]
        cells = []
        for iy in range(g["ny"]):
            for ix in range(g["nx"]):
                cx, cy = ix + 0.5, iy + 0.5
                if any(_pip(cx, cy, ring) for ring in rings_grid):
                    hv = min(g["nz"], max(1, math.ceil(float(h) / g["cell_z"])))
                    for z in range(hv):
                        blocked.add((ix, iy, z))
                    cells.append([ix, iy, hv])
        if cells:
            buildings.append(dict(id=props.get("identificatie"),
                                  height_m=round(float(h), 3),
                                  height_voxels=max(c[2] for c in cells),
                                  footprint_cells=len(cells)))
    return blocked, buildings


def build(save=True):
    if os.path.exists(RAW):
        payload = json.load(open(RAW, encoding="utf-8"))
        url = f"{WFS}?... cached from data/3dbag_raw.geojson"
    else:
        payload, url = download_raw()
    blocked, buildings = rasterize(payload)
    out = dict(
        source={**SOURCE, "query_url": url, "bbox_rd": list(BBOX_RD)},
        grid=GRID,
        blocked=sorted([list(v) for v in blocked]),
        buildings=buildings,
        stats=dict(n_features=len(payload.get("features", [])),
                   n_buildings=len(buildings),
                   n_blocked_voxels=len(blocked),
                   total_voxels=GRID["nx"] * GRID["ny"] * GRID["nz"],
                   blocked_ratio=round(len(blocked) /
                                       (GRID["nx"] * GRID["ny"] * GRID["nz"]), 4)),
    )
    if save:
        os.makedirs(DATA, exist_ok=True)
        with open(VOXELS, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def build_tile(bbox, tag):
    """从官方 WFS 取另一个 100m×100m 子区并栅格化（多 tile 外部效度实验用）。"""
    url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeNames={LAYER}&outputFormat=application/json&count=1000"
           f"&bbox={','.join(map(str, bbox))},urn:ogc:def:crs:EPSG::28992")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    blocked, buildings = rasterize(payload, bbox)
    meta = dict(source={**SOURCE, "query_url": url, "bbox_rd": list(bbox),
                        "tile_tag": tag},
                grid=GRID, buildings=buildings,
                stats=dict(n_features=len(payload.get("features", [])),
                           n_buildings=len(buildings),
                           n_blocked_voxels=len(blocked),
                           blocked_ratio=round(len(blocked) / 2400, 4)))
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"bag_tile_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(meta, blocked=sorted([list(v) for v in blocked])),
                  f, ensure_ascii=False, indent=2)
    return set(blocked), meta


def load_layout(path=VOXELS):
    """运行时读取：返回 (blocked_voxels:set, metadata:dict)。零第三方依赖。"""
    if not os.path.exists(path):
        build()
    j = json.load(open(path, encoding="utf-8"))
    return {tuple(v) for v in j["blocked"]}, j


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--info", action="store_true")
    a = p.parse_args()
    if a.download and not os.path.exists(RAW):
        download_raw()
    out = build(save=not a.info)
    print(json.dumps(out["stats"], ensure_ascii=False, indent=2))
    print("bbox_rd", BBOX_RD, "grid", GRID)
    if not a.info:
        print("saved", VOXELS)


if __name__ == "__main__":
    main()
