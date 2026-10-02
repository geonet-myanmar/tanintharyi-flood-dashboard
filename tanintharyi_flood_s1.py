#!/usr/bin/env python3
"""
Sentinel-1 flood inundation, seven Tanintharyi townships, 20 Sep -> 2 Oct 2026.

Imagery from the Copernicus Data Space Ecosystem (Sentinel Hub Process API),
analysed locally, summarised against MIMU townships, village tracts, villages
and towns, and written as a hand-built Leaflet dashboard (index.html + assets/).

WHY NOT flood_mapping_cdse.py AS-IS
  That script holds every layer of the AOI grid in memory at once.  Here the
  AOI bbox at Sentinel-1's native 10 m is ~21,200 x 50,600 px = 1.07 Gpx per
  layer -- a single float32 layer is 4.3 GB, and the chain needs about ten of
  them.  This driver keeps the same method and the same thresholds but works
  block by block on the 2048 px Sentinel Hub tile grid (with a halo for the
  speckle filter), streaming every product into tiled GeoTIFFs on disk.
  Peak memory is a few hundred MB.

WHAT IS SHARED (imported, not copied)
  flood_mapping.py  -- MIMU loaders, palettes, PNG chunking, the dashboard
  cdse_source.py    -- CDSE token, catalogue query, Sentinel Hub tile request

PAIR
  2026-09-20 and 2026-10-02 are both Sentinel-1D ASCENDING relative orbit 70,
  4 IW GRDH frames each, same pass time (11:35 UTC), 12 days = one repeat
  cycle.  Same track, same geometry: a valid change-detection pair.

USAGE
    python tanintharyi_flood_s1.py             # full run (resumes from caches)
    python tanintharyi_flood_s1.py --check     # catalogue check only
    python tanintharyi_flood_s1.py --page-only # re-render index.html only
"""

import argparse
import json
import math
import os
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import ee
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import box as shp_box
from shapely.geometry import mapping, shape
from shapely.prepared import prep

import cdse_source as CD
import flood_mapping as F

# ── configuration ────────────────────────────────────────────────────────────
F.AOI_GEOJSON = "townships.geojson"
F.PRE_FLOOD_DATE = "2026-09-20"
F.POST_FLOOD_DATE = "2026-10-02"
F.ORBIT_PASS = "ASCENDING"
F.REL_ORBIT = 70

FLOOD_DB_THRESH = F.FLOOD_DB_THRESH       # -3 dB
SPECKLE_RADIUS_M = F.SPECKLE_RADIUS       # 30 m
MAX_SLOPE_DEG = F.MAX_SLOPE_DEG           # 5 deg
R_PX = max(1, int(round(SPECKLE_RADIUS_M / 10.0)))
HALO = R_PX

# Static masks are fetched from Earth Engine on the same 10 m grid as the SAR
# (F.EXPORT_SCALE), never coarser: every input to the classification and every
# statistic is computed at Sentinel-1's native spacing.
MASK_SCALE = F.EXPORT_SCALE

CDSE_CACHE = "cdse_cache"                 # metered imagery: never auto-deleted
WORK_DIR = "work"                         # 10 m products (GeoTIFF)
FETCH_WORKERS = 4

# MIMU village tracts (adm4) for Tanintharyi -- the unit below township that
# local responders actually work in.  Summarised from the same 10 m flags.
VT_LAYER = "mmr_tni_polbnda_adm4_unhcr_mimu_250k"

# flag bits in work/flags.tif
B_VALID, B_FLOOD, B_WATER, B_LAND, B_AOI = 1, 2, 4, 8, 16


# ── grid ─────────────────────────────────────────────────────────────────────

def tiles_for(aoi_shape, width, height, bbox):
    """(r, c, y0, y1, x0, x1) for every 2048 px tile that touches the AOI."""
    T = CD.SH_TILE_PX
    pad = 0.002                    # ~220 m, so the speckle halo is always fetched
    pa = prep(aoi_shape.buffer(pad))
    out = []
    for r in range(math.ceil(height / T)):
        for c in range(math.ceil(width / T)):
            y0, x0 = r * T, c * T
            y1, x1 = min(height, y0 + T), min(width, x0 + T)
            w = bbox["west"] + x0 * CD.DEG_PER_10M
            e = bbox["west"] + x1 * CD.DEG_PER_10M
            n = bbox["north"] - y0 * CD.DEG_PER_10M
            s = bbox["north"] - y1 * CD.DEG_PER_10M
            if pa.intersects(shp_box(w, s, e, n)):
                out.append((r, c, y0, y1, x0, x1))
    return out


def tile_path(date, r, c):
    return os.path.join(CDSE_CACHE, date, f"r{r:02d}_c{c:02d}.tif")


# ── catalogue ────────────────────────────────────────────────────────────────

def catalogue(date, bbox):
    """GRDH frames on one UTC day with direction and relative orbit."""
    poly = (f"POLYGON(({bbox['west']} {bbox['south']},{bbox['east']} "
            f"{bbox['south']},{bbox['east']} {bbox['north']},{bbox['west']} "
            f"{bbox['north']},{bbox['west']} {bbox['south']}))")
    flt = (f"Collection/Name eq 'SENTINEL-1' and contains(Name,'_IW_GRDH') and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
           f"ContentDate/Start gt {date}T00:00:00.000Z and "
           f"ContentDate/Start lt {date}T23:59:59.999Z")
    url = CD.CATALOGUE_URL + "?" + urllib.parse.urlencode(
        {"$filter": flt, "$top": "50", "$expand": "Attributes"})
    with urllib.request.urlopen(url, timeout=120) as r:
        vals = json.load(r)["value"]
    out, seen = [], set()
    for v in vals:
        if "_COG" in v["Name"]:
            continue
        a = {x["Name"]: x["Value"] for x in v.get("Attributes", [])}
        start = v["ContentDate"]["Start"]
        if start in seen:
            continue
        seen.add(start)
        out.append({"name": v["Name"], "start": start,
                    "dir": a.get("orbitDirection"),
                    "rel": a.get("relativeOrbitNumber"),
                    "platform": v["Name"][:3], "online": v.get("Online")})
    return sorted(out, key=lambda d: d["start"])


# ── imagery ──────────────────────────────────────────────────────────────────

def fetch_tile(date, t, bbox):
    r, c, y0, y1, x0, x1 = t
    path = tile_path(date, r, c)
    if os.path.exists(path):
        return path, False
    tb = [bbox["west"] + x0 * CD.DEG_PER_10M,
          bbox["north"] - y1 * CD.DEG_PER_10M,
          bbox["west"] + x1 * CD.DEG_PER_10M,
          bbox["north"] - y0 * CD.DEG_PER_10M]
    arr = CD._request_tile(date, tb, x1 - x0, y1 - y0, F.ORBIT_PASS)
    # One band: linear gamma0 VV, 0 where Sentinel Hub's dataMask says no data.
    vv = np.where(arr[1] > 0.5, arr[0], 0.0).astype(np.float32)
    prof = {"driver": "GTiff", "height": y1 - y0, "width": x1 - x0, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326",
            "transform": from_origin(tb[0], tb[3], CD.DEG_PER_10M,
                                     CD.DEG_PER_10M),
            "compress": "deflate", "predictor": 3, "tiled": True,
            "blockxsize": 512, "blockysize": 512}
    tmp = path + ".part"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(vv, 1)
    os.replace(tmp, path)
    return path, True


def fetch_all(tiles, bbox):
    CD.token()                         # mint once before the threads start
    for date in (F.PRE_FLOOD_DATE, F.POST_FLOOD_DATE):
        os.makedirs(os.path.join(CDSE_CACHE, date), exist_ok=True)
        todo = [t for t in tiles if not os.path.exists(tile_path(date, t[0], t[1]))]
        print(f"  {date}: {len(tiles)} tile(s), {len(tiles) - len(todo)} cached, "
              f"{len(todo)} to fetch")
        t0, done = time.time(), 0
        with ThreadPoolExecutor(FETCH_WORKERS) as ex:
            futs = {ex.submit(fetch_tile, date, t, bbox): t for t in todo}
            for f in as_completed(futs):
                f.result()
                done += 1
                if done % 10 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"    {done}/{len(todo)}  {el / 60:.1f} min elapsed, "
                          f"~{el / done * (len(todo) - done) / 60:.1f} min left",
                          flush=True)


def read_region(date, tile_index, ya, yb, xa, xb):
    """Linear VV for grid rows ya:yb, cols xa:xb, assembled from cached tiles."""
    T = CD.SH_TILE_PX
    out = np.zeros((yb - ya, xb - xa), np.float32)
    for r in range(ya // T, (yb - 1) // T + 1):
        for c in range(xa // T, (xb - 1) // T + 1):
            if (r, c) not in tile_index:
                continue
            _, _, y0, y1, x0, x1 = tile_index[(r, c)]
            iy0, iy1 = max(ya, y0), min(yb, y1)
            ix0, ix1 = max(xa, x0), min(xb, x1)
            if iy0 >= iy1 or ix0 >= ix1:
                continue
            with rasterio.open(tile_path(date, r, c)) as ds:
                out[iy0 - ya:iy1 - ya, ix0 - xa:ix1 - xa] = ds.read(
                    1, window=Window(ix0 - x0, iy0 - y0, ix1 - ix0, iy1 - iy0))
    return out


# ── processing helpers (same maths as flood_mapping_cdse.py) ────────────────

def disk(radius_px):
    r = int(radius_px)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def focal_mean(a, valid, radius_px):
    k = disk(radius_px).astype(np.float32)
    filled = np.where(valid, a, 0.0).astype(np.float32)
    s = ndimage.convolve(filled, k, mode="constant", cval=0.0)
    n = ndimage.convolve(valid.astype(np.float32), k, mode="constant", cval=0.0)
    out = np.full(a.shape, np.nan, np.float32)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    return out


def to_db(vv):
    ok = vv > 0
    db = np.full(vv.shape, np.nan, np.float32)
    db[ok] = 10.0 * np.log10(vv[ok])
    return db, ok


def sar_byte(db, ok):
    v = np.clip((db + 25.0) / 25.0, 0, 1)
    out = (v * 254 + 1).astype(np.uint8)
    out[~ok | ~np.isfinite(db)] = 0
    return out


def gee_masks(bbox, aoi_shape):
    """JRC permanent water and the HydroSHEDS terrain class, 10 m, on disk."""
    rect = ee.Geometry.Rectangle([bbox["west"], bbox["south"],
                                  bbox["east"], bbox["north"]])
    # CLIP BEFORE UNMASK (see flood_mapping_cdse.py): unmask() on the derived
    # projection of ee.Terrain.slope() otherwise yields an empty image.
    water_img = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality")
                 .gte(4).clip(rect).unmask(0).toByte())
    slope = ee.Terrain.slope(ee.Image("WWF/HydroSHEDS/03VFDEM"))
    # 0 = no DEM (sea), 1 = land too steep to hold water, 2 = flat land.
    terr_img = slope.lt(MAX_SLOPE_DEG).add(1).clip(rect).unmask(0).toByte()
    out = []
    for img, name in ((water_img, "water10.tif"), (terr_img, "terr10.tif")):
        dst = os.path.join(WORK_DIR, name)
        if not os.path.exists(dst):
            stream_mask(img, name, aoi_shape, bbox, dst)
        out.append(dst)
    return out


def stream_mask(image, name, aoi_shape, bbox, dst):
    """
    Download a uint8 GEE mask in tiles and mosaic it onto the 10 m analysis
    grid ONE TILE AT A TIME.

    F.download_as_geotiff merges with rasterio.merge, which holds the whole
    mosaic in memory -- 1.07 Gpx here, plus its mask -- and on 2026-10-02 that
    drove this 16 GB machine into memory exhaustion. Tile names and order match
    F.download_as_geotiff, so its cached tiles are reused.
    """
    os.makedirs(F.TEMP_DIR, exist_ok=True)
    tile_deg = F.tile_deg_for(MASK_SCALE)
    defs = []
    lat = bbox["south"]
    while lat < bbox["north"]:
        lat_end = min(lat + tile_deg, bbox["north"])
        lon = bbox["west"]
        while lon < bbox["east"]:
            lon_end = min(lon + tile_deg, bbox["east"])
            if shp_box(lon, lat, lon_end, lat_end).intersects(aoi_shape):
                defs.append((lon, lat, lon_end, lat_end))
            lon = lon_end
        lat = lat_end
    print(f"  {name}: {len(defs)} tile(s) at {MASK_SCALE} m")
    width, height, transform = CD.grid_for(bbox)
    prof = {"driver": "GTiff", "height": height, "width": width, "count": 1,
            "dtype": "uint8", "crs": "EPSG:4326", "transform": transform,
            "compress": "deflate", "tiled": True, "blockxsize": 512,
            "blockysize": 512, "BIGTIFF": "IF_SAFER"}
    with rasterio.open(dst + ".part", "w+", **prof) as out:
        for i, (w, s, e, n) in enumerate(defs, 1):
            tp = os.path.join(F.TEMP_DIR, f"_tile_{i:03d}_{name}")
            if not os.path.exists(tp):
                region = ee.Geometry.Rectangle([w, s, e, n])
                if not F._download_tile(image.clip(region), tp, MASK_SCALE, region):
                    raise RuntimeError(f"{name} tile {i} failed; re-run resumes")
            with rasterio.open(tp) as src:
                b = src.bounds
                win = out.window(b.left, b.bottom, b.right, b.top)
                win = win.round_offsets().round_lengths().intersection(
                    Window(0, 0, width, height))
                cur = out.read(1, window=win)
                new = np.zeros_like(cur)
                reproject(source=rasterio.band(src, 1), destination=new,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=out.window_transform(win),
                          dst_crs="EPSG:4326", resampling=Resampling.nearest)
                out.write(np.maximum(cur, new), 1, window=win)
            if i % 10 == 0 or i == len(defs):
                print(f"    mosaicked {i}/{len(defs)}", flush=True)
    os.replace(dst + ".part", dst)


def mask_on_block(ds, transform, h, w):
    """The 10 m GEE mask on this block's grid (nearest; a pure re-index when
    the two grids coincide, which they do to within float rounding)."""
    west, north = transform.c, transform.f
    east, south = west + w * transform.a, north + h * transform.e
    win = ds.window(west, south, east, north).round_offsets().round_lengths()
    win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
    arr = ds.read(1, window=win, boundless=True, fill_value=0)
    out = np.zeros((h, w), np.uint8)
    reproject(source=arr, destination=out,
              src_transform=ds.window_transform(win), src_crs="EPSG:4326",
              dst_transform=transform, dst_crs="EPSG:4326",
              resampling=Resampling.nearest)
    return out


# ── main analysis ────────────────────────────────────────────────────────────

def analyse(aoi_info, townships, tiles, width, height, transform, bbox):
    """Block-wise change detection; returns AOI and per-township sums."""
    tile_index = {(t[0], t[1]): t for t in tiles}
    wpath, tpath = gee_masks(bbox, aoi_info["shape"])
    masks = {"water": rasterio.open(wpath), "terr": rasterio.open(tpath)}
    for k, ds in masks.items():
        print(f"    {k} mask {ds.width:,} x {ds.height:,} px, "
              f"{abs(ds.transform.a) * 111319.49:.2f} m")
    ov = masks["terr"].read(1, out_shape=(masks["terr"].height // 64,
                                          masks["terr"].width // 64))
    if ov.max() < 2:
        raise SystemExit("[STOP] Terrain mask has no flat land -- an EE "
                         "expression was probably unmasked before it was clipped.")

    shp = aoi_info["shape"]
    aoi_parts = list(shp.geoms) if shp.geom_type == "MultiPolygon" else [shp]
    ts_geoms = [shape(t["geom"]) for t in townships]

    prof = {"driver": "GTiff", "height": height, "width": width, "count": 1,
            "dtype": "uint8", "crs": "EPSG:4326", "transform": transform,
            "compress": "deflate", "tiled": True, "blockxsize": 512,
            "blockysize": 512, "BIGTIFF": "IF_SAFER"}
    names = ["flags", "flood", "water", "pre_sar", "post_sar"]
    outs = {n: rasterio.open(os.path.join(WORK_DIR, f"{n}.tif.part"), "w", **prof)
            for n in names}

    lat = bbox["north"] - (np.arange(height) + 0.5) * CD.DEG_PER_10M
    m_side = CD.DEG_PER_10M * 111_319.49
    px_area = (m_side * m_side * np.cos(np.radians(lat))).astype(np.float64)

    keys = ("area", "flood", "valid", "land", "water")
    tot = dict.fromkeys(keys, 0.0)
    per_ts = [dict.fromkeys(keys, 0.0) for _ in townships]
    hist = np.zeros(161, np.int64)                    # diff -20..+20 dB, 0.25 step

    t0 = time.time()
    for i, (r, c, y0, y1, x0, x1) in enumerate(tiles, 1):
        h, w = y1 - y0, x1 - x0
        ya, yb = max(0, y0 - HALO), min(height, y1 + HALO)
        xa, xb = max(0, x0 - HALO), min(width, x1 + HALO)
        core = (slice(y0 - ya, y0 - ya + h), slice(x0 - xa, x0 - xa + w))

        pre_db, pre_ok = to_db(read_region(F.PRE_FLOOD_DATE, tile_index, ya, yb, xa, xb))
        post_db, post_ok = to_db(read_region(F.POST_FLOOD_DATE, tile_index, ya, yb, xa, xb))
        pre_s = focal_mean(pre_db, pre_ok, R_PX)[core]
        post_s = focal_mean(post_db, post_ok, R_PX)[core]
        pre_ok, post_ok = pre_ok[core], post_ok[core]

        btf = rasterio.transform.from_origin(
            bbox["west"] + x0 * CD.DEG_PER_10M, bbox["north"] - y0 * CD.DEG_PER_10M,
            CD.DEG_PER_10M, CD.DEG_PER_10M)
        aoi = rasterize([(g, 1) for g in aoi_parts], out_shape=(h, w),
                        transform=btf, fill=0, dtype="uint8").astype(bool)
        water = (mask_on_block(masks["water"], btf, h, w) > 0) & aoi
        terr = mask_on_block(masks["terr"], btf, h, w)
        land, flat = (terr >= 1) & aoi, (terr == 2)

        valid = pre_ok & post_ok & land
        with np.errstate(invalid="ignore"):
            diff = np.where(valid, post_s - pre_s, np.nan)
        flood = np.zeros((h, w), bool)
        np.less(diff, FLOOD_DB_THRESH, out=flood, where=np.isfinite(diff))
        flood &= ~water & flat & valid

        fin = diff[np.isfinite(diff) & flat & ~water]
        if fin.size:
            hist += np.bincount(np.clip(((fin + 20) * 4).astype(int), 0, 160),
                                minlength=161)

        area = np.broadcast_to(px_area[y0:y1, None], (h, w))
        for k, m in (("area", aoi), ("flood", flood), ("valid", valid),
                     ("land", land), ("water", water)):
            tot[k] += float(area[m].sum())

        bb = shp_box(*rasterio.transform.array_bounds(h, w, btf))
        shapes = [(g, j + 1) for j, g in enumerate(ts_geoms) if g.intersects(bb)]
        if shapes:
            tid = rasterize(shapes, out_shape=(h, w), transform=btf, fill=0,
                            dtype="uint8")
            tid[~aoi] = 0              # clip every statistics band to the AOI
            for k, m in (("area", aoi), ("flood", flood), ("valid", valid),
                         ("land", land), ("water", water)):
                s = np.bincount(tid[m], weights=area[m],
                                minlength=len(townships) + 1)
                for j in range(len(townships)):
                    per_ts[j][k] += float(s[j + 1])

        flags = (valid * B_VALID | flood * B_FLOOD | water * B_WATER
                 | land * B_LAND | aoi * B_AOI).astype(np.uint8)
        win = Window(x0, y0, w, h)
        outs["flags"].write(flags, 1, window=win)
        outs["flood"].write(flood.astype(np.uint8), 1, window=win)
        outs["water"].write(water.astype(np.uint8), 1, window=win)
        outs["pre_sar"].write(sar_byte(pre_s, pre_ok & aoi), 1, window=win)
        outs["post_sar"].write(sar_byte(post_s, post_ok & aoi), 1, window=win)

        if i % 10 == 0 or i == len(tiles):
            el = time.time() - t0
            print(f"    block {i}/{len(tiles)}  flood so far "
                  f"{tot['flood'] / 1e6:,.1f} km2  ({el / 60:.1f} min)", flush=True)

    for ds in masks.values():
        ds.close()
    for n, ds in outs.items():
        ds.close()
        os.replace(os.path.join(WORK_DIR, f"{n}.tif.part"),
                   os.path.join(WORK_DIR, f"{n}.tif"))
    a = {k: v / 1e6 for k, v in tot.items()}
    ts = [{k: v / 1e6 for k, v in d.items()} for d in per_ts]
    return a, ts, hist


def sample_points(points, flags_ds, bbox):
    """Same rule as the GEE pipeline: share of a disc with new water."""
    r_at = int(round(F.VILLAGE_AT_M / 10.0))
    r_near = int(round(F.VILLAGE_NEAR_M / 10.0))
    d_at, d_near = disk(r_at), disk(r_near)
    H, W = flags_ds.height, flags_ds.width
    for p in points:
        col = int((p["lon"] - bbox["west"]) / CD.DEG_PER_10M)
        row = int((bbox["north"] - p["lat"]) / CD.DEG_PER_10M)
        y0, y1 = max(0, row - r_near), min(H, row + r_near + 1)
        x0, x1 = max(0, col - r_near), min(W, col + r_near + 1)
        if y0 >= y1 or x0 >= x1:
            p["flag"], p["f_at"], p["f_near"] = -1, 0.0, 0.0
            continue
        blk = flags_ds.read(1, window=Window(x0, y0, x1 - x0, y1 - y0))
        fl = (blk & B_FLOOD) > 0
        va = (blk & B_VALID) > 0
        for rad, kern, key in ((r_at, d_at, "at"), (r_near, d_near, "near")):
            ky0, kx0 = (row - rad) - y0, (col - rad) - x0
            k = np.zeros(blk.shape, bool)
            sy0, sx0 = max(0, ky0), max(0, kx0)
            sy1 = min(blk.shape[0], ky0 + 2 * rad + 1)
            sx1 = min(blk.shape[1], kx0 + 2 * rad + 1)
            k[sy0:sy1, sx0:sx1] = kern[sy0 - ky0:sy1 - ky0, sx0 - kx0:sx1 - kx0]
            n = k.sum()
            p[f"f_{key}"] = float((fl & k).sum()) / n if n else 0.0
            p[f"v_{key}"] = float((va & k).sum()) / n if n else 0.0
        if p["v_near"] < 0.5:
            p["flag"] = -1
        elif p["f_at"] >= F.VILLAGE_AT_FRAC:
            p["flag"] = 2
        elif p["f_near"] >= F.VILLAGE_NEAR_FRAC:
            p["flag"] = 1
        else:
            p["flag"] = 0


def village_tracts(aoi_shape, townships, flags_ds, bbox, px_area_fn):
    """MIMU village-tract (adm4) flood sums from the 10 m flags raster."""
    gj = F.fetch_mimu_layer(VT_LAYER)
    pcodes = {t["pcode"] for t in townships}
    pa = prep(aoi_shape)
    out = []
    for feat in gj["features"]:
        p, g = feat["properties"], feat.get("geometry")
        if not g or p.get("TS_PCODE") not in pcodes:
            continue
        geom = shape(g)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not pa.intersects(geom):
            continue
        w, s, e, n = geom.bounds
        x0 = max(0, int((w - bbox["west"]) / CD.DEG_PER_10M))
        x1 = min(flags_ds.width, int(math.ceil((e - bbox["west"]) / CD.DEG_PER_10M)))
        y0 = max(0, int((bbox["north"] - n) / CD.DEG_PER_10M))
        y1 = min(flags_ds.height, int(math.ceil((bbox["north"] - s) / CD.DEG_PER_10M)))
        if x0 >= x1 or y0 >= y1:
            continue
        blk = flags_ds.read(1, window=Window(x0, y0, x1 - x0, y1 - y0))
        btf = rasterio.transform.from_origin(
            bbox["west"] + x0 * CD.DEG_PER_10M, bbox["north"] - y0 * CD.DEG_PER_10M,
            CD.DEG_PER_10M, CD.DEG_PER_10M)
        m = rasterize([(geom, 1)], out_shape=blk.shape, transform=btf, fill=0,
                      dtype="uint8").astype(bool) & ((blk & B_AOI) > 0)
        area = np.broadcast_to(px_area_fn(y0, y1)[:, None], blk.shape)
        sums = {k: float(area[m & ((blk & b) > 0)].sum()) / 1e6
                for k, b in (("flood", B_FLOOD), ("valid", B_VALID),
                             ("land", B_LAND), ("water", B_WATER))}
        sums["area"] = float(area[m].sum()) / 1e6
        if sums["area"] < 0.01:
            continue
        out.append({
            "vt_pcode": p.get("VT_PCODE", ""), "vt": p.get("VT", ""),
            "vt_mmr": p.get("VT_MMR", ""), "ts": p.get("TS", ""),
            "ts_pcode": p.get("TS_PCODE", ""),
            "flood_km2": sums["flood"], "area_km2": sums["area"],
            "land_km2": sums["land"], "valid_km2": sums["valid"],
            "water_km2": sums["water"],
            "flood_pct": 100 * sums["flood"] / sums["area"],
            "land_frac": sums["valid"] / sums["land"] if sums["land"] else 1.0,
            "geom": F._round_geom(geom.simplify(F.TS_SIMPLIFY_DEG,
                                                preserve_topology=True), 5),
        })
    out.sort(key=lambda d: -d["flood_km2"])
    return out


def patch_sizes(path, bbox):
    """
    Flooded area by connected-patch size (8-connected), km2.

    The method has no minimum mapping unit -- same as every dashboard in this
    series -- so isolated speckle pixels count toward the total. This makes
    that share visible instead of silently filtering it. Patches are labelled
    within 4096 px windows, so a patch cut by a window edge is counted as two
    pieces: a slight over-count of the small classes, never an under-count.
    """
    B = 4096
    edges = [(0, 10), (10, 100), (100, 1000), (1000, 10 ** 12)]
    keys = ["lt_0.1ha", "0.1_1ha", "1_10ha", "ge_10ha"]
    out = dict.fromkeys(keys, 0.0)
    st = np.ones((3, 3), bool)
    m_side = CD.DEG_PER_10M * 111_319.49
    with rasterio.open(path) as d:
        for y in range(0, d.height, B):
            h = min(B, d.height - y)
            lat = bbox["north"] - (y + h / 2) * CD.DEG_PER_10M
            px_km2 = m_side * m_side * math.cos(math.radians(lat)) / 1e6
            for x in range(0, d.width, B):
                a = d.read(1, window=Window(x, y, min(B, d.width - x), h)) > 0
                if not a.any():
                    continue
                lab, _ = ndimage.label(a, st)
                sz = np.bincount(lab.ravel())[1:]
                for k, (lo, hi) in zip(keys, edges):
                    out[k] += float(sz[(sz >= lo) & (sz < hi)].sum()) * px_km2
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--page-only", action="store_true")
    args = ap.parse_args()

    if args.page_only:
        import dashboard_extras as X           # template patches for this AOI
        X.patch_template()
        fp = os.path.join(WORK_DIR, "flood.tif")
        if os.path.exists(fp):
            with open(F.AOI_GEOJSON, encoding="utf-8") as fh:
                from shapely.ops import unary_union
                b = unary_union([shape(f["geometry"]) for f in
                                 json.load(fh)["features"]]).bounds
            X.extra_meta = {"patches": patch_sizes(
                fp, {"west": b[0], "south": b[1], "east": b[2], "north": b[3]})}
            print("  patches:", {k: round(v, 1)
                                 for k, v in X.extra_meta["patches"].items()})
        F.rebuild_page(F.OUTPUT_HTML)
        return

    F.init_gee()
    aoi_info = F.load_aoi(F.AOI_GEOJSON)
    aoi_info["name"] = ("Seven Tanintharyi townships: Dawei, Launglon, "
                        "Thayetchaung, Yebyu, Palaw, Tanintharyi, Bokpyin")
    bbox = aoi_info["bbox"]
    width, height, transform = CD.grid_for(bbox)
    tiles = tiles_for(aoi_info["shape"], width, height, bbox)
    T = CD.SH_TILE_PX
    print(f"\n> AOI: {aoi_info['name']}")
    print(f"  grid {width:,} x {height:,} px at 10 m ({width * height / 1e9:.2f} Gpx); "
          f"{len(tiles)} of {math.ceil(width / T) * math.ceil(height / T)} "
          f"{T} px tiles touch the AOI")

    print("\n> Copernicus Data Space catalogue ...")
    have = {}
    for d in (F.PRE_FLOOD_DATE, F.POST_FLOOD_DATE):
        sc = catalogue(d, bbox)
        have[d] = sc
        for s in sc:
            print(f"    {d} {s['start'][11:19]} {s['platform']} {s['dir']} "
                  f"rel {s['rel']} online={s['online']}")
        asc = [s for s in sc if s["dir"] == F.ORBIT_PASS]
        rels = {s["rel"] for s in asc}
        if not asc:
            raise SystemExit(f"[STOP] no {F.ORBIT_PASS} frames on {d}")
        if rels != {F.REL_ORBIT}:
            # Sentinel Hub filters by direction, not track: two tracks on one
            # day would be mosaicked together, mixing viewing geometries.
            raise SystemExit(f"[STOP] {d}: {F.ORBIT_PASS} tracks {rels}, "
                             f"expected only {F.REL_ORBIT}")
    if args.check:
        print("\n[OK] Both dates in the ESA archive on one track.")
        return

    print("\n> MIMU administrative data ...")
    townships = F.load_townships(aoi_info["shape"])
    villages = F.load_villages(aoi_info["shape"], townships)
    towns = F.load_towns(aoi_info["shape"], townships)

    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"\n> Fetching Sentinel-1 VV from CDSE Sentinel Hub "
          f"(gamma0 terrain-corrected, Copernicus DEM, 10 m) ...")
    fetch_all(tiles, bbox)

    print(f"\n> Change detection, block by block "
          f"({R_PX} px speckle radius, {FLOOD_DB_THRESH} dB) ...")
    if os.path.exists(os.path.join(WORK_DIR, "analysis.json")):
        with open(os.path.join(WORK_DIR, "analysis.json")) as fh:
            cached = json.load(fh)
        a, ts_sums, hist = cached["a"], cached["ts"], np.array(cached["hist"])
        print("  using cached work/analysis.json")
    else:
        a, ts_sums, hist = analyse(aoi_info, townships, tiles, width, height,
                                   transform, bbox)
        with open(os.path.join(WORK_DIR, "analysis.json"), "w") as fh:
            json.dump({"a": a, "ts": ts_sums, "hist": hist.tolist()}, fh)

    land_frac = a["valid"] / a["land"] if a["land"] else 0.0
    valid_frac = a["valid"] / a["area"] if a["area"] else 0.0
    if a["land"] <= 0 or land_frac < 0.5:
        raise SystemExit(f"[STOP] only {land_frac:.1%} of land assessed")
    print(f"  => Newly inundated : {a['flood']:,.1f} km2 "
          f"({a['flood'] / a['area'] * 100:.2f}% of AOI)")
    print(f"  => Permanent water : {a['water']:,.1f} km2")
    print(f"  => Land assessed   : {land_frac:.2%} of {a['land']:,.0f} km2")

    for t, z in zip(townships, ts_sums):
        t.update({"flood_km2": z["flood"], "valid_km2": z["valid"],
                  "land_km2": z["land"], "water_km2": z["water"],
                  "area_km2": z["area"]})
        t["flood_pct"] = 100 * t["flood_km2"] / t["area_km2"] if t["area_km2"] else 0.0
        t["valid_frac"] = t["valid_km2"] / t["area_km2"] if t["area_km2"] else 0.0
        t["land_frac"] = t["valid_km2"] / t["land_km2"] if t["land_km2"] else 1.0

    lat_of = lambda y0, y1: (bbox["north"] - (np.arange(y0, y1) + 0.5)
                             * CD.DEG_PER_10M)
    m_side = CD.DEG_PER_10M * 111_319.49
    px_area_fn = lambda y0, y1: m_side * m_side * np.cos(np.radians(lat_of(y0, y1)))

    with rasterio.open(os.path.join(WORK_DIR, "flags.tif")) as fds:
        print(f"> Sampling {len(villages):,} village and {len(towns)} town point(s) ...")
        sample_points(villages, fds, bbox)
        sample_points(towns, fds, bbox)
        print("> Village tracts (MIMU adm4) ...")
        vts = village_tracts(aoi_info["shape"], townships, fds, bbox, px_area_fn)
    vil = F.roll_up_villages(townships, villages)
    print(f"  {len(vts)} village tracts; villages: {vil['at']} at, "
          f"{vil['near']} near, {vil['nodata']} unassessed of {vil['total']}")
    for t in sorted(townships, key=lambda x: -x["flood_km2"]):
        print(f"    {t['ts']:<16s} {t['flood_km2']:8,.1f} km2  "
              f"{t['flood_pct']:5.2f}%  {t['vil_at']:4d} village(s) hit  "
              f"land assessed {t['land_frac']:.1%}")

    print("\n> Encoding rasters at native 10 m ...")
    F.clear_assets()
    W = WORK_DIR
    overlays = {
        "flood": F.geotiff_to_layer(os.path.join(W, "flood.tif"),
                                    F._palette_solid(F.C_FLOOD), "flood"),
        "water": F.geotiff_to_layer(os.path.join(W, "water.tif"),
                                    F._palette_solid(F.C_WATER), "water"),
        "pre_sar": F.geotiff_to_layer(os.path.join(W, "pre_sar.tif"),
                                      F._palette_gray(), "pre_sar", True),
        "post_sar": F.geotiff_to_layer(os.path.join(W, "post_sar.tif"),
                                       F._palette_gray(), "post_sar", True),
    }

    stats = {
        "flood_km2": a["flood"], "water_km2": a["water"],
        "aoi_km2": a["area"], "polygon_km2": a["area"],
        "flood_pct": a["flood"] / a["area"] * 100,
        "valid_frac": valid_frac, "land_frac": land_frac,
        "land_km2": a["land"], "sea_km2": max(0.0, a["area"] - a["land"]),
        "vil_total": vil["total"], "vil_at": vil["at"],
        "vil_near": vil["near"], "vil_nodata": vil["nodata"],
    }
    sc_pre, sc_post = have[F.PRE_FLOOD_DATE], have[F.POST_FLOOD_DATE]
    meta = {
        "title": f"Tanintharyi flood inundation {F.PRE_FLOOD_DATE} → {F.POST_FLOOD_DATE}",
        "pre": F.PRE_FLOOD_DATE, "post": F.POST_FLOOD_DATE,
        "pre_time": sc_pre[0]["start"][11:16] + " UTC",
        "post_time": sc_post[0]["start"][11:16] + " UTC",
        "platform": "Sentinel-1" + sc_post[0]["platform"][-1],
        "window": 0, "gap": F._gap_text(),
        "pass": F.ORBIT_PASS.title(), "pass_short": "Asc.",
        "track": F.REL_ORBIT,
        "source": "CDSE Sentinel Hub, gamma0 terrain-corrected",
        "n_pre": len(sc_pre), "cov_pre": 100.0,
        "n_post": len(sc_post), "cov_post": 100.0,
        "thresh": FLOOD_DB_THRESH, "speckle": SPECKLE_RADIUS_M,
        "slope": MAX_SLOPE_DEG, "scale": F.EXPORT_SCALE,
        "vil_at_m": F.VILLAGE_AT_M, "vil_near_m": F.VILLAGE_NEAR_M,
        "vil_at_frac": round(F.VILLAGE_AT_FRAC * 100, 1),
        "vil_near_frac": round(F.VILLAGE_NEAR_FRAC * 100, 1),
        "ramp": F.CHORO_RAMP, "breaks": F.CHORO_BREAKS,
        "c_flood": F.C_FLOOD, "c_water": F.C_WATER, "c_village": F.C_VILLAGE,
        "mb_pre": round(overlays["pre_sar"]["mb"]),
        "mb_post": round(overlays["post_sar"]["mb"]),
        "diff_hist": hist.tolist(),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    meta["patches"] = patch_sizes(os.path.join(WORK_DIR, "flood.tif"), bbox)
    # Per-date valid coverage of the AOI, measured, replaces the nominal 100%.
    meta["cov_pre"] = meta["cov_post"] = round(valid_frac * 100, 2)

    print("> Building dashboard ...")
    import dashboard_extras as X               # template patches for this AOI
    X.patch_template()
    X.extra_payload = {"vts": [{k: v for k, v in d.items() if k != "geom"}
                               for d in vts],
                       "vtGeo": {"type": "FeatureCollection", "features": [
                           {"type": "Feature", "properties": {"i": i},
                            "geometry": d["geom"]} for i, d in enumerate(vts)]}}
    F.build_dashboard(overlays, stats, aoi_info, townships, villages, meta,
                      F.OUTPUT_HTML, towns)

    with open(F.STATS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": {k: v for k, v in meta.items() if k != "diff_hist"},
                   "aoi": stats,
                   "townships": [{k: v for k, v in t.items() if k != "geom"}
                                 for t in townships],
                   "village_tracts": [{k: v for k, v in d.items() if k != "geom"}
                                      for d in vts]},
                  fh, indent=1, ensure_ascii=False)

    html_mb = os.path.getsize(F.OUTPUT_HTML) / 1e6
    asset_mb = sum(v["mb"] for v in overlays.values() if v["external"])
    print(f"\n  Done: {F.OUTPUT_HTML} ({html_mb:.1f} MB) + {F.ASSET_DIR}/ "
          f"({asset_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
