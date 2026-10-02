#!/usr/bin/env python3
"""
Sentinel-1 flood mapping with imagery from Copernicus Data Space, analysed locally.

WHY A SECOND PIPELINE
  flood_mapping.py does everything inside Earth Engine.  That breaks the moment
  GEE does not have the scene -- which is exactly what happened for Kale on
  2026-09-03: ESA had both frames online within the hour, GEE had nothing and
  was still empty a day later.  This script keeps the AOI, the MIMU analysis and
  the dashboard identical, and swaps only the imagery source and the arithmetic:
  Sentinel Hub for the SAR, numpy for the differencing.

WHAT IS SHARED WITH flood_mapping.py
  AOI loading, MIMU townships / villages / town points, the palettes, the PNG
  chunking, and the whole dashboard.  Those are imported, not copied, so the two
  pipelines cannot drift apart in how they present a result.

WHAT DIFFERS, AND WHY IT IS STATED ON THE PAGE
  GEE serves sigma0 (ellipsoid); Sentinel Hub here serves gamma0 with terrain
  correction against the Copernicus DEM.  On sloping ground those differ by
  several dB.  Both dates come from the SAME source, so the difference cancels
  and the -3 dB threshold stays meaningful -- but a figure from this pipeline is
  not interchangeable with one from the GEE pipeline, and the dashboard says so.

ANCILLARY LAYERS STILL COME FROM GEE
  JRC permanent water and the HydroSHEDS terrain mask are static masks, not part
  of the differenced pair, so taking them from Earth Engine introduces no
  cross-source bias.  They are resampled onto the CDSE grid.

USAGE
    python flood_mapping_cdse.py            # full run
    python flood_mapping_cdse.py --check    # confirm both dates exist at ESA
"""

import argparse
import glob
import json
import os
import shutil
from datetime import datetime, timezone

import ee
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage

import cdse_source as CD
import flood_mapping as F

OUTPUT_HTML = F.OUTPUT_HTML
STATS_JSON = F.STATS_JSON
TEMP_DIR = F.TEMP_DIR
# Metered CDSE imagery lives here, OUTSIDE TEMP_DIR, so end-of-run cleanup
# cannot bin it. Sentinel Hub charges for every fetch.
CDSE_CACHE = "cdse_cache"

# The dB drop that marks new open water, and the speckle filter radius, are the
# same as the GEE pipeline so the two remain comparable in method even though
# they are not interchangeable in calibration.
FLOOD_DB_THRESH = F.FLOOD_DB_THRESH
SPECKLE_RADIUS_M = F.SPECKLE_RADIUS
MAX_SLOPE_DEG = F.MAX_SLOPE_DEG


# ── grid helpers ─────────────────────────────────────────────────────────────

def disk(radius_px: int) -> np.ndarray:
    """Boolean disk, so the speckle filter is circular like GEE's focal_mean."""
    r = int(radius_px)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def focal_mean(a: np.ndarray, valid: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Circular focal mean that ignores invalid pixels instead of averaging them in.

    A plain convolution would drag NaN, or zero, across every neighbourhood that
    touches the scene edge; dividing two convolutions -- summed values over
    counted valid pixels -- keeps the edge honest.
    """
    k = disk(radius_px).astype(np.float32)
    filled = np.where(valid, a, 0.0).astype(np.float32)
    s = ndimage.convolve(filled, k, mode="constant", cval=0.0)
    n = ndimage.convolve(valid.astype(np.float32), k, mode="constant", cval=0.0)
    out = np.full(a.shape, np.nan, np.float32)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    return out


def pixel_area_m2(height: int, bbox: dict) -> np.ndarray:
    """
    Ground area of one pixel per raster row, in m2.

    The grid is in degrees, so a pixel shrinks east-west as latitude rises. Kale
    spans 1.07 deg, about 1% of pixel area top to bottom -- small, but free to
    get right, and area is what every headline number is made of.
    """
    lat = bbox["north"] - (np.arange(height) + 0.5) * CD.DEG_PER_10M
    m_lat = CD.DEG_PER_10M * 111_319.49
    m_lon = CD.DEG_PER_10M * 111_319.49 * np.cos(np.radians(lat))
    return (m_lat * m_lon).astype(np.float64)


def gee_mask_on_grid(image: ee.Image, name: str, aoi_shape, bbox: dict,
                     width: int, height: int, transform) -> np.ndarray:
    """Download a uint8 GEE mask and resample it onto the CDSE grid."""
    path = F.download_as_geotiff(image, name, aoi_shape, F.EXPORT_SCALE, bbox)
    out = np.zeros((height, width), np.uint8)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest)
    return out


# ── statistics ───────────────────────────────────────────────────────────────

def zonal(geom, flood, valid, land, water, px_area, width, height, transform):
    """Area sums (km2) for one polygon, rasterised onto the analysis grid."""
    m = rasterize([(geom, 1)], out_shape=(height, width), transform=transform,
                  fill=0, dtype="uint8").astype(bool)
    if not m.any():
        return None
    area = np.broadcast_to(px_area[:, None], (height, width))
    inside = area * m
    return {
        "flood_km2": float((inside * flood).sum()) / 1e6,
        "valid_km2": float((inside * valid).sum()) / 1e6,
        "land_km2":  float((inside * land).sum()) / 1e6,
        "water_km2": float((inside * water).sum()) / 1e6,
        "area_km2":  float(inside.sum()) / 1e6,
    }


def sample_points(points, flood, valid, bbox, width, height):
    """
    Classify settlement points by new water within two radii.

    Same thresholds and the same meaning as the GEE pipeline: a fraction of the
    disc around the point, not "the village is under water".
    """
    r_at = int(round(F.VILLAGE_AT_M / 10.0))
    r_near = int(round(F.VILLAGE_NEAR_M / 10.0))
    d_at, d_near = disk(r_at), disk(r_near)
    for p in points:
        col = int((p["lon"] - bbox["west"]) / CD.DEG_PER_10M)
        row = int((bbox["north"] - p["lat"]) / CD.DEG_PER_10M)
        for rad, kern, key in ((r_at, d_at, "at"), (r_near, d_near, "near")):
            y0, y1 = max(0, row - rad), min(height, row + rad + 1)
            x0, x1 = max(0, col - rad), min(width, col + rad + 1)
            if y0 >= y1 or x0 >= x1:
                p[f"f_{key}"] = p[f"v_{key}"] = 0.0
                continue
            k = kern[y0 - (row - rad):y1 - (row - rad),
                     x0 - (col - rad):x1 - (col - rad)]
            n = k.sum()
            p[f"f_{key}"] = float((flood[y0:y1, x0:x1] * k).sum()) / n if n else 0.0
            p[f"v_{key}"] = float((valid[y0:y1, x0:x1] * k).sum()) / n if n else 0.0
        if p.get("v_near", 0.0) < 0.5:
            p["flag"] = -1
        elif p.get("f_at", 0.0) >= F.VILLAGE_AT_FRAC:
            p["flag"] = 2
        elif p.get("f_near", 0.0) >= F.VILLAGE_NEAR_FRAC:
            p["flag"] = 1
        else:
            p["flag"] = 0


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true",
                    help="confirm both dates exist at ESA, then stop")
    ap.add_argument("--keep-tif", action="store_true")
    args = ap.parse_args()

    F.init_gee()
    aoi_info = F.load_aoi(F.AOI_GEOJSON)
    bbox = aoi_info["bbox"]
    width, height, transform = CD.grid_for(bbox)

    print(f"\n> AOI: {aoi_info['name']}")
    print(f"  grid {width} x {height} px at 10 m "
          f"({width * height / 1e6:.0f} Mpx per layer)")

    # ── the catalogue is ESA's own index: it, not GEE, decides availability ──
    print(f"\n> Copernicus Data Space catalogue ...")
    have = {}
    for d in (F.PRE_FLOOD_DATE, F.POST_FLOOD_DATE):
        sc = CD.scenes_on(d, bbox)
        have[d] = sc
        print(f"    {d}: {len(sc)} IW GRDH frame(s)"
              + ("  " + ", ".join(s["start"][11:19] for s in sc) if sc else ""))
    missing = [d for d, s in have.items() if not s]
    if missing:
        raise SystemExit(
            f"\n[STOP] Not in the ESA archive either: {', '.join(missing)}.\n"
            f"       Nothing was substituted and nothing was written.\n")
    if args.check:
        print("\n[OK] Both dates are in the ESA archive. Re-run without "
              "--check to build the map.")
        return

    print("\n> MIMU administrative data ...")
    townships = F.load_townships(aoi_info["shape"])
    villages = F.load_villages(aoi_info["shape"], townships)
    towns = F.load_towns(aoi_info["shape"], townships)

    # ── imagery: both dates from CDSE, same processing chain ────────────────
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(CDSE_CACHE, exist_ok=True)
    print(f"\n> Fetching Sentinel-1 from Copernicus Data Space "
          f"(gamma0, terrain-corrected, {F.ORBIT_PASS.lower()}) ...")
    pre_path = os.path.join(CDSE_CACHE, f"vv_{F.PRE_FLOOD_DATE}.tif")
    post_path = os.path.join(CDSE_CACHE, f"vv_{F.POST_FLOOD_DATE}.tif")
    if not os.path.exists(pre_path):
        CD.fetch_vv_db(F.PRE_FLOOD_DATE, bbox, F.ORBIT_PASS, pre_path)
    if not os.path.exists(post_path):
        CD.fetch_vv_db(F.POST_FLOOD_DATE, bbox, F.ORBIT_PASS, post_path)

    with rasterio.open(pre_path) as ds:
        pre_db, pre_ok = ds.read(1), ds.read(2) > 0.5
    with rasterio.open(post_path) as ds:
        post_db, post_ok = ds.read(1), ds.read(2) > 0.5

    print("> Speckle filter (circular focal mean, "
          f"{SPECKLE_RADIUS_M} m radius) ...")
    r_px = max(1, int(round(SPECKLE_RADIUS_M / 10.0)))
    pre_s = focal_mean(pre_db, pre_ok, r_px)
    post_s = focal_mean(post_db, post_ok, r_px)

    # ── static masks from GEE, resampled onto this grid ─────────────────────
    print("> Permanent water and terrain masks (Earth Engine) ...")
    aoi_ee = aoi_info["ee_geom"]
    # CLIP BEFORE UNMASK, always. unmask() on an image whose projection is
    # derived rather than fixed -- ee.Terrain.slope() over HydroSHEDS is
    # exactly that -- yields an image every reducer reports as EMPTY.
    # Measured on this AOI: slope.lt(5).add(1) histograms 25,547 steep and
    # 37,949 flat pixels; put .unmask(0) ahead of .clip() and it histograms
    # nothing at all. The wrong order silently produced a 0 km2 flood map
    # rather than failing, which is the worst way for this to go wrong.
    water_img = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality")
                 .gte(4).clip(aoi_ee).unmask(0).toByte())
    slope = ee.Terrain.slope(ee.Image("WWF/HydroSHEDS/03VFDEM"))
    # 0 = no DEM (never land here), 1 = land but too steep, 2 = flat land.
    terr_img = slope.lt(MAX_SLOPE_DEG).add(1).clip(aoi_ee).unmask(0).toByte()
    water = gee_mask_on_grid(water_img, "water.tif", aoi_info["shape"], bbox,
                             width, height, transform) > 0
    terr = gee_mask_on_grid(terr_img, "terr.tif", aoi_info["shape"], bbox,
                            width, height, transform)
    land, flat = terr >= 1, terr == 2

    print("> Detecting new inundation ...")
    valid = pre_ok & post_ok & land
    # np.where evaluates both branches, so the subtraction touches NaN
    # outside the valid mask. That is expected here, not a problem.
    with np.errstate(invalid="ignore"):
        diff = np.where(valid, post_s - pre_s, np.nan)
    flood = np.zeros((height, width), bool)
    np.less(diff, FLOOD_DB_THRESH, out=flood, where=np.isfinite(diff))
    flood &= ~water & flat & valid

    # ── statistics, all on the native grid ──────────────────────────────────
    px_area = pixel_area_m2(height, bbox)
    area2d = np.broadcast_to(px_area[:, None], (height, width))
    shp = aoi_info["shape"]
    parts = list(shp.geoms) if shp.geom_type == "MultiPolygon" else [shp]
    aoi_mask = rasterize([(g, 1) for g in parts],
                         out_shape=(height, width), transform=transform,
                         fill=0, dtype="uint8").astype(bool)
    inside = area2d * aoi_mask
    a = {"flood_a": float((inside * flood).sum()) / 1e6,
         "valid_a": float((inside * valid).sum()) / 1e6,
         "land_a": float((inside * land).sum()) / 1e6,
         "water_a": float((inside * water).sum()) / 1e6,
         "total_a": float(inside.sum()) / 1e6}
    land_frac = a["valid_a"] / a["land_a"] if a["land_a"] else 0.0

    # A product with nothing assessed is a bug, not a finding. The first run of
    # this pipeline wrote a cheerful 0 km2 dashboard because the terrain mask
    # came back empty. Refuse, and name the input that failed.
    if a["land_a"] <= 0 or land_frac < 0.5:
        raise SystemExit(
            f"\n[STOP] Only {land_frac:.1%} of the land here could be "
            f"assessed -- refusing to write a map.\n"
            f"       land {a['land_a']:,.1f} km2 | usable on both dates "
            f"{a['valid_a']:,.1f} km2 | AOI {a['total_a']:,.1f} km2\n"
            f"       pre valid {float(pre_ok.mean()):.1%} | post valid "
            f"{float(post_ok.mean()):.1%} | terrain mask "
            f"{float(land.mean()):.1%} of grid\n"
            f"       A terrain mask near 0% means an Earth Engine expression "
            f"was unmasked\n"
            f"       before it was clipped. Nothing was written.\n")

    valid_frac = a["valid_a"] / a["total_a"] if a["total_a"] else 0.0
    print(f"  => Newly inundated : {a['flood_a']:,.1f} km2 "
          f"({a['flood_a'] / a['total_a'] * 100:.2f}% of AOI)")
    print(f"  => Permanent water : {a['water_a']:,.1f} km2")
    print(f"  => Land assessed   : {land_frac:.2%}")

    print(f"> Zonal statistics for {len(townships)} township(s) ...")
    for t in townships:
        from shapely.geometry import shape as _shape
        z = zonal(_shape(t["geom"]), flood, valid, land, water, px_area,
                  width, height, transform)
        t.update(z or {"flood_km2": 0.0, "valid_km2": 0.0, "land_km2": 0.0,
                       "water_km2": 0.0, "area_km2": 0.0})
        t["flood_pct"] = (100 * t["flood_km2"] / t["area_km2"]
                          if t["area_km2"] else 0.0)
        t["valid_frac"] = (t["valid_km2"] / t["area_km2"]
                           if t["area_km2"] else 0.0)
        t["land_frac"] = (t["valid_km2"] / t["land_km2"]
                          if t["land_km2"] else 1.0)

    print(f"> Sampling {len(villages):,} village and {len(towns)} town "
          f"point(s) ...")
    sample_points(villages, flood, valid, bbox, width, height)
    sample_points(towns, flood, valid, bbox, width, height)
    vil = F.roll_up_villages(townships, villages)

    for t in sorted(townships, key=lambda x: -x["flood_km2"]):
        print(f"    {t['ts']:<16s} {t['flood_km2']:8,.1f} km2  "
              f"{t['flood_pct']:5.1f}%  {t['vil_at']:4d} village(s) hit")
    print(f"  Villages: {vil['at']:,} with water at the settlement, "
          f"{vil['near']:,} more within {F.VILLAGE_NEAR_M} m")

    # ── rasters -> palette PNG chunks -> dashboard ──────────────────────────
    def write_byte(arr, path):
        prof = {"driver": "GTiff", "height": height, "width": width, "count": 1,
                "dtype": "uint8", "crs": "EPSG:4326", "transform": transform,
                "compress": "deflate", "tiled": True}
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr.astype(np.uint8), 1)
        return path

    def sar_byte(db, ok):
        v = np.clip((db + 25.0) / 25.0, 0, 1)
        out = (v * 254 + 1).astype(np.uint8)
        out[~ok | ~np.isfinite(db)] = 0
        return out

    print("\n> Encoding rasters at native 10 m ...")
    F.clear_assets()
    paths = {
        "flood": write_byte(flood & aoi_mask, os.path.join(TEMP_DIR, "flood.tif")),
        "water": write_byte(water & aoi_mask, os.path.join(TEMP_DIR, "waterb.tif")),
        "pre_sar": write_byte(sar_byte(pre_s, pre_ok & aoi_mask),
                              os.path.join(TEMP_DIR, "pre_vv.tif")),
        "post_sar": write_byte(sar_byte(post_s, post_ok & aoi_mask),
                               os.path.join(TEMP_DIR, "post_vv.tif")),
    }
    overlays = {
        "flood": F.geotiff_to_layer(paths["flood"], F._palette_solid(F.C_FLOOD),
                                    "flood"),
        "water": F.geotiff_to_layer(paths["water"], F._palette_solid(F.C_WATER),
                                    "water"),
        "pre_sar": F.geotiff_to_layer(paths["pre_sar"], F._palette_gray(),
                                      "pre_sar", True),
        "post_sar": F.geotiff_to_layer(paths["post_sar"], F._palette_gray(),
                                       "post_sar", True),
    }

    stats = {
        "flood_km2": a["flood_a"], "water_km2": a["water_a"],
        "aoi_km2": a["total_a"], "polygon_km2": a["total_a"],
        "flood_pct": a["flood_a"] / a["total_a"] * 100 if a["total_a"] else 0.0,
        "valid_frac": valid_frac, "land_frac": land_frac,
        "land_km2": a["land_a"], "sea_km2": max(0.0, a["total_a"] - a["land_a"]),
        "vil_total": vil["total"], "vil_at": vil["at"],
        "vil_near": vil["near"], "vil_nodata": vil["nodata"],
    }
    meta = {
        "title": f"Flood inundation {F.PRE_FLOOD_DATE} → {F.POST_FLOOD_DATE}",
        "pre": F.PRE_FLOOD_DATE, "post": F.POST_FLOOD_DATE,
        "window": F.DATE_WINDOW, "gap": F._gap_text(),
        "pass": F.ORBIT_PASS.title(),
        "pass_short": "Asc." if F.ORBIT_PASS == "ASCENDING" else "Desc.",
        "track": F.REL_ORBIT, "source": "CDSE Sentinel Hub, gamma0 terrain",
        "n_pre": len(have[F.PRE_FLOOD_DATE]), "cov_pre": 100.0,
        "n_post": len(have[F.POST_FLOOD_DATE]), "cov_post": 100.0,
        "thresh": FLOOD_DB_THRESH, "speckle": SPECKLE_RADIUS_M,
        "slope": MAX_SLOPE_DEG, "scale": F.EXPORT_SCALE,
        "vil_at_m": F.VILLAGE_AT_M, "vil_near_m": F.VILLAGE_NEAR_M,
        "vil_at_frac": round(F.VILLAGE_AT_FRAC * 100, 1),
        "vil_near_frac": round(F.VILLAGE_NEAR_FRAC * 100, 1),
        "ramp": F.CHORO_RAMP, "breaks": F.CHORO_BREAKS,
        "c_flood": F.C_FLOOD, "c_water": F.C_WATER, "c_village": F.C_VILLAGE,
        "mb_pre": round(overlays["pre_sar"]["mb"]),
        "mb_post": round(overlays["post_sar"]["mb"]),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    print("> Building dashboard ...")
    F.build_dashboard(overlays, stats, aoi_info, townships, villages, meta,
                      OUTPUT_HTML, towns)
    with open(STATS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "aoi": stats,
                   "townships": [{k: v for k, v in t.items()
                                  if k not in ("geom", "_lyr")}
                                 for t in townships]},
                  fh, indent=1, ensure_ascii=False)

    if not args.keep_tif:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    html_mb = os.path.getsize(OUTPUT_HTML) / 1e6
    asset_mb = sum(v["mb"] for v in overlays.values() if v["external"])
    print(f"\n{'-' * 66}")
    print(f"  Done!  '{OUTPUT_HTML}' ({html_mb:.1f} MB)")
    if asset_mb:
        print(f"  Plus {F.ASSET_DIR}/ -- "
              f"{len(glob.glob(os.path.join(F.ASSET_DIR, '*.png')))} PNG(s), "
              f"{asset_mb:,.0f} MB.")
    print(f"  Imagery: Copernicus Data Space, both dates, 10 m native.")
    print(f"{'-' * 66}")


if __name__ == "__main__":
    main()
