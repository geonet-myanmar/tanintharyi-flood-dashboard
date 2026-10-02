#!/usr/bin/env python3
"""
Sentinel-1 Flood Inundation Mapping -- Township Dashboard, Permanent Web Map
============================================================================
WHY THIS APPROACH
  geemap.addLayer() writes GEE tile-server URLs into the HTML.  Those URLs
  expire after ~2 days, causing layers to go blank.  This script avoids that
  by downloading every layer as a GeoTIFF, converting it to a palette PNG and
  publishing that PNG as page-local data -- either a base64 data URI inside the
  HTML or a file in ASSET_DIR next to it.  Either way nothing points at a
  Google server, so the map never expires.

METHOD
  Change-detection: VV backscatter difference (post - pre, dB).
  Flooded open water causes specular reflection away from the sensor,
  producing a sharp drop in VV backscatter.

  The result is then summarised per MIMU township and per MIMU village point,
  so the product is not just a picture but an impact table.

DATA      Copernicus Sentinel-1 GRD IW (via Google Earth Engine)
          MIMU township boundaries (adm3, 250k, v9.4) and village points
          (pplp2, 250k, PCode v9.7) -- geonode.themimu.info GeoServer WFS
AOI       Loaded from AOI_GEOJSON = merged_6_townships.geojson -- six whole
          MIMU townships in Ayeyarwady Region, 6,667 km2: Hinthada and
          Lemyethna (Hinthada District), plus Kyaunggon, Kyonpyaw, Thabaung
          and Yegyi (Pathein District).  Because the AOI IS the townships,
          nothing is clipped and every "% flooded" is of the whole township.
Pre-flood  2026-07-24
Post-flood 2026-08-29
Output     index.html + ASSET_DIR/*.png  -- self-hosted, never-expiring

DATA AVAILABILITY (checked against the GEE catalogue on 2026-08-30 ~10:50 UTC)
  Three Sentinel-1 tracks image this AOI, each on a 12-day repeat:

      ASCENDING  track  41 : 07-20, 08-01, 08-13, 08-25
      ASCENDING  track 143 : 07-27, 08-08, 08-20
      DESCENDING track 106 : 07-24, 08-05, 08-17, [08-29]   <- USED

  Both requested dates fall on DESCENDING track 106, 36 days apart -- exactly
  three repeat cycles -- so this is the same ground track imaged three cycles
  over: same viewing geometry, same incidence angle, the cleanest possible
  basis for differencing.  Each date is covered by 2 scenes at 100.0% of the
  AOI, measured rather than assumed, in BOTH collections.

  2026-08-29 crosses this AOI at 23:32 UTC (06:02 local on the 30th) and was
  fully ingested about 11 hours later, so no waiting was needed.  The guards
  are still in force: availability is judged on COVERAGE, not scene presence,
  and a pass younger than FRESH_PASS_HOURS that is short of MIN_AOI_COVERAGE
  stops the run rather than mapping a fraction of the area.

  CAVEAT ON THE BASELINE: 2026-07-24 is itself mid-monsoon, not a dry-season
  image.  What is mapped is water that appeared BETWEEN 24 July and 29 August,
  so ground already under water on 24 July is not counted.  This is the
  intended change-detection product but a lower bound on total standing water
  on 29 August.  Note that the 24 Jul -> 17 Aug product over the wider western
  AOI shares this baseline, so the two are directly comparable on these six
  townships: this run extends the same window by two more repeat cycles.

RESOLUTION
  Every layer -- flood mask, permanent water AND both SAR VV backdrops -- is
  produced and displayed at Sentinel-1's native 10 m GRD pixel spacing.  Every
  statistic, AOI-wide and per township, is reduced at 10 m too.  Nothing is
  downsampled anywhere in this script.

  That has a real cost, which is why the output is not a single file.  The AOI
  bbox is 1.667 deg x 2.253 deg, i.e. 18,559 x 25,077 px (465 Mpx) per layer at
  10 m.  Measured on the 2026-08-17 run:

      flood mask        @10 m ->    7.2 MB PNG  (binary, compresses ~65x)
      permanent water   @10 m ->    1.2 MB PNG  (embedded in the page)
      SAR VV backdrop   @10 m -> 260-268 MB PNG (speckle, near-incompressible)

  Base64 inflates by 1.33x, so embedding two native SAR backdrops would make a
  ~450 MB HTML file -- far over GitHub's hard 100 MB per-file push limit, and
  hopeless for a browser to parse.  So each layer is:

    * cut into chunks of at most MAX_CHUNK_PX pixels, with all-nodata chunks
      dropped -- worth a lot here, because the slanted AOI leaves large empty
      wedges in the bbox corners; and
    * embedded as base64 if the whole layer is under EMBED_MAX_MB, otherwise
      written to ASSET_DIR/ and referenced by relative path.

  Chunks of a layer live in one Leaflet layer group, so they toggle as a unit.
  The SAR layers start hidden, so their bytes are only fetched if the user
  actually turns them on.  Download tiles that miss the AOI polygon entirely
  are never requested, and each layer's GeoTIFF is encoded and deleted before
  the next one starts, so peak disk stays around 1 GB rather than 3 GB.

  Deploying therefore means publishing index.html *and* the ASSET_DIR folder.

REQUIREMENTS
    pip install earthengine-api geemap rasterio numpy Pillow shapely

FIRST-TIME GEE SETUP
    1. Sign up at https://earthengine.google.com
    2. Enable the Earth Engine API in a Google Cloud project
    3. Run:  earthengine authenticate
       (or let this script call ee.Authenticate() automatically)

USAGE
    python flood_mapping.py              # full run (~75 min at 10 m)
    python flood_mapping.py --check      # availability only, no downloads
    python flood_mapping.py --no-sar     # skip the two heavy SAR backdrops
    python flood_mapping.py --page-only  # re-render index.html, same numbers
    python watch_and_run.py              # wait for the scene, then run
"""

import argparse
import base64
import glob
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from io import BytesIO

import ee
import geemap
import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge as rio_merge
from rasterio.windows import Window
from shapely.geometry import box as shp_box
from shapely.geometry import mapping, shape
from shapely.prepared import prep

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# AOI is read from a GeoJSON file (FeatureCollection, Feature or bare geometry).
AOI_GEOJSON = "aoi.geojson"

PRE_FLOOD_DATE  = "2026-08-24"
POST_FLOOD_DATE = "2026-09-05"

# +/- days searched around each target date.  DELIBERATELY 0: the two dates
# requested are the two dates used.  A window of even 1 day would let the
# post composite fall back to 2026-08-16 -- a different track, a different
# viewing geometry, and not the date that was asked for.  With 0, each
# composite is exactly the scenes acquired on that UTC day, and a date that is
# not in the catalogue yet stops the run instead of being quietly replaced.
DATE_WINDOW     = 0

FLOOD_DB_THRESH = -3.0    # VV drop (dB) below which pixel is classified flooded
SPECKLE_RADIUS  = 30      # focal-mean kernel radius in metres
MAX_SLOPE_DEG   = 5.0     # steeper terrain cannot hold standing water

# Sentinel-1 orbit pass used for BOTH pre- and post-flood composites.
# Change detection is only valid when both images share the same viewing
# geometry (same pass, ideally same relative orbit).  Mixing ASCENDING and
# DESCENDING scenes introduces a systematic backscatter bias that destroys the
# flood signal.  Over this AOI both 2026-07-24 and 2026-08-29 are DESCENDING
# track 106 passes -- three 12-day repeats apart.
ORBIT_PASS = "DESCENDING"

# Pin a single Sentinel-1 relative orbit (ground track) so pre and post use the
# identical acquisition path.  Set to None to use every track of the chosen
# pass.  Three tracks image this AOI -- ascending 41 and 143, descending 106.
# Track 106 carries BOTH requested dates at 100% AOI coverage; its repeats run
# 07-24, 08-05, 08-17, 08-29, 09-10.  The 36-day separation is exactly three
# repeat cycles, so the pair is the same ground track imaged three times over.
REL_ORBIT = 33

# Source collections, in order of preference.  (collection_id, is_linear)
#
# COPERNICUS/S1_GRD is the canonical product and is already in dB.
# COPERNICUS/S1_GRD_FLOAT holds the SAME scenes in linear power, and in
# practice is ingested into the catalogue EARLIER -- a scene acquired today is
# routinely in _FLOAT while S1_GRD still lags a day or two.  Since
#     S1_GRD == 10 * log10(S1_GRD_FLOAT)     (verified exact on this AOI)
# reading _FLOAT and converting loses nothing.
#
# preflight() picks the first collection that carries BOTH dates on the
# configured track and uses it for the pre AND post composite.  It never mixes
# collections across the pair -- that is the same discipline as not mixing
# orbit passes.
S1_COLLECTIONS = [
    ("COPERNICUS/S1_GRD",       False),
    ("COPERNICUS/S1_GRD_FLOAT", True),
]

# Minimum AOI coverage, on BOTH dates, for a collection to be preferred.
#
# Presence is not enough.  Ingestion of one pass is not atomic: the scenes of a
# three-scene pass land minutes to hours apart, so for a while S1_GRD can hold
# 2 of 3 scenes (95% of this AOI) while S1_GRD_FLOAT already holds all three.
# Choosing on presence alone picked the incomplete one and left 5% of the AOI
# permanently unassessed in the output -- honestly labelled, but avoidable.
# choose_collection() now prefers a collection only if both dates clear this,
# and falls back to the best available with a loud warning.
MIN_AOI_COVERAGE = 0.99

# How recently a pass must have been acquired for INCOMPLETE coverage to read as
# "still ingesting" rather than "this is all there will ever be".
#
# That distinction is the whole game. Some AOIs are genuinely wider than one
# pass and can never reach MIN_AOI_COVERAGE; for those, mapping the part that IS
# covered is the right answer, and choose_collection() falls back to it. But a
# pass acquired three hours ago sitting at 12% is not a narrow swath, it is a
# catalogue still filling, and mapping it would discard most of the AOI out of
# impatience. Below this age, partial coverage stops the run instead.
FRESH_PASS_HOURS = 36

# Your GEE Cloud project ID -- leave "" to let GEE auto-detect from credentials
GEE_PROJECT = "gee-python-419405"

# Native Sentinel-1 GRD IW pixel spacing.  EVERY layer and EVERY statistic --
# AOI-wide, per township and per village -- use this.  No downsampling anywhere.
EXPORT_SCALE = 10

# SAR VV backdrops are rendered at native resolution too (see RESOLUTION above).
# Kept as a separate name so the heavy display layers can be dropped to a
# coarser scale for a quick test run without touching the flood product.
SAR_DISPLAY_SCALE = 10

# ── MIMU administrative data ──────────────────────────────────────────────────
# Myanmar Information Management Unit, GeoNode GeoServer WFS, EPSG:4326.
# Everything is cached under DATA_DIR so a re-run is offline and reproducible.
MIMU_WFS   = "https://geonode.themimu.info/geoserver/wfs"
DATA_DIR   = os.path.join("data", "mimu")

# Township boundaries: a local copy is used if present (that is the file already
# in this folder), otherwise the layer is pulled from the WFS.
TOWNSHIP_GEOJSON = "MyanmarTownshipBoundaries.geojson"
TOWNSHIP_LAYER   = "mmr_polbnda_adm3_250k_mimu_1"

# MIMU town points -- the gazetteer of towns, one per township seat plus
# larger settlements. Only 494 nationwide, so on a single-township AOI this
# is a place label rather than a statistical population; the village points
# carry the settlement-level impact.
TOWN_POINTS_LAYER = "mmr_pplp1_mimu250k"

# Village points are published per state/region, so only the layers for the
# states the AOI actually touches are downloaded.  Keyed by ST_PCODE.
VILLAGE_LAYERS = {
    "MMR001": "mmr_kcn_pplp2_250k_mimu",          # Kachin
    "MMR002": "mmr_kyh_pplp2_250k_mimu",          # Kayah
    "MMR003": "mmr_kyn_pplp2_250k_mimu",          # Kayin
    "MMR004": "mmr_chn_pplp2_250k_wfp_mimu",      # Chin
    "MMR005": "mmr_sag_pplp2_250k_mimu",          # Sagaing
    "MMR006": "mmr_tni_pplp2_250k_mimu",          # Tanintharyi
    "MMR007": "mmr_bgo_pplp2_250k_mimu",          # Bago (East)
    "MMR008": "mmr_bgo_pplp2_250k_mimu",          # Bago (West)
    "MMR009": "mmr_mgy_pplp2_250k_wfp_mimu",      # Magway
    "MMR010": "mmr_mdy_pplp2_250k_mimu_1",        # Mandalay
    "MMR011": "mmr_mon_pplp2_250k_mimu",          # Mon
    "MMR012": "mmr_rke_pplp2_250k_unicef_mimu",   # Rakhine
    "MMR013": "mmr_ygn_pplp2_250k_mimu",          # Yangon
    "MMR014": "mmr_shn_pplp2_250k_unodc_mimu",    # Shan (South)
    "MMR015": "mmr_shn_pplp2_250k_unodc_mimu",    # Shan (North)
    "MMR016": "mmr_shn_pplp2_250k_unodc_mimu",    # Shan (East)
    "MMR017": "mmr_ady_pplp2_250k_mimu",          # Ayeyarwady
    "MMR018": "mmr_npt_pplp2_250k_mimu",          # Nay Pyi Taw
}

# Township polygons are simplified before they are used -- for the zonal
# statistics AND for the map, deliberately the same geometry, so the table and
# the drawn boundary can never disagree.  0.0005 deg is ~55 m, well inside the
# ~125 m positional accuracy of a 1:250,000 boundary, and it takes the 43
# clipped townships from ~1.9 MB of coordinates to ~0.2 MB.
TS_SIMPLIFY_DEG = 0.0005
TS_COORD_DP     = 5        # ~1.1 m -- rounding below the simplification error

# Townships whose overlap with the AOI is smaller than this are dropped: a
# 2 km2 clip of a 900 km2 township cannot support a "% of township flooded"
# figure, and listing it invites exactly that misreading.
MIN_TS_OVERLAP_KM2 = 5.0

# Village impact radii.  MIMU village points are settlement locations, not
# built-up outlines, so impact is expressed as new water within a radius of the
# point, never as "the village is under water".
VILLAGE_AT_M    = 150      # settlement neighbourhood (7.1 ha)
VILLAGE_NEAR_M  = 500      # surroundings / access (78.5 ha)
VILLAGE_AT_FRAC   = 0.05   # >= 5% of the 150 m disc  = 0.35 ha of new water
VILLAGE_NEAR_FRAC = 0.02   # >= 2% of the 500 m disc  = 1.57 ha of new water

# Features per reduceRegions request.  Zonal sums at 10 m are the heaviest
# server-side work in the script; batching keeps every request inside the
# synchronous 5-minute limit and lets a failure retry cheaply.
TS_BATCH      = 4
VILLAGE_BATCH = 400
REDUCE_TILE_SCALE = 4      # trades speed for memory headroom on GEE's workers
REDUCE_MAX_ATTEMPTS = 3

# Written to the repo root as index.html so GitHub Pages serves it directly.
# Never hand-edit index.html -- re-run this script to regenerate it.
OUTPUT_HTML = "index.html"
ASSET_DIR   = "assets"        # page-local PNGs for layers too big to embed
TEMP_DIR    = "_gee_tmp"      # deleted automatically after the map is built
STATS_JSON  = "flood_stats.json"   # machine-readable copy of every figure

# Target pixels per download tile.  GEE's getDownloadURL rejects requests over
# ~48 MB; every layer is fetched as uint8 (1 byte/px), so ~15 M px per tile
# leaves generous headroom.  Tile size in degrees is derived from this.
TILE_TARGET_PX = 15_000_000

# Attempts per tile before giving up. Earth Engine download failures are
# usually transient; at ~100 tiles a run that cannot survive one is a run that
# rarely finishes.
TILE_MAX_ATTEMPTS = 4

# Max pixels in one displayed PNG chunk.  A 75 Mpx single image decodes to
# ~300 MB of RGBA in the browser and is a hard failure on mobile Safari;
# ~16 Mpx chunks decode to ~64 MB each and can be freed independently.
MAX_CHUNK_PX = 16_000_000

# Layers whose total PNG payload exceeds this are written to ASSET_DIR instead
# of being base64-embedded in the HTML.  Keeps index.html small enough to open
# instantly and to push to GitHub (hard 100 MB per-file limit).  At 462 Mpx even
# the binary masks run to several MB, and base64 inflates them by a third, so
# the threshold is low: the page itself should stay around a megabyte and open
# instantly, with the pixels streamed from ASSET_DIR as layers are switched on.
EMBED_MAX_MB = 3.0

# Rough free-space needed for the SAR run (tiles + merged GeoTIFF + PNGs).
DISK_WARN_GB = 3.0

DEG_PER_M = 1.0 / 111_320.0   # approx degrees of latitude per metre

# ── Display palette ───────────────────────────────────────────────────────────
# Map marks are one fixed categorical set, identical in both page themes,
# because they are baked into the PNGs.  Validated all-pairs: worst CVD
# separation (deutan) dE 15.3, worst normal-vision dE 20.8, both far above the
# 8 / 15 floors.  Yellow is under 3:1 on a light surface, so every swatch is
# label-bearing and a table view of the same numbers is always present.
C_FLOOD    = "#e34948"      # new flood inundation (raster)
C_WATER    = "#2a78d6"      # JRC permanent water (raster)
C_VILLAGE  = "#eda100"      # affected village points
# Single-hue sequential ramp for the township choropleth: L 0.94 -> 0.39, hue
# held at 287-295 deg.  Neither blue nor red: blue is permanent water and red
# is the flood raster itself.  An orange ramp was tried first and failed in
# practice -- most townships here are 10-30% flooded, so the whole AOI filled
# with mid-ramp orange and the 10 m red flood pixels stopped reading against
# it.  The choropleth is the summary; it must sit UNDER the evidence, not
# camouflage it.
CHORO_RAMP = ["#ece9f8", "#d5cff0", "#b9afe6", "#9a8dd8",
              "#7d6cc4", "#5f4ea8", "#443586"]
# Upper bounds, in % of the township inside the AOI that is newly flooded.
CHORO_BREAKS = [0.5, 1, 3, 7, 15, 30]

# ──────────────────────────────────────────────────────────────────────────────
# 0. AOI LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_aoi(path: str) -> dict:
    """
    Read an AOI from GeoJSON and return everything the rest of the script needs.

    Accepts a FeatureCollection, a single Feature, or a bare geometry so the
    AOI file can be swapped without touching code.
    """
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)

    if gj.get("type") == "FeatureCollection":
        features = gj["features"]
    elif gj.get("type") == "Feature":
        features = [gj]
    else:                                    # bare geometry
        features = [{"type": "Feature", "geometry": gj, "properties": {}}]

    if not features:
        raise RuntimeError(f"{path} contains no features.")

    geoms = [f["geometry"] for f in features]
    props = features[0].get("properties", {}) or {}

    # Union multiple features into one analysis geometry, server side and local.
    ee_geom = ee.Geometry(geoms[0])
    for g in geoms[1:]:
        ee_geom = ee_geom.union(ee.Geometry(g), maxError=1)

    shp = shape(geoms[0])
    for g in geoms[1:]:
        shp = shp.union(shape(g))

    b = shp.bounds
    bbox = {"west": b[0], "south": b[1], "east": b[2], "north": b[3]}

    # An AOI file exported from MIMU carries TS/DT/ST per feature but no
    # Name_EN, so falling back to the filename gives something like
    # "merged_6_townships". Build a real name out of the attributes instead.
    name = props.get("Name_EN") or props.get("name")
    if not name:
        townships = sorted({f.get("properties", {}).get("TS")
                            for f in features
                            if f.get("properties", {}).get("TS")})
        regions = sorted({f.get("properties", {}).get("ST")
                          for f in features
                          if f.get("properties", {}).get("ST")})
        if townships:
            listed = ", ".join(townships[:6])
            if len(townships) > 6:
                listed += f" and {len(townships) - 6} more"
            name = (f"{len(townships)} MIMU township"
                    f"{'s' if len(townships) != 1 else ''}: {listed}"
                    + (f" ({', '.join(regions)})" if regions else ""))
    if not name:
        name = os.path.splitext(os.path.basename(path))[0]

    return {
        "ee_geom": ee_geom,
        "shape":   shp,
        "geojson": {"type": "FeatureCollection", "features": features},
        "bbox":    bbox,
        "name":    name,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 0b. MIMU ADMINISTRATIVE DATA  (townships + village points)
# ──────────────────────────────────────────────────────────────────────────────

def _wfs_url(layer: str) -> str:
    return (f"{MIMU_WFS}?service=WFS&version=1.0.0&request=GetFeature"
            f"&typeName=geonode:{layer}&outputFormat=application/json"
            f"&srsName=EPSG:4326")


def fetch_mimu_layer(layer: str) -> dict:
    """
    Load one MIMU layer as GeoJSON, from DATA_DIR if it is already there.

    The cache is what makes a re-run reproducible: the flood figures are joined
    to a fixed snapshot of the boundaries, not to whatever the WFS serves that
    day.  Delete the cached file to refresh a layer.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{layer}.json")

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    print(f"  Downloading MIMU layer {layer} ...")
    req = urllib.request.Request(_wfs_url(layer),
                                 headers={"User-Agent": "flood-mapping/2.0"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Could not download MIMU layer '{layer}' from {MIMU_WFS} "
                    f"({exc}).\nPut a GeoJSON copy at {path} and re-run."
                ) from exc
            print(f"    attempt {attempt} failed ({exc}); retrying ...")
            time.sleep(5 * attempt)

    gj = json.loads(raw.decode("utf-8"))
    with open(path, "wb") as fh:
        fh.write(raw)
    print(f"    cached -> {path} ({len(gj.get('features', []))} features)")
    return gj


def _round_geom(geom, dp: int):
    """GeoJSON mapping of a shapely geometry with coordinates rounded to dp."""
    def walk(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], dp), round(c[1], dp)]
        return [walk(x) for x in c]

    gj = mapping(geom)
    return {"type": gj["type"], "coordinates": walk(gj["coordinates"])}


def _km2(geom) -> float:
    """Rough planar area in km2 -- only used to reject slivers before GEE runs.

    Every reported area comes from ee.Image.pixelArea() on the sphere; this is
    a local screening estimate, so a cos(lat) scaling is accurate enough.
    """
    lat = geom.centroid.y
    return geom.area * (111.32 ** 2) * math.cos(math.radians(lat))


def load_townships(aoi_shape) -> list:
    """
    MIMU adm3 townships clipped to the AOI, simplified, ready for GEE and Leaflet.

    Returns one dict per township, sorted by name, each carrying its P-code, the
    clipped geometry as GeoJSON, and a planar area estimate for screening.
    """
    if os.path.exists(TOWNSHIP_GEOJSON):
        print(f"  Townships: local {TOWNSHIP_GEOJSON}")
        with open(TOWNSHIP_GEOJSON, encoding="utf-8") as fh:
            gj = json.load(fh)
    else:
        gj = fetch_mimu_layer(TOWNSHIP_LAYER)

    prepared = prep(aoi_shape)
    out, dropped = [], []

    for feat in gj["features"]:
        if not feat.get("geometry"):
            continue
        geom = shape(feat["geometry"])
        if not prepared.intersects(geom):
            continue

        clipped = geom.intersection(aoi_shape)
        if clipped.is_empty:
            continue
        if not clipped.is_valid:
            clipped = clipped.buffer(0)

        p = feat["properties"]
        area = _km2(clipped)
        row = {
            "pcode":     p.get("TS_PCODE", ""),
            "ts":        p.get("TS", ""),
            "ts_mmr":    p.get("TS_MMR", ""),
            "dt":        p.get("DT", ""),
            "st":        p.get("ST", ""),
            "st_pcode":  p.get("ST_PCODE", ""),
            "share_of_township": clipped.area / geom.area if geom.area else 0.0,
        }

        if area < MIN_TS_OVERLAP_KM2:
            dropped.append((row["ts"], area))
            continue

        simple = clipped.simplify(TS_SIMPLIFY_DEG, preserve_topology=True)
        if simple.is_empty or not simple.is_valid:
            simple = clipped
        row["geom"]      = _round_geom(simple, TS_COORD_DP)
        row["est_km2"]   = area
        out.append(row)

    out.sort(key=lambda r: (r["st"], r["dt"], r["ts"]))
    print(f"  Townships intersecting AOI: {len(out)}"
          + (f"  ({len(dropped)} sliver(s) under {MIN_TS_OVERLAP_KM2:g} km2 "
             f"dropped: {', '.join(t for t, _ in dropped)})" if dropped else ""))
    return out


def load_towns(aoi_shape, townships: list) -> list:
    """
    MIMU town points inside the AOI, in the same shape as village records.

    Deliberately the same record shape, so village_stats() and the impact
    thresholds apply unchanged -- a town is sampled exactly like a village, and
    the two differ only in how the dashboard draws them.
    """
    idx_of = {t["pcode"]: i for i, t in enumerate(townships)}
    gj = fetch_mimu_layer(TOWN_POINTS_LAYER)
    prepared = prep(aoi_shape)
    towns = []
    for feat in gj["features"]:
        g = feat.get("geometry")
        if not g or g.get("type") != "Point" or not prepared.contains(shape(g)):
            continue
        p = feat["properties"]
        towns.append({
            "lon": g["coordinates"][0], "lat": g["coordinates"][1],
            "name": p.get("Town") or "(unnamed)",
            "name_mmr": p.get("Town_MMR4") or "",
            "vt": p.get("Level") or "",
            "pcode": p.get("Town_Pcode"),
            "ts": idx_of.get(p.get("TS_Pcode"), -1),
            "ts_pcode": p.get("TS_Pcode", ""),
        })
    print(f"  Town points inside AOI: {len(towns)}"
          + (f"  ({', '.join(t['name'] for t in towns[:6])})" if towns else ""))
    return towns


def load_villages(aoi_shape, townships: list) -> list:
    """
    MIMU village points inside the AOI, joined to townships by P-code.

    The join is by TS_PCODE, never by name -- village and township names repeat
    across Myanmar, P-codes do not.  Points whose township was dropped as a
    sliver keep ts = -1: they still count in the AOI totals, they just have no
    township row to sit in.
    """
    idx_of = {t["pcode"]: i for i, t in enumerate(townships)}
    layers, seen = [], set()
    for t in townships:
        lyr = VILLAGE_LAYERS.get(t["st_pcode"])
        if lyr and lyr not in seen:
            seen.add(lyr)
            layers.append((t["st"], lyr))

    prepared = prep(aoi_shape)
    villages = []
    for st, lyr in layers:
        gj = fetch_mimu_layer(lyr)
        n_in = 0
        for feat in gj["features"]:
            g = feat.get("geometry")
            if not g or g.get("type") != "Point":
                continue
            lon, lat = g["coordinates"][0], g["coordinates"][1]
            if not prepared.contains(shape(g)):
                continue
            p = feat["properties"]
            villages.append({
                "lon":   lon,
                "lat":   lat,
                "name":  p.get("VILLAGE") or "(unnamed)",
                "name_mmr": p.get("VLG_MMR") or "",
                "vt":    p.get("VT") or "",
                "pcode": p.get("VLG_PCODE"),
                "ts":    idx_of.get(p.get("TS_PCODE"), -1),
                "ts_pcode": p.get("TS_PCODE", ""),
            })
            n_in += 1
        print(f"    {lyr:34s} {n_in:>6,d} point(s) inside AOI   ({st})")

    print(f"  Village points inside AOI: {len(villages):,}")
    return villages

# ──────────────────────────────────────────────────────────────────────────────
# 1. GEE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

def init_gee() -> None:
    kwargs = {"project": GEE_PROJECT} if GEE_PROJECT else {}
    try:
        ee.Initialize(**kwargs)
        print("[OK] GEE initialised.")
    except ee.EEException:
        print("[!] Not authenticated -- running ee.Authenticate() ...")
        ee.Authenticate()
        ee.Initialize(**kwargs)
        print("[OK] GEE initialised.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. SENTINEL-1 RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def _date_range(center_date: str, window: int) -> tuple:
    """Inclusive +/-window-day range as (start, end) for ee filterDate.

    filterDate's end is EXCLUSIVE, so the end bound is advanced one extra day.
    With window = 0 this is exactly the one UTC day named -- which is the point:
    the requested date cannot be silently swapped for a neighbouring pass.
    """
    return (ee.Date(center_date).advance(-window, "day"),
            ee.Date(center_date).advance(window + 1, "day"))


def _s1_filtered(cid: str, aoi: ee.Geometry, d_start, d_end,
                 orbit_pass, rel_orbit) -> ee.ImageCollection:
    """Sentinel-1 IW dual-pol scenes matching the AOI, dates and geometry.

    orbit_pass=None drops the pass filter; rel_orbit=None uses every ground
    track of the chosen pass.
    """
    c = (
        ee.ImageCollection(cid)
        .filterBounds(aoi)
        .filterDate(d_start, d_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if orbit_pass is not None:
        c = c.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    if rel_orbit is not None:
        c = c.filter(ee.Filter.eq("relativeOrbitNumber_start", rel_orbit))
    return c.select(["VV", "VH"])


def _to_db(img: ee.Image) -> ee.Image:
    """Linear backscatter power -> dB, matching COPERNICUS/S1_GRD exactly.

    Non-positive samples are masked first: log10(0) is -Infinity and would
    otherwise poison the focal-mean speckle filter for a whole neighbourhood.
    """
    img = img.updateMask(img.gt(0))
    return (img.log10().multiply(10)
            .copyProperties(img, ["system:time_start"]))


def load_s1(aoi: ee.Geometry, center_date: str, window: int,
            orbit_pass: str = ORBIT_PASS, rel_orbit=REL_ORBIT,
            cid: str = "COPERNICUS/S1_GRD", is_linear: bool = False) -> ee.Image:
    """Mean composite, in dB, of Sentinel-1 IW GRD scenes on the given date(s).

    Both composites must share the same viewing geometry.  orbit_pass fixes
    ASCENDING vs DESCENDING; rel_orbit pins one relative orbit / ground track so
    the pair uses the identical acquisition path -- the cleanest possible basis
    for change detection.

    When the source is linear (S1_GRD_FLOAT), each scene is converted to dB
    BEFORE compositing.  Order matters: mean(dB) != dB(mean) once more than one
    scene falls in the window, and mean-of-dB is what the dB collection would
    have given, so converting first keeps the two sources interchangeable.

    There is no fallback to the other orbit pass here.  preflight() has already
    established that this exact date/track pair exists; silently widening the
    filter after that would only ever produce a mixed-geometry composite.
    """
    d_start, d_end = _date_range(center_date, window)
    col = _s1_filtered(cid, aoi, d_start, d_end, orbit_pass, rel_orbit)
    if is_linear:
        col = col.map(_to_db)

    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No Sentinel-1 data on {center_date} "
            f"(pass={orbit_pass}, rel_orbit={rel_orbit}, window=+/-{window}d)."
        )

    track = f", track {rel_orbit}" if rel_orbit is not None else ""
    print(f"  [{center_date}] {n} scene(s) found{track} in {cid.split('/')[-1]}"
          f" -> mean composite (dB).")
    return col.mean().clip(aoi)

# ──────────────────────────────────────────────────────────────────────────────
# 2b. PRE-FLIGHT DATA AVAILABILITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

def s1_availability(aoi: ee.Geometry, center_date: str, window: int,
                    orbit_pass: str, rel_orbit, cid: str) -> tuple:
    """Scene count and AOI coverage fraction for one date on the chosen track."""
    d0, d1 = _date_range(center_date, window)
    col = _s1_filtered(cid, aoi, d0, d1, orbit_pass, rel_orbit).select(["VV"])

    n = col.size().getInfo()
    if n == 0:
        return 0, 0.0
    frac = col.mosaic().mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=200, maxPixels=1e10
    ).get("VV")
    return n, (ee.Number(frac).getInfo() or 0.0)


def recent_acquisitions(aoi: ee.Geometry, days_back: int = 50) -> list:
    """
    Every S1 IW acquisition over the AOI in the last days_back days, as a list
    of (date, pass, track, sources) -- one row per unique combination, with the
    short names of the collections that carry it.

    Both collections are scanned because they ingest at different speeds: the
    newest scene is often in _FLOAT alone, and a table built from S1_GRD only
    would wrongly report it as missing.
    """
    end   = ee.Date(POST_FLOOD_DATE).advance(3, "day")
    start = end.advance(-days_back, "day")

    rows = {}
    for cid, _ in S1_COLLECTIONS:
        col = (
            ee.ImageCollection(cid)
            .filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .sort("system:time_start")
        )
        if col.size().getInfo() == 0:
            continue
        dated = col.map(lambda im: im.set(
            "_d", ee.Date(im.get("system:time_start")).format("YYYY-MM-dd")))
        info = ee.Dictionary({
            "d": dated.aggregate_array("_d"),
            "p": col.aggregate_array("orbitProperties_pass"),
            "t": col.aggregate_array("relativeOrbitNumber_start"),
        }).getInfo()
        for d, p, t in zip(info["d"], info["p"], info["t"]):
            rows.setdefault((d, p, int(t)), set()).add(cid.split("/")[-1])

    return [(d, p, t, sorted(src)) for (d, p, t), src in sorted(rows.items())]


def ingestion_frontier() -> dict:
    """
    Latest acquisition time present in each S1 collection, worldwide.

    This is what separates "the satellite has not passed yet" from "the pass
    happened but Earth Engine has not ingested it yet".  Only the second one is
    fixed by waiting a few hours, so the distinction belongs in the message the
    user gets.
    """
    out = {}
    for cid, _ in S1_COLLECTIONS:
        col = ee.ImageCollection(cid).filterDate(
            ee.Date(POST_FLOOD_DATE).advance(-6, "day"),
            ee.Date(POST_FLOOD_DATE).advance(6, "day"))
        n = col.size().getInfo()
        latest = (ee.Date(col.aggregate_max("system:time_start"))
                  .format("YYYY-MM-dd HH:mm").getInfo()) if n else None
        out[cid.split("/")[-1]] = latest
    return out


def post_pass_age_hours(aoi: ee.Geometry, cid: str):
    """
    Hours since the post-flood pass crossed this AOI, or None if unknowable.

    Uses the acquisition time of whatever is already ingested for that date; if
    nothing is, it borrows the time of day from the PRE-flood pass. That is
    exact rather than approximate whenever the pair shares a relative orbit:
    the satellite crosses this AOI at the same UTC time of day on both dates.
    """
    d0, d1 = _date_range(POST_FLOOD_DATE, DATE_WINDOW)
    t_ms = _s1_filtered(cid, aoi, d0, d1, ORBIT_PASS,
                        REL_ORBIT).aggregate_max("system:time_start").getInfo()
    if t_ms is None:
        p0, p1 = _date_range(PRE_FLOOD_DATE, DATE_WINDOW)
        t_ms = _s1_filtered(cid, aoi, p0, p1, ORBIT_PASS,
                            REL_ORBIT).aggregate_max("system:time_start").getInfo()
        if t_ms is None:
            return None
        gap = (date.fromisoformat(POST_FLOOD_DATE)
               - date.fromisoformat(PRE_FLOOD_DATE)).days
        t_ms += gap * 86_400_000
    return (datetime.now(timezone.utc).timestamp() * 1000 - t_ms) / 3_600_000



def post_track(aoi: ee.Geometry, cid: str):
    """The relative orbit that actually carries the post-flood date, or None."""
    d0, d1 = _date_range(POST_FLOOD_DATE, DATE_WINDOW)
    col = _s1_filtered(cid, aoi, d0, d1, ORBIT_PASS, REL_ORBIT)
    tracks = col.aggregate_array("relativeOrbitNumber_start").getInfo() or []
    return int(max(set(tracks), key=tracks.count)) if tracks else None


def baseline_candidates(aoi: ee.Geometry, cid: str, track: int,
                        n_back: int = 5) -> list:
    """
    Earlier repeats of the post-flood track, with measured AOI coverage.

    A change-detection pair has to share a ground track, so the usable
    baselines for a given post date are exactly that track's previous repeats.
    Sentinel-1 repeats every 12 days, so stepping back in 12-day increments
    enumerates them; each is then measured rather than assumed, because a track
    can clip an AOI at one latitude and cover it at another.
    """
    out = []
    for k in range(1, n_back + 1):
        d = (date.fromisoformat(POST_FLOOD_DATE)
             - timedelta(days=12 * k)).isoformat()
        n, cov = s1_availability(aoi, d, DATE_WINDOW, ORBIT_PASS, track, cid)
        out.append((d, n, cov, k))
    return out


def frontier_ms() -> int:
    """Acquisition time of the newest scene anywhere on Earth, in epoch ms."""
    col = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT").filterDate(
        ee.Date(POST_FLOOD_DATE).advance(-4, "day"),
        ee.Date(POST_FLOOD_DATE).advance(6, "day"))
    return col.aggregate_max("system:time_start").getInfo()


def expected_pass_ms(aoi: ee.Geometry, cid: str):
    """
    When the post-flood pass crosses this AOI, in epoch ms, or None.

    Taken from the PRE-flood pass on the same relative orbit and shifted by the
    whole number of repeat cycles between the dates: same track means the same
    UTC time of day, so this is exact rather than estimated.
    """
    d0, d1 = _date_range(PRE_FLOOD_DATE, DATE_WINDOW)
    t = _s1_filtered(cid, aoi, d0, d1, ORBIT_PASS,
                     REL_ORBIT).aggregate_max("system:time_start").getInfo()
    if t is None:
        return None
    gap = (date.fromisoformat(POST_FLOOD_DATE)
           - date.fromisoformat(PRE_FLOOD_DATE)).days
    return t + gap * 86_400_000


def track_history(aoi: ee.Geometry, track: int, days_back: int = 90) -> list:
    """
    Dates the configured track appears in GEE over this AOI, oldest first.

    Reports GOOGLE EARTH ENGINE, never the ESA archive. GEE routinely runs days
    behind for a region and fills in later -- dates that looked permanently
    absent on 2026-09-01 (08-23, 08-26, 08-31) had all arrived by 09-04 -- so a
    gap here means "not in GEE yet", never "not acquired".
    """
    end = ee.Date(POST_FLOOD_DATE).advance(2, "day")
    col = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
           .filterBounds(aoi).filterDate(end.advance(-days_back, "day"), end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("relativeOrbitNumber_start", track)))
    if col.size().getInfo() == 0:
        return []
    dated = col.map(lambda im: im.set(
        "_d", ee.Date(im.get("system:time_start")).format("YYYY-MM-dd")))
    return sorted(set(dated.aggregate_array("_d").getInfo()))



def choose_collection(aoi: ee.Geometry) -> tuple:
    """
    Pick the collection to read BOTH composites from.

    Preference order is S1_COLLECTIONS, but only among collections that carry
    both dates at >= MIN_AOI_COVERAGE.  A collection mid-ingestion can hold part
    of a pass -- 2 of 3 scenes is 95% of this AOI -- and picking it costs the
    missing slice permanently: the VV difference is masked there, so it is
    reported as unassessed rather than analysed.  When the other collection
    already has the whole pass, that loss is pure choice, not a data limit.

    If nothing clears the bar, the collection with the best worst-date coverage
    is used and the shortfall is stated, since for some AOIs no single pass
    covers everything and a partial map is still the right answer.

    Returns (cid, is_linear, report) where report maps date -> (n, coverage),
    or (None, None, {}) if no collection carries both dates at all.
    """
    surveyed = []
    for cid, is_linear in S1_COLLECTIONS:
        report, ok = {}, True
        for d in (PRE_FLOOD_DATE, POST_FLOOD_DATE):
            n, cov = s1_availability(aoi, d, DATE_WINDOW,
                                     ORBIT_PASS, REL_ORBIT, cid)
            report[d] = (n, cov)
            ok = ok and n > 0
        state = ", ".join(
            f"{d}: {v[0] or 'none'}" + (f" ({v[1]:.0%})" if v[0] else "")
            for d, v in report.items())
        print(f"    {cid.split('/')[-1]:14s} {state}")
        if ok:
            surveyed.append((cid, is_linear, report,
                             min(cov for _, cov in report.values())))

    if not surveyed:
        return None, None, {}, 0.0

    for cid, is_linear, report, worst in surveyed:
        if worst >= MIN_AOI_COVERAGE:
            return cid, is_linear, report, worst

    cid, is_linear, report, worst = max(surveyed, key=lambda s: s[3])
    return cid, is_linear, report, worst


NEWLINE = chr(10)


def preflight(aoi: ee.Geometry) -> tuple:
    """
    Verify both dates have imagery on the configured track before doing any
    heavy work, and decide which collection to read.

    Sentinel-1 reaches the Earth Engine catalogue a few hours to a few days
    after acquisition, and the two collections do not land together, so a very
    recent post-flood date may be in _FLOAT only -- or in neither yet.  Failing
    here, with the real catalogue contents printed, beats crashing halfway
    through a multi-gigabyte download run, and beats quietly mapping some other
    date's flood.
    """
    print(f"\n> Pre-flight: checking {ORBIT_PASS} track {REL_ORBIT} availability ...")
    cid, is_linear, report, worst = choose_collection(aoi)

    if cid is None:
        rows = recent_acquisitions(aoi)
        table = "\n".join(
            f"         {d}   {p:<10s} track {t:<4d} {'+'.join(src)}"
            + ("   <- configured track" if (t == REL_ORBIT and p == ORBIT_PASS) else "")
            for d, p, t, src in rows
        ) or "         (nothing found)"
        front = "\n".join(
            f"         {k:<14s} latest scene anywhere on Earth: {v or 'n/a'} UTC"
            for k, v in ingestion_frontier().items())
        missing = [d for d in (PRE_FLOOD_DATE, POST_FLOOD_DATE)
                   if not any(r[0] == d for r in rows)]

        # THE decisive test, and it is a comparison, not a guess: has Earth
        # Engine ingested past the moment this pass crosses the AOI?
        #   frontier < pass time  -> the scene cannot be here yet. Wait.
        #   frontier > pass time  -> GEE skipped it, at least for now. GEE is
        #                            NOT ESA, so check Copernicus Browser
        #                            before concluding the image does not exist.
        cid0 = S1_COLLECTIONS[0][0]
        exp = expected_pass_ms(aoi, cid0)
        front = frontier_ms()
        hist = track_history(aoi, REL_ORBIT)
        hist_txt = ", ".join(hist) if hist else "(none in the last 90 days)"

        def _fmt(ms):
            return (ee.Date(ms).format("YYYY-MM-dd HH:mm").getInfo()
                    if ms else "unknown")

        # A margin, not a knife-edge. Ingestion is not strictly ordered,
        # so a frontier that has only just passed the acquisition moment
        # means this segment is still in the pipeline. Measured live on
        # 2026-09-04: frontier 23:40 UTC against a 23:39 UTC pass -- one
        # minute past, and the scene was plainly still landing. Only a
        # frontier well clear of the pass says GEE has moved on without it.
        FRONTIER_MARGIN_MS = 12 * 3_600_000
        if exp and front and front < exp + FRONTIER_MARGIN_MS:
            lag_h = (exp - front) / 3_600_000
            near = ("is only %.0f h short of it" % lag_h if lag_h > 0
                    else "has just reached it (%.0f h past)" % -lag_h)
            verdict = (
                f"\n       THIS IS ORDINARY INGESTION LAG -- WAIT.\n"
                f"       The pass crosses this AOI at {_fmt(exp)} UTC.\n"
                f"       Earth Engine has ingested to {_fmt(front)} UTC and "
                f"{near}.\n"
                f"       Expect it within hours. Re-run "
                f"unchanged later, or:\n"
                f"           python watch_and_run.py\n")
        else:
            verdict = (
                f"\n       Earth Engine has ingested past this pass "
                f"({_fmt(front)} UTC vs a pass at\n"
                f"       {_fmt(exp)} UTC), so the scene is missing from GEE "
                f"rather than merely late.\n"
                f"       GEE IS NOT ESA and routinely runs days behind for a "
                f"region before filling\n"
                f"       in. Confirm at browser.dataspace.copernicus.eu -- if "
                f"ESA has the scene,\n"
                f"       this is a GEE delay or gap, not a missing "
                f"acquisition.\n\n"
                f"       Track {REL_ORBIT} in GEE over this AOI, last 90 days:"
                f"\n         {hist_txt}\n")

        verdict += (f"\n       Nothing was substituted and nothing was "
                    f"written.\n")

        raise SystemExit(
            f"\n[STOP] Requested date not in the Earth Engine catalogue: "
            f"{', '.join(missing) or 'see table'}\n"
            f"       (pass={ORBIT_PASS}, track={REL_ORBIT}, "
            f"window=+/-{DATE_WINDOW}d -- an exact-date match by design)\n\n"
            f"       Nothing was substituted. This script will not build the\n"
            f"       post-flood composite from any other date.\n\n"
            f"       How far Earth Engine has ingested:\n{front}\n\n"
            f"       Acquisitions over this AOI currently in the catalogue:\n"
            f"{table}\n\n"
            f"{verdict}"
        )

    # Thin coverage has three quite different causes, and the fix differs for
    # each: wait (the pass is still landing), pick another baseline (this date's
    # track misses the AOI), or accept it (no pass covers the AOI at all).
    cov_pre = report[PRE_FLOOD_DATE][1]
    cov_post = report[POST_FLOOD_DATE][1]
    short = cid.split("/")[-1]

    if cov_pre < MIN_AOI_COVERAGE <= cov_post:
        track = post_track(aoi, cid) or REL_ORBIT
        cands = baseline_candidates(aoi, cid, track)
        table = NEWLINE.join(
            f"           {d}   {cov:6.1%}   {n} scene(s)   "
            f"{12 * k:>2d} days before {POST_FLOOD_DATE} "
            f"({k} repeat cycle{'s' if k != 1 else ''})"
            + ("   <- usable" if cov >= MIN_AOI_COVERAGE else "")
            for d, n, cov, k in cands)
        raise SystemExit(
            f"\n[STOP] The BASELINE date does not cover this AOI.\n"
            f"       {PRE_FLOOD_DATE} reaches only {cov_pre:.1%} of it, while "
            f"{POST_FLOOD_DATE} covers {cov_post:.1%}.\n\n"
            f"       This is not an ingestion delay -- it is swath geometry. "
            f"The two dates sit on\n"
            f"       different ground tracks, and the baseline's track clips "
            f"only the edge of this\n"
            f"       AOI. Waiting will not change it.\n\n"
            f"       A change-detection pair must share a track, so the usable "
            f"baselines are the\n"
            f"       earlier repeats of track {track}, the one that carries "
            f"{POST_FLOOD_DATE}:\n\n"
            f"{table}\n\n"
            f"       Set PRE_FLOOD_DATE to one of those and REL_ORBIT to "
            f"{track}, then re-run.\n"
            f"       A longer gap means a drier baseline and a larger measured "
            f"change; a shorter\n"
            f"       one isolates the most recent flooding. Nothing was "
            f"written.\n"
        )

    if worst < MIN_AOI_COVERAGE:
        age = post_pass_age_hours(aoi, cid)
        if age is not None and age < FRESH_PASS_HOURS:
            raise SystemExit(
                f"\n[STOP] {POST_FLOOD_DATE} is still being ingested -- only "
                f"{worst:.1%} of the AOI is covered.\n"
                f"       The pass crossed this AOI {age:.1f} h ago, and the best "
                f"collection ({short})\n"
                f"       holds {report[POST_FLOOD_DATE][0]} scene(s) so far. "
                f"Earth Engine fills a pass scene by\n"
                f"       scene over several hours, so this should keep climbing."
                f"\n\n"
                f"       Running now would map {worst:.0%} of the AOI and report "
                f"the other\n"
                f"       {1 - worst:.0%} as UNASSESSED -- not flood-free, just "
                f"unknown. Nothing was written.\n\n"
                f"       Wait and re-run, or let the watcher do it:\n"
                f"           python watch_and_run.py\n"
            )
        # Old enough that this IS the extent of the pass rather than a snapshot
        # of one landing: say so plainly and let the caller decide.
        print(f"    [!] No collection covers the whole AOI on both dates. "
              f"Using {short} at {worst:.1%}.")
        if age is not None:
            print(f"        The pass is {age:.0f} h old, past the "
                  f"{FRESH_PASS_HOURS} h ingestion window, so this is the "
                  f"swath itself, not a partial load.")
        print(f"        The missing {1 - worst:.1%} will be reported as "
              f"UNASSESSED, not as flood-free.")


    for d, (n, cov) in report.items():
        if cov < 0.5:
            print(f"    [!] {d}: only {cov:.0%} coverage -- most of the AOI "
                  f"cannot be assessed on this date.")

    short = cid.split("/")[-1]
    print(f"    [OK] both dates available in {short}.")
    if is_linear:
        print(f"    [i] {short} is linear power; every scene is converted with\n"
              f"        10*log10() before compositing, which reproduces S1_GRD\n"
              f"        exactly. Both dates come from this one collection.")
    return cid, is_linear, report

# ──────────────────────────────────────────────────────────────────────────────
# 3. SPECKLE FILTERING
# ──────────────────────────────────────────────────────────────────────────────

def speckle_filter(image: ee.Image, radius: int = SPECKLE_RADIUS) -> ee.Image:
    """Boxcar (focal-mean) speckle suppression."""
    return image.focal_mean(radius=radius, kernelType="circle", units="meters")

# ──────────────────────────────────────────────────────────────────────────────
# 4. FLOOD DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_floods(pre: ee.Image, post: ee.Image,
                  aoi: ee.Geometry,
                  thresh: float = FLOOD_DB_THRESH):
    """
    Returns
    -------
    flood_mask  binary (1 = newly flooded)
    perm_water  binary (1 = JRC permanent/seasonal water)
    diff_db     continuous VV change image (dB)
    """
    diff_db = post.select("VV").subtract(pre.select("VV")).rename("VV_change")

    # JRC Global Surface Water v1.4 -- seasonality >= 4 months/year.
    #
    # unmask(0) is essential: the seasonality band carries NO DATA over land
    # that has never been observed as water.  Because ee .And() intersects
    # masks, feeding the raw (masked) image into .And(perm_water.Not()) deletes
    # every never-been-water pixel from the flood mask -- i.e. exactly the dry
    # land a flood would newly cover.  Unmasking to 0 makes it a filter ("this
    # pixel is not permanent water") instead of a data-availability constraint.
    perm_water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("seasonality").gte(4).unmask(0).clip(aoi)
    )

    # HydroSHEDS terrain slope -- exclude steep pixels (unlikely to flood).
    # Left masked on purpose: the DEM has no data over sea, so its mask doubles
    # as a land mask and keeps open ocean out of the result.  That matters more
    # on this AOI than the last one: it reaches the Bay of Bengal at Gwa.
    slope = ee.Terrain.slope(ee.Image("WWF/HydroSHEDS/03VFDEM")).clip(aoi)

    flood_mask = (
        diff_db.lt(thresh)              # large VV decrease
        .And(perm_water.Not())          # not already open water
        .And(slope.lt(MAX_SLOPE_DEG))   # flat terrain only
        .rename("flood")
    )
    return flood_mask, perm_water, diff_db

# ──────────────────────────────────────────────────────────────────────────────
# 5. STATISTICS  (AOI-wide, per township, per village)
# ──────────────────────────────────────────────────────────────────────────────
#
# Every reduction below runs at EXPORT_SCALE (10 m).  Scale-dependent QA numbers
# are artifacts: an earlier version of this script reduced a coverage mask at
# 60 m and reported 97.7% of the AOI "assessed" where the true figure at 10 m
# was 99.98% -- pure boundary rasterisation.  Area is summed from
# ee.Image.pixelArea() rather than counted in pixels, so it is true surface area.

def _stat_image(flood_mask: ee.Image, perm_water: ee.Image,
                aoi: ee.Geometry) -> ee.Image:
    """Five area bands, in m2 per pixel, ready to be summed over any region.

    flood_a  new inundation
    valid_a  where both dates AND the DEM had data -- the analysed area
    land_a   where the terrain model has data at all -- i.e. land, not sea
    water_a  JRC permanent water (context, excluded from flood_a by definition)
    total_a  the region itself, so every share has a measured denominator

    land_a answers a question valid_a cannot: WHY was something not assessed.
    HydroSHEDS carries no data over the sea, and detect_floods leans on that on
    purpose -- the DEM mask doubles as a land mask and keeps open ocean out of
    the flood class. On a coastal AOI that shows up as a few per cent
    "unassessed", which reads like a data gap over land. valid_a / land_a is
    the share of LAND actually analysed; total_a - land_a is simply sea.

    EVERY band is clipped to the AOI so they share one footprint. Township
    polygons are simplified AFTER clipping, so their boundary can sit ~55 m
    outside the AOI; there, the global DEM still reports land while the
    AOI-clipped flood mask reports nothing. Unclipped, that thin rind invents a
    data gap and puts warning icons on townships that have none.
    """
    px = ee.Image.pixelArea().clip(aoi)
    land = ee.Terrain.slope(
        ee.Image("WWF/HydroSHEDS/03VFDEM")).mask().clip(aoi)
    return ee.Image.cat([
        flood_mask.unmask(0).multiply(px).rename("flood_a"),
        flood_mask.mask().multiply(px).rename("valid_a"),
        land.multiply(px).rename("land_a"),
        perm_water.multiply(px).rename("water_a"),
        px.rename("total_a"),
    ])


def _retry(fn, what: str, attempts: int = REDUCE_MAX_ATTEMPTS):
    """Run a GEE call, retrying the transient failures that plague big reducers."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:                       # noqa: BLE001
            if i == attempts:
                raise
            wait = 15 * i
            print(f"      [!] {what}: {type(exc).__name__} -- retry {i}/"
                  f"{attempts - 1} in {wait}s")
            time.sleep(wait)


def aoi_stats(stat_img: ee.Image, aoi: ee.Geometry,
              scale: int = EXPORT_SCALE) -> dict:
    """The four area bands summed over the whole AOI, in km2."""
    res = _retry(
        lambda: stat_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e11, tileScale=REDUCE_TILE_SCALE,
        ).getInfo(),
        "AOI reduction")
    return {k: (res.get(k) or 0.0) / 1e6 for k in
            ("flood_a", "valid_a", "land_a", "water_a", "total_a")}


def township_stats(stat_img: ee.Image, townships: list,
                   scale: int = EXPORT_SCALE) -> None:
    """
    Sum the area bands inside every township-clipped-to-AOI polygon, in place.

    Batched: a 10 m zonal sum over a 1,000 km2 township is 10 M pixels, and the
    synchronous API gives one request 5 minutes.  Small batches also make a
    retry cheap -- one failure costs TS_BATCH townships, not the whole run.
    """
    print(f"> Zonal statistics for {len(townships)} township(s) at {scale} m ...")
    done = 0
    for i in range(0, len(townships), TS_BATCH):
        chunk = townships[i:i + TS_BATCH]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(t["geom"]), {"p": t["pcode"]}) for t in chunk
        ])
        res = _retry(
            lambda fc=fc: stat_img.reduceRegions(
                collection=fc, reducer=ee.Reducer.sum(), scale=scale,
                tileScale=REDUCE_TILE_SCALE,
            ).getInfo(),
            f"townships {i + 1}-{i + len(chunk)}")

        by_pcode = {f["properties"]["p"]: f["properties"]
                    for f in res["features"]}
        for t in chunk:
            p = by_pcode.get(t["pcode"], {})
            t["flood_km2"] = (p.get("flood_a") or 0.0) / 1e6
            t["valid_km2"] = (p.get("valid_a") or 0.0) / 1e6
            t["land_km2"]  = (p.get("land_a") or 0.0) / 1e6
            t["water_km2"] = (p.get("water_a") or 0.0) / 1e6
            t["area_km2"]  = (p.get("total_a") or 0.0) / 1e6
            t["flood_pct"] = (100.0 * t["flood_km2"] / t["area_km2"]
                              if t["area_km2"] else 0.0)
            t["valid_frac"] = (t["valid_km2"] / t["area_km2"]
                               if t["area_km2"] else 0.0)
            # Share of LAND assessed -- the number that means something on a
            # coastal township, where area_km2 includes open sea.
            t["land_frac"] = (t["valid_km2"] / t["land_km2"]
                              if t["land_km2"] else 1.0)
        done += len(chunk)
        print(f"    {done}/{len(townships)} townships")


def village_stats(flood_mask: ee.Image, villages: list,
                  scale: int = EXPORT_SCALE) -> None:
    """
    Classify every village point by the new water around it, in place.

    MIMU village points are settlement locations, not built-up outlines, so the
    honest question is not "is this village under water" but "how much new water
    is there around this point".  Two radii are sampled:

      f_at    fraction of the VILLAGE_AT_M disc that is newly flooded
      f_near  fraction of the VILLAGE_NEAR_M disc that is newly flooded

    and a village is called inundated only above VILLAGE_AT_FRAC -- a handful of
    speckle pixels next to a settlement is not a flooded village.

    flag  2 = new water at the settlement
          1 = new water within VILLAGE_NEAR_M
          0 = none detected
         -1 = not assessed (no usable data on one of the two dates)
    """
    print(f"> Sampling {len(villages):,} MIMU village point(s) at {scale} m ...")
    img = ee.Image.cat([
        flood_mask.unmask(0).rename("f"),
        flood_mask.mask().rename("v"),
    ])

    for radius, key in ((VILLAGE_AT_M, "at"), (VILLAGE_NEAR_M, "near")):
        done = 0
        for i in range(0, len(villages), VILLAGE_BATCH):
            chunk = villages[i:i + VILLAGE_BATCH]
            fc = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([v["lon"], v["lat"]]).buffer(radius),
                    {"i": i + j})
                for j, v in enumerate(chunk)
            ])
            res = _retry(
                lambda fc=fc: img.reduceRegions(
                    collection=fc, reducer=ee.Reducer.mean(), scale=scale,
                    tileScale=REDUCE_TILE_SCALE,
                ).getInfo(),
                f"villages {key} {i + 1}-{i + len(chunk)}")

            for feat in res["features"]:
                p = feat["properties"]
                v = villages[p["i"]]
                v[f"f_{key}"] = p.get("f") or 0.0
                v[f"v_{key}"] = p.get("v") or 0.0
            done += len(chunk)
            print(f"    {key:<4s} {done:,}/{len(villages):,}")

    for v in villages:
        if v.get("v_near", 0.0) < 0.5:
            v["flag"] = -1
        elif v.get("f_at", 0.0) >= VILLAGE_AT_FRAC:
            v["flag"] = 2
        elif v.get("f_near", 0.0) >= VILLAGE_NEAR_FRAC:
            v["flag"] = 1
        else:
            v["flag"] = 0


def roll_up_villages(townships: list, villages: list) -> dict:
    """Village counts per township and for the AOI as a whole."""
    for t in townships:
        t["vil_total"] = t["vil_at"] = t["vil_near"] = t["vil_nodata"] = 0

    totals = {"total": len(villages), "at": 0, "near": 0, "nodata": 0}
    for v in villages:
        flag = v.get("flag", -1)
        totals["at"]     += flag == 2
        totals["near"]   += flag == 1
        totals["nodata"] += flag == -1
        ti = v["ts"]
        if 0 <= ti < len(townships):
            t = townships[ti]
            t["vil_total"]  += 1
            t["vil_at"]     += flag == 2
            t["vil_near"]   += flag == 1
            t["vil_nodata"] += flag == -1
    return totals

# ──────────────────────────────────────────────────────────────────────────────
# 6. DOWNLOAD GEE IMAGE -> LOCAL GEOTIFF
# ──────────────────────────────────────────────────────────────────────────────

def tile_deg_for(scale: int) -> float:
    """Tile edge in degrees that keeps one uint8 tile near TILE_TARGET_PX."""
    return math.sqrt(TILE_TARGET_PX) * scale * DEG_PER_M


def _download_tile(image: ee.Image, tile_path: str, scale: int,
                   region: ee.Geometry) -> bool:
    """
    Fetch one tile, retrying transient Earth Engine failures.

    geemap.ee_export_image does NOT raise on a failed download -- it prints and
    returns -- so success is judged by the file being present, non-empty and
    actually openable.  A truncated file is deleted so the retry starts clean,
    and so a later run does not mistake it for a cached success.
    """
    for attempt in range(1, TILE_MAX_ATTEMPTS + 1):
        try:
            geemap.ee_export_image(
                image, filename=tile_path, scale=scale, region=region,
                crs="EPSG:4326", file_per_band=False,
            )
        except Exception as exc:                      # noqa: BLE001
            print(f"      attempt {attempt} raised: {type(exc).__name__}: {exc}")

        if os.path.exists(tile_path) and os.path.getsize(tile_path) > 0:
            try:
                with rasterio.open(tile_path) as ds:
                    if ds.width and ds.height:
                        return True
            except Exception:                         # noqa: BLE001
                pass
            os.remove(tile_path)                      # truncated / unreadable

        if attempt < TILE_MAX_ATTEMPTS:
            wait = 10 * attempt
            print(f"      [!] attempt {attempt}/{TILE_MAX_ATTEMPTS} failed "
                  f"-- retrying in {wait}s ...")
            time.sleep(wait)

    return False


def download_as_geotiff(image: ee.Image, name: str, aoi_shape,
                        scale: int, bbox: dict) -> str:
    """
    Download a uint8 GEE image to TEMP_DIR/<name> (EPSG:4326).

    GEE's getDownloadURL API rejects requests larger than ~48 MB, which at 10 m
    is only a fraction of a degree.  The bbox is therefore split into tiles
    sized by tile_deg_for(), downloaded separately, then merged with
    rasterio.merge -- so any resolution works, including native 10 m.

    Tiles that do not touch the AOI polygon are never requested.  On a slanted
    quadrilateral like this one that is a quarter of the grid: the corners of
    the bbox are empty sea and empty Bago hills, and downloading them would cost
    minutes each for an all-nodata chunk that gets dropped later anyway.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    final_path = os.path.join(TEMP_DIR, name)

    if os.path.exists(final_path):
        print(f"  Using cached {name}.")
        return final_path

    tile_deg = tile_deg_for(scale)

    tile_defs, skipped = [], 0
    lat = bbox["south"]
    while lat < bbox["north"]:
        lat_end = min(lat + tile_deg, bbox["north"])
        lon = bbox["west"]
        while lon < bbox["east"]:
            lon_end = min(lon + tile_deg, bbox["east"])
            if shp_box(lon, lat, lon_end, lat_end).intersects(aoi_shape):
                tile_defs.append((lon, lat, lon_end, lat_end))
            else:
                skipped += 1
            lon = lon_end
        lat = lat_end

    n_tiles = len(tile_defs)
    print(f"  Downloading {name}: {n_tiles} tile(s) at {scale} m/px "
          f"({tile_deg:.3f} deg tiles"
          + (f", {skipped} outside AOI skipped" if skipped else "") + ") ...")

    tile_paths = []
    for i, (w, s, e, n) in enumerate(tile_defs, 1):
        tile_region = ee.Geometry.Rectangle([w, s, e, n])
        tile_path   = os.path.join(TEMP_DIR, f"_tile_{i:03d}_{name}")
        tile_paths.append(tile_path)

        if not os.path.exists(tile_path):
            print(f"    Tile {i}/{n_tiles} ...")
            if not _download_tile(image.clip(tile_region), tile_path,
                                  scale, tile_region):
                raise RuntimeError(
                    f"Tile {i}/{n_tiles} download failed for {name} after "
                    f"{TILE_MAX_ATTEMPTS} attempts.\n"
                    f"Completed tiles are cached in {TEMP_DIR}/, so re-running "
                    f"resumes from here rather than starting over.\n"
                    f"If it keeps failing on the same tile, lower "
                    f"TILE_TARGET_PX (currently {TILE_TARGET_PX})."
                )

    if n_tiles == 1:
        shutil.move(tile_paths[0], final_path)
    else:
        print(f"  Merging {n_tiles} tiles -> {name} ...")
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = rio_merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({
            "height":    mosaic.shape[1],
            "width":     mosaic.shape[2],
            "transform": transform,
            "compress":  "deflate",
        })
        for ds in datasets:
            ds.close()
        with rasterio.open(final_path, "w", **profile) as dst:
            dst.write(mosaic)
        for p in tile_paths:
            if os.path.exists(p):
                os.remove(p)

    print(f"  [OK] {name} saved ({os.path.getsize(final_path) / 1e6:,.0f} MB).")
    return final_path

# ──────────────────────────────────────────────────────────────────────────────
# 7. GEOTIFF -> PALETTE PNG CHUNKS  (page-local -- never expires)
# ──────────────────────────────────────────────────────────────────────────────
#
# Palette ("P" mode) PNGs are used rather than RGBA.  At 10 m this AOI's bbox
# is ~462 Mpx: RGBA would allocate nearly two gigabytes per layer in memory and
# compress far worse, while a palette image is 1 byte/px and lets PNG's filters
# do their job.
# Index 0 is reserved for "no data / nothing here" and made transparent.

def _hex_rgb(h: str) -> list:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _palette_solid(color_hex: str) -> list:
    """0 = transparent, 1 = the given colour."""
    return [0, 0, 0] + _hex_rgb(color_hex) + [0, 0, 0] * 254


def _palette_gray() -> list:
    pal = []
    for i in range(256):                  # 0 = transparent, 1..255 = grays
        pal += [i, i, i]
    return pal


def _encode_png(arr: np.ndarray, palette: list, fast: bool = False) -> bytes:
    """uint8 array -> palette PNG bytes, index 0 transparent.

    frombytes("P", ...) treats the buffer as raw palette indices, so values
    survive exactly.  Going via fromarray().convert("P") would invite PIL to
    re-quantise, which for the gray ramp would silently shift DN values.

    fast=True skips PIL's exhaustive filter search.  On a binary mask optimize
    is nearly free and worth it; on 16 Mpx of SAR speckle it costs minutes per
    chunk for about 1% -- so the heavy layers use the plain deflate path.
    """
    h, w = arr.shape
    img = Image.frombytes("P", (w, h), np.ascontiguousarray(arr).tobytes())
    img.putpalette(palette)
    buf = BytesIO()
    if fast:
        img.save(buf, format="PNG", compress_level=6, transparency=0)
    else:
        img.save(buf, format="PNG", optimize=True, transparency=0)
    return buf.getvalue()


def _chunk_grid(width: int, height: int, max_px: int = None) -> tuple:
    """Rows/cols needed so every chunk stays under max_px pixels."""
    max_dim = int(math.sqrt(MAX_CHUNK_PX if max_px is None else max_px))
    return (max(1, math.ceil(height / max_dim)),
            max(1, math.ceil(width  / max_dim)))


def clear_assets() -> None:
    """
    Drop PNGs written by a previous run so stale chunks can never be served.

    Only files matching the names this script generates are removed -- anything
    else a user has put in ASSET_DIR is left alone.
    """
    if not os.path.isdir(ASSET_DIR):
        return
    stale = glob.glob(os.path.join(ASSET_DIR, "*_[0-9][0-9]*.png"))
    for p in stale:
        os.remove(p)
    if stale:
        print(f"  Removed {len(stale)} PNG(s) from a previous run in {ASSET_DIR}/.")


def geotiff_chunks(path: str, palette: list, fast: bool) -> tuple:
    """
    Cut a uint8 GeoTIFF into palette-PNG chunks of at most MAX_CHUNK_PX pixels.

    Returns (pieces, skipped) where pieces is [(bounds, png_bytes), ...] with
    bounds in Leaflet order [[s,w],[n,e]].  Chunks that are entirely nodata --
    the bbox corners outside the AOI polygon -- are dropped rather than shipped
    as megabytes of transparent pixels.

    Windowed reads keep peak memory at one chunk rather than the whole raster,
    which matters at 10 m where a full layer here is ~462 Mpx.
    """
    with rasterio.open(path) as src:
        W, H = src.width, src.height
        b    = src.bounds                    # EPSG:4326 from the download step
        xres = (b.right - b.left) / W
        yres = (b.top   - b.bottom) / H

        n_rows, n_cols = _chunk_grid(W, H)
        r_step = math.ceil(H / n_rows)
        c_step = math.ceil(W / n_cols)

        pieces, skipped = [], 0
        for r in range(n_rows):
            r0, r1 = r * r_step, min(H, (r + 1) * r_step)
            if r0 >= r1:
                continue
            for c in range(n_cols):
                c0, c1 = c * c_step, min(W, (c + 1) * c_step)
                if c0 >= c1:
                    continue

                arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if not arr.any():            # all nodata -- nothing to draw
                    skipped += 1
                    continue

                bounds = [[b.top - r1 * yres, b.left + c0 * xres],
                          [b.top - r0 * yres, b.left + c1 * xres]]
                pieces.append((bounds, _encode_png(arr, palette, fast)))

    return pieces, skipped


def geotiff_to_layer(path: str, palette: list, key: str,
                     fast: bool = False) -> dict:
    """
    Read a uint8 GeoTIFF and turn it into browser-ready PNG chunks.

    Returns {"pieces": [(image_ref, [[s,w],[n,e]]), ...],
             "external": bool, "mb": float}

    image_ref is either a base64 data URI (small layers, embedded in the HTML)
    or a relative path into ASSET_DIR (big layers).  Neither points at a GEE
    tile server, so neither expires.
    """
    with rasterio.open(path) as src:
        W, H = src.width, src.height
    pieces, skipped = geotiff_chunks(path, palette, fast)

    total_mb = sum(len(p) for _, p in pieces) / 1e6
    external = total_mb > EMBED_MAX_MB

    refs = []
    for i, (bounds, png) in enumerate(pieces, 1):
        if external:
            os.makedirs(ASSET_DIR, exist_ok=True)
            fname = f"{key}_{i:02d}.png"
            with open(os.path.join(ASSET_DIR, fname), "wb") as fh:
                fh.write(png)
            ref = f"{ASSET_DIR}/{fname}"     # forward slash: this is a URL
        else:
            ref = "data:image/png;base64," + base64.b64encode(png).decode()
        refs.append([ref, bounds])

    where = f"{ASSET_DIR}/" if external else "embedded"
    print(f"    {key:9s} {W}x{H} px -> {len(refs)} chunk(s), "
          f"{total_mb:,.1f} MB, {where}"
          + (f" ({skipped} empty chunk(s) dropped)" if skipped else ""))

    return {"pieces": refs, "external": external, "mb": total_mb}


def build_layer(image: ee.Image, name: str, key: str, palette: list,
                aoi_shape, scale: int, bbox: dict, fast: bool,
                keep_tif: bool) -> dict:
    """Download one layer, encode it, and drop the GeoTIFF before the next.

    Holding all four native-resolution GeoTIFFs at once needs ~1.9 GB; doing
    them one at a time needs ~0.5 GB, and the PNGs are the actual deliverable.
    """
    path = download_as_geotiff(image, name, aoi_shape, scale, bbox)
    layer = geotiff_to_layer(path, palette, key, fast)
    if not keep_tif:
        os.remove(path)
    return layer

# ──────────────────────────────────────────────────────────────────────────────
# 8. DASHBOARD  (hand-built Leaflet page -- no GEE URLs, nothing expires)
# ──────────────────────────────────────────────────────────────────────────────

PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta name="description" content="__DESC__">
<meta property="og:title" content="__TITLE__">
<meta property="og:description" content="__DESC__">
<meta property="og:type" content="website">
<!-- Inline SVG favicon: GitHub Pages serves these files raw, and without one
     every visit also fetches a 404 for /favicon.ico. -->
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230d0d0d'/%3E%3Ccircle cx='16' cy='10' r='4.2' fill='%23e34948'/%3E%3Cpath d='M4 19c4-3 6 2 10-1s6 2 10-1' stroke='%232a78d6' stroke-width='3' fill='none' stroke-linecap='round'/%3E%3Cpath d='M4 26c4-3 6 2 10-1s6 2 10-1' stroke='%232a78d6' stroke-width='3' fill='none' stroke-linecap='round' opacity='.5'/%3E%3C/svg%3E">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
      crossorigin="">
<style>
/* ── tokens ──────────────────────────────────────────────────────────────── */
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --surface-2:#f2f1ed; --plane:#f9f9f7;
  --ink-1:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
  --rule:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --shadow:0 1px 2px rgba(11,11,11,.06),0 8px 24px rgba(11,11,11,.10);
  --track:#e8e7e1;
  --flood:__C_FLOOD__; --water:__C_WATER__; --village:__C_VILLAGE__;
  --warn:#fab219; --crit:#d03b3b;
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232322; --plane:#0d0d0d;
  --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-3:#898781;
  --rule:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --shadow:0 1px 2px rgba(0,0,0,.5),0 8px 24px rgba(0,0,0,.55);
  --track:#2c2c2a;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--ink-1); background:var(--plane);
  -webkit-font-smoothing:antialiased;
}
#app{display:flex;height:100%;overflow:hidden}
#map{flex:1;min-width:0;background:var(--surface-2)}

/* ── panel ───────────────────────────────────────────────────────────────── */
#panel{
  position:relative;
  width:390px;flex:none;display:flex;flex-direction:column;
  background:var(--surface-1);border-right:1px solid var(--rule);
  box-shadow:var(--shadow);z-index:900;
}
#panel-head{padding:14px 16px 12px;border-bottom:1px solid var(--rule)}
.eyebrow{
  font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;display:flex;align-items:center;gap:8px;
}
.eyebrow .dot{width:8px;height:8px;border-radius:50%;background:var(--flood)}
h1{font-size:17px;line-height:1.25;margin:6px 0 2px;font-weight:650}
.sub{color:var(--ink-2);font-size:12.5px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}
.chip{
  font-size:11.5px;padding:2px 8px;border-radius:999px;
  border:1px solid var(--ring);color:var(--ink-2);background:var(--surface-2);
  white-space:nowrap;
}
.chip b{color:var(--ink-1);font-weight:600}
#theme{
  position:absolute;top:12px;right:12px;width:28px;height:28px;
  border:1px solid var(--ring);border-radius:8px;background:var(--surface-2);
  color:var(--ink-2);cursor:pointer;font-size:13px;line-height:1;
}
#theme:hover{color:var(--ink-1)}
#panel-head{position:relative}

/* ── hero + stat tiles ───────────────────────────────────────────────────── */
#hero{padding:14px 16px 4px}
.hero-label{font-size:12px;color:var(--ink-2)}
.hero-value{font-size:48px;line-height:1.05;font-weight:650;letter-spacing:-.02em}
.hero-value .u{font-size:19px;font-weight:550;color:var(--ink-2);margin-left:5px}
.hero-note{font-size:12px;color:var(--ink-3);margin-top:3px}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px 14px}
.tile{
  border:1px solid var(--ring);border-radius:10px;padding:8px 10px;
  background:var(--surface-2);
}
.tile .l{font-size:11px;color:var(--ink-2);line-height:1.3}
.tile .v{font-size:20px;font-weight:600;margin-top:2px}
.tile .s{font-size:11px;color:var(--ink-3)}

/* ── tabs ────────────────────────────────────────────────────────────────── */
#tabs{display:flex;gap:2px;padding:0 10px;border-bottom:1px solid var(--rule)}
.tab{
  appearance:none;border:0;background:none;cursor:pointer;
  font:inherit;font-size:12.5px;color:var(--ink-2);
  padding:9px 10px;border-bottom:2px solid transparent;
}
.tab:hover{color:var(--ink-1)}
.tab[aria-selected="true"]{color:var(--ink-1);font-weight:600;border-bottom-color:var(--flood)}
#views{flex:1;overflow-y:auto;overscroll-behavior:contain}
.view{display:none;padding:12px 16px 24px}
.view.on{display:block}

/* ── controls ────────────────────────────────────────────────────────────── */
.toolbar{display:flex;gap:6px;margin-bottom:10px}
input[type=search],select{
  font:inherit;font-size:12.5px;padding:6px 8px;color:var(--ink-1);
  background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;
}
input[type=search]{flex:1;min-width:0}
input[type=search]:focus,select:focus{outline:2px solid var(--water);outline-offset:-1px}
.seg{display:flex;border:1px solid var(--ring);border-radius:8px;overflow:hidden}
/* The UA rule for [hidden] is display:none, which an author display:flex
   silently outranks -- so el.hidden did nothing to the grouping control. */
.seg[hidden]{display:none}
.seg button{
  appearance:none;border:0;background:var(--surface-1);color:var(--ink-2);
  font:inherit;font-size:12px;padding:6px 9px;cursor:pointer;
}
.seg button[aria-pressed="true"]{background:var(--surface-2);color:var(--ink-1);font-weight:600}

/* ── ranked bar rows ─────────────────────────────────────────────────────── */
.rows{display:flex;flex-direction:column;gap:2px}
.row{
  display:grid;grid-template-columns:1fr auto;gap:4px 8px;align-items:baseline;
  padding:6px 7px;border-radius:8px;cursor:pointer;border:1px solid transparent;
}
.row:hover,.row.sel{background:var(--surface-2);border-color:var(--ring)}
.row .nm{font-size:12.5px;font-weight:550;line-height:1.25}
.row .nm small{font-weight:400;color:var(--ink-3);font-size:11px;display:block}
.row .val{font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.row .val small{display:block;font-weight:400;color:var(--ink-3);font-size:11px}
.row .bar{grid-column:1/3;height:10px;background:var(--track);border-radius:3px;overflow:hidden}
.row .bar i{display:block;height:100%;border-radius:0 4px 4px 0}
.flagged{color:var(--crit);font-weight:600}
.warnrow{font-size:11px;color:var(--ink-2);grid-column:1/3;display:flex;gap:5px;align-items:center}

/* ── table view ──────────────────────────────────────────────────────────── */
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:5px 6px;border-bottom:1px solid var(--rule);text-align:right;
      font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--ink-2);font-weight:600;text-align:right;position:sticky;top:0;
   background:var(--surface-1)}
th:first-child,td:first-child{text-align:left}
.tblwrap{overflow-x:auto}

/* ── layers tab ──────────────────────────────────────────────────────────── */
.lyr{display:flex;gap:9px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--rule)}
.lyr:last-child{border-bottom:0}
.lyr input{margin-top:3px;accent-color:var(--water)}
.lyr .t{flex:1;min-width:0}
.lyr .t b{font-weight:600;font-size:12.5px}
.lyr .t span{display:block;color:var(--ink-3);font-size:11px}
.sw{width:12px;height:12px;border-radius:3px;flex:none;margin-top:3px;
    border:1px solid var(--ring)}
input[type=range]{width:100%;accent-color:var(--water);margin-top:6px}
.h3{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
    font-weight:600;margin:14px 0 6px}
.h3:first-child{margin-top:0}

/* ── method tab ──────────────────────────────────────────────────────────── */
.kv{display:flex;justify-content:space-between;gap:10px;padding:4px 0;
    border-bottom:1px solid var(--rule);font-size:12.5px}
.kv span{color:var(--ink-2)}
.kv b{font-weight:600;text-align:right}
.note{font-size:12px;color:var(--ink-2);margin:10px 0 0;line-height:1.55}
.note b{color:var(--ink-1)}
.callout{
  border:1px solid var(--ring);border-left:3px solid var(--warn);
  background:var(--surface-2);border-radius:8px;padding:9px 11px;margin-top:12px;
  font-size:12px;color:var(--ink-2);line-height:1.55;
}
.callout b{color:var(--ink-1)}
a{color:var(--water)}

/* ── map furniture ───────────────────────────────────────────────────────── */
.leaflet-container{background:var(--surface-2);font-family:inherit}
.leaflet-image-layer.crisp{image-rendering:pixelated}
.leaflet-image-layer.smooth{image-rendering:auto;image-rendering:smooth}
#legend{
  position:absolute;right:12px;bottom:26px;z-index:1100;max-width:250px;
  background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;
  box-shadow:var(--shadow);font-size:11.5px;color:var(--ink-2);
}
#legend h4{margin:0;padding:8px 11px;font-size:11px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink-3);cursor:pointer;display:flex;gap:8px;
  align-items:center;user-select:none}
#legend h4 .c{margin-left:auto}
#legend .body{padding:0 11px 10px;border-top:1px solid var(--rule)}
#legend.min .body{display:none}
.lg{display:flex;gap:7px;align-items:center;margin-top:6px}
.lg i{width:12px;height:12px;border-radius:3px;flex:none;border:1px solid var(--ring)}
.lg .dot{border-radius:50%}
.ramp{display:flex;margin-top:7px;border-radius:3px;overflow:hidden}
.ramp i{height:10px;flex:1}
.ramp-ax{display:flex;justify-content:space-between;color:var(--ink-3);font-size:10.5px;margin-top:2px}
.leaflet-popup-content-wrapper,.leaflet-tooltip{
  background:var(--surface-1);color:var(--ink-1);border-radius:10px;
  box-shadow:var(--shadow);border:1px solid var(--ring)}
.leaflet-popup-tip{background:var(--surface-1)}
.leaflet-tooltip{font-size:12px;padding:6px 9px}
.leaflet-tooltip.town-lbl{
  background:rgba(255,255,255,.88);color:#1a1a19;border:0;box-shadow:none;
  font-size:11px;font-weight:650;padding:1px 5px;border-radius:4px}
.leaflet-tooltip:before{display:none}
.pop b{font-size:13px}
.pop .kv{font-size:12px}
.leaflet-control-attribution{background:var(--surface-1)!important;color:var(--ink-3)!important}
.leaflet-control-attribution a{color:var(--ink-2)!important}
.leaflet-bar a{background:var(--surface-1);color:var(--ink-1);border-color:var(--rule)}
.leaflet-bar a:hover{background:var(--surface-2)}
#readout{
  position:absolute;left:12px;bottom:40px;z-index:1100;font-size:11px;
  color:var(--ink-2);background:var(--surface-1);border:1px solid var(--ring);
  border-radius:8px;padding:3px 8px;font-variant-numeric:tabular-nums;
}
#sheet-toggle{display:none}

/* ── responsive: panel becomes a sheet ───────────────────────────────────── */
@media (max-width:820px){
  #app{flex-direction:column-reverse}
  #panel{width:auto;height:56%;border-right:0;border-top:1px solid var(--rule)}
  #map{height:44%}
  #legend{bottom:12px}
  #sheet-toggle{
    display:block;position:absolute;left:50%;transform:translateX(-50%);
    top:-13px;z-index:901;border:1px solid var(--ring);border-radius:999px;
    background:var(--surface-1);color:var(--ink-2);font:inherit;font-size:11px;
    padding:2px 12px;cursor:pointer;box-shadow:var(--shadow);
  }
  #app.collapsed #panel{height:44px;overflow:hidden}
  #app.collapsed #map{height:calc(100% - 44px)}
}
</style>
</head>
<body>
<div id="app">
  <aside id="panel">
    <button id="sheet-toggle" type="button">▾ hide</button>
    <div id="panel-head">
      <button id="theme" type="button" title="Switch light / dark">◐</button>
      <div class="eyebrow"><span class="dot"></span>Sentinel-1 flood mapping</div>
      <h1 id="ttl"></h1>
      <div class="sub" id="sub"></div>
      <div class="chips" id="chips"></div>
    </div>
    <div id="hero">
      <div class="hero-label">New inundation detected</div>
      <div class="hero-value"><span id="heroval"></span><span class="u">km²</span></div>
      <div class="hero-note" id="heronote"></div>
    </div>
    <div class="tiles" id="tiles"></div>
    <div id="tabs" role="tablist"></div>
    <div id="views">
      <section class="view on" id="v-ts">
        <div class="toolbar">
          <input type="search" id="ts-q" placeholder="Filter townships, districts, regions…" aria-label="Filter">
          <select id="ts-sort" aria-label="Sort">
            <option value="flood_km2">Flooded area</option>
            <option value="flood_pct">% flooded</option>
            <option value="vil_at">Villages affected</option>
            <option value="name">Name</option>
          </select>
        </div>
        <div class="toolbar">
          <div class="seg" role="group" aria-label="Group by">
            <button type="button" id="ts-g-ts" aria-pressed="true">Township</button>
            <button type="button" id="ts-g-dt" aria-pressed="false">District</button>
            <button type="button" id="ts-g-st" aria-pressed="false">Region</button>
          </div>
          <div class="seg" style="margin-left:auto" role="group" aria-label="View as">
            <button type="button" id="ts-bars" aria-pressed="true">Bars</button>
            <button type="button" id="ts-table" aria-pressed="false">Table</button>
          </div>
        </div>
        <div id="ts-rows" class="rows"></div>
        <div id="ts-tbl" class="tblwrap" hidden></div>
        <p class="note" id="ts-foot"></p>
      </section>
      <section class="view" id="v-vil">
        <div class="toolbar">
          <input type="search" id="vil-q" placeholder="Search village or township…" aria-label="Search villages">
          <select id="vil-filter" aria-label="Filter villages">
            <option value="2">Water at settlement</option>
            <option value="1">Water within 500 m</option>
            <option value="-1">Not assessed</option>
          </select>
        </div>
        <div id="vil-rows" class="rows"></div>
        <p class="note" id="vil-foot"></p>
      </section>
      <section class="view" id="v-lyr"></section>
      <section class="view" id="v-met"></section>
    </div>
  </aside>
  <main id="map">
    <div id="legend">
      <h4 id="lg-h">Legend<span class="c">▾</span></h4>
      <div class="body" id="lg-b"></div>
    </div>
    <div id="readout">–</div>
  </main>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
        crossorigin=""></script>
<script>
const FLOOD = __PAYLOAD__;
</script>
<script>
/* ═══ helpers ═════════════════════════════════════════════════════════════ */
const $  = (s, r) => (r || document).querySelector(s);
const el = (t, c, txt) => { const e = document.createElement(t);
  if (c) e.className = c; if (txt !== undefined) e.textContent = txt; return e; };
const num = (v, d) => v.toLocaleString(undefined,
  {minimumFractionDigits: d, maximumFractionDigits: d});
const km2 = v => v >= 100 ? num(v, 0) : v >= 10 ? num(v, 1) : num(v, 2);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const M = FLOOD.meta, S = FLOOD.stats, TS = FLOOD.townships;
const RAMP = M.ramp, BREAKS = M.breaks;

function choro(pct){
  if (!(pct > 0)) return null;
  for (let i = 0; i < BREAKS.length; i++) if (pct < BREAKS[i]) return RAMP[i];
  return RAMP[RAMP.length - 1];
}

/* ═══ theme ═══════════════════════════════════════════════════════════════ */
const mq = window.matchMedia('(prefers-color-scheme: dark)');
function setTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  try { localStorage.setItem('fm-theme', t); } catch (e) {}
}
setTheme((() => { try { return localStorage.getItem('fm-theme'); } catch (e) { return null; } })()
         || (mq.matches ? 'dark' : 'light'));
$('#theme').onclick = () =>
  setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');

/* ═══ header ══════════════════════════════════════════════════════════════ */
$('#ttl').textContent = M.title;
$('#sub').textContent = FLOOD.aoi.name;
[['Pre', M.pre], ['Post', M.post], [M.pass_short, 'track ' + M.track],
 ['Native', M.scale + ' m'], ['Source', M.source]]
  .forEach(([k, v]) => {
    const c = el('span', 'chip'); c.append(k + ' ');
    c.append(Object.assign(el('b'), {textContent: v})); $('#chips').append(c);
  });

$('#heroval').textContent = km2(S.flood_km2);
$('#heronote').textContent =
  num(S.flood_pct, 1) + '% of the ' + num(S.aoi_km2, 0) + ' km² area of interest · '
  + 'water that arrived between ' + M.pre + ' and ' + M.post;

const nAffected = TS.filter(t => t.flood_km2 > 0.5).length;
[['Townships with flooding', nAffected + ' / ' + TS.length,
  TS.every(t => t.share_of_township >= 0.999)
    ? 'MIMU adm3 — the AOI is these townships'
    : 'MIMU adm3, clipped to the AOI'],
 ['Villages with water at the settlement', num(S.vil_at, 0),
  'of ' + num(S.vil_total, 0) + ' MIMU village points'],
 ['Villages with water within 500 m', num(S.vil_near + S.vil_at, 0),
  'includes the settlements above'],
 ['Land assessed',
  num((S.land_frac !== undefined ? S.land_frac : S.valid_frac) * 100, 1) + '%',
  (S.land_frac !== undefined && S.land_frac > 0.995)
    ? (S.sea_km2 > 1 ? num(S.sea_km2, 0) + ' km² of sea excluded'
                     : 'both dates usable')
    : 'blank ≠ flood-free']
].forEach(([l, v, s]) => {
  const t = el('div', 'tile');
  t.append(el('div', 'l', l), el('div', 'v', v), el('div', 's', s));
  $('#tiles').append(t);
});

/* ═══ map ═════════════════════════════════════════════════════════════════ */
const map = L.map('map', {zoomControl: true, preferCanvas: true,
                          attributionControl: true});
map.fitBounds(FLOOD.aoi.bounds);

const bases = {
  'Satellite (Esri)': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {maxZoom: 19, attribution: 'Esri, Maxar, Earthstar Geographics'}),
  'Terrain (Esri)': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
    {maxZoom: 13, attribution: 'Esri'}),
  'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'})
};
let baseName = 'Satellite (Esri)';
bases[baseName].addTo(map);

/* Explicit panes fix the stack once and for all: Leaflet appends a re-added
   layer to the end of its pane, so without them switching the choropleth off
   and on again would leave its wash sitting on top of the flood raster. */
[['p-choro', 400], ['p-sar', 405], ['p-water', 410], ['p-flood', 415],
 ['p-line', 420], ['p-vil', 430]].forEach(([n, z]) => {
  map.createPane(n).style.zIndex = z;
});

function rasterGroup(spec, opacity, cls, pane){
  const g = L.layerGroup();
  (spec || []).forEach(p => L.imageOverlay(p[0], p[1],
    {opacity: opacity, className: cls, pane: pane, interactive: false}).addTo(g));
  g._opacity = opacity;
  return g;
}
const R = FLOOD.layers;
const gWater = rasterGroup(R.water,    0.75, 'crisp',  'p-water');
const gFlood = rasterGroup(R.flood,    0.85, 'crisp',  'p-flood');
const gPre   = rasterGroup(R.pre_sar,  0.90, 'smooth', 'p-sar');
const gPost  = rasterGroup(R.post_sar, 0.90, 'smooth', 'p-sar');

/* township choropleth ---------------------------------------------------- */
const byPcode = {};
TS.forEach((t, i) => { t._i = i; byPcode[t.pcode] = t; });

const tsStyle = t => ({
  color: '#ffffff', weight: 0.8, opacity: 0.55,
  fillColor: choro(t.flood_pct) || '#ffffff',
  fillOpacity: choro(t.flood_pct) ? 0.45 : 0.03
});
const gChoro = L.geoJSON(FLOOD.townshipGeo, {
  pane: 'p-choro',
  style: f => tsStyle(byPcode[f.properties.p] || {flood_pct: 0}),
  onEachFeature: (f, lyr) => {
    const t = byPcode[f.properties.p];
    if (!t) return;
    lyr._ts = t; t._lyr = lyr;
    lyr.bindTooltip(() =>
      '<b>' + esc(t.ts) + '</b> township<br>' +
      km2(t.flood_km2) + ' km² newly flooded · ' + num(t.flood_pct, 1) + '% of it<br>' +
      t.vil_at + ' village' + (t.vil_at === 1 ? '' : 's') + ' with water at the settlement',
      {sticky: true});
    lyr.on('mouseover', () => lyr.setStyle({weight: 2.2, opacity: 1}));
    lyr.on('mouseout',  () => lyr.setStyle({weight: 0.8, opacity: 0.55}));
    lyr.on('click', () => selectTs(t, true));
  }
});
const gOutline = L.geoJSON(FLOOD.townshipGeo, {
  interactive: false, pane: 'p-line',
  style: {color: '#ffffff', weight: 0.9, opacity: 0.5, fill: false}
});
const gAoi = L.geoJSON(FLOOD.aoi.geojson, {
  interactive: false, pane: 'p-line',
  style: {color: '#ffe066', weight: 2, opacity: 0.95, dashArray: '6 4', fill: false}
});

/* villages --------------------------------------------------------------- */
const V = FLOOD.villages;           // [lon, lat, flag, tsIndex, name, at%, near%]
const vCanvas = L.canvas({padding: 0.3, pane: 'p-vil'});

/* Marker radius follows the zoom.  Over 3,000 affected settlements at a fixed
   4.5 px cover this AOI in a solid sheet of dots at overview zoom and hide the
   10 m raster underneath -- the dots have to shrink to near-pixels when the
   whole AOI is on screen and only become clickable targets once you are in. */
const vScale = [];                  // [marker, size factor]
function vilRadius(z){
  return z <= 8 ? 1.5 : z <= 9 ? 2.1 : z <= 10 ? 2.9
       : z <= 11 ? 3.6 : z <= 12 ? 4.4 : 5.5;
}
function resizeVillages(){
  const r = vilRadius(map.getZoom());
  vScale.forEach(([m, f]) => {
    const rr = Math.max(0.8, r * f);
    m.setRadius(rr);
    m.setStyle({weight: rr >= 3 ? m._w : 0});
  });
}
map.on('zoomend', resizeVillages);

function villageLayer(match, style, factor){
  const g = L.layerGroup();
  V.forEach(v => {
    if (!match(v[2])) return;
    const m = L.circleMarker([v[1], v[0]], Object.assign(
      {renderer: vCanvas, pane: 'p-vil',
       radius: vilRadius(map.getZoom()) * factor}, style));
    m._v = v; m._w = style.weight || 0;
    m.bindPopup(() => villagePopup(v));
    vScale.push([m, factor]);
    m.addTo(g);
  });
  return g;
}
function villagePopup(v){
  const t = v[3] >= 0 ? TS[v[3]] : null;
  const state = v[2] === 2 ? 'New water <b>at the settlement</b>'
              : v[2] === 1 ? 'New water <b>within 500 m</b>'
              : v[2] === -1 ? '<b>Not assessed</b> — no usable data on one date'
              : 'No new water detected';
  return '<div class="pop"><b>' + esc(v[4]) + '</b><br>' +
    (t ? esc(t.ts) + ' township, ' + esc(t.st) : '') + '<hr style="border:0;border-top:1px solid var(--rule);margin:6px 0">' +
    state + '<br>' +
    'Flooded within 150 m: <b>' + v[5] + '%</b><br>' +
    'Flooded within 500 m: <b>' + v[6] + '%</b></div>';
}
/* Towns are place labels, not a statistical population -- a national gazetteer
   of 494 points. Drawn neutral (white, dark ring) and permanently labelled so
   they read as "where this is" rather than as another impact category. */
const TOWNS = FLOOD.towns || [];
const gTowns = L.layerGroup();
TOWNS.forEach(t => {
  const m = L.circleMarker([t[1], t[0]], {
    renderer: vCanvas, pane: 'p-vil', radius: 5,
    color: '#1a1a19', weight: 1.6, fillColor: '#ffffff', fillOpacity: 1});
  m.bindTooltip(t[4], {permanent: true, direction: 'right', offset: [7, 0],
                       className: 'town-lbl'});
  m.bindPopup(() =>
    '<div class="pop"><b>' + esc(t[4]) + '</b><br>town (MIMU gazetteer)' +
    '<hr style="border:0;border-top:1px solid var(--rule);margin:6px 0">' +
    (t[2] === 2 ? 'New water <b>at the town</b>'
     : t[2] === 1 ? 'New water <b>within 500 m</b>'
     : t[2] === -1 ? '<b>Not assessed</b>' : 'No new water detected') +
    '<br>Flooded within 150 m: <b>' + t[5] + '%</b>' +
    '<br>Flooded within 500 m: <b>' + t[6] + '%</b></div>');
  m.addTo(gTowns);
});

const gVilAt   = villageLayer(f => f === 2,
  {color: '#1a1a19', fillColor: M.c_village, fillOpacity: 1, weight: 1.2}, 1.0);
const gVilNear = villageLayer(f => f === 1,
  {color: M.c_village, fillColor: M.c_village, fillOpacity: 0.35, weight: 1}, 0.78);
const gVilAll  = villageLayer(f => f === 0 || f === -1,
  {color: '#898781', fillColor: '#898781', fillOpacity: 0.5, weight: 0}, 0.4);
resizeVillages();

/* default stack ---------------------------------------------------------- */
const LAYERS = [
  {k: 'flood',  g: gFlood,  n: 'New flood inundation',   sw: M.c_flood,
   d: M.scale + ' m raster · ' + km2(S.flood_km2) + ' km²', on: true,  op: true},
  {k: 'choro',  g: gChoro,  n: 'Township flood intensity', ramp: true,
   d: '% of each township newly flooded', on: true},
  /* The village layers start off. At AOI zoom 3,043 dots are a solid sheet
     that hides the 10 m raster and says only "many"; the tiles and the
     Villages tab carry that count in a form you can actually read. Clicking a
     village row switches the layer on and flies there. */
  {k: 'towns', g: gTowns, n: 'Towns (MIMU gazetteer)', sw: '#ffffff',
   d: TOWNS.length + ' town point' + (TOWNS.length === 1 ? '' : 's') +
      ', labelled', on: TOWNS.length > 0},
  {k: 'vil_at', g: gVilAt,  n: 'Villages — water at settlement', sw: M.c_village,
   d: num(S.vil_at, 0) + ' MIMU village points · clearest zoomed in', on: false},
  {k: 'vil_near', g: gVilNear, n: 'Villages — water within 500 m', sw: M.c_village,
   d: num(S.vil_near, 0) + ' points, hollow', on: false},
  {k: 'vil_all', g: gVilAll, n: 'All other village points', sw: '#898781',
   d: num(S.vil_total - S.vil_at - S.vil_near, 0) + ' points', on: false},
  {k: 'water',  g: gWater,  n: 'Permanent water (JRC)',  sw: M.c_water,
   d: M.scale + ' m raster · excluded from the flood class', on: true, op: true},
  {k: 'outline', g: gOutline, n: 'Township boundaries',  sw: 'transparent',
   d: 'MIMU adm3 250k', on: false},
  {k: 'aoi',    g: gAoi,    n: 'Area of interest',       sw: 'transparent',
   d: num(S.aoi_km2, 0) + ' km²', on: true},
  {k: 'pre_sar', g: gPre,   n: 'Pre-flood VV SAR ' + M.pre, sw: '#9a9a9a',
   d: M.scale + ' m backdrop · ' + num(M.mb_pre, 0) + ' MB, loads on demand',
   on: false, op: true},
  {k: 'post_sar', g: gPost, n: 'Post-flood VV SAR ' + M.post, sw: '#9a9a9a',
   d: M.scale + ' m backdrop · ' + num(M.mb_post, 0) + ' MB, loads on demand',
   on: false, op: true}
];
LAYERS.forEach(l => { if (l.on) l.g.addTo(map); });

/* One switch for both the map and the Layers checkbox, so a layer turned on
   from somewhere else in the UI cannot leave the panel showing otherwise. */
function setLayer(k, on){
  const l = LAYERS.find(x => x.k === k);
  if (!l) return;
  if (on && !map.hasLayer(l.g)) l.g.addTo(map);
  if (!on && map.hasLayer(l.g)) map.removeLayer(l.g);
  if (l._cb) l._cb.checked = on;
}

/* ═══ tabs ════════════════════════════════════════════════════════════════ */
const TABS = [['ts', 'Townships'], ['vil', 'Villages'],
              ['lyr', 'Layers'], ['met', 'Method']];
TABS.forEach(([k, label], i) => {
  const b = el('button', 'tab', label);
  b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.k = k;
  b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
  b.onclick = () => showTab(k, true);
  $('#tabs').append(b);
});
function showTab(k, push){
  const view = $('#v-' + k);
  if (!view) return;                       // unknown hash: leave the tab alone
  document.querySelectorAll('.tab').forEach(x =>
    x.setAttribute('aria-selected', String(x.dataset.k === k)));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('on'));
  view.classList.add('on');
  $('#views').scrollTop = 0;
  if (push) { try { history.replaceState(null, '', '#' + k); } catch (e) {} }
}
/* The tab lives in the URL fragment, so one view of this dashboard can be
   linked to directly: #ts, #vil, #lyr, #met. */
if (location.hash) showTab(location.hash.slice(1), false);

/* ═══ townships view ══════════════════════════════════════════════════════ */
/* Grouping exists because an AOI like this one spans several regions: a flat
   ranked list answers "which township is worst" but buries "which region is
   worst", and in a response that is the first question. Every level is
   aggregated from the SAME 10 m township sums -- a district figure is the sum
   of its townships' flooded area, never a re-measurement, so the levels cannot
   disagree. Percentages are re-derived (sum flooded / sum area), never
   averaged: averaging township percentages would weight a 200 km² township the
   same as a 2,000 km² one. */
let tsSort = 'flood_km2', tsQuery = '', tsMode = 'bars';
let tsGroup = 'ts', tsSelKey = null;

const GROUP_LABEL = {ts: 'township', dt: 'district', st: 'region'};

function tsRows(){
  const q = tsQuery.toLowerCase();
  const hits = TS.filter(t => !q ||
    (t.ts + ' ' + t.dt + ' ' + t.st).toLowerCase().includes(q));

  let rows;
  if (tsGroup === 'ts'){
    rows = hits.map(t => ({
      key: t.pcode, name: t.ts, sub: t.dt + ' district · ' + t.st,
      flood_km2: t.flood_km2, area_km2: t.area_km2, valid_km2: t.valid_km2,
      land_km2: t.land_km2 !== undefined ? t.land_km2 : t.area_km2,
      vil_at: t.vil_at, vil_total: t.vil_total, members: [t],
    }));
  } else {
    const by = new Map();
    hits.forEach(t => {
      const k = t[tsGroup];
      if (!by.has(k)) by.set(k, {
        key: k, name: k, sub: '', flood_km2: 0, area_km2: 0, valid_km2: 0,
        land_km2: 0, vil_at: 0, vil_total: 0, members: [],
      });
      const g = by.get(k);
      g.flood_km2 += t.flood_km2; g.area_km2 += t.area_km2;
      g.valid_km2 += t.valid_km2;
      g.land_km2 += (t.land_km2 !== undefined ? t.land_km2 : t.area_km2);
      g.vil_at += t.vil_at; g.vil_total += t.vil_total; g.members.push(t);
    });
    rows = [...by.values()];
    rows.forEach(g => {
      const n = g.members.length;
      g.sub = (tsGroup === 'dt' ? g.members[0].st + ' · ' : '') +
              n + ' township' + (n === 1 ? '' : 's');
    });
  }

  rows.forEach(r => {
    r.flood_pct  = r.area_km2 ? 100 * r.flood_km2 / r.area_km2 : 0;
    /* Coverage is judged against LAND. A delta township that is a third open
       sea is not a third unassessed -- the sea was never assessable, and
       flagging it would bury the rows that are real gaps. */
    r.land_frac  = r.land_km2 ? r.valid_km2 / r.land_km2 : 1;
  });
  rows.sort((a, b) => tsSort === 'name' ? a.name.localeCompare(b.name)
                                        : b[tsSort] - a[tsSort]);
  return rows;
}

function hover(row, on){
  row.members.forEach(t => t._lyr && t._lyr.setStyle(
    on ? {weight: 2.2, opacity: 1} : {weight: 0.8, opacity: 0.55}));
}

function renderTs(){
  const rows = tsRows();
  const box = $('#ts-rows'); box.textContent = '';
  const sortKey = tsSort === 'name' ? 'flood_km2' : tsSort;
  const max = Math.max(1e-9, ...rows.map(r => r[sortKey]));

  rows.forEach(row => {
    const r = el('div', 'row' + (row.key === tsSelKey ? ' sel' : ''));
    const nm = el('div', 'nm', row.name);
    nm.append(el('small', null, row.sub));
    const val = el('div', 'val', km2(row.flood_km2) + ' km²');
    val.append(el('small', null, num(row.flood_pct, 1) + '% · ' +
      num(row.vil_at, 0) + '/' + num(row.vil_total, 0) + ' villages'));
    const bar = el('div', 'bar');
    const fill = el('i');
    fill.style.width = (100 * Math.max(row[sortKey], 0) / max) + '%';
    fill.style.background = choro(row.flood_pct) || 'var(--axis)';
    bar.append(fill);
    r.append(nm, val, bar);
    if (row.land_frac < 0.98){
      const w = el('div', 'warnrow');
      w.append(el('span', null, '⚠'),
        el('span', null, 'only ' + num(row.land_frac * 100, 0) +
           '% of the land in this ' + GROUP_LABEL[tsGroup] +
           ' had usable data on both dates'));
      r.append(w);
    }
    r.onclick = () => selectTs(row, true);
    r.onmouseenter = () => hover(row, true);
    r.onmouseleave = () => hover(row, false);
    box.append(r);
  });

  const head1 = {ts: 'Township', dt: 'District', st: 'Region'}[tsGroup];
  const head2 = {ts: 'District', dt: 'Region', st: 'Townships'}[tsGroup];
  const tbl = el('table');
  const thead = el('thead'), head = el('tr');
  [head1, head2, 'Flooded km²', '% flooded', 'Villages hit', 'Villages',
   'Land assessed %'].forEach(h => head.append(el('th', null, h)));
  thead.append(head); tbl.append(thead);
  const tb = el('tbody');
  rows.forEach(row => {
    const second = tsGroup === 'ts' ? row.members[0].dt
                 : tsGroup === 'dt' ? row.members[0].st
                 : row.members.length;
    const tr = el('tr');
    [row.name, second, km2(row.flood_km2), num(row.flood_pct, 1),
     row.vil_at, row.vil_total, num(row.land_frac * 100, 1)]
      .forEach(c => tr.append(el('td', null, String(c))));
    tb.append(tr);
  });
  tbl.append(tb);
  $('#ts-tbl').textContent = ''; $('#ts-tbl').append(tbl);

  const unit = GROUP_LABEL[tsGroup];
  const shown = rows.length + ' ' + unit + (rows.length === 1 ? '' : 's') +
    ' shown' + (tsGroup === 'ts' ? '. '
      : ', aggregated from ' + rows.reduce((n, r) => n + r.members.length, 0) +
        ' townships. ');
  /* If the AOI was drawn AS whole townships, nothing is clipped and the usual
     caveat would be misleading in the other direction -- these really are
     whole-township percentages. */
  const whole = TS.every(t => t.share_of_township >= 0.999);
  $('#ts-foot').textContent = shown + 'Area is summed from 10 m pixels' +
    (whole
      ? '. Each township lies entirely inside the AOI, so "% flooded" is of the '
        + 'whole ' + unit + '.'
      : ' inside the ' + unit + ' ∩ AOI polygon, so "% flooded" is of that '
        + 'clipped area, not of the whole ' + unit + '.');
}

$('#ts-q').oninput  = e => { tsQuery = e.target.value; renderTs(); };
$('#ts-sort').onchange = e => { tsSort = e.target.value; renderTs(); };
$('#ts-bars').onclick  = () => setTsMode('bars');
$('#ts-table').onclick = () => setTsMode('table');
['ts', 'dt', 'st'].forEach(g => { $('#ts-g-' + g).onclick = () => setTsGroup(g); });
/* With one township in the AOI, Township / District / Region all render the
   same single row, so the control is noise. Hide it rather than offer three
   buttons that do nothing. */
if (TS.length < 2) {
  const seg = $('#ts-g-ts').parentNode;
  seg.hidden = true;
}

function setTsMode(m){
  tsMode = m;
  $('#ts-bars').setAttribute('aria-pressed', m === 'bars');
  $('#ts-table').setAttribute('aria-pressed', m === 'table');
  $('#ts-rows').hidden = m !== 'bars';
  $('#ts-tbl').hidden  = m !== 'table';
}
function setTsGroup(g){
  tsGroup = g; tsSelKey = null;
  ['ts', 'dt', 'st'].forEach(x =>
    $('#ts-g-' + x).setAttribute('aria-pressed', String(x === g)));
  renderTs();
}
function selectTs(row, zoom){
  tsSelKey = row.key;
  renderTs();
  if (!zoom) return;
  /* A fresh bounds object, NOT the first member's: Polygon.getBounds() hands
     back the layer's internal _bounds, so extending it would quietly grow that
     township's own bounds every time a group row was clicked. */
  const bounds = L.latLngBounds([]);
  row.members.forEach(t => { if (t._lyr) bounds.extend(t._lyr.getBounds()); });
  if (bounds.isValid()) map.fitBounds(bounds, {padding: [24, 24]});
  if (row.members.length === 1 && row.members[0]._lyr)
    row.members[0]._lyr.openTooltip();
}

/* ═══ villages view ═══════════════════════════════════════════════════════ */
let vilFlag = 2, vilQuery = '';
const VIL_CAP = 400;
function renderVil(){
  const q = vilQuery.toLowerCase();
  let rows = V.filter(v => v[2] === vilFlag);
  if (q) rows = rows.filter(v =>
    (v[4] + ' ' + (v[3] >= 0 ? TS[v[3]].ts : '')).toLowerCase().includes(q));
  rows.sort((a, b) => b[5] - a[5] || b[6] - a[6]);

  const box = $('#vil-rows'); box.textContent = '';
  rows.slice(0, VIL_CAP).forEach(v => {
    const r = el('div', 'row');
    const nm = el('div', 'nm', v[4]);
    nm.append(el('small', null, v[3] >= 0
      ? TS[v[3]].ts + ' township · ' + TS[v[3]].st : 'township outside the table'));
    const val = el('div', 'val', v[5] + '%');
    val.append(el('small', null, 'within 500 m: ' + v[6] + '%'));
    const bar = el('div', 'bar');
    const fill = el('i');
    fill.style.width = Math.min(100, v[5]) + '%';
    fill.style.background = M.c_village;
    bar.append(fill);
    r.append(nm, val, bar);
    r.onclick = () => {
      setLayer(v[2] === 2 ? 'vil_at' : v[2] === 1 ? 'vil_near' : 'vil_all', true);
      map.setView([v[1], v[0]], 14);
      L.popup().setLatLng([v[1], v[0]]).setContent(villagePopup(v)).openOn(map);
    };
    box.append(r);
  });

  $('#vil-foot').innerHTML =
    '<b>' + num(rows.length, 0) + '</b> village point' + (rows.length === 1 ? '' : 's') +
    ' in this class' + (rows.length > VIL_CAP
      ? ', first ' + VIL_CAP + ' shown — refine with the search box' : '') +
    '. Bars show the share of the 150 m around the settlement that is newly ' +
    'flooded. MIMU village points are settlement <i>locations</i>, not built-up ' +
    'outlines, so this measures water reaching the settlement, not buildings flooded.';
}
$('#vil-q').oninput = e => { vilQuery = e.target.value; renderVil(); };
$('#vil-filter').onchange = e => { vilFlag = +e.target.value; renderVil(); };

/* ═══ layers view ═════════════════════════════════════════════════════════ */
(function(){
  const v = $('#v-lyr');
  v.append(el('div', 'h3', 'Overlays'));
  LAYERS.forEach(l => {
    const row = el('div', 'lyr');
    const empty = l.g.getLayers && l.g.getLayers().length === 0;
    const cb = el('input'); cb.type = 'checkbox'; cb.checked = !!l.on && !empty;
    cb.disabled = empty;
    cb.onchange = () => cb.checked ? l.g.addTo(map) : map.removeLayer(l.g);
    l._cb = cb;
    if (empty) l.d = 'not built in this run (--no-sar)';
    const sw = el('div', 'sw');
    if (l.ramp) sw.style.background =
      'linear-gradient(90deg,' + RAMP[1] + ',' + RAMP[RAMP.length - 1] + ')';
    else sw.style.background = l.sw;
    const t = el('div', 't');
    t.append(Object.assign(el('b'), {textContent: l.n}),
             Object.assign(el('span'), {textContent: l.d}));
    if (l.op){
      const s = el('input'); s.type = 'range'; s.min = 0; s.max = 100;
      s.value = Math.round(l.g._opacity * 100);
      s.oninput = () => l.g.eachLayer(x => x.setOpacity(s.value / 100));
      t.append(s);
    }
    row.append(cb, sw, t);
    v.append(row);
  });
  v.append(el('div', 'h3', 'Basemap'));
  Object.keys(bases).forEach(nm => {
    const row = el('div', 'lyr');
    const rb = el('input'); rb.type = 'radio'; rb.name = 'base';
    rb.checked = nm === baseName;
    rb.onchange = () => { map.removeLayer(bases[baseName]);
      baseName = nm; bases[nm].addTo(map); };
    const t = el('div', 't'); t.append(Object.assign(el('b'), {textContent: nm}));
    row.append(rb, el('div', 'sw'), t);
    v.append(row);
  });
})();

/* ═══ method view ═════════════════════════════════════════════════════════ */
(function(){
  const v = $('#v-met');
  const kv = (k, val) => { const d = el('div', 'kv');
    d.append(el('span', null, k), Object.assign(el('b'), {textContent: val}));
    v.append(d); };
  v.append(el('div', 'h3', 'Imagery pair'));
  kv('Pre-flood', M.pre + '  (' + M.n_pre + ' scenes, ' + M.cov_pre + '% of AOI)');
  kv('Post-flood', M.post + '  (' + M.n_post + ' scenes, ' + M.cov_post + '% of AOI)');
  kv('Separation', M.gap);
  kv('Orbit', M.pass + ', relative orbit ' + M.track);
  kv('Collection', 'COPERNICUS/' + M.source);
  kv('Date window', '± ' + M.window + ' day — exact dates only');
  v.append(el('div', 'h3', 'Detection'));
  kv('Change threshold', 'VV drop < ' + M.thresh + ' dB');
  kv('Speckle filter', 'focal mean, ' + M.speckle + ' m radius');
  kv('Excluded', 'JRC permanent water; slope ≥ ' + M.slope + '°');
  kv('Resolution', M.scale + ' m native — layers and statistics');
  v.append(el('div', 'h3', 'Impact units'));
  kv('Townships', 'MIMU adm3 250k v9.4, clipped to AOI');
  kv('Village points', 'MIMU pplp2 250k, PCode v9.7');
  kv('Settlement radius', M.vil_at_m + ' m, flagged above ' + M.vil_at_frac + '%');
  kv('Surroundings radius', M.vil_near_m + ' m, flagged above ' + M.vil_near_frac + '%');
  kv('Join', 'by P-code, never by name');
  v.append(el('div', 'h3', 'Provenance'));
  kv('Generated', M.generated);
  kv('AOI area', num(S.aoi_km2, 0) + ' km² summed from 10 m pixels');
  kv('Assessed area', num(S.valid_frac * 100, 2) + '% of the AOI');
  if (S.land_frac !== undefined)
    kv('Assessed land', num(S.land_frac * 100, 2) + '% of the '
       + num(S.land_km2, 0) + ' km² of land');

  const c1 = el('div', 'callout');
  c1.innerHTML = '<b>This is change, not total water.</b> ' + M.pre +
    ' is itself a monsoon image, so ground already under water then is not in ' +
    'the red layer. Treat the figure as new inundation between the two dates — ' +
    'a lower bound on standing water on ' + M.post + '.';
  v.append(c1);
  const c2 = el('div', 'callout');
  c2.innerHTML = '<b>Blank is not dry.</b> Where either date had no usable ' +
    'Sentinel-1 or DEM data the pixel is unassessed, not flood-free. ' +
    'Coverage is judged against <i>land</i>: the terrain model carries no ' +
    'data over the sea, which is deliberate — it is what keeps open ' +
    'ocean out of the flood class' +
    (S.sea_km2 > 1 ? ', and it accounts for the ' + num(S.sea_km2, 0) +
      ' km² of this AOI that sits offshore' : '') +
    '. Rows below 98% of their land assessed carry a ⚠ in the list.';
  v.append(c2);
  const c3 = el('div', 'callout');
  c3.innerHTML = '<b>Radar sees water, not damage.</b> Smooth dry surfaces ' +
    '(fresh tarmac, sand, dry riverbed) can mimic the specular drop that marks ' +
    'flooding, and buildings or trees standing in water can hide it. ' +
    'Ground-truth before operational use.';
  v.append(c3);
  const c4 = el('div', 'note');
  c4.innerHTML = 'Imagery: Copernicus Sentinel-1 (ESA) via Google Earth Engine. ' +
    'Permanent water: JRC Global Surface Water v1.4. Terrain: WWF HydroSHEDS. ' +
    'Boundaries and village points: <a href="https://geonode.themimu.info" ' +
    'target="_blank" rel="noopener">MIMU</a>. Basemaps: Esri, OpenStreetMap.';
  v.append(c4);
})();

/* ═══ legend ══════════════════════════════════════════════════════════════ */
(function(){
  const b = $('#lg-b');
  const line = (color, text, round) => {
    const d = el('div', 'lg'); const i = el('i', round ? 'dot' : null);
    i.style.background = color; d.append(i, el('span', null, text)); b.append(d);
  };
  line(M.c_flood, 'New flood inundation');
  line(M.c_water, 'Permanent water (JRC)');
  line(M.c_village, 'Village — water at settlement', true);
  const cap = el('div', 'lg');
  cap.style.marginTop = '10px';
  cap.append(el('span', null, 'Township newly flooded, % of area'));
  b.append(cap);
  const ramp = el('div', 'ramp');
  RAMP.forEach(c => { const i = el('i'); i.style.background = c; ramp.append(i); });
  b.append(ramp);
  const ax = el('div', 'ramp-ax');
  ax.append(el('span', null, '0'), el('span', null, BREAKS[2] + '%'),
            el('span', null, '≥' + BREAKS[BREAKS.length - 1] + '%'));
  b.append(ax);
  $('#lg-h').onclick = () => {
    const lg = $('#legend'); lg.classList.toggle('min');
    $('#lg-h .c').textContent = lg.classList.contains('min') ? '▸' : '▾';
  };
})();

/* ═══ map furniture ═══════════════════════════════════════════════════════ */
L.control.scale({imperial: false, position: 'bottomleft'}).addTo(map);
map.on('mousemove', e => { $('#readout').textContent =
  e.latlng.lat.toFixed(4) + ' , ' + e.latlng.lng.toFixed(4); });
map.on('mouseout', () => { $('#readout').textContent = '–'; });

$('#sheet-toggle').onclick = () => {
  const a = $('#app'); a.classList.toggle('collapsed');
  $('#sheet-toggle').textContent = a.classList.contains('collapsed') ? '▴ show' : '▾ hide';
};

setTsMode('bars');
renderTs();
renderVil();
</script>
</body>
</html>
"""


def build_dashboard(overlays: dict, stats: dict, aoi_info: dict,
                    townships: list, villages: list, meta: dict,
                    output: str, towns: list = None) -> None:
    """
    Write the standalone dashboard.

    Every raster reference is page-local (base64 or ASSET_DIR), so nothing
    points at a Google server and nothing expires.  The whole payload is one
    JSON literal, which keeps the template free of string interpolation and
    makes the numbers auditable in the page source.
    """
    bbox = aoi_info["bbox"]

    ts_out, ts_geo = [], []
    for t in townships:
        ts_out.append({
            "pcode": t["pcode"], "ts": t["ts"], "ts_mmr": t.get("ts_mmr", ""),
            "dt": t["dt"], "st": t["st"],
            "flood_km2": round(t["flood_km2"], 3),
            "flood_pct": round(t["flood_pct"], 3),
            "area_km2":  round(t["area_km2"], 2),
            "water_km2": round(t["water_km2"], 2),
            "valid_frac": round(t["valid_frac"], 4),
            "land_km2":   round(t.get("land_km2", t["area_km2"]), 2),
            "vil_total": t["vil_total"], "vil_at": t["vil_at"],
            "vil_near": t["vil_near"], "vil_nodata": t["vil_nodata"],
            "share_of_township": round(t["share_of_township"], 3),
        })
        ts_geo.append({"type": "Feature", "properties": {"p": t["pcode"]},
                       "geometry": t["geom"]})

    # Villages as positional arrays: 9,900 objects with named keys would add
    # ~1 MB of repeated key text to the page for no gain.
    vil_out = [[round(v["lon"], 4), round(v["lat"], 4), v.get("flag", -1),
                v["ts"], v["name"],
                int(round(100 * v.get("f_at", 0.0))),
                int(round(100 * v.get("f_near", 0.0)))]
               for v in villages]

    payload = {
        "meta": meta,
        "aoi": {
            "name": aoi_info["name"],
            "geojson": aoi_info["geojson"],
            "bounds": [[bbox["south"], bbox["west"]],
                       [bbox["north"], bbox["east"]]],
        },
        "stats": stats,
        "townships": ts_out,
        "townshipGeo": {"type": "FeatureCollection", "features": ts_geo},
        "villages": vil_out,
        "towns": [[round(t["lon"], 4), round(t["lat"], 4), t.get("flag", -1),
                   t["ts"], t["name"],
                   int(round(100 * t.get("f_at", 0.0))),
                   int(round(100 * t.get("f_near", 0.0)))]
                  for t in (towns or [])],
        "layers": {k: v["pieces"] for k, v in overlays.items()},
    }

    _write_page(payload, output)


def _write_page(payload: dict, output: str) -> None:
    """Fill the template with a payload and write it."""
    m, s = payload["meta"], payload["stats"]
    # Search-result and link-preview text.  Built from the figures rather than
    # written by hand so it cannot drift from what the page actually shows.
    # No quotes or angle brackets: it goes straight into an HTML attribute.
    desc = (f"{s['flood_km2']:,.0f} km2 of new flood inundation mapped from "
            f"Sentinel-1 SAR between {m['pre']} and {m['post']}, at 10 m native "
            f"resolution. {s['vil_at']:,} MIMU village points with water at the "
            f"settlement across {len(payload['townships'])} townships.")

    html = (PAGE_TEMPLATE
            .replace("__DESC__", desc)
            .replace("__TITLE__", payload["meta"]["title"])
            .replace("__C_FLOOD__", C_FLOOD)
            .replace("__C_WATER__", C_WATER)
            .replace("__C_VILLAGE__", C_VILLAGE)
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))

    with open(output, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[OK] Dashboard saved -> {output}")


PAYLOAD_HEAD = "const FLOOD = "
PAYLOAD_TAIL = ";\n</script>"


def rebuild_page(output: str = OUTPUT_HTML) -> None:
    """
    Re-render an existing dashboard with the current template, same numbers.

    The page already carries every figure it displays, so a change to the
    layout, palette or interaction does not need another 500 MB of downloads --
    the payload is lifted out of the old HTML and poured into the new template.
    Raster references are relative paths into ASSET_DIR (or data URIs), so they
    survive the round trip untouched.

    Only the presentation may change this way.  Anything that alters a NUMBER
    means re-running the analysis, because nothing here recomputes.
    """
    if not os.path.exists(output):
        raise SystemExit(f"[STOP] {output} does not exist yet -- run the full "
                         f"script once before --page-only.")

    with open(output, encoding="utf-8") as fh:
        html = fh.read()

    i = html.find(PAYLOAD_HEAD)
    j = html.find(PAYLOAD_TAIL, i)
    if i < 0 or j < 0:
        raise SystemExit(f"[STOP] Could not find the data payload in {output}. "
                         f"It was probably not written by this script.")

    payload = json.loads(html[i + len(PAYLOAD_HEAD):j])
    payload["meta"]["ramp"]   = CHORO_RAMP
    payload["meta"]["breaks"] = CHORO_BREAKS
    payload["meta"]["c_flood"], payload["meta"]["c_water"] = C_FLOOD, C_WATER
    payload["meta"]["c_village"] = C_VILLAGE

    n_chunks = sum(len(v) for v in payload["layers"].values())
    print(f"  Reusing payload: {len(payload['townships'])} townships, "
          f"{len(payload['villages']):,} villages, {n_chunks} raster chunk(s), "
          f"generated {payload['meta']['generated']}.")
    _write_page(payload, output)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def _gap_text() -> str:
    days = (date.fromisoformat(POST_FLOOD_DATE)
            - date.fromisoformat(PRE_FLOOD_DATE)).days
    cycles = days / 12.0
    if REL_ORBIT is not None and days % 12 == 0:
        n = days // 12
        return f"{days} days ({n} repeat cycle{'s' if n != 1 else ''})"
    return f"{days} days ({cycles:.1f} repeat cycles)"


def check_disk(need_gb: float) -> None:
    free = shutil.disk_usage(os.getcwd()).free / 1e9
    if free < need_gb:
        print(f"  [!] Only {free:.1f} GB free on this drive. A native-resolution "
              f"run needs roughly {need_gb:.0f} GB for tiles, merged GeoTIFFs and "
              f"PNGs.\n      Free some space or run with --no-sar (the flood and "
              f"water layers alone need well under 1 GB).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true",
                    help="only test Sentinel-1 availability for the two dates")
    ap.add_argument("--no-sar", action="store_true",
                    help="skip the two native-resolution SAR backdrops")
    ap.add_argument("--keep-tif", action="store_true",
                    help="keep the intermediate GeoTIFFs in " + TEMP_DIR)
    ap.add_argument("--page-only", action="store_true",
                    help="re-render " + OUTPUT_HTML + " from its own payload "
                         "(layout/palette changes, no downloads, no GEE)")
    args = ap.parse_args()

    if args.page_only:
        rebuild_page(OUTPUT_HTML)
        return

    init_gee()

    aoi_info = load_aoi(AOI_GEOJSON)
    aoi      = aoi_info["ee_geom"]
    bbox     = aoi_info["bbox"]
    aoi_km2  = aoi.area(maxError=10).divide(1e6).getInfo()

    print(f"\n> AOI: {aoi_info['name']}")
    print(f"  Area: {aoi_km2:,.0f} km2   bbox "
          f"{bbox['west']:.3f},{bbox['south']:.3f} -> "
          f"{bbox['east']:.3f},{bbox['north']:.3f}")

    # Fail fast and clearly if either date's imagery is not in GEE yet, and
    # settle on one collection for both composites.
    cid, is_linear, report = preflight(aoi)
    if args.check:
        # Report the MEASURED coverage, not just "available". A date can be
        # present and still be 12% of the AOI, and a bare "available" would
        # send someone off to build exactly the map preflight refuses to build.
        cov = min(c for _, c in report.values())
        print(f"\n[OK] Both dates are on {ORBIT_PASS} track {REL_ORBIT} in "
              f"{cid.split('/')[-1]}, covering {cov:.1%} of the AOI.")
        print("     Re-run without --check to build the map.")
        return

    if not args.no_sar:
        check_disk(DISK_WARN_GB)

    # ── MIMU impact geography ────────────────────────────────────────────────
    print("\n> Loading MIMU administrative data ...")
    townships = load_townships(aoi_info["shape"])
    villages  = load_villages(aoi_info["shape"], townships)
    towns     = load_towns(aoi_info["shape"], townships)

    # ── GEE server-side processing ───────────────────────────────────────────
    print(f"\n> Loading S1 pre-flood  ({PRE_FLOOD_DATE}, {ORBIT_PASS}) ...")
    pre_raw  = load_s1(aoi, PRE_FLOOD_DATE,  DATE_WINDOW, ORBIT_PASS,
                       cid=cid, is_linear=is_linear)

    print(f"> Loading S1 post-flood ({POST_FLOOD_DATE}, {ORBIT_PASS}) ...")
    post_raw = load_s1(aoi, POST_FLOOD_DATE, DATE_WINDOW, ORBIT_PASS,
                       cid=cid, is_linear=is_linear)

    print("> Applying speckle filter ...")
    pre  = speckle_filter(pre_raw)
    post = speckle_filter(post_raw)

    print("> Detecting flooded pixels ...")
    flood_mask, perm_water, _ = detect_floods(pre, post, aoi)

    # ── Statistics, all at native scale ──────────────────────────────────────
    stat_img = _stat_image(flood_mask, perm_water, aoi)

    print(f"\n> Computing AOI totals at {EXPORT_SCALE} m ...")
    a = aoi_stats(stat_img, aoi)
    valid_frac = a["valid_a"] / a["total_a"] if a["total_a"] else 0.0
    land_frac  = a["valid_a"] / a["land_a"] if a["land_a"] else 0.0
    sea_km2    = max(0.0, a["total_a"] - a["land_a"])
    # Every share on the page uses total_a -- the AOI area MEASURED by summing
    # 10 m pixels -- not aoi_km2, which is the geodesic polygon area.  The two
    # differ by ~0.3% here because ee.Geometry edges are geodesic while the
    # reduction grid follows parallels and meridians; mixing them would make
    # the numerator and denominator come from different geometries.
    print(f"  => Newly inundated : {a['flood_a']:,.1f} km2 "
          f"({a['flood_a'] / a['total_a'] * 100:.2f}% of AOI)")
    print(f"  => Permanent water : {a['water_a']:,.1f} km2")
    print(f"  => AOI assessed    : {valid_frac:.2%}  "
          f"(of land: {land_frac:.2%}; {sea_km2:,.0f} km2 of the AOI is sea, "
          f"where the terrain model has no data by design)")
    if land_frac < 0.995:
        print(f"  [!] WARNING: only {land_frac:.0%} of the LAND was assessed. The "
              f"rest lacked usable Sentinel-1 or DEM data on one of the two "
              f"dates. Unassessed areas are unknown, not dry.")

    township_stats(stat_img, townships)
    village_stats(flood_mask, villages)
    vil_totals = roll_up_villages(townships, villages)
    if towns:
        print(f"> Sampling {len(towns)} MIMU town point(s) ...")
        village_stats(flood_mask, towns)

    hit = [t for t in townships if t["flood_km2"] > 0.5]
    hit.sort(key=lambda t: -t["flood_km2"])
    print(f"\n  Townships with >0.5 km2 of new water: {len(hit)}/{len(townships)}")
    for t in hit[:10]:
        print(f"    {t['ts']:<16s} {t['flood_km2']:8,.1f} km2  "
              f"{t['flood_pct']:5.1f}% of township  "
              f"{t['vil_at']:4d} village(s) with water at the settlement")
    print(f"  Villages: {vil_totals['at']:,} with water at the settlement, "
          f"{vil_totals['near']:,} more within {VILLAGE_NEAR_M} m, "
          f"{vil_totals['nodata']:,} not assessed.")

    # ── Rasters (all uint8 so 10 m stays tractable) ──────────────────────────
    # SAR dB is rescaled server-side to 1..255 (0 reserved as nodata): the PNG
    # is 8-bit anyway, so quantising here costs no visible fidelity but cuts
    # transfer and memory 4x versus float32.  This is a radiometric mapping for
    # display, not a spatial one -- pixel spacing stays at 10 m.
    def sar_byte(img: ee.Image) -> ee.Image:
        return (img.select("VV").unitScale(-25, 0)
                .multiply(254).add(1).clamp(1, 255).toByte())

    print("\n> Downloading and encoding rasters (native "
          f"{EXPORT_SCALE} m) ...")
    clear_assets()
    shp = aoi_info["shape"]
    overlays = {
        "flood": build_layer(flood_mask.toByte(), "flood.tif", "flood",
                             _palette_solid(C_FLOOD), shp, EXPORT_SCALE, bbox,
                             False, args.keep_tif),
        "water": build_layer(perm_water.toByte(), "water.tif", "water",
                             _palette_solid(C_WATER), shp, EXPORT_SCALE, bbox,
                             False, args.keep_tif),
    }
    if args.no_sar:
        overlays["pre_sar"]  = {"pieces": [], "external": False, "mb": 0.0}
        overlays["post_sar"] = {"pieces": [], "external": False, "mb": 0.0}
        print("  [i] --no-sar: the two 10 m SAR backdrops were not built.")
    else:
        overlays["pre_sar"] = build_layer(
            sar_byte(pre), "pre_vv.tif", "pre_sar", _palette_gray(), shp,
            SAR_DISPLAY_SCALE, bbox, True, args.keep_tif)
        overlays["post_sar"] = build_layer(
            sar_byte(post), "post_vv.tif", "post_sar", _palette_gray(), shp,
            SAR_DISPLAY_SCALE, bbox, True, args.keep_tif)

    # ── Build the dashboard ──────────────────────────────────────────────────
    stats = {
        "flood_km2": a["flood_a"], "water_km2": a["water_a"],
        "aoi_km2": a["total_a"], "polygon_km2": aoi_km2,
        "flood_pct": (a["flood_a"] / a["total_a"] * 100) if a["total_a"] else 0.0,
        "valid_frac": valid_frac, "land_frac": land_frac,
        "land_km2": a["land_a"], "sea_km2": sea_km2,
        "vil_total": vil_totals["total"], "vil_at": vil_totals["at"],
        "vil_near": vil_totals["near"], "vil_nodata": vil_totals["nodata"],
    }
    meta = {
        "title": "Flood inundation " + PRE_FLOOD_DATE + " → " + POST_FLOOD_DATE,
        "pre": PRE_FLOOD_DATE, "post": POST_FLOOD_DATE,
        "window": DATE_WINDOW, "gap": _gap_text(),
        "pass": ORBIT_PASS.title(), "pass_short": ORBIT_PASS.title()[:4] + ".",
        "track": REL_ORBIT, "source": cid.split("/")[-1],
        "n_pre": report[PRE_FLOOD_DATE][0],
        "cov_pre": round(report[PRE_FLOOD_DATE][1] * 100, 1),
        "n_post": report[POST_FLOOD_DATE][0],
        "cov_post": round(report[POST_FLOOD_DATE][1] * 100, 1),
        "thresh": FLOOD_DB_THRESH, "speckle": SPECKLE_RADIUS,
        "slope": MAX_SLOPE_DEG, "scale": EXPORT_SCALE,
        "vil_at_m": VILLAGE_AT_M, "vil_near_m": VILLAGE_NEAR_M,
        "vil_at_frac": round(VILLAGE_AT_FRAC * 100, 1),
        "vil_near_frac": round(VILLAGE_NEAR_FRAC * 100, 1),
        "ramp": CHORO_RAMP, "breaks": CHORO_BREAKS,
        "c_flood": C_FLOOD, "c_water": C_WATER, "c_village": C_VILLAGE,
        "mb_pre": round(overlays["pre_sar"]["mb"]),
        "mb_post": round(overlays["post_sar"]["mb"]),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    print("> Building dashboard ...")
    build_dashboard(overlays, stats, aoi_info, townships, villages, meta,
                    OUTPUT_HTML, towns)

    with open(STATS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "aoi": stats,
                   "townships": [{k: v for k, v in t.items()
                                  if k not in ("geom", "_lyr")}
                                 for t in townships]},
                  fh, indent=1, ensure_ascii=False)
    print(f"[OK] Figures also written to {STATS_JSON}")

    if not args.keep_tif:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    html_mb  = os.path.getsize(OUTPUT_HTML) / 1e6
    asset_mb = sum(v["mb"] for v in overlays.values() if v["external"])
    print(f"\n{'-' * 66}")
    print(f"  Done!  Open  '{OUTPUT_HTML}'  ({html_mb:.1f} MB) in any browser.")
    if asset_mb:
        n_files = len(glob.glob(os.path.join(ASSET_DIR, "*.png")))
        print(f"  Plus {ASSET_DIR}/ -- {n_files} PNG(s), {asset_mb:,.0f} MB.")
        print(f"  Publish BOTH: the page loads its big layers from {ASSET_DIR}/.")
    print(f"  All layers and statistics at {EXPORT_SCALE} m native. "
          f"No GEE URLs -- nothing expires.")
    print(f"{'-' * 66}")


if __name__ == "__main__":
    main()
