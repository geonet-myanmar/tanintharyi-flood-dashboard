#!/usr/bin/env python3
"""
Copernicus Data Space Ecosystem imagery backend for flood_mapping.

WHY THIS EXISTS
  Google Earth Engine runs days behind ESA for some regions and occasionally
  never catches up.  On 2026-09-04 the Kale pass of 2026-09-03 23:38 UTC was in
  ESA's archive and online, while GEE held nothing for it -- so the requested
  pair was unbuildable from GEE alone.  This module fetches the same imagery
  straight from CDSE instead.

WHAT IT RETURNS
  Terrain-corrected gamma0 VV in dB, on an exact EPSG:4326 grid at Sentinel-1's
  native 10 m spacing.  Sentinel Hub does the orbit correction, radiometric
  calibration and orthorectification server-side against the Copernicus DEM, so
  what comes back is analysis-ready and directly comparable between dates.

BOTH DATES COME FROM HERE, NEVER ONE FROM EACH
  GEE's S1_GRD is sigma0 ellipsoid; this is gamma0 terrain-corrected.  The two
  differ by several dB on sloping ground, and a change-detection product is a
  DIFFERENCE, so mixing them would write that offset straight into the flood
  mask.  Same discipline as never mixing orbit passes.

CREDENTIALS
  Read from CRED_FILE, two lines: CDSE account e-mail, then password.  A token
  lasts 30 minutes and is refreshed automatically -- a native-resolution fetch
  takes longer than that.  Nothing is logged: no token, no password.

PROCESSING UNITS
  Sentinel Hub meters usage.  One 10 m fetch of this AOI is roughly
  (width x height / 512^2) x 2 PU per date for FLOAT32 output -- about 400 PU
  for a 44 Mpx AOI across both dates, against a 30,000 PU/month free tier.
"""

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_origin

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Two lines: e-mail, password.
# Searched in order. The file moves between project layouts, and failing to
# authenticate because a path changed wastes a whole run.
CRED_FILES = [
    "credentials_CDSE.txt",
    os.path.join("identity.dataspace.copernicus.eu_credentials",
                 "credentials.txt"),
    "credentials.txt",
]
CRED_FILE = CRED_FILES[0]        # the one named in error messages

# Sentinel Hub caps a single response at 2500 px per side; 2048 leaves room and
# keeps one float32 tile at 16 MB.
SH_TILE_PX = 2048
SH_MAX_ATTEMPTS = 4

# 10 m expressed in degrees, the same convention Earth Engine's scale=10 uses in
# EPSG:4326, so rasters from either backend land on the same grid.
DEG_PER_10M = 10.0 / 111319.49

_token = {"value": None, "expires": 0.0}


def _creds() -> tuple:
    """
    CDSE e-mail and password.

    Environment first (CDSE_USER / CDSE_PASSWORD) so a caller can supply them
    without a file on disk; otherwise CRED_FILE. Credentials go only to the CDSE
    token endpoint and are never logged.
    """
    env_u, env_p = os.environ.get("CDSE_USER"), os.environ.get("CDSE_PASSWORD")
    if env_u and env_p:
        return env_u, env_p
    path = next((c for c in CRED_FILES if os.path.exists(c)), None)
    if path is None:
        raise SystemExit(
            "[STOP] CDSE credentials not found. Looked in: "
            + ", ".join(CRED_FILES)
            + " -- two lines: account e-mail, then password."
            + " Or set CDSE_USER and CDSE_PASSWORD in the environment.")
    lines = [l.strip() for l in io.open(path, encoding="utf-8")
             .read().splitlines() if l.strip()]
    if len(lines) < 2:
        raise SystemExit(f"[STOP] {path} needs two lines: e-mail, password.")
    return lines[0], lines[1]


def token() -> str:
    """A valid access token, refreshed when it is close to expiring."""
    if _token["value"] and time.time() < _token["expires"] - 120:
        return _token["value"]

    user, password = _creds()
    body = urllib.parse.urlencode({
        "grant_type": "password", "client_id": "cdse-public",
        "username": user, "password": password}).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            tok = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"[STOP] CDSE rejected the credentials in {CRED_FILE} "
            f"(HTTP {e.code}).\n"
            f"       Check the e-mail and password are current -- a stale file "
            f"is the usual cause.") from e

    _token["value"] = tok["access_token"]
    _token["expires"] = time.time() + float(tok.get("expires_in", 1800))
    return _token["value"]


def scenes_on(date: str, bbox: dict) -> list:
    """
    IW GRDH products intersecting the bbox on one UTC day, from the PUBLIC
    catalogue -- no credentials needed.

    This is the authority on whether an image exists.  GEE's absence proves
    nothing; this endpoint is ESA's own index.
    """
    poly = (f"POLYGON(({bbox['west']} {bbox['south']},{bbox['east']} "
            f"{bbox['south']},{bbox['east']} {bbox['north']},{bbox['west']} "
            f"{bbox['north']},{bbox['west']} {bbox['south']}))")
    flt = (f"Collection/Name eq 'SENTINEL-1' and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
           f"ContentDate/Start gt {date}T00:00:00.000Z and "
           f"ContentDate/Start lt {date}T23:59:59.999Z")
    url = CATALOGUE_URL + "?" + urllib.parse.urlencode({"$filter": flt,
                                                        "$top": "50"})
    with urllib.request.urlopen(url, timeout=120) as r:
        vals = json.load(r)["value"]
    # One acquisition is published twice, as .SAFE and as a COG variant; keep
    # one row per sensing start so counts mean "frames", not "packagings".
    seen, out = set(), []
    for v in vals:
        name = v.get("Name", "")
        if "_IW_GRDH" not in name:
            continue
        start = (v.get("ContentDate") or {}).get("Start", "")
        if start in seen:
            continue
        seen.add(start)
        out.append({"name": name, "start": start,
                    "online": v.get("Online", False)})
    return sorted(out, key=lambda d: d["start"])


EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "dataMask"]}],
          output: {id: "default", bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.VV, s.dataMask]; }
"""


def _request_tile(date: str, bbox: list, width: int, height: int,
                  orbit_direction: str) -> np.ndarray:
    """One Process API call -> (2, h, w) float32 array of [VV, dataMask]."""
    body = {
        "input": {
            "bounds": {"bbox": bbox,
                       "properties": {"crs":
                                      "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": f"{date}T00:00:00Z",
                                  "to": f"{date}T23:59:59Z"},
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                    "orbitDirection": orbit_direction,
                },
                # Terrain correction matters here: Kale sits against the Chin
                # Hills, and an ellipsoid product would put slope-driven
                # brightness differences into the pre/post difference.
                "processing": {"orthorectify": True,
                               "demInstance": "COPERNICUS",
                               "backCoeff": "GAMMA0_TERRAIN"},
            }],
        },
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default",
                                  "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    last = None
    for attempt in range(1, SH_MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(
                PROCESS_URL, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + token(),
                         "Content-Type": "application/json",
                         "Accept": "image/tiff"})
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = r.read()
            with rasterio.io.MemoryFile(raw) as mem, mem.open() as ds:
                return ds.read().astype(np.float32)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:180]}"
            # 401 means the token aged out mid-run; drop it and let token()
            # mint a fresh one on the retry.
            if e.code == 401:
                _token["value"] = None
        except Exception as e:                              # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < SH_MAX_ATTEMPTS:
            wait = 15 * attempt
            print(f"      [!] tile attempt {attempt}/{SH_MAX_ATTEMPTS} failed "
                  f"({last}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Sentinel Hub tile failed after "
                       f"{SH_MAX_ATTEMPTS} attempts -- {last}")


def grid_for(bbox: dict) -> tuple:
    """(width, height, transform) for the AOI bbox at native 10 m."""
    width = int(round((bbox["east"] - bbox["west"]) / DEG_PER_10M))
    height = int(round((bbox["north"] - bbox["south"]) / DEG_PER_10M))
    transform = from_origin(bbox["west"], bbox["north"],
                            DEG_PER_10M, DEG_PER_10M)
    return width, height, transform


def fetch_vv_db(date: str, bbox: dict, orbit_direction: str,
                out_path: str) -> str:
    """
    Terrain-corrected gamma0 VV in dB for one date, at native 10 m.

    Written as a two-band float32 GeoTIFF: band 1 dB, band 2 the data mask.
    Areas the pass did not cover come back masked rather than as zeros, so the
    caller can tell "no data" from "very dark water" -- the distinction the
    whole flood product rests on.
    """
    width, height, transform = grid_for(bbox)
    cols = int(np.ceil(width / SH_TILE_PX))
    rows = int(np.ceil(height / SH_TILE_PX))
    print(f"  CDSE {date}: {width} x {height} px at 10 m, "
          f"{cols * rows} tile(s) of <= {SH_TILE_PX} px")

    vv = np.zeros((height, width), np.float32)
    dm = np.zeros((height, width), np.float32)

    n = 0
    for r in range(rows):
        y0 = r * SH_TILE_PX
        y1 = min(height, y0 + SH_TILE_PX)
        for c in range(cols):
            x0 = c * SH_TILE_PX
            x1 = min(width, x0 + SH_TILE_PX)
            n += 1
            tb = [bbox["west"] + x0 * DEG_PER_10M,
                  bbox["north"] - y1 * DEG_PER_10M,
                  bbox["west"] + x1 * DEG_PER_10M,
                  bbox["north"] - y0 * DEG_PER_10M]
            print(f"    tile {n}/{cols * rows} ({x1 - x0} x {y1 - y0}) ...")
            arr = _request_tile(date, tb, x1 - x0, y1 - y0, orbit_direction)
            vv[y0:y1, x0:x1] = arr[0]
            dm[y0:y1, x0:x1] = arr[1]

    # Linear gamma0 -> dB. Non-positive samples are masked first: log10(0) is
    # -inf and would poison the speckle filter's whole neighbourhood.
    valid = (dm > 0.5) & (vv > 0)
    db = np.full((height, width), np.nan, np.float32)
    db[valid] = 10.0 * np.log10(vv[valid])

    profile = {"driver": "GTiff", "height": height, "width": width, "count": 2,
               "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
               "compress": "deflate", "predictor": 2, "tiled": True}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(db, 1)
        dst.write(valid.astype(np.float32), 2)
    cov = float(valid.mean())
    print(f"  [OK] {out_path}  valid {cov:.1%} of the bbox")
    return out_path
