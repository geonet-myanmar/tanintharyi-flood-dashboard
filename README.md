# Tanintharyi flood inundation dashboard — 20 September → 2 October 2026

Sentinel-1 SAR flood mapping over seven townships of Tanintharyi Region (Dawei,
Launglon, Thayetchaung, Yebyu, Palaw, Tanintharyi, Bokpyin), summarised per MIMU
township, village tract, village point and town.

**Live dashboard → https://geonet-myanmar.github.io/tanintharyi-flood-dashboard/**

## Headline figures

| | |
|---|---|
| New inundation | **315 km²** (0.94 % of the 33,591 km² AOI) — an upper bound, see below |
| In patches of 1 ha or more | **170 km²** — the conservative figure |
| Land assessed | **99.9 %** (88 km² of sea excluded) |
| Villages with water at the settlement | **103** of 902 MIMU village points |
| Villages with water within 500 m | **341** (includes the 103 above) |
| Village tracts | 223 MIMU adm4 tracts summarised |
| Resolution | **10 m native**, every layer and every statistic |

| Township | District | New water km² | % of township | Villages hit / total |
|---|---|---:|---:|---:|
| Tanintharyi | Myeik | 181.3 | 1.6 | 73 / 194 |
| Launglon | Dawei | 34.0 | 4.3 | 3 / 113 |
| Dawei | Dawei | 22.8 | 0.3 | 5 / 94 |
| Thayetchaung | Dawei | 20.8 | 1.0 | 6 / 103 |
| Yebyu | Dawei | 18.8 | 0.5 | 5 / 140 |
| Palaw | Myeik | 18.7 | 0.8 | 11 / 155 |
| Bokpyin | Kawthoung | 18.7 | 0.3 | 0 / 103 |

The clearest signal is the **Tanintharyi River floodplain** south of Tanintharyi
town — Ta Moke Chone (35 km², 16 % of the tract), Tanintharyi Sin Chay Hpone
(21 km², 12 %) and Ta Ku (13 km², 13 %) village tracts. Every figure is in
[flood_stats.json](flood_stats.json).

## How much of it is coherent water?

The method has no minimum mapping unit (the same as the other dashboards in this
series), so isolated pixels count. Of the 315 km², 112.6 km² is in patches of
10 ha or more and 57.9 km² in 1–10 ha patches; **145 km² (46 %) is in patches
under 1 ha**, 42 km² of it in specks of fewer than ten 10 m pixels — mostly
residual speckle or vegetation change rather than standing water.

## Imagery

| | |
|---|---|
| Pre-flood | 2026-09-20, 11:35 UTC — 4 IW GRDH frames |
| Post-flood | 2026-10-02, 11:35 UTC — 4 IW GRDH frames |
| Platform / orbit | Sentinel-1D, ASCENDING, relative orbit 70 — same track, 12 days = 1 repeat cycle |
| Source | Copernicus Data Space Ecosystem, Sentinel Hub Process API |
| Radiometry | gamma0, terrain-corrected against the Copernicus DEM |
| Threshold | VV drop < −3 dB after a 30 m circular focal mean |
| Excluded | JRC permanent water (seasonality ≥ 4 months); HydroSHEDS slope ≥ 5°; sea |
| Village impact | new water within 150 m (settlement, ≥ 5 %) and 500 m (surroundings, ≥ 2 %) |

Gamma0 from CDSE is **not interchangeable** with the sigma0 (Earth Engine)
dashboards in this account: on slopes the two calibrations differ by several dB.
Both dates here come from the same source and track, so the change rule is
internally consistent. The JRC and HydroSHEDS masks are fetched from Earth Engine
on the same 10 m grid.

This is **change, not total water**: 20 September is itself a monsoon image, so
ground already flooded then is not in the red layer.

## Data

- Imagery: Copernicus Sentinel-1 (ESA), via the Copernicus Data Space Ecosystem.
- Boundaries, village tracts, village points and towns:
  [MIMU](https://geonode.themimu.info) — adm3 250k, adm4 (UNHCR/MIMU) 250k,
  pplp2 250k (Tanintharyi), pplp1 town gazetteer. Cached in `data/mimu/`.
- Permanent water: JRC Global Surface Water v1.4. Terrain: WWF HydroSHEDS.

## Rebuilding

```
python tanintharyi_flood_s1.py             # full run; resumes from local caches
python tanintharyi_flood_s1.py --check     # catalogue check only
python tanintharyi_flood_s1.py --page-only # re-render index.html, same numbers
```

The AOI is 1.07 gigapixels per layer at 10 m, so the analysis runs block by block
on the 2048 px Sentinel Hub tile grid. `flood_mapping.py` and `cdse_source.py` are
shared libraries (MIMU loaders, dashboard template, CDSE access);
`dashboard_extras.py` adds the village-tract tab and CDSE provenance to the
template. A full run needs a CDSE account (`credentials_CDSE.txt`, two lines:
e-mail, password — gitignored) and an Earth Engine project.

`assets/` (98 palette PNGs, ~490 MB) holds the 10 m SAR backdrops and **must be
committed with `index.html`** — the page loads them by relative path. Never
hand-edit `index.html`; re-run the script.
