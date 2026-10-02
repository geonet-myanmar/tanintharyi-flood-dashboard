"""
Patches to flood_mapping.PAGE_TEMPLATE for the CDSE Tanintharyi product.

Kept separate so flood_mapping.py (shared with the other dashboards) is never
edited: this module rewrites the template string in memory before the page is
built.  It adds

  * a "Village tracts" tab and map layer -- MIMU adm4 (UNHCR/MIMU 250k), the
    unit below township, with 10 m flood sums from the same flags raster;
  * provenance that says where the imagery actually came from (CDSE Sentinel
    Hub, gamma0 terrain-corrected), not Earth Engine;
  * the acquisition time and platform in the Method tab.

Every patch asserts its anchor exists, so a template change upstream fails
loudly instead of silently dropping a feature.
"""

import flood_mapping as F

SITE = "https://geonet-myanmar.github.io/tanintharyi-flood-dashboard/"

extra_payload = None          # set by the driver; None on --page-only
extra_meta = None             # meta keys added on --page-only

_orig_write_page = F._write_page


def _write_page(payload, output):
    if extra_payload:
        payload.update(extra_payload)
    if extra_meta:
        payload["meta"].update(extra_meta)
    _orig_write_page(payload, output)


def _sub(t, old, new):
    assert old in t, f"template anchor missing: {old[:60]!r}"
    return t.replace(old, new, 1)


VT_SECTION = """      <section class="view" id="v-vt">
        <div class="toolbar">
          <input type="search" id="vt-q" placeholder="Search village tract or township…" aria-label="Search village tracts">
          <select id="vt-sort" aria-label="Sort village tracts">
            <option value="flood_km2">Flooded area</option>
            <option value="flood_pct">% flooded</option>
            <option value="name">Name</option>
          </select>
        </div>
        <div class="toolbar">
          <select id="vt-ts" aria-label="Township"><option value="">All townships</option></select>
        </div>
        <div id="vt-rows" class="rows"></div>
        <p class="note" id="vt-foot"></p>
      </section>
      <section class="view" id="v-lyr"></section>"""

VT_JS = r"""/* ═══ village tracts (MIMU adm4) ══════════════════════════════════════════ */
/* Same 10 m flags as the township sums, rasterised per village tract and
   clipped to the AOI. Off by default on the map: at AOI zoom ~300 small
   polygons are a mosaic; the tab is the readable form, and clicking a row
   switches the layer on and zooms to the tract. */
const VT = FLOOD.vts || [];
const vtStyle = t => ({
  color: '#ffffff', weight: 0.6, opacity: 0.7,
  fillColor: choro(t.flood_pct) || '#ffffff',
  fillOpacity: choro(t.flood_pct) ? 0.5 : 0.03
});
const gVt = L.geoJSON(FLOOD.vtGeo || {type: 'FeatureCollection', features: []}, {
  pane: 'p-choro',
  style: f => vtStyle(VT[f.properties.i]),
  onEachFeature: (f, lyr) => {
    const t = VT[f.properties.i]; t._lyr = lyr;
    lyr.bindTooltip(() =>
      '<b>' + esc(t.vt) + '</b> village tract' +
      (t.vt_mmr ? ' · ' + esc(t.vt_mmr) : '') + '<br>' + esc(t.ts) +
      ' township<br>' + km2(t.flood_km2) + ' km² newly flooded · ' +
      num(t.flood_pct, 1) + '% of it', {sticky: true});
    lyr.on('mouseover', () => lyr.setStyle({weight: 2, opacity: 1}));
    lyr.on('mouseout',  () => lyr.setStyle({weight: 0.6, opacity: 0.7}));
  }
});
LAYERS.splice(2, 0, {k: 'vt', g: gVt, n: 'Village tract flood intensity',
  ramp: true, d: VT.length + ' MIMU village tracts · % newly flooded', on: false});

let vtSort = 'flood_km2', vtQuery = '', vtTs = '';
[...new Set(VT.map(t => t.ts))].sort().forEach(n => {
  const o = el('option', null, n); o.value = n; $('#vt-ts').append(o);
});
function renderVt(){
  const q = vtQuery.toLowerCase();
  let rows = VT.filter(t => (!vtTs || t.ts === vtTs) &&
    (!q || (t.vt + ' ' + t.vt_mmr + ' ' + t.ts).toLowerCase().includes(q)));
  rows = rows.slice().sort((a, b) => vtSort === 'name'
    ? a.vt.localeCompare(b.vt) : b[vtSort] - a[vtSort]);
  const box = $('#vt-rows'); box.textContent = '';
  const sk = vtSort === 'name' ? 'flood_km2' : vtSort;
  const max = Math.max(1e-9, ...rows.map(r => r[sk]));
  rows.slice(0, 400).forEach(t => {
    const r = el('div', 'row');
    const nm = el('div', 'nm', t.vt);
    nm.append(el('small', null, (t.vt_mmr ? t.vt_mmr + ' · ' : '') + t.ts));
    const val = el('div', 'val', km2(t.flood_km2) + ' km²');
    val.append(el('small', null, num(t.flood_pct, 1) + '% of ' +
                  num(t.area_km2, 1) + ' km²'));
    const bar = el('div', 'bar'); const fill = el('i');
    fill.style.width = (100 * Math.max(t[sk], 0) / max) + '%';
    fill.style.background = choro(t.flood_pct) || 'var(--axis)';
    bar.append(fill);
    r.append(nm, val, bar);
    if (t.land_frac < 0.98){
      const w = el('div', 'warnrow');
      w.append(el('span', null, '⚠'), el('span', null, 'only ' +
        num(t.land_frac * 100, 0) + '% of its land had usable data on both dates'));
      r.append(w);
    }
    r.onclick = () => {
      setLayer('vt', true);
      if (t._lyr){ map.fitBounds(t._lyr.getBounds(), {padding: [24, 24]});
                   t._lyr.openTooltip(); }
    };
    r.onmouseenter = () => t._lyr && t._lyr.setStyle({weight: 2, opacity: 1});
    r.onmouseleave = () => t._lyr && t._lyr.setStyle({weight: 0.6, opacity: 0.7});
    box.append(r);
  });
  const fl = rows.filter(t => t.flood_km2 >= 0.1).length;
  $('#vt-foot').innerHTML = '<b>' + rows.length + '</b> village tract' +
    (rows.length === 1 ? '' : 's') + ' shown, ' + fl + ' with at least 0.1 km² ' +
    'of new water. Boundaries: MIMU adm4 (UNHCR/MIMU 1:250,000), clipped to the ' +
    'AOI. Area summed from 10 m pixels; "% flooded" is of the whole tract, ' +
    'hills and sea-free land included, so a lowland tract reads lower than its ' +
    'paddy actually is.';
}
$('#vt-q').oninput = e => { vtQuery = e.target.value; renderVt(); };
$('#vt-sort').onchange = e => { vtSort = e.target.value; renderVt(); };
$('#vt-ts').onchange = e => { vtTs = e.target.value; renderVt(); };
renderVt();

/* ═══ layers view ═════════════════════════════════════════════════════════ */"""


C5_JS = r"""  const c5 = el('div', 'callout');
  c5.innerHTML = '<b>Gamma0, terrain-corrected.</b> Both dates come ' +
    'from the same source and track, so the -3 dB change rule is ' +
    'internally consistent. The absolute figures are not ' +
    'interchangeable with the Earth Engine (sigma0) dashboards in ' +
    'this series: on slopes the two calibrations differ by several dB.';
  v.append(c5);
  if (M.patches){
    const P = M.patches, tot = P['lt_0.1ha'] + P['0.1_1ha'] + P['1_10ha'] + P['ge_10ha'];
    const small = P['lt_0.1ha'] + P['0.1_1ha'];
    const c6 = el('div', 'callout');
    c6.innerHTML = '<b>How much is coherent water?</b> Of the ' + km2(tot) +
      ' km² flagged, ' + km2(P['ge_10ha']) + ' km² lies in patches of 10 ha or more and ' +
      km2(P['1_10ha']) + ' km² in 1–10 ha patches. ' + km2(small) + ' km² (' +
      num(100 * small / tot, 0) + '%) is in patches under 1 ha — ' +
      km2(P['lt_0.1ha']) + ' km² in specks of fewer than ten 10 m pixels. ' +
      'Like the other dashboards in this series the method has no minimum ' +
      'mapping unit, so that fringe is counted; much of it is residual speckle or ' +
      'vegetation change rather than standing water. Read the headline as an ' +
      'upper bound, the ≥ 1 ha figure (' + km2(P['1_10ha'] + P['ge_10ha']) +
      ' km²) as the conservative one.';
    v.append(c6);
  }
  const c4 = el('div', 'note');"""


def patch_template():
    t = F.PAGE_TEMPLATE
    if "id=\"v-vt\"" in t:
        return
    t = _sub(t, '<meta property="og:type" content="website">',
             '<meta property="og:type" content="website">\n'
             f'<meta property="og:url" content="{SITE}">\n'
             f'<link rel="canonical" href="{SITE}">')
    t = _sub(t, '      <section class="view" id="v-lyr"></section>', VT_SECTION)
    t = _sub(t, "const TABS = [['ts', 'Townships'], ['vil', 'Villages'],",
             "const TABS = [['ts', 'Townships'], ['vil', 'Villages'], "
             "['vt', 'Village tracts'],")
    t = _sub(t, "/* ═══ layers view ═════════════════════════════════════════════════════════ */",
             VT_JS)
    t = _sub(t, "Imagery: Copernicus Sentinel-1 (ESA) via Google Earth Engine. ",
             "Imagery: Copernicus Sentinel-1 (ESA), fetched from the Copernicus "
             "Data Space Ecosystem (Sentinel Hub Process API) and analysed "
             "locally. Masks via Google Earth Engine. ")
    t = _sub(t, "Boundaries and village points: <a",
             "Township, village-tract, village and town data: <a")
    t = _sub(t, "  kv('Collection', 'COPERNICUS/' + M.source);",
             "  if (M.platform) kv('Platform', M.platform + ', ' + M.pre_time + "
             "' on both dates');\n"
             "  kv('Imagery source', M.source);")
    t = _sub(t, "  kv('Village points', 'MIMU pplp2 250k, PCode v9.7');",
             "  kv('Village tracts', 'MIMU adm4 (UNHCR/MIMU) 250k, clipped to AOI');\n"
             "  kv('Village points', 'MIMU pplp2 250k, PCode v9.7');")
    t = _sub(t, "  const c4 = el('div', 'note');", C5_JS)
    F.PAGE_TEMPLATE = t
    F._write_page = _write_page
