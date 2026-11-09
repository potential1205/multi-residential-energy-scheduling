"""Collect and process extra household datasets into one common format.

    python -m experiments.datasets fetch      # download raw TMY3 loads, NOAA stations, Ausgrid, PVGIS PV (~170 MB)
    python -m experiments.datasets build      # raw files -> data/processed/<collection>/<site>/
    python -m experiments.datasets summary    # table of all processed sites

Common format (one folder per household):
    load.npy, generation.npy   float32, shape (365, 24), kWh per hour, local standard time,
                               day 0 = January 1, hour h = [h:00, h+1:00)
    meta.json                  source, scaling, coordinates, time zone, statistics, raw-file SHA-256

Units: physical kWh, no rounding. The schedulers work in 0.1 kWh units; the conversion (x10)
happens once, in `load_profile`. The original repository data (data/load, data/generation) were
built as floor(kWh x 5): halved and rounded *down* to 0.1 kWh, which drops ~11% of the energy.
They are kept untouched for paper reproduction.

Scaling: households are halved as in the original preprocessing. Where a halved profile would still
exceed the grid purchase limit C_max = 2 kWh (demand could then be impossible to serve), the site is
scaled further so that its peak is PEAK_LIMIT_KWH. PV is a 0.5 kWp system (as in the paper).
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data'/'raw'
PROCESSED = ROOT/'data'/'processed'
HOUSEHOLD_SCALE = 0.5
PEAK_LIMIT_KWH = 1.95
PV_KWP = 0.5
PVGIS_YEAR = 2019
PVGIS_URL = ('https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat={lat}&lon={lon}&peakpower={kwp}&loss=14'
             '&startyear={year}&endyear={year}&pvcalculation=1&optimalangles=1&outputformat=json')
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# UTC offset of local *standard* time (TMY3 files use standard time, no daylight saving).
STATE_UTC = dict(AK=-9, HI=-10, WA=-8, OR=-8, CA=-8, NV=-8, AZ=-7, CO=-7, TX=-6, OK=-6, LA=-6, MO=-6, TN=-6,
                 IL=-6, MN=-6, WI=-6, ND=-6, KY=-5, GA=-5, NC=-5, VA=-5, MD=-5, NY=-5, MA=-5, PA=-5, OH=-5,
                 FL=-5, ME=-5)
UTC_EXCEPTIONS = {'722700': -7}  # El Paso, TX is on Mountain time
EXTRA_PV_SITES = {'KOR_Seoul': dict(lat=37.5665, lon=126.978, utc=9)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(load, gen):
    return dict(load_kwh_year=float(load.sum()), load_peak_kwh=float(load.max()),
                load_mean_kwh=float(load.mean()), pv_kwh_year=float(gen.sum()), pv_peak_kwh=float(gen.max()),
                hours_over_cmax=int((load > 2.0).sum()))


def write_site(collection, site, load, gen, meta):
    assert load.shape == gen.shape == (365, 24) and np.isfinite(load).all() and np.isfinite(gen).all()
    assert (load >= 0).all() and (gen >= 0).all() and load.max() <= 2.0 + 1e-6, site
    folder = PROCESSED/collection/site
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/'load.npy', load.astype(np.float32))
    np.save(folder/'generation.npy', gen.astype(np.float32))
    meta = dict(collection=collection, site=site, units='kWh per hour', shape=[365, 24],
                time='local standard time; day 0 = Jan 1; hour h covers [h, h+1)', **meta, **stats(load, gen))
    (folder/'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def load_profile(name, kind):
    """Profile in scheduler units (0.1 kWh). `name` is a legacy file ('x.npy') or a processed site
    ('tmy3/USA_CA_Oakland...'). Shared by TDAS.py, Co-TDAS.py, core and reports."""
    if name.endswith('.npy'):
        return np.load(ROOT/'data'/kind/name, allow_pickle=False)
    return np.load(PROCESSED/name/f'{kind}.npy', allow_pickle=False).astype(np.float64) * 10


# ---------------------------------------------------------------- sources
def stations():
    """USAF -> (lat, lon) from NOAA's ISD station history."""
    import csv
    out = {}
    with (RAW/'noaa'/'isd-history.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                out.setdefault(row['USAF'], (float(row['LAT']), float(row['LON'])))
            except ValueError:
                pass
    return out


def tmy3_sites():
    for path in sorted((RAW/'tmy3_base').glob('USA_*_TMY3_BASE.csv')):
        name = path.name.removesuffix('.csv')
        usaf = name.split('_TMY3')[0].rsplit('.', 1)[-1]
        state = name.split('_')[1]
        yield path, name, usaf, state


def pv_targets():
    coords = stations()
    for _, name, usaf, state in tmy3_sites():
        lat, lon = coords[usaf]
        yield name, dict(lat=lat, lon=lon, utc=UTC_EXCEPTIONS.get(usaf, STATE_UTC[state]), usaf=usaf)
    yield from EXTRA_PV_SITES.items()


TMY3_BASE_URL = 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/'
DOWNLOADS = {
    RAW/'noaa'/'isd-history.csv': 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv',
    RAW/'ausgrid'/'Ausgrid_solar_home_data.zip': 'https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip',
}


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=600) as response, temporary.open('wb') as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    temporary.replace(target)
    print('downloaded', target.relative_to(ROOT), target.stat().st_size, flush=True)


def fetch():
    sites = (Path(__file__).with_name('tmy3_sites.txt')).read_text().split()
    for name in sites:
        download(TMY3_BASE_URL + name, RAW/'tmy3_base'/name)
    for target, url in DOWNLOADS.items():
        download(url, target)
    fetch_pvgis()
    record = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(RAW.rglob('*')) if p.is_file() and not p.name.startswith('_')}
    (RAW/'manifest.json').write_text(json.dumps(record, indent=1), encoding='utf-8')


def fetch_pvgis(delay=1.0):
    folder = RAW/'pvgis'
    folder.mkdir(parents=True, exist_ok=True)
    for name, site in pv_targets():
        target = folder/f'{name}.json'
        if target.exists():
            continue
        url = PVGIS_URL.format(lat=site['lat'], lon=site['lon'], kwp=1, year=PVGIS_YEAR)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    body = response.read()
                json.loads(body)
                target.write_bytes(body)
                print('fetched', name, len(body), flush=True)
                break
            except Exception as exc:  # network hiccups and PVGIS rate limits
                print('retry', name, attempt, exc, flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f'PVGIS failed for {name}')
        time.sleep(delay)


def pvgis_profile(name, utc, kwp=PV_KWP):
    """Hourly kWh for a `kwp` system, shifted from UTC to local standard time, Feb 29 never present."""
    data = json.loads((RAW/'pvgis'/f'{name}.json').read_text())
    hourly = data['outputs']['hourly']
    watts = np.array([h['P'] for h in hourly], dtype=float)  # W from a 1 kWp system, hour starting at stamp
    if len(watts) != 8760:
        raise ValueError(f'{name}: expected 8760 hours, got {len(watts)}')
    # Local hour index k corresponds to UTC hour k - utc; wrap the few hours at the year edge.
    local = np.roll(watts, utc)
    return (local / 1000 * kwp).reshape(365, 24), data['inputs']


def build_tmy3():
    import pandas as pd
    coords = stations()
    metas = []
    for path, name, usaf, state in tmy3_sites():
        kwh = pd.read_csv(path).iloc[:, 1].to_numpy(dtype=float)  # Electricity:Facility [kW], hourly
        if len(kwh) != 8760:
            raise ValueError(name)
        scale = min(HOUSEHOLD_SCALE, PEAK_LIMIT_KWH / kwh.max())
        load = (kwh * scale).reshape(365, 24)
        gen, pv_inputs = pvgis_profile(name, UTC_EXCEPTIONS.get(usaf, STATE_UTC[state]))
        lat, lon = coords[usaf]
        metas.append(write_site('tmy3', name, load, gen, dict(
            source='NREL Commercial and Residential Hourly Load Profiles for all TMY3 Locations (BASE), OpenEI dataset 961',
            source_url=f'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/{path.name}',
            raw_sha256=sha256(path), load_scale=scale, load_scale_reason='halved' if scale == HOUSEHOLD_SCALE else f'peak limited to {PEAK_LIMIT_KWH} kWh',
            pv=dict(source='PVGIS v5.3 seriescalc', year=PVGIS_YEAR, kwp=PV_KWP, loss_percent=14, angles='optimal',
                    radiation_db=pv_inputs['meteo_data']['radiation_db'], raw_sha256=sha256(RAW/'pvgis'/f'{name}.json')),
            state=state, usaf=usaf, lat=lat, lon=lon, utc_offset=UTC_EXCEPTIONS.get(usaf, STATE_UTC[state]),
            note='Typical-year (TMY3) load with 2019 PV weather; load and PV are not from the same year.')))
    return metas


def build_ausgrid(min_days=365):
    """Ausgrid Solar Home data, financial year 2012-13, re-ordered to a calendar year (Jan..Dec).
    Load = general consumption (GC); PV = gross generation (GG). Controlled load (CL, off-peak hot water on
    its own meter and tariff) is excluded: it is not schedulable under TOU/DC and its 2.5 kWh bursts would
    force most homes far below the half-size scale."""
    import pandas as pd
    archive = RAW/'ausgrid'/'Ausgrid_solar_home_data.zip'
    with zipfile.ZipFile(archive) as z:
        raw = z.read('Solar home 2012-2013.csv')
    frame = pd.read_csv(io.BytesIO(raw), skiprows=1)
    frame['date'] = pd.to_datetime(frame['date'], format='%d/%m/%Y')
    halfhours = [c for c in frame.columns if ':' in c]
    if len(halfhours) != 48:
        raise ValueError('Unexpected Ausgrid layout')
    metas = []
    for customer, rows in frame.groupby('Customer'):
        table = {cat: part.set_index('date')[halfhours] for cat, part in rows.groupby('Consumption Category')}
        if 'GC' not in table or 'GG' not in table:
            continue
        days = pd.date_range('2012-07-01', '2013-06-30')
        gc = table['GC'].reindex(days)
        gg = table['GG'].reindex(days)
        if gc.isna().any().any() or gg.isna().any().any():
            continue
        def hourly(part):
            values = part.to_numpy(dtype=float)
            return values.reshape(len(days), 24, 2).sum(axis=2)
        load_fy, gen_fy = hourly(gc), hourly(gg)
        # Calendar order: Jan 1 - Jun 30 2013, then Jul 1 - Dec 31 2012.
        order = np.r_[184:365, 0:184]
        load_raw, gen_raw = load_fy[order], gen_fy[order]
        capacity = float(rows['Generator Capacity'].iloc[0])
        scale = min(HOUSEHOLD_SCALE, PEAK_LIMIT_KWH / max(load_raw.max(), 1e-9))
        pv_scale = PV_KWP / capacity if capacity > 0 else 0
        if load_raw.sum() < 365 or gen_raw.sum() <= 0:
            continue
        site = f'AU_NSW_customer{int(customer):03d}'
        metas.append(write_site('ausgrid', site, load_raw * scale, gen_raw * pv_scale, dict(
            source='Ausgrid Solar Home Electricity Data, 1 July 2012 - 30 June 2013 (CC BY 3.0 AU)',
            source_url='https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip (mirror; original Ausgrid page retired)',
            raw_sha256=sha256(archive), customer=int(customer), postcode=int(rows['Postcode'].iloc[0]),
            generator_kwp=capacity, load_scale=scale, pv_scale=pv_scale, controlled_load_excluded=bool('CL' in table),
            note='Southern hemisphere: Jan is summer. Calendar order Jan-Jun 2013 then Jul-Dec 2012. '
                 'PV rescaled to 0.5 kWp; load halved (or peak-limited) as for TMY3.')))
    return metas


def build_seoul():
    """Seoul load from the original repository (only the 0.1 kWh rounded version exists) + PVGIS PV."""
    load = np.load(ROOT/'data'/'load'/'table1_load.npy').astype(float) / 10
    gen, inputs = pvgis_profile('KOR_Seoul', 9)
    return [write_site('korea', 'KOR_Seoul', load, gen, dict(
        source='Original repository table1_load.npy (KEPCO Seoul profile, already 0.1 kWh-rounded)',
        load_scale=1.0, pv=dict(source='PVGIS v5.3 seriescalc', year=PVGIS_YEAR, kwp=PV_KWP,
                                radiation_db=inputs['meteo_data']['radiation_db']),
        lat=37.5665, lon=126.978, utc_offset=9))]


def build():
    metas = build_tmy3() + build_ausgrid() + build_seoul()
    index = {m['collection'] + '/' + m['site']: {k: m[k] for k in ('load_kwh_year', 'load_peak_kwh', 'pv_kwh_year', 'load_scale')}
             for m in metas}
    (PROCESSED/'index.json').write_text(json.dumps(index, indent=1), encoding='utf-8')
    print(f'{len(metas)} sites written to {PROCESSED}')


def summary():
    index = json.loads((PROCESSED/'index.json').read_text())
    by = {}
    for name, row in index.items():
        by.setdefault(name.split('/')[0], []).append(row)
    for collection, rows in by.items():
        load = np.array([r['load_kwh_year'] for r in rows]);pv = np.array([r['pv_kwh_year'] for r in rows])
        print(f'{collection:8} sites {len(rows):3}  load {load.mean():6.0f} kWh/yr (min {load.min():.0f}, max {load.max():.0f})'
              f'  PV {pv.mean():5.0f} kWh/yr  peak-limited {sum(r["load_scale"] < HOUSEHOLD_SCALE for r in rows)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['fetch', 'pvgis', 'build', 'summary'])
    args = parser.parse_args()
    {'fetch': fetch, 'pvgis': fetch_pvgis, 'build': build, 'summary': summary}[args.command]()


if __name__ == '__main__':
    main()
