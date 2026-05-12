#!/usr/bin/env python3
"""Build CSI300 + CSI500 combined index constituent cache.

CSI300 (000300.SH): large-cap, top 300 stocks
CSI500 (000905.SH): mid-cap, stocks ranked 301-800
Together: top 800 stocks by market cap + liquidity
"""

import os, sys, pickle, time
import pandas as pd
import tushare as ts

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, 'output', 'index_constituent_lookup.pkl')
OLD_CACHE = os.path.join(BASE, 'output', 'index_constituent_lookup.pkl')

pro = ts.pro_api()

# Fetch CSI300 constituents for all months where data exists
INDEX_CODE = '000300.SH'
MONTHS = [f'{y}{m:02d}01' for y in range(2016, 2027) for m in range(1, 13) if (y < 2026 or m <= 5)]

print(f"Fetching CSI300 ({INDEX_CODE}) constituents for {len(MONTHS)} months...")

csi300_data = {}  # ym -> set of con_codes

for i, m in enumerate(MONTHS):
    # Convert 20160101 -> last day of that month
    trade_date = m
    ym = f'{m[:4]}-{m[4:6]}'

    try:
        df = pro.index_weight(index_code=INDEX_CODE, trade_date=trade_date)
        if len(df) > 0:
            codes = set(df['con_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6])
            csi300_data[ym] = codes
            print(f"  [{i+1}/{len(MONTHS)}] {trade_date} → {len(codes)} stocks")
        else:
            csi300_data[ym] = set()
            print(f"  [{i+1}/{len(MONTHS)}] {trade_date} → 0 stocks (no data)")
    except Exception as e:
        # Try last day of month
        try:
            dt = pd.Timestamp(trade_date)
            last_day = dt + pd.offsets.MonthEnd(0)
            trade_date2 = last_day.strftime('%Y%m%d')
            df = pro.index_weight(index_code=INDEX_CODE, trade_date=trade_date2)
            if len(df) > 0:
                codes = set(df['con_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6])
                csi300_data[ym] = codes
                print(f"  [{i+1}/{len(MONTHS)}] {trade_date2} (alt) → {len(codes)} stocks")
            else:
                csi300_data[ym] = set()
                print(f"  [{i+1}/{len(MONTHS)}] {trade_date2} (alt) → 0 stocks")
        except Exception as e2:
            csi300_data[ym] = set()
            print(f"  [{i+1}/{len(MONTHS)}] {trade_date} ERROR: {e2}")

    time.sleep(0.15)  # rate limit

# Load existing CSI500 data from old cache
print("\nLoading existing CSI500 data...")
if os.path.exists(OLD_CACHE):
    with open(OLD_CACHE, 'rb') as f:
        old = pickle.load(f)
    print(f"  Old cache has indices: {old.get('indices')}")

    # Extract CSI500 only from the old lookup
    # Old lookup: (ym, symbol) -> True for CSI500+CSI1000
    # We need to filter to CSI500 only. But the lookup doesn't separate by index.
    # We need to re-fetch CSI500 or extract from stats.

    # Actually, let's just re-fetch CSI500 too for consistency
    print("  Will re-fetch CSI500 for consistency...")
else:
    print("  No old cache found, will fetch CSI500 fresh")

# Fetch CSI500
INDEX_CODE_500 = '000905.SH'
print(f"\nFetching CSI500 ({INDEX_CODE_500}) constituents...")

csi500_data = {}

for i, m in enumerate(MONTHS):
    trade_date = m
    ym = f'{m[:4]}-{m[4:6]}'

    try:
        df = pro.index_weight(index_code=INDEX_CODE_500, trade_date=trade_date)
        if len(df) > 0:
            codes = set(df['con_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6])
            csi500_data[ym] = codes
            print(f"  [{i+1}/{len(MONTHS)}] {trade_date} → {len(codes)} stocks")
        else:
            csi500_data[ym] = set()
            print(f"  [{i+1}/{len(MONTHS)}] {trade_date} → 0 stocks (no data)")
    except Exception as e:
        try:
            dt = pd.Timestamp(trade_date)
            last_day = dt + pd.offsets.MonthEnd(0)
            trade_date2 = last_day.strftime('%Y%m%d')
            df = pro.index_weight(index_code=INDEX_CODE_500, trade_date=trade_date2)
            if len(df) > 0:
                codes = set(df['con_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6])
                csi500_data[ym] = codes
                print(f"  [{i+1}/{len(MONTHS)}] {trade_date2} (alt) → {len(codes)} stocks")
            else:
                csi500_data[ym] = set()
                print(f"  [{i+1}/{len(MONTHS)}] {trade_date2} (alt) → 0 stocks")
        except Exception as e2:
            csi500_data[ym] = set()
            print(f"  [{i+1}/{len(MONTHS)}] {trade_date} ERROR: {e2}")

    time.sleep(0.15)

# Forward-fill gaps
print("\nForward-filling gaps...")
sorted_yms = sorted(MONTHS)
sorted_yms_fmt = [f'{m[:4]}-{m[4:6]}' for m in sorted_yms]

for data_dict, name in [(csi300_data, 'CSI300'), (csi500_data, 'CSI500')]:
    last_data = set()
    gap_count = 0
    for ym in sorted_yms_fmt:
        if ym in data_dict and len(data_dict[ym]) > 0:
            last_data = data_dict[ym]
        elif last_data:
            data_dict[ym] = last_data.copy()
            gap_count += 1
    print(f"  {name}: {gap_count} gaps forward-filled")

# Build combined lookup
print("\nBuilding combined CSI300+CSI500 lookup...")
lookup = {}
stats = {}

for ym in sorted_yms_fmt:
    c300 = csi300_data.get(ym, set())
    c500 = csi500_data.get(ym, set())
    union = c300 | c500

    for sym in union:
        lookup[(ym, sym)] = True

    stats[ym] = {'CSI300': len(c300), 'CSI500': len(c500), 'union': len(union)}

total_entries = len(lookup)
print(f"  Total lookup entries: {total_entries}")

# Save
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
with open(CACHE_PATH, 'wb') as f:
    pickle.dump({
        'lookup': lookup,
        'stats': stats,
        'indices': ['CSI300', 'CSI500']
    }, f)

print(f"\nCache saved: {CACHE_PATH}")
print(f"Size: {os.path.getsize(CACHE_PATH) / 1024 / 1024:.1f} MB")

# Print summary stats
print("\nSummary (first/last 3 months):")
for ym in sorted_yms_fmt[:3] + ['...'] + sorted_yms_fmt[-3:]:
    if ym == '...':
        print('...')
    elif ym in stats:
        print(f"  {ym}: CSI300={stats[ym]['CSI300']}, CSI500={stats[ym]['CSI500']}, union={stats[ym]['union']}")
