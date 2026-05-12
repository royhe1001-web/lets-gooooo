#!/usr/bin/env python3
"""Rebuild CSI300 + CSI500 combined index constituent cache.

Uses month-end dates for tushare index_weight API.
CSI300 (000300.SH): large-cap, top 300
CSI500 (000905.SH): mid-cap, stocks 301-800
"""

import os, pickle, time
import pandas as pd
import tushare as ts

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, 'output', 'index_constituent_lookup.pkl')
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

pro = ts.pro_api()

# Build list of all year-months and their actual last day
months = []
for y in range(2015, 2027):
    for m in range(1, 13):
        if y == 2015 and m < 6:
            continue  # start from 2015-06
        if y == 2026 and m > 5:
            continue  # end at 2026-05
        dt = pd.Timestamp(f'{y}-{m:02d}-01') + pd.offsets.MonthEnd(0)
        trade_date = dt.strftime('%Y%m%d')
        ym = f'{y}-{m:02d}'
        months.append((ym, trade_date))

print(f"Total months: {len(months)}")
print(f"Date range: {months[0]} to {months[-1]}")

INDICES = [
    ('000300.SH', 'CSI300'),
    ('000905.SH', 'CSI500'),
]

all_data = {}  # index_name -> {ym: set of codes}

for index_code, index_name in INDICES:
    print(f"\n{'='*60}")
    print(f"Fetching {index_name} ({index_code})...")
    print(f"{'='*60}")

    data = {}
    calls = 0

    for i, (ym, trade_date) in enumerate(months):
        try:
            df = pro.index_weight(index_code=index_code, trade_date=trade_date)
            calls += 1

            if len(df) > 0:
                codes = set(df['con_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6])
                data[ym] = codes
                if i % 20 == 0 or len(data) <= 3:
                    print(f"  [{i+1}/{len(months)}] {trade_date} ({ym}) -> {len(codes)} stocks")
            else:
                data[ym] = set()
                if i < 5:
                    print(f"  [{i+1}/{len(months)}] {trade_date} ({ym}) -> 0 stocks")

        except Exception as e:
            err_msg = str(e)
            if '频率' in err_msg or 'frequency' in err_msg.lower():
                print(f"  Rate limit hit at {ym}, waiting 65s...")
                time.sleep(65)
                # Retry
                try:
                    df = pro.index_weight(index_code=index_code, trade_date=trade_date)
                    calls += 1
                    if len(df) > 0:
                        codes = set(df['con_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6])
                        data[ym] = codes
                        print(f"  [{i+1}/{len(months)}] {trade_date} ({ym}) -> {len(codes)} stocks (retry OK)")
                    else:
                        data[ym] = set()
                except Exception as e2:
                    print(f"  [{i+1}/{len(months)}] {trade_date} ({ym}) -> retry FAIL: {e2}")
                    data[ym] = set()
            else:
                print(f"  [{i+1}/{len(months)}] {trade_date} ({ym}) -> ERROR: {e}")
                data[ym] = set()

        # Rate limit: 200/min -> ~0.35s between calls
        if calls % 180 == 0 and calls > 0:
            print(f"  ... {calls} calls, pausing 10s for rate limit ...")
            time.sleep(10)
        else:
            time.sleep(0.35)

    all_data[index_name] = data

    # Count non-empty months
    non_empty = sum(1 for s in data.values() if len(s) > 0)
    print(f"  {index_name}: {non_empty}/{len(months)} months have data")

# Forward-fill gaps for each index
print(f"\n{'='*60}")
print("Forward-filling gaps...")
print(f"{'='*60}")

for index_name in ['CSI300', 'CSI500']:
    data = all_data[index_name]
    last_codes = set()
    filled = 0

    for ym, trade_date in months:
        if ym in data and len(data[ym]) > 0:
            last_codes = data[ym]
        elif last_codes:
            data[ym] = last_codes.copy()
            filled += 1

    print(f"  {index_name}: {filled} gaps forward-filled")
    all_data[index_name] = data

# Build combined lookup
print(f"\n{'='*60}")
print("Building combined CSI300+CSI500 lookup...")
print(f"{'='*60}")

lookup = {}
stats = {}

for ym, trade_date in months:
    c300 = all_data['CSI300'].get(ym, set())
    c500 = all_data['CSI500'].get(ym, set())
    union = c300 | c500

    for sym in union:
        lookup[(ym, sym)] = True

    stats[ym] = {
        'CSI300': len(c300),
        'CSI500': len(c500),
        'union': len(union)
    }

print(f"  Total lookup entries: {len(lookup)}")

# Save
with open(CACHE_PATH, 'wb') as f:
    pickle.dump({
        'lookup': lookup,
        'stats': stats,
        'indices': ['CSI300', 'CSI500']
    }, f)

print(f"\nCache saved: {CACHE_PATH}")
print(f"Size: {os.path.getsize(CACHE_PATH) / 1024 / 1024:.1f} MB")

# Print summary
print(f"\nSummary (key months):")
for ym, trade_date in months:
    if ym in stats:
        s = stats[ym]
        # Print first 5, every 10th, and last 5
        idx = months.index((ym, trade_date))
        if idx < 5 or idx > len(months) - 6 or idx % 12 == 0:
            print(f"  {ym}: CSI300={s['CSI300']}, CSI500={s['CSI500']}, union={s['union']}")
