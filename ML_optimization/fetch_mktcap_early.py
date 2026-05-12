#!/usr/bin/env python3
"""Fetch monthly market cap snapshots for 2008-2017 using single-date queries.

Each query targets the last trading day of each month (from trade_cal).
Merges with existing 2017-12+ data and rebuilds pickle lookup.
"""

import os, pickle, time
import pandas as pd
import tushare as ts

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(BASE, 'output', 'monthly_mktcap_full.csv')
CACHE_PATH = os.path.join(BASE, 'output', 'mktcap_lookup.pkl')
OLD_CSV = os.path.join(BASE, 'output', 'monthly_mktcap.csv')

pro = ts.pro_api()

# Step 1: Get trading calendar for Shanghai exchange
print("Getting SHSE trading calendar...")
cal = pro.trade_cal(exchange='SSE', start_date='20080101', end_date='20171231')
cal = cal[cal['is_open'] == 1].copy()
cal['trade_date'] = pd.to_datetime(cal['cal_date'], format='%Y%m%d')
cal['ym'] = cal['trade_date'].dt.strftime('%Y-%m')

# Last trading day of each month
last_days = cal.groupby('ym')['cal_date'].max().reset_index()
last_days.columns = ['ym', 'last_trade_date']
print(f"  {len(last_days)} months from {last_days['ym'].iloc[0]} to {last_days['ym'].iloc[-1]}")

# Step 2: Fetch market cap for each month-end
print("\nFetching market cap (2008-2017)...")
all_data = []
n_months = len(last_days)

for i, (_, row) in enumerate(last_days.iterrows()):
    ym = row['ym']
    trade_date = row['last_trade_date']

    for attempt in range(3):
        try:
            df = pro.daily_basic(
                trade_date=trade_date,
                fields='ts_code,trade_date,total_mv'
            )
            if len(df) > 0:
                break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)

    if len(df) == 0:
        print(f"  [{i+1:3d}/{n_months}] {trade_date} ({ym}) -> 0 rows")
        continue

    df['symbol'] = df['ts_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6]
    df['mv_yi'] = pd.to_numeric(df['total_mv'], errors='coerce') / 1e8
    df['ym'] = ym

    all_data.append(df[['ym', 'symbol', 'mv_yi']])

    if (i+1) % 20 == 0:
        print(f"  [{i+1:3d}/{n_months}] {trade_date} ({ym}) -> {len(df)} stocks")

    time.sleep(0.15)

# Combine all early data
early = pd.concat(all_data, ignore_index=True)
print(f"\nEarly data: {len(early)} records, {early['symbol'].nunique()} unique stocks")
print(f"Range: {early['ym'].min()} -> {early['ym'].max()}")

# Step 3: Merge with existing data
print("\nMerging with existing 2017-12+ data...")
old = pd.read_csv(OLD_CSV)
old['ym'] = pd.to_datetime(old['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m')
old = old[['ym', 'symbol', 'mv_yi']].copy()
old['symbol'] = old['symbol'].astype(str).str.zfill(6)
print(f"  Existing: {len(old)} records, {old['ym'].min()} -> {old['ym'].max()}")

combined = pd.concat([early, old], ignore_index=True)
combined = combined.drop_duplicates(subset=['ym', 'symbol'], keep='last')
combined = combined.sort_values(['ym', 'symbol']).reset_index(drop=True)
print(f"  Combined: {len(combined)} records, {combined['ym'].min()} -> {combined['ym'].max()}")

# Save
combined.to_csv(OUT_CSV, index=False)
print(f"\nCSV saved: {OUT_CSV}")

# Step 4: Build pickle lookup
print("Building pickle lookup...")
lookup = {}
for _, row in combined.iterrows():
    lookup[(row['ym'], str(row['symbol']).zfill(6))] = float(row['mv_yi'])

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
with open(CACHE_PATH, 'wb') as f:
    pickle.dump(lookup, f)

print(f"Pickle: {CACHE_PATH} ({len(lookup)} entries, {os.path.getsize(CACHE_PATH)/1024/1024:.1f} MB)")

# Quick stats
print(f"\nCoverage check (300-1000亿 stocks per sample month):")
for y in [2010, 2013, 2017, 2020, 2024]:
    ym = f'{y}-06'
    in_range = sum(1 for (m, s), v in lookup.items() if m == ym and 300 <= v <= 1000)
    below = sum(1 for (m, s), v in lookup.items() if m == ym and v < 300)
    above = sum(1 for (m, s), v in lookup.items() if m == ym and v > 1000)
    total = in_range + below + above
    print(f"  {ym}: 300-1000亿={in_range}, <300亿={below}, >1000亿={above}, total={total}")
