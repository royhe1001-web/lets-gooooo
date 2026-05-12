#!/usr/bin/env python3
"""Extend monthly market cap data backward to 2008.

Fetches from tushare daily_basic (yearly batches), takes last trading
day of each month as monthly snapshot. Merges with existing data.
"""

import os, sys, pickle, time
import pandas as pd
import tushare as ts

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, 'output', 'mktcap_lookup.pkl')
CSV_PATH = os.path.join(BASE, 'output', 'monthly_mktcap.csv')

pro = ts.pro_api()

YEARS = range(2008, 2018)  # 2008-2017

print(f"Fetching daily_basic for years {YEARS[0]}-{YEARS[-1]}...")

all_frames = []

for year in YEARS:
    start = f'{year}0101'
    end = f'{year}1231'

    for attempt in range(3):
        try:
            df = pro.daily_basic(
                start_date=start, end_date=end,
                fields='ts_code,trade_date,total_mv'
            )
            if len(df) > 0:
                break
        except Exception as e:
            print(f"  {year} attempt {attempt+1}/3: {e}")
            time.sleep(5)

    if len(df) == 0:
        print(f"  {year}: 0 rows — SKIP")
        continue

    # Parse
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df['symbol'] = df['ts_code'].str.replace('.SZ', '').str.replace('.SH', '').str[:6]
    df['mv_yi'] = pd.to_numeric(df['total_mv'], errors='coerce') / 1e8  # 元→亿

    # Keep only the last trading day of each month
    df['ym'] = df['trade_date'].dt.strftime('%Y-%m')
    last_day = df.groupby(['ym', 'symbol'])['trade_date'].transform('max')
    df = df[df['trade_date'] == last_day].copy()
    df = df.drop_duplicates(subset=['ym', 'symbol'], keep='last')

    print(f"  {year}: {len(df)} monthly records, {df['symbol'].nunique()} stocks")
    all_frames.append(df[['ym', 'symbol', 'mv_yi']])

    time.sleep(0.5)

# Combine
new_data = pd.concat(all_frames, ignore_index=True)
print(f"\nTotal new records: {len(new_data)}")
print(f"Date range: {new_data['ym'].min()} -> {new_data['ym'].max()}")

# Merge with existing data if available
if os.path.exists(CSV_PATH):
    existing = pd.read_csv(CSV_PATH)
    print(f"Existing data: {len(existing)} records, {existing['ym'].min()} -> {existing['ym'].max()}")
    combined = pd.concat([new_data, existing], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ym', 'symbol'], keep='last')
    combined = combined.sort_values(['ym', 'symbol'])
    print(f"Combined: {len(combined)} records")
else:
    combined = new_data

# Save CSV
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
combined.to_csv(CSV_PATH, index=False)
print(f"CSV saved: {CSV_PATH} ({len(combined)} rows)")

# Build pickle lookup: (ym, symbol) -> mv_yi
print("\nBuilding pickle lookup...")
lookup = {}
for _, row in combined.iterrows():
    lookup[(row['ym'], row['symbol'])] = row['mv_yi']

with open(CACHE_PATH, 'wb') as f:
    pickle.dump(lookup, f)

print(f"Lookup saved: {CACHE_PATH} ({len(lookup)} entries, {os.path.getsize(CACHE_PATH)/1024/1024:.1f} MB)")
