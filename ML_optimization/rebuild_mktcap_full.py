#!/usr/bin/env python3
"""Rebuild full monthly market cap data: merge 2008-2017 new data with existing 2017-2026 data."""

import os, pickle
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_CSV = os.path.join(BASE, 'output', 'monthly_mktcap_early.csv')
OLD_CSV = os.path.join(BASE, 'output', 'monthly_mktcap.csv')
OUT_CSV = os.path.join(BASE, 'output', 'monthly_mktcap_full.csv')
CACHE_PATH = os.path.join(BASE, 'output', 'mktcap_lookup.pkl')

# Load existing data
print("Loading existing market cap data...")
old = pd.read_csv(OLD_CSV)
# Add ym column from trade_date
old['ym'] = pd.to_datetime(old['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m')
old = old[['ym', 'symbol', 'mv_yi']]
print(f"  Existing: {len(old)} records, {old['ym'].min()} -> {old['ym'].max()}")

# Load new early data
if os.path.exists(NEW_CSV):
    print("Loading early market cap data...")
    new = pd.read_csv(NEW_CSV)
    print(f"  Early: {len(new)} records, {new['ym'].min()} -> {new['ym'].max()}")
else:
    print("No early data CSV found — only using existing data")
    new = pd.DataFrame(columns=['ym', 'symbol', 'mv_yi'])

# Combine, deduplicate
combined = pd.concat([new, old], ignore_index=True)
combined = combined.drop_duplicates(subset=['ym', 'symbol'], keep='last')
combined = combined.sort_values(['ym', 'symbol']).reset_index(drop=True)

print(f"  Combined: {len(combined)} records, {combined['ym'].min()} -> {combined['ym'].max()}")

# Save full CSV
combined.to_csv(OUT_CSV, index=False)
print(f"Full CSV: {OUT_CSV}")

# Build pickle lookup
print("Building pickle lookup...")
lookup = {}
for _, row in combined.iterrows():
    lookup[(row['ym'], str(row['symbol']).zfill(6))] = row['mv_yi']

with open(CACHE_PATH, 'wb') as f:
    pickle.dump(lookup, f)

print(f"Pickle: {CACHE_PATH} ({len(lookup)} entries, {os.path.getsize(CACHE_PATH)/1024/1024:.1f} MB)")

# Quick stats on coverage
print(f"\nCoverage check:")
for y in [2008, 2010, 2013, 2017, 2020, 2026]:
    ym = f'{y}-06'
    count = sum(1 for (m, s), v in lookup.items() if m == ym and v >= 100)
    print(f"  {ym}: {count} stocks with mv >= 100亿")
