#!/usr/bin/env python3
"""
Phase 1: Validate B+C+D stock pool with current best params (Explosive_def2.0+None).

Compares:
  OLD: absolute 500-1000亿, no liquidity filter
  NEW: percentile-based (P30-P70 normal), OAMV regime adjustment, + liquidity filter

Runs on Test 2026 only.
"""

import os, sys, time, copy
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from ML_optimization.mktcap_utils import (
    build_mktcap_lookup, get_mktcap_universe, is_in_mktcap_range,
    compute_mktcap_percentiles, get_pct_universe, get_oamv_regime_range,
    is_in_pct_range, check_liquidity,
    OAMV_REGIME_PCT, DEFAULT_PCT_LOWER, DEFAULT_PCT_UPPER,
    LIQUIDITY_MIN_AMOUNT, LIQUIDITY_MIN_TURNOVER,
)
import ML_optimization.phase2c_oamv_grid_search as p2c_mod
from ML_optimization.phase2c_bull_grid_search import (
    build_param_sets, _prepare_oamv_and_b2_params,
    _get_stock_files_mainboard, preload_stock_data,
    StopOptimEngine, STOP_PROFILES,
)

# ── Config ──────────────────────────────────────────────────────────
TEST_START = pd.Timestamp('2026-01-01')
TEST_END   = pd.Timestamp('2026-05-07')
BEST_SPRING_NAME = 'Explosive_def2.0'

OUT_DIR = os.path.join(BASE, 'output')
FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')

# Old absolute filter
OLD_MIN, OLD_MAX = 500, 1000

print("=" * 70)
print("  B+C+D Stock Pool Validation — Phase 1")
print(f"  Params: {BEST_SPRING_NAME} + None stop")
print(f"  Period: {TEST_START.date()} ~ {TEST_END.date()}")
print("=" * 70)

# ── 1. Load OAMV ────────────────────────────────────────────────────
print("\n[1/5] Loading OAMV...")
t0 = time.time()
oamv_df = fetch_market_data(start='20100101')
oamv_df = calc_oamv(oamv_df)
oamv_df = gen_oamv(oamv_df)
print(f"  OAMV: {len(oamv_df)} days ({time.time()-t0:.0f}s)")

# ── 2. Build market cap data ─────────────────────────────────────────
print("\n[2/5] Building market cap filters...")
t0 = time.time()

# Force rebuild for clean test
import pickle
for cache_path in [os.path.join(BASE, 'output', 'mktcap_lookup.pkl'),
                   os.path.join(BASE, 'output', 'mktcap_percentiles.pkl')]:
    if os.path.exists(cache_path):
        os.remove(cache_path)

mktcap_lookup = build_mktcap_lookup()
percentiles = compute_mktcap_percentiles()
print(f"  Lookup: {len(mktcap_lookup)} entries, {len(percentiles)} months ({time.time()-t0:.0f}s)")

# Show what the old/new ranges look like for test period
print(f"\n  Market cap ranges for Test 2026 period:")
for ym in ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05']:
    if ym in percentiles:
        lo, hi = get_oamv_regime_range(ym, 'normal', percentiles)
        lo_a, hi_a = get_oamv_regime_range(ym, 'aggressive', percentiles)
        lo_d, hi_d = get_oamv_regime_range(ym, 'defensive', percentiles)
        print(f"    {ym}: normal={lo:.0f}-{hi:.0f}  aggressive={lo_a:.0f}-{hi_a:.0f}  defensive={lo_d:.0f}-{hi_d:.0f}  (old fixed: {OLD_MIN}-{OLD_MAX})")

# ── 3. Build stock universes ─────────────────────────────────────────
print("\n[3/5] Building stock universes...")
t0 = time.time()

# OLD universe: absolute 500-1000亿
ym_s, ym_e = '2026-01', '2026-05'
old_codes = get_mktcap_universe(mktcap_lookup, ym_s, OLD_MIN, OLD_MAX, ym_end=ym_e)
print(f"  OLD (absolute {OLD_MIN}-{OLD_MAX}亿): {len(old_codes)} stocks")

# NEW universe: widest percentile range (aggressive P20-P80) for pre-filtering
# + per-signal regime-aware filtering happens in the engine
new_codes = get_pct_universe(mktcap_lookup, percentiles, ym_s, ym_end=ym_e, regime='aggressive')
print(f"  NEW (pct aggressive P20-P80):       {len(new_codes)} stocks")

# Overlap
overlap = old_codes & new_codes
print(f"  Overlap: {len(overlap)} stocks")
print(f"  OLD only: {len(old_codes - new_codes)}, NEW only: {len(new_codes - old_codes)}")

# ── 4. Load stock data ───────────────────────────────────────────────
print(f"\n[4/5] Loading stock data... ({time.time()-t0:.0f}s prep)")
t0 = time.time()

# Load for OLD universe
old_stock_data = {}
all_files = _get_stock_files_mainboard()
for f in all_files:
    code = os.path.basename(f).replace('.parquet', '')
    if code not in old_codes:
        continue
    try:
        df = pd.read_parquet(f)
        from ML_optimization.phase2c_bull_grid_search import NEEDED_COLS_S2
        cols_avail = [c for c in NEEDED_COLS_S2 if c in df.columns]
        df = df[cols_avail].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        if len(df) >= 120:
            old_stock_data[code] = df
    except Exception:
        continue
print(f"  OLD stock data: {len(old_stock_data)} stocks loaded ({time.time()-t0:.0f}s)")

t0 = time.time()
# Load for NEW universe
new_stock_data = {}
for f in all_files:
    code = os.path.basename(f).replace('.parquet', '')
    if code not in new_codes:
        continue
    try:
        df = pd.read_parquet(f)
        from ML_optimization.phase2c_bull_grid_search import NEEDED_COLS_S2
        cols_avail = [c for c in NEEDED_COLS_S2 if c in df.columns]
        df = df[cols_avail].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        if len(df) >= 120:
            new_stock_data[code] = df
    except Exception:
        continue
print(f"  NEW stock data: {len(new_stock_data)} stocks loaded ({time.time()-t0:.0f}s)")

# ── 5. Get best params ──────────────────────────────────────────────
param_sets = build_param_sets()
best_ps = None
for ps in param_sets:
    if ps['name'] == BEST_SPRING_NAME:
        best_ps = ps
        break
if best_ps is None:
    raise ValueError(f"Param set '{BEST_SPRING_NAME}' not found")

b2_params, oamv_params = _prepare_oamv_and_b2_params(best_ps)

# ── 6. Run backtests ─────────────────────────────────────────────────
print(f"\n[5/5] Running backtests...")
print("=" * 70)

p2c_mod.SIM_START = TEST_START
p2c_mod.SIM_END = TEST_END

# --- OLD: absolute 500-1000亿, no liquidity ---
print("\n  >>> OLD: absolute 500-1000亿, no liquidity...")
t0 = time.time()
engine_old = p2c_mod.OAMVSimEngine(old_stock_data, b2_params, oamv_df, oamv_params)
metrics_old = engine_old.run()
print(f"  Return={metrics_old['total_return_pct']:+.1f}%  "
      f"Sharpe={metrics_old['sharpe']:.3f}  WR={metrics_old['win_rate']:.1%}  "
      f"MaxDD={metrics_old['max_dd']:.1%}  Trades={metrics_old['n_trades']}  "
      f"Signals={metrics_old['n_signals']}  ({time.time()-t0:.0f}s)")

# --- NEW: percentile + OAMV regime + liquidity ---
print("\n  >>> NEW: percentile (regime-aware) + liquidity...")
t0 = time.time()
engine_new = p2c_mod.OAMVSimEngine(new_stock_data, b2_params, oamv_df, oamv_params,
                                   mktcap_lookup=mktcap_lookup,
                                   percentiles=percentiles,
                                   enable_liquidity=True)
metrics_new = engine_new.run()
fs = metrics_new.get('filter_stats', {})
print(f"  Return={metrics_new['total_return_pct']:+.1f}%  "
      f"Sharpe={metrics_new['sharpe']:.3f}  WR={metrics_new['win_rate']:.1%}  "
      f"MaxDD={metrics_new['max_dd']:.1%}  Trades={metrics_new['n_trades']}  "
      f"Signals={metrics_new['n_signals']}  ({time.time()-t0:.0f}s)")
if fs:
    print(f"  Filter stats: total={fs.get('total_signals',0)}  "
          f"pct_pass={fs.get('pct_pass',0)} pct_fail={fs.get('pct_fail',0)}  "
          f"liq_pass={fs.get('liq_pass',0)} liq_fail={fs.get('liq_fail',0)}")

# --- NEW: percentile only (no liquidity) ---
print("\n  >>> NEW: percentile only (no liquidity)...")
t0 = time.time()
engine_pct = p2c_mod.OAMVSimEngine(new_stock_data, b2_params, oamv_df, oamv_params,
                                   mktcap_lookup=mktcap_lookup,
                                   percentiles=percentiles,
                                   enable_liquidity=False)
metrics_pct = engine_pct.run()
fs2 = metrics_pct.get('filter_stats', {})
print(f"  Return={metrics_pct['total_return_pct']:+.1f}%  "
      f"Sharpe={metrics_pct['sharpe']:.3f}  WR={metrics_pct['win_rate']:.1%}  "
      f"MaxDD={metrics_pct['max_dd']:.1%}  Trades={metrics_pct['n_trades']}  "
      f"Signals={metrics_pct['n_signals']}  ({time.time()-t0:.0f}s)")
if fs2:
    print(f"  Filter stats: total={fs2.get('total_signals',0)}  "
          f"pct_pass={fs2.get('pct_pass',0)} pct_fail={fs2.get('pct_fail',0)}")

# ── 7. Comparison ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Comparison: OLD vs NEW vs PCT-ONLY")
print(f"{'='*70}")
print(f"  {'Metric':<20} {'OLD (500-1000亿)':>20} {'PCT-only':>20} {'NEW (PCT+Liq)':>20}")
print(f"  {'-'*70}")
for label, key in [
    ('Return', 'total_return_pct'),
    ('Sharpe', 'sharpe'),
    ('Win Rate', 'win_rate'),
    ('Max DD', 'max_dd'),
    ('Trades', 'n_trades'),
    ('Signals', 'n_signals'),
]:
    def fmt_val(v, k):
        if 'return' in k.lower():
            return f'{v:>+20.1f}%'
        elif 'sharpe' in k.lower():
            return f'{v:>20.3f}'
        elif 'rate' in k.lower() or 'dd' in k.lower():
            return f'{v:>20.1%}'
        else:
            return f'{v:>20d}'
    print(f"  {label:<20} {fmt_val(metrics_old.get(key,0), key)} {fmt_val(metrics_pct.get(key,0), key)} {fmt_val(metrics_new.get(key,0), key)}")

# Universe sizes
print(f"\n  Universe sizes:")
print(f"    OLD: {len(old_codes)} eligible → {len(old_stock_data)} loaded")
print(f"    NEW: {len(new_codes)} eligible → {len(new_stock_data)} loaded")

# ── 8. Save trade logs ───────────────────────────────────────────────
print(f"\n[6/5] Saving reports...")

def save_trades(engine, label):
    rows = []
    for t in engine.trade_log:
        row = {
            'action': t['action'],
            'date': t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date']),
            'code': t['code'],
            'price': round(t['price'], 2),
            'shares': t['shares'],
        }
        if t['action'] == 'BUY':
            row['cost'] = round(t.get('cost', 0), 2)
            row['weight'] = t.get('weight', 0)
            row['regime'] = t.get('regime', '')
            row['pnl'] = ''
            row['pnl_pct'] = ''
            row['reason'] = ''
        else:
            row['proceeds'] = round(t.get('proceeds', 0), 2)
            row['pnl'] = round(t.get('pnl', 0), 2)
            row['pnl_pct'] = round(t.get('pnl_pct', 0), 2)
            row['reason'] = t.get('reason', '')
            row['cost'] = ''
            row['weight'] = ''
            row['regime'] = ''
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f'bcd_test_trades_{label}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  {label} trades: {len(df)} rows → {path}")
    return df

save_trades(engine_old, 'old')
save_trades(engine_new, 'new')
save_trades(engine_pct, 'pct')

print(f"\n{'='*70}")
print(f"  Phase 1 B+C+D validation complete.")
print(f"{'='*70}")
