"""
Generate daily NAV, daily positions, and trade report for the best
Explosive_def2.0+None config on Test 2026.
"""
import os, sys, json, copy
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
import ML_optimization.phase2c_oamv_grid_search as p2c_mod
from ML_optimization.phase2c_mainboard import build_param_sets
from ML_optimization.phase2c_bull_grid_search import (
    _prepare_oamv_and_b2_params, _get_stock_files_mainboard, preload_stock_data,
)
from ML_optimization.mktcap_utils import build_mktcap_lookup, get_mktcap_universe

# ── Config ──────────────────────────────────────────────────────────
TEST_START = pd.Timestamp('2026-01-01')
TEST_END   = pd.Timestamp('2026-05-07')
MktCap_MIN = 500
MktCap_MAX = 1000
BEST_SPRING_NAME = 'Explosive_def2.0'

OUT_DIR = os.path.join(BASE, 'output')

print("=" * 60)
print("  Generating Reports: Explosive_def2.0+None on Test 2026")
print("=" * 60)

# ── Load OAMV ───────────────────────────────────────────────────────
print("\n[1/5] Loading OAMV data...")
oamv_df = fetch_market_data(start='20100101')
oamv_df = calc_oamv(oamv_df)
oamv_df = gen_oamv(oamv_df)
print(f"  OAMV: {len(oamv_df)} days")

# ── Find best B2 params ─────────────────────────────────────────────
param_sets = build_param_sets()
best_ps = None
for ps in param_sets:
    if ps['name'] == BEST_SPRING_NAME:
        best_ps = ps
        break
if best_ps is None:
    raise ValueError(f"Param set '{BEST_SPRING_NAME}' not found")

b2_params, oamv_params = _prepare_oamv_and_b2_params(best_ps)
print(f"  B2: {BEST_SPRING_NAME}")

# ── Load eligible stocks (SAME as grid search: union across ALL periods) ─
print("\n[2/5] Loading eligible stocks...")
mktcap_lookup = build_mktcap_lookup()

# Replicate the exact union logic from phase2c_bull_grid_search.py
TRAIN_BULL_1 = (pd.Timestamp('2014-07-01'), pd.Timestamp('2015-06-12'))
TRAIN_BULL_2 = (pd.Timestamp('2020-03-23'), pd.Timestamp('2021-12-13'))
TRAIN_BULL_3 = (pd.Timestamp('2024-01-01'), pd.Timestamp('2025-06-30'))
TRAIN_PERIODS = [TRAIN_BULL_1, TRAIN_BULL_2, TRAIN_BULL_3]

eligible_codes = set()
for s, e in TRAIN_PERIODS:
    ym_s = s.strftime('%Y-%m')
    ym_e = e.strftime('%Y-%m')
    codes = get_mktcap_universe(mktcap_lookup, ym_s, MktCap_MIN, MktCap_MAX, ym_end=ym_e)
    eligible_codes.update(codes)
codes_val = get_mktcap_universe(mktcap_lookup, '2025-07', MktCap_MIN, MktCap_MAX, ym_end='2026-05')
eligible_codes.update(codes_val)
print(f"  Eligible (union across all periods): {len(eligible_codes)} stocks (500-1000亿)")

# Use preload_stock_data from grid search for exact same loading
stock_data = preload_stock_data(eligible_codes)
print(f"  Loaded {len(stock_data)} stocks")

# ── Run backtest ────────────────────────────────────────────────────
print("\n[3/5] Running backtest on Test 2026...")
p2c_mod.SIM_START = TEST_START
p2c_mod.SIM_END = TEST_END

engine = p2c_mod.OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params)
metrics = engine.run()

print(f"  Return: {metrics['total_return_pct']:+.1f}%")
print(f"  Sharpe: {metrics['sharpe']:.3f}")
print(f"  Win Rate: {metrics['win_rate']:.1%}")
print(f"  Max DD: {metrics['max_dd']:.1%}")
print(f"  Trades: {metrics['n_trades']}")

# ── Build reports ───────────────────────────────────────────────────
print("\n[5/5] Building report CSVs...")

# --- Daily NAV ---
nav_rows = []
for dv in engine.daily_values:
    nav_rows.append({
        'date': dv['date'].strftime('%Y-%m-%d'),
        'nav_total': round(dv['value'], 2),
        'nav_normalized': round(dv['value'] / engine.initial_capital, 6),
        'cash': round(dv['cash'], 2),
        'positions_count': dv['positions'],
        'regime': dv.get('regime', 'normal'),
    })
nav_df = pd.DataFrame(nav_rows)
nav_path = os.path.join(OUT_DIR, 'report_daily_nav.csv')
nav_df.to_csv(nav_path, index=False, encoding='utf-8-sig')
print(f"  Daily NAV: {len(nav_df)} rows → {nav_path}")

# --- Daily Positions (reconstruct from trade_log) ---
running_positions = {}
sorted_trades = sorted(engine.trade_log, key=lambda t: t['date'])
trade_idx = 0
all_dates = sorted(dv['date'] for dv in engine.daily_values)

pos_rows = []
for date in all_dates:
    while trade_idx < len(sorted_trades) and sorted_trades[trade_idx]['date'] <= date:
        t = sorted_trades[trade_idx]
        code = t['code']
        if t['action'] == 'BUY':
            running_positions[code] = {
                'buy_date': t['date'],
                'buy_price': t['price'],
                'shares': t['shares'],
                'cost': t['cost'],
                'weight': t.get('weight', 0),
            }
        elif t['action'] == 'SELL':
            if code in running_positions:
                sold_shares = t['shares']
                pos = running_positions[code]
                pos['shares'] -= sold_shares
                if pos['shares'] < 100:
                    del running_positions[code]
        trade_idx += 1

    if running_positions:
        for code, pos in running_positions.items():
            df_s = stock_data.get(code)
            current_price = pos['buy_price']
            if df_s is not None:
                if date in df_s.index:
                    current_price = float(df_s.loc[date, 'close'])
                else:
                    nearby = df_s.index[df_s.index >= date]
                    if len(nearby) > 0:
                        current_price = float(df_s.loc[nearby[0], 'close'])

            market_value = pos['shares'] * current_price
            pos_rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'code': code,
                'shares': pos['shares'],
                'buy_price': round(pos['buy_price'], 2),
                'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                'current_price': round(current_price, 2),
                'market_value': round(market_value, 2),
                'unrealized_pnl': round(market_value - pos['cost'], 2),
                'unrealized_pnl_pct': round((market_value / pos['cost'] - 1) * 100, 2) if pos['cost'] > 0 else 0,
                'weight': pos['weight'],
            })
    else:
        pos_rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'code': '', 'shares': 0, 'buy_price': 0, 'buy_date': '',
            'current_price': 0, 'market_value': 0, 'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0, 'weight': 0,
        })

pos_df = pd.DataFrame(pos_rows)
pos_df = pos_df.sort_values(['date', 'code']).reset_index(drop=True)
pos_path = os.path.join(OUT_DIR, 'report_daily_positions.csv')
pos_df.to_csv(pos_path, index=False, encoding='utf-8-sig')
print(f"  Daily Positions: {len(pos_df)} rows → {pos_path}")

# --- Trade Report ---
trade_rows = []
for t in engine.trade_log:
    row = {
        'action': t['action'],
        'date': t['date'].strftime('%Y-%m-%d'),
        'code': t['code'],
        'price': round(t['price'], 2),
        'shares': t['shares'],
    }
    if t['action'] == 'BUY':
        row['cost'] = round(t['cost'], 2)
        row['weight'] = t.get('weight', 0)
        row['regime'] = t.get('regime', 'normal')
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
    trade_rows.append(row)

trade_df = pd.DataFrame(trade_rows)
trade_path = os.path.join(OUT_DIR, 'report_trades.csv')
trade_df.to_csv(trade_path, index=False, encoding='utf-8-sig')
print(f"  Trade Report: {len(trade_df)} rows → {trade_path}")

# --- Closed Positions Summary ---
closed_rows = []
for p in engine.closed_positions:
    sell_date = p['sell_date']
    if hasattr(sell_date, 'strftime'):
        sell_date = sell_date.strftime('%Y-%m-%d')
    buy_date_str = ''
    if 'buy_date' in p and hasattr(p['buy_date'], 'strftime'):
        buy_date_str = p['buy_date'].strftime('%Y-%m-%d')
    closed_rows.append({
        'code': p['code'],
        'buy_price': round(p['buy_price'], 2),
        'sell_date': sell_date,
        'sell_price': round(p['sell_price'], 2),
        'shares': p['shares'],
        'pnl': round(p['pnl'], 2),
        'pnl_pct': round(p['pnl_pct'], 2),
        'reason': p.get('reason', ''),
        'weight': p.get('weight', 0),
    })
closed_df = pd.DataFrame(closed_rows)
closed_path = os.path.join(OUT_DIR, 'report_closed_positions.csv')
closed_df.to_csv(closed_path, index=False, encoding='utf-8-sig')
print(f"  Closed Positions: {len(closed_df)} rows → {closed_path}")

# ── Upload to GitHub ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Uploading to GitHub...")

# Git operations
os.chdir(BASE)

# Check which files need to be added
report_files = [
    'output/report_daily_nav.csv',
    'output/report_daily_positions.csv',
    'output/report_trades.csv',
    'output/report_closed_positions.csv',
]

for rf in report_files:
    if os.path.exists(os.path.join(BASE, rf)):
        print(f"  Found: {rf} ({os.path.getsize(os.path.join(BASE, rf)):,} bytes)")

# Use git to add and commit
import subprocess
def run_git(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=BASE)
    if result.returncode != 0 and result.stderr:
        print(f"  Git warning: {result.stderr.strip()}")
    return result.stdout.strip()

# Check if files are tracked
status = run_git('git status --short output/report_*.csv')
print(f"  Git status: {status if status else 'no changes'}")

if status:
    run_git('git add output/report_daily_nav.csv output/report_daily_positions.csv output/report_trades.csv output/report_closed_positions.csv')
    commit_msg = f'[auto] Explosive_def2.0+None Test2026 reports — Ret+{metrics["total_return_pct"]:.1f}% Sharpe{metrics["sharpe"]:.3f}'
    run_git(f'git commit -m "{commit_msg}"')
    run_git('git push origin main')
    print(f"  Pushed to GitHub: royhe1001-web/lets-gooooo")
else:
    print(f"  No new changes to push")

print(f"\n{'='*60}")
print(f"  Done! Reports at:")
for rf in report_files:
    print(f"    https://github.com/royhe1001-web/lets-gooooo/blob/main/{rf}")
print(f"{'='*60}")
