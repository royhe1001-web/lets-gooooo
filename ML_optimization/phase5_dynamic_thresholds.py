#!/usr/bin/env python3
"""
Phase 5: Dynamic Threshold Optimization (per OAMV Regime)
==========================================================
Optimizes entry thresholds (gain_min, j_today_max, shadow_max) separately
for each OAMV regime: aggressive / normal / defensive.

Training:  2024-01 ~ 2025-09
Validation: 2025-Q4
Test:       2026 (full simulation engine)

Key improvement over Phase 3:
  - T+5 returns weighted by b2_position_weight (fixes the flat-surface problem)
  - Per-regime thresholds instead of fixed global values
  - Expanded training period to prevent overfitting
"""

import os, sys, time, json, copy
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from strategy_b2 import generate_b2_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# Date splits
TRAIN_START = pd.Timestamp('2024-01-01')
TRAIN_END   = pd.Timestamp('2025-09-30')
VAL_START   = pd.Timestamp('2025-10-01')
VAL_END     = pd.Timestamp('2025-12-31')
TEST_START  = pd.Timestamp('2026-01-01')
TEST_END    = pd.Timestamp('2026-05-07')

# Baseline params (Phase 3 optimal, fixed thresholds)
BASE_PARAMS = {
    'j_prev_max': 20, 'gain_min': 0.054, 'j_today_max': 59,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.09, 'weight_gain_2x': 2.5,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'oamv_aggressive_buffer': 0.95, 'oamv_defensive_buffer': 0.99,
    'oamv_defensive_ban_entry': True,
    'oamv_aggressive_threshold': 3.0, 'oamv_defensive_threshold': -2.0,
}

# Per-regime profile candidates
# Each regime has 3 profiles: loose (easy entry), medium (baseline), tight (strict)
AGGRESSIVE_PROFILES = {
    'agg_loose':  {'gain_min_aggressive': 0.03, 'j_today_max_aggressive': 70, 'shadow_max_aggressive': 0.050},
    'agg_medium': {'gain_min_aggressive': 0.04, 'j_today_max_aggressive': 65, 'shadow_max_aggressive': 0.040},
    'agg_tight':  {'gain_min_aggressive': 0.05, 'j_today_max_aggressive': 60, 'shadow_max_aggressive': 0.035},
}

NORMAL_PROFILES = {
    'norm_loose':  {'gain_min_normal': 0.04, 'j_today_max_normal': 65, 'shadow_max_normal': 0.040},
    'norm_medium': {'gain_min_normal': 0.054, 'j_today_max_normal': 59, 'shadow_max_normal': 0.035},
    'norm_tight':  {'gain_min_normal': 0.06, 'j_today_max_normal': 55, 'shadow_max_normal': 0.030},
}

DEFENSIVE_PROFILES = {
    'def_loose':  {'gain_min_defensive': 0.05, 'j_today_max_defensive': 60, 'shadow_max_defensive': 0.035},
    'def_medium': {'gain_min_defensive': 0.06, 'j_today_max_defensive': 55, 'shadow_max_defensive': 0.025},
    'def_tight':  {'gain_min_defensive': 0.07, 'j_today_max_defensive': 50, 'shadow_max_defensive': 0.020},
}


def preload_data(n_stocks=None, mktcap_min=None, mktcap_max=None,
                 oamv_change_map=None, eligible_codes=None):
    """Preload stocks. Returns stock_data dict.

    If mktcap_min/max are set, filters stocks to those ever in the market cap
    range during 2010-2026 (dynamic, time-aware filtering)."""

    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])

    if oamv_change_map is None or eligible_codes is None:
        from ML_optimization.mktcap_utils import load_mktcap, build_mktcap_lookup

        mktcap_lookup = {}
        if mktcap_min is not None and mktcap_max is not None:
            try:
                mktcap_lookup = build_mktcap_lookup(load_mktcap())
                print(f"  市值过滤: {mktcap_min}-{mktcap_max}亿, "
                      f"{load_mktcap()['ym'].nunique()}个月快照", flush=True)
            except Exception as e:
                print(f"  市值数据加载失败: {e}", flush=True)

        oamv_df = fetch_market_data(start='20100101')
        oamv_df = calc_oamv(oamv_df)
        oamv_df = gen_oamv(oamv_df)

        oamv_change_map = {}
        for _, row in oamv_df.iterrows():
            d = row['date']
            if hasattr(d, 'date'):
                d = d.date()
            oamv_change_map[pd.Timestamp(d)] = float(row['oamv_change_pct'])

        if mktcap_lookup:
            eligible_codes = set()
            ym_start = '2010-01'; ym_end = '2026-05'
            for (ym, sym), mv in mktcap_lookup.items():
                if ym_start <= ym <= ym_end and mktcap_min <= mv <= mktcap_max:
                    eligible_codes.add(sym)
            print(f"  市值范围内股票: {len(eligible_codes)} 只 (2010-2026)", flush=True)
        else:
            eligible_codes = None

    needed_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                   'white_line', 'yellow_line',
                   'brick_val', 'brick_rising', 'brick_falling',
                   'brick_green_to_red', 'brick_red_to_green',
                   'brick_green_streak', 'brick_red_streak',
                   'cross_below_yellow']

    stock_data = {}
    t_start = time.time()
    total_checked = 0
    total_skipped = 0
    for i, f in enumerate(feat_files):
        code = f.replace('.parquet', '')
        total_checked += 1

        # Market cap pre-filter: quick set lookup
        if eligible_codes is not None and code not in eligible_codes:
            total_skipped += 1
            continue

        try:
            df = pd.read_parquet(os.path.join(FEAT_DIR, f))
            cols_avail = [c for c in needed_cols if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])

            oamv_changes = [oamv_change_map.get(d, 0.0) for d in df['date']]
            df['_oamv_change_pct'] = oamv_changes

            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df

            if len(stock_data) % 25 == 0:
                elapsed = time.time() - t_start
                print(f"    [{len(stock_data)} loaded, {total_checked} checked, "
                      f"{total_skipped} skipped] {elapsed:.0f}s", flush=True)
        except Exception as e:
            if len(stock_data) < 5:
                print(f"    ERROR loading {code}: {e}", flush=True)
            continue

    print(f"    Done: {len(stock_data)} stocks, {total_checked} checked, "
          f"{total_skipped} skipped, {time.time()-t_start:.0f}s", flush=True)

    if n_stocks and len(stock_data) > n_stocks:
        # Trim to n_stocks if specified
        codes = sorted(stock_data.keys())[:n_stocks]
        stock_data = {c: stock_data[c] for c in codes}

    return stock_data


def eval_params_weighted(stock_data, params, sim_start, sim_end):
    """Evaluate params on date range, returning b2_position_weight-weighted T+5 return.

    Weights each signal by its b2_position_weight so that high-conviction
    signals contribute more to the metric (fixes Phase 3's flat-weight-surface problem).
    """
    agg_thresh = params.get('oamv_aggressive_threshold', 3.0)
    def_thresh = params.get('oamv_defensive_threshold', -2.0)

    b2_params = {k: v for k, v in params.items()
                 if k not in ('oamv_aggressive_threshold', 'oamv_defensive_threshold')}

    total_weighted_ret = 0.0
    total_weight = 0.0
    signal_count = 0

    for code, df in stock_data.items():
        try:
            df_copy = df.copy()
            chg = df_copy['_oamv_change_pct'].values
            regime = np.where(chg > agg_thresh, 'aggressive',
                     np.where(chg < def_thresh, 'defensive', 'normal'))
            df_copy['oamv_regime'] = regime

            sdf = generate_b2_signals(df_copy, board_type='main', precomputed=True, params=b2_params)
            mask = (sdf.index >= sim_start) & (sdf.index <= sim_end)
            sig_idx = sdf.index[mask & (sdf['b2_entry_signal'] == 1)]

            for i in sig_idx:
                pos = sdf.index.get_loc(i)
                if pos + 5 < len(sdf):
                    buy_px = float(sdf.loc[i, 'close'])
                    sell_px = float(sdf.iloc[pos + 5]['open'])
                    ret = (sell_px / buy_px - 1)
                    w = float(sdf.loc[i, 'b2_position_weight'])
                    total_weighted_ret += ret * w
                    total_weight += w
                    signal_count += 1
        except Exception:
            continue

    if total_weight == 0:
        return 0.0, 0
    return float(total_weighted_ret / total_weight), signal_count


def build_dynamic_params(agg_profile, norm_profile, def_profile):
    """Build a full params dict combining per-regime profile choices with base params."""
    p = copy.deepcopy(BASE_PARAMS)
    # Enable dynamic thresholds
    p['dynamic_thresholds'] = True
    # Merge per-regime profile values
    p.update(agg_profile)
    p.update(norm_profile)
    p.update(def_profile)
    return p


def run_grid_search(stock_data):
    """Grid search over 3x3x3 = 27 regime profile combinations on training data."""
    print("=" * 70)
    print("  Phase 5: Dynamic Threshold Grid Search")
    print(f"  Training: {TRAIN_START.date()} -> {TRAIN_END.date()}")
    print(f"  Stocks: {len(stock_data)}")
    print("=" * 70)

    agg_items = list(AGGRESSIVE_PROFILES.items())
    norm_items = list(NORMAL_PROFILES.items())
    def_items = list(DEFENSIVE_PROFILES.items())

    # Baseline: fixed thresholds (no dynamic)
    print("\n  --- Baseline: Fixed thresholds ---")
    t0 = time.time()
    base_params_no_dynamic = copy.deepcopy(BASE_PARAMS)
    base_params_no_dynamic['dynamic_thresholds'] = False
    base_z, base_n = eval_params_weighted(stock_data, base_params_no_dynamic, TRAIN_START, TRAIN_END)
    print(f"  Fixed baseline: weighted_z={base_z*100:.3f}%, signals={base_n} ({time.time()-t0:.0f}s)")

    # Dynamic with all medium profiles (= should be similar to fixed if gain/j/shadow match)
    print("\n  --- Dynamic (all medium profiles) ---")
    t0 = time.time()
    all_medium = build_dynamic_params(
        AGGRESSIVE_PROFILES['agg_medium'],
        NORMAL_PROFILES['norm_medium'],
        DEFENSIVE_PROFILES['def_medium'])
    med_z, med_n = eval_params_weighted(stock_data, all_medium, TRAIN_START, TRAIN_END)
    print(f"  All-medium dynamic: weighted_z={med_z*100:.3f}%, signals={med_n} ({time.time()-t0:.0f}s)")

    # Full grid: 27 combinations
    print(f"\n  --- Grid Search: {len(agg_items)}x{len(norm_items)}x{len(def_items)} = {len(agg_items)*len(norm_items)*len(def_items)} combinations ---")
    results = []
    t_start = time.time()

    for idx, ((agg_name, agg_p), (norm_name, norm_p), (def_name, def_p)) in enumerate(
        product(agg_items, norm_items, def_items)):

        t1 = time.time()
        p = build_dynamic_params(agg_p, norm_p, def_p)
        z, n = eval_params_weighted(stock_data, p, TRAIN_START, TRAIN_END)
        elapsed = time.time() - t1

        name = f"{agg_name}+{norm_name}+{def_name}"
        results.append({
            'name': name, 'z': z, 'n_signals': n,
            'agg_profile': agg_name, 'norm_profile': norm_name, 'def_profile': def_name,
            'params': {k: v for k, v in p.items() if 'dynamic' in k or 'regime' in k or
                       k.startswith(('gain_min_', 'j_today_max_', 'shadow_max_'))},
            'elapsed': elapsed,
        })

        eta = (time.time() - t_start) / (idx + 1) * (27 - idx - 1)
        print(f"  [{idx+1:>2}/27] {name:<40s} z={z*100:+.3f}% n={n:>5d} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    results.sort(key=lambda r: r['z'], reverse=True)
    return results, base_z, base_n


def validate_top(results, stock_data, top_n=5):
    """Validate top-N on 2025 data."""
    print(f"\n{'='*70}")
    print(f"  Validation: Top-{top_n} on 2025")
    print(f"  Period: {VAL_START.date()} -> {VAL_END.date()}")
    print(f"{'='*70}")

    # Baseline on validation
    base_params_no_dynamic = copy.deepcopy(BASE_PARAMS)
    base_params_no_dynamic['dynamic_thresholds'] = False
    base_z_val, base_n_val = eval_params_weighted(stock_data, base_params_no_dynamic, VAL_START, VAL_END)
    print(f"  Fixed baseline: weighted_z={base_z_val*100:.3f}%, signals={base_n_val}")

    validated = []
    for r in results[:top_n]:
        p = build_dynamic_params(
            AGGRESSIVE_PROFILES[r['agg_profile']],
            NORMAL_PROFILES[r['norm_profile']],
            DEFENSIVE_PROFILES[r['def_profile']])
        z, n = eval_params_weighted(stock_data, p, VAL_START, VAL_END)
        z_delta = (z - base_z_val) / abs(base_z_val) * 100 if base_z_val != 0 else 0
        validated.append({**r, 'z_val': z, 'n_val': n, 'val_delta_pct': z_delta})
        tag = " BEST" if z == max(v['z_val'] for v in validated) else ""
        print(f"  {r['name']:<40s} z_val={z*100:+.3f}% (vs base {z_delta:+.1f}%) n={n}{tag}")

    validated.sort(key=lambda r: r['z_val'], reverse=True)
    return validated, base_z_val


def run_full_simulation_test(stock_data, params, name):
    """Run full OAMVSimEngine on 2026 test window."""
    import importlib.util
    p2c_path = os.path.join(BASE, 'ML_optimization', 'phase2c_oamv_grid_search.py')
    spec = importlib.util.spec_from_file_location('phase2c', p2c_path)
    p2c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2c)
    OAMVSimEngine = p2c.OAMVSimEngine

    # Rebuild OAMV data for sim engine
    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)

    oamv_params = {
        'oamv_aggressive_threshold': params.get('oamv_aggressive_threshold', 3.0),
        'oamv_defensive_threshold': params.get('oamv_defensive_threshold', -2.0),
        'oamv_aggressive_buffer': params.get('oamv_aggressive_buffer', 0.95),
        'oamv_defensive_buffer': params.get('oamv_defensive_buffer', 0.99),
        'oamv_defensive_ban_entry': params.get('oamv_defensive_ban_entry', True),
    }

    b2_params = {k: v for k, v in params.items()
                 if k not in ('oamv_aggressive_threshold', 'oamv_defensive_threshold')}

    engine = OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params)
    metrics = engine.run()

    print(f"  {name:<40s} Ret={metrics['total_return_pct']:+.1f}% "
          f"Sharpe={metrics['sharpe']:.3f} WR={metrics['win_rate']:.1%} "
          f"DD={metrics['max_dd']:.1%} Trades={metrics['n_trades']} "
          f"Sig={metrics['n_signals']} "
          f"A/D/N={metrics.get('aggressive_buys',0)}/{metrics.get('defensive_buys',0)}/{metrics.get('normal_buys',0)}")

    return metrics


def main():
    t0 = time.time()

    # 0. One-time setup: OAMV + mktcap 500-1000亿 filter
    print("=" * 70)
    print("  Phase 5: Setup (OAMV + MktCap 500-1000亿)")
    print("=" * 70)

    from ML_optimization.mktcap_utils import build_mktcap_lookup

    mktcap_lookup = build_mktcap_lookup()
    ym_start, ym_end = '2010-01', '2026-05'
    eligible_codes = set()
    for (ym, sym), mv in mktcap_lookup.items():
        if ym_start <= ym <= ym_end and 500 <= mv <= 1000:
            eligible_codes.add(sym)
    print(f"  市值500-1000亿范围内: {len(eligible_codes)} 只 ({ym_start}~{ym_end})", flush=True)

    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    oamv_change_map = {}
    for _, row in oamv_df.iterrows():
        d = row['date']
        if hasattr(d, 'date'):
            d = d.date()
        oamv_change_map[pd.Timestamp(d)] = float(row['oamv_change_pct'])
    print(f"  OAMV: {len(oamv_df)} days", flush=True)

    # 1. Preload training stock data
    print(f"\n  Phase 5: Preloading training data...")
    stock_data = preload_data(oamv_change_map=oamv_change_map,
                               eligible_codes=eligible_codes)
    print(f"  Loaded {len(stock_data)} stocks")

    # 2. Grid search on training data (2018-2024)
    grid_results, base_z, base_n = run_grid_search(stock_data)

    # 3. Validate top-5 on 2025
    validated, base_z_val = validate_top(grid_results, stock_data, top_n=5)

    # 4. Full simulation test on 2026
    print(f"\n{'='*70}")
    print(f"  Final Test: Full Simulation on 2026")
    print(f"{'='*70}")

    # Need full stock data for sim engine (not just sampled)
    print("  Loading full stock set for final simulation...")
    full_stock_data = preload_data(oamv_change_map=oamv_change_map,
                                    eligible_codes=eligible_codes)
    print(f"  Loaded {len(full_stock_data)} stocks for final test")

    # Test baseline
    base_params_no_dynamic = copy.deepcopy(BASE_PARAMS)
    base_params_no_dynamic['dynamic_thresholds'] = False
    print("\n  --- Baseline (Fixed Thresholds) ---")
    base_metrics = run_full_simulation_test(full_stock_data, base_params_no_dynamic, "BASELINE_Fixed")

    # Test best dynamic
    best_val = validated[0]
    best_params = build_dynamic_params(
        AGGRESSIVE_PROFILES[best_val['agg_profile']],
        NORMAL_PROFILES[best_val['norm_profile']],
        DEFENSIVE_PROFILES[best_val['def_profile']])
    print("\n  --- Best Dynamic ---")
    best_metrics = run_full_simulation_test(full_stock_data, best_params, "BEST_Dynamic")

    # Also test all-medium dynamic for comparison
    all_medium = build_dynamic_params(
        AGGRESSIVE_PROFILES['agg_medium'],
        NORMAL_PROFILES['norm_medium'],
        DEFENSIVE_PROFILES['def_medium'])
    print("\n  --- Dynamic All-Medium (control) ---")
    med_metrics = run_full_simulation_test(full_stock_data, all_medium, "DYNAMIC_AllMedium")

    # 5. Summary
    print(f"\n{'='*70}")
    print(f"  Phase 5 Summary")
    print(f"{'='*70}")

    print(f"\n  Grid Search (Train {TRAIN_START.date()}~{TRAIN_END.date()}):")
    print(f"    Fixed baseline z: {base_z*100:.3f}%")
    print(f"    Top-5 dynamic z range: {grid_results[0]['z']*100:.3f}% - {grid_results[4]['z']*100:.3f}%")

    print(f"\n  Validation (2025):")
    print(f"    Fixed baseline z_val: {base_z_val*100:.3f}%")
    for v in validated:
        print(f"    {v['name']:<45s} z_val={v['z_val']*100:+.3f}% (delta {v['val_delta_pct']:+.1f}%)")

    print(f"\n  Full Simulation (Test 2026):")
    for name, m in [("BASELINE Fixed", base_metrics), ("BEST Dynamic", best_metrics), ("DYNAMIC All-Medium", med_metrics)]:
        print(f"    {name:<25s} Ret={m['total_return_pct']:+.1f}% Sharpe={m['sharpe']:.3f} "
              f"WR={m['win_rate']:.1%} DD={m['max_dd']:.1%} Trades={m['n_trades']}")

    # Improvement
    if best_metrics['sharpe'] > base_metrics['sharpe']:
        delta_sharpe = best_metrics['sharpe'] - base_metrics['sharpe']
        delta_ret = best_metrics['total_return_pct'] - base_metrics['total_return_pct']
        print(f"\n  Dynamic improves over fixed: Delta Sharpe=+{delta_sharpe:.3f}, Delta Ret={delta_ret:+.1f}%")
    else:
        print(f"\n  Dynamic does NOT improve over fixed in full simulation.")
        print(f"  Best grid z may have overfit to train/val period.")

    # Best per-regime thresholds
    print(f"\n  Best Per-Regime Thresholds:")
    print(f"    Aggressive:  {AGGRESSIVE_PROFILES[best_val['agg_profile']]}")
    print(f"    Normal:      {NORMAL_PROFILES[best_val['norm_profile']]}")
    print(f"    Defensive:   {DEFENSIVE_PROFILES[best_val['def_profile']]}")

    # 6. Save
    out = {
        'phase': '5_dynamic_thresholds',
        'date_splits': {
            'train': f'{TRAIN_START.date()}~{TRAIN_END.date()}',
            'val': f'{VAL_START.date()}~{VAL_END.date()}',
            'test': f'{TEST_START.date()}~{TEST_END.date()}',
        },
        'n_stocks_train': len(stock_data),
        'n_stocks_test': len(full_stock_data),
        'baseline': {
            'fixed_train_z': round(base_z, 6),
            'fixed_val_z': round(base_z_val, 6),
            'fixed_test_sharpe': round(base_metrics['sharpe'], 4),
            'fixed_test_return_pct': round(base_metrics['total_return_pct'], 2),
            'fixed_test_win_rate': round(base_metrics['win_rate'], 4),
            'fixed_test_trades': base_metrics['n_trades'],
        },
        'grid_top10': [{
            'rank': i+1,
            'name': r['name'],
            'z_train': round(r['z'], 6),
            'n_signals': r['n_signals'],
            'agg_profile': r['agg_profile'],
            'norm_profile': r['norm_profile'],
            'def_profile': r['def_profile'],
        } for i, r in enumerate(grid_results[:10])],
        'validation_top5': [{
            'name': v['name'],
            'z_val': round(v['z_val'], 6),
            'val_delta_pct': round(v['val_delta_pct'], 2),
        } for v in validated],
        'best_dynamic': {
            'name': best_val['name'],
            'test_sharpe': round(best_metrics['sharpe'], 4),
            'test_return_pct': round(best_metrics['total_return_pct'], 2),
            'test_win_rate': round(best_metrics['win_rate'], 4),
            'test_trades': best_metrics['n_trades'],
            'per_regime_params': {
                'aggressive': AGGRESSIVE_PROFILES[best_val['agg_profile']],
                'normal': NORMAL_PROFILES[best_val['norm_profile']],
                'defensive': DEFENSIVE_PROFILES[best_val['def_profile']],
            },
        },
        'elapsed_seconds': time.time() - t0,
    }

    out_path = os.path.join(OUT_DIR, 'phase5_dynamic_thresholds.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Phase 5 Complete ({time.time()-t0:.0f}s)")
    print(f"  Results: {out_path}")


if __name__ == '__main__':
    main()
