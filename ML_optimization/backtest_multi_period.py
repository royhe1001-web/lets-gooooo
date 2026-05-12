#!/usr/bin/env python3
"""
Multi-Period Backtest — B2 Optimal Parameters
==============================================
Runs full OAMVSimEngine backtests on three horizons:
  10Y: 2016-05 ~ 2026-05
  5Y:  2021-05 ~ 2026-05
  3Y:  2023-05 ~ 2026-05

Configuration:
  Stock pool:  Main board + 500-1000亿 dynamic mktcap filter
  B2 params:   Phase 2c Explosive_def2.0 (optimal)
  Thresholds:  Phase 5 dynamic per-regime (agg_loose + norm_tight + def_tight)
"""

import os, sys, time, json, copy
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Monkey-patch SIM_START / SIM_END before importing the engine
import ML_optimization.phase2c_oamv_grid_search as p2c_mod

from ML_optimization.mktcap_utils import build_mktcap_lookup
from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from quant_strategy.strategy_spring import generate_spring_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# ============================================================
# Optimal Parameters (from research report v2.0)
# ============================================================

# Phase 2c Explosive B2 base params
B2_BASE = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.09, 'weight_gain_2x': 2.5,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'oamv_aggressive_threshold': 3.0, 'oamv_defensive_threshold': -2.0,
    'oamv_aggressive_buffer': 0.95, 'oamv_defensive_buffer': 0.99,
    'oamv_defensive_ban_entry': True,
}

# Phase 5 optimal dynamic thresholds: agg_loose + norm_tight + def_tight (merged with B2_BASE)
DYNAMIC_PARAMS = copy.deepcopy(B2_BASE)
DYNAMIC_PARAMS.update({
    'dynamic_thresholds': True,
    # Aggressive (牛市宽松, 乘胜追击)
    'gain_min_aggressive': 0.03, 'j_today_max_aggressive': 70, 'shadow_max_aggressive': 0.050,
    # Normal (收紧过滤噪声)
    'gain_min_normal': 0.06, 'j_today_max_normal': 55, 'shadow_max_normal': 0.030,
    # Defensive (最紧, 保守等待高质信号)
    'gain_min_defensive': 0.07, 'j_today_max_defensive': 50, 'shadow_max_defensive': 0.020,
})

# Fixed-threshold comparison (Phase 2c optimal)
FIXED_PARAMS = copy.deepcopy(B2_BASE)
FIXED_PARAMS['dynamic_thresholds'] = False

# ============================================================
# Multi-period configuration
# ============================================================
PERIODS = [
    {
        'name': '10Y (2016-2026)',
        'sim_start': pd.Timestamp('2016-05-01'),
        'sim_end':   pd.Timestamp('2026-05-07'),
        'train_start': pd.Timestamp('2010-01-01'),
        'train_end':   pd.Timestamp('2016-04-30'),
    },
    {
        'name': '5Y (2021-2026)',
        'sim_start': pd.Timestamp('2021-05-01'),
        'sim_end':   pd.Timestamp('2026-05-07'),
        'train_start': pd.Timestamp('2010-01-01'),
        'train_end':   pd.Timestamp('2021-04-30'),
    },
    {
        'name': '3Y (2023-2026)',
        'sim_start': pd.Timestamp('2023-05-01'),
        'sim_end':   pd.Timestamp('2026-05-07'),
        'train_start': pd.Timestamp('2010-01-01'),
        'train_end':   pd.Timestamp('2023-04-30'),
    },
]


def preload_stock_data(eligible_codes, oamv_change_map):
    """Load stock data for eligible codes, with OAMV pre-attached."""
    stock_data = {}
    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    MAIN_PREFIXES = ('000', '002', '003', '600', '601', '603', '605')

    for f in feat_files:
        code = f.replace('.parquet', '')
        if not any(code.startswith(p) for p in MAIN_PREFIXES):
            continue
        if code not in eligible_codes:
            continue

        try:
            df = pd.read_parquet(os.path.join(FEAT_DIR, f))
            if 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            needed = ['open', 'high', 'low', 'close', 'volume',
                      'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                      'white_line', 'yellow_line',
                      'brick_val', 'brick_rising', 'brick_falling',
                      'brick_green_to_red', 'brick_red_to_green',
                      'brick_green_streak', 'brick_red_streak',
                      'cross_below_yellow']
            cols_avail = [c for c in needed if c in df.columns]
            if len(cols_avail) < 8:
                continue

            sdf = df[cols_avail].copy()
            stock_data[code] = sdf
        except Exception:
            continue

    return stock_data


def run_one_backtest(stock_data, params, sim_start, sim_end, oamv_df, label):
    """Run OAMVSimEngine for a specific period."""
    # Monkey-patch SIM_START/SIM_END for this run
    p2c_mod.SIM_START = sim_start
    p2c_mod.SIM_END = sim_end

    sim_params = copy.deepcopy(params)
    oamv_p = {
        'oamv_aggressive_threshold': sim_params.pop('oamv_aggressive_threshold', 3.0),
        'oamv_defensive_threshold': sim_params.pop('oamv_defensive_threshold', -2.0),
        'oamv_aggressive_buffer': sim_params.pop('oamv_aggressive_buffer', 0.95),
        'oamv_defensive_buffer': sim_params.pop('oamv_defensive_buffer', 0.99),
        'oamv_defensive_ban_entry': sim_params.pop('oamv_defensive_ban_entry', True),
    }

    engine = p2c_mod.OAMVSimEngine(stock_data, sim_params, oamv_df, oamv_p)
    metrics = engine.run()

    n_days = (sim_end - sim_start).days
    ann_ret = metrics['total_return']  # already a fraction
    ann_factor = 365.25 / max(n_days, 1)
    cagr = ((1 + ann_ret) ** ann_factor - 1) * 100 if ann_ret > -1 else None

    print(f"  {label:<20s} "
          f"Ret={metrics['total_return_pct']:+.1f}%  "
          f"Sharpe={metrics['sharpe']:.3f}  "
          f"WR={metrics['win_rate']:.1%}  "
          f"DD={metrics['max_dd']:.1%}  "
          f"Trades={metrics['n_trades']}  "
          f"CAGR={cagr:+.1f}%" if cagr is not None else f"  {label:<20s} Ret={metrics['total_return_pct']:+.1f}%")

    return {**metrics, 'cagr_pct': cagr, 'n_days': n_days,
            'sim_start': str(sim_start.date()), 'sim_end': str(sim_end.date())}


def main():
    t0 = time.time()
    print("=" * 80)
    print("  Multi-Period Backtest — B2 Optimal Strategy")
    print("  Params: Phase 2c Explosive_def2.0 + Phase 5 Dynamic (agg_loose/norm_tight/def_tight)")
    print("  Filter: Main Board + 500-1000亿 mktcap")
    print("=" * 80)

    # 1. Load mktcap filter
    print("\n[1/4] Loading market cap lookup...")
    mktcap_lookup = build_mktcap_lookup()
    eligible_codes = set()
    ym_all = ('2010-01', '2026-05')
    for (ym, sym), mv in mktcap_lookup.items():
        if ym_all[0] <= ym <= ym_all[1] and 500 <= mv <= 1000:
            eligible_codes.add(sym)
    print(f"  Eligible codes (500-1000亿, 2010-2026): {len(eligible_codes)}")

    # 2. Load OAMV
    print("\n[2/4] Loading OAMV data...")
    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    print(f"  OAMV: {len(oamv_df)} days, {oamv_df['date'].min().date()} ~ {oamv_df['date'].max().date()}")

    # 3. Load stock data
    print("\n[3/4] Loading stock data...")
    oamv_change_map = {}
    for _, row in oamv_df.iterrows():
        d = row['date']
        if hasattr(d, 'date'):
            d = d.date()
        oamv_change_map[pd.Timestamp(d)] = float(row['oamv_change_pct'])

    stock_data = preload_stock_data(eligible_codes, oamv_change_map)
    print(f"  Loaded {len(stock_data)} stocks")

    # 4. Run backtests
    print(f"\n[4/4] Running backtests ({len(PERIODS)} periods x 2 configs)...")
    print(f"  {'='*75}")

    all_results = []

    for period in PERIODS:
        print(f"\n  --- {period['name']} ---")

        # Dynamic threshold run
        dyn_result = run_one_backtest(
            stock_data, copy.deepcopy(DYNAMIC_PARAMS),
            period['sim_start'], period['sim_end'],
            oamv_df, "Dynamic (最优)"
        )

        # Fixed threshold run
        fix_result = run_one_backtest(
            stock_data, copy.deepcopy(FIXED_PARAMS),
            period['sim_start'], period['sim_end'],
            oamv_df, "Fixed (Phase 2c)"
        )

        all_results.append({
            'period': period['name'],
            'sim_start': str(period['sim_start'].date()),
            'sim_end': str(period['sim_end'].date()),
            'n_days': dyn_result['n_days'],
            'dynamic': {
                'return_pct': round(dyn_result['total_return_pct'], 2),
                'cagr_pct': round(dyn_result['cagr_pct'], 2) if dyn_result['cagr_pct'] else None,
                'sharpe': round(dyn_result['sharpe'], 4),
                'win_rate': round(dyn_result['win_rate'], 4),
                'max_dd': round(dyn_result['max_dd'], 4),
                'n_trades': dyn_result['n_trades'],
            },
            'fixed': {
                'return_pct': round(fix_result['total_return_pct'], 2),
                'cagr_pct': round(fix_result['cagr_pct'], 2) if fix_result['cagr_pct'] else None,
                'sharpe': round(fix_result['sharpe'], 4),
                'win_rate': round(fix_result['win_rate'], 4),
                'max_dd': round(fix_result['max_dd'], 4),
                'n_trades': fix_result['n_trades'],
            },
        })

    # 5. Summary table
    print(f"\n{'='*80}")
    print(f"  Multi-Period Backtest — Summary")
    print(f"{'='*80}")
    print(f"  {'Period':<20s} {'Config':<20s} {'Return':>10s} {'CAGR':>10s} {'Sharpe':>8s} {'WR':>8s} {'MaxDD':>8s} {'Trades':>8s}")
    print(f"  {'-'*80}")

    for r in all_results:
        for cfg_key, cfg_label in [('dynamic', 'Dynamic (Phase 5)'), ('fixed', 'Fixed (Phase 2c)')]:
            cfg = r[cfg_key]
            cagr_str = f"{cfg['cagr_pct']:+.1f}%" if cfg['cagr_pct'] is not None else "N/A"
            print(f"  {r['period']:<20s} {cfg_label:<20s} "
                  f"{cfg['return_pct']:>+9.1f}% {cagr_str:>10s} "
                  f"{cfg['sharpe']:>8.3f} {cfg['win_rate']:>7.1%} "
                  f"{cfg['max_dd']:>7.1%} {cfg['n_trades']:>8d}")

    # Save results
    out = {
        'phase': 'backtest_multi_period',
        'date': '2026-05-11',
        'config': {
            'b2_params': 'Phase 2c Explosive_def2.0',
            'dynamic_thresholds': 'Phase 5 agg_loose + norm_tight + def_tight',
            'stock_filter': 'Main board + 500-1000亿 mktcap',
            'n_eligible_codes': len(eligible_codes),
            'n_loaded_stocks': len(stock_data),
        },
        'results': all_results,
        'elapsed_seconds': time.time() - t0,
    }

    out_path = os.path.join(OUT_DIR, 'backtest_multi_period.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Backtest Complete ({elapsed:.0f}s)")
    print(f"  Results: {out_path}")


if __name__ == '__main__':
    main()
