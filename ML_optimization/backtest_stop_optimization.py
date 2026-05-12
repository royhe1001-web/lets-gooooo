#!/usr/bin/env python3
"""
Stop-Loss Parameter Optimization for Explosive_def2.0
======================================================
Uses the proven Phase 2c Explosive_def2.0 (fixed threshold) as base strategy.
Train 2024→2025Q3, Val 2025Q4, Test 2026 — matching Phase 2c/5 window.
Sweeps stop-loss parameters one-at-a-time on VALIDATION (2025Q4),
then tests best config on 2026 + multi-period (3Y/5Y/10Y).

Key insight from Phase 2c/5: Explosive_def2.0 is the "iron parameter" —
consistent +17.1% Test 2026 across ALL training windows. Stop optimization
builds on this solid foundation.
"""

import os, sys, time, json, copy, itertools
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import ML_optimization.phase2c_oamv_grid_search as p2c_mod

from ML_optimization.mktcap_utils import build_mktcap_lookup
from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# ============================================================
# Phase 2c Explosive_def2.0 — proven optimal across all windows
# ============================================================
B2_PARAMS = {
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
    'dynamic_thresholds': False,
}

# Baseline stop config (v2 from previous optimization)
BASE_STOP = {
    'hard_stop': {'aggressive': -0.18, 'normal': -0.12, 'defensive': -0.08},
    'hard_stop_min_days': 5,
    'trailing_activate': 0.10, 'trailing_pct': 0.06,
    'take_profit_partial': 0.30, 'take_profit_partial_frac': 0.5,
    'take_profit_full': 0.60,
    'time_stop_loser_days': 30, 'time_stop_loser_pct': -0.05,
    'peak_dd_stop': -0.15,
    'checkpoint_60d_min_ret': 0.05,
    'max_hold_days': 180,
}

# Parameter sweep ranges (one-at-a-time from baseline)
SWEEP_PARAMS = {
    'hard_stop_normal': {
        'path': ['hard_stop', 'normal'],
        'values': [-0.08, -0.10, -0.12, -0.15, -0.18],
    },
    'trailing_activate': {
        'path': ['trailing_activate'],
        'values': [0.05, 0.08, 0.10, 0.12, 0.15],
    },
    'trailing_pct': {
        'path': ['trailing_pct'],
        'values': [0.03, 0.04, 0.06, 0.08, 0.10],
    },
    'take_profit_partial': {
        'path': ['take_profit_partial'],
        'values': [0.15, 0.20, 0.25, 0.30, 0.40],
    },
    'time_stop_loser_days': {
        'path': ['time_stop_loser_days'],
        'values': [15, 20, 25, 30, 40],
    },
    'peak_dd_stop': {
        'path': ['peak_dd_stop'],
        'values': [-0.10, -0.12, -0.15, -0.18, -0.20],
    },
    'checkpoint_60d_min_ret': {
        'path': ['checkpoint_60d_min_ret'],
        'values': [0.0, 0.03, 0.05, 0.08, 0.10],
    },
    'max_hold_days': {
        'path': ['max_hold_days'],
        'values': [90, 120, 150, 180, 252],
    },
}

# Date split matching Phase 2c/5: Train 2024→2025Q3, Val 2025Q4, Test 2026
TRAIN_START = pd.Timestamp('2024-01-01')
TRAIN_END   = pd.Timestamp('2025-09-30')
VAL_START   = pd.Timestamp('2025-10-01')
VAL_END     = pd.Timestamp('2025-12-31')
TEST_START  = pd.Timestamp('2026-01-01')
TEST_END    = pd.Timestamp('2026-05-07')

# Parameter sweep on VALIDATION (2025Q4), final test on 2026 + long-term
VAL_PERIOD  = {'name': 'Val_2025Q4', 'sim_start': VAL_START, 'sim_end': VAL_END}
TEST_PERIOD = {'name': 'Test_2026',  'sim_start': TEST_START, 'sim_end': TEST_END}

PERIODS = [
    TEST_PERIOD,
    {'name': '3Y',    'sim_start': pd.Timestamp('2023-05-01'), 'sim_end': TEST_END},
    {'name': '5Y',    'sim_start': pd.Timestamp('2021-05-01'), 'sim_end': TEST_END},
    {'name': '10Y',   'sim_start': pd.Timestamp('2016-05-01'), 'sim_end': TEST_END},
]


def set_nested(d, path, value):
    """Set a nested dict value by path list."""
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


class StopOptimEngine(p2c_mod.OAMVSimEngine):
    """OAMVSimEngine with configurable stop-loss."""

    def __init__(self, *args, stop_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_cfg = stop_config or BASE_STOP

    def check_stops(self, date):
        to_sell = []
        agg_buf = self.oamv_params.get('oamv_aggressive_buffer', 0.95)
        def_buf = self.oamv_params.get('oamv_defensive_buffer', 0.99)

        for code, pos in list(self.positions.items()):
            df = self.stock_data.get(code)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            close_today = float(row['close'])
            regime = self._get_oamv_regime(date)

            buy_idx = df.index.get_loc(pos.buy_date) if pos.buy_date in df.index else None
            today_idx = df.index.get_loc(date) if date in df.index else None
            days_held = (today_idx - buy_idx) if buy_idx is not None and today_idx is not None else 0
            is_buy_day = (date == pos.buy_date)
            cum_return = (close_today / pos.buy_price - 1)

            if close_today > pos.peak_price:
                pos.peak_price = close_today

            # 1. Time stop for early losers
            if (not is_buy_day and days_held >= self.stop_cfg['time_stop_loser_days']
                    and cum_return < self.stop_cfg['time_stop_loser_pct']):
                to_sell.append((code, date, close_today,
                               f'time_loser', 1.0))
                continue

            # 2. Hard stop-loss (per-regime, with min hold days)
            hard_stop_pct = self.stop_cfg['hard_stop'].get(regime, -0.12)
            min_days = self.stop_cfg.get('hard_stop_min_days', 5)
            if (not is_buy_day and days_held >= min_days
                    and cum_return <= hard_stop_pct):
                to_sell.append((code, date, close_today,
                               f'hard_stop_{regime}', 1.0))
                continue

            # 3. Peak drawdown stop
            if not is_buy_day and pos.peak_price > 0:
                peak_dd = (close_today / pos.peak_price - 1)
                if peak_dd <= self.stop_cfg['peak_dd_stop']:
                    to_sell.append((code, date, close_today, 'peak_dd', 1.0))
                    continue

            # 4. Trailing stop
            if (not is_buy_day and cum_return >= self.stop_cfg['trailing_activate']
                    and pos.peak_price > pos.buy_price):
                trail_level = pos.peak_price * (1 - self.stop_cfg['trailing_pct'])
                if close_today < trail_level:
                    to_sell.append((code, date, close_today, 'trail', 1.0))
                    continue

            # 5. Take profit
            if not is_buy_day and cum_return >= self.stop_cfg['take_profit_full']:
                to_sell.append((code, date, close_today, 'take_full', 1.0))
                continue

            if not is_buy_day and cum_return >= self.stop_cfg['take_profit_partial'] and not pos.half_sold:
                to_sell.append((code, date, close_today, 'take_half',
                               self.stop_cfg['take_profit_partial_frac']))
                continue

            # 6. 60-day checkpoint
            if not is_buy_day and days_held >= 60 and cum_return < self.stop_cfg['checkpoint_60d_min_ret']:
                to_sell.append((code, date, close_today, '60d_weak', 1.0))
                continue

            # 7. Max hold
            if not is_buy_day and days_held >= self.stop_cfg['max_hold_days']:
                to_sell.append((code, date, close_today, 'max_hold', 1.0))
                continue

            # 8. Original stops (candle/yellow/signal)
            if regime == 'aggressive':
                candle_level = pos.entry_low * agg_buf
            elif regime == 'defensive':
                candle_level = pos.entry_low * def_buf
            else:
                candle_level = pos.entry_low

            candle_stop = close_today < candle_level
            signal_stop = close_today < pos.signal_close
            yellow = float(row.get('yellow_line', 0))
            yellow_stop = yellow > 0 and close_today < yellow

            if candle_stop:
                reason = f'candle_{regime}'
            elif signal_stop:
                reason = 'sig_close'
            elif yellow_stop:
                reason = 'yellow'
            else:
                continue

            if is_buy_day:
                next_dates = df.index[df.index > date]
                if len(next_dates) > 0:
                    t2_date = next_dates[0]
                    t2_open = float(df.loc[t2_date, 'open'])
                    to_sell.append((code, t2_date, t2_open, f'buyday_{reason}', 1.0))
                else:
                    to_sell.append((code, date, close_today, f'buyday_{reason}', 1.0))
            else:
                to_sell.append((code, date, close_today, reason, 1.0))

        for code, exit_date, exit_price, reason, partial in to_sell:
            if code in self.positions:
                self.sell(code, exit_date, exit_price, reason, partial=partial)


def preload_stock_data(eligible_codes):
    stock_data = {}
    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    MAIN_PREFIXES = ('000', '002', '003', '600', '601', '603', '605')
    needed = ['open', 'high', 'low', 'close', 'volume',
              'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
              'white_line', 'yellow_line',
              'brick_val', 'brick_rising', 'brick_falling',
              'brick_green_to_red', 'brick_red_to_green',
              'brick_green_streak', 'brick_red_streak', 'cross_below_yellow']

    loaded = 0
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
            cols_avail = [c for c in needed if c in df.columns]
            if len(cols_avail) < 8:
                continue
            stock_data[code] = df[cols_avail].copy()
            loaded += 1
            if loaded % 100 == 0:
                print(f"  Loaded {loaded} stocks...")
        except Exception:
            continue
    print(f"  Loaded {loaded} stocks total")
    return stock_data


def run_one(stock_data, stop_config, sim_start, sim_end, oamv_df):
    """Run single backtest and return metrics.
    If stop_config is None, uses the original OAMVSimEngine (no improved stops)."""
    p2c_mod.SIM_START = sim_start
    p2c_mod.SIM_END = sim_end

    sim_params = copy.deepcopy(B2_PARAMS)
    oamv_p = {
        'oamv_aggressive_threshold': sim_params.pop('oamv_aggressive_threshold', 3.0),
        'oamv_defensive_threshold': sim_params.pop('oamv_defensive_threshold', -2.0),
        'oamv_aggressive_buffer': sim_params.pop('oamv_aggressive_buffer', 0.95),
        'oamv_defensive_buffer': sim_params.pop('oamv_defensive_buffer', 0.99),
        'oamv_defensive_ban_entry': sim_params.pop('oamv_defensive_ban_entry', True),
    }

    if stop_config is not None:
        engine = StopOptimEngine(stock_data, sim_params, oamv_df, oamv_p,
                                 stop_config=stop_config)
    else:
        engine = p2c_mod.OAMVSimEngine(stock_data, sim_params, oamv_df, oamv_p)
    metrics = engine.run()

    # Stop reason breakdown
    stop_counts = {}
    for t in engine.trade_log:
        if t['action'] == 'SELL':
            reason = t.get('reason', 'unknown')
            base = reason.split('(')[0].split('_')[0] if '_' in reason else reason.split('(')[0]
            stop_counts[base] = stop_counts.get(base, 0) + 1

    return {
        'return_pct': round(metrics['total_return_pct'], 2),
        'sharpe': round(metrics['sharpe'], 4),
        'win_rate': round(metrics['win_rate'], 4),
        'max_dd': round(metrics['max_dd'], 4),
        'n_trades': metrics['n_trades'],
        'n_signals': metrics.get('n_signals', 0),
        'stop_counts': stop_counts,
        'total_return': metrics['total_return'],
    }


def sweep_param(param_name, param_info, stock_data, oamv_df, period):
    """Sweep one parameter, return sorted results."""
    print(f"\n  --- Sweeping {param_name} ---")
    results = []
    for val in param_info['values']:
        cfg = copy.deepcopy(BASE_STOP)
        set_nested(cfg, param_info['path'], val)
        label = f"{param_name}={val}"
        t0 = time.time()
        m = run_one(stock_data, cfg, period['sim_start'], period['sim_end'], oamv_df)
        elapsed = time.time() - t0
        print(f"    {label:<30s} Ret={m['return_pct']:+.1f}%  Sharpe={m['sharpe']:.3f}  "
              f"WR={m['win_rate']:.1%}  DD={m['max_dd']:.1%}  Trades={m['n_trades']}  ({elapsed:.0f}s)")
        results.append({'param': param_name, 'value': val, 'label': label, **m})
    results.sort(key=lambda r: r['sharpe'], reverse=True)
    best = results[0]
    print(f"    BEST: {best['label']} (Sharpe={best['sharpe']:.3f})")
    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("  Stop-Loss Parameter Optimization")
    print("  Base Strategy: Phase 2c Explosive_def2.0 (Fixed Threshold)")
    print("=" * 80)

    # Load data
    print("\n[1/3] Loading market cap + OAMV + stock data...")
    mktcap_lookup = build_mktcap_lookup()
    eligible_codes = set()
    for (ym, sym), mv in mktcap_lookup.items():
        if '2010-01' <= ym <= '2026-05' and 500 <= mv <= 1000:
            eligible_codes.add(sym)
    print(f"  Eligible: {len(eligible_codes)} stocks (500-1000亿)")

    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    print(f"  OAMV: {len(oamv_df)} days")

    stock_data = preload_stock_data(eligible_codes)
    print(f"  Stocks loaded: {len(stock_data)}")

    # Phase 1: Baseline (no improved stops) on Val + Test
    print(f"\n{'='*80}")
    print("  PHASE 1: Baseline — Original Engine (no improved stops)")
    print(f"{'='*80}")
    baseline_val = run_one(stock_data, None, VAL_PERIOD['sim_start'], VAL_PERIOD['sim_end'], oamv_df)
    print(f"  Val_2025Q4:  Ret={baseline_val['return_pct']:+.1f}%  Sharpe={baseline_val['sharpe']:.3f}  "
          f"WR={baseline_val['win_rate']:.1%}  DD={baseline_val['max_dd']:.1%}  Trades={baseline_val['n_trades']}")
    baseline = run_one(stock_data, None, TEST_PERIOD['sim_start'], TEST_PERIOD['sim_end'], oamv_df)
    print(f"  Test_2026:   Ret={baseline['return_pct']:+.1f}%  Sharpe={baseline['sharpe']:.3f}  "
          f"WR={baseline['win_rate']:.1%}  DD={baseline['max_dd']:.1%}  Trades={baseline['n_trades']}")

    # Phase 2: Baseline improved stops on Val + Test
    print(f"\n{'='*80}")
    print("  PHASE 2: Baseline Improved Stops (v2 config)")
    print(f"{'='*80}")
    impr_base_val = run_one(stock_data, BASE_STOP, VAL_PERIOD['sim_start'], VAL_PERIOD['sim_end'], oamv_df)
    print(f"  Val_2025Q4:  Ret={impr_base_val['return_pct']:+.1f}%  Sharpe={impr_base_val['sharpe']:.3f}  "
          f"WR={impr_base_val['win_rate']:.1%}  DD={impr_base_val['max_dd']:.1%}  Trades={impr_base_val['n_trades']}")
    impr_base = run_one(stock_data, BASE_STOP, TEST_PERIOD['sim_start'], TEST_PERIOD['sim_end'], oamv_df)
    print(f"  Test_2026:   Ret={impr_base['return_pct']:+.1f}%  Sharpe={impr_base['sharpe']:.3f}  "
          f"WR={impr_base['win_rate']:.1%}  DD={impr_base['max_dd']:.1%}  Trades={impr_base['n_trades']}")
    print(f"  Stop reasons: {impr_base['stop_counts']}")

    # Phase 3: One-at-a-time parameter sweeps on VALIDATION (2025Q4)
    print(f"\n{'='*80}")
    print("  PHASE 3: Parameter Sweeps (Val 2025Q4, one-at-a-time)")
    print(f"{'='*80}")

    all_sweeps = {}
    best_config = copy.deepcopy(BASE_STOP)

    for param_name, param_info in SWEEP_PARAMS.items():
        sweep_results = sweep_param(param_name, param_info, stock_data, oamv_df, VAL_PERIOD)
        all_sweeps[param_name] = sweep_results
        # Update best config with best value for this param
        best_val = sweep_results[0]['value']
        set_nested(best_config, param_info['path'], best_val)

    # Phase 4: Multi-period validation with best config
    print(f"\n{'='*80}")
    print("  PHASE 4: Multi-Period Validation (Best Config)")
    print(f"{'='*80}")

    print(f"\n  Best Stop Config:")
    for k, v in best_config.items():
        print(f"    {k}: {v}")

    print(f"\n  {'Period':<10s} {'Engine':<15s} {'Return':>10s} {'Sharpe':>8s} "
          f"{'WR':>8s} {'MaxDD':>8s} {'Trades':>8s}  Stop Reasons")
    print(f"  {'-'*90}")

    multi_results = []
    for period in PERIODS:
        # Original
        orig = run_one(stock_data, None, period['sim_start'], period['sim_end'], oamv_df)
        # Best improved
        impr = run_one(stock_data, best_config, period['sim_start'], period['sim_end'], oamv_df)
        # Baseline improved (for comparison)
        impr_bl = run_one(stock_data, BASE_STOP, period['sim_start'], period['sim_end'], oamv_df)

        top_stops = sorted(impr['stop_counts'].items(), key=lambda x: -x[1])[:3]
        reasons_str = ', '.join(f'{k}:{v}' for k, v in top_stops)

        print(f"  {period['name']:<10s} {'Original':<15s} "
              f"{orig['return_pct']:>+9.1f}% {orig['sharpe']:>8.3f} "
              f"{orig['win_rate']:>7.1%} {orig['max_dd']:>7.1%} {orig['n_trades']:>8d}")
        print(f"  {period['name']:<10s} {'Improved(v2)':<15s} "
              f"{impr_bl['return_pct']:>+9.1f}% {impr_bl['sharpe']:>8.3f} "
              f"{impr_bl['win_rate']:>7.1%} {impr_bl['max_dd']:>7.1%} {impr_bl['n_trades']:>8d}")
        print(f"  {period['name']:<10s} {'Improved(BEST)':<15s} "
              f"{impr['return_pct']:>+9.1f}% {impr['sharpe']:>8.3f} "
              f"{impr['win_rate']:>7.1%} {impr['max_dd']:>7.1%} {impr['n_trades']:>8d}  [{reasons_str}]")

        multi_results.append({
            'period': period['name'],
            'original': orig,
            'improved_v2': impr_bl,
            'improved_best': impr,
        })

    # Save
    out = {
        'phase': 'stop_optimization',
        'date': '2026-05-12',
        'train_val_test': 'Train 2024→2025Q3, Val 2025Q4, Test 2026',
        'base_strategy': 'Phase 2c Explosive_def2.0 (Fixed Threshold)',
        'baseline_val_2025Q4': baseline_val,
        'improved_val_2025Q4': impr_base_val,
        'baseline_test_2026': baseline,
        'improved_baseline_test_2026': impr_base,
        'best_config': {str(k): v for k, v in best_config.items()},
        'sweeps': {k: [{kk: vv for kk, vv in r.items() if kk != 'total_return'}
                       for r in v] for k, v in all_sweeps.items()},
        'multi_period': multi_results,
        'elapsed_seconds': time.time() - t0,
    }
    with open(os.path.join(OUT_DIR, 'backtest_stop_optimization.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Optimization Complete ({elapsed:.0f}s)")
    print(f"  Results: {os.path.join(OUT_DIR, 'backtest_stop_optimization.json')}")


if __name__ == '__main__':
    main()
