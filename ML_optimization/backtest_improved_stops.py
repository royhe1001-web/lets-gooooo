#!/usr/bin/env python3
"""
Improved Stop-Loss Backtest
===========================
Overrides OAMVSimEngine.check_stops with enhanced risk management:
  1. Hard stop-loss (per-regime): aggressive -12% / normal -8% / defensive -5%
  2. Trailing stop: activates at +10%, trails at -6% from peak
  3. Take-profit: +30% take half, +50% take all
  4. Time stop: cut losers after 20 days if still < -3%
  5. Peak-DD stop: -10% from position peak triggers exit
  6. Improved 60-day checkpoint: tighter criteria

Compares original vs improved on 10Y / 5Y / 3Y periods.
"""

import os, sys, time, json, copy
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import ML_optimization.phase2c_oamv_grid_search as p2c_mod

from ML_optimization.mktcap_utils import build_mktcap_lookup
from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from quant_strategy.strategy_spring import generate_spring_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# ============================================================
# Improved Stop-Loss Parameters
# ============================================================
STOP_CONFIG = {
    # Hard stop: maximum loss from buy price (per-regime)
    # B2 is a pullback-buying strategy — stops must be wide enough to
    # survive the pullback itself, especially in bull markets.
    'hard_stop': {
        'aggressive': -0.18,   # very loose — bull market pullbacks are shallow but can gap
        'normal':    -0.12,    # standard 12% max loss (widened from 8%)
        'defensive': -0.08,    # tighter in bear — preserve capital
    },
    'hard_stop_min_days': 5,   # don't trigger hard stop in first 5 days (avoid gap-down whipsaw)
    # Trailing stop: activates when position reaches this gain
    'trailing_activate': 0.10,   # activate trailing after +10%
    'trailing_pct': 0.06,        # trail 6% below peak close
    # Take profit
    'take_profit_partial': 0.30,  # sell half at +30%
    'take_profit_partial_frac': 0.5,
    'take_profit_full': 0.60,     # sell all at +60%
    # Time stop: cut losers if held too long without recovery
    'time_stop_loser_days': 30,
    'time_stop_loser_pct': -0.05,  # worse than -5% after 30 days
    # Peak drawdown stop
    'peak_dd_stop': -0.15,  # -15% from position peak (widened)
    # Improved 60-day checkpoint
    'checkpoint_60d_min_ret': 0.05,  # require +5% by day 60
    # Max hold: force close at 180 days (was 252)
    'max_hold_days': 180,
}


class ImprovedOAMVSimEngine(p2c_mod.OAMVSimEngine):
    """OAMVSimEngine with enhanced stop-loss and risk management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_cfg = STOP_CONFIG

    def check_stops(self, date):
        to_sell = []
        agg_buf = self.oamv_params.get('oamv_aggressive_buffer', 0.95)
        def_buf = self.oamv_params.get('oamv_defensive_buffer', 0.99)
        norm_buf = self.oamv_params.get('oamv_normal_buffer', 0.97)
        grace_days = self.oamv_params.get('oamv_grace_days', 2)

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

            # Update peak price for trailing stop tracking
            if close_today > pos.peak_price:
                pos.peak_price = close_today

            # --- NEW: Time stop for early losers ---
            if (not is_buy_day and days_held >= self.stop_cfg['time_stop_loser_days']
                    and cum_return < self.stop_cfg['time_stop_loser_pct']):
                to_sell.append((code, date, close_today,
                               f'time_loser(T+{days_held},{cum_return*100:+.1f}%)', 1.0))
                continue

            # --- NEW: Hard stop-loss (per-regime, with min hold days) ---
            hard_stop_pct = self.stop_cfg['hard_stop'].get(regime, -0.12)
            min_days = self.stop_cfg.get('hard_stop_min_days', 5)
            if (not is_buy_day and days_held >= min_days
                    and cum_return <= hard_stop_pct):
                to_sell.append((code, date, close_today,
                               f'hard_stop_{regime}(T+{days_held},{cum_return*100:+.1f}%)', 1.0))
                continue

            # --- NEW: Peak drawdown stop ---
            if not is_buy_day and pos.peak_price > 0:
                peak_dd = (close_today / pos.peak_price - 1)
                if peak_dd <= self.stop_cfg['peak_dd_stop']:
                    to_sell.append((code, date, close_today,
                                   f'peak_dd(T+{days_held},pk_dd{peak_dd*100:+.1f}%)', 1.0))
                    continue

            # --- NEW: Trailing stop ---
            if (not is_buy_day and cum_return >= self.stop_cfg['trailing_activate']
                    and pos.peak_price > pos.buy_price):
                trail_level = pos.peak_price * (1 - self.stop_cfg['trailing_pct'])
                if close_today < trail_level:
                    to_sell.append((code, date, close_today,
                                   f'trail(T+{days_held},pk{pos.peak_price:.2f},{cum_return*100:+.1f}%)', 1.0))
                    continue

            # --- NEW: Take profit ---
            if not is_buy_day and cum_return >= self.stop_cfg['take_profit_full']:
                to_sell.append((code, date, close_today,
                               f'take_full(T+{days_held},{cum_return*100:+.0f}%)', 1.0))
                continue

            if not is_buy_day and cum_return >= self.stop_cfg['take_profit_partial'] and not pos.half_sold:
                to_sell.append((code, date, close_today,
                               f'take_half(T+{days_held},{cum_return*100:+.0f}%)',
                               self.stop_cfg['take_profit_partial_frac']))
                continue

            # --- IMPROVED: 60-day checkpoint (tighter — sell weak performers) ---
            if not is_buy_day and days_held >= 60 and cum_return < self.stop_cfg['checkpoint_60d_min_ret']:
                to_sell.append((code, date, close_today,
                               f'60d_weak(T+{days_held},{cum_return*100:+.0f}%)', 1.0))
                continue

            # --- IMPROVED: Max hold 180 days (was 252) ---
            if not is_buy_day and days_held >= self.stop_cfg['max_hold_days']:
                to_sell.append((code, date, close_today,
                               f'max_hold(T+{days_held})', 1.0))
                continue

            # --- KEPT: Original candle/yellow/signal stops (last resort) ---
            if regime == 'aggressive':
                candle_level = pos.entry_low * agg_buf
            elif regime == 'defensive':
                candle_level = pos.entry_low * def_buf
            else:
                candle_level = pos.entry_low * norm_buf

            candle_stop = close_today < candle_level and days_held >= grace_days
            signal_stop = close_today < pos.signal_close
            yellow = float(row.get('yellow_line', 0))
            yellow_stop = yellow > 0 and close_today < yellow

            if candle_stop:
                reason = f'candle_{regime}(T+{days_held})'
            elif signal_stop:
                reason = f'sig_close(T+{days_held})'
            elif yellow_stop:
                reason = f'yellow(T+{days_held})'
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


# ============================================================
# Parameters (same as previous backtest)
# ============================================================
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

DYNAMIC_PARAMS = copy.deepcopy(B2_BASE)
DYNAMIC_PARAMS.update({
    'dynamic_thresholds': True,
    'gain_min_aggressive': 0.03, 'j_today_max_aggressive': 70, 'shadow_max_aggressive': 0.050,
    'gain_min_normal': 0.06, 'j_today_max_normal': 55, 'shadow_max_normal': 0.030,
    'gain_min_defensive': 0.07, 'j_today_max_defensive': 50, 'shadow_max_defensive': 0.020,
})

FIXED_PARAMS = copy.deepcopy(B2_BASE)
FIXED_PARAMS['dynamic_thresholds'] = False

PERIODS = [
    {'name': '10Y (2016-2026)', 'sim_start': pd.Timestamp('2016-05-01'), 'sim_end': pd.Timestamp('2026-05-07')},
    {'name': '5Y (2021-2026)',  'sim_start': pd.Timestamp('2021-05-01'), 'sim_end': pd.Timestamp('2026-05-07')},
    {'name': '3Y (2023-2026)',  'sim_start': pd.Timestamp('2023-05-01'), 'sim_end': pd.Timestamp('2026-05-07')},
]


def preload_stock_data(eligible_codes):
    """Load stock data for eligible codes."""
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
            stock_data[code] = df[cols_avail].copy()
        except Exception:
            continue
    return stock_data


def run_one_backtest(stock_data, params, sim_start, sim_end, oamv_df, label, engine_cls):
    """Run backtest with given engine class."""
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

    engine = engine_cls(stock_data, sim_params, oamv_df, oamv_p)
    metrics = engine.run()

    n_days = (sim_end - sim_start).days
    ann_factor = 365.25 / max(n_days, 1)
    cagr = ((1 + metrics['total_return']) ** ann_factor - 1) * 100 if metrics['total_return'] > -1 else None

    # Stop reason breakdown
    stop_reasons = {}
    for t in engine.trade_log:
        if t['action'] == 'SELL':
            reason = t.get('reason', 'unknown')
            base_reason = reason.split('(')[0]
            stop_reasons[base_reason] = stop_reasons.get(base_reason, 0) + 1

    print(f"  {label:<30s} "
          f"Ret={metrics['total_return_pct']:+.1f}%  "
          f"Sharpe={metrics['sharpe']:.3f}  "
          f"WR={metrics['win_rate']:.1%}  "
          f"DD={metrics['max_dd']:.1%}  "
          f"Trades={metrics['n_trades']}  "
          f"CAGR={cagr:+.1f}%" if cagr is not None else f"  {label:<30s} Ret={metrics['total_return_pct']:+.1f}%")

    # Show top stop reasons
    top_reasons = sorted(stop_reasons.items(), key=lambda x: -x[1])[:3]
    reasons_str = ', '.join(f'{r}:{n}' for r, n in top_reasons)
    print(f"    Stops: {reasons_str}")

    return {**metrics, 'cagr_pct': cagr, 'n_days': n_days,
            'sim_start': str(sim_start.date()), 'sim_end': str(sim_end.date()),
            'stop_reasons': stop_reasons}


def main():
    t0 = time.time()
    print("=" * 85)
    print("  Improved Stop-Loss Backtest — B2 Optimal Strategy")
    print("  Comparing: Original Engine vs Improved Engine (hard stop + trail + take-profit)")
    print("=" * 85)

    # 1. Load mktcap filter
    print("\n[1/4] Loading market cap lookup...")
    mktcap_lookup = build_mktcap_lookup()
    eligible_codes = set()
    for (ym, sym), mv in mktcap_lookup.items():
        if '2010-01' <= ym <= '2026-05' and 500 <= mv <= 1000:
            eligible_codes.add(sym)
    print(f"  Eligible codes (500-1000亿): {len(eligible_codes)}")

    # 2. Load OAMV
    print("\n[2/4] Loading OAMV data...")
    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    print(f"  OAMV: {len(oamv_df)} days")

    # 3. Load stock data
    print("\n[3/4] Loading stock data...")
    stock_data = preload_stock_data(eligible_codes)
    print(f"  Loaded {len(stock_data)} stocks")

    # 4. Run backtests
    print(f"\n[4/4] Running backtests: Original vs Improved...")
    print(f"  {'='*80}")

    all_results = []

    for period in PERIODS:
        print(f"\n  --- {period['name']} ---")

        # Original engine - Dynamic
        orig_dyn = run_one_backtest(
            stock_data, copy.deepcopy(DYNAMIC_PARAMS),
            period['sim_start'], period['sim_end'], oamv_df,
            f"Original_Dynamic", p2c_mod.OAMVSimEngine)

        # Improved engine - Dynamic
        impr_dyn = run_one_backtest(
            stock_data, copy.deepcopy(DYNAMIC_PARAMS),
            period['sim_start'], period['sim_end'], oamv_df,
            f"Improved_Dynamic", ImprovedOAMVSimEngine)

        # Original engine - Fixed
        orig_fix = run_one_backtest(
            stock_data, copy.deepcopy(FIXED_PARAMS),
            period['sim_start'], period['sim_end'], oamv_df,
            f"Original_Fixed", p2c_mod.OAMVSimEngine)

        # Improved engine - Fixed
        impr_fix = run_one_backtest(
            stock_data, copy.deepcopy(FIXED_PARAMS),
            period['sim_start'], period['sim_end'], oamv_df,
            f"Improved_Fixed", ImprovedOAMVSimEngine)

        all_results.append({
            'period': period['name'],
            'original_dynamic': orig_dyn,
            'improved_dynamic': impr_dyn,
            'original_fixed': orig_fix,
            'improved_fixed': impr_fix,
        })

    # 5. Summary
    print(f"\n{'='*85}")
    print(f"  Summary: Original vs Improved Stop-Loss")
    print(f"{'='*85}")
    header = (f"  {'Period':<20s} {'Engine':<22s} {'Return':>10s} {'CAGR':>8s} "
              f"{'Sharpe':>8s} {'WR':>7s} {'MaxDD':>7s} {'Trades':>7s}")
    print(header)
    print(f"  {'-'*85}")

    for r in all_results:
        for key, label, engine_type in [
            ('original_fixed', 'Original Fixed', 'original'),
            ('improved_fixed', 'Improved Fixed', 'improved'),
            ('original_dynamic', 'Original Dynamic', 'original'),
            ('improved_dynamic', 'Improved Dynamic', 'improved'),
        ]:
            m = r[key]
            cagr_str = f"{m['cagr_pct']:+.1f}%" if m.get('cagr_pct') is not None else "N/A"
            print(f"  {r['period']:<20s} {label:<22s} "
                  f"{m['total_return_pct']:>+9.1f}% {cagr_str:>8s} "
                  f"{m['sharpe']:>8.3f} {m['win_rate']:>6.1%} "
                  f"{m['max_dd']:>6.1%} {m['n_trades']:>7d}")

        # Delta row
        orig = r['original_dynamic']
        impr = r['improved_dynamic']
        dd_delta = impr['max_dd'] - orig['max_dd']
        ret_delta = impr['total_return_pct'] - orig['total_return_pct']
        print(f"  {'':20s} {'Delta (Impr-Orig)':<22s} "
              f"{ret_delta:>+9.1f}% {'':>8s} "
              f"{impr['sharpe']-orig['sharpe']:>+8.3f} {'':>7s} "
              f"{dd_delta:>+6.1%} {impr['n_trades']-orig['n_trades']:>+7d}")

    # Save
    out = {
        'phase': 'backtest_improved_stops',
        'date': '2026-05-11',
        'stop_config': STOP_CONFIG,
        'results': [{
            'period': r['period'],
            'original_dynamic': {k: v for k, v in r['original_dynamic'].items() if k != 'stop_reasons'},
            'improved_dynamic': {k: v for k, v in r['improved_dynamic'].items() if k != 'stop_reasons'},
            'original_fixed': {k: v for k, v in r['original_fixed'].items() if k != 'stop_reasons'},
            'improved_fixed': {k: v for k, v in r['improved_fixed'].items() if k != 'stop_reasons'},
        } for r in all_results],
        'elapsed_seconds': time.time() - t0,
    }
    with open(os.path.join(OUT_DIR, 'backtest_improved_stops.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Backtest Complete ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()
