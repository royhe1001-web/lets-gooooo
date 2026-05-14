#!/usr/bin/env python3
"""
Phase 2c Bull Grid Search: B2 + OAMV + Stop-Loss Integrated Optimization
=========================================================================
Multi-bull-market training (3 periods) + stop profiles embedded in grid.
4-stage pipeline: Fast Screen → Grid Backtest → Validate → Final Test.

Training: 3 Chinese A-share bull markets (each includes trend-testing pullbacks)
  - Bull 1: 2014-07-01 ~ 2015-06-12 (杠杆牛, incl. Dec 2014 10% correction)
  - Bull 2: 2020-03-23 ~ 2021-12-13 (结构牛, incl. Feb-Mar 2021 抱团瓦解)
  - Bull 3: 2024-01-01 ~ 2025-06-30 (政策牛, incl. Jun-Aug 2024 阴跌, Oct 急刹)
Validation: 2025-07-01 ~ 2025-12-31
Test:       2026-01-01 ~ 2026-05-07
"""

import os, sys, time, json, copy, pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import ML_optimization.phase2c_oamv_grid_search as p2c_mod
from ML_optimization.mktcap_utils import (
    build_mktcap_lookup, get_mktcap_universe, is_in_mktcap_range,
    compute_mktcap_percentiles, get_pct_universe, is_in_pct_range,
)
from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from strategy_spring import generate_spring_signals


FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# ============================================================
# Date Constants
# ============================================================
TRAIN_BULL_1 = (pd.Timestamp('2014-07-01'), pd.Timestamp('2015-06-12'))
TRAIN_RANGE_2016_2017 = (pd.Timestamp('2016-01-01'), pd.Timestamp('2017-12-31'))
TRAIN_BULL_2 = (pd.Timestamp('2020-03-23'), pd.Timestamp('2021-12-13'))
TRAIN_BULL_3 = (pd.Timestamp('2024-01-01'), pd.Timestamp('2025-06-30'))
TRAIN_PERIODS = [TRAIN_BULL_1, TRAIN_RANGE_2016_2017, TRAIN_BULL_2, TRAIN_BULL_3]
TRAIN_PERIOD_NAMES = ['Bull1_2014-2015', 'Range_2016-2017', 'Bull2_2020-2021', 'Bull3_2024-2025H1']

VAL_START = pd.Timestamp('2025-07-01')
VAL_END   = pd.Timestamp('2025-12-31')
TEST_START = pd.Timestamp('2026-01-01')
TEST_END   = pd.Timestamp.now().normalize()  # 默认最新, 网格搜索按需覆盖

MktCap_USE_PERCENTILE = True   # B+C: percentile-based with OAMV regime
MAIN_BOARD_PREFIXES = ('000', '002', '003', '600', '601', '603', '605')

# Sanity check: no overlap
assert TRAIN_BULL_3[1] < VAL_START, "Train Bull 3 must end before Val starts"
assert VAL_END < TEST_START, "Val must end before Test starts"

# ============================================================
# B2 + OAMV Parameter Sets (same 17 as Phase 2c)
# ============================================================
EXPLOSIVE_B2 = {
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
    'sector_momentum_enabled': False,
    'sector_weight_strong': 1.15,
    'sector_weight_weak': 0.85,
}
DEFAULT_B2 = {**EXPLOSIVE_B2, 'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
              'gain_min': 0.04, 'j_today_max': 65,
              'sector_momentum_enabled': False}
PULLBACK_B2 = {**EXPLOSIVE_B2, 'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
               'weight_pullback_thresh': -0.07, 'weight_pullback': 1.5,
               'sector_momentum_enabled': False}

DEFAULT_OAMV = {
    'oamv_aggressive_threshold': 3.0, 'oamv_defensive_threshold': -2.35,
    'oamv_aggressive_buffer': 0.95, 'oamv_defensive_buffer': 0.99,
    'oamv_normal_buffer': 0.97, 'oamv_grace_days': 2,
    'oamv_defensive_ban_entry': True,
}


def build_param_sets():
    sets = []
    sets.append({'name': 'DEFAULT+OAMV_default', 'b2': copy.deepcopy(DEFAULT_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV)})
    sets.append({'name': 'DEFAULT_no_OAMV', 'b2': copy.deepcopy(DEFAULT_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV), 'no_oamv': True})
    for agg_t, s in [(2.5, 'agg2.5'), (3.0, 'agg3.0'), (3.5, 'agg3.5')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_aggressive_threshold'] = agg_t
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    for def_t, s in [(-2.0, 'def2.0'), (-2.35, 'def2.35'), (-2.5, 'def2.5')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_defensive_threshold'] = def_t
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    for buf, s in [(0.93, 'abuf93'), (0.97, 'abuf97')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_aggressive_buffer'] = buf
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    for buf, s in [(0.97, 'dbuf97'), (1.0, 'dbuf100')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_defensive_buffer'] = buf
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    o = copy.deepcopy(DEFAULT_OAMV); o['oamv_defensive_ban_entry'] = False
    sets.append({'name': 'Explosive_noDefBan', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    sets.append({'name': 'Explosive_no_OAMV', 'b2': copy.deepcopy(EXPLOSIVE_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV), 'no_oamv': True})
    sets.append({'name': 'Pullback+OAMV', 'b2': copy.deepcopy(PULLBACK_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV)})
    o = copy.deepcopy(DEFAULT_OAMV)
    o.update({'oamv_aggressive_threshold': 2.5, 'oamv_aggressive_buffer': 0.93,
              'oamv_defensive_buffer': 0.97, 'oamv_defensive_ban_entry': False})
    sets.append({'name': 'OAMV_Aggressive', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    o = copy.deepcopy(DEFAULT_OAMV)
    o.update({'oamv_aggressive_threshold': 3.5, 'oamv_aggressive_buffer': 0.97,
              'oamv_defensive_threshold': -2.0, 'oamv_defensive_buffer': 1.0,
              'oamv_defensive_ban_entry': True})
    sets.append({'name': 'OAMV_Conservative', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    return sets


# ============================================================
# Stop Profiles
# ============================================================
FROZEN_STOP_DEFAULTS = {
    'hard_stop': {'aggressive': -0.18, 'normal': -0.10, 'defensive': -0.08},
    'hard_stop_min_days': 5,
    'take_profit_full': 0.60,
    'take_profit_partial_frac': 0.5,
    'time_stop_loser_days': 20,
    'time_stop_loser_pct': -0.05,
    'checkpoint_60d_min_ret': 0.05,
    'max_hold_days': 180,
}

STOP_PROFILES = {
    'None': None,
    'Tight': {
        'trailing_activate': 0.05, 'trailing_pct': 0.03,
        'take_profit_partial': 0.40, 'peak_dd_stop': -0.10,
        **FROZEN_STOP_DEFAULTS,
    },
    'Medium': {
        'trailing_activate': 0.08, 'trailing_pct': 0.05,
        'take_profit_partial': 0.30, 'peak_dd_stop': -0.12,
        **FROZEN_STOP_DEFAULTS,
    },
    'Loose': {
        'trailing_activate': 0.12, 'trailing_pct': 0.08,
        'take_profit_partial': 0.20, 'peak_dd_stop': -0.15,
        **FROZEN_STOP_DEFAULTS,
    },
}


# ============================================================
# StopOptimEngine (from backtest_stop_optimization.py)
# ============================================================
class StopOptimEngine(p2c_mod.OAMVSimEngine):
    """OAMVSimEngine with configurable stop-loss hierarchy."""

    def __init__(self, *args, stop_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_cfg = stop_config or FROZEN_STOP_DEFAULTS

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

            if close_today > pos.peak_price:
                pos.peak_price = close_today

            # 1. Time stop for early losers
            if (not is_buy_day and days_held >= self.stop_cfg['time_stop_loser_days']
                    and cum_return < self.stop_cfg['time_stop_loser_pct']):
                to_sell.append((code, date, close_today, 'time_loser', 1.0))
                continue

            # 2. Hard stop-loss (per-regime, with min hold days)
            hard_stop_pct = self.stop_cfg['hard_stop'].get(regime, -0.10)
            min_days = self.stop_cfg.get('hard_stop_min_days', 5)
            if (not is_buy_day and days_held >= min_days
                    and cum_return <= hard_stop_pct):
                to_sell.append((code, date, close_today, f'hard_stop_{regime}', 1.0))
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
                candle_level = pos.entry_low * norm_buf

            candle_stop = close_today < candle_level and days_held >= grace_days
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


# ============================================================
# Stage 1: Fast T+4 Screening on Multi-Period
# ============================================================
NEEDED_COLS_S1 = ['date', 'open', 'high', 'low', 'close', 'volume',
                  'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                  'white_line', 'yellow_line',
                  'brick_val', 'brick_rising', 'brick_falling',
                  'brick_green_to_red', 'brick_red_to_green',
                  'brick_green_streak', 'brick_red_streak',
                  'cross_below_yellow']


def is_in_any_period(idx, periods):
    """Boolean mask: which indices fall in any period."""
    mask = np.zeros(len(idx), dtype=bool)
    for start, end in periods:
        mask |= (idx >= start) & (idx <= end)
    return mask


def _eval_stock_chunk_bull(args):
    """Worker: evaluate ALL param sets on assigned stock chunk over bull periods."""
    stock_files, oamv_change_map, param_sets, train_periods, worker_id, sector_lookup_path = args

    try:
        mktcap_lookup = build_mktcap_lookup()
        percentiles = compute_mktcap_percentiles()
    except Exception:
        mktcap_lookup = {}
        percentiles = {}

    # Load sector data if available
    sector_lookup = None
    if sector_lookup_path and os.path.exists(sector_lookup_path):
        try:
            with open(sector_lookup_path, 'rb') as f:
                sector_lookup = pickle.load(f)
        except Exception:
            pass

    stock_data = {}
    for f in stock_files:
        code = os.path.basename(f).replace('.parquet', '')
        try:
            df = pd.read_parquet(f)
            cols_avail = [c for c in NEEDED_COLS_S1 if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            oamv_changes = [oamv_change_map.get(d, 0.0) for d in df['date']]
            df['_oamv_change_pct'] = oamv_changes
            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
        except Exception:
            continue

    results = []
    n_codes = len(stock_data)
    for pi, ps in enumerate(param_sets):
        agg_thresh = ps['oamv'].get('oamv_aggressive_threshold', 3.0)
        def_thresh = ps['oamv'].get('oamv_defensive_threshold', -2.35)
        no_oamv = ps.get('no_oamv', False)
        if no_oamv:
            agg_thresh, def_thresh = 999.0, -999.0

        b2p = copy.deepcopy(ps['b2'])

        total_w = 0.0; total_wr = 0.0; n = 0
        for ci, (code, df) in enumerate(stock_data.items()):
            try:
                dc = df.copy()
                chg = dc['_oamv_change_pct'].values
                regime = np.where(chg > agg_thresh, 'aggressive',
                         np.where(chg < def_thresh, 'defensive', 'normal'))
                dc['oamv_regime'] = regime

                sdf = generate_spring_signals(dc, board_type='main', precomputed=True, params=b2p)

                # Filter to signals within ANY training period
                bull_mask = is_in_any_period(sdf.index, train_periods)
                sig_idx = sdf.index[bull_mask & (sdf['b2_entry_signal'] == 1)]

                for i in sig_idx:
                    pos = sdf.index.get_loc(i)
                    if pos + 4 < len(sdf):
                        if mktcap_lookup and percentiles:
                            regime_i = sdf.loc[i, 'oamv_regime'] if 'oamv_regime' in sdf.columns else 'normal'
                            if not is_in_pct_range(code, i, mktcap_lookup, percentiles, regime=regime_i):
                                continue
                        buy_px = float(sdf.loc[i, 'close'])
                        sell_px = float(sdf.iloc[pos + 4]['open'])
                        ret = sell_px / buy_px - 1
                        w = float(sdf.loc[i, 'b2_position_weight'])
                        total_wr += ret * w; total_w += w; n += 1
            except Exception:
                continue
            if ci > 0 and ci % 500 == 0:
                print(f"    Param {pi+1}/{len(param_sets)}: {ci}/{n_codes} stocks done", flush=True)

        z = float(total_wr / total_w) if total_w > 0 else 0.0
        results.append({'name': ps['name'], 'z': z, 'n_signals': n,
                        'total_w': total_w, 'total_wr': total_wr})

    return results


def stage1_fast_screen(stock_files, oamv_df, param_sets, train_periods,
                       n_workers=0):
    """Fast T+5 screening on concatenated bull periods.
    If n_workers=0, runs sequentially in main process (safer, less memory).
    If n_workers>=2, uses ProcessPoolExecutor (faster but may OOM on Windows)."""
    oamv_change_map = {}
    for _, row in oamv_df.iterrows():
        d = row['date']
        if hasattr(d, 'date'): d = d.date()
        oamv_change_map[pd.Timestamp(d)] = float(row['oamv_change_pct'])

    if n_workers <= 1:
        # Sequential mode — single process, no OOM risk
        print(f"  Stocks: {len(stock_files)} main board, sequential mode")
        print(f"  Train periods: {[(s.date(), e.date()) for s, e in train_periods]}")
        print(f"  Param sets: {len(param_sets)}", flush=True)
        t0 = time.time()
        all_results = _eval_stock_chunk_bull((stock_files, oamv_change_map, param_sets, train_periods, 0, sector_lookup_path))
        elapsed = time.time() - t0
        print(f"  Sequential done ({elapsed:.0f}s, {len(all_results)} results)", flush=True)
    else:
        # Parallel mode
        chunk_size = max(1, len(stock_files) // n_workers)
        chunks = [stock_files[i:i+chunk_size] for i in range(0, len(stock_files), chunk_size)]
        print(f"  Stocks: {len(stock_files)} main board, {n_workers} workers ({len(chunks)} chunks)")
        print(f"  Train periods: {[(s.date(), e.date()) for s, e in train_periods]}")
        print(f"  Param sets: {len(param_sets)}", flush=True)
        t0 = time.time()
        all_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            args_list = [(chunk, oamv_change_map, param_sets, train_periods, i, sector_lookup_path)
                         for i, chunk in enumerate(chunks)]
            futures = {executor.submit(_eval_stock_chunk_bull, args): i
                       for i, args in enumerate(args_list)}
            for f in as_completed(futures):
                worker_id = futures[f]
                try:
                    wr = f.result()
                    all_results.extend(wr)
                    elapsed = time.time() - t0
                    print(f"  Worker {worker_id+1}/{len(chunks)} done ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  Worker {worker_id+1} ERROR: {e}", flush=True)

    merged = {}
    for r in all_results:
        name = r['name']
        if name not in merged:
            merged[name] = {'total_w': 0.0, 'total_wr': 0.0, 'n_signals': 0}
        merged[name]['total_w'] += r['total_w']
        merged[name]['total_wr'] += r['total_wr']
        merged[name]['n_signals'] += r['n_signals']

    stage1_results = []
    for name, m in merged.items():
        z = float(m['total_wr'] / m['total_w']) if m['total_w'] > 0 else 0.0
        stage1_results.append({'name': name, 'z': z, 'n_signals': m['n_signals']})

    stage1_results.sort(key=lambda r: r['z'], reverse=True)
    return stage1_results


# ============================================================
# Stage 2: Full Backtest Grid on Multi-Period Training
# ============================================================
NEEDED_COLS_S2 = ['date', 'open', 'high', 'low', 'close', 'volume',
                  'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                  'white_line', 'yellow_line',
                  'brick_val', 'brick_rising', 'brick_falling',
                  'brick_green_to_red', 'brick_red_to_green',
                  'brick_green_streak', 'brick_red_streak',
                  'cross_below_yellow']


def _get_stock_files_mainboard():
    """Get all mainboard stock file paths."""
    files = []
    for f in sorted(os.listdir(FEAT_DIR)):
        if not f.endswith('.parquet'):
            continue
        code = f.replace('.parquet', '')
        if any(code.startswith(p) for p in MAIN_BOARD_PREFIXES):
            files.append(os.path.join(FEAT_DIR, f))
    return files


def preload_stock_data(eligible_codes):
    """Load stock data for backtest (single process)."""
    stock_data = {}
    loaded = 0
    for f in _get_stock_files_mainboard():
        code = os.path.basename(f).replace('.parquet', '')
        if code not in eligible_codes:
            continue
        try:
            df = pd.read_parquet(f)
            cols_avail = [c for c in NEEDED_COLS_S2 if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
                loaded += 1
                if loaded % 100 == 0:
                    print(f"  Loaded {loaded} stocks...")
        except Exception:
            continue
    print(f"  Loaded {loaded} stocks total")
    return stock_data


def _prepare_oamv_and_b2_params(param_set):
    """Extract b2_params and oamv_params from a param_set dict."""
    b2 = copy.deepcopy(param_set['b2'])
    oamv = copy.deepcopy(param_set['oamv'])
    no_oamv = param_set.get('no_oamv', False)
    if no_oamv:
        oamv['oamv_aggressive_threshold'] = 999.0
        oamv['oamv_defensive_threshold'] = -999.0
    return b2, oamv


def run_one_period(stock_data, b2_params, oamv_params, oamv_df,
                   period_start, period_end, stop_config,
                   mktcap_lookup=None, percentiles=None,
                   sector_lookup=None):
    """Run backtest on a single contiguous period."""
    p2c_mod.SIM_START = period_start
    p2c_mod.SIM_END = period_end

    if stop_config is not None:
        engine = StopOptimEngine(stock_data, b2_params, oamv_df, oamv_params,
                                 stop_config=stop_config,
                                 mktcap_lookup=mktcap_lookup,
                                 percentiles=percentiles,
                                 sector_lookup=sector_lookup)
    else:
        engine = p2c_mod.OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params,
                                       mktcap_lookup=mktcap_lookup,
                                       percentiles=percentiles,
                                       sector_lookup=sector_lookup)

    metrics = engine.run()
    metrics['n_days'] = max(1, len([d for d in pd.date_range(period_start, period_end)
                                    if d.weekday() < 5]))  # approximate trading days

    # Stop reason breakdown
    stop_counts = {}
    for t in engine.trade_log:
        if t['action'] == 'SELL':
            reason = t.get('reason', 'unknown')
            base = reason.split('_')[0] if '_' in reason else reason.split('(')[0]
            stop_counts[base] = stop_counts.get(base, 0) + 1
    metrics['stop_counts'] = stop_counts

    return metrics


def aggregate_multi_period(period_results):
    """Aggregate results from multiple sub-period backtests."""
    if not period_results:
        return None

    weights = [np.sqrt(r['n_days']) for r in period_results]
    total_weight = sum(weights)
    combined_sharpe = (sum(r['sharpe'] * w for r, w in zip(period_results, weights)) / total_weight
                       if total_weight > 0 else 0.0)

    total_return = sum(r['total_return'] for r in period_results)
    total_return_pct = total_return * 100
    win_rate = np.mean([r['win_rate'] for r in period_results])
    max_dd = max(r['max_dd'] for r in period_results)
    n_trades = sum(r['n_trades'] for r in period_results)

    # Merge stop counts
    all_stop_counts = {}
    for r in period_results:
        for k, v in r.get('stop_counts', {}).items():
            all_stop_counts[k] = all_stop_counts.get(k, 0) + v

    return {
        'sharpe': combined_sharpe,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'stop_counts': all_stop_counts,
        'per_period': period_results,
    }


def _run_grid_job(args_dict):
    """Subprocess entry for one (B2, Stop) combo across all training periods."""
    import os, sys, json, copy, pickle
    import pandas as pd
    import importlib.util

    BASE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE2)

    p2c_path = os.path.join(BASE2, 'ML_optimization', 'phase2c_oamv_grid_search.py')
    spec = importlib.util.spec_from_file_location('p2c_engine', p2c_path)
    p2c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2c)

    # Rebuild the StopOptimEngine in subprocess context
    class StopEng(p2c.OAMVSimEngine):
        def __init__(self, *args2, stop_config=None, **kwargs2):
            super().__init__(*args2, **kwargs2)
            self.stop_cfg = stop_config or FROZEN_STOP_DEFAULTS

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

                if close_today > pos.peak_price:
                    pos.peak_price = close_today

                # 1. Time stop
                if (not is_buy_day and days_held >= self.stop_cfg['time_stop_loser_days']
                        and cum_return < self.stop_cfg['time_stop_loser_pct']):
                    to_sell.append((code, date, close_today, 'time_loser', 1.0))
                    continue
                # 2. Hard stop
                hard_stop_pct = self.stop_cfg['hard_stop'].get(regime, -0.10)
                min_days = self.stop_cfg.get('hard_stop_min_days', 5)
                if (not is_buy_day and days_held >= min_days and cum_return <= hard_stop_pct):
                    to_sell.append((code, date, close_today, f'hard_stop_{regime}', 1.0))
                    continue
                # 3. Peak DD
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
                # 8. Original stops
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

    from ML_optimization.mktcap_utils import (build_mktcap_lookup, get_mktcap_universe,
        compute_mktcap_percentiles)

    # Load stock data
    data = json.loads(args_dict['stock_data_json'])
    stock_data = {}
    for code, fpath in data.items():
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_parquet(fpath)
            cols_avail = [c for c in NEEDED_COLS_S2 if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
        except Exception:
            continue

    oamv_df = pd.read_pickle(args_dict['oamv_path'])

    b2_params = json.loads(args_dict['b2_json'])
    oamv_params = json.loads(args_dict['oamv_json'])
    stop_config = json.loads(args_dict['stop_config_json']) if args_dict['stop_config_json'] else None
    periods = json.loads(args_dict['periods_json'])

    # Load sector data if provided
    sector_lookup = None
    sector_lookup_path = args_dict.get('sector_lookup_path')
    if sector_lookup_path and os.path.exists(sector_lookup_path):
        try:
            with open(sector_lookup_path, 'rb') as f:
                sector_lookup = pickle.load(f)
        except Exception:
            pass

    # Load mktcap data for percentile filtering
    try:
        mktcap_lookup = build_mktcap_lookup()
        percentiles = compute_mktcap_percentiles()
    except Exception:
        mktcap_lookup = {}
        percentiles = {}

    period_results = []
    for p_start_str, p_end_str in periods:
        p_start = pd.Timestamp(p_start_str)
        p_end = pd.Timestamp(p_end_str)
        p2c.SIM_START = p_start
        p2c.SIM_END = p_end

        if stop_config is not None:
            engine = StopEng(stock_data, b2_params, oamv_df, oamv_params,
                           stop_config=stop_config,
                           mktcap_lookup=mktcap_lookup,
                           percentiles=percentiles,
                           sector_lookup=sector_lookup)
        else:
            engine = p2c.OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params,
                                       mktcap_lookup=mktcap_lookup,
                                       percentiles=percentiles,
                                       sector_lookup=sector_lookup)

        metrics = engine.run()
        metrics['n_days'] = max(1, len(pd.date_range(p_start, p_end, freq='B')))

        stop_counts = {}
        for t in engine.trade_log:
            if t['action'] == 'SELL':
                reason = t.get('reason', 'unknown')
                base = reason.split('_')[0] if '_' in reason else reason.split('(')[0]
                stop_counts[base] = stop_counts.get(base, 0) + 1
        metrics['stop_counts'] = stop_counts
        period_results.append(metrics)

    # Aggregate
    weights = [np.sqrt(r['n_days']) for r in period_results]
    total_weight = sum(weights)
    combined_sharpe = (sum(r['sharpe'] * w for r, w in zip(period_results, weights)) / total_weight
                       if total_weight > 0 else 0.0)
    total_return = sum(r['total_return'] for r in period_results)
    win_rate = float(np.mean([r['win_rate'] for r in period_results]))
    max_dd = float(max(r['max_dd'] for r in period_results))
    n_trades = sum(r['n_trades'] for r in period_results)

    all_stop_counts = {}
    for r in period_results:
        for k, v in r.get('stop_counts', {}).items():
            all_stop_counts[k] = all_stop_counts.get(k, 0) + v

    return {
        'b2_name': args_dict['b2_name'],
        'stop_profile': args_dict['stop_profile'],
        'sharpe': combined_sharpe,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'stop_counts': all_stop_counts,
        'per_period': period_results,
    }


def stage2_train_grid(param_sets, stock_data_paths, oamv_df_path,
                      train_periods, n_workers=2):
    """Full backtest grid: N B2 combos across training periods (OAMVSimEngine only)."""
    combos = []
    for ps in param_sets:
        b2, oamv = _prepare_oamv_and_b2_params(ps)
        combos.append({
            'b2_name': ps['name'],
            'stop_profile': 'None',
            'b2_json': json.dumps(b2),
            'oamv_json': json.dumps(oamv),
            'stop_config_json': None,
        })

    periods_json = json.dumps([(str(s), str(e)) for s, e in train_periods])
    stock_data_json = json.dumps(stock_data_paths)

    print(f"  Grid: {len(param_sets)} B2 combos (OAMVSimEngine)")
    print(f"  Periods: {len(train_periods)} → {len(combos) * len(train_periods)} backtests")
    print(f"  Workers: {n_workers}", flush=True)

    t0 = time.time()
    results = []

    if n_workers <= 1:
        for i, combo in enumerate(combos):
            args_dict = {
                **combo,
                'stock_data_json': stock_data_json,
                'oamv_path': oamv_df_path,
                'periods_json': periods_json,
            }
            try:
                r = _run_grid_job(args_dict)
                results.append(r)
                done = len(results)
                elapsed = time.time() - t0
                est_total = elapsed / done * len(combos) if done > 0 else 0
                remaining = est_total - elapsed
                print(f"  [{done}/{len(combos)}] {r['b2_name']} "
                      f"Sharpe={r['sharpe']:.3f} Ret={r['total_return_pct']:+.1f}% "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)
            except Exception as e:
                print(f"  Combo {i} ERROR: {e}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for i, combo in enumerate(combos):
                args_dict = {
                    **combo,
                    'stock_data_json': stock_data_json,
                    'oamv_path': oamv_df_path,
                    'periods_json': periods_json,
                }
                futures[executor.submit(_run_grid_job, args_dict)] = i

            for f in as_completed(futures):
                idx = futures[f]
                try:
                    r = f.result()
                    results.append(r)
                    done = len(results)
                    elapsed = time.time() - t0
                    est_total = elapsed / done * len(combos) if done > 0 else 0
                    remaining = est_total - elapsed
                    print(f"  [{done}/{len(combos)}] {r['b2_name']} "
                          f"Sharpe={r['sharpe']:.3f} Ret={r['total_return_pct']:+.1f}% "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)
                except Exception as e:
                    print(f"  Combo {idx} ERROR: {e}", flush=True)

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    return results


# ============================================================
# Stages 3-4: Validation and Test
# ============================================================
def run_val_or_test(stock_data, b2_params, oamv_params, oamv_df,
                    start, end, stop_config, label,
                    mktcap_lookup=None, percentiles=None,
                    sector_lookup=None):
    """Run a single backtest on Val or Test period."""
    p2c_mod.SIM_START = start
    p2c_mod.SIM_END = end

    if stop_config is not None:
        engine = StopOptimEngine(stock_data, b2_params, oamv_df, oamv_params,
                                 stop_config=stop_config,
                                 mktcap_lookup=mktcap_lookup,
                                 percentiles=percentiles,
                                 sector_lookup=sector_lookup)
    else:
        engine = p2c_mod.OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params,
                                       mktcap_lookup=mktcap_lookup,
                                       percentiles=percentiles,
                                       sector_lookup=sector_lookup)

    metrics = engine.run()

    stop_counts = {}
    for t in engine.trade_log:
        if t['action'] == 'SELL':
            reason = t.get('reason', 'unknown')
            base = reason.split('_')[0] if '_' in reason else reason.split('(')[0]
            stop_counts[base] = stop_counts.get(base, 0) + 1
    metrics['stop_counts'] = stop_counts
    metrics['label'] = label

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  Phase 2c Bull Grid Search — B2 × Stop Profile Integrated Optimization")
    print("=" * 80)
    print(f"  Train: {len(TRAIN_PERIODS)} bull markets "
          f"{[(s.date(), e.date()) for s, e in TRAIN_PERIODS]}")
    print(f"  Val:   {VAL_START.date()} ~ {VAL_END.date()}")
    print(f"  Test:  {TEST_START.date()} ~ {TEST_END.date()}")
    print(f"  MktCap: percentile-based (B+C), OAMV regime-aware")

    # ---- Load shared data ----
    print(f"\n[0/4] Loading OAMV + market cap + stock list...")
    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    print(f"  OAMV: {len(oamv_df)} days")

    mktcap_lookup = build_mktcap_lookup()
    percentiles = compute_mktcap_percentiles()

    # Load sector momentum data
    print("  Loading sector momentum data...")
    sector_lookup = None

    # Get stock files
    all_files = _get_stock_files_mainboard()
    print(f"  Main board stocks: {len(all_files)}")

    # Build eligible universe using percentile-based filtering (widest = aggressive)
    eligible_codes = set()
    for s, e in TRAIN_PERIODS:
        ym_s = s.strftime('%Y-%m')
        ym_e = e.strftime('%Y-%m')
        codes = get_pct_universe(mktcap_lookup, percentiles, ym_s, ym_end=ym_e, regime='aggressive')
        eligible_codes.update(codes)
    # Also add Val + Test period codes
    codes_val = get_pct_universe(mktcap_lookup, percentiles, '2025-07', ym_end='2026-05', regime='aggressive')
    eligible_codes.update(codes_val)
    print(f"  Eligible codes (percentile, aggressive prefilter): {len(eligible_codes)}")

    # ---- Preload stock data ----
    print("  Preloading stock data...")
    stock_data = preload_stock_data(eligible_codes)

    # Save OAMV to pickle for subprocess reuse
    oamv_tmp = os.path.join(tempfile.gettempdir(), 'bull_grid_oamv.pkl')
    oamv_df.to_pickle(oamv_tmp)

    # ---- Stage 1: Fast Screening ----
    print(f"\n[1/4] Stage 1: Fast T+4 Screening on Bull Periods")
    print("-" * 60)
    param_sets = build_param_sets()
    stage1_results = stage1_fast_screen(all_files, oamv_df, param_sets, TRAIN_PERIODS,
                                        n_workers=2)
    top5 = stage1_results[:5]

    print(f"\n  Top-5 B2 combos:")
    for i, r in enumerate(top5):
        print(f"  {i+1}. {r['name']:<28s} z={r['z']:+.6f}  signals={r['n_signals']}")

    # ---- Stage 2: Full Backtest Grid ----
    print(f"\n[2/4] Stage 2: Full Backtest Grid ({len(top5)} B2 combos, OAMVSimEngine)")
    print("-" * 60)

    # Build stock_data_paths dict for subprocess use
    stock_data_paths = {}
    for f in all_files:
        code = os.path.basename(f).replace('.parquet', '')
        if code in eligible_codes:
            stock_data_paths[code] = f

    # Build top-5 param sets list
    top5_names = {r['name'] for r in top5}
    top5_param_sets = [ps for ps in param_sets if ps['name'] in top5_names]

    grid_results = stage2_train_grid(top5_param_sets, stock_data_paths,
                                     oamv_tmp, TRAIN_PERIODS, n_workers=2)

    print(f"\n  Top-5 Train (Aggregate) Results (OAMVSimEngine):")
    for i, r in enumerate(grid_results[:5]):
        print(f"  {i+1}. {r['b2_name']:<35s} "
              f"Sharpe={r['sharpe']:.3f}  Ret={r['total_return_pct']:+.1f}%  "
              f"MaxDD={r['max_dd']:.1%}  Trades={r['n_trades']}")

    # ---- Stage 3: Validation ----
    print(f"\n[3/4] Stage 3: Validate Top-3 on Val 2025H2")
    print("-" * 60)

    top3 = grid_results[:3]
    val_results = []
    for r in top3:
        ps = next(ps for ps in param_sets if ps['name'] == r['b2_name'])
        b2, oamv = _prepare_oamv_and_b2_params(ps)
        m = run_val_or_test(stock_data, b2, oamv, oamv_df,
                           VAL_START, VAL_END, None,
                           r['b2_name'],
                           mktcap_lookup=mktcap_lookup,
                           percentiles=percentiles,
                           sector_lookup=sector_lookup)
        m['b2_name'] = r['b2_name']
        m['stop_profile'] = 'None'
        val_results.append(m)
        print(f"  {r['b2_name']:<35s} "
              f"Sharpe={m['sharpe']:.3f}  Ret={m['total_return_pct']:+.1f}%  "
              f"MaxDD={m['max_dd']:.1%}  Trades={m['n_trades']}")

    val_results.sort(key=lambda r: r['sharpe'], reverse=True)
    best_val = val_results[0]
    print(f"\n  Best on Val: {best_val['b2_name']} "
          f"(Sharpe={best_val['sharpe']:.3f})")

    # ---- Stage 4: Final Test ----
    print(f"\n[4/4] Stage 4: Final Test on 2026 (OAMVSimEngine)")
    print("-" * 60)

    test_results = []
    # Test top-3 best val configs on Test 2026
    for vr in val_results[:3]:
        ps = next(ps for ps in param_sets if ps['name'] == vr['b2_name'])
        b2, oamv = _prepare_oamv_and_b2_params(ps)
        m = run_val_or_test(stock_data, b2, oamv, oamv_df,
                            TEST_START, TEST_END, None,
                            vr['b2_name'],
                            mktcap_lookup=mktcap_lookup,
                            percentiles=percentiles,
                            sector_lookup=sector_lookup)
        m['b2_name'] = vr['b2_name']
        test_results.append(m)

    print(f"\n  {'Config':<35s} {'Return':>10s} {'Sharpe':>8s} {'WR':>8s} {'MaxDD':>8s} {'Trades':>8s}")
    print(f"  {'-'*80}")
    for m in test_results:
        print(f"  {m['label']:<35s} {m['total_return_pct']:>+9.1f}% {m['sharpe']:>8.3f} "
              f"{m['win_rate']:>7.1%} {m['max_dd']:>7.1%} {m['n_trades']:>8d}")

    # ---- Save ----
    elapsed = time.time() - t0
    out = {
        'phase': '2c_bull_grid_search',
        'engine': 'OAMVSimEngine',
        'date_ranges': {
            'train': [{'name': TRAIN_PERIOD_NAMES[i], 'start': str(s.date()), 'end': str(e.date())}
                      for i, (s, e) in enumerate(TRAIN_PERIODS)],
            'val': f'{VAL_START.date()} ~ {VAL_END.date()}',
            'test': f'{TEST_START.date()} ~ {TEST_END.date()}',
        },
        'stage1_top5': stage1_results[:5],
        'stage2_grid_top10': [{k: v for k, v in r.items() if k != 'per_period'}
                              for r in grid_results[:10]],
        'stage3_val_results': val_results,
        'stage4_test_results': [{k: v for k, v in r.items() if k != 'total_return'}
                               for r in test_results],
        'best_config': {
            'b2_name': best_val['b2_name'],
        },
        'elapsed_seconds': elapsed,
    }
    out_path = os.path.join(OUT_DIR, 'phase2c_bull_grid_search.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Complete ({elapsed:.0f}s)")
    print(f"  Results: {out_path}")

    # Cleanup
    try:
        os.remove(oamv_tmp)
    except Exception:
        pass
    if sector_tmp:
        try:
            os.remove(sector_tmp)
        except Exception:
            pass


if __name__ == '__main__':
    main()
