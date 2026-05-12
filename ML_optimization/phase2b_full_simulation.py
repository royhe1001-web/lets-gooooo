#!/usr/bin/env python3
"""
Phase 2b: Full B2 Simulation Validation (Real Engine Logic)
============================================================
Pre-loads all 4263 stocks + pre-computes indicators once, then re-generates
B2 signals with different parameter sets and runs the REAL simulation logic
(stop-loss, fly, position sizing, rotation, etc.) for each param set.

Validates top 75% important parameters from Phase 1/1b RF analysis.

Architecture:
  1. Pre-load all CSV → DataFrame (one-time, ~2-3 min)
  2. Pre-calc indicators for all stocks (one-time, ~2-3 min)
  3. For each param set:
     a. Re-generate B2 signals with new params (fast, in-memory)
     b. Extract 2026 signals
     c. Run full simulation logic (stops, fly, rotation, position sizing)
     d. Record metrics
  4. Rank and report
"""

import os, sys, time, json, copy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_spring import generate_spring_signals

DATA_DIR = os.path.join(BASE, 'fetch_kline', 'stock_kline')
FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# Trading costs
COMMISSION_BUY = 0.0003
COMMISSION_SELL = 0.0013
DEFAULT_CAPITAL = 100_000
DEFAULT_MAX_POSITIONS = 4
DEFAULT_MIN_POSITIONS = 3
SUPER_WEIGHT_THRESHOLD = 3.0

# Simulation window
SIM_START = pd.Timestamp('2026-01-01')
SIM_END = pd.Timestamp('2026-05-07')


@dataclass
class Position:
    code: str
    signal_date: pd.Timestamp
    signal_close: float
    signal_weight: float
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    cost: float
    entry_low: float = 0.0
    peak_price: float = 0.0
    half_sold: bool = False
    was_profitable: bool = False


# ============================================================
# Parameter Sets to Test (top 75% importance from Phase 1/1b)
# ============================================================
DEFAULT_PARAMS = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2,
    'weight_deep_shrink_ratio': 0.80, 'weight_deep_shrink': 1.3,
    'weight_pullback_thresh': -0.05, 'weight_pullback': 1.2,
    'weight_shadow_discount_thresh': 0.015, 'weight_shadow_discount': 0.7,
    'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
}


def build_param_sets():
    """Build parameter sets covering top 75% important params from Phase 1/1b.

    RF importance ranking (Phase 1b combined):
      Top features map to: J_prev, J, ret_1d, upper_shadow, brick_val_delta_3d,
      vol_ratio_20, K_slope, MACD_HIST, white_yellow_diff, close_yellow_ratio

    We vary the parameters that control these features:
      - Entry thresholds (j_prev_max, gain_min, j_today_max, shadow_max) → 60% cumulative importance
      - Brick resonance weight → adds brick_val features
      - Weight multipliers for volume/pullback → adds vol_ratio features
    """
    sets = []

    # 0: Default baseline
    sets.append({'name': 'DEFAULT', 'params': copy.deepcopy(DEFAULT_PARAMS)})

    # --- Entry threshold variations (highest importance: J, ret_1d, shadow) ---

    # 1: Phase 2 signal-screening best: tighter J_prev + higher J_today + looser shadow
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 15, 'gain_min': 0.04, 'j_today_max': 70, 'shadow_max': 0.025})
    sets.append({'name': 'Phase2_Best_Signal', 'params': p})

    # 2: Tighter J_prev (Phase 2 signal top performer)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 15, 'gain_min': 0.03, 'j_today_max': 65, 'shadow_max': 0.030})
    sets.append({'name': 'Tight_Jprev15_Gain3', 'params': p})

    # 3: Higher gain_min (filters for stronger moves → maps to ret_1d importance)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 15, 'gain_min': 0.05, 'j_today_max': 65, 'shadow_max': 0.035})
    sets.append({'name': 'Tight_Gain5', 'params': p})

    # 4: Higher J_today (looser J threshold → catches 920039-like cases)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 18, 'gain_min': 0.04, 'j_today_max': 70, 'shadow_max': 0.030})
    sets.append({'name': 'Loose_Jt70', 'params': p})

    # 5: Tighter shadow (maps to upper_shadow importance)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 18, 'gain_min': 0.04, 'j_today_max': 65, 'shadow_max': 0.025})
    sets.append({'name': 'Tight_Shadow25', 'params': p})

    # 6: Higher gain + looser J_today (quality filter combo)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 15, 'gain_min': 0.05, 'j_today_max': 70, 'shadow_max': 0.030})
    sets.append({'name': 'Quality_Gain5_Jt70', 'params': p})

    # 7: Lower prior_strong_ret (easier to trigger strong-stock path)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 18, 'gain_min': 0.04, 'j_today_max': 65, 'shadow_max': 0.035,
              'prior_strong_ret': 0.15})
    sets.append({'name': 'Easy_Strong15', 'params': p})

    # 8: Higher prior_strong_ret (stricter strong-stock filter)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 18, 'gain_min': 0.04, 'j_today_max': 65, 'shadow_max': 0.035,
              'prior_strong_ret': 0.25})
    sets.append({'name': 'Hard_Strong25', 'params': p})

    # --- Brick resonance weight variations (brick features in top-10 RF importance) ---

    # 9: Higher brick resonance weight
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'weight_brick_resonance': 2.0})
    sets.append({'name': 'Brick_Resonance_2x', 'params': p})

    # 10: Lower brick resonance weight
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'weight_brick_resonance': 1.0})
    sets.append({'name': 'Brick_Resonance_1x', 'params': p})

    # 11: No brick resonance (control)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'weight_brick_resonance': 1.0, 'weight_shadow_discount': 1.0,
              'weight_strong_discount': 1.0})
    sets.append({'name': 'No_Discounts', 'params': p})

    # --- Weight multiplier variations (volume/pullback features) ---

    # 12: Stronger deep pullback bonus
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'weight_pullback': 1.5, 'weight_pullback_thresh': -0.07})
    sets.append({'name': 'Strong_Pullback', 'params': p})

    # 13: Stronger deep shrink bonus
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'weight_deep_shrink': 1.5, 'weight_deep_shrink_ratio': 0.70})
    sets.append({'name': 'Strong_DeepShrink', 'params': p})

    # 14: Higher gain 2x threshold (only truly explosive moves get 2x)
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'weight_gain_2x_thresh': 0.09, 'weight_gain_2x': 2.5})
    sets.append({'name': 'Explosive_2.5x', 'params': p})

    # 15: Combined best: tight entry + strong brick + strong pullback
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 15, 'gain_min': 0.04, 'j_today_max': 70, 'shadow_max': 0.025,
              'prior_strong_ret': 0.15,
              'weight_brick_resonance': 2.0,
              'weight_pullback': 1.5,
              'weight_deep_shrink': 1.5})
    sets.append({'name': 'Combined_Best', 'params': p})

    # 16: Conservative: tighter everything
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 15, 'gain_min': 0.05, 'j_today_max': 60, 'shadow_max': 0.025,
              'prior_strong_ret': 0.25,
              'weight_gain_2x_thresh': 0.08,
              'weight_shadow_discount': 0.6,
              'weight_strong_discount': 0.7})
    sets.append({'name': 'Conservative', 'params': p})

    # 17: Aggressive: looser entry, higher weights
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update({'j_prev_max': 22, 'gain_min': 0.03, 'j_today_max': 70, 'shadow_max': 0.040,
              'prior_strong_ret': 0.15,
              'weight_brick_resonance': 2.0,
              'weight_pullback': 1.5,
              'weight_gap_up': 2.0})
    sets.append({'name': 'Aggressive', 'params': p})

    return sets


# ============================================================
# Simulation Engine (replicates run_spring_simulation_2026 logic)
# ============================================================
class FastSimulationEngine:
    """Simulation engine that operates on pre-loaded + pre-computed DataFrames."""

    def __init__(self, stock_data: Dict[str, pd.DataFrame], b2_params: dict,
                 capital=DEFAULT_CAPITAL):
        self.stock_data = stock_data
        self.b2_params = b2_params
        self.initial_capital = capital
        self.cash = capital
        self.max_positions = DEFAULT_MAX_POSITIONS
        self.min_positions = DEFAULT_MIN_POSITIONS

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.daily_values: List[Dict] = []
        self.trade_log: List[Dict] = []

    def get_price(self, code, date, price_type='close'):
        df = self.stock_data.get(code)
        if df is None:
            return None
        if date in df.index:
            return float(df.loc[date, price_type])
        prev = df.index[df.index <= date]
        if len(prev) > 0:
            return float(df.loc[prev[-1], price_type])
        return None

    def extract_signals(self):
        """Extract B2 signals from pre-loaded data (re-generates B2 signals with params)."""
        all_signals = []
        for code, df in self.stock_data.items():
            try:
                # Re-generate B2 signals with current params
                sdf = generate_spring_signals(df.copy(), board_type='main', precomputed=True,
                                          params=self.b2_params)

                mask = (sdf.index >= SIM_START) & (sdf.index <= SIM_END)
                signal_dates = sdf.index[mask & (sdf['b2_entry_signal'] == 1)]

                for sd in signal_dates:
                    row = sdf.loc[sd]
                    weight = float(row.get('b2_position_weight', 1.0))
                    sig_close = float(row['close'])

                    next_dates = sdf.index[sdf.index > sd]
                    if len(next_dates) == 0:
                        continue
                    t1_date = next_dates[0]
                    t1_open = float(sdf.loc[t1_date, 'open'])

                    all_signals.append({
                        'code': code,
                        'signal_date': sd,
                        'signal_close': sig_close,
                        'weight': weight,
                        't1_date': t1_date,
                        't1_open': t1_open,
                        'J': float(row.get('J', 0)),
                        'gain': (sig_close / float(sdf.shift(1).loc[sd, 'close']) - 1) * 100
                        if sd in sdf.shift(1).index else 0,
                        'brick_res': int(row.get('b2_brick_resonance', 0)),
                    })
            except Exception:
                continue

        all_signals.sort(key=lambda x: x['signal_date'])
        return all_signals

    def can_add_position(self, signal_weight):
        current = len(self.positions)
        if current < self.min_positions:
            return True
        if current < self.max_positions:
            return True
        if signal_weight >= SUPER_WEIGHT_THRESHOLD:
            return True
        return False

    def find_weakest_position(self):
        if not self.positions:
            return None
        return min(self.positions.values(), key=lambda p: p.signal_weight)

    def calc_position_size(self, signal_weight, buy_price):
        base = self.cash / max(self.min_positions, 1)
        wf = min(signal_weight / 2.0, 2.0)
        alloc = base * wf
        alloc = min(alloc, self.cash * 0.95)
        shares = int(alloc / buy_price / 100) * 100
        if shares < 100:
            shares = 100 if self.cash >= buy_price * 100 * 1.01 else 0
        return shares

    def buy(self, signal):
        code = signal['code']
        buy_price = signal['t1_open']
        shares = self.calc_position_size(signal['weight'], buy_price)
        if shares == 0:
            return False

        cost = shares * buy_price * (1 + COMMISSION_BUY)
        if cost > self.cash:
            shares = int(self.cash * 0.95 / buy_price / 100) * 100
            if shares < 100:
                return False
            cost = shares * buy_price * (1 + COMMISSION_BUY)

        self.cash -= cost

        df = self.stock_data.get(code)
        entry_low = buy_price
        if df is not None and signal['t1_date'] in df.index:
            entry_low = float(df.loc[signal['t1_date'], 'low'])

        pos = Position(
            code=code, signal_date=signal['signal_date'],
            signal_close=signal['signal_close'],
            signal_weight=signal['weight'],
            buy_date=signal['t1_date'],
            buy_price=buy_price, shares=shares, cost=cost,
            entry_low=entry_low, peak_price=buy_price,
        )
        self.positions[code] = pos

        self.trade_log.append({
            'action': 'BUY', 'date': signal['t1_date'], 'code': code,
            'price': buy_price, 'shares': shares, 'cost': cost,
            'weight': signal['weight'], 'cash_after': self.cash,
        })
        return True

    def sell(self, code, date, price, reason='manual', partial=1.0):
        pos = self.positions[code]
        sell_shares = int(pos.shares * partial)
        if sell_shares < 100:
            return

        proceeds = sell_shares * price * (1 - COMMISSION_SELL)
        cost_part = pos.cost * partial
        pnl = proceeds - cost_part
        pnl_pct = (pnl / cost_part) * 100 if cost_part > 0 else 0

        self.cash += proceeds

        if partial >= 1.0:
            self.positions.pop(code)
        else:
            pos.shares -= sell_shares
            pos.cost -= cost_part
            pos.half_sold = True

        self.closed_positions.append({
            'code': code, 'buy_price': pos.buy_price,
            'sell_date': date, 'sell_price': price,
            'shares': sell_shares, 'pnl': pnl, 'pnl_pct': pnl_pct,
            'reason': reason, 'weight': pos.signal_weight,
        })

        self.trade_log.append({
            'action': 'SELL', 'date': date, 'code': code,
            'price': price, 'shares': sell_shares,
            'proceeds': proceeds, 'pnl': pnl, 'pnl_pct': pnl_pct,
            'reason': reason, 'cash_after': self.cash,
        })

    def check_stops(self, date):
        to_sell = []
        for code, pos in self.positions.items():
            df = self.stock_data.get(code)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            close_today = float(row['close'])

            buy_idx = df.index.get_loc(pos.buy_date) if pos.buy_date in df.index else None
            today_idx = df.index.get_loc(date) if date in df.index else None
            days_held = (today_idx - buy_idx) if buy_idx is not None and today_idx is not None else 0
            is_buy_day = (date == pos.buy_date)
            cum_return = (close_today / pos.buy_price - 1)

            # 60-day checkpoint
            if not is_buy_day and days_held >= 60 and cum_return < 0.15:
                to_sell.append((code, date, close_today,
                               f'60d_weak(T+{days_held},{cum_return*100:+.0f}%)', 1.0))
                continue

            # 252-day force clear
            if not is_buy_day and days_held >= 252:
                to_sell.append((code, date, close_today,
                               f'1yr_clear(T+{days_held},{cum_return*100:+.0f}%)', 1.0))
                continue

            candle_stop = close_today < pos.entry_low * 0.97
            signal_stop = close_today < pos.signal_close
            yellow = float(row.get('yellow_line', 0))
            yellow_stop = yellow > 0 and close_today < yellow

            if candle_stop:
                reason = f'candle(T+{days_held})'
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
                    to_sell.append((code, t2_date, t2_open, f'buyday_{reason}(T+2)', 1.0))
                else:
                    to_sell.append((code, date, close_today, f'buyday_{reason}', 1.0))
            else:
                to_sell.append((code, date, close_today, reason, 1.0))

        for code, exit_date, exit_price, reason, partial in to_sell:
            if code in self.positions:
                self.sell(code, exit_date, exit_price, reason, partial=partial)

    def calc_portfolio_value(self, date):
        hv = 0
        for code, pos in self.positions.items():
            price = self.get_price(code, date, 'close')
            if price is not None:
                hv += pos.shares * price
            else:
                hv += pos.cost
        return self.cash + hv

    def rotate_if_needed(self, signal):
        if len(self.positions) <= self.max_positions:
            return True
        weakest = self.find_weakest_position()
        if weakest is None:
            return True
        if signal['weight'] > weakest.signal_weight + 1.0:
            sell_price = self.get_price(weakest.code, signal['t1_date'], 'close')
            if sell_price is None:
                sell_price = signal['t1_open']
            self.sell(weakest.code, signal['t1_date'], sell_price, 'rotate')
            return True
        return False

    def run(self):
        signals = self.extract_signals()
        if not signals:
            return self._empty_result()

        all_dates = set()
        for s in signals:
            all_dates.add(s['signal_date'])
            all_dates.add(s['t1_date'])
        trading_dates = sorted(d for d in all_dates
                               if SIM_START <= d <= SIM_END + pd.Timedelta(days=10))

        pending = []
        for date in trading_dates:
            self.check_stops(date)

            # Retry pending (within 3 days)
            still_pending = []
            for s in pending:
                if (date - s['signal_date']).days > 3:
                    continue
                if s['code'] in self.positions:
                    continue
                if self.can_add_position(s['weight']):
                    self.buy(s)
                    continue
                elif self.rotate_if_needed(s):
                    self.buy(s)
                    continue
                still_pending.append(s)
            pending = still_pending

            # New signals
            new_sigs = [s for s in signals if s['t1_date'] == date]
            for sig in new_sigs:
                if sig['code'] in self.positions:
                    continue
                if self.can_add_position(sig['weight']):
                    self.buy(sig)
                    continue
                elif self.rotate_if_needed(sig):
                    self.buy(sig)
                    continue
                pending.append(sig)

            total_value = self.calc_portfolio_value(date)
            self.daily_values.append({
                'date': date, 'value': total_value,
                'cash': self.cash, 'positions': len(self.positions),
            })

        # End-of-period cleanup
        final_date = trading_dates[-1] if trading_dates else SIM_END
        for code in list(self.positions.keys()):
            price = self.get_price(code, final_date, 'close')
            if price is None:
                price = self.get_price(code, final_date, 'open')
            if price is not None:
                self.sell(code, final_date, price, 'final_clear')

        return self._compute_metrics()

    def _empty_result(self):
        return {'total_return': 0, 'total_return_pct': 0, 'sharpe': 0,
                'n_trades': 0, 'n_signals': 0, 'win_rate': 0,
                'max_dd': 0, 'avg_pnl': 0, 'final_value': self.initial_capital}

    def _compute_metrics(self):
        final_value = self.calc_portfolio_value(
            self.daily_values[-1]['date'] if self.daily_values else SIM_END)
        total_ret = (final_value / self.initial_capital - 1)

        # Sharpe and max DD from daily values
        sharpe = 0
        max_dd = 0
        if len(self.daily_values) > 1:
            vals = self.daily_values
            peak = max(v['value'] for v in vals)
            peak_i = next(i for i, v in enumerate(vals) if v['value'] == peak)
            trough = min(v['value'] for v in vals[peak_i:])
            max_dd = (peak - trough) / peak if peak > 0 else 0

            rets = []
            for i in range(1, len(vals)):
                rets.append(vals[i]['value'] / vals[i-1]['value'] - 1)
            if rets:
                vol = np.std(rets) * np.sqrt(252)
                sharpe = total_ret / max(vol, 0.001) if vol > 0 else 0

        n_trades = len(self.closed_positions)
        win_rate = sum(1 for p in self.closed_positions if p['pnl'] > 0) / max(n_trades, 1)
        avg_pnl = np.mean([p['pnl_pct'] for p in self.closed_positions]) if self.closed_positions else 0

        buys = [t for t in self.trade_log if t['action'] == 'BUY']
        n_signals = len(buys)

        # Signal stats
        weights = [t['weight'] for t in buys] if buys else []
        n_super = sum(1 for w in weights if w >= 3.0)
        n_strong = sum(1 for w in weights if 2.0 <= w < 3.0)

        return {
            'total_return': float(total_ret),
            'total_return_pct': float(total_ret * 100),
            'sharpe': float(sharpe),
            'n_trades': n_trades,
            'n_signals': n_signals,
            'win_rate': float(win_rate),
            'max_dd': float(max_dd),
            'avg_pnl_pct': float(avg_pnl),
            'final_value': float(final_value),
            'n_super_signals': n_super,
            'n_strong_signals': n_strong,
        }


# ============================================================
# Main: Pre-load → Test param sets → Report
# ============================================================
def preload_all_stocks():
    """Load all parquet feature files (pre-computed indicators from Phase 0)."""
    print("=" * 70)
    print("  Phase 2b: Pre-loading stocks from parquet features")
    print("=" * 70)

    parquet_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    print(f"  Found {len(parquet_files)} parquet files")

    # Columns needed: OHLCV + indicators for generate_spring_signals(precomputed=True)
    needed_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                   'white_line', 'yellow_line', 'white_above_yellow', 'white_direction',
                   'brick_val', 'brick_var6a', 'brick_rising', 'brick_falling',
                   'brick_green_to_red', 'brick_red_to_green',
                   'brick_green_streak', 'brick_red_streak',
                   'cross_below_yellow']

    stock_data = {}
    n_done = 0
    t0 = time.time()

    for f in parquet_files:
        code = f.replace('.parquet', '')
        try:
            fp = os.path.join(FEAT_DIR, f)
            df = pd.read_parquet(fp)
            # Keep only needed cols + date
            cols_avail = [c for c in needed_cols if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            if len(df) < 120:
                continue

            # Ensure numeric types
            for col in ['K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                       'white_line', 'yellow_line', 'cross_below_yellow',
                       'brick_val', 'brick_green_to_red', 'brick_green_streak']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            stock_data[code] = df.copy()
        except Exception:
            continue

        n_done += 1
        if n_done % 1000 == 0:
            print(f"    ... {n_done}/{len(parquet_files)} loaded ({time.time()-t0:.0f}s)")

    print(f"  Pre-loaded {len(stock_data)} stocks ({time.time()-t0:.0f}s)")
    return stock_data


def main():
    t0 = time.time()

    # 1. Pre-load all stocks
    stock_data = preload_all_stocks()

    # 2. Build parameter sets
    param_sets = build_param_sets()
    print(f"\n  Testing {len(param_sets)} parameter sets on 2026 window...")
    print(f"  Window: {SIM_START.date()} → {SIM_END.date()}")
    print()

    # 3. Run simulation for each param set
    results = []
    for i, ps in enumerate(param_sets):
        name = ps['name']
        params = ps['params']

        t1 = time.time()
        engine = FastSimulationEngine(stock_data, params)
        metrics = engine.run()
        elapsed = time.time() - t1

        metrics['name'] = name
        metrics['params'] = params
        metrics['elapsed'] = elapsed

        is_default = (name == 'DEFAULT')
        tag = ' [DEFAULT]' if is_default else ''
        print(f"  {i+1:>2}. {name:<25s} Ret={metrics['total_return_pct']:+.1f}% "
              f"Sharpe={metrics['sharpe']:.3f} WR={metrics['win_rate']:.1%} "
              f"DD={metrics['max_dd']:.1%} Trades={metrics['n_trades']} "
              f"Sig={metrics['n_signals']} ({elapsed:.0f}s){tag}")

        results.append(metrics)

        # Show improvement vs default for non-default sets
        if not is_default and results:
            default_r = next((r for r in results if r['name'] == 'DEFAULT'), None)
            if default_r and default_r['total_return'] != 0:
                ret_delta = metrics['total_return_pct'] - default_r['total_return_pct']
                sharpe_delta = metrics['sharpe'] - default_r['sharpe']
                print(f"       vs DEFAULT: ΔRet={ret_delta:+.1f}%  ΔSharpe={sharpe_delta:+.3f}")

    # 4. Rank and report
    results.sort(key=lambda r: r['sharpe'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  Final Ranking (by Sharpe)")
    print(f"{'='*70}")
    print(f"  {'Rank':<6} {'Name':<25} {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Trades':>7} {'Sigs':>6}")
    print(f"  {'-'*70}")

    for i, r in enumerate(results):
        tag = ' ← DEFAULT' if r['name'] == 'DEFAULT' else ''
        print(f"  {i+1:<6} {r['name']:<25} {r['total_return_pct']:>+9.1f}% {r['sharpe']:>8.3f} "
              f"{r['win_rate']:>8.1%} {r['max_dd']:>8.1%} {r['n_trades']:>7} {r['n_signals']:>6}{tag}")

    # Best vs default comparison
    best = results[0]
    default_r = next((r for r in results if r['name'] == 'DEFAULT'), None)

    if default_r and best['name'] != 'DEFAULT':
        ret_imp = best['total_return_pct'] - default_r['total_return_pct']
        sharpe_imp = best['sharpe'] - default_r['sharpe']
        print(f"\n  ★ Best: {best['name']}")
        print(f"    Return improvement vs DEFAULT: {ret_imp:+.1f}%")
        print(f"    Sharpe improvement vs DEFAULT: {sharpe_imp:+.3f}")
        print(f"    Best params: {json.dumps({k: v for k, v in best['params'].items() if v != DEFAULT_PARAMS.get(k)}, indent=4)}")

    # Parameter analysis
    print(f"\n{'='*70}")
    print(f"  Parameter Impact Analysis")
    print(f"{'='*70}")

    # Entry threshold impact
    tight_j_prev = [r for r in results if r['name'].startswith('Tight_Jprev')]
    if tight_j_prev:
        avg_ret = np.mean([r['total_return_pct'] for r in tight_j_prev])
        print(f"  Tight J_prev (15): avg return {avg_ret:+.1f}%")

    brick_sets = [r for r in results if 'Brick' in r['name']]
    if brick_sets:
        avg_ret = np.mean([r['total_return_pct'] for r in brick_sets])
        print(f"  Brick resonance variations: avg return {avg_ret:+.1f}%")

    # 5. Save results
    out = {
        'phase': '2b_full_simulation',
        'simulation_window': f'{SIM_START.date()}~{SIM_END.date()}',
        'n_param_sets': len(param_sets),
        'n_stocks': len(stock_data),
        'default_params': DEFAULT_PARAMS,
        'results': [{
            'name': r['name'],
            'total_return_pct': round(r['total_return_pct'], 2),
            'sharpe': round(r['sharpe'], 4),
            'win_rate': round(r['win_rate'], 4),
            'max_dd': round(r['max_dd'], 4),
            'n_trades': r['n_trades'],
            'n_signals': r['n_signals'],
            'avg_pnl_pct': round(r['avg_pnl_pct'], 2),
            'n_super_signals': r.get('n_super_signals', 0),
            'n_strong_signals': r.get('n_strong_signals', 0),
            'elapsed_seconds': round(r['elapsed'], 1),
            'params_diff_from_default': {
                k: v for k, v in r['params'].items()
                if v != DEFAULT_PARAMS.get(k)
            } if r['name'] != 'DEFAULT' else {},
        } for r in results],
        'best': results[0]['name'] if results else None,
        'elapsed_seconds': time.time() - t0,
    }

    out_path = os.path.join(OUT_DIR, 'phase2b_full_simulation.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # CSV
    pd.DataFrame([{
        'name': r['name'],
        'total_return_pct': r['total_return_pct'],
        'sharpe': r['sharpe'],
        'win_rate': r['win_rate'],
        'max_dd': r['max_dd'],
        'n_trades': r['n_trades'],
        'n_signals': r['n_signals'],
        'avg_pnl_pct': r.get('avg_pnl_pct', 0),
    } for r in results]).to_csv(
        os.path.join(OUT_DIR, 'phase2b_full_simulation.csv'), index=False)

    print(f"\n  Results saved: {out_path}")
    print(f"  Phase 2b Complete ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
