#!/usr/bin/env python3
"""
Phase 2c: B2 + 0AMV Integrated Grid Search
============================================
Tests 0AMV market regime filtering combined with the best B2 parameters
from Phase 2b. Uses the real simulation engine with 0AMV-aware stops.

Parameter space:
  - 0AMV aggressive threshold: when daily oamv change exceeds X%, enter aggressive mode
  - 0AMV defensive threshold: when daily oamv change drops below -Y%, enter defensive mode
  - Stop buffer multipliers for aggressive/defensive modes
  - Defensive ban entry: whether to forbid new positions in defensive mode
"""

import os, sys, time, json, copy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals
from quant_strategy.strategy_spring import generate_spring_signals
from ML_optimization.mktcap_utils import (is_in_pct_range, check_liquidity,
                                           get_oamv_regime_range)
from ML_optimization.sector_utils import preload_sector_data, get_sector_score

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

COMMISSION_BUY = 0.0003
COMMISSION_SELL = 0.0013
DEFAULT_CAPITAL = 100_000
SIM_START = pd.Timestamp('2026-01-01')
SIM_END = pd.Timestamp('2026-05-07')

SUPER_WEIGHT_THRESHOLD = 3.0


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


class OAMVSimEngine:
    """Simulation engine with 0AMV market regime integration."""

    def __init__(self, stock_data, b2_params, oamv_df, oamv_params,
                 mktcap_lookup=None, percentiles=None, enable_liquidity=False,
                 sector_lookup=None):
        self.stock_data = stock_data
        self.b2_params = b2_params
        self.oamv_df = oamv_df.set_index('date')
        self.oamv_params = oamv_params
        self.cash = DEFAULT_CAPITAL
        self.initial_capital = DEFAULT_CAPITAL
        self.max_positions = 4
        self.min_positions = 0   # allow empty in defensive
        self._last_check_date = None
        self.max_entries_per_day = oamv_params.get('max_entries_per_day', 3)

        # B+C+D filter support (Phase 1 validation)
        self.mktcap_lookup = mktcap_lookup
        self.percentiles = percentiles        # from compute_mktcap_percentiles()
        self.enable_liquidity = enable_liquidity

        # Sector momentum factor support
        self.sector_lookup = sector_lookup  # {'stock_board_map': ..., 'board_momentum': ...}

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.daily_values: List[Dict] = []
        self.trade_log: List[Dict] = []

        # Filter stats for reporting
        self.filter_stats = {'pct_pass': 0, 'pct_fail': 0,
                             'liq_pass': 0, 'liq_fail': 0,
                             'total_signals': 0, 'filtered_signals': 0}

    def _get_oamv_regime(self, date):
        """Determine 0AMV regime for a given date."""
        if date not in self.oamv_df.index:
            # Find nearest prior date
            prev = self.oamv_df.index[self.oamv_df.index <= date]
            if len(prev) == 0:
                return 'normal'
            date = prev[-1]

        change_pct = float(self.oamv_df.loc[date, 'oamv_change_pct'])
        agg_thresh = self.oamv_params.get('oamv_aggressive_threshold', 3.0)
        def_thresh = self.oamv_params.get('oamv_defensive_threshold', -2.35)

        if change_pct > agg_thresh:
            return 'aggressive'
        elif change_pct < def_thresh:
            return 'defensive'
        return 'normal'

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

    def extract_signals(self, lookback=None):
        """Extract B2 signals with 0AMV regime embedded in each stock's DataFrame.

        lookback: 仅处理最近N个交易日 (实时模式用120, 回测用None=全量)
        """
        all_signals = []
        agg_buf = self.oamv_params.get('oamv_aggressive_buffer', 0.95)
        def_buf = self.oamv_params.get('oamv_defensive_buffer', 0.99)
        ban_entry = self.oamv_params.get('oamv_defensive_ban_entry', True)

        for code, df in self.stock_data.items():
            try:
                # 实时加速: 只取最近N天, 但至少保留足够的指标计算窗口
                if lookback and len(df) > lookback + 60:
                    sdf = df.iloc[-lookback - 60:].copy()
                else:
                    sdf = df.copy()

                # Embed 0AMV regime into DataFrame for generate_spring_signals
                regimes = []
                for idx in sdf.index:
                    regimes.append(self._get_oamv_regime(idx))
                sdf['oamv_regime'] = regimes

                # Generate B2 signals with 0AMV-aware params
                b2p = copy.deepcopy(self.b2_params)
                b2p['oamv_aggressive_buffer'] = agg_buf
                b2p['oamv_defensive_buffer'] = def_buf
                b2p['oamv_normal_buffer'] = self.oamv_params.get('oamv_normal_buffer', 0.97)
                b2p['oamv_grace_days'] = self.oamv_params.get('oamv_grace_days', 2)
                b2p['oamv_defensive_ban_entry'] = ban_entry

                sdf = generate_spring_signals(sdf, board_type='main', precomputed=True, params=b2p)

                mask = (sdf.index >= SIM_START) & (sdf.index <= SIM_END)
                signal_dates = sdf.index[mask & (sdf['b2_entry_signal'] == 1)]

                for sd in signal_dates:
                    row = sdf.loc[sd]
                    weight = float(row.get('b2_position_weight', 1.0))
                    sig_close = float(row['close'])
                    regime = self._get_oamv_regime(sd)

                    self.filter_stats['total_signals'] += 1

                    # B+C: Percentile-based market cap filter (regime-aware)
                    if self.mktcap_lookup and self.percentiles:
                        if not is_in_pct_range(code, sd, self.mktcap_lookup,
                                               self.percentiles, regime=regime):
                            self.filter_stats['pct_fail'] += 1
                            self.filter_stats['filtered_signals'] += 1
                            continue
                        self.filter_stats['pct_pass'] += 1

                    # D: Liquidity filter
                    if self.enable_liquidity:
                        if not check_liquidity(sdf, sd, code, self.mktcap_lookup):
                            self.filter_stats['liq_fail'] += 1
                            self.filter_stats['filtered_signals'] += 1
                            continue
                        self.filter_stats['liq_pass'] += 1

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
                        'regime': regime,
                    })
            except Exception:
                continue

        all_signals.sort(key=lambda x: x['signal_date'])
        return all_signals

    # --- Position management (same as Phase 2b) ---
    def can_add_position(self, signal_weight):
        if len(self.positions) < self.min_positions:
            return True
        if len(self.positions) < self.max_positions:
            return True
        if signal_weight >= SUPER_WEIGHT_THRESHOLD:
            return True
        return False

    def find_weakest_position(self):
        if not self.positions:
            return None
        return min(self.positions.values(), key=lambda p: p.signal_weight)

    def calc_position_size(self, signal_weight, buy_price):
        # P15: 基于总权益(现金+持仓市值)分配, 利润自动再投资
        total_eq = self.cash
        for code, pos in self.positions.items():
            px = self.get_price(code, self._last_check_date, 'close')
            if px is None:
                px = pos.buy_price
            total_eq += px * pos.shares
        capital = max(total_eq, self.initial_capital)

        # P15: 牛市集中 — 剩余仓位越少单票越大
        slots_open = max(1, self.max_positions - len(self.positions) + 1)
        base = capital / slots_open
        wf = min(signal_weight / 2.0, 2.5)
        # 单票上限50%基于当前总权益, 利润越大仓位越大
        alloc = min(base * wf, capital * 0.50)
        # 实际不超可用现金
        alloc = min(alloc, self.cash * 0.98)

        shares = int(alloc / buy_price / 100) * 100
        # P15: min_single_amount ≥ 8000
        if shares < 100 or shares * buy_price < 8000:
            return 0
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
            buy_date=signal['t1_date'], buy_price=buy_price,
            shares=shares, cost=cost,
            entry_low=entry_low, peak_price=buy_price,
        )
        self.positions[code] = pos
        self.trade_log.append({
            'action': 'BUY', 'date': signal['t1_date'], 'code': code,
            'price': buy_price, 'shares': shares, 'cost': cost,
            'weight': signal['weight'], 'cash_after': self.cash,
            'regime': signal.get('regime', 'normal'),
            'sector_score': signal.get('sector_momentum_score', signal.get('b2_sector_score', None)),
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
        agg_buf = self.oamv_params.get('oamv_aggressive_buffer', 0.95)
        def_buf = self.oamv_params.get('oamv_defensive_buffer', 0.99)
        norm_buf = self.oamv_params.get('oamv_normal_buffer', 0.97)
        grace_days = self.oamv_params.get('oamv_grace_days', 2)

        for code, pos in self.positions.items():
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

            # 60-day checkpoint
            if not is_buy_day and days_held >= 60 and cum_return < 0.15:
                to_sell.append((code, date, close_today,
                               f'60d_weak(T+{days_held},{cum_return*100:+.0f}%)', 1.0))
                continue
            # 252-day force clear
            if not is_buy_day and days_held >= 252:
                to_sell.append((code, date, close_today,
                               f'1yr_clear(T+{days_held})', 1.0))
                continue

            # 0AMV-aware candle stop
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

    def calc_portfolio_value(self, date):
        hv = 0
        for code, pos in self.positions.items():
            price = self.get_price(code, date, 'close')
            hv += pos.shares * (price if price else pos.buy_price)
        return self.cash + hv

    def rotate_if_needed(self, signal):
        if len(self.positions) < self.max_positions:
            return True
        weakest = self.find_weakest_position()
        if weakest is None:
            return True

        # Calculate current return of weakest position
        sell_price = self.get_price(weakest.code, signal['t1_date'], 'close')
        if sell_price is None:
            sell_price = signal['t1_open']
        weakest_return = (sell_price / weakest.buy_price - 1) if weakest.buy_price > 0 else 0

        should_rotate = False

        # 1. New signal significantly stronger → rotate (lowered threshold: 0.3 from 1.0)
        if signal['weight'] > weakest.signal_weight + 0.3:
            should_rotate = True
        # 2. Profitable rotation: weakest has small profit + new signal is any stronger
        elif weakest_return > 0.03 and signal['weight'] > weakest.signal_weight:
            should_rotate = True
        # 3. Loss protection: weakest in loss + new signal has higher weight
        elif weakest_return < -0.02 and signal['weight'] > weakest.signal_weight:
            should_rotate = True

        if should_rotate:
            self.sell(weakest.code, signal['t1_date'], sell_price, 'rotate')
            return True
        return False

    def run(self):
        signals = self.extract_signals()
        if not signals:
            return self._empty_result()

        # Build all trading dates in simulation range (for accurate MaxDD)
        all_dates = set()
        for s in signals:
            all_dates.add(s['signal_date'])
            all_dates.add(s['t1_date'])
        sim_dates = sorted(d for d in all_dates)
        # Use real A-share trading dates from OAMV index (not US BDay)
        oamv_dates_in_range = [d for d in self.oamv_df.index
                               if sim_dates[0] <= d <= sim_dates[-1]]
        all_trading_dates = sorted(set(oamv_dates_in_range) | set(sim_dates))

        # Pre-index signals by t1_date
        sigs_by_t1 = {}
        for s in signals:
            sigs_by_t1.setdefault(s['t1_date'], []).append(s)

        pending = []
        for date in all_trading_dates:
            self.check_stops(date)
            regime = self._get_oamv_regime(date)
            self._last_check_date = date
            entries_today = 0

            # Process new signals on this date
            new_sigs = sigs_by_t1.get(date, [])
            pending.extend(new_sigs)

            still_pending = []
            for s in pending:
                if (date - s['signal_date']).days > 3:
                    continue
                if s['code'] in self.positions:
                    continue
                if entries_today >= self.max_entries_per_day:
                    still_pending.append(s)
                    continue
                if regime == 'defensive' and self.oamv_params.get('oamv_defensive_ban_entry', True):
                    still_pending.append(s)
                    continue
                if self.can_add_position(s['weight']):
                    if self.buy(s):
                        entries_today += 1
                    else:
                        still_pending.append(s)
                    continue
                elif self.rotate_if_needed(s):
                    if self.buy(s):
                        entries_today += 1
                    else:
                        still_pending.append(s)
                    continue
                still_pending.append(s)
            pending = still_pending

            # Record equity daily (P15: accurate MaxDD tracking)
            total_value = self.calc_portfolio_value(date)
            self.daily_values.append({
                'date': date, 'value': total_value,
                'cash': self.cash, 'positions': len(self.positions),
                'regime': regime,
            })

        final_date = all_trading_dates[-1] if all_trading_dates else SIM_END
        for code in list(self.positions.keys()):
            price = self.get_price(code, final_date, 'close')
            if price is None:
                price = self.get_price(code, final_date, 'open')
            if price is not None:
                self.sell(code, final_date, price, 'final_clear')

        return self._compute_metrics()

    def export_trade_log(self):
        """导出 trade_log 为 DataFrame（供回测持仓重建）"""
        import pandas as pd
        rows = []
        for t in self.trade_log:
            if t['action'] == 'BUY':
                rows.append({
                    'date': t['date'], 'symbol': t['code'], 'action': 'BUY',
                    'shares': t['shares'], 'price': t['price'], 'cost': t['cost'],
                    'weight': t.get('weight', 1.0),
                })
            elif t['action'] == 'SELL':
                rows.append({
                    'date': t['date'], 'symbol': t['code'], 'action': 'SELL',
                    'shares': t.get('shares', 0), 'price': t['price'],
                    'reason': t.get('reason', ''),
                })
        return pd.DataFrame(rows)

    def _empty_result(self):
        return {'total_return': 0, 'total_return_pct': 0, 'sharpe': 0,
                'n_trades': 0, 'n_signals': 0, 'win_rate': 0,
                'max_dd': 0, 'final_value': DEFAULT_CAPITAL}

    def _compute_metrics(self):
        final_value = self.calc_portfolio_value(
            self.daily_values[-1]['date'] if self.daily_values else SIM_END)
        total_ret = (final_value / self.initial_capital - 1)
        sharpe = 0
        max_dd = 0
        if len(self.daily_values) > 1:
            vals = self.daily_values
            # MaxDD from ANY peak to any subsequent trough
            peak_so_far = vals[0]['value']
            for v in vals[1:]:
                if v['value'] > peak_so_far:
                    peak_so_far = v['value']
                else:
                    dd = (peak_so_far - v['value']) / peak_so_far
                    if dd > max_dd:
                        max_dd = dd
            rets = []
            for i in range(1, len(vals)):
                rets.append(vals[i]['value'] / vals[i-1]['value'] - 1)
            if rets:
                vol = np.std(rets) * np.sqrt(252)
                sharpe = total_ret / max(vol, 0.001) if vol > 0 else 0

        n_trades = len(self.closed_positions)
        win_rate = sum(1 for p in self.closed_positions if p['pnl'] > 0) / max(n_trades, 1)
        buys = [t for t in self.trade_log if t['action'] == 'BUY']

        # Regime breakdown
        agg_buys = sum(1 for t in buys if t.get('regime') == 'aggressive')
        def_buys = sum(1 for t in buys if t.get('regime') == 'defensive')
        norm_buys = sum(1 for t in buys if t.get('regime') == 'normal')

        return {
            'total_return': float(total_ret),
            'total_return_pct': float(total_ret * 100),
            'sharpe': float(sharpe),
            'n_trades': n_trades,
            'n_signals': len(buys),
            'win_rate': float(win_rate),
            'max_dd': float(max_dd),
            'final_value': float(final_value),
            'aggressive_buys': agg_buys,
            'defensive_buys': def_buys,
            'normal_buys': norm_buys,
            'filter_stats': dict(self.filter_stats),
        }


# ============================================================
# Preload
# ============================================================
def preload_data():
    """Load parquet files with indicators pre-computed."""
    print("=" * 70)
    print("  Phase 2c: Pre-loading stocks + 0AMV data")
    print("=" * 70)

    # 0AMV
    print("  Computing 0AMV market data...")
    oamv_df = fetch_market_data(start='20180101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = generate_signals(oamv_df)

    # Stocks
    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    print(f"  Loading {len(feat_files)} stocks from parquet...")

    needed_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                   'white_line', 'yellow_line', 'white_above_yellow', 'white_direction',
                   'brick_val', 'brick_var6a', 'brick_rising', 'brick_falling',
                   'brick_green_to_red', 'brick_red_to_green',
                   'brick_green_streak', 'brick_red_streak',
                   'cross_below_yellow']

    stock_data = {}
    t0 = time.time()
    for i, f in enumerate(feat_files):
        code = f.replace('.parquet', '')
        try:
            fp = os.path.join(FEAT_DIR, f)
            df = pd.read_parquet(fp)
            cols_avail = [c for c in needed_cols if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
        except Exception:
            continue
        if (i + 1) % 2000 == 0:
            print(f"    ... {i+1}/{len(feat_files)} ({time.time()-t0:.0f}s)")

    print(f"  Pre-loaded {len(stock_data)} stocks + 0AMV {len(oamv_df)} days ({time.time()-t0:.0f}s)")

    # Sector momentum
    print("  Loading sector momentum data...")
    try:
        sector_lookup = preload_sector_data(stock_data_dict=stock_data, start='20180101')
        print(f"  Sector data: {len(sector_lookup.get('stock_board_map', {}))} stocks mapped, "
              f"{len(sector_lookup.get('board_momentum', {}))} board-date scores")
    except Exception as e:
        print(f"  WARNING: Sector data load failed ({e}), continuing without sector factor")
        sector_lookup = None

    return stock_data, oamv_df, sector_lookup


# ============================================================
# Parameter Sets
# ============================================================
# Best B2 params from Phase 2b
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

PULLBACK_B2 = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.07,
    'weight_pullback': 1.5, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'sector_momentum_enabled': False,
}

DEFAULT_B2 = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'sector_momentum_enabled': False,
}


def build_param_sets():
    """Build parameter combinations: B2 base × OAMV settings."""
    sets = []

    # Default OAMV settings
    default_oamv = {
        'oamv_aggressive_threshold': 3.0,
        'oamv_defensive_threshold': -2.35,
        'oamv_aggressive_buffer': 0.95,
        'oamv_defensive_buffer': 0.99,
        'oamv_normal_buffer': 0.97,
        'oamv_grace_days': 2,
        'oamv_defensive_ban_entry': True,
    }

    # 0: DEFAULT B2 + DEFAULT OAMV
    sets.append({'name': 'DEFAULT+OAMV_default', 'b2': copy.deepcopy(DEFAULT_B2),
                 'oamv': copy.deepcopy(default_oamv)})

    # 1: DEFAULT B2 + NO OAMV (baseline control)
    sets.append({'name': 'DEFAULT_no_OAMV', 'b2': copy.deepcopy(DEFAULT_B2),
                 'oamv': copy.deepcopy(default_oamv), 'no_oamv': True})

    # 2-4: Best B2 (Explosive) with different OAMV thresholds
    for agg_t, name_suffix in [(2.5, 'agg2.5'), (3.0, 'agg3.0'), (3.5, 'agg3.5')]:
        o = copy.deepcopy(default_oamv)
        o['oamv_aggressive_threshold'] = agg_t
        sets.append({'name': f'Explosive_{name_suffix}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # 5-7: Best B2 (Explosive) with different defensive thresholds
    for def_t, name_suffix in [(-2.0, 'def2.0'), (-2.35, 'def2.35'), (-2.5, 'def2.5')]:
        o = copy.deepcopy(default_oamv)
        o['oamv_defensive_threshold'] = def_t
        sets.append({'name': f'Explosive_{name_suffix}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # 8-9: Vary aggressive buffer
    for buf, name_suffix in [(0.93, 'abuf93'), (0.97, 'abuf97')]:
        o = copy.deepcopy(default_oamv)
        o['oamv_aggressive_buffer'] = buf
        sets.append({'name': f'Explosive_{name_suffix}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # 10-11: Vary defensive buffer
    for buf, name_suffix in [(0.97, 'dbuf97'), (1.0, 'dbuf100')]:
        o = copy.deepcopy(default_oamv)
        o['oamv_defensive_buffer'] = buf
        sets.append({'name': f'Explosive_{name_suffix}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # 12: Explosive without defensive ban
    o = copy.deepcopy(default_oamv)
    o['oamv_defensive_ban_entry'] = False
    sets.append({'name': 'Explosive_noDefBan', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # 13: Explosive + no OAMV (control)
    sets.append({'name': 'Explosive_no_OAMV', 'b2': copy.deepcopy(EXPLOSIVE_B2),
                 'oamv': copy.deepcopy(default_oamv), 'no_oamv': True})

    # 14: Pullback B2 + OAMV
    sets.append({'name': 'Pullback+OAMV', 'b2': copy.deepcopy(PULLBACK_B2),
                 'oamv': copy.deepcopy(default_oamv)})

    # 15: Aggressive OAMV (wider stops, more entries)
    o = copy.deepcopy(default_oamv)
    o.update({'oamv_aggressive_threshold': 2.5, 'oamv_aggressive_buffer': 0.93,
              'oamv_defensive_buffer': 0.97, 'oamv_defensive_ban_entry': False})
    sets.append({'name': 'OAMV_Aggressive', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # 16: Conservative OAMV (tighter stops, fewer entries)
    o = copy.deepcopy(default_oamv)
    o.update({'oamv_aggressive_threshold': 3.5, 'oamv_aggressive_buffer': 0.97,
              'oamv_defensive_threshold': -2.0, 'oamv_defensive_buffer': 1.0,
              'oamv_defensive_ban_entry': True})
    sets.append({'name': 'OAMV_Conservative', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})

    # Sector momentum variants (7 configs × 2 B2 bases = 14)
    SECTOR_CONFIGS = [
        ('SectorOff',    False, None,  None),   # 无板块动量对照
        ('SectorWide',   True,  1.25,  0.75),   # 极宽区间
        ('SectorAggr',   True,  1.20,  0.80),   # 激进
        ('SectorStd',    True,  1.15,  0.85),   # 标准（原默认）
        ('SectorMild',   True,  1.10,  0.90),   # 温和
        ('SectorTight',  True,  1.05,  0.95),   # 极窄区间
        ('SectorUpOnly', True,  1.20,  0.90),   # 非对称：强加多弱折少
    ]
    # B2 base 1: Explosive_def2.0
    base_oamv_explosive = copy.deepcopy(default_oamv)
    base_oamv_explosive['oamv_defensive_threshold'] = -2.0
    for tag, enabled, strong, weak in SECTOR_CONFIGS:
        b2 = copy.deepcopy(EXPLOSIVE_B2)
        b2['sector_momentum_enabled'] = enabled
        if strong is not None:
            b2['sector_weight_strong'] = strong
        if weak is not None:
            b2['sector_weight_weak'] = weak
        sets.append({'name': f'Explosive_{tag}', 'b2': b2,
                     'oamv': copy.deepcopy(base_oamv_explosive)})

    # B2 base 2: OAMV_Conservative (o from above = conservative OAMV params)
    for tag, enabled, strong, weak in SECTOR_CONFIGS:
        b2 = copy.deepcopy(EXPLOSIVE_B2)
        b2['sector_momentum_enabled'] = enabled
        if strong is not None:
            b2['sector_weight_strong'] = strong
        if weak is not None:
            b2['sector_weight_weak'] = weak
        sets.append({'name': f'OAMV_Conservative_{tag}', 'b2': b2,
                     'oamv': copy.deepcopy(o)})

    return sets


def main():
    t0 = time.time()

    # 1. Preload
    stock_data, oamv_df, sector_lookup = preload_data()

    # 2. Build param sets
    param_sets = build_param_sets()
    print(f"\n  Testing {len(param_sets)} parameter combinations on 2026 window...\n")

    # 3. Run simulations
    results = []
    for i, ps in enumerate(param_sets):
        name = ps['name']
        b2_params = ps['b2']
        oamv_params = ps['oamv']
        no_oamv = ps.get('no_oamv', False)

        t1 = time.time()

        if no_oamv:
            # Disable 0AMV: set thresholds to extreme values so regime is always normal
            oamv_params_use = copy.deepcopy(oamv_params)
            oamv_params_use['oamv_aggressive_threshold'] = 999.0
            oamv_params_use['oamv_defensive_threshold'] = -999.0
        else:
            oamv_params_use = oamv_params

        engine = OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params_use,
                               sector_lookup=sector_lookup)
        metrics = engine.run()
        elapsed = time.time() - t1

        metrics['name'] = name
        metrics['elapsed'] = elapsed

        tag = ' [CONTROL]' if no_oamv else ''
        print(f"  {i+1:>2}. {name:<25s} Ret={metrics['total_return_pct']:+.1f}% "
              f"Sharpe={metrics['sharpe']:.3f} WR={metrics['win_rate']:.1%} "
              f"DD={metrics['max_dd']:.1%} Trades={metrics['n_trades']} "
              f"Sig={metrics['n_signals']} "
              f"A/D/N={metrics.get('aggressive_buys',0)}/{metrics.get('defensive_buys',0)}/{metrics.get('normal_buys',0)} "
              f"({elapsed:.0f}s){tag}")

        results.append(metrics)

    # 4. Rank
    results.sort(key=lambda r: r['sharpe'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  Phase 2c Final Ranking (by Sharpe)")
    print(f"{'='*70}")
    print(f"  {'Rank':<6} {'Name':<25} {'Return':>9} {'Sharpe':>8} {'WR':>8} {'DD':>8} {'Trades':>6}")
    print(f"  {'-'*70}")

    for i, r in enumerate(results):
        tag = ''
        if 'no_OAMV' in r['name']:
            tag = ' ← CONTROL'
        elif r['name'].startswith('DEFAULT'):
            tag = ' ← BASELINE'
        print(f"  {i+1:<6} {r['name']:<25} {r['total_return_pct']:>+8.1f}% "
              f"{r['sharpe']:>8.3f} {r['win_rate']:>8.1%} {r['max_dd']:>8.1%} "
              f"{r['n_trades']:>6}{tag}")

    # 5. OAMV vs no-OAMV comparison
    oamv_on = [r for r in results if 'no_OAMV' not in r['name']]
    oamv_off = [r for r in results if 'no_OAMV' in r['name']]
    if oamv_on and oamv_off:
        avg_on = np.mean([r['sharpe'] for r in oamv_on])
        avg_off = np.mean([r['sharpe'] for r in oamv_off])
        print(f"\n  Average Sharpe: OAMV ON={avg_on:.3f} vs OFF={avg_off:.3f}")

    # 6. Save
    out = {
        'phase': '2c_oamv_grid_search',
        'n_param_sets': len(param_sets),
        'results': [{
            'name': r['name'],
            'total_return_pct': round(r['total_return_pct'], 2),
            'sharpe': round(r['sharpe'], 4),
            'win_rate': round(r['win_rate'], 4),
            'max_dd': round(r['max_dd'], 4),
            'n_trades': r['n_trades'],
            'n_signals': r['n_signals'],
            'aggressive_buys': r.get('aggressive_buys', 0),
            'defensive_buys': r.get('defensive_buys', 0),
            'normal_buys': r.get('normal_buys', 0),
        } for r in results],
        'best': results[0]['name'],
        'elapsed_seconds': time.time() - t0,
    }

    with open(os.path.join(OUT_DIR, 'phase2c_oamv_results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Phase 2c Complete ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
