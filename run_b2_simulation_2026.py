#!/usr/bin/env python3
"""
b2 策略 2026年 T+1 模拟交易系统
================================
规则:
  - 初始资金: 100,000
  - 持仓: 3-4只, 权重≥3.0x 可突破上限 (特强信号)
  - T+1 开盘买入, 信号日收盘价止损 (T+3后生效)
  - 买入当天不可止损 (触发则在T+2开盘离场)
  - 信号更强可调仓换股
  - 无持仓期限限制

用法:
  python run_b2_simulation_2026.py                    # 默认参数
  python run_b2_simulation_2026.py --capital 200000   # 自定义资金
  python run_b2_simulation_2026.py --start 2026-01-01 --end 2026-05-07
"""

import argparse, os, sys, time, warnings
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_b2 import generate_b2_signals

DATA_DIR = os.path.join(ROOT, 'fetch_kline', 'stock_kline')
OUT = os.path.join(ROOT, 'output')

# 交易成本
COMMISSION_BUY = 0.0003   # 买入佣金 0.03%
COMMISSION_SELL = 0.0013  # 卖出佣金+印花税 0.13%

# 默认参数
DEFAULT_CAPITAL = 100_000
DEFAULT_MAX_POSITIONS = 4
DEFAULT_MIN_POSITIONS = 3
SUPER_WEIGHT_THRESHOLD = 3.0  # 权重≥3.0可突破持仓上限


@dataclass
class Position:
    code: str
    signal_date: pd.Timestamp
    signal_close: float      # 信号日收盘价 (止损基准)
    signal_weight: float     # 信号权重
    buy_date: pd.Timestamp   # T+1 日期
    buy_price: float          # T+1 开盘价
    shares: int
    cost: float
    entry_low: float = 0.0   # 入场日最低价
    peak_price: float = 0.0  # 持仓期间最高收盘价
    half_sold: bool = False  # 是否已减半仓
    was_profitable: bool = False  # 曾浮盈>5%
    days_held: int = 0


class SimulationEngine:
    def __init__(self, capital=DEFAULT_CAPITAL, max_positions=DEFAULT_MAX_POSITIONS,
                 min_positions=DEFAULT_MIN_POSITIONS, start_date='2026-01-01',
                 end_date='2026-05-07', commission_buy=COMMISSION_BUY,
                 commission_sell=COMMISSION_SELL,
                 dynamic_sizing=False, sizing_step=50000, sizing_add=(1, 2)):
        self.initial_capital = capital
        self.cash = capital
        self.base_max = max_positions
        self.base_min = min_positions
        self.max_positions = max_positions
        self.min_positions = min_positions
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell

        # 动态仓位: 每盈利 sizing_step, 增加 sizing_add 只持仓
        self.dynamic_sizing = dynamic_sizing
        self.sizing_step = sizing_step
        self.sizing_add = sizing_add

        # State
        self.positions: Dict[str, Position] = {}
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.closed_positions: List[Dict] = []
        self.daily_values: List[Dict] = []
        self.trade_log: List[Dict] = []

    def update_position_limits(self):
        """根据当前总资产动态调整持仓上限"""
        if not self.dynamic_sizing:
            return
        total_value = self.calc_portfolio_value(pd.Timestamp.now())  # approximate
        profit = total_value - self.initial_capital
        steps = max(0, int(profit / self.sizing_step))
        add_low, add_high = self.sizing_add
        self.max_positions = self.base_max + steps * add_high
        self.min_positions = self.base_min + steps * add_low

    def load_stock(self, code):
        if code in self.stock_data:
            return self.stock_data[code]
        path = os.path.join(DATA_DIR, f'{code}.csv')
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=['date'],
                         dtype={'open': np.float32, 'close': np.float32,
                                'high': np.float32, 'low': np.float32, 'volume': np.float32})
        df = df.set_index('date').sort_index()
        self.stock_data[code] = df
        return df

    def get_price(self, code, date, price_type='close'):
        """获取指定日期的价格，若无数据则取最近一个交易日"""
        df = self.stock_data.get(code)
        if df is None:
            return None
        if date in df.index:
            return float(df.loc[date, price_type])
        # 向前找最近交易日
        prev_dates = df.index[df.index <= date]
        if len(prev_dates) > 0:
            return float(df.loc[prev_dates[-1], price_type])
        return None

    def scan_signals(self, min_vol_ratio=1.0, code_allowlist=None,
                     mktcap_df=None, mktcap_min=None, mktcap_max=None):
        """扫描所有股票在2026年的b2信号.
           min_vol_ratio: 信号日量比底线 (<此值则过滤)
           code_allowlist: 可选的白名单股票代码列表
           mktcap_df: 月度市值快照 DataFrame (trade_date, symbol, mv_yi)
           mktcap_min/max: 市值范围过滤 (亿元), 动态逐月筛选"""
        print("扫描 b2 入场信号 (2026)...", flush=True)
        all_signals = []
        if code_allowlist is not None:
            codes = code_allowlist
        else:
            codes = sorted([f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
        total = len(codes)
        t0 = time.time()

        # Build monthly market cap lookup: (year_month_str, symbol) -> mv_yi
        mktcap_lookup = {}
        if mktcap_df is not None and mktcap_min is not None:
            mktcap_df['ym'] = mktcap_df['trade_date'].astype(str).str[:7]
            # Ensure symbol is zero-padded 6-digit string
            mktcap_df['symbol_str'] = mktcap_df['symbol'].astype(str).str.zfill(6)
            for _, r in mktcap_df.iterrows():
                mktcap_lookup[(r['ym'], r['symbol_str'])] = r['mv_yi']
            print(f"  动态市值过滤: {mktcap_min}-{mktcap_max}亿, "
                  f"{mktcap_df['ym'].nunique()}个月快照", flush=True)

        for i, code in enumerate(codes):
            df = self.load_stock(code)
            if df is None or len(df) < 120:
                continue

            # Filter to 2026
            df_2026 = df[df.index >= self.start_date]
            if len(df_2026) < 5:
                continue

            try:
                # Need full history for indicators
                full_df = df.loc[:self.end_date + pd.Timedelta(days=5)]
                if len(full_df) < 120:
                    continue

                full_df = calc_all_indicators(full_df, board_type='main')
                full_df = generate_b2_signals(full_df, board_type='main', precomputed=True)

                # Find signals in date range
                mask = (full_df.index >= self.start_date) & (full_df.index <= self.end_date)
                signal_dates = full_df.index[mask & (full_df['b2_entry_signal'] == 1)]

                for sd in signal_dates:
                    row = full_df.loc[sd]

                    # 量比过滤
                    if min_vol_ratio > 1.0:
                        prev_vol = float(full_df.shift(1).loc[sd, 'volume']) if sd in full_df.shift(1).index else 0
                        if prev_vol > 0 and float(row['volume']) / prev_vol < min_vol_ratio:
                            continue

                    # 动态市值过滤: 按月快照检查是否在范围内
                    if mktcap_lookup:
                        ym = sd.strftime('%Y-%m')
                        mv = mktcap_lookup.get((ym, code))
                        if mv is None or mv < mktcap_min or mv > mktcap_max:
                            continue

                    weight = float(row.get('b2_position_weight', 1.0))
                    sig_close = float(row['close'])

                    # Find T+1 open (next trading day)
                    next_dates = full_df.index[full_df.index > sd]
                    if len(next_dates) == 0:
                        continue
                    t1_date = next_dates[0]
                    t1_open = float(full_df.loc[t1_date, 'open'])
                    t1_close = float(full_df.loc[t1_date, 'close'])

                    all_signals.append({
                        'code': code,
                        'signal_date': sd,
                        'signal_close': sig_close,
                        'weight': weight,
                        't1_date': t1_date,
                        't1_open': t1_open,
                        't1_close': t1_close,
                        'J_prev': float(full_df.loc[sd, 'J']),  # J at signal day
                        'gain': (sig_close / float(full_df.shift(1).loc[sd, 'close']) - 1) * 100
                        if sd in full_df.shift(1).index else 0,
                    })

            except Exception as e:
                continue

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{total} {elapsed:.0f}s  signals={len(all_signals)}", flush=True)

        elapsed = time.time() - t0
        print(f"  完成: {total}只, {elapsed:.0f}s, 共 {len(all_signals)} 个信号", flush=True)

        # Sort by signal date
        all_signals.sort(key=lambda x: x['signal_date'])
        return all_signals

    def can_add_position(self, signal_weight):
        """检查是否可以新增仓位"""
        current = len(self.positions)
        if current < self.min_positions:
            return True
        if current < self.max_positions:
            return True
        # 特强信号可突破上限
        if signal_weight >= SUPER_WEIGHT_THRESHOLD:
            return True
        return False

    def find_weakest_position(self):
        """找到最弱的持仓用于调仓"""
        if not self.positions:
            return None
        return min(self.positions.values(), key=lambda p: p.signal_weight)

    def calc_position_size(self, signal_weight, buy_price):
        """计算买入股数 (100股整数倍)"""
        base_per_position = self.cash / max(self.min_positions, 1)
        # 权重调整
        weight_factor = min(signal_weight / 2.0, 2.0)  # weight相对2.0的比率, 上限2x
        alloc = base_per_position * weight_factor
        alloc = min(alloc, self.cash * 0.95)  # 最多用95%现金
        shares = int(alloc / buy_price / 100) * 100
        if shares < 100:
            shares = 100 if self.cash >= buy_price * 100 * 1.01 else 0
        return shares

    def buy(self, signal, date=None):
        """执行买入"""
        code = signal['code']
        buy_price = signal['t1_open']
        shares = self.calc_position_size(signal['weight'], buy_price)
        if shares == 0:
            return False

        cost = shares * buy_price * (1 + self.commission_buy)
        if cost > self.cash:
            shares = int(self.cash * 0.95 / buy_price / 100) * 100
            if shares < 100:
                return False
            cost = shares * buy_price * (1 + self.commission_buy)

        self.cash -= cost

        # 获取入场日最低价
        df = self.stock_data.get(code)
        entry_low = buy_price  # fallback
        if df is not None and signal['t1_date'] in df.index:
            entry_low = float(df.loc[signal['t1_date'], 'low'])

        pos = Position(
            code=code,
            signal_date=signal['signal_date'],
            signal_close=signal['signal_close'],
            signal_weight=signal['weight'],
            buy_date=signal['t1_date'],
            buy_price=buy_price,
            shares=shares,
            cost=cost,
            entry_low=entry_low,
            peak_price=buy_price,
        )
        self.positions[code] = pos

        self.trade_log.append({
            'action': 'BUY',
            'date': signal['t1_date'],
            'code': code,
            'price': buy_price,
            'shares': shares,
            'cost': cost,
            'weight': signal['weight'],
            'cash_after': self.cash,
        })
        return True

    def sell(self, code, date, price, reason='manual', partial=1.0):
        """执行卖出. partial: 卖出比例 (1.0=全仓, 0.5=半仓)"""
        pos = self.positions[code]
        sell_shares = int(pos.shares * partial)
        if sell_shares < 100:
            return  # 不足1手

        proceeds = sell_shares * price * (1 - self.commission_sell)
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
            'code': code,
            'buy_date': pos.buy_date,
            'buy_price': pos.buy_price,
            'sell_date': date,
            'sell_price': price,
            'shares': sell_shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': pos.days_held,
            'reason': reason,
            'weight': pos.signal_weight,
            'partial': partial,
        })

        self.trade_log.append({
            'action': 'SELL',
            'date': date,
            'code': code,
            'price': price,
            'shares': sell_shares,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'cash_after': self.cash,
        })

    def check_stops_and_tp(self, date):
        """止损 + 检查点:
           T+60: 收益<15% → 释放弱股
           T+252: 强制清仓 (捕获峰值)
           常规止损: 蜡烛破位(3%缓冲) | 信号收盘 | 黄线破位
        """
        to_sell = []
        for code, pos in self.positions.items():
            df = self.stock_data.get(code)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            close_today = float(row['close'])

            buy_idx = df.index.get_loc(pos.buy_date) if pos.buy_date in df.index else None
            today_idx = df.index.get_loc(date) if date in df.index else None
            if buy_idx is not None and today_idx is not None:
                pos.days_held = today_idx - buy_idx

            is_buy_day = (date == pos.buy_date)
            cum_return = (close_today / pos.buy_price - 1)

            # ── 60日检查点: 收益<15% → 释放弱股 ──
            if not is_buy_day and pos.days_held >= 60 and cum_return < 0.15:
                to_sell.append((code, date, close_today,
                               f'60日弱(T+{pos.days_held}, {cum_return*100:+.0f}%)', 1.0))
                print(f"  🔄 {date.date()} 60日检查 {code}: {pos.days_held}天 {cum_return*100:+.0f}%", flush=True)
                continue

            # ── 252日硬限 ──
            if not is_buy_day and pos.days_held >= 252:
                to_sell.append((code, date, close_today,
                               f'1年清仓(T+{pos.days_held}, {cum_return*100:+.0f}%)', 1.0))
                print(f"  🔄 {date.date()} 1年清仓 {code}: {pos.days_held}天 {cum_return*100:+.0f}%", flush=True)
                continue

            candle_stop_level = pos.entry_low * 0.97
            candle_triggered = close_today < candle_stop_level
            signal_triggered = close_today < pos.signal_close
            yellow = float(row.get('yellow_line', 0))
            yellow_triggered = yellow > 0 and close_today < yellow

            if candle_triggered:
                reason = f'蜡烛破位(缓冲3%, T+{pos.days_held})'
            elif signal_triggered:
                reason = f'破信号收盘({pos.signal_close:.2f}, T+{pos.days_held})'
            elif yellow_triggered:
                reason = f'破黄线(T+{pos.days_held})'
            else:
                continue

            if is_buy_day:
                next_dates = df.index[df.index > date]
                if len(next_dates) > 0:
                    t2_date = next_dates[0]
                    t2_open = float(df.loc[t2_date, 'open'])
                    to_sell.append((code, t2_date, t2_open, f'买入日{reason}(T+2开盘)', 1.0))
                else:
                    to_sell.append((code, date, close_today, f'买入日{reason}(无T+2)', 1.0))
            else:
                to_sell.append((code, date, close_today, reason, 1.0))

        for code, exit_date, exit_price, reason, partial in to_sell:
            if code in self.positions:
                if partial < 1.0:
                    self.sell(code, exit_date, exit_price, reason, partial=partial)
                else:
                    print(f"  ⚠ {exit_date.date()} 止损 {code}: {reason} @{exit_price:.2f}", flush=True)
                    self.sell(code, exit_date, exit_price, reason)

    def calc_portfolio_value(self, date):
        """计算当日持仓市值"""
        holdings_value = 0
        for code, pos in self.positions.items():
            price = self.get_price(code, date, 'close')
            if price is not None:
                holdings_value += pos.shares * price
            else:
                holdings_value += pos.cost  # 无数据, 按成本计
        return self.cash + holdings_value

    def rotate_if_needed(self, signal):
        """如果需要调仓换股, 卖出最弱换入最强"""
        if len(self.positions) <= self.max_positions:
            return True  # 还有空位

        weakest = self.find_weakest_position()
        if weakest is None:
            return True

        # 只有新信号明显更强才换 (权重差距>1.0)
        if signal['weight'] > weakest.signal_weight + 1.0:
            # 找最近卖出日
            sell_date = signal['t1_date']
            sell_price = self.get_price(weakest.code, sell_date, 'close')
            if sell_price is None:
                sell_price = self.get_price(weakest.code, sell_date, 'open')
                if sell_price is None:
                    return False

            print(f"  🔄 {sell_date.date()} 调仓: 卖{weakest.code}(w={weakest.signal_weight:.1f}) "
                  f"→ 买{signal['code']}(w={signal['weight']:.1f})", flush=True)
            self.sell(weakest.code, sell_date, sell_price, '换仓-更强信号')
            return True

        return False  # 不调仓

    def run(self, min_vol_ratio=1.0, code_allowlist=None,
            mktcap_df=None, mktcap_min=None, mktcap_max=None):
        """主模拟循环"""
        print("=" * 70)
        print(f"  b2 策略 2026年 T+1 模拟交易")
        print(f"  初始资金: {self.initial_capital:,.0f}  |  "
              f"持仓: {self.min_positions}-{self.max_positions}只  |  "
              f"特强阈值: ≥{SUPER_WEIGHT_THRESHOLD}x")
        print(f"  期间: {self.start_date.date()} → {self.end_date.date()}")
        print("=" * 70)

        # 1. Scan signals
        signals = self.scan_signals(min_vol_ratio=min_vol_ratio, code_allowlist=code_allowlist,
                                    mktcap_df=mktcap_df, mktcap_min=mktcap_min,
                                    mktcap_max=mktcap_max)
        if not signals:
            print("未发现任何交易信号!")
            return

        print(f"\n>>> 开始模拟 ({len(signals)} 个信号)...\n", flush=True)

        # 2. Get all trading dates in range
        all_dates = set()
        for s in signals:
            all_dates.add(s['signal_date'])
            all_dates.add(s['t1_date'])
        trading_dates = sorted(d for d in all_dates
                               if self.start_date <= d <= self.end_date + pd.Timedelta(days=10))

        # 3. Process signals chronologically (with retry for unbought signals)
        pending = []  # signals not yet bought
        total_signals = len(signals)

        for date in trading_dates:
            if self.dynamic_sizing:
                total_val = self.calc_portfolio_value(date)
                profit = total_val - self.initial_capital
                steps = max(0, int(profit / self.sizing_step))
                add_low, add_high = self.sizing_add
                self.max_positions = self.base_max + steps * add_high
                self.min_positions = self.base_min + steps * add_low

            self.check_stops_and_tp(date)

            # Retry pending signals first (within 3 days of signal)
            still_pending = []
            for s in pending:
                if (date - s['signal_date']).days > 3:
                    continue  # expired
                if s['code'] in self.positions:
                    continue
                if self.can_add_position(s['weight']):
                    if self.buy(s, date):
                        print(f"  ✓ {date.date()} 补买 {s['code']} w={s['weight']:.1f}x", flush=True)
                        continue
                elif self.rotate_if_needed(s):
                    if self.buy(s, date):
                        print(f"  ✓ {date.date()} 补买(换仓) {s['code']} w={s['weight']:.1f}x", flush=True)
                        continue
                still_pending.append(s)
            pending = still_pending

            # Process new signals for this date
            new_pending = [s for s in signals if s['t1_date'] == date]
            for signal in new_pending:
                if signal['code'] in self.positions:
                    continue
                if self.can_add_position(signal['weight']):
                    if self.buy(signal, date):
                        tag = "★★★" if signal['weight'] >= SUPER_WEIGHT_THRESHOLD else ""
                        print(f"  ✓ {date.date()} 买入 {signal['code']} "
                              f"@{signal['t1_open']:.2f} w={signal['weight']:.1f}x "
                              f"现金={self.cash:,.0f} {tag}", flush=True)
                        continue
                elif self.rotate_if_needed(signal):
                    if self.buy(signal, date):
                        print(f"  ✓ {date.date()} 买入 {signal['code']} "
                              f"@{signal['t1_open']:.2f} w={signal['weight']:.1f}x "
                              f"(调仓后) 现金={self.cash:,.0f}", flush=True)
                        continue
                pending.append(signal)  # try again later

            # 记录每日净值
            total_value = self.calc_portfolio_value(date)
            self.daily_values.append({
                'date': date,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions),
            })

        # 4. 清仓
        final_date = trading_dates[-1] if trading_dates else self.end_date
        for code in list(self.positions.keys()):
            price = self.get_price(code, final_date, 'close')
            if price is None:
                price = self.get_price(code, final_date, 'open')
            if price is not None:
                self.sell(code, final_date, price, '期末清仓')

        # 5. Report
        self.report()

    def report(self):
        """输出模拟报告"""
        final_value = self.cash
        total_return = (final_value / self.initial_capital - 1) * 100

        print(f"\n{'='*70}")
        print(f"  模拟结果")
        print(f"{'='*70}")

        print(f"\n  初始资金: {self.initial_capital:,.0f}")
        print(f"  最终资金: {final_value:,.0f}")
        print(f"  总收益率: {total_return:+.2f}%")

        # 交易统计
        buys = [t for t in self.trade_log if t['action'] == 'BUY']
        sells = [t for t in self.trade_log if t['action'] == 'SELL']
        print(f"\n  交易次数: {len(buys)} 买 / {len(sells)} 卖")

        # 已完结交易
        if self.closed_positions:
            wins = [p for p in self.closed_positions if p['pnl'] > 0]
            losses = [p for p in self.closed_positions if p['pnl'] <= 0]
            print(f"\n  ┌─ 已完结交易 ({len(self.closed_positions)}笔) ─────────────────────────────┐")
            print(f"  │ 盈利: {len(wins)}笔  亏损: {len(losses)}笔  胜率: {len(wins)/len(self.closed_positions)*100:.1f}%")
            if wins:
                avg_win = np.mean([p['pnl_pct'] for p in wins])
                print(f"  │ 平均盈利: {avg_win:+.1f}%")
            if losses:
                avg_loss = np.mean([p['pnl_pct'] for p in losses])
                print(f"  │ 平均亏损: {avg_loss:+.1f}%")
            total_pnl = sum(p['pnl'] for p in self.closed_positions)
            print(f"  │ 已实现盈亏: {total_pnl:+,.0f}")
            print(f"  └──────────────────────────────────────────────────────┘")

            # 每笔详情
            print(f"\n  交易明细:")
            print(f"  {'日期':<12} {'代码':<8} {'方向':<4} {'价格':>8} {'股数':>6} {'金额':>12} {'盈亏%':>8} {'原因':<20}")
            print(f"  {'─'*12} {'─'*8} {'─'*4} {'─'*8} {'─'*6} {'─'*12} {'─'*8} {'─'*20}")
            for t in self.trade_log:
                pnl_str = f"{t.get('pnl_pct', 0):+.1f}%" if 'pnl_pct' in t else '-'
                print(f"  {str(t['date'].date()):<12} {t['code']:<8} {t['action']:<4} "
                      f"{t['price']:>8.2f} {t['shares']:>6} {t.get('cost', t.get('proceeds', 0)):>12,.0f} "
                      f"{pnl_str:>8} {t.get('reason', ''):<20}")

        # 权重分布
        if buys:
            weights = [t['weight'] for t in buys]
            print(f"\n  买入信号权重分布:")
            w3 = sum(1 for w in weights if w >= 3.0)
            w2 = sum(1 for w in weights if 2.0 <= w < 3.0)
            w1 = sum(1 for w in weights if w < 2.0)
            print(f"    ★★★ 特强(≥3.0x): {w3} 次  ★★ 强(2.0-3.0x): {w2} 次  ★ (<2.0x): {w1} 次")

        # 净值曲线摘要
        if len(self.daily_values) > 1:
            vals = self.daily_values
            peak = max(v['value'] for v in vals)

            # 滚动最大回撤: 每个点相对历史最高点的最大跌幅
            cummax = 0
            max_dd = 0
            for v in vals:
                if v['value'] > cummax:
                    cummax = v['value']
                drawdown = (cummax - v['value']) / cummax * 100
                if drawdown > max_dd:
                    max_dd = drawdown

            # 年化收益率 & 夏普比率
            rets = []
            for i in range(1, len(vals)):
                rets.append(vals[i]['value'] / vals[i-1]['value'] - 1)
            if rets:
                days = len(rets)
                total_ret_decimal = vals[-1]['value'] / vals[0]['value'] - 1
                ann_ret = (1 + total_ret_decimal) ** (252 / days) - 1
                ann_vol = np.std(rets) * np.sqrt(252)
                risk_free = 0.02
                sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0
            else:
                ann_ret = 0
                ann_vol = 0
                sharpe = 0

            print(f"\n  风险指标:")
            print(f"    峰值资金: {peak:,.0f}  最大回撤: {max_dd:.1f}%")
            print(f"    年化收益率: {ann_ret*100:.1f}%  年化波动率: {ann_vol*100:.1f}%")
            print(f"    夏普比率: {sharpe:.2f}")

        # Export
        os.makedirs(OUT, exist_ok=True)
        fp = os.path.join(OUT, 'b2_simulation_2026.csv')
        pd.DataFrame(self.trade_log).to_csv(fp, index=False)
        print(f"\n  交易记录: {fp}")

        fp2 = os.path.join(OUT, 'b2_simulation_2026_nav.csv')
        pd.DataFrame(self.daily_values).to_csv(fp2, index=False)
        print(f"  净值曲线: {fp2}")


def parse_args():
    p = argparse.ArgumentParser(description='b2策略 T+1 模拟交易系统 (2026年)')
    p.add_argument('--capital', type=float, default=DEFAULT_CAPITAL, help='初始资金')
    p.add_argument('--max-positions', type=int, default=DEFAULT_MAX_POSITIONS, help='最大持仓数')
    p.add_argument('--min-positions', type=int, default=DEFAULT_MIN_POSITIONS, help='最小持仓数')
    p.add_argument('--start', default='2026-01-01', help='开始日期')
    p.add_argument('--end', default='2026-05-07', help='结束日期')
    p.add_argument('--super-threshold', type=float, default=SUPER_WEIGHT_THRESHOLD,
                   help='特强信号权重阈值')
    p.add_argument('--min-vol-ratio', type=float, default=1.5,
                   help='信号日量比底线, >1.0可过滤弱信号 (推荐1.5)')
    p.add_argument('--dynamic-sizing', action='store_true',
                   help='启用动态仓位 (每盈利5万多1-2只持仓)')
    p.add_argument('--allowlist', default=None, help='股票白名单文件路径')
    return p.parse_args()


def main():
    args = parse_args()
    global SUPER_WEIGHT_THRESHOLD
    SUPER_WEIGHT_THRESHOLD = args.super_threshold

    # 动态市值过滤
    mktcap_df = None
    mktcap_min = mktcap_max = None
    mktcap_path = os.path.join(OUT, "monthly_mktcap.csv")
    if os.path.exists(mktcap_path):
        mktcap_df = pd.read_csv(mktcap_path)
        mktcap_min = getattr(args, 'mktcap_min', None) or 500
        mktcap_max = getattr(args, 'mktcap_max', None) or 1000
        print(f"动态市值过滤: {mktcap_min}-{mktcap_max}亿 "
              f"({len(mktcap_df)}条, {mktcap_df['trade_date'].nunique()}个月)")

    # 静态白名单（与动态过滤互斥）
    allowlist = None
    if mktcap_df is None:
        allowlist_path = getattr(args, 'allowlist', None)
        if not allowlist_path:
            allowlist_path = os.path.join(OUT, "qualified_stocks.txt")
        if os.path.exists(allowlist_path):
            with open(allowlist_path) as f:
                allowlist = [line.strip() for line in f if line.strip()]
            print(f"加载白名单: {len(allowlist)} 只 ({os.path.basename(allowlist_path)})")

    engine = SimulationEngine(
        capital=args.capital,
        max_positions=args.max_positions,
        min_positions=args.min_positions,
        start_date=args.start,
        end_date=args.end,
        dynamic_sizing=args.dynamic_sizing,
    )
    engine.run(min_vol_ratio=args.min_vol_ratio, code_allowlist=allowlist,
               mktcap_df=mktcap_df, mktcap_min=mktcap_min, mktcap_max=mktcap_max)


if __name__ == '__main__':
    main()
