#!/usr/bin/env python3
"""
0AMV 活筹指数增强回测 — 与基准 b2 策略对比

规则:
  - 0AMV单日涨>3% → 猛干模式: 止损放宽(entry_low*0.95), 让利润奔跑
  - 0AMV单日跌>2.35% → 防守模式: 止损收紧(entry_low*0.99), 正常买入(信号质量高)
  - 其余 → 基准模式（b2 默认参数）

用法:
  python simulation/run_oamv_backtest.py
  python simulation/run_oamv_backtest.py --start 2025-01-01 --end 2026-05-07
"""

import argparse, os, sys, time, warnings
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from copy import deepcopy

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)

from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_spring import generate_spring_signals

DATA_DIR = os.path.join(ROOT, "fetch_kline", "stock_kline")
OUT = os.path.join(ROOT, "output")

COMMISSION_BUY = 0.0003
COMMISSION_SELL = 0.0013

# ============================================================
# Position dataclass
# ============================================================
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
    days_held: int = 0


# ============================================================
# OAMV-enhanced Simulation Engine
# ============================================================
class OAMVSimEngine:
    def __init__(self, capital=100_000, max_positions=4, min_positions=3,
                 start_date="2026-01-01", end_date="2026-05-07",
                 oamv_df=None, use_oamv=False):
        self.initial_capital = capital
        self.cash = capital
        self.base_max = max_positions
        self.base_min = min_positions
        self.max_positions = max_positions
        self.min_positions = min_positions
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.use_oamv = use_oamv

        # OAMV data
        self.oamv_df = oamv_df
        self.current_regime = "normal"

        # State
        self.positions: Dict[str, Position] = {}
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.closed_positions: List[Dict] = []
        self.daily_values: List[Dict] = []
        self.trade_log: List[Dict] = []

    def get_oamv_regime(self, date):
        """根据 0AMV 单日涨跌幅判定当前市场状态"""
        if not self.use_oamv or self.oamv_df is None:
            return "normal"

        row = self.oamv_df[self.oamv_df["date"] == date]
        if row.empty:
            return "normal"

        chg = float(row["oamv_change_pct"].iloc[0])
        if chg > 3.0:
            return "aggressive"
        elif chg < -2.35:
            return "defensive"
        return "normal"

    def load_stock(self, code):
        if code in self.stock_data:
            return self.stock_data[code]
        path = os.path.join(DATA_DIR, f"{code}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=["date"],
                         dtype={"open": np.float32, "close": np.float32,
                                "high": np.float32, "low": np.float32, "volume": np.float32})
        df = df.set_index("date").sort_index()
        self.stock_data[code] = df
        return df

    def get_price(self, code, date, price_type="close"):
        df = self.stock_data.get(code)
        if df is None:
            return None
        if date in df.index:
            return float(df.loc[date, price_type])
        prev_dates = df.index[df.index <= date]
        if len(prev_dates) > 0:
            return float(df.loc[prev_dates[-1], price_type])
        return None

    def scan_signals(self):
        """扫描 b2 入场信号"""
        print(f"  [{'OAMV增强' if self.use_oamv else '基准'}] 扫描 b2 信号...", flush=True)
        all_signals = []
        codes = sorted([f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
        t0 = time.time()
        total = len(codes)

        for i, code in enumerate(codes):
            df = self.load_stock(code)
            if df is None or len(df) < 120:
                continue

            df_period = df[df.index >= self.start_date]
            if len(df_period) < 5:
                continue

            try:
                full_df = df.loc[:self.end_date + pd.Timedelta(days=5)]
                if len(full_df) < 120:
                    continue

                full_df = calc_all_indicators(full_df, board_type="main")
                full_df = generate_spring_signals(full_df, board_type="main", precomputed=True)

                mask = (full_df.index >= self.start_date) & (full_df.index <= self.end_date)
                signal_dates = full_df.index[mask & (full_df["b2_entry_signal"] == 1)]

                for sd in signal_dates:
                    row = full_df.loc[sd]
                    weight = float(row.get("b2_position_weight", 1.0))
                    sig_close = float(row["close"])

                    next_dates = full_df.index[full_df.index > sd]
                    if len(next_dates) == 0:
                        continue
                    t1_date = next_dates[0]
                    t1_open = float(full_df.loc[t1_date, "open"])

                    all_signals.append({
                        "code": code, "signal_date": sd,
                        "signal_close": sig_close, "weight": weight,
                        "t1_date": t1_date, "t1_open": t1_open,
                    })
            except Exception:
                continue

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{total} {time.time()-t0:.0f}s  sigs={len(all_signals)}", flush=True)

        print(f"    完成: {len(codes)}只, {time.time()-t0:.0f}s, {len(all_signals)} 信号", flush=True)
        all_signals.sort(key=lambda x: x["signal_date"])
        return all_signals

    # ── 核心：OAMV 模式下的止损/仓位逻辑 ──

    def get_candle_stop_ratio(self, regime):
        """蜡烛止损比例"""
        if regime == "aggressive":
            return 0.95  # 放宽止损
        elif regime == "defensive":
            return 0.99  # 极紧止损
        return 0.97  # 基准

    def get_position_multiplier(self, regime):
        """仓位倍数 — 猛干模式下不加仓，只放宽止损"""
        return 1.0

    def can_buy_in_regime(self, regime):
        """当前状态是否允许买入 — 所有模式均允许（防御日信号质量反而高）"""
        return True

    def should_force_sell_defensive(self, pos, date, df):
        """防守模式下是否强制清仓浮亏持仓"""
        if df is None or date not in df.index:
            return False
        close_today = float(df.loc[date, "close"])
        cum_return = (close_today / pos.buy_price - 1)
        return cum_return < 0  # 浮亏即清

    # ── 买入 / 卖出 / 止损 ──

    def calc_position_size(self, signal_weight, buy_price, regime="normal"):
        base_per_position = self.cash / max(self.min_positions, 1)
        weight_factor = min(signal_weight / 2.0, 2.0)
        alloc = base_per_position * weight_factor * self.get_position_multiplier(regime)
        alloc = min(alloc, self.cash * 0.95)
        shares = int(alloc / buy_price / 100) * 100
        if shares < 100:
            shares = 100 if self.cash >= buy_price * 100 * 1.01 else 0
        return shares

    def can_add_position(self, signal_weight):
        current = len(self.positions)
        if current < self.min_positions:
            return True
        if current < self.max_positions:
            return True
        if signal_weight >= 3.0:
            return True
        return False

    def find_weakest_position(self):
        if not self.positions:
            return None
        return min(self.positions.values(), key=lambda p: p.signal_weight)

    def buy(self, signal, date=None, regime="normal"):
        code = signal["code"]
        buy_price = signal["t1_open"]
        shares = self.calc_position_size(signal["weight"], buy_price, regime)
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
        if df is not None and signal["t1_date"] in df.index:
            entry_low = float(df.loc[signal["t1_date"], "low"])

        pos = Position(
            code=code, signal_date=signal["signal_date"],
            signal_close=signal["signal_close"], signal_weight=signal["weight"],
            buy_date=signal["t1_date"], buy_price=buy_price,
            shares=shares, cost=cost, entry_low=entry_low, peak_price=buy_price,
        )
        self.positions[code] = pos

        self.trade_log.append({
            "action": "BUY", "date": signal["t1_date"], "code": code,
            "price": buy_price, "shares": shares, "cost": cost,
            "weight": signal["weight"], "cash_after": self.cash,
            "regime": regime,
        })
        return True

    def sell(self, code, date, price, reason="manual", partial=1.0):
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
            "code": code, "buy_date": pos.buy_date, "buy_price": pos.buy_price,
            "sell_date": date, "sell_price": price, "shares": sell_shares,
            "pnl": pnl, "pnl_pct": pnl_pct, "days_held": pos.days_held,
            "reason": reason, "weight": pos.signal_weight, "partial": partial,
        })
        self.trade_log.append({
            "action": "SELL", "date": date, "code": code, "price": price,
            "shares": sell_shares, "proceeds": proceeds,
            "pnl": pnl, "pnl_pct": pnl_pct, "reason": reason,
            "cash_after": self.cash,
        })

    def check_stops(self, date):
        regime = self.get_oamv_regime(date)
        stop_ratio = self.get_candle_stop_ratio(regime)
        to_sell = []

        for code, pos in self.positions.items():
            df = self.stock_data.get(code)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            close_today = float(row["close"])

            buy_idx = df.index.get_loc(pos.buy_date) if pos.buy_date in df.index else None
            today_idx = df.index.get_loc(date) if date in df.index else None
            if buy_idx is not None and today_idx is not None:
                pos.days_held = today_idx - buy_idx

            is_buy_day = (date == pos.buy_date)
            cum_return = (close_today / pos.buy_price - 1)

            # ── 防守模式: 不强制清仓，仅依赖收紧的止损 ──

            # ── 60日检查点 ──
            if not is_buy_day and pos.days_held >= 60 and cum_return < 0.15:
                to_sell.append((code, date, close_today,
                               f"60日弱(T+{pos.days_held}, {cum_return*100:+.0f}%)", 1.0))
                continue

            # ── 252日硬限 ──
            if not is_buy_day and pos.days_held >= 252:
                to_sell.append((code, date, close_today,
                               f"1年清仓(T+{pos.days_held})", 1.0))
                continue

            # ── 蜡烛止损（OAMV 调整）──
            candle_stop = pos.entry_low * stop_ratio
            candle_triggered = close_today < candle_stop
            signal_triggered = close_today < pos.signal_close
            yellow = float(row.get("yellow_line", 0))
            yellow_triggered = yellow > 0 and close_today < yellow

            if candle_triggered:
                reason = f"蜡烛破位({stop_ratio:.0%}缓冲, T+{pos.days_held})"
            elif signal_triggered:
                reason = f"破信号收盘(T+{pos.days_held})"
            elif yellow_triggered:
                reason = f"破黄线(T+{pos.days_held})"
            else:
                continue

            if is_buy_day:
                next_dates = df.index[df.index > date]
                if len(next_dates) > 0:
                    t2_date = next_dates[0]
                    t2_open = float(df.loc[t2_date, "open"])
                    to_sell.append((code, t2_date, t2_open, f"买入日{reason}(T+2)", 1.0))
                else:
                    to_sell.append((code, date, close_today, f"买入日{reason}", 1.0))
            else:
                to_sell.append((code, date, close_today, reason, 1.0))

        for code, exit_date, exit_price, reason, partial in to_sell:
            if code in self.positions:
                self.sell(code, exit_date, exit_price, reason)

    def rotate_if_needed(self, signal):
        if len(self.positions) < self.max_positions:
            return True
        weakest = self.find_weakest_position()
        if weakest is None:
            return True
        if signal["weight"] > weakest.signal_weight + 1.0:
            sell_date = signal["t1_date"]
            sell_price = self.get_price(weakest.code, sell_date, "close")
            if sell_price is None:
                sell_price = self.get_price(weakest.code, sell_date, "open")
                if sell_price is None:
                    return False
            self.sell(weakest.code, sell_date, sell_price, "换仓")
            return True
        return False

    def calc_portfolio_value(self, date):
        holdings_value = 0
        for code, pos in self.positions.items():
            price = self.get_price(code, date, "close")
            if price is not None:
                holdings_value += pos.shares * price
            else:
                holdings_value += pos.cost
        return self.cash + holdings_value

    def run(self):
        label = "OAMV增强" if self.use_oamv else "基准"
        print(f"\n{'='*70}")
        print(f"  {label} | {self.start_date.date()} → {self.end_date.date()}")
        print(f"  资金: {self.initial_capital:,.0f}  持仓: {self.min_positions}-{self.max_positions}只")
        print(f"{'='*70}")

        signals = self.scan_signals()
        if not signals:
            print(f"  [{label}] 无信号!")
            return

        all_dates = set()
        for s in signals:
            all_dates.add(s["signal_date"])
            all_dates.add(s["t1_date"])
        trading_dates = sorted(d for d in all_dates
                               if self.start_date <= d <= self.end_date + pd.Timedelta(days=10))

        pending = []
        for date in trading_dates:
            regime = self.get_oamv_regime(date)
            self.current_regime = regime

            self.check_stops(date)

            # 防守模式只收紧止损，不阻断买入（防御日信号质量反而高）
            if not self.can_buy_in_regime(regime):
                total_value = self.calc_portfolio_value(date)
                self.daily_values.append({
                    "date": date, "value": total_value,
                    "cash": self.cash, "positions": len(self.positions),
                    "regime": regime,
                })
                continue

            # Retry pending
            still_pending = []
            for s in pending:
                if (date - s["signal_date"]).days > 3:
                    continue
                if s["code"] in self.positions:
                    continue
                if self.can_add_position(s["weight"]):
                    if self.buy(s, date, regime):
                        continue
                elif self.rotate_if_needed(s):
                    if self.buy(s, date, regime):
                        continue
                still_pending.append(s)
            pending = still_pending

            # New signals
            new_pending = [s for s in signals if s["t1_date"] == date]
            for signal in new_pending:
                if signal["code"] in self.positions:
                    continue
                # 猛干模式: 不强过滤弱信号；防守模式已跳过
                if self.can_add_position(signal["weight"]):
                    if self.buy(signal, date, regime):
                        continue
                elif self.rotate_if_needed(signal):
                    if self.buy(signal, date, regime):
                        continue
                pending.append(signal)

            total_value = self.calc_portfolio_value(date)
            self.daily_values.append({
                "date": date, "value": total_value,
                "cash": self.cash, "positions": len(self.positions),
                "regime": regime,
            })

        # 期末清仓
        final_date = trading_dates[-1] if trading_dates else self.end_date
        for code in list(self.positions.keys()):
            price = self.get_price(code, final_date, "close")
            if price is None:
                price = self.get_price(code, final_date, "open")
            if price is not None:
                self.sell(code, final_date, price, "期末清仓")

        return self._compute_stats(label)

    def _compute_stats(self, label):
        final_value = sum(p.get("proceeds", 0) for p in self.closed_positions) + self.cash
        total_return = (self.cash / self.initial_capital - 1) * 100

        if self.closed_positions:
            wins = [p for p in self.closed_positions if p["pnl"] > 0]
            losses = [p for p in self.closed_positions if p["pnl"] <= 0]
            wr = len(wins) / len(self.closed_positions) * 100
            avg_win = np.mean([p["pnl_pct"] for p in wins]) if wins else 0
            avg_loss = np.mean([p["pnl_pct"] for p in losses]) if losses else 0
            total_pnl = sum(p["pnl"] for p in self.closed_positions)
        else:
            wr = avg_win = avg_loss = total_pnl = 0

        # Max drawdown
        if len(self.daily_values) > 1:
            vals = [v["value"] for v in self.daily_values]
            peak = vals[0]
            max_dd = 0
            for v in vals:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak * 100
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        n_trades = len(self.closed_positions)

        # Regime stats
        regime_days = {"aggressive": 0, "defensive": 0, "normal": 0}
        for dv in self.daily_values:
            r = dv.get("regime", "normal")
            regime_days[r] = regime_days.get(r, 0) + 1

        return {
            "label": label,
            "final_cash": self.cash,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "win_rate": wr,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "n_trades": n_trades,
            "n_buys": len([t for t in self.trade_log if t["action"] == "BUY"]),
            "n_sells": len([t for t in self.trade_log if t["action"] == "SELL"]),
            "regime_days": regime_days,
            "daily_values": self.daily_values,
            "trade_log": self.trade_log,
            "closed_positions": self.closed_positions,
        }


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="0AMV 活筹指数增强回测")
    p.add_argument("--capital", type=float, default=100_000, help="初始资金")
    p.add_argument("--max-positions", type=int, default=4)
    p.add_argument("--min-positions", type=int, default=3)
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-05-07")
    args = p.parse_args()

    # Load OAMV data
    print("加载 0AMV 数据...", flush=True)
    from quant_strategy.oamv import calc_oamv, generate_signals, fetch_market_data
    oamv_raw = fetch_market_data(start="20240101", end=args.end)
    oamv_df = calc_oamv(oamv_raw)
    oamv_df = generate_signals(oamv_df)

    # Show regime summary
    chg = oamv_df["oamv_change_pct"].dropna()
    agg_days = (chg > 3.0).sum()
    def_days = (chg < -2.35).sum()
    print(f"  0AMV 猛干日(涨>3%): {agg_days}天  防守日(跌>2.35%): {def_days}天")

    # Run baseline
    print("\n>>> 运行基准回测...", flush=True)
    t0 = time.time()
    base_engine = OAMVSimEngine(
        capital=args.capital, max_positions=args.max_positions,
        min_positions=args.min_positions, start_date=args.start, end_date=args.end,
        oamv_df=None, use_oamv=False,
    )
    base_result = base_engine.run()
    base_time = time.time() - t0

    # Run OAMV enhanced
    print("\n>>> 运行 OAMV 增强回测...", flush=True)
    t0 = time.time()
    oamv_engine = OAMVSimEngine(
        capital=args.capital, max_positions=args.max_positions,
        min_positions=args.min_positions, start_date=args.start, end_date=args.end,
        oamv_df=oamv_df, use_oamv=True,
    )
    oamv_result = oamv_engine.run()
    oamv_time = time.time() - t0

    # ── Comparison Report ──
    print(f"\n{'='*80}")
    print(f"  OAMV 活筹指数增强 vs 基准 — 对比报告")
    print(f"  期间: {args.start} → {args.end}  |  初始资金: {args.capital:,.0f}")
    print(f"{'='*80}")

    header = f"  {'指标':<20} {'基准':>15} {'OAMV增强':>15} {'差异':>15}"
    print(header)
    print(f"  {'─'*20} {'─'*15} {'─'*15} {'─'*15}")

    for key, label, fmt in [
        ("total_return", "总收益率", "+.2f"),
        ("max_drawdown", "最大回撤", ".2f"),
        ("win_rate", "胜率", ".1f"),
        ("avg_win", "平均盈利", "+.1f"),
        ("avg_loss", "平均亏损", "+.1f"),
        ("n_trades", "完结交易数", ".0f"),
        ("n_buys", "买入次数", ".0f"),
        ("n_sells", "卖出次数", ".0f"),
    ]:
        b_val = base_result[key]
        o_val = oamv_result[key]
        diff = o_val - b_val
        diff_str = f"{diff:{fmt}}" if isinstance(diff, (int, float)) else "-"
        b_str = f"{b_val:{fmt}}" if isinstance(b_val, (int, float)) else str(b_val)
        o_str = f"{o_val:{fmt}}" if isinstance(o_val, (int, float)) else str(o_val)
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else " "
        if key == "max_drawdown":
            arrow = "↓" if diff < 0 else "↑" if diff > 0 else " "  # lower is better
        print(f"  {label:<20} {b_str:>15} {o_str:>15} {arrow}{diff_str:>14}")

    print(f"\n  状态分布:")
    print(f"    基准: 无状态区分（全天候交易）")
    rd = oamv_result["regime_days"]
    print(f"    OAMV: 猛干{rd['aggressive']}天 / 防守{rd['defensive']}天 / 正常{rd['normal']}天")

    # Net value comparison
    b_vals = base_result["daily_values"]
    o_vals = oamv_result["daily_values"]
    if b_vals and o_vals:
        # Compute daily returns correlation
        b_dates = {v["date"]: v["value"] for v in b_vals}
        o_dates = {v["date"]: v["value"] for v in o_vals}
        common = sorted(set(b_dates) & set(o_dates))
        if len(common) > 1:
            b_nav = [b_dates[d] for d in common]
            o_nav = [o_dates[d] for d in common]
            print(f"\n  净值对比 (共{len(common)}个交易日):")
            print(f"    基准最终: {b_nav[-1]:,.0f}")
            print(f"    增强最终: {o_nav[-1]:,.0f}")
            print(f"    增强超额: {o_nav[-1] - b_nav[-1]:+,.0f} ({(o_nav[-1]/b_nav[-1]-1)*100:+.2f}%)")

    # Regime analysis
    print(f"\n  猛干日分析 (0AMV涨>3%):")
    agg_trades = [t for t in oamv_result["trade_log"]
                  if t.get("regime") == "aggressive" and t["action"] == "BUY"]
    if agg_trades:
        agg_returns = []
        for bt in agg_trades:
            sells = [s for s in oamv_result["closed_positions"]
                     if s["code"] == bt["code"] and s["buy_date"] == bt["date"]]
            if sells:
                agg_returns.append(sells[0]["pnl_pct"])
        if agg_returns:
            print(f"    猛干日买入: {len(agg_trades)}笔, 平均收益: {np.mean(agg_returns):+.1f}%, "
                  f"胜率: {sum(1 for r in agg_returns if r>0)/len(agg_returns)*100:.0f}%")
        else:
            print(f"    猛干日买入: {len(agg_trades)}笔")
    else:
        print(f"    无猛干日入场")

    print(f"\n  防守日分析 (0AMV跌>2.35%):")
    def_sells = [t for t in oamv_result["trade_log"]
                 if t["action"] == "SELL" and "防守" in t.get("reason", "")]
    print(f"    防守清仓: {len(def_sells)}笔")

    # Export
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(base_result["trade_log"]).to_csv(
        os.path.join(OUT, "oamv_backtest_baseline.csv"), index=False)
    pd.DataFrame(oamv_result["trade_log"]).to_csv(
        os.path.join(OUT, "oamv_backtest_enhanced.csv"), index=False)
    print(f"\n  交易记录已导出: output/oamv_backtest_*.csv")


if __name__ == "__main__":
    main()
