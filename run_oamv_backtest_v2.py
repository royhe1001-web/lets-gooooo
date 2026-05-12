#!/usr/bin/env python3
"""
0AMV 活筹指数增强回测 v2 — 趋势区域 + 仓位管理

对比方案:
  baseline    : b2 基准（max=4, min=3, hold=60, stop=0.97）
  v1_zone_pos : 持续上涨区→max+1, 持续下跌区→max-1 & min-1
  v2_zone_hold: 持续上涨区→hold=90, 持续下跌区→hold=45
  v3_combined : v1 + v2 组合

用法:
  python simulation/run_oamv_backtest_v2.py --start 2024-01-01 --end 2026-05-07
"""

import argparse, os, sys, time, warnings
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)

from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_spring import generate_spring_signals
from quant_strategy.oamv import calc_oamv, generate_signals, fetch_market_data

DATA_DIR = os.path.join(ROOT, "fetch_kline", "stock_kline")
OUT = os.path.join(ROOT, "output")

COMMISSION_BUY = 0.0003
COMMISSION_SELL = 0.0013


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
    days_held: int = 0


class SharedSignals:
    """一次扫描，多方案复用"""

    def __init__(self, start_date, end_date):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.signals: List[Dict] = []
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.trading_dates: List[pd.Timestamp] = []
        self.oamv_df: pd.DataFrame = None

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
        prev = df.index[df.index <= date]
        if len(prev) > 0:
            return float(df.loc[prev[-1], price_type])
        return None

    def scan(self):
        print("扫描 b2 入场信号...", flush=True)
        codes = sorted([f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
        t0 = time.time()
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
                    nd = full_df.index[full_df.index > sd]
                    if len(nd) == 0:
                        continue
                    self.signals.append({
                        "code": code, "signal_date": sd,
                        "signal_close": float(row["close"]),
                        "weight": float(row.get("b2_position_weight", 1.0)),
                        "t1_date": nd[0],
                        "t1_open": float(full_df.loc[nd[0], "open"]),
                    })
            except Exception:
                continue
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(codes)} {time.time()-t0:.0f}s sigs={len(self.signals)}", flush=True)
        self.signals.sort(key=lambda x: x["signal_date"])
        print(f"  完成: {len(codes)}只, {time.time()-t0:.0f}s, {len(self.signals)} 信号", flush=True)

        all_dates = set()
        for s in self.signals:
            all_dates.add(s["signal_date"])
            all_dates.add(s["t1_date"])
        self.trading_dates = sorted(
            d for d in all_dates if self.start_date <= d <= self.end_date + pd.Timedelta(days=10))


# ============================================================
# Simulation runner — 纯函数式，无 cash mutation bug
# ============================================================
def simulate(shared: SharedSignals, label: str, zone_config: dict,
             stop_ratio=0.97, capital=100_000, base_max=4, base_min=3) -> dict:
    """
    zone_config: {zone_name: {max_delta, min_delta, hold_days}}
    """
    cash = capital
    positions: Dict[str, Position] = {}
    closed: List[Dict] = []
    tlog: List[Dict] = []
    nav: List[Dict] = []

    zone_map = {}
    if shared.oamv_df is not None:
        for _, r in shared.oamv_df.iterrows():
            zone_map[r["date"]] = r.get("oamv_zone", "震荡")

    def get_zone(date):
        return zone_map.get(date, "震荡")

    def get_limits(zone):
        cfg = zone_config.get(zone, {})
        mx = base_max + cfg.get("max_delta", 0)
        mn = base_min + cfg.get("min_delta", 0)
        hd = cfg.get("hold_days", 60)
        return max(1, mx), max(1, mn), hd

    pending = []
    for date in shared.trading_dates:
        zone = get_zone(date)
        mx, mn, hd = get_limits(zone)

        # ── Check stops ──
        to_sell = []
        for code, pos in positions.items():
            df = shared.stock_data.get(code)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            close_today = float(row["close"])

            bi = df.index.get_loc(pos.buy_date) if pos.buy_date in df.index else None
            ti = df.index.get_loc(date) if date in df.index else None
            if bi is not None and ti is not None:
                pos.days_held = ti - bi

            is_buy_day = (date == pos.buy_date)
            cum_ret = (close_today / pos.buy_price - 1)

            # Zone-adjusted hold check
            if not is_buy_day and pos.days_held >= hd and cum_ret < 0.15:
                to_sell.append((code, date, close_today,
                               f'{hd}d弱(T+{pos.days_held},{cum_ret*100:+.0f}%)', 1.0))
                continue
            # 252 hard limit
            if not is_buy_day and pos.days_held >= 252:
                to_sell.append((code, date, close_today, '1年清仓', 1.0))
                continue
            # Stops
            candle_hit = close_today < pos.entry_low * stop_ratio
            signal_hit = close_today < pos.signal_close
            yellow = float(row.get("yellow_line", 0))
            yellow_hit = yellow > 0 and close_today < yellow
            if candle_hit:
                reason = f'蜡烛({stop_ratio:.0%},T+{pos.days_held})'
            elif signal_hit:
                reason = f'破信号(T+{pos.days_held})'
            elif yellow_hit:
                reason = f'破黄线(T+{pos.days_held})'
            else:
                continue
            if is_buy_day:
                nd = df.index[df.index > date]
                if len(nd) > 0:
                    to_sell.append((code, nd[0], float(df.loc[nd[0], "open"]),
                                   f'买日{reason}(T+2)', 1.0))
            else:
                to_sell.append((code, date, close_today, reason, 1.0))

        for code, xd, xp, reason, partial in to_sell:
            if code not in positions:
                continue
            pos = positions[code]
            ss = int(pos.shares * partial)
            if ss < 100:
                continue
            proceeds = ss * xp * (1 - COMMISSION_SELL)
            cp = pos.cost * partial
            pnl = proceeds - cp
            pnl_pct = (pnl / cp) * 100 if cp > 0 else 0
            cash += proceeds
            if partial >= 1.0:
                positions.pop(code)
            closed.append({
                "code": code, "buy_date": pos.buy_date, "buy_price": pos.buy_price,
                "sell_date": xd, "sell_price": xp, "shares": ss,
                "pnl": pnl, "pnl_pct": pnl_pct, "days_held": pos.days_held,
                "reason": reason, "weight": pos.signal_weight, "zone": zone,
            })
            tlog.append({
                "action": "SELL", "date": xd, "code": code, "price": xp,
                "shares": ss, "proceeds": proceeds, "pnl": pnl,
                "pnl_pct": pnl_pct, "reason": reason, "zone": zone,
            })

        # ── Process pending ──
        still_pending = []
        for s in pending:
            if (date - s["signal_date"]).days > 3:
                continue
            if s["code"] in positions:
                continue
            if len(positions) < mn or (len(positions) < mx) or s["weight"] >= 3.0:
                ok, cost = _do_buy(shared, s, date, positions, tlog, zone, cash, base_min)
                if ok:
                    cash -= cost
                    continue
            ok, net_cash = _do_rotate(shared, s, date, positions, tlog, zone, cash, base_min)
            if ok:
                cash += net_cash
                continue
            still_pending.append(s)
        pending = still_pending

        # ── New signals ──
        for s in [x for x in shared.signals if x["t1_date"] == date]:
            if s["code"] in positions:
                continue
            if len(positions) < mn or (len(positions) < mx) or s["weight"] >= 3.0:
                ok, cost = _do_buy(shared, s, date, positions, tlog, zone, cash, base_min)
                if ok:
                    cash -= cost
                    continue
            ok, net_cash = _do_rotate(shared, s, date, positions, tlog, zone, cash, base_min)
            if ok:
                cash += net_cash
                continue
            pending.append(s)

        # NAV
        hv = 0
        for code, pos in positions.items():
            p = shared.get_price(code, date, "close")
            hv += pos.shares * (p if p is not None else pos.buy_price)
        nav.append({"date": date, "value": cash + hv, "cash": cash,
                     "positions": len(positions), "zone": zone})

    # ── Final liquidation ──
    fd = shared.trading_dates[-1] if shared.trading_dates else shared.end_date
    for code in list(positions.keys()):
        p = shared.get_price(code, fd, "close") or shared.get_price(code, fd, "open")
        if p is not None:
            pos = positions[code]
            proceeds = pos.shares * p * (1 - COMMISSION_SELL)
            cash += proceeds
            closed.append({
                "code": code, "buy_date": pos.buy_date, "buy_price": pos.buy_price,
                "sell_date": fd, "sell_price": p, "shares": pos.shares,
                "pnl": proceeds - pos.cost,
                "pnl_pct": (proceeds / pos.cost - 1) * 100 if pos.cost > 0 else 0,
                "days_held": pos.days_held, "reason": "期末", "weight": pos.signal_weight,
            })

    return _compute_stats(label, cash, capital, closed, tlog, nav)


def _do_buy(shared, signal, date, positions, tlog, zone, available_cash, base_min):
    """买入。返回 (success, cost)。调用方负责 cash -= cost"""
    bp = signal["t1_open"]
    base_alloc = available_cash / max(base_min, 1)
    wf = min(signal["weight"] / 2.0, 2.0)
    alloc = min(base_alloc * wf, available_cash * 0.95)
    shares = int(alloc / bp / 100) * 100
    if shares < 100:
        shares = 100 if available_cash >= bp * 100 * 1.01 else 0
    if shares == 0:
        return False, 0
    cost = shares * bp * (1 + COMMISSION_BUY)
    if cost > available_cash:
        shares = int(available_cash * 0.95 / bp / 100) * 100
        if shares < 100:
            return False, 0
        cost = shares * bp * (1 + COMMISSION_BUY)

    df = shared.stock_data.get(signal["code"])
    entry_low = bp
    if df is not None and signal["t1_date"] in df.index:
        entry_low = float(df.loc[signal["t1_date"], "low"])

    positions[signal["code"]] = Position(
        code=signal["code"], signal_date=signal["signal_date"],
        signal_close=signal["signal_close"], signal_weight=signal["weight"],
        buy_date=signal["t1_date"], buy_price=bp,
        shares=shares, cost=cost, entry_low=entry_low, peak_price=bp,
    )
    tlog.append({"action": "BUY", "date": signal["t1_date"], "code": signal["code"],
                 "price": bp, "shares": shares, "cost": cost,
                 "weight": signal["weight"], "zone": zone})
    return True, cost


def _do_rotate(shared, signal, date, positions, tlog, zone, available_cash, base_min):
    """换仓。返回 (success, net_cash_change)。调用方负责 cash += net_cash_change"""
    if not positions:
        ok, cost = _do_buy(shared, signal, date, positions, tlog, zone, available_cash, base_min)
        return ok, -cost
    weakest = min(positions.values(), key=lambda p: p.signal_weight)
    if signal["weight"] <= weakest.signal_weight + 1.0:
        return False, 0
    sp = shared.get_price(weakest.code, signal["t1_date"], "close")
    if sp is None:
        return False, 0
    pos = positions.pop(weakest.code)
    proceeds = pos.shares * sp * (1 - COMMISSION_SELL)
    tlog.append({
        "action": "SELL", "date": signal["t1_date"], "code": weakest.code,
        "price": sp, "shares": pos.shares, "proceeds": proceeds,
        "pnl": proceeds - pos.cost,
        "pnl_pct": (proceeds / pos.cost - 1) * 100 if pos.cost > 0 else 0,
        "reason": "换仓", "zone": zone,
    })
    ok, cost = _do_buy(shared, signal, date, positions, tlog, zone,
                       available_cash + proceeds, base_min)
    if ok:
        return True, proceeds - cost
    # Buy failed, put back and return proceeds
    positions[weakest.code] = pos
    return False, 0


def _compute_stats(label, cash, initial_capital, closed, tlog, nav):
    if closed:
        wins = [p for p in closed if p["pnl"] > 0]
        losses = [p for p in closed if p["pnl"] <= 0]
        wr = len(wins) / len(closed) * 100
        avg_win = np.mean([p["pnl_pct"] for p in wins]) if wins else 0
        avg_loss = np.mean([p["pnl_pct"] for p in losses]) if losses else 0
        total_pnl = sum(p["pnl"] for p in closed)
    else:
        wr = avg_win = avg_loss = total_pnl = 0

    tr = (cash / initial_capital - 1) * 100

    if len(nav) > 1:
        vals = [v["value"] for v in nav]
        peak = vals[0]
        max_dd = 0
        for v in vals:
            if v > peak:
                peak = v
            max_dd = max(max_dd, (peak - v) / peak * 100)
    else:
        max_dd = 0

    n_buys = len([t for t in tlog if t["action"] == "BUY"])
    n_sells = len([t for t in tlog if t["action"] == "SELL"])

    zs = defaultdict(lambda: {"buys": 0, "sells": 0, "pnls": []})
    for t in tlog:
        z = t.get("zone", "?")
        if t["action"] == "BUY":
            zs[z]["buys"] += 1
        else:
            zs[z]["sells"] += 1
            if "pnl_pct" in t:
                zs[z]["pnls"].append(t["pnl_pct"])

    return {
        "label": label, "final_cash": cash, "total_return": tr,
        "max_drawdown": max_dd, "win_rate": wr,
        "avg_win": avg_win, "avg_loss": avg_loss, "total_pnl": total_pnl,
        "n_trades": len(closed), "n_buys": n_buys, "n_sells": n_sells,
        "nav": nav, "zone_stats": dict(zs), "tlog": tlog,
    }


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="0AMV 趋势区域增强回测 v2")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--max", type=int, default=4)
    p.add_argument("--min", type=int, default=3)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-05-07")
    args = p.parse_args()

    print("加载 0AMV...", flush=True)
    oamv_raw = fetch_market_data(start="20240101", end=args.end)
    oamv_df = calc_oamv(oamv_raw)
    oamv_df = generate_signals(oamv_df)

    print("\n>>> 扫描信号...", flush=True)
    shared = SharedSignals(args.start, args.end)
    shared.oamv_df = oamv_df
    shared.scan()

    configs = {
        "baseline": ({  # all default
            "持续上涨区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "上涨犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "持续下跌区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "下跌犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "震荡":      {"max_delta": 0, "min_delta": 0, "hold_days": 60},
        }, 0.97),
        "v1_zone_pos": ({
            "持续上涨区": {"max_delta": +1, "min_delta": 0, "hold_days": 60},
            "上涨犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "持续下跌区": {"max_delta": -1, "min_delta": -1, "hold_days": 60},
            "下跌犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "震荡":      {"max_delta": 0, "min_delta": 0, "hold_days": 60},
        }, 0.97),
        "v2_zone_hold": ({
            "持续上涨区": {"max_delta": 0, "min_delta": 0, "hold_days": 90},
            "上涨犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "持续下跌区": {"max_delta": 0, "min_delta": 0, "hold_days": 45},
            "下跌犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "震荡":      {"max_delta": 0, "min_delta": 0, "hold_days": 60},
        }, 0.97),
        "v3_combined": ({
            "持续上涨区": {"max_delta": +1, "min_delta": 0, "hold_days": 90},
            "上涨犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "持续下跌区": {"max_delta": -1, "min_delta": -1, "hold_days": 45},
            "下跌犹豫区": {"max_delta": 0, "min_delta": 0, "hold_days": 60},
            "震荡":      {"max_delta": 0, "min_delta": 0, "hold_days": 60},
        }, 0.97),
    }

    results = {}
    for name, (zc, sr) in configs.items():
        print(f"\n>>> {name}...", flush=True)
        t0 = time.time()
        results[name] = simulate(shared, name, zc, sr, args.capital, args.max, args.min)
        print(f"    {time.time()-t0:.0f}s", flush=True)

    # ── Report ──
    base = results["baseline"]
    print(f"\n{'='*85}")
    print(f"  0AMV 趋势区域增强 — 多方案对比")
    print(f"  {args.start} → {args.end}  资金: {args.capital:,.0f}  持仓: {args.min}-{args.max}只")
    print(f"{'='*85}")
    hdr = f"  {'方案':<16} {'总收益':>12} {'vs基准':>10} {'回撤':>8} {'vs基准':>8} {'胜率':>7} {'交易':>6}"
    print(hdr)
    print(f"  {'─'*16} {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*6}")
    for k in ["baseline", "v1_zone_pos", "v2_zone_hold", "v3_combined"]:
        r = results[k]
        td = r["total_return"] - base["total_return"]
        dd = r["max_drawdown"] - base["max_drawdown"]
        print(f"  {k:<16} {r['total_return']:>+10.2f}% {td:>+9.2f}% "
              f"{r['max_drawdown']:>7.2f}% {dd:>+7.2f}% {r['win_rate']:>6.1f}% {r['n_trades']:>5}")

    # Best variant
    best = max([k for k in results if k != "baseline"],
               key=lambda k: results[k]["total_return"] - results[k]["max_drawdown"] * 0.3)
    r = results[best]
    print(f"\n  最佳: {best}  (收益 {r['total_return']-base['total_return']:+.2f}%, "
          f"回撤 {r['max_drawdown']-base['max_drawdown']:+.2f}%)")
    for zone in ["持续上涨区", "上涨犹豫区", "持续下跌区", "下跌犹豫区"]:
        zs = r["zone_stats"].get(zone, {})
        pnls = zs.get("pnls", [])
        print(f"    {zone}: 买{zs.get('buys',0)}卖{zs.get('sells',0)} "
              f"均{np.mean(pnls):+.1f}% ({len(pnls)}笔)")

    os.makedirs(OUT, exist_ok=True)
    for k, r in results.items():
        pd.DataFrame(r["nav"]).to_csv(os.path.join(OUT, f"oamv_v2_{k}_nav.csv"), index=False)
    print(f"\n  NAV: output/oamv_v2_*_nav.csv")


if __name__ == "__main__":
    main()
