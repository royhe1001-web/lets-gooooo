#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0AMV 活筹指数（活跃市值指数）—— 指南针经典大盘资金面指标

核心公式:
  0AMV = SMA(沪深两市总成交额, 10, 1)

其中 SMA 为通达信标准加权移动平均，项目已有的 numba 加速实现。

交易逻辑:
  - 0AMV 持续在 5日成本均线上方 → 资金持续入场，多头区域，可积极操作
  - 0AMV 持续在 5日成本均线下方 → 资金持续出逃，空头区域，建议持币观望
  - 0AMV 走平 → 变盘信号，密切关注方向选择
  - 顶背离（指数涨但 0AMV 跌）→ 上升行情将见顶
  - 底背离（指数跌但 0AMV 涨）→ 下跌行情将见底

用法:
  python quant_strategy/oamv.py                  # 显示最新状态
  python quant_strategy/oamv.py --plot           # 显示+保存图表
  python quant_strategy/oamv.py --start 20240101 # 指定起始日期
  python quant_strategy/oamv.py --csv            # 导出CSV
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from quant_strategy.indicators import sma

logger = logging.getLogger("oamv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE_PATH = ROOT / "output" / "market_turnover.csv"
CHART_PATH = ROOT / "output" / "oamv_chart.png"

# ============================================================
# 1. 数据获取
# ============================================================


def _fetch_index_ts(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """通过 tushare 获取指数日线数据（含成交额）。"""
    import tushare as ts

    ts_token = "c2a96ee78a444f422063ba90ed4460a8f252f3fd136ea893f6c310ae"
    ts.set_token(ts_token)
    pro = ts.pro_api()

    for attempt in range(1, 4):
        try:
            df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
            if df is None or df.empty:
                raise ValueError("空响应")
            break
        except Exception as e:
            logger.warning("tushare 获取 %s 失败(%d/3): %s", ts_code, attempt, e)
            time.sleep(random.uniform(1, 3) * attempt)
    else:
        raise RuntimeError(f"tushare 连续三次获取 {ts_code} 失败")

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    # tushare index_daily amount 单位: 千元，转为 元
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_market_data(start: str = "20190101", end: Optional[str] = None) -> pd.DataFrame:
    """
    获取沪深两市日频总成交额。

    数据源: 东方财富 HTTP API（上证综指 1.000001 + 深证综指 0.399106）
    缓存: output/market_turnover.csv（增量更新）
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")

    # 增量：从缓存最后日期开始
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        last_date = cached["date"].max().strftime("%Y%m%d")
        if last_date >= end:
            logger.info("缓存已是最新 (%s → %s)", cached["date"].min().date(), cached["date"].max().date())
            return cached
        start = last_date
        logger.info("增量更新: %s → %s", start, end)

    logger.info("获取沪深两市指数数据: %s → %s", start, end)
    # 上证综指 000001.SH, 深证综指 399106.SZ
    sh = _fetch_index_ts("000001.SH", start, end)
    sz = _fetch_index_ts("399106.SZ", start, end)

    # 合并：总成交额 = 上证成交额 + 深证成交额
    df = pd.DataFrame({
        "date": sh["date"],
        "close_sh": sh["close"].values,
        "amount_sh": sh["amount"].values,
        "amount_sz": sz["amount"].values,
        "amount_total": sh["amount"].values + sz["amount"].values,
    })

    # 合并缓存
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        df = pd.concat([cached, df], ignore_index=True).drop_duplicates(subset="date").sort_values("date")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    logger.info("缓存已更新: %s, %d 条记录", CACHE_PATH, len(df))
    return df


# ============================================================
# 2. 0AMV 指标计算
# ============================================================


def calc_oamv(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    计算 0AMV 活跃市值指数。

    参数
    ----
    df  : 必须包含 'amount_total' 列（沪深总成交额）
    n   : SMA 周期，默认 10

    返回
    ----
    DataFrame, 新增列:
        oamv           : 活跃市值（SMA 平滑后的总成交额，单位：亿元）
        oamv_cyc5      : 5日成本均线（EMA）
        oamv_cyc13     : 13日成本均线
        oamv_prev      : 前一日 OAMV
        oamv_diff      : OAMV 与 CYC5 差值
        oamv_trend     : 趋势状态 (1=多头, -1=空头, 0=震荡)
        oamv_change_pct: OAMV 日涨跌幅%
        oamv_direction : 连续上升/下降天数（正=上升，负=下降）
    """
    result = df.copy()

    # 核心：SMA(总成交额, 10, 1)，转为亿元便于阅读
    amount_total = result["amount_total"]
    result["oamv"] = sma(amount_total, n=n, m=1) / 1e8

    # 成本均线
    result["oamv_cyc5"] = result["oamv"].ewm(span=5, adjust=False).mean()
    result["oamv_cyc13"] = result["oamv"].ewm(span=13, adjust=False).mean()

    # 前一日值
    result["oamv_prev"] = result["oamv"].shift(1)

    # 差值
    result["oamv_diff"] = result["oamv"] - result["oamv_cyc5"]

    # 日涨跌幅
    result["oamv_change_pct"] = result["oamv"].pct_change() * 100

    # 趋势状态
    above_cyc5 = result["oamv"] > result["oamv_cyc5"]
    below_cyc5 = result["oamv"] < result["oamv_cyc5"]
    result["oamv_trend"] = 0
    result.loc[above_cyc5, "oamv_trend"] = 1
    result.loc[below_cyc5, "oamv_trend"] = -1

    # 连续方向计数（正=连续上升天数，负=连续下降天数）
    direction = np.sign(result["oamv"].diff().fillna(0))
    streak = np.zeros(len(direction), dtype=int)
    cnt = 0
    for i in range(len(direction)):
        if direction.iloc[i] == 0:
            cnt = 0
        elif i == 0 or direction.iloc[i] == direction.iloc[i - 1]:
            cnt += 1
        else:
            cnt = 1
        streak[i] = cnt * int(direction.iloc[i])
    result["oamv_direction"] = streak

    return result


# ============================================================
# 3. 交易信号
# ============================================================


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 0AMV 生成交易信号。

    新增列:
        oamv_signal       : 1=多头入场, -1=空头出场, 0=无信号
        oamv_divergence   : 1=顶背离, -1=底背离, 0=无
        oamv_zone         : 区域标签
    """
    result = df.copy()
    result["oamv_signal"] = 0

    # 金叉：0AMV 上穿 CYC5
    cross_up = (
        (result["oamv"] > result["oamv_cyc5"])
        & (result["oamv"].shift(1) <= result["oamv_cyc5"].shift(1))
    )
    result.loc[cross_up, "oamv_signal"] = 1

    # 死叉：0AMV 下穿 CYC5
    cross_down = (
        (result["oamv"] < result["oamv_cyc5"])
        & (result["oamv"].shift(1) >= result["oamv_cyc5"].shift(1))
    )
    result.loc[cross_down, "oamv_signal"] = -1

    # 背离检测（5日窗口）
    result["oamv_divergence"] = 0
    close_chg = result["close_sh"].pct_change(5) * 100
    oamv_chg = result["oamv"].pct_change(5) * 100

    top_div = (close_chg > 2) & (oamv_chg < -2)  # 指数涨但资金出逃
    bot_div = (close_chg < -2) & (oamv_chg > 2)  # 指数跌但资金入场
    result.loc[top_div, "oamv_divergence"] = 1
    result.loc[bot_div, "oamv_divergence"] = -1

    # 区域标签
    def _zone(row):
        t = row["oamv_trend"]
        if t == 1 and row["oamv_cyc5"] > row["oamv_cyc13"]:
            return "持续上涨区"
        elif t == 1:
            return "上涨犹豫区"
        elif t == -1 and row["oamv_cyc5"] < row["oamv_cyc13"]:
            return "持续下跌区"
        elif t == -1:
            return "下跌犹豫区"
        return "震荡"

    result["oamv_zone"] = result.apply(_zone, axis=1)

    return result


# ============================================================
# 4. 状态摘要
# ============================================================


def print_summary(df: pd.DataFrame) -> None:
    """打印当前 0AMV 状态摘要。"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_map = {1: "多头", -1: "空头", 0: "震荡"}
    signal_map = {1: "金叉 ↑", -1: "死叉 ↓", 0: "无"}

    recent = df.tail(20)
    cross_up_count = (recent["oamv_signal"] == 1).sum()
    cross_down_count = (recent["oamv_signal"] == -1).sum()
    div_count = (recent["oamv_divergence"] != 0).sum()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           指南针 0AMV 活跃市值（活筹指数）                 ║
╠══════════════════════════════════════════════════════════╣
║  日期       : {last['date'].strftime('%Y-%m-%d'):>34s}  ║
║  上证指数   : {last['close_sh']:>30.2f}  ║
║  总成交额   : {last['amount_total']/1e8:>30.2f} 亿元    ║
╠══════════════════════════════════════════════════════════╣
║  0AMV       : {last['oamv']:>30.2f} 亿元    ║
║  CYC5       : {last['oamv_cyc5']:>30.2f} 亿元    ║
║  CYC13      : {last['oamv_cyc13']:>30.2f} 亿元    ║
║  日涨跌     : {last['oamv_change_pct']:>+30.2f}%         ║
╠══════════════════════════════════════════════════════════╣
║  趋势       : {trend_map[last['oamv_trend']]:>34s}  ║
║  区域       : {last['oamv_zone']:>34s}  ║
║  当日信号   : {signal_map[last['oamv_signal']]:>34s}  ║
║  背离       : {last['oamv_divergence']:>34d}  ║
╠══════════════════════════════════════════════════════════╣
║  近20日统计:                                              ║
║    金叉: {cross_up_count:<3d}  死叉: {cross_down_count:<3d}  背离: {div_count:<3d}                        ║
╚══════════════════════════════════════════════════════════╝
""")

    # 仓位建议
    zone = last["oamv_zone"]
    suggestions = {
        "持续上涨区": "重仓持股，场外资金持续入场",
        "上涨犹豫区": "逢低买入，仓位 ≤50%，CYC5 尚未站稳 CYC13",
        "持续下跌区": "轻仓/空仓观望，场内资金持续出逃",
        "下跌犹豫区": "逢高卖出，仓位 ≤50%，关注是否转入持续下跌",
        "震荡": "观望为主，等待方向选择",
    }
    print(f"  操作建议: {suggestions.get(zone, '观望')}")


# ============================================================
# 5. 可视化
# ============================================================


def plot_oamv(df: pd.DataFrame, save: bool = True) -> None:
    """绘制 0AMV 三子图并保存。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")
        return

    # 中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti SC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("指南针 0AMV 活跃市值（活筹指数）", fontsize=16, fontweight="bold")

    # 图1: 上证指数
    ax = axes[0]
    ax.plot(df["date"], df["close_sh"], color="black", linewidth=0.8, label="上证综指")
    ax.set_ylabel("上证综指", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图2: 0AMV + CYC5 + CYC13
    ax = axes[1]
    ax.fill_between(
        df["date"], df["oamv"], df["oamv_cyc5"],
        where=df["oamv"] > df["oamv_cyc5"],
        color="red", alpha=0.15, label="多头区域",
    )
    ax.fill_between(
        df["date"], df["oamv"], df["oamv_cyc5"],
        where=df["oamv"] < df["oamv_cyc5"],
        color="blue", alpha=0.15, label="空头区域",
    )
    ax.plot(df["date"], df["oamv"], color="black", linewidth=1.0, label="0AMV")
    ax.plot(df["date"], df["oamv_cyc5"], color="#FF6600", linewidth=0.8, linestyle="--", label="CYC5")
    ax.plot(df["date"], df["oamv_cyc13"], color="#9966CC", linewidth=0.8, linestyle=":", label="CYC13")

    # 标记金叉/死叉
    cross_up = df[df["oamv_signal"] == 1]
    cross_down = df[df["oamv_signal"] == -1]
    ax.scatter(cross_up["date"], cross_up["oamv"], color="red", marker="^", s=40, zorder=5, label="金叉")
    ax.scatter(cross_down["date"], cross_down["oamv"], color="green", marker="v", s=40, zorder=5, label="死叉")

    ax.set_ylabel("0AMV (亿元)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 图3: 趋势状态
    ax = axes[2]
    colors = ["red" if t == 1 else "blue" if t == -1 else "gray" for t in df["oamv_trend"]]
    ax.bar(df["date"], df["oamv_trend"], color=colors, width=0.8)
    ax.set_ylabel("趋势", fontsize=11)
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["空头", "震荡", "多头"])
    ax.grid(True, alpha=0.3, axis="y")

    # 标记背离
    top_div = df[df["oamv_divergence"] == 1]
    bot_div = df[df["oamv_divergence"] == -1]
    ax.scatter(top_div["date"], [1.3] * len(top_div), color="orange", marker="v", s=50, label="顶背离", zorder=5)
    ax.scatter(bot_div["date"], [-1.3] * len(bot_div), color="lime", marker="^", s=50, label="底背离", zorder=5)
    ax.legend(loc="upper left", fontsize=8)

    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    if save:
        CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
        logger.info("图表已保存: %s", CHART_PATH)
    else:
        plt.show()
    plt.close()


# ============================================================
# 6. CLI 入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="指南针 0AMV 活跃市值（活筹指数）")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument("--plot", action="store_true", help="生成图表")
    parser.add_argument("--csv", action="store_true", help="导出 0AMV 数据到 CSV")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存，强制重新获取")
    args = parser.parse_args()

    if args.no_cache and CACHE_PATH.exists():
        CACHE_PATH.unlink()
        logger.info("已清除缓存")

    print("正在获取沪深两市数据...")
    df = fetch_market_data(start=args.start, end=args.end)

    print("正在计算 0AMV 指标...")
    df = calc_oamv(df)
    df = generate_signals(df)

    # 输出
    print_summary(df)

    if args.csv:
        csv_out = ROOT / "output" / "oamv_data.csv"
        cols = [
            "date", "close_sh", "amount_total", "oamv", "oamv_cyc5",
            "oamv_cyc13", "oamv_trend", "oamv_signal", "oamv_divergence", "oamv_zone",
        ]
        df[cols].to_csv(csv_out, index=False, encoding="utf-8-sig")
        logger.info("CSV 已导出: %s", csv_out)

    if args.plot:
        print("正在生成图表...")
        plot_oamv(df, save=True)

    # 返回 df 供外部调用
    return df


def get_oamv_snapshot() -> dict:
    """
    供策略模块调用的快照接口。
    返回最新一日的 0AMV 关键值，用于仓位管理/市场过滤。

    用法:
        from quant_strategy.oamv import get_oamv_snapshot
        snap = get_oamv_snapshot()
        if snap['oamv_trend'] == 1:
            ...  # 多头区域，可积极操作
    """
    df = fetch_market_data()
    df = calc_oamv(df)
    df = generate_signals(df)
    last = df.iloc[-1]
    return {
        "date": last["date"],
        "close_sh": last["close_sh"],
        "amount_total": last["amount_total"],
        "oamv": last["oamv"],
        "oamv_cyc5": last["oamv_cyc5"],
        "oamv_trend": last["oamv_trend"],
        "oamv_zone": last["oamv_zone"],
        "oamv_signal": last["oamv_signal"],
        "oamv_divergence": last["oamv_divergence"],
    }


if __name__ == "__main__":
    main()
