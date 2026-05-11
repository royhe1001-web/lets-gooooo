#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义技术指标计算模块
========================

将通达信指标逻辑翻译为Python实现，包括：
  - KDJ 指标
  - MACD 指标
  - 砖型图（Brick Chart）—— 4日周期摆动指标
  - 白黄线 —— 双层EMA + 四均线平均值
  - 成交量形态（倍量/地量/长阴短柱/天量比等）
  - 活跃市值（0AMV 活筹指数）: 见 quant_strategy/oamv.py，已完整实现

注：博弈K线（GameKLine）已移除以提升性能。

2026-05-04  基于"转折点交易技法修正版与量化实现.md"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ============================================================
# 1. KDJ 指标
# ============================================================
def calc_kdj(df: pd.DataFrame,
             n: int = 9,
             m1: int = 3,
             m2: int = 3) -> pd.DataFrame:
    """
    计算 KDJ 指标。

    参数
    ----
    df : DataFrame, 必须包含 'high', 'low', 'close'
    n  : RSV 周期，默认9
    m1 : K 平滑周期，默认3
    m2 : D 平滑周期，默认3

    返回
    ----
    DataFrame, 新增列 'K', 'D', 'J'
    """
    low_n = df['low'].rolling(window=n).min()
    high_n = df['high'].rolling(window=n).max()

    # RSV (Raw Stochastic Value)
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)  # 极端情况回补

    K = rsv.ewm(span=m1, adjust=False).mean()
    D = K.ewm(span=m2, adjust=False).mean()
    J = 3 * K - 2 * D

    result = df.copy()
    result['K'] = K
    result['D'] = D
    result['J'] = J
    return result


# ============================================================
# 2. MACD 指标
# ============================================================
def calc_macd(df: pd.DataFrame,
              fast: int = 12,
              slow: int = 26,
              signal: int = 9) -> pd.DataFrame:
    """
    计算 MACD 指标。

    返回
    ----
    DataFrame, 新增列 'MACD_DIF', 'MACD_DEA', 'MACD_HIST'
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)

    result = df.copy()
    result['MACD_DIF'] = dif
    result['MACD_DEA'] = dea
    result['MACD_HIST'] = hist
    return result


# ============================================================
# 3. 砖型图（Brick Chart）—— 通达信代码直译
# ============================================================
def calc_brick_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    砖型图指标 —— 4日周期摆动指标（通达信原码直译）。

    通达信原代码：
        VAR1A:=(HHV(HIGH,4)-CLOSE)/(HHV(HIGH,4)-LLV(LOW,4))*100-90;
        VAR2A:=SMA(VAR1A,4,1)+100;
        VAR3A:=(CLOSE-LLV(LOW,4))/(HHV(HIGH,4)-LLV(LOW,4))*100;
        VAR4A:=SMA(VAR3A,6,1);
        VAR5A:=SMA(VAR4A,6,1)+100;
        VAR6A:=VAR5A-VAR2A;
        砖型图:=IF(VAR6A>4,VAR6A-4,0);
        AA:=REF(砖型图,1)<砖型图;               ← 砖值上升=红柱/持股
        CC:=REF(AA,1)=0 AND AA=1;               ← 前日未上升，今日首次上升=买点

    返回
    ----
    DataFrame, 新增列:
        'brick_val'          : 砖型图指标值 IF(VAR6A>4, VAR6A-4, 0)
        'brick_var6a'        : VAR6A 原始值
        'brick_rising'       : 红柱=1 (砖值上升), 绿柱=0
        'brick_falling'      : 下降=1 (砖值下降)
        'brick_red'          : = brick_rising (兼容)
        'brick_green_to_red' : CC 买入信号 (绿转红)
        'brick_red_to_green' : 卖出信号 (红转绿)
        'brick_green_streak' : 连续绿柱天数
        'brick_red_streak'   : 连续红柱天数
    """
    hhv4 = df['high'].rolling(4).max()
    llv4 = df['low'].rolling(4).min()
    diff_range = hhv4 - llv4
    diff_range = diff_range.replace(0, np.nan)

    var1a = (hhv4 - df['close']) / diff_range * 100 - 90
    var2a = sma(var1a, n=4, m=1) + 100
    var3a = (df['close'] - llv4) / diff_range * 100
    var4a = sma(var3a, n=6, m=1)
    var5a = sma(var4a, n=6, m=1) + 100
    var6a = var5a - var2a

    # 砖型图指标值: IF(VAR6A>4, VAR6A-4, 0)
    brick_val = pd.Series(np.where(var6a.values > 4, var6a.values - 4, 0), index=df.index)

    # AA: 砖值上升 → 红柱/持股；砖值下降 → 绿柱/观望
    brick_rising = (brick_val > brick_val.shift(1)).astype(int)
    brick_falling = (brick_val < brick_val.shift(1)).astype(int)

    # CC/XG: 前日未上升，今日首次上升 → 绿转红买点
    green_to_red = ((brick_rising.shift(1) == 0) & (brick_rising == 1)).astype(int)
    # 红转绿: 前日上升，今日下降 → 卖点
    red_to_green = ((brick_rising.shift(1) == 1) & (brick_falling == 1)).astype(int)

    # 连续红/绿柱计数
    green_streak = _calc_streak(brick_rising == 0)
    red_streak = _calc_streak(brick_rising == 1)

    result = df.copy()
    result['brick_val'] = brick_val
    result['brick_var6a'] = var6a
    result['brick_rising'] = brick_rising
    result['brick_falling'] = brick_falling
    result['brick_red'] = brick_rising           # 兼容旧语义
    result['brick_green_to_red'] = green_to_red
    result['brick_red_to_green'] = red_to_green
    result['brick_green_streak'] = green_streak
    result['brick_red_streak'] = red_streak
    return result


# ============================================================
# 4. 白黄线指标
# ============================================================
def calc_white_yellow_line(df: pd.DataFrame,
                           m1: int = 14,
                           m2: int = 28,
                           m3: int = 57,
                           m4: int = 114) -> pd.DataFrame:
    """
    白黄线指标。

    通达信原代码：
        白线:EMA(EMA(C,10),10),COLORFFFFFF;
        黄线:(MA(CLOSE,14)+MA(CLOSE,28)+MA(CLOSE,57)+MA(CLOSE,114))/4;

    白线 = 双层EMA，对价格变化更敏感
    黄线 = 四均线平均值，代表中期趋势综合成本

    返回
    ----
    DataFrame, 新增列:
        'white_line'    : 白线
        'yellow_line'   : 黄线
        'white_above_yellow': 白线在黄线上方=1
        'white_direction': 白线方向 (1=向上, -1=向下, 0=平)
    """
    # 白线：双层EMA
    ema_first = df['close'].ewm(span=10, adjust=False).mean()
    white_line = ema_first.ewm(span=10, adjust=False).mean()

    # 黄线：四均线平均值
    ma14 = df['close'].rolling(m1).mean()
    ma28 = df['close'].rolling(m2).mean()
    ma57 = df['close'].rolling(m3).mean()
    ma114 = df['close'].rolling(m4).mean()
    yellow_line = (ma14 + ma28 + ma57 + ma114) / 4

    result = df.copy()
    result['white_line'] = white_line
    result['yellow_line'] = yellow_line
    result['white_above_yellow'] = (white_line > yellow_line).astype(int)
    result['white_direction'] = np.sign(white_line.diff())
    return result


# ============================================================
# 5. 成交量形态识别
# ============================================================
# 注：原 5.博弈K线 / 6.博弈K线长 已移除，以提升性能。
# GameKLine 依赖 WINNER 函数近似（250日滚动窗口逐行循环），
# 计算成本极高（占 ~75% 指标计算时间），且对入场信号非必须。
def calc_volume_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别成交量关键形态。

    返回
    ----
    DataFrame, 新增列:
        'vol_double'       : 倍量柱（当日量≥前日1.9倍 且 前日非倍量）
        'vol_floor'        : 地量（20日最低量）
        'vol_sky'          : 天量（20日最高量）
        'vol_long_yin_short': 长阴短柱（跌幅>3% 且 量<前日80%）
        'vol_shrink_after_double': 倍量后缩量（<55%=熄火）
        'vol_half_above_4d': 倍量后连续4天维持半量以上
        'amplitude'        : 当日振幅%
    """
    result = df.copy()
    vol = df['volume']
    close = df['close']

    # 倍量柱：当日量≥前日1.9倍 且前日量<前前日1.3倍（参考文档5.2）
    result['vol_double'] = (
        (vol >= vol.shift(1) * 1.9) &
        (vol.shift(1) < vol.shift(2) * 1.3)
    ).astype(int)

    # 地量：20日最低成交量
    result['vol_floor'] = (vol == vol.rolling(20).min()).astype(int)

    # 天量：20日最高成交量
    result['vol_sky'] = (vol == vol.rolling(20).max()).astype(int)

    # 长阴短柱：跌幅>3% 且成交量<前日80%
    result['vol_long_yin_short'] = (
        (close < close.shift(1) * 0.97) &
        (vol < vol.shift(1) * 0.8)
    ).astype(int)

    # 倍量后再倍量 → 天量（顶部预警）
    result['vol_double_again'] = (
        (result['vol_double'] == 1) &
        (result['vol_double'].shift(1) == 1)
    ).astype(int)

    # 倍量后缩量信号
    result['vol_after_double_shrink'] = 0
    result['vol_half_above_count'] = 0

    double_indices = result.index[result['vol_double'] == 1]
    for idx_pos in range(len(double_indices)):
        idx = double_indices[idx_pos]
        loc = result.index.get_loc(idx)
        if loc + 1 >= len(result):
            continue
        next_loc = result.index[loc + 1]
        if vol.loc[next_loc] < vol.loc[idx] * 0.55:
            result.loc[next_loc, 'vol_after_double_shrink'] = 1  # 熄火

        # 后续4天维持半量以上
        count = 0
        for offset in range(1, 5):
            check_loc = loc + offset
            if check_loc >= len(result):
                break
            check_idx = result.index[check_loc]
            if vol.loc[check_idx] >= vol.loc[idx] * 0.5:
                count += 1
        result.loc[idx, 'vol_half_above_count'] = count
        if count >= 4:
            result.loc[idx, 'vol_half_above_4d'] = 1

    result['vol_half_above_4d'] = result.get('vol_half_above_4d', pd.Series(0, index=result.index, dtype=int)).fillna(0).astype(int)

    # 天量/地量比
    vol_20_max = vol.rolling(20).max()
    vol_20_min = vol.rolling(20).min()
    result['vol_sky_floor_ratio'] = vol_20_max / vol_20_min.replace(0, np.nan)

    # 振幅
    result['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100

    return result


# ============================================================
# 6. 综合指标计算（一站式调用）
# ============================================================
def calc_all_indicators(df: pd.DataFrame,
                        board_type: str = 'main',
                        lookback: int = 250) -> pd.DataFrame:
    """
    一次性计算所有自定义指标。

    参数
    ----
    df         : 原始OHLCV数据
    board_type : 'main'(主板) / 'gem'(创业板) —— 影响振幅阈值等
    lookback   : WINNER函数近似窗口

    返回
    ----
    DataFrame, 包含所有原始列及计算后的指标列
    """
    df = calc_kdj(df)
    df = calc_macd(df)
    df = calc_brick_chart(df)
    df = calc_white_yellow_line(df)
    df = calc_volume_patterns(df)

    # 添加通用的交叉判断辅助列
    df['cross_below_yellow'] = (  # 跌破黄线
        (df['close'] < df['yellow_line']) &
        (df['close'].shift(1) >= df['yellow_line'].shift(1))
    ).astype(int)

    df['board_type'] = board_type

    return df


# ============================================================
# 辅助函数
# ============================================================
try:
    from numba import jit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def jit(*args, **kwargs):
        """Fallback no-op decorator when numba is not available."""
        def wrapper(fn):
            return fn
        return wrapper


@jit(nopython=True, cache=True)
def _sma_numba(vals, n, m):
    """numba-accelerated SMA core (operates on float64 numpy arrays)."""
    out = np.full(len(vals), np.nan)
    # Find first non-NaN
    first = -1
    for i in range(len(vals)):
        if not np.isnan(vals[i]):
            first = i
            out[first] = vals[first]
            break
    if first < 0:
        return out
    for i in range(first + 1, len(vals)):
        prev = out[i - 1]
        cur = vals[i]
        if np.isnan(prev):
            out[i] = cur
        elif np.isnan(cur):
            out[i] = prev
        else:
            out[i] = (cur * m + prev * (n - m)) / n
    return out


@jit(nopython=True, cache=True)
def _calc_streak_numba(cond_array):
    """numba-accelerated streak counter (consecutive True counts)."""
    out = np.zeros(len(cond_array), dtype=np.int32)
    count = 0
    for i in range(len(cond_array)):
        if cond_array[i]:
            count += 1
        else:
            count = 0
        out[i] = count
    return out


def sma(series: pd.Series, n: int, m: int) -> pd.Series:
    """
    SMA (Smoothed Moving Average) —— 通达信SMA函数（numba JIT 优化版）。

    公式: SMA(X, N, M) = (X*M + SMA'*(N-M)) / N
    """
    vals = series.values.astype(np.float64)
    out = _sma_numba(vals, n, m)
    return pd.Series(out, index=series.index)


def _calc_streak(condition: pd.Series, direction: str = 'same') -> pd.Series:
    """计算连续满足条件的计数（numba JIT 优化版）。"""
    cond_array = condition.values.astype(np.bool_)
    out = _calc_streak_numba(cond_array)
    return pd.Series(out, index=condition.index, dtype=int)


def cross_over(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """判断 series_a 上穿 series_b（前一刻在下方，当前在上方）。"""
    return (
        (series_a > series_b) &
        (series_a.shift(1) <= series_b.shift(1))
    ).astype(int)


def cross_under(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """判断 series_a 下穿 series_b。"""
    return (
        (series_a < series_b) &
        (series_a.shift(1) >= series_b.shift(1))
    ).astype(int)


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    # 用随机数据做冒烟测试
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='B')
    close = 10 + np.cumsum(np.random.randn(500) * 0.2)
    close = np.maximum(close, 1)

    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(500) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(500) * 0.02)),
        'low': close * (1 - np.abs(np.random.randn(500) * 0.02)),
        'close': close,
        'volume': np.random.randint(1e6, 1e7, 500),
        'turnover': np.random.uniform(0.5, 8, 500),
    }, index=dates)

    df = calc_all_indicators(df)
    print("指标计算完成，数据形状:", df.shape)
    print("指标列:", [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'turnover']])
    print()
    print("b1信号计数:", (df['J'] <= 13).sum())
    print("砖型图绿转红:", df['brick_green_to_red'].sum())
    print("倍量柱:", df['vol_double'].sum())
    print("长阴短柱:", df['vol_long_yin_short'].sum())
