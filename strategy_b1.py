#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略一：b1 超卖反转策略（短线 T+1 ~ T+5）— 优化版
====================================================

优化项:
  1. 大盘过滤: 大盘在黄线下时 b1 信号降权
  2. 弹性J值: 大盘强→收紧J≤10（要确定性），大盘弱→放宽J≤18（要赔率）
  3. T+5差异化: 高波动股放宽至8%，低波动股收紧至3%
  4. 缩量检查: 入场前5日缩量→加分，确认抛压衰竭

止损体系（两级四类，不变）:
  Tier 1: 黄线二次跌破 / S1 / S2 → 无条件清仓
  Tier 2: 放飞+滴滴 / T+5时间止损
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

try:
    from quant_strategy.indicators import calc_all_indicators
except ImportError:
    from indicators import calc_all_indicators


def generate_b1_signals(df: pd.DataFrame,
                        board_type: str = 'main',
                        use_active_market: bool = False,
                        active_market_df: Optional[pd.DataFrame] = None,
                        precomputed: bool = False,
                        market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    参数
    ----
    df          : OHLCV 日线数据
    board_type  : 'main' / 'gem'
    precomputed : 是否已预先计算指标
    market_df   : 大盘指数 DataFrame（需含 'close', 'yellow_line'），用于大盘过滤和弹性 J 值

    返回新增列:
        b1_entry_signal, b1_entry_score, b1_score_pct
        b2_add_signal, b1_stop_signal
        b1_fly_signal, b1_didi_stop, b1_t5_stop
        b1_state
    """
    if not precomputed:
        df = calc_all_indicators(df, board_type=board_type)

    # === 大盘过滤 ===
    mkt_above_yellow = True  # 默认大盘健康
    if market_df is not None and 'close' in market_df.columns:
        aligned = market_df.reindex(df.index, method='ffill')
        if 'yellow_line' in aligned.columns:
            mkt_above_yellow = bool(
                (aligned['close'].iloc[-1] > aligned['yellow_line'].iloc[-1])
                if len(aligned) > 0 else True
            )

    # === 振幅阈值 ===
    amp_max = 4.0 if board_type == 'main' else 7.0

    # ============================================================
    # 入场条件（13项 + 优化）
    # ============================================================
    conditions = {}

    # 优化1: 弹性 J 值 — 大盘强收紧，大盘弱放宽
    j_tight, j_loose = 10, 18
    j_threshold = j_tight if mkt_above_yellow else j_loose
    conditions['c1_j_le'] = df['J'] <= j_threshold

    # 条件2: J 最优值（加分）
    conditions['c2_j_optimal'] = (df['J'] == 7) | (df['J'] == 11)

    # 条件3: 股价在黄线之上
    conditions['c3_above_yellow'] = df['close'] > df['yellow_line']

    # 条件4: MACD DIF 在零轴之上
    conditions['c4_macd_above_zero'] = df['MACD_DIF'] > 0

    # 条件5: 排除 MACD 刚出第一根绿柱
    macd_first_green = (df['MACD_HIST'] < 0) & (df['MACD_HIST'].shift(1) >= 0)
    conditions['c5_not_first_green'] = ~macd_first_green

    # 条件6: 前日小阴 + 当日小阳（分歧转一致）
    prev_change = df['close'].shift(1) / df['close'].shift(2) - 1
    cur_change = df['close'] / df['close'].shift(1) - 1
    conditions['c6_doji_to_yang'] = (
        (prev_change < -0.012) & (prev_change > -0.03) &
        (cur_change > 0.01) & (cur_change < 0.03)
    )

    # 条件7: 振幅控制
    conditions['c7_amp_ok'] = df['amplitude'] <= amp_max

    # 条件8/9: 活跃市值（默认通过）
    conditions['c8_active_market'] = pd.Series(True, index=df.index)
    conditions['c9_active_market_dash'] = pd.Series(False, index=df.index)
    if use_active_market and active_market_df is not None:
        am = active_market_df.reindex(df.index, method='ffill')['active_market_pct']
        conditions['c8_active_market'] = (am > -3.5) & (am < -1.0)
        conditions['c9_active_market_dash'] = am <= -4.0

    # 条件10: 非跳空低开放量下砸
    gap_down_vol = (
        (df['open'] < df['close'].shift(1) * 0.98) &
        (df['volume'] > df['volume'].rolling(5).mean() * 1.5)
    )
    conditions['c10_no_gap_smash'] = ~gap_down_vol

    # 条件11: 排除 A 杀
    ret_5d = df['close'] / df['close'].shift(5) - 1
    conditions['c11_not_a_kill'] = ret_5d > -0.15

    # 条件12: 排除大阴线次日 b1
    big_yin_prev = df['close'].shift(1) / df['close'].shift(2) - 1 < -0.05
    conditions['c12_not_big_yin_b1'] = ~(big_yin_prev & (df['J'] <= j_threshold))

    # 优化4: 缩量检查 — 近5日均量 < 近10日均量（抛压衰竭）
    vol_shrink = df['volume'].rolling(5).mean() < df['volume'].rolling(10).mean()

    # === 评分 ===
    score = pd.Series(0.0, index=df.index)
    must_conditions = ['c1_j_le', 'c3_above_yellow', 'c10_no_gap_smash',
                       'c11_not_a_kill', 'c12_not_big_yin_b1']
    for key in must_conditions:
        score += conditions[key].astype(float)

    bonus_conditions = ['c2_j_optimal', 'c4_macd_above_zero', 'c5_not_first_green',
                        'c6_doji_to_yang', 'c7_amp_ok', 'c8_active_market',
                        'c9_active_market_dash']
    for key in bonus_conditions:
        score += conditions[key].astype(float) * 0.5

    # 缩量加分: +0.5
    score += vol_shrink.astype(float) * 0.5

    max_score = len(must_conditions) + len(bonus_conditions) * 0.5 + 0.5

    # 入场信号
    entry_signal = pd.Series(True, index=df.index)
    for key in must_conditions:
        entry_signal &= conditions[key]

    df['b1_entry_signal'] = entry_signal.astype(int)
    df['b1_entry_score'] = score
    df['b1_score_pct'] = score / max_score

    # 优化3: T+5 阈值差异化 — 高波动股放宽
    vol_20 = df['amplitude'].rolling(20).mean()
    df['b1_t5_threshold'] = 0.05  # 默认5%
    df.loc[vol_20 > 6, 'b1_t5_threshold'] = 0.08
    df.loc[vol_20 < 3, 'b1_t5_threshold'] = 0.03

    # === b2 加仓条件 ===
    b2_conditions = (
        (df['close'] / df['close'].shift(1) - 1 > 0.04) &
        (df['J'] < 55) &
        (df['volume'] > df['volume'].shift(1) * 1.3) &
        ((df['high'] - df['close']) / df['close'] < 0.015)
    )
    df['b2_add_signal'] = b2_conditions.astype(int)

    # === Tier 1 止损 ===
    ever_below = (df['close'] < df['yellow_line']).rolling(20).max() > 0
    back_above = ((df['cross_below_yellow'] == 0) &
                  ((df['close'] > df['yellow_line']).rolling(3).min() > 0))
    second_break = ((df['close'] < df['yellow_line']) &
                    (df['close'].shift(1) >= df['yellow_line'].shift(1)))
    stop_yellow = ever_below & back_above & second_break

    s1 = ((df['close'] / df['close'].shift(5) - 1 > 0.10) &
          (df['close'] / df['open'] - 1 < -0.03) &
          (df['volume'] > df['volume'].rolling(20).mean() * 1.5))

    new_high_20 = df['high'] == df['high'].rolling(20).max()
    macd_decl = df['MACD_HIST'] < df['MACD_HIST'].shift(10)
    vol_f = ((df['volume'] / df['volume'].rolling(5).mean() > 0.8) &
             (df['volume'] / df['volume'].rolling(5).mean() < 1.2))
    s2 = new_high_20 & macd_decl & vol_f

    df['b1_stop_signal'] = (stop_yellow | s1 | s2).astype(int)
    df['b1_stop_yellow'] = stop_yellow.astype(int)
    df['b1_s1_signal'] = s1.astype(int)
    df['b1_s2_signal'] = s2.astype(int)

    # Tier 2 信号（状态机填充）
    df['b1_fly_signal'] = 0
    df['b1_didi_stop'] = 0
    df['b1_t5_stop'] = 0

    # === 状态机 ===
    state = pd.Series('wait', index=df.index, dtype=str)
    close_vals = df['close'].values
    low_vals = df['low'].values
    t5_thresh = df['b1_t5_threshold'].values
    stop_t1 = df['b1_stop_signal'].values
    entry_sig = df['b1_entry_signal'].values

    in_pos = False; has_fly = False
    entry_idx = None; entry_price = None

    for i in range(len(df)):
        idx = df.index[i]
        if not in_pos:
            if entry_sig[i] == 1:
                state.loc[idx] = 'entry'
                in_pos = True; has_fly = False
                entry_idx = i; entry_price = close_vals[i]
            else:
                state.loc[idx] = 'wait'
        else:
            gain = (close_vals[i] / entry_price - 1) if entry_price else 0

            if stop_t1[i] == 1:
                state.loc[idx] = 'exit'
                in_pos = False; has_fly = False
            elif i - entry_idx >= 5 and gain < t5_thresh[i]:
                state.loc[idx] = 't5_stop'
                df.loc[idx, 'b1_t5_stop'] = 1
                in_pos = False; has_fly = False
            elif not has_fly and gain > 0.10:
                state.loc[idx] = 'fly'
                df.loc[idx, 'b1_fly_signal'] = 1
                has_fly = True
            elif has_fly:
                prev_low = low_vals[i - 1] if i > 0 else close_vals[i]
                if close_vals[i] < prev_low:
                    state.loc[idx] = 'didi_stop'
                    df.loc[idx, 'b1_didi_stop'] = 1
                    in_pos = False; has_fly = False
                else:
                    state.loc[idx] = 'hold'
            elif df['b2_add_signal'].iloc[i] == 1:
                state.loc[idx] = 'add'
            else:
                state.loc[idx] = 'hold'

    df['b1_state'] = state
    return df


def b1_strategy_summary(df: pd.DataFrame) -> Dict:
    entry_count = int(df['b1_entry_signal'].sum())
    stop_count = int(df['b1_stop_signal'].sum())
    fly_count = int(df['b1_fly_signal'].sum())
    didi_count = int(df['b1_didi_stop'].sum())
    t5_count = int(df['b1_t5_stop'].sum())
    add_count = int(df['b2_add_signal'].sum())
    entry_indices = df.index[df['b1_entry_signal'] == 1]

    return {
        'strategy': 'b1超卖反转',
        'total_bars': len(df),
        'entry_signals': entry_count,
        'add_signals': add_count,
        'stop_signals': stop_count,
        'fly_signals': fly_count,
        'didi_stops': didi_count,
        't5_stops': t5_count,
        'signal_density': f"{entry_count / len(df) * 100:.2f}%" if len(df) > 0 else '0%',
        'last_entry_dates': (
            entry_indices[-5:].strftime('%Y-%m-%d').tolist()
            if len(entry_indices) > 0 else []
        ),
        'avg_entry_score': round(
            df.loc[df['b1_entry_signal'] == 1, 'b1_score_pct'].mean() * 100, 1
        ) if entry_count > 0 else 0,
    }


if __name__ == '__main__':
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', periods=500, freq='B')
    close = 10 + np.cumsum(np.random.randn(500) * 0.15)
    close = np.maximum(close, 1)
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(500) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(500) * 0.02)),
        'low': close * (1 - np.abs(np.random.randn(500) * 0.02)),
        'close': close,
        'volume': np.random.randint(1e6, 1e7, 500),
        'turnover': np.random.uniform(0.5, 8, 500),
    }, index=dates)
    df = generate_b1_signals(df, board_type='main')
    s = b1_strategy_summary(df)
    for k, v in s.items():
        print(f"  {k}: {v}")
