#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略二：b2 强势确认反转策略（短线 T+1 ~ T+5）— 案例优化版
========================================================

基于10个成功案例（2025年5-8月）的统计优化:
  案例: 002987, 920039, 688170, 300418, 601778, 688787, 002345, 688202, 688318, 920748
  T+5 胜率 100% | T+5 均值 +24.7% | 最大收益均值 +75.2%

优化项（案例驱动）:
  1. J当日阈值: 55→65 (920039 J=58.7→T+5=+89.7%, 920748 J=64.6→T+5=+50.6%)
  2. 上影线: 1.5%→3.0% + 降级权重×0.7 (3/10有上影仍获利)
  3. 回调深度: 前10日回调>5%→权重×1.2 (10/10案例回调均值-8.7%)
  4. 深度缩量: 前日量<20日均量80%→权重×1.3 (案例T-1 vs20=0.63x)
  5. 共振加分: b2∩砖型图CC信号→权重×1.5 (双策略确认)
  6. 涨幅>7%→2x仓位 | 跳空>2%→1.5x | setup缩量→1.2x (保留)

入场条件（7项，优化后）:
  前日 J<20 | 当日涨>4% | J<65 | 放量 | 上影线<3% | 白>黄 | 收>黄
"""

import numpy as np
import pandas as pd
from typing import Dict

try:
    from quant_strategy.indicators import calc_all_indicators
except ImportError:
    from indicators import calc_all_indicators


def generate_b2_signals(df: pd.DataFrame,
                        board_type: str = 'main',
                        precomputed: bool = False) -> pd.DataFrame:
    if not precomputed:
        df = calc_all_indicators(df, board_type=board_type)

    # === 入场（7条件 + 前期强势股放宽）===
    gain = df['close'] / df['close'].shift(1) - 1

    c1 = df['J'].shift(1) < 20                        # setup: 前日超卖
    c2 = gain > 0.04                                    # 长阳确认
    c4 = df['volume'] > df['volume'].shift(1)           # 放量
    c6 = df['white_line'] > df['yellow_line']           # 白>黄
    c7 = df['close'] > df['yellow_line']                # 收>黄

    # 前期强势股识别: 近20日涨幅>20% (已证明强势, 回调后放宽阈值)
    ret_20d = df['close'] / df['close'].shift(20) - 1
    is_prior_strong = ret_20d > 0.20

    # 标准阈值
    c3_std = df['J'] < 65
    c5_std = (df['high'] - df['close']) / df['close'] < 0.035

    # 强势股放宽阈值 (案例: 000547 J今=67→放宽到70, 上影=3.8%→放宽到4.5%)
    c3_loose = df['J'] < 70
    c5_loose = (df['high'] - df['close']) / df['close'] < 0.045

    # 标准入场 或 强势股放宽入场
    entry_std = c1 & c2 & c3_std & c4 & c5_std & c6 & c7
    entry_loose = c1 & c2 & c3_loose & c4 & c5_loose & c6 & c7 & is_prior_strong
    entry = entry_std | entry_loose
    df['b2_entry_signal'] = entry.astype(int)
    df['b2_prior_strong'] = (is_prior_strong & entry_loose).astype(int)

    # === 仓位权重（多层叠加）===
    df['b2_position_weight'] = 1.0

    # 涨幅>7%→2x
    df.loc[gain > 0.07, 'b2_position_weight'] = 2.0

    # 跳空高开>2%→1.5x
    gap_up = df['open'] > df['close'].shift(1) * 1.02
    df.loc[gap_up, 'b2_position_weight'] *= 1.5

    # setup日缩量→1.2x (抛压衰竭确认)
    setup_shrink = df['volume'].shift(1) < df['volume'].shift(2)
    df.loc[setup_shrink, 'b2_position_weight'] *= 1.2

    # 案例优化: 前日量缩至20日均量80%以下→1.3x (深度缩量筑底, 案例T-1 vs20=0.63x)
    vol_20_avg = df['volume'].shift(1).rolling(20).mean()
    deep_shrink = df['volume'].shift(1) < vol_20_avg * 0.8
    df.loc[deep_shrink, 'b2_position_weight'] *= 1.3

    # 案例优化: 前10日回调>5%→1.2x (超卖深度确认)
    lookback = 10
    hh10 = df['close'].rolling(lookback).max().shift(1)
    pullback = (df['close'].shift(1) / hh10 - 1)
    deep_pullback = pullback < -0.05
    df.loc[deep_pullback, 'b2_position_weight'] *= 1.2

    # 案例优化: 上影线折价→0.7x (有上影仍有获利但风险更高)
    has_shadow = (df['high'] - df['close']) / df['close'] >= 0.015
    df.loc[has_shadow & entry, 'b2_position_weight'] *= 0.7

    # 强势股放宽入场 → 权重×0.8 (阈值放宽的信号质量略低)
    df.loc[entry_loose & ~entry_std, 'b2_position_weight'] *= 0.8

    # 共振加分: b2 ∩ 砖型图CC信号 → 1.5x (双策略确认提高确定性)
    if 'brick_green_to_red' in df.columns:
        brick_resonance = (df['brick_green_to_red'] == 1) & entry
        df.loc[brick_resonance, 'b2_position_weight'] *= 1.5
        df['b2_brick_resonance'] = brick_resonance.astype(int)
    else:
        df['b2_brick_resonance'] = 0


    # === Tier 1 止损（同 b1）===
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
    s2 = (new_high_20 &
          (df['MACD_HIST'] < df['MACD_HIST'].shift(10)) &
          (df['volume'] / df['volume'].rolling(5).mean()).between(0.8, 1.2))

    df['b2_stop_signal'] = (stop_yellow | s1 | s2).astype(int)

    # Tier 2 + b2特有信号
    df['b2_fly_signal'] = 0
    df['b2_didi_stop'] = 0
    df['b2_t5_stop'] = 0
    df['b2_candle_stop'] = 0
    df['b2_loss_stop'] = 0

    # === 状态机 ===
    state = pd.Series('wait', index=df.index, dtype=str)
    close_v = df['close'].values; low_v = df['low'].values
    stop_v = df['b2_stop_signal'].values; entry_v = df['b2_entry_signal'].values

    in_pos = False; has_fly = False; was_profitable = False
    entry_i = None; entry_px = None; entry_low = None

    for i in range(len(df)):
        idx = df.index[i]
        if not in_pos:
            if entry_v[i] == 1:
                state.loc[idx] = 'entry'
                in_pos = True; has_fly = False; was_profitable = False
                entry_i = i; entry_px = close_v[i]; entry_low = low_v[i]
            else:
                state.loc[idx] = 'wait'
        else:
            gain_px = (close_v[i] / entry_px - 1) if entry_px else 0
            if gain_px > 0: was_profitable = True

            # Tier 1
            if stop_v[i] == 1:
                state.loc[idx] = 'exit'
                in_pos = False; has_fly = False
            # 蜡烛跌破
            elif close_v[i] < entry_low:
                state.loc[idx] = 'candle_stop'
                df.loc[idx, 'b2_candle_stop'] = 1
                in_pos = False; has_fly = False
            # loss_stop缓冲 — 浮盈>5%后才启用
            elif was_profitable and gain_px > 0.05 and gain_px < 0:
                state.loc[idx] = 'loss_stop'
                df.loc[idx, 'b2_loss_stop'] = 1
                in_pos = False; has_fly = False
            elif was_profitable and gain_px < 0 and gain_px <= 0.05:
                state.loc[idx] = 'hold'
            # T+5
            elif i - entry_i >= 5 and gain_px < 0.05:
                state.loc[idx] = 't5_stop'
                df.loc[idx, 'b2_t5_stop'] = 1
                in_pos = False; has_fly = False
            # 放飞
            elif not has_fly and gain_px > 0.10:
                state.loc[idx] = 'fly'
                df.loc[idx, 'b2_fly_signal'] = 1
                has_fly = True
            # 放飞后滴滴
            elif has_fly:
                prev_low = low_v[i - 1] if i > 0 else close_v[i]
                if close_v[i] < prev_low:
                    state.loc[idx] = 'didi_stop'
                    df.loc[idx, 'b2_didi_stop'] = 1
                    in_pos = False; has_fly = False
                else:
                    state.loc[idx] = 'hold'
            else:
                state.loc[idx] = 'hold'

    df['b2_state'] = state
    return df


def b2_strategy_summary(df: pd.DataFrame) -> Dict:
    return {
        'strategy': 'b2强势确认反转',
        'total_bars': len(df),
        'entry_signals': int(df['b2_entry_signal'].sum()),
        'stop_signals': int(df['b2_stop_signal'].sum()),
        'fly_signals': int(df['b2_fly_signal'].sum()),
        'didi_stops': int(df['b2_didi_stop'].sum()),
        't5_stops': int(df['b2_t5_stop'].sum()),
        'candle_stops': int(df['b2_candle_stop'].sum()),
        'loss_stops': int(df['b2_loss_stop'].sum()),
        'signal_density': f"{df['b2_entry_signal'].sum() / len(df) * 100:.2f}%" if len(df) > 0 else '0%',
        'last_entry_dates': (
            df.index[df['b2_entry_signal'] == 1].strftime('%Y-%m-%d').tolist()[-5:]
            if df['b2_entry_signal'].sum() > 0 else []
        ),
    }


if __name__ == '__main__':
    np.random.seed(456)
    dates = pd.date_range('2023-01-01', periods=500, freq='B')
    close = 10 + np.cumsum(np.random.randn(500) * 0.12)
    close = np.maximum(close, 1)
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(500) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(500) * 0.02)),
        'low': close * (1 - np.abs(np.random.randn(500) * 0.02)),
        'close': close,
        'volume': np.random.randint(1e6, 1e7, 500),
        'turnover': np.random.uniform(0.5, 8, 500),
    }, index=dates)
    df = generate_b2_signals(df, board_type='main')
    s = b2_strategy_summary(df)
    for k, v in s.items():
        print(f"  {k}: {v}")
