#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略二：b2 强势确认反转策略（短线 T+1 ~ T+5）— 案例优化版
========================================================

基于10个成功案例（2025年5-8月）的统计优化:
  案例: 002987, 920039, 688170, 300418, 601778, 688787, 002345, 688202, 688318, 920748
  T+5 胜率 100% | T+5 均值 +24.7% | 最大收益均值 +75.2%

优化项（ML Phase 1-5 + Phase 2c 多牛市网格搜索，2026-05）:
  1. B2+OAMV联合优化: Pullback+OAMV 三段牛市网格搜索最优 (Test +28.3%, Sharpe 0.728)
  2. 入场门槛: gain_min=4%, j_today_max=65 (放宽信号捕捉更多机会)
  3. 回调权重: 前10日回调>7%→权重×1.5 (超卖深度确认增强)
  4. 涨幅2x: gain>7%→权重×2.0 (适度加成)
  5. 止损策略: 原始引擎(None) — 牛市中让利润奔跑，不用改进止损
  6. 动态阈值: 按 OAMV regime 区分入场门槛 (Phase 5, Sharpe +0.722)
  7. 上影线: 普通3.0%/猛干4.0%/防守2.5% + 降级权重×0.7
  8. 深度缩量: 前日量<20日均量80%→权重×1.3
  9. 共振加分: b2∩砖型图CC信号→权重×1.5
  10. OAMV 市场状态: 防守禁止开仓, 动态止损缓冲
  11. LightGBM 信号过滤: AUC=0.546, 胜率+8.5pp (Phase 4)

入场条件（7项 + 动态OAMV阈值）:
  前日 J<20 | 当日涨>动态阈值 | J<动态阈值 | 放量 | 上影线<动态阈值 | 白>黄 | 收>黄
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
                        precomputed: bool = False,
                        params: dict = None) -> pd.DataFrame:
    """Generate B2 signals with optional parameter overrides.

    params dict keys (all optional, defaults used if missing):
      # Entry thresholds (Phase 3 optimal, Phase 5 dynamic)
      j_prev_max, gain_min, j_today_max, shadow_max,
      j_today_loose, shadow_loose, prior_strong_ret,
      # Dynamic thresholds (dynamic_thresholds=True enables per-regime values)
      dynamic_thresholds, gain_min_aggressive, gain_min_defensive,
      gain_min_normal, j_today_max_aggressive, j_today_max_defensive,
      j_today_max_normal, shadow_max_aggressive, shadow_max_defensive,
      shadow_max_normal, vol_scale,
      # Weight multipliers
      weight_gain_2x_thresh, weight_gain_2x, weight_gap_thresh,
      weight_gap_up, weight_shrink, weight_deep_shrink_ratio,
      weight_deep_shrink, weight_pullback_thresh, weight_pullback,
      weight_shadow_discount_thresh, weight_shadow_discount,
      weight_strong_discount, weight_brick_resonance,
      # 0AMV 市场状态 (活筹指数)
      oamv_aggressive_buffer    : 猛干模式止损缓冲 (默认 0.95)
      oamv_defensive_buffer     : 防守模式止损缓冲 (默认 0.99)
      oamv_defensive_ban_entry  : 防守模式禁止开仓 (默认 True)

    0AMV 集成: 若 DataFrame 中包含 'oamv_regime' 列
      (值: 'aggressive' / 'defensive' / 'normal'), 状态机会自动调整
      止损阈值: 猛干放宽→entry_low*0.95, 防守收紧→entry_low*0.99.
    """
    if not precomputed:
        df = calc_all_indicators(df, board_type=board_type)

    p = params or {}

    # === 入场（7条件 + 前期强势股放宽）===
    gain = df['close'] / df['close'].shift(1) - 1

    c1 = df['J'].shift(1) < p.get('j_prev_max', 20)
    c4 = df['volume'] > df['volume'].shift(1)
    c6 = df['white_line'] > df['yellow_line']
    c7 = df['close'] > df['yellow_line']

    # 前期强势股识别
    ret_20d = df['close'] / df['close'].shift(20) - 1
    is_prior_strong = ret_20d > p.get('prior_strong_ret', 0.20)

    # === 动态阈值: 根据 OAMV regime + 波动率调整入场门槛 ===
    has_oamv = 'oamv_regime' in df.columns
    use_dynamic = p.get('dynamic_thresholds', False)

    if use_dynamic and has_oamv:
        regime_arr = df['oamv_regime'].values

        # Per-regime gain_min (Phase 5 最优)
        gain_agg = p.get('gain_min_aggressive', 0.04)
        gain_def = p.get('gain_min_defensive', 0.06)
        gain_norm = p.get('gain_min_normal', 0.06)
        gain_min_arr = np.where(regime_arr == 'aggressive', gain_agg,
                       np.where(regime_arr == 'defensive', gain_def, gain_norm))

        # Per-regime j_today_max
        j_agg = p.get('j_today_max_aggressive', 65)
        j_def = p.get('j_today_max_defensive', 55)
        j_norm = p.get('j_today_max_normal', 55)
        j_max_arr = np.where(regime_arr == 'aggressive', j_agg,
                    np.where(regime_arr == 'defensive', j_def, j_norm))

        # Per-regime shadow_max
        sh_agg = p.get('shadow_max_aggressive', 0.040)
        sh_def = p.get('shadow_max_defensive', 0.025)
        sh_norm = p.get('shadow_max_normal', 0.030)
        shad_arr = np.where(regime_arr == 'aggressive', sh_agg,
                   np.where(regime_arr == 'defensive', sh_def, sh_norm))

        # Volatility scaling (optional)
        if p.get('vol_scale', False) and 'volatility_20d' in df.columns:
            vol = df['volatility_20d'].values
            vol_factor = np.clip(vol / 0.03, 0.5, 2.0)
            gain_min_arr = gain_min_arr * vol_factor
            shad_arr = shad_arr / np.clip(vol_factor, 0.5, 2.0)

        effective_gain_min = pd.Series(gain_min_arr, index=df.index)
        effective_j_max = pd.Series(j_max_arr, index=df.index)
        effective_shad_max = pd.Series(shad_arr, index=df.index)
    else:
        effective_gain_min = p.get('gain_min', 0.04)
        effective_j_max = p.get('j_today_max', 65)
        effective_shad_max = p.get('shadow_max', 0.035)

    # 标准阈值
    c2 = gain > effective_gain_min
    c3_std = df['J'] < effective_j_max
    c5_std = (df['high'] - df['close']) / df['close'] < effective_shad_max

    # 强势股放宽阈值 (保持固定，强势股自身特征已足够)
    c3_loose = df['J'] < p.get('j_today_loose', 70)
    c5_loose = (df['high'] - df['close']) / df['close'] < p.get('shadow_loose', 0.045)

    # 标准入场 或 强势股放宽入场
    entry_std = c1 & c2 & c3_std & c4 & c5_std & c6 & c7
    entry_loose = c1 & c2 & c3_loose & c4 & c5_loose & c6 & c7 & is_prior_strong
    entry = entry_std | entry_loose
    df['b2_entry_signal'] = entry.astype(int)
    df['b2_prior_strong'] = (is_prior_strong & entry_loose).astype(int)

    # === 仓位权重（多层叠加）===
    df['b2_position_weight'] = 1.0

    # 涨幅>7%→2x
    gain_2x_thresh = p.get('weight_gain_2x_thresh', 0.07)
    gain_2x_mult = p.get('weight_gain_2x', 2.0)
    df.loc[gain > gain_2x_thresh, 'b2_position_weight'] = gain_2x_mult

    # 跳空高开>2%→1.5x
    gap_thresh = p.get('weight_gap_thresh', 0.02)
    gap_mult = p.get('weight_gap_up', 1.5)
    gap_up = df['open'] > df['close'].shift(1) * (1 + gap_thresh)
    df.loc[gap_up, 'b2_position_weight'] *= gap_mult

    # setup日缩量→1.2x (抛压衰竭确认)
    setup_shrink_mult = p.get('weight_shrink', 1.2)
    setup_shrink = df['volume'].shift(1) < df['volume'].shift(2)
    df.loc[setup_shrink, 'b2_position_weight'] *= setup_shrink_mult

    # 深度缩量: 前日量缩至20日均量80%以下→1.3x
    deep_shrink_ratio = p.get('weight_deep_shrink_ratio', 0.80)
    deep_shrink_mult = p.get('weight_deep_shrink', 1.3)
    vol_20_avg = df['volume'].shift(1).rolling(20).mean()
    deep_shrink = df['volume'].shift(1) < vol_20_avg * deep_shrink_ratio
    df.loc[deep_shrink, 'b2_position_weight'] *= deep_shrink_mult

    # 前10日回调>7%→1.5x (超卖深度确认，优化自Pullback+OAMV)
    pullback_thresh = p.get('weight_pullback_thresh', -0.07)
    pullback_mult = p.get('weight_pullback', 1.5)
    lookback = 10
    hh10 = df['close'].rolling(lookback).max().shift(1)
    pullback = (df['close'].shift(1) / hh10 - 1)
    deep_pullback = pullback < pullback_thresh
    df.loc[deep_pullback, 'b2_position_weight'] *= pullback_mult

    # 上影线折价→0.7x (有上影仍有获利但风险更高)
    shadow_discount_thresh = p.get('weight_shadow_discount_thresh', 0.015)
    shadow_discount_mult = p.get('weight_shadow_discount', 0.7)
    has_shadow = (df['high'] - df['close']) / df['close'] >= shadow_discount_thresh
    df.loc[has_shadow & entry, 'b2_position_weight'] *= shadow_discount_mult

    # 强势股放宽入场 → 权重×0.8 (阈值放宽的信号质量略低)
    strong_discount_mult = p.get('weight_strong_discount', 0.8)
    df.loc[entry_loose & ~entry_std, 'b2_position_weight'] *= strong_discount_mult

    # 共振加分: b2 ∩ 砖型图CC信号 → 1.5x (双策略确认提高确定性)
    brick_resonance_mult = p.get('weight_brick_resonance', 1.5)
    if 'brick_green_to_red' in df.columns:
        brick_resonance = (df['brick_green_to_red'] == 1) & entry
        df.loc[brick_resonance, 'b2_position_weight'] *= brick_resonance_mult
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

    # === 0AMV 市场状态（活筹指数）===
    if has_oamv:
        oamv_regime_v = df['oamv_regime'].values
    else:
        oamv_regime_v = None

    # 0AMV 止损缓冲参数: aggressive(猛干)放宽, defensive(防守)收紧
    oamv_aggressive_buffer = p.get('oamv_aggressive_buffer', 0.95)
    oamv_defensive_buffer = p.get('oamv_defensive_buffer', 0.99)
    # 防守模式禁止开仓
    oamv_defensive_ban_entry = p.get('oamv_defensive_ban_entry', True)

    # === 状态机 ===
    state = pd.Series('wait', index=df.index, dtype=str)
    close_v = df['close'].values; low_v = df['low'].values
    stop_v = df['b2_stop_signal'].values; entry_v = df['b2_entry_signal'].values

    in_pos = False; has_fly = False; was_profitable = False
    entry_i = None; entry_px = None; entry_low = None

    for i in range(len(df)):
        idx = df.index[i]
        if not in_pos:
            # 防守模式: 可选禁止新开仓
            can_entry = True
            if oamv_regime_v is not None and oamv_defensive_ban_entry:
                if oamv_regime_v[i] == 'defensive':
                    can_entry = False

            if entry_v[i] == 1 and can_entry:
                state.loc[idx] = 'entry'
                in_pos = True; has_fly = False; was_profitable = False
                entry_i = i; entry_px = close_v[i]; entry_low = low_v[i]
                df.loc[idx, 'b2_entry_oamv_regime'] = oamv_regime_v[i] if oamv_regime_v is not None else 'normal'
            else:
                state.loc[idx] = 'wait'
        else:
            gain_px = (close_v[i] / entry_px - 1) if entry_px else 0
            if gain_px > 0: was_profitable = True

            # 计算当日蜡烛止损阈值 (0AMV 动态调整)
            current_regime = oamv_regime_v[i] if oamv_regime_v is not None else 'normal'
            if current_regime == 'aggressive':
                candle_stop_level = entry_low * oamv_aggressive_buffer  # 放宽, 如 0.95
            elif current_regime == 'defensive':
                candle_stop_level = entry_low * oamv_defensive_buffer  # 收紧, 如 0.99
            else:
                candle_stop_level = entry_low

            # Tier 1
            if stop_v[i] == 1:
                state.loc[idx] = 'exit'
                in_pos = False; has_fly = False
            # 蜡烛跌破 (0AMV 动态阈值)
            elif close_v[i] < candle_stop_level:
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
