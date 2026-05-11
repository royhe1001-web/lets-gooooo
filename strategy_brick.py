#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略三：砖型图波段策略（四定式分类）
====================================

通达信原码逻辑：
  VAR1A:=(HHV(H,4)-C)/(HHV(H,4)-LLV(L,4))*100-90;
  VAR2A:=SMA(VAR1A,4,1)+100;
  VAR3A:=(C-LLV(L,4))/(HHV(H,4)-LLV(L,4))*100;
  VAR4A:=SMA(VAR3A,6,1);
  VAR5A:=SMA(VAR4A,6,1)+100;
  VAR6A:=VAR5A-VAR2A;
  砖型图:=IF(VAR6A>4,VAR6A-4,0);         ← 指标值

  STICKLINE(REF<砖,砖,REF,3,0),COLORRED;   ← 红柱=砖值上升
  STICKLINE(REF>砖,砖,REF,3,0),COLOR00FF00; ← 绿柱=砖值下降

  AA:=REF(砖型图,1)<砖型图;               ← 今日上升
  CC:=REF(AA,1)=0 AND AA=1;               ← 前日未上升，今日首次上升
  XG:=CC>0;                                ← 选股信号

用法: 红砖持股，绿砖观望。绿转红时买入（CC 信号），红转绿时卖出。
      持股周期约 4 天（基于 4 日高低点的短周期摆动指标）。

四定式分类（叠加在 CC 买入信号上）:
  定式一（N型上涨）  : 上涨→回调>10%→不破前低→CC买入
  定式二（横盘突破）  : 横盘>8日→放量长阳→CC买入
  定式三（拉升回踩）  : 前期拉升>10%→短暂调整→CC买入
  定式四（不满足定式）: 仅满足总原则
"""

import numpy as np
import pandas as pd
from typing import Dict

try:
    from quant_strategy.indicators import calc_brick_chart, calc_white_yellow_line
except ImportError:
    from indicators import calc_brick_chart, calc_white_yellow_line


# ============================================================
# 砖型图 + 白黄线（委托 indicators 模块）
# ============================================================
def _calc_brick_and_wy(df: pd.DataFrame) -> pd.DataFrame:
    """砖型图 + 白黄线指标，委托调用 indicators 模块统一实现。"""
    df = calc_brick_chart(df)
    df = calc_white_yellow_line(df)
    return df


# ============================================================
# 总原则
# ============================================================
def _meets_general(row) -> bool:
    """总原则：白线>黄线 + CC买入信号（绿转红）"""
    if row['white_line'] <= row['yellow_line']:
        return False
    if row['brick_green_to_red'] != 1:
        return False
    return True


# ============================================================
# 定式判断（不变）
# ============================================================
def _check_pattern_1(stock: pd.DataFrame, target_loc: int) -> bool:
    """定式一：N型上涨 — 回调>10%不破前低"""
    close, low = stock['close'], stock['low']
    if target_loc < 60: return False
    peak_start = max(0, target_loc - 60)
    peak_end = target_loc - 5
    if peak_end <= peak_start: return False
    peak_slice = close.iloc[peak_start:peak_end]
    if len(peak_slice) == 0: return False
    peak_price = peak_slice.max()
    peak_loc = peak_slice.idxmax()
    peak_pos = stock.index.get_loc(peak_loc)
    trough_slice = low.iloc[peak_pos:target_loc]
    if len(trough_slice) == 0: return False
    trough_price = trough_slice.min()
    if (trough_price / peak_price - 1) * 100 > -10: return False
    prev_low_start = max(0, peak_pos - 80)
    prev_low_end = peak_pos - 1
    if prev_low_end <= prev_low_start: return False
    return trough_price > low.iloc[prev_low_start:prev_low_end].min()


def _check_pattern_2(stock: pd.DataFrame, target_loc: int) -> bool:
    """定式二：横盘突破 — 横盘>8日+放量长阳"""
    close, high, low, vol = stock['close'], stock['high'], stock['low'], stock['volume']
    if target_loc < 20: return False
    cur, prev = stock.iloc[target_loc], stock.iloc[target_loc - 1]
    if (cur['close'] / prev['close'] - 1) * 100 < 3: return False
    vol_20_avg = vol.iloc[max(0, target_loc - 20):target_loc].mean()
    if vol_20_avg <= 0 or cur['volume'] < vol_20_avg * 1.5: return False
    consol_end = target_loc - 1; consol_start = consol_end
    for i in range(consol_end - 1, max(0, target_loc - 40), -1):
        ph = high.iloc[i:consol_end + 1].max()
        pl = low.iloc[i:consol_end + 1].min()
        pa = close.iloc[i:consol_end + 1].mean()
        if pa > 0 and (ph - pl) / pa * 100 < 12: consol_start = i
        else: break
    if consol_end - consol_start + 1 < 8: return False
    return cur['high'] > high.iloc[consol_start:consol_end + 1].max()


def _check_pattern_3(stock: pd.DataFrame, target_loc: int) -> bool:
    """定式三：拉升回踩 — 前期拉升>10%+短暂调整"""
    close, low = stock['close'], stock['low']
    if target_loc < 30: return False
    gl = int(stock['brick_green_streak'].iloc[target_loc - 1]) if target_loc >= 1 else 0
    if gl < 1 or gl > 7: return False
    gs = target_loc - 1 - gl + 1
    if gs < 10: return False
    rss, rse = max(0, gs - 40), gs - 1
    if rse <= rss: return False
    sc, sl = close.iloc[rss:rse + 1], low.iloc[rss:rse + 1]
    if len(sc) < 10: return False
    mr = 0
    for ws in [10, 15, 20]:
        for s in range(0, len(sc) - ws):
            rp = (sc.iloc[s + ws - 1] / sl.iloc[s:s + ws].min() - 1) * 100
            if rp > mr: mr = rp
    return mr >= 10


# ============================================================
# 公开的四定式分类函数（供筛选器复用，消除代码重复）
# ============================================================
def classify_brick_pattern(stock: pd.DataFrame, target_loc: int) -> str:
    """
    对单只股票在 target_loc 位置进行四定式分类。
    优先级: 定式一 > 定式二 > 定式三 > 定式四
    """
    if _check_pattern_1(stock, target_loc):
        return '定式一'
    if _check_pattern_2(stock, target_loc):
        return '定式二'
    if _check_pattern_3(stock, target_loc):
        return '定式三'
    return '定式四'


# ============================================================
# 策略主函数
# ============================================================
def generate_brick_signals(df: pd.DataFrame,
                            board_type: str = 'main',
                            green_streak_min: int = 2,
                            consolidation_window: int = 20,
                            precomputed: bool = False) -> pd.DataFrame:
    """
    生成砖型图 CC 买入信号 + 四定式分类。

    优化项:
      1. 绿砖连续≥2: 前日绿柱至少连续2天，过滤单日噪音反转
      2. 黄线趋势: 股价>黄线，顺势而为
      3. 最小持仓2天: 交易层规则（非信号层），防同日进出

    总原则: 绿砖≥green_streak_min + 白线>黄线 + CC信号

    返回新增列:
        brick_val, brick_rising, brick_falling, brick_green_to_red,
        brick_red_to_green, brick_green_streak, brick_red_streak,
        white_line, yellow_line, white_above_yellow, white_direction,
        brick_entry_signal, brick_pattern
    """
    df = _calc_brick_and_wy(df)

    # 优化: 绿砖连续≥green_streak_min + CC信号 + 黄线趋势
    general = (
        (df['brick_green_streak'].shift(1) >= green_streak_min) &
        (df['brick_green_to_red'] == 1) &
        (df['white_line'] > df['yellow_line']) &
        (df['close'] > df['yellow_line'])
    )
    df['brick_entry_signal'] = general.astype(int)

    # === 质量评分: 涨幅 + 砖值 + 量能 ===
    gain = df['close'] / df['close'].shift(1) - 1
    vol_ratio = df['volume'] / df['volume'].shift(1)
    df['brick_quality'] = 0.0
    # 涨幅分
    df.loc[gain > 0.03, 'brick_quality'] += 2.0
    df.loc[(gain > 0.01) & (gain <= 0.03), 'brick_quality'] += 1.0
    # 砖值分
    df.loc[df['brick_val'] > 80, 'brick_quality'] += 2.0
    df.loc[(df['brick_val'] > 60) & (df['brick_val'] <= 80), 'brick_quality'] += 1.5
    df.loc[(df['brick_val'] > 40) & (df['brick_val'] <= 60), 'brick_quality'] += 1.0
    # 量能分
    df.loc[vol_ratio > 2.0, 'brick_quality'] += 1.5
    df.loc[(vol_ratio > 1.3) & (vol_ratio <= 2.0), 'brick_quality'] += 1.0
    df.loc[(vol_ratio > 1.0) & (vol_ratio <= 1.3), 'brick_quality'] += 0.5
    # 高质量标记 (≥3.5分)
    df['brick_high_quality'] = ((df['brick_quality'] >= 3.5) & general).astype(int)

    # 四定式分类
    df['brick_pattern'] = ''
    entry_indices = df.index[df['brick_entry_signal'] == 1]
    for idx in entry_indices:
        target_loc = df.index.get_loc(idx)
        if _check_pattern_1(df, target_loc):
            df.loc[idx, 'brick_pattern'] = '定式一'
        elif _check_pattern_2(df, target_loc):
            df.loc[idx, 'brick_pattern'] = '定式二'
        elif _check_pattern_3(df, target_loc):
            df.loc[idx, 'brick_pattern'] = '定式三'
        else:
            df.loc[idx, 'brick_pattern'] = '定式四'

    return df


def brick_strategy_summary(df: pd.DataFrame) -> Dict:
    entry_count = int(df['brick_entry_signal'].sum())
    pattern_counts = {}
    if entry_count > 0 and 'brick_pattern' in df.columns:
        pattern_counts = (df.loc[df['brick_entry_signal'] == 1, 'brick_pattern']
                          .value_counts().to_dict())
    return {
        'strategy': '砖型图波段（CC信号+四定式）',
        'total_bars': len(df),
        'entry_signals': entry_count,
        'exit_signals': int(df['brick_red_to_green'].sum()),
        'signal_density': f"{entry_count / len(df) * 100:.2f}%" if len(df) > 0 else '0%',
        'pattern_1': int(pattern_counts.get('定式一', 0)),
        'pattern_2': int(pattern_counts.get('定式二', 0)),
        'pattern_3': int(pattern_counts.get('定式三', 0)),
        'pattern_4': int(pattern_counts.get('定式四', 0)),
        'last_entry_dates': (df.index[df['brick_entry_signal'] == 1]
                             .strftime('%Y-%m-%d').tolist()[-5:]
                             if entry_count > 0 else []),
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
    df = generate_brick_signals(df, board_type='main')
    s = brick_strategy_summary(df)
    for k, v in s.items():
        print(f"  {k}: {v}")
