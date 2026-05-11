#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三信号离场引擎

实现方案第六章的三重卖出信号投票:
  ① 截面排名检查: T+4日重排序，跌出Top-30% → 标记卖出
  ② 砖型图周期确认: brick趋势逆转 → 降级/卖出
  ③ 风控止损: 最大回撤/移动止损/OAMV防御强制退出

信号逻辑:
  - 全部通过 → 续持进入下一周期
  - 任一失败 → T+5 开盘卖出
  - 防御市场 + 排名跌出Top-50% → 强制卖出 (OAMV优先)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ExitConfig:
    """离场信号配置 (Gamma曲面优化后的最优参数)"""
    # 截面排名
    rank_sell_threshold: float = 0.20      # 排名跌出Top-20% → 卖出 (优化后从30%收紧)
    rank_defensive_threshold: float = 0.50  # 防御市场放宽到Top-50%

    # 砖型图
    brick_reversal_window: int = 3          # 砖型逆转确认窗口

    # 止损
    max_drawdown_pct: float = 0.05          # 最大回撤 5% (优化后从8%收紧)
    trailing_stop_pct: float = 0.03         # 移动止损 3% (优化后从5%收紧)
    hard_stop_pct: float = 0.10             # 硬止损 10%

    # 卖出阈值
    sell_score_threshold: int = 3           # 需要3个信号同时触发才卖出 (优化后从1提高)

    # OAMV防御
    oamv_defensive_force_exit: bool = True  # 防御市场强制退出
    oamv_divergence_override: bool = True   # 底背离时豁免防御强制退出


class ExitSignalEngine:
    """三信号离场引擎。

    用法:
        engine = ExitSignalEngine(config)
        decision = engine.evaluate(
            symbol='000001',
            entry_date=T_date,
            current_date=T_plus_4,
            held_data={...},
            market_data={...},
        )
    """

    def __init__(self, config: ExitConfig = None):
        self.config = config or ExitConfig()

    def evaluate(self,
                 symbol: str,
                 entry_date: pd.Timestamp,
                 current_date: pd.Timestamp,
                 current_rank_pct: float,        # 当前截面排名百分位 (0~1, 越小越好)
                 brick_signal: dict,              # 砖型图信号
                 price_history: pd.Series,        # 持仓期间价格序列
                 oamv_state: str,                 # 'aggressive'/'bullish'/'neutral'/'defensive'
                 entry_price: float,
                 current_price: float,
                 oamv_divergence: str = 'none',  # 'bottom'=底背离, 'top'=顶背离, 'none'=无
                 ) -> dict:
        """综合评估离场信号。

        返回:
          dict with 'action' ('hold'/'sell'), 'reasons', 'score' (0~3, 越高越该卖)
        """
        reasons = []
        sell_score = 0

        # ---- ① 截面排名检查 ----
        rank_threshold = self.config.rank_sell_threshold
        if oamv_state == 'defensive':
            rank_threshold = self.config.rank_defensive_threshold

        if current_rank_pct > rank_threshold:
            reasons.append(f'排名{current_rank_pct:.0%} > {rank_threshold:.0%}')
            sell_score += 1

        # ---- ② 砖型图周期确认 ----
        if brick_signal.get('trend_reversed', False):
            reasons.append('砖型逆转')
            sell_score += 1
        if brick_signal.get('brick_color') == 'red' and brick_signal.get('consecutive_red', 0) >= 2:
            reasons.append(f'砖型连阴({brick_signal["consecutive_red"]}根)')
            sell_score += 1

        # ---- ③ 风控止损 ----
        if price_history is not None and len(price_history) > 0:
            ret = (current_price - entry_price) / entry_price

            # 硬止损
            if ret <= -self.config.hard_stop_pct:
                reasons.append(f'硬止损({ret:.1%})')
                sell_score += 2  # 硬止损直接+2，强制卖出

            # 最大回撤
            peak = price_history.max()
            dd = (current_price - peak) / peak
            if dd <= -self.config.max_drawdown_pct:
                reasons.append(f'回撤({dd:.1%})')
                sell_score += 1

            # 移动止损
            if len(price_history) >= 3:
                recent_high = price_history.iloc[-3:].max()
                trail_dd = (current_price - recent_high) / recent_high
                if trail_dd <= -self.config.trailing_stop_pct:
                    reasons.append(f'移动止损({trail_dd:.1%})')
                    sell_score += 1

        # ---- OAMV 防御强制退出 (v2: 底背离豁免) ----
        if oamv_state == 'defensive' and self.config.oamv_defensive_force_exit:
            if oamv_divergence == 'bottom' and self.config.oamv_divergence_override:
                pass  # 底背离时不强制卖出: 资金在暗中进场
            elif current_rank_pct > self.config.rank_defensive_threshold:
                reasons.append('防御市场强制退出')
                sell_score += 1

        action = 'sell' if sell_score >= self.config.sell_score_threshold else 'hold'

        return {
            'symbol': symbol,
            'action': action,
            'sell_score': sell_score,
            'reasons': reasons,
            'entry_price': entry_price,
            'current_price': current_price,
            'return_pct': (current_price - entry_price) / entry_price,
            'holding_days': (current_date - entry_date).days,
        }

    def batch_evaluate(self,
                       positions: pd.DataFrame,
                       scores: pd.Series,
                       brick_data: dict,
                       prices: pd.DataFrame,
                       oamv_state: str,
                       current_date: pd.Timestamp,
                       oamv_divergence: str = 'none',
                       ) -> pd.DataFrame:
        """批量评估所有持仓的离场信号。

        positions: columns=['symbol', 'entry_date', 'entry_price']
        scores: symbol → ML排名分 (越小越好/越大越好取决于定义，这里假设越大越好)
        brick_data: {symbol: brick_signal_dict}
        prices: index=date, columns=symbol
        """
        if positions.empty:
            return pd.DataFrame()

        # 计算截面排名百分位
        n = len(scores)
        rank_pcts = {}
        for i, (sym, score) in enumerate(scores.sort_values(ascending=False).items()):
            rank_pcts[sym] = i / n

        results = []
        for _, pos in positions.iterrows():
            sym = pos['symbol']
            entry_date = pd.Timestamp(pos['entry_date'])
            entry_price = float(pos['entry_price'])
            current_price = float(prices[sym].dropna().iloc[-1]) if sym in prices.columns else entry_price

            # 持仓期间价格
            mask = (prices.index > entry_date) & (prices.index <= current_date)
            price_hist = prices[sym][mask].dropna() if sym in prices.columns else pd.Series()

            brick = brick_data.get(sym, {})

            decision = self.evaluate(
                symbol=sym,
                entry_date=entry_date,
                current_date=current_date,
                current_rank_pct=rank_pcts.get(sym, 0.5),
                brick_signal=brick,
                price_history=price_hist,
                oamv_state=oamv_state,
                entry_price=entry_price,
                current_price=current_price,
                oamv_divergence=oamv_divergence,
            )
            results.append(decision)

        return pd.DataFrame(results)


# ================================================================
# 砖型图辅助函数 (从B2因子提取)
# ================================================================

def compute_brick_signals(df: pd.DataFrame,
                          brick_size: float = 0.03,
                          lookback: int = 5) -> dict:
    """从日线数据计算砖型图信号。

    返回:
      brick_color: 'red'(阳/上升) / 'green'(阴/下降)
      consecutive_red: 连续阴线根数
      consecutive_green: 连续阳线根数
      trend_reversed: 趋势是否逆转 (最近lookback内方向改变)
      brick_count: 最近N日砖块数
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    if len(close) < 20:
        return {'brick_color': 'unknown', 'trend_reversed': False}

    # 简化砖型: 用收盘价变动方向代替完整砖型图
    brick_size_val = brick_size * close[-1]

    # 计算价格摆动
    swings = []
    ref_price = close[0]
    for i in range(1, len(close)):
        change = close[i] - ref_price
        if abs(change) >= brick_size_val:
            swings.append(1 if change > 0 else -1)
            ref_price = close[i]

    if len(swings) < 2:
        recent_dir = 1 if close[-1] > close[-5] else -1
        return {
            'brick_color': 'green' if recent_dir > 0 else 'red',
            'consecutive_red': 1 if recent_dir < 0 else 0,
            'consecutive_green': 1 if recent_dir > 0 else 0,
            'trend_reversed': False,
            'brick_count': 1,
        }

    # 最近方向
    recent_swings = swings[-lookback:]
    current_dir = swings[-1]

    # 连续同向
    consecutive = 0
    for s in reversed(swings):
        if s == current_dir:
            consecutive += 1
        else:
            break

    # 趋势逆转检测
    if len(swings) >= 3:
        prev_dir = swings[-3]
        trend_reversed = (prev_dir != current_dir)
    else:
        trend_reversed = False

    direction = 'green' if current_dir > 0 else 'red'

    return {
        'brick_color': direction,
        'consecutive_red': consecutive if current_dir < 0 else 0,
        'consecutive_green': consecutive if current_dir > 0 else 0,
        'trend_reversed': trend_reversed,
        'brick_count': len(recent_swings),
    }


# ================================================================
# 回测辅助: T+4 持仓跟踪
# ================================================================

def simulate_exits(stock_data: dict,
                   entry_date: pd.Timestamp,
                   positions: pd.DataFrame,
                   ranker,            # LambdaRanker 实例 (用于T+4重排序)
                   factor_engine,     # FactorEngine 实例
                   exit_engine: ExitSignalEngine,
                   oamv_state: str,
                   ) -> pd.DataFrame:
    """模拟 T+4 离场信号。

    positions: T日买入的持仓, columns=['symbol', 'entry_price']
    返回: 离场决策 DataFrame
    """
    # T+4 日期 (近似: 跳过周末)
    exit_date = entry_date + pd.Timedelta(days=7)  # 近似T+4

    # 获取T+4日数据
    exit_data = {}
    for sym in positions['symbol']:
        if sym in stock_data:
            df = stock_data[sym]
            mask = df.index <= exit_date
            if mask.any():
                exit_data[sym] = df[mask].copy()

    if len(exit_data) < 3:
        return pd.DataFrame()

    # T+4日截面重排序
    panel = factor_engine.compute_panel(exit_data)
    matrix = factor_engine.compute_matrix(panel, fillna=True)

    # 预测分数
    # 注意: matrix的index是(date, symbol), 取最后一个日期
    last_date = matrix.index.get_level_values('date').max()
    X_latest = matrix.xs(last_date, level='date')

    # 确保因子列与训练时一致
    try:
        scores = ranker.predict(X_latest)
    except Exception:
        return pd.DataFrame()

    # 获取价格
    prices = {}
    for sym in exit_data:
        prices[sym] = float(exit_data[sym]['close'].iloc[-1])
    price_df = pd.DataFrame(prices).T
    if not price_df.empty:
        price_df.columns = [last_date]

    # 计算砖型信号
    brick_signals = {}
    for sym in exit_data:
        brick_signals[sym] = compute_brick_signals(exit_data[sym])

    # 执行离场评估
    decisions = exit_engine.batch_evaluate(
        positions=positions,
        scores=scores,
        brick_data=brick_signals,
        prices=pd.DataFrame({last_date: prices}),
        oamv_state=oamv_state,
        current_date=last_date,
    )

    return decisions
