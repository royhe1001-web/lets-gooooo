#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: OAMV 驱动底仓/卫星动态配比引擎

根据 OAMV 活筹指数(指南针CYC5/CYC13双均线体系)判断市场状态，
动态调整底仓(指数跟踪)与卫星(Alpha增强)的配比。

四状态 (基于指南针官方四区域框架):
  持续上涨区 → AGGRESSIVE: OAMV>CYC5, CYC5>CYC13, 日涨>2% → 卫星60-70%
  上涨犹豫区 → BULLISH:    OAMV>CYC5, CYC5<CYC13          → 卫星40-50%
  下跌犹豫区 → NEUTRAL:    OAMV<CYC5, CYC5>CYC13          → 卫星30-40%
  持续下跌区 → DEFENSIVE:  OAMV<CYC5, CYC5<CYC13          → 卫星0-20%

新增机制(v2):
  - 防御入场需连续2日确认(防假突破)
  - 底背离(指数跌+OAMV涨)豁免防御强制卖出
  - 顶背离(指数涨+OAMV跌)提前升级防御
  - CYC5斜率改用5日多项式拟合(更稳定)

用法:
    detector = OAMVStateDetector(oamv_df)
    state = detector.detect(today)           # 返回 MarketState (向后兼容)
    detail = detector.detect_with_detail(today)  # 返回详细状态+背离信息
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class MarketState(Enum):
    AGGRESSIVE = "aggressive"   # 积极: 乘胜追击
    BULLISH = "bullish"         # 偏多: 适度进攻
    NEUTRAL = "neutral"         # 震荡: 攻守平衡
    DEFENSIVE = "defensive"     # 防御: 收缩防守


# 各状态配比表 (底仓%, 卫星%)
STATE_ALLOCATION = {
    MarketState.AGGRESSIVE: (0.35, 0.65),
    MarketState.BULLISH:    (0.55, 0.45),
    MarketState.NEUTRAL:    (0.65, 0.35),
    MarketState.DEFENSIVE:  (0.90, 0.10),
}

# 各状态持仓数参考
STATE_POSITIONS = {
    MarketState.AGGRESSIVE: {'base': (3, 5), 'sat': (3, 5)},
    MarketState.BULLISH:    {'base': (4, 6), 'sat': (2, 4)},
    MarketState.NEUTRAL:    {'base': (5, 7), 'sat': (2, 3)},
    MarketState.DEFENSIVE:  {'base': (5, 7), 'sat': (0, 1)},
}


@dataclass
class AllocationResult:
    """配比结果"""
    state: MarketState
    base_ratio: float
    sat_ratio: float
    base_capital: float
    sat_capital: float
    base_positions: pd.DataFrame   # 底仓持仓
    sat_positions: pd.DataFrame    # 卫星持仓
    oamv_snapshot: dict            # OAMV 当日快照


@dataclass
class OAMVStateDetail:
    """OAMV 详细状态信息（供离场引擎等下游使用）"""
    state: MarketState
    zone: str                  # 四区域标签: 持续上涨区/上涨犹豫区/下跌犹豫区/持续下跌区
    divergence: str            # 'bottom'=底背离, 'top'=顶背离, 'none'=无
    consecutive_days: int      # 当前区域已持续天数
    oamv_above_cyc5: bool
    cyc5_above_cyc13: bool
    oamv_chg: float            # OAMV 日涨跌幅%
    cyc5_slope: float          # CYC5 5日多项式拟合斜率


class OAMVStateDetector:
    """OAMV 市场状态检测器(v2) — 指南针双均线四区域体系。

    四区域定义:
      持续上涨区: OAMV > CYC5 且 CYC5 > CYC13 → AGGRESSIVE/BULLISH
      上涨犹豫区: OAMV > CYC5 但 CYC5 < CYC13 → BULLISH
      下跌犹豫区: OAMV < CYC5 但 CYC5 > CYC13 → NEUTRAL
      持续下跌区: OAMV < CYC5 且 CYC5 < CYC13 → DEFENSIVE

    缓冲机制(v2增强):
      - AGGRESSIVE: 需连续2日确认(防假突破)
      - DEFENSIVE:  需连续2日确认(防假跌破)，而非v1的立即触发
      - 退出DEFENSIVE: OAMV上穿CYC5且不再满足持续下跌条件

    背离检测:
      - 底背离: 5日窗口 CSI500跌>1% AND OAMV涨>1% → 防御不强制卖出
      - 顶背离: 5日窗口 CSI500涨>2% AND OAMV跌>2% → 提前升级到防御
    """

    def __init__(self, oamv_df: pd.DataFrame):
        """
        oamv_df: 日频 OAMV 数据，需含 'oamv', 'cyc5', 'cyc13' 列。
                 可选含 'close_sh' 列用于背离检测。
        """
        self.df = oamv_df.sort_index().copy()
        self._last_detail: Optional[OAMVStateDetail] = None
        self._precompute()

    def _precompute(self):
        df = self.df
        df['oamv_chg'] = df['oamv'].pct_change() * 100
        # v2: 多项式拟合斜率(5日窗口)，比简单pct_change更稳定
        df['cyc5_slope'] = self._polyfit_slope(df['cyc5'].values, window=5)
        df['above_cyc5'] = df['oamv'] > df['cyc5']
        df['above_cyc13'] = df['cyc5'] > df['cyc13']
        df['cyc5_rising'] = df['cyc5_slope'] > 0

        # v2: 连续方向计数
        above_streak = np.zeros(len(df), dtype=int)
        cnt = 0
        for i in range(len(df)):
            if df['above_cyc5'].iloc[i]:
                cnt += 1
            else:
                cnt = 0
            above_streak[i] = cnt
        df['above_cyc5_streak'] = above_streak

        below_streak = np.zeros(len(df), dtype=int)
        cnt = 0
        for i in range(len(df)):
            if not df['above_cyc5'].iloc[i]:
                cnt += 1
            else:
                cnt = 0
            below_streak[i] = cnt
        df['below_cyc5_streak'] = below_streak

        # v2: 背离预计算
        self._divergence_series = pd.Series('none', index=df.index, dtype=str)
        if 'close_sh' in df.columns:
            for i in range(5, len(df)):
                idx_chg = (df['close_sh'].iloc[i] - df['close_sh'].iloc[i - 5]) / df['close_sh'].iloc[i - 5] * 100
                oamv_chg = (df['oamv'].iloc[i] - df['oamv'].iloc[i - 5]) / df['oamv'].iloc[i - 5] * 100
                if idx_chg < -1 and oamv_chg > 1:
                    self._divergence_series.iloc[i] = 'bottom'
                elif idx_chg > 2 and oamv_chg < -2:
                    self._divergence_series.iloc[i] = 'top'

    @staticmethod
    def _polyfit_slope(values: np.ndarray, window: int = 5) -> np.ndarray:
        """5日线性多项式拟合斜率，比简单pct_change(3)更稳定"""
        from numpy.polynomial.polynomial import polyfit
        slopes = np.full(len(values), np.nan)
        for i in range(window, len(values)):
            y = values[i - window + 1:i + 1]
            x = np.arange(window)
            try:
                slope, _ = polyfit(x, y, 1)
                slopes[i] = slope / values[i] * 100 if values[i] != 0 else 0
            except Exception:
                pass
        return slopes

    def detect(self, date: pd.Timestamp) -> MarketState:
        """检测指定日期的市场状态 (向后兼容接口)"""
        detail = self.detect_with_detail(date)
        return detail.state

    def detect_with_detail(self, date: pd.Timestamp) -> OAMVStateDetail:
        """检测指定日期的详细市场状态（含区域、背离、确认天数）"""
        if date not in self.df.index:
            avail = self.df.index[self.df.index <= date]
            if len(avail) == 0:
                return OAMVStateDetail(
                    state=MarketState.NEUTRAL, zone='未知', divergence='none',
                    consecutive_days=0, oamv_above_cyc5=False,
                    cyc5_above_cyc13=False, oamv_chg=0, cyc5_slope=0,
                )
            date = avail[-1]

        idx = self.df.index.get_loc(date)
        row = self.df.iloc[idx]
        above_cyc5 = bool(row['above_cyc5'])
        above_cyc13 = bool(row['above_cyc13'])
        chg = float(row['oamv_chg'])
        slope = float(row['cyc5_slope'])
        div = self._divergence_series.iloc[idx]

        # 双均线四区域判断
        if above_cyc5 and above_cyc13:
            zone = '持续上涨区'
            below_streak = int(row['below_cyc5_streak'])
            # → AGGRESSIVE: 2日确认+涨幅确认
            if int(row['above_cyc5_streak']) >= 2 and chg > 2:
                if not np.isnan(slope) and slope > 0:
                    state = MarketState.AGGRESSIVE
                else:
                    state = MarketState.BULLISH
            else:
                state = MarketState.BULLISH
            consecutive = int(row['above_cyc5_streak'])
        elif above_cyc5 and not above_cyc13:
            zone = '上涨犹豫区'
            state = MarketState.BULLISH
            consecutive = int(row['above_cyc5_streak'])
        elif not above_cyc5 and above_cyc13:
            zone = '下跌犹豫区'
            state = MarketState.NEUTRAL
            consecutive = int(row['below_cyc5_streak'])
        else:  # not above_cyc5 and not above_cyc13
            zone = '持续下跌区'
            # → DEFENSIVE: 需连续2日跌破CYC5确认(防假跌破)
            if int(row['below_cyc5_streak']) >= 2:
                state = MarketState.DEFENSIVE
            else:
                state = MarketState.NEUTRAL  # 首日跌破→暂观望
            consecutive = int(row['below_cyc5_streak'])

        # 顶背离强化: 即使不满足持续下跌条件也升级到防御
        if div == 'top' and state != MarketState.DEFENSIVE:
            state = MarketState.DEFENSIVE

        detail = OAMVStateDetail(
            state=state, zone=zone, divergence=div,
            consecutive_days=consecutive, oamv_above_cyc5=above_cyc5,
            cyc5_above_cyc13=above_cyc13, oamv_chg=chg, cyc5_slope=slope,
        )
        self._last_detail = detail
        return detail

    @property
    def last_divergence(self) -> str:
        """上次 detect 调用后的背离状态"""
        if self._last_detail:
            return self._last_detail.divergence
        return 'none'

    def snapshot(self, date: pd.Timestamp) -> dict:
        """OAMV 当日快照"""
        if date not in self.df.index:
            avail = self.df.index[self.df.index <= date]
            if len(avail) == 0:
                return {}
            date = avail[-1]
        row = self.df.loc[date]
        return {
            'date': date,
            'oamv': round(float(row['oamv']), 1),
            'cyc5': round(float(row['cyc5']), 1),
            'cyc13': round(float(row['cyc13']), 1),
            'chg_pct': round(float(row['oamv_chg']), 2),
            'above_cyc5': bool(row['above_cyc5']),
            'cyc5_rising': bool(row['cyc5_rising']),
        }


class DynamicAllocator:
    """OAMV 驱动的底仓/卫星动态配比器。

    底仓: 市值加权，跟踪指数风格，防守属性
    卫星: ML 排名分加权，Alpha 攻击属性

    用法:
        allocator = DynamicAllocator(capital=200_000)
        result = allocator.allocate(scores, prices, market_caps, state)
    """

    def __init__(self,
                 capital: float = 200_000,
                 commission_rate: float = 0.0003,
                 lot_size: int = 100):
        self.capital = capital
        self.commission_rate = commission_rate
        self.lot_size = lot_size

    def allocate(self,
                 scores: pd.Series,          # symbol → ML排名分
                 prices: pd.Series,           # symbol → 最新股价
                 market_caps: pd.Series,      # symbol → 市值(亿)
                 state: MarketState,
                 industries: pd.Series = None, # symbol → 行业
                 ) -> AllocationResult:
        """执行动态配比。

        返回 AllocationResult 含底仓和卫星两个子组合。
        """
        base_pct, sat_pct = STATE_ALLOCATION[state]
        pos_range = STATE_POSITIONS[state]

        # 合并可用股票信息
        pool = pd.DataFrame({
            'symbol': scores.index,
            'score': scores.values,
            'price': [prices.get(s, np.nan) for s in scores.index],
            'mcap': [market_caps.get(s, np.nan) for s in scores.index],
        }).dropna(subset=['price', 'mcap'])
        pool = pool[pool['price'] > 0].copy()

        if industries is not None:
            pool['industry'] = pool['symbol'].map(industries)

        # ---- 底仓: 市值加权, 选大市值 ----
        base_pool = pool.nlargest(min(30, len(pool)), 'mcap').copy()
        base_capital = self.capital * base_pct
        n_base = min(pos_range['base'][1], len(base_pool))
        n_base = max(pos_range['base'][0], n_base)
        base_holdings = self._equal_weight_allocate(
            base_pool.head(n_base), base_capital, 'mcap'
        )

        # ---- 卫星: ML分数加权, 选Top排名 ----
        sat_pool = pool.nlargest(min(30, len(pool)), 'score').copy()
        sat_capital = self.capital * sat_pct
        n_sat = min(pos_range['sat'][1], len(sat_pool))
        n_sat = max(pos_range['sat'][0], n_sat) if sat_capital > 5000 else 0
        sat_holdings = self._score_weighted_allocate(
            sat_pool.head(n_sat), sat_capital
        )

        return AllocationResult(
            state=state,
            base_ratio=base_pct,
            sat_ratio=sat_pct,
            base_capital=base_capital,
            sat_capital=sat_capital,
            base_positions=base_holdings,
            sat_positions=sat_holdings,
            oamv_snapshot={},
        )

    def _equal_weight_allocate(self, pool: pd.DataFrame,
                                capital: float, weight_col: str = 'mcap') -> pd.DataFrame:
        """等权/市值加权分配底仓（整数手约束）"""
        if pool.empty or capital <= 0:
            return pd.DataFrame(columns=['symbol', 'shares', 'lots', 'amount', 'weight'])

        n = len(pool)
        target_per_stock = capital / n
        results = []
        used = 0.0

        for _, row in pool.iterrows():
            price = row['price']
            max_shares = int(target_per_stock / price / self.lot_size) * self.lot_size
            max_shares = max(self.lot_size, max_shares)
            cost = max_shares * price * (1 + self.commission_rate)
            if cost > target_per_stock * 1.1:
                max_shares = int(target_per_stock * 0.95 / price / self.lot_size) * self.lot_size
                max_shares = max(0, max_shares)
                cost = max_shares * price * (1 + self.commission_rate) if max_shares > 0 else 0

            if max_shares > 0 and cost <= target_per_stock * 1.2:
                results.append({
                    'symbol': row['symbol'],
                    'shares': max_shares,
                    'lots': max_shares // self.lot_size,
                    'amount': round(cost, 2),
                    'weight': 0.0,  # 后续计算
                })
                used += cost

        result = pd.DataFrame(results)
        if not result.empty:
            result['weight'] = result['amount'] / result['amount'].sum()
        return result

    def _score_weighted_allocate(self, pool: pd.DataFrame,
                                  capital: float) -> pd.DataFrame:
        """分数加权分配卫星仓（高分数→多配）"""
        if pool.empty or capital <= 0:
            return pd.DataFrame(columns=['symbol', 'shares', 'lots', 'amount', 'weight'])

        # 分数归一化为权重
        scores = pool['score'].values
        if scores.std() > 0:
            weights = np.exp((scores - scores.mean()) / scores.std())
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(pool)) / len(pool)

        results = []
        used = 0.0

        for i, (_, row) in enumerate(pool.iterrows()):
            target = capital * weights[i]
            price = row['price']
            max_shares = int(target / price / self.lot_size) * self.lot_size
            max_shares = max(self.lot_size, max_shares)
            cost = max_shares * price * (1 + self.commission_rate)
            if cost > target * 1.2:
                max_shares = max(0, max_shares - self.lot_size)
                cost = max_shares * price * (1 + self.commission_rate) if max_shares > 0 else 0

            if max_shares > 0 and cost <= target * 1.3:
                results.append({
                    'symbol': row['symbol'],
                    'shares': max_shares,
                    'lots': max_shares // self.lot_size,
                    'amount': round(cost, 2),
                    'weight': 0.0,
                })
                used += cost

        result = pd.DataFrame(results)
        if not result.empty:
            result['weight'] = result['amount'] / result['amount'].sum()
        return result

    def summary(self, result: AllocationResult) -> dict:
        """生成配比摘要"""
        base_count = len(result.base_positions)
        sat_count = len(result.sat_positions)
        base_amt = result.base_positions['amount'].sum() if base_count > 0 else 0
        sat_amt = result.sat_positions['amount'].sum() if sat_count > 0 else 0
        total_used = base_amt + sat_amt

        base_symbols = result.base_positions['symbol'].tolist() if base_count > 0 else []
        sat_symbols = result.sat_positions['symbol'].tolist() if sat_count > 0 else []

        return {
            'state': result.state.value,
            'base_ratio': f"{result.base_ratio:.0%}",
            'sat_ratio': f"{result.sat_ratio:.0%}",
            'base_capital': round(base_amt, 0),
            'sat_capital': round(sat_amt, 0),
            'total_used': round(total_used, 0),
            'utilization': f"{total_used/self.capital:.1%}",
            'base_count': base_count,
            'sat_count': sat_count,
            'total_count': base_count + sat_count,
            'base_symbols': base_symbols,
            'sat_symbols': sat_symbols,
        }
