#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数增强完整策略引擎 — 选股→持仓检查→离场闭环

集成组件:
  1. ML截面排序选股 (LambdaRanker, Top-30因子)
  2. OAMV动态底仓/卫星配比 (DynamicAllocator)
  3. 三信号离场检查 (ExitSignalEngine, Gamma优化参数)
  4. 整数手约束执行 (IntegerLotOptimizer)

策略流程 (每日):
  T日:
    ① 计算因子截面 → 预测得分 → 排名
    ② 检测OAMV市场状态 → 确定底仓/卫星比例
    ③ 选股: 底仓(市值加权) + 卫星(得分加权)
    ④ 持仓检查: 持有≥T+4日的仓位 → 三信号离场评估
    ⑤ 执行: 卖出信号 → T+1开盘卖出; 现金充足 → 买入
    ⑥ 盯市: 按收盘价计算持仓市值和收益

用法:
  # 回测模式
  engine = IndexEnhancedStrategy(config)
  engine.train(stock_data, oamv_df, train_years=(2015,2024))
  report = engine.backtest(stock_data, oamv_df, start='2025-01-01', end='2026-05-08')

  # 模型持久化
  engine.save('output/strategy_model')
  engine = IndexEnhancedStrategy.load('output/strategy_model')
"""

import sys
import os

# Windows控制台UTF-8编码
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


# ================================================================
# 配置
# ================================================================

@dataclass
class StrategyConfig:
    """策略全局配置"""
    # 资金
    total_capital: float = 200_000
    commission_rate: float = 0.0003       # 万三
    lot_size: int = 100

    # 持仓周期
    holding_period: int = 4               # T+4 离场检查
    rebalance_check_days: int = 5         # 每5个交易日检查调仓

    # 选股
    top_k_candidates: int = 30            # Alpha模型候选池大小
    n_factors: int = 40                   # 最优: 3yr窗口 + 40因子

    # B2筛选层 (方案A: Alpha Top-K → B2综合打分 → 精选)
    use_b2_filter: bool = True            # 是否启用B2筛选层
    b2_candidate_pool: int = 30           # Alpha初筛保留数
    b2_select_count: int = 10             # B2精选数 (用于卫星仓位)
    b2_boost_strength: float = 0.5        # B2得分对Alpha得分的最大提升倍率
    b2_factor_names: list = field(default_factory=list)   # B2因子名列表
    b2_factor_weights: dict = field(default_factory=dict)  # B2因子IC归一化权重

    # 风控
    max_single_weight: float = 0.20       # 单只最大仓位
    max_single_amount: float = 40_000     # 单只最大金额
    min_single_amount: float = 5_000      # 单只最低金额
    max_positions: int = 15               # 最大持仓数
    min_positions: int = 5                # 最小持仓数

    # 模型 (短窗口+强正则化: 防过拟合配置)
    model_params: dict = field(default_factory=lambda: dict(
        objective='regression',
        num_leaves=10,
        learning_rate=0.03,
        min_data_in_leaf=30,
        feature_fraction=0.5,
        bagging_fraction=0.7,
        bagging_freq=1,
        early_stopping_rounds=50,
        num_boost_round=3000,
    ))


# ================================================================
# 策略引擎
# ================================================================

class IndexEnhancedStrategy:
    """指数增强完整策略引擎。

    状态机: 选股 → 持仓 → 离场检查 → 执行 → 盯市

    B2筛选层 (方案A): Alpha模型初筛Top-K → B2综合得分重排 → 精选买入
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.model = None               # LambdaRanker
        self.factor_names = None        # 训练时筛选的因子名
        self.preprocessor = None        # 已fit的PreprocessPipeline
        self.cleaner_stats = None       # Cleaner拟合统计量

        # 运行时状态
        self.positions = pd.DataFrame() # 当前持仓
        self.cash = self.config.total_capital
        self.trade_log = []             # 交易记录
        self.daily_snapshots = []       # 每日快照

    def _compute_b2_composite(self, X_t: pd.DataFrame, candidates: pd.Index) -> pd.Series:
        """在候选池内计算B2综合得分 (截面归一化 + IC加权)。

        X_t: 当日因子面板 (index=symbol, columns=factors)
        candidates: Alpha初筛的候选股票列表
        返回: B2综合得分 Series (0~1, 越高越好)
        """
        weights = self.config.b2_factor_weights
        if not weights:
            return pd.Series(0.5, index=candidates)

        composite = pd.Series(0.0, index=candidates)
        total_w = 0.0
        for fname, w in weights.items():
            if fname not in X_t.columns:
                continue
            vals = X_t.loc[X_t.index.intersection(candidates), fname].fillna(0).astype(float)
            if len(vals) < 2:
                continue
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                vals = (vals - vmin) / (vmax - vmin)
            composite.loc[vals.index] += vals * w
            total_w += w

        if total_w > 0:
            composite = composite / total_w
        return composite.clip(0, 1)

    def _apply_b2_filter(self, alpha_scores: pd.Series,
                          X_t: pd.DataFrame) -> pd.Series:
        """B2筛选层: Alpha Top-K → B2重排 → 调整得分。

        返回调整后的得分 Series:
          - 候选池内: Alpha得分 × (1 + boost × B2综合)
          - 候选池外: 原Alpha得分不变
        """
        n_candidates = min(self.config.b2_candidate_pool, len(alpha_scores))
        top_k = alpha_scores.nlargest(n_candidates).index

        b2_composite = self._compute_b2_composite(X_t, top_k)

        adjusted = alpha_scores.copy()
        boost = self.config.b2_boost_strength
        for sym in top_k:
            if sym in b2_composite.index:
                adjusted[sym] = alpha_scores[sym] * (1 + boost * b2_composite[sym])

        return adjusted

    # ================================================================
    # 训练
    # ================================================================

    def train(self,
              stock_data: dict,
              oamv_df: pd.DataFrame,
              train_years: Tuple[int, int] = (2023, 2025),
              val_years: Tuple[int, int] = (2025, 2025),
              verbose: bool = True) -> 'IndexEnhancedStrategy':
        """训练模型并拟合预处理器。

        stock_data: {symbol: DataFrame with OHLCV}
        oamv_df: OAMV日频数据
        """
        from quant_strategy.factors.factor_engine import FactorEngine
        from quant_strategy.preprocess.pipeline import PreprocessPipeline
        from quant_strategy.cross_section.data_builder import RankingDataBuilder
        from quant_strategy.cross_section.ranker import LambdaRanker
        from scipy.stats import spearmanr

        logger.info("=" * 50)
        logger.info("训练指数增强策略模型")

        # 确保date为索引
        stock_data_idx = {}
        for sym, df in stock_data.items():
            if 'date' in df.columns:
                df = df.set_index('date').sort_index()
            stock_data_idx[sym] = df

        # 1. 因子计算
        logger.info("Step 1/5: 计算因子...")
        engine = FactorEngine(oamv_df=oamv_df)
        panel = engine.compute_panel(stock_data_idx)
        matrix = engine.compute_matrix(panel)
        logger.info("  因子矩阵: %d × %d", len(matrix), len(matrix.columns))

        # 收盘价矩阵
        close_dict = {}
        for sym, df in stock_data_idx.items():
            close_dict[sym] = df['close']
        close_df = pd.DataFrame(close_dict).sort_index()

        # 2. 预处理
        logger.info("Step 2/5: 预处理...")
        matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.preprocessor = PreprocessPipeline(
            fill_method='cross_section',
            industry_neutralize=True,
            mcap_neutralize=True,
        )
        cleaned = self.preprocessor.fit_transform(matrix)
        logger.info("  预处理后: %d × %d", *cleaned.shape)

        # 3. 构建训练数据
        logger.info("Step 3/5: 构建训练数据...")
        builder = RankingDataBuilder(
            horizons=(4,),
            n_grades=0,
            train_years=train_years,
            val_years=val_years,
            test_years=(2025, 2026),
        )
        data = builder.build(cleaned, close_df, primary_horizon=4)
        X_t, y_t, q_t = data['train']
        X_v, y_v, q_v = data['val']

        # 4. 因子筛选
        logger.info("Step 4/5: 筛选Top-%d因子...", self.config.n_factors)
        factor_ics = {}
        for col in X_t.columns:
            valid = X_t[col].notna() & y_t.notna()
            if valid.sum() < 100:
                factor_ics[col] = 0.0
            else:
                ic, _ = spearmanr(X_t[col][valid], y_t[valid])
                factor_ics[col] = abs(ic)
        self.factor_names = sorted(factor_ics, key=factor_ics.get, reverse=True)[:self.config.n_factors]
        X_t = X_t[self.factor_names]
        X_v = X_v[self.factor_names]
        logger.info("  保留因子: %s ...", ', '.join(self.factor_names[:5]))

        # 5. 训练模型
        logger.info("Step 5/5: 训练LightGBM...")
        self.model = LambdaRanker(**self.config.model_params)
        self.model.fit(X_t, y_t, q_t, X_v, y_v, q_v, verbose=verbose)
        logger.info("  训练完成, best_iter=%d", self.model.best_iteration_)

        return self

    # ================================================================
    # 回测
    # ================================================================

    def backtest(self,
                 stock_data: dict,
                 oamv_df: pd.DataFrame,
                 start: str = '2025-01-01',
                 end: str = '2026-05-08',
                 precomputed_panel: pd.DataFrame = None,
                 ) -> dict:
        """运行完整回测。

        precomputed_panel: 预计算的因子面板(MultiIndex date,symbol)。
                          提供后跳过最耗时的因子计算步骤。
        返回 业绩指标 + 交易记录 + 每日快照。
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("模型未训练，请先调用 train() 或 load()")

        from quant_strategy.factors.factor_engine import FactorEngine
        from quant_strategy.cross_section.exit_signals import ExitSignalEngine, ExitConfig, compute_brick_signals
        from quant_strategy.cross_section.dynamic_allocator import (
            DynamicAllocator, OAMVStateDetector, MarketState
        )

        logger.info("=" * 50)
        logger.info("回测: %s → %s", start, end)

        # 准备数据
        stock_data_idx = {}
        for sym, df in stock_data.items():
            if 'date' in df.columns:
                df = df.set_index('date').sort_index()
            stock_data_idx[sym] = df

        # 因子预计算（支持缓存复用，避免重复计算）
        factor_engine = FactorEngine(oamv_df=oamv_df)
        if precomputed_panel is not None:
            logger.info("使用预计算因子面板 (跳过 %.0f分钟因子计算)",
                        len(precomputed_panel) / 1e6 * 0.5)
            panel = precomputed_panel
        else:
            logger.info("预计算因子面板...")
            panel = factor_engine.compute_panel(stock_data_idx, n_jobs=-1)
        matrix = factor_engine.compute_matrix(panel)
        matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
        cleaned = self.preprocessor.transform(matrix)
        # 拆分预测列(Alpha)与全量列(Alpha+B2), 避免feature mismatch
        pred_cols = [c for c in self.factor_names if c in cleaned.columns]
        b2_extra = []
        if self.config.use_b2_filter and self.config.b2_factor_weights:
            b2_extra = [c for c in self.config.b2_factor_names
                       if c in cleaned.columns and c not in pred_cols]
        all_cols = pred_cols + b2_extra
        cleaned = cleaned[all_cols]
        logger.info("  因子面板: %d × %d (含 %d B2筛选因子), 日期范围 %s ~ %s",
                    len(cleaned), len(all_cols), len(b2_extra),
                    cleaned.index.get_level_values('date').min().date(),
                    cleaned.index.get_level_values('date').max().date())

        # 收盘价矩阵
        close_dict = {}
        for sym, df in stock_data_idx.items():
            close_dict[sym] = df['close']
        close_df = pd.DataFrame(close_dict).sort_index()

        # OHLC数据（砖型图用）
        ohlc_data = {}
        for sym, df in stock_data_idx.items():
            if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
                ohlc_data[sym] = df

        # 市值数据
        mcap_series = self._load_mcap_series()

        # 组件初始化
        exit_engine = ExitSignalEngine(ExitConfig())
        oamv_detector = OAMVStateDetector(oamv_df) if oamv_df is not None else None
        allocator = DynamicAllocator(
            capital=self.config.total_capital,
            commission_rate=self.config.commission_rate,
            lot_size=self.config.lot_size,
        )

        # 回测状态
        trading_dates = sorted(close_df.index)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        test_dates = [d for d in trading_dates if start_ts <= d <= end_ts]

        if len(test_dates) < 20:
            return {'error': f'回测日期不足: {len(test_dates)}天'}

        self.positions = pd.DataFrame(columns=[
            'symbol', 'shares', 'entry_date', 'entry_price', 'side'
        ])
        self.cash = self.config.total_capital
        self.trade_log = []
        self.daily_snapshots = []

        date_to_pos = {d: i for i, d in enumerate(trading_dates)}
        last_rebalance = None

        logger.info("开始逐日模拟 (%d 个交易日)...", len(test_dates))

        for i, t_date in enumerate(test_dates):
            # ---- 1. 截面因子 + 预测得分 ----
            try:
                X_t = cleaned.xs(t_date, level='date')
            except KeyError:
                continue

            if len(X_t) < 5:
                continue

            scores = self.model.predict(X_t[pred_cols])

            # B2筛选层: Alpha初筛 → B2重排
            if self.config.use_b2_filter and self.config.b2_factor_weights:
                scores = self._apply_b2_filter(scores, X_t)

            n_stocks = len(scores)
            rank_pcts = pd.Series(index=scores.index, dtype=float)
            for j, (sym, _) in enumerate(scores.sort_values(ascending=False).items()):
                rank_pcts[sym] = j / n_stocks

            # ---- 2. OAMV状态 (v2: 详细状态含背离) ----
            oamv_state = 'neutral'
            oamv_divergence = 'none'
            if oamv_detector:
                try:
                    detail = oamv_detector.detect_with_detail(t_date)
                    oamv_state = detail.state.value
                    oamv_divergence = detail.divergence
                except Exception:
                    pass

            # ---- 3. 获取当日收盘价 ----
            prices_t = pd.Series(index=scores.index, dtype=float)
            for sym in scores.index:
                if sym in close_df.columns:
                    try:
                        prices_t[sym] = float(close_df[sym].loc[t_date])
                    except KeyError:
                        prices_t[sym] = np.nan
            prices_t = prices_t.dropna()

            # ---- 4. 持仓离场检查 (≥T+4) ----
            sell_orders = []
            if not self.positions.empty:
                remaining = []
                for _, pos in self.positions.iterrows():
                    sym = pos['symbol']
                    entry_date = pd.Timestamp(pos['entry_date'])
                    held_days = (t_date - entry_date).days

                    if held_days < self.config.holding_period:
                        remaining.append(pos.to_dict())
                        continue

                    # 获取当前数据
                    if sym not in prices_t.index:
                        remaining.append(pos.to_dict())
                        continue

                    current_price = prices_t[sym]
                    rank_pct = float(rank_pcts.get(sym, 0.5))

                    # 价格历史
                    mask_p = (close_df.index > entry_date) & (close_df.index <= t_date)
                    price_hist = close_df[sym][mask_p].dropna() if sym in close_df.columns else pd.Series()

                    # 砖型信号
                    brick = {}
                    if sym in ohlc_data:
                        sym_ohlc = ohlc_data[sym]
                        mask_b = sym_ohlc.index <= t_date
                        if mask_b.sum() >= 20:
                            brick = compute_brick_signals(sym_ohlc[mask_b],
                                                          brick_size=0.03)

                    # 三信号评估
                    decision = exit_engine.evaluate(
                        symbol=sym,
                        entry_date=entry_date,
                        current_date=t_date,
                        current_rank_pct=rank_pct,
                        brick_signal=brick,
                        price_history=price_hist,
                        oamv_state=oamv_state,
                        entry_price=float(pos['entry_price']),
                        current_price=current_price,
                        oamv_divergence=oamv_divergence,
                    )

                    if decision['action'] == 'sell':
                        sell_orders.append({
                            **pos.to_dict(),
                            'exit_date': t_date,
                            'exit_price': current_price,
                            'return_pct': decision['return_pct'],
                            'reasons': ','.join(decision['reasons']),
                        })
                    else:
                        remaining.append(pos.to_dict())

                self.positions = pd.DataFrame(remaining) if remaining else pd.DataFrame(
                    columns=['symbol', 'shares', 'entry_date', 'entry_price', 'side'])

            # ---- 5. 执行卖出 ----
            for sold in sell_orders:
                proceeds = sold['shares'] * sold['exit_price'] * (1 - self.config.commission_rate)
                self.cash += proceeds
                self.trade_log.append({
                    'date': t_date,
                    'symbol': sold['symbol'],
                    'action': 'SELL',
                    'shares': sold['shares'],
                    'price': sold['exit_price'],
                    'amount': proceeds,
                    'return_pct': sold['return_pct'],
                    'reasons': sold['reasons'],
                    'entry_date': sold['entry_date'],
                    'entry_price': sold['entry_price'],
                })

            # ---- 6. 选股买入 ----
            should_rebalance = (
                last_rebalance is None
                or (t_date - last_rebalance).days >= self.config.rebalance_check_days
                or len(self.positions) < self.config.min_positions
            )

            if should_rebalance and self.cash > self.config.min_single_amount:
                n_current = len(self.positions)
                max_new = self.config.max_positions - n_current

                if max_new > 0:
                    # 获取市值
                    mcaps = pd.Series(index=scores.index, dtype=float)
                    if mcap_series is not None:
                        for sym in scores.index:
                            mcaps[sym] = mcap_series.get((t_date.strftime('%Y-%m'), sym), np.nan)
                    mcaps = mcaps.dropna()
                    if mcaps.empty:
                        mcaps = pd.Series(1e10, index=scores.index)  # fallback

                    try:
                        alloc_result = allocator.allocate(
                            scores=scores,
                            prices=prices_t,
                            market_caps=mcaps,
                            state=MarketState(oamv_state),
                        )
                    except Exception:
                        alloc_result = None

                    # 合并底仓+卫星候选
                    buy_candidates = []
                    existing_symbols = set(self.positions['symbol'].tolist()) if not self.positions.empty else set()

                    if alloc_result is not None:
                        for _, row in alloc_result.base_positions.iterrows():
                            if row['symbol'] not in existing_symbols:
                                buy_candidates.append({
                                    'symbol': row['symbol'],
                                    'target_amount': row['amount'],
                                    'side': 'base',
                                })
                        for _, row in alloc_result.sat_positions.iterrows():
                            if row['symbol'] not in existing_symbols:
                                buy_candidates.append({
                                    'symbol': row['symbol'],
                                    'target_amount': row['amount'],
                                    'side': 'sat',
                                })

                    # 如果动态分配失败，直接用排名选
                    if not buy_candidates:
                        top_syms = scores.nlargest(min(max_new, len(scores))).index
                        top_syms = [s for s in top_syms if s not in existing_symbols]
                        per_stock = self.cash * 0.8 / max(len(top_syms), 1)
                        for sym in top_syms[:max_new]:
                            buy_candidates.append({
                                'symbol': sym,
                                'target_amount': min(per_stock, self.config.max_single_amount),
                                'side': 'score',
                            })

                    # 执行买入
                    for bc in buy_candidates[:max_new]:
                        sym = bc['symbol']
                        if sym not in prices_t.index:
                            continue
                        price = prices_t[sym]
                        target = min(bc['target_amount'], self.cash * 0.25,
                                     self.config.max_single_amount)
                        if target < self.config.min_single_amount:
                            continue

                        n_shares = int(target / price / self.config.lot_size) * self.config.lot_size
                        n_shares = max(self.config.lot_size, n_shares)
                        cost = n_shares * price * (1 + self.config.commission_rate)

                        if cost > self.cash * 0.3:
                            n_shares = int(self.cash * 0.25 / price / self.config.lot_size) * self.config.lot_size
                            n_shares = max(0, n_shares)
                            cost = n_shares * price * (1 + self.config.commission_rate) if n_shares > 0 else 0

                        if n_shares > 0 and cost <= self.cash:
                            self.cash -= cost
                            new_pos = {
                                'symbol': sym,
                                'shares': n_shares,
                                'entry_date': t_date,
                                'entry_price': price,
                                'side': bc['side'],
                            }
                            self.positions = pd.concat([
                                self.positions,
                                pd.DataFrame([new_pos])
                            ], ignore_index=True)

                            self.trade_log.append({
                                'date': t_date,
                                'symbol': sym,
                                'action': 'BUY',
                                'shares': n_shares,
                                'price': price,
                                'amount': cost,
                                'return_pct': 0.0,
                                'reasons': f"side={bc['side']}",
                                'entry_date': t_date,
                                'entry_price': price,
                            })

                last_rebalance = t_date

            # ---- 7. 盯市 ----
            position_value = 0.0
            pos_details = []
            if not self.positions.empty:
                for _, pos in self.positions.iterrows():
                    sym = pos['symbol']
                    if sym in prices_t.index:
                        mkt_val = pos['shares'] * prices_t[sym]
                    else:
                        mkt_val = pos['shares'] * pos['entry_price']
                    position_value += mkt_val
                    pos_details.append({
                        'symbol': sym,
                        'shares': pos['shares'],
                        'entry_price': pos['entry_price'],
                        'current_price': prices_t.get(sym, pos['entry_price']),
                        'mkt_value': mkt_val,
                        'return_pct': (prices_t.get(sym, pos['entry_price']) - pos['entry_price']) / pos['entry_price'],
                        'held_days': (t_date - pd.Timestamp(pos['entry_date'])).days,
                        'side': pos.get('side', 'unknown'),
                    })

            total_equity = self.cash + position_value

            self.daily_snapshots.append({
                'date': t_date,
                'cash': self.cash,
                'position_value': position_value,
                'total_equity': total_equity,
                'n_positions': len(self.positions),
                'oamv_state': oamv_state,
                'oamv_divergence': oamv_divergence,
                'n_sells_today': len(sell_orders),
                'n_buys_today': len([t for t in self.trade_log if t['date'] == t_date and t['action'] == 'BUY']),
            })

            if (i + 1) % 50 == 0:
                ret = (total_equity / self.config.total_capital - 1) * 100
                logger.info("  %s | 净值 %.4f (%+.2f%%) | 持仓 %d | 现金 %.0f",
                            t_date.date(), total_equity / self.config.total_capital,
                            ret, len(self.positions), self.cash)

        # ---- 汇总统计 ----
        logger.info("回测完成，计算业绩指标...")
        metrics = self._compute_metrics(test_dates)

        return {
            'metrics': metrics,
            'trade_log': pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame(),
            'daily_snapshots': pd.DataFrame(self.daily_snapshots),
            'final_positions': self.positions,
        }

    def backtest_with_data(self,
                           cleaned: pd.DataFrame,
                           close_df: pd.DataFrame,
                           ohlc_data: dict,
                           oamv_df: pd.DataFrame,
                           start: str = '2025-01-01',
                           end: str = '2026-05-08',
                           ) -> dict:
        """使用预计算因子数据运行回测（跳过因子计算，用于并行多窗口回测）。

        cleaned: 预处理后的因子矩阵, MultiIndex(date, symbol), 已筛选至 self.factor_names
        close_df: 收盘价矩阵, index=date, columns=symbol
        ohlc_data: {symbol: DataFrame with OHLC}
        oamv_df: OAMV日频数据
        """
        if self.model is None:
            raise RuntimeError("模型未训练")

        from quant_strategy.cross_section.exit_signals import ExitSignalEngine, ExitConfig, compute_brick_signals
        from quant_strategy.cross_section.dynamic_allocator import (
            DynamicAllocator, OAMVStateDetector, MarketState
        )

        # 拆分预测列(Alpha)与全量列(Alpha+B2)
        pred_cols = [c for c in self.factor_names if c in cleaned.columns]
        b2_extra = []
        if self.config.use_b2_filter and self.config.b2_factor_weights:
            b2_extra = [c for c in self.config.b2_factor_names
                       if c in cleaned.columns and c not in pred_cols]
        all_cols = pred_cols + b2_extra
        cleaned = cleaned[all_cols]

        # 市值数据
        mcap_series = self._load_mcap_series()

        # 组件初始化
        exit_engine = ExitSignalEngine(ExitConfig())
        oamv_detector = OAMVStateDetector(oamv_df) if oamv_df is not None else None
        allocator = DynamicAllocator(
            capital=self.config.total_capital,
            commission_rate=self.config.commission_rate,
            lot_size=self.config.lot_size,
        )

        # 回测状态
        trading_dates = sorted(close_df.index)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        test_dates = [d for d in trading_dates if start_ts <= d <= end_ts]

        if len(test_dates) < 20:
            return {'error': f'回测日期不足: {len(test_dates)}天'}

        self.positions = pd.DataFrame(columns=[
            'symbol', 'shares', 'entry_date', 'entry_price', 'side'
        ])
        self.cash = self.config.total_capital
        self.trade_log = []
        self.daily_snapshots = []
        date_to_pos = {d: i for i, d in enumerate(trading_dates)}
        last_rebalance = None

        logger.info("开始逐日模拟 (%d 个交易日)...", len(test_dates))

        for i, t_date in enumerate(test_dates):
            # ---- 1. 截面因子 + 预测得分 ----
            try:
                X_t = cleaned.xs(t_date, level='date')
            except KeyError:
                continue
            if len(X_t) < 5:
                continue

            scores = self.model.predict(X_t[pred_cols])

            # B2筛选层: Alpha初筛 → B2重排
            if self.config.use_b2_filter and self.config.b2_factor_weights:
                scores = self._apply_b2_filter(scores, X_t)

            n_stocks = len(scores)
            rank_pcts = pd.Series(index=scores.index, dtype=float)
            for j, (sym, _) in enumerate(scores.sort_values(ascending=False).items()):
                rank_pcts[sym] = j / n_stocks

            # ---- 2. OAMV状态 (v2: 详细状态含背离) ----
            oamv_state = 'neutral'
            oamv_divergence = 'none'
            if oamv_detector:
                try:
                    detail = oamv_detector.detect_with_detail(t_date)
                    oamv_state = detail.state.value
                    oamv_divergence = detail.divergence
                except Exception:
                    pass

            # ---- 3. 当日收盘价 ----
            prices_t = pd.Series(index=scores.index, dtype=float)
            for sym in scores.index:
                if sym in close_df.columns:
                    try:
                        prices_t[sym] = float(close_df[sym].loc[t_date])
                    except KeyError:
                        prices_t[sym] = np.nan
            prices_t = prices_t.dropna()

            # ---- 4. 离场检查 (≥T+4) ----
            sell_orders = []
            if not self.positions.empty:
                remaining = []
                for _, pos in self.positions.iterrows():
                    sym = pos['symbol']
                    entry_date = pd.Timestamp(pos['entry_date'])
                    held_days = (t_date - entry_date).days
                    if held_days < self.config.holding_period:
                        remaining.append(pos.to_dict())
                        continue
                    if sym not in prices_t.index:
                        remaining.append(pos.to_dict())
                        continue

                    current_price = prices_t[sym]
                    rank_pct = float(rank_pcts.get(sym, 0.5))
                    mask_p = (close_df.index > entry_date) & (close_df.index <= t_date)
                    price_hist = close_df[sym][mask_p].dropna() if sym in close_df.columns else pd.Series()

                    brick = {}
                    if sym in ohlc_data:
                        sym_ohlc = ohlc_data[sym]
                        mask_b = sym_ohlc.index <= t_date
                        if mask_b.sum() >= 20:
                            brick = compute_brick_signals(sym_ohlc[mask_b], brick_size=0.03)

                    decision = exit_engine.evaluate(
                        symbol=sym, entry_date=entry_date, current_date=t_date,
                        current_rank_pct=rank_pct, brick_signal=brick,
                        price_history=price_hist, oamv_state=oamv_state,
                        entry_price=float(pos['entry_price']), current_price=current_price,
                        oamv_divergence=oamv_divergence,
                    )
                    if decision['action'] == 'sell':
                        sell_orders.append({**pos.to_dict(), 'exit_date': t_date,
                                           'exit_price': current_price,
                                           'return_pct': decision['return_pct'],
                                           'reasons': ','.join(decision['reasons'])})
                    else:
                        remaining.append(pos.to_dict())
                self.positions = pd.DataFrame(remaining) if remaining else pd.DataFrame(
                    columns=['symbol', 'shares', 'entry_date', 'entry_price', 'side'])

            # ---- 5. 执行卖出 ----
            for sold in sell_orders:
                proceeds = sold['shares'] * sold['exit_price'] * (1 - self.config.commission_rate)
                self.cash += proceeds
                self.trade_log.append({
                    'date': t_date, 'symbol': sold['symbol'], 'action': 'SELL',
                    'shares': sold['shares'], 'price': sold['exit_price'],
                    'amount': proceeds, 'return_pct': sold['return_pct'],
                    'reasons': sold['reasons'],
                    'entry_date': sold['entry_date'], 'entry_price': sold['entry_price'],
                })

            # ---- 6. 选股买入 ----
            should_rebalance = (
                last_rebalance is None
                or (t_date - last_rebalance).days >= self.config.rebalance_check_days
                or len(self.positions) < self.config.min_positions
            )
            if should_rebalance and self.cash > self.config.min_single_amount:
                n_current = len(self.positions)
                max_new = self.config.max_positions - n_current
                if max_new > 0:
                    mcaps = pd.Series(index=scores.index, dtype=float)
                    if mcap_series is not None:
                        for sym in scores.index:
                            mcaps[sym] = mcap_series.get((t_date.strftime('%Y-%m'), sym), np.nan)
                    mcaps = mcaps.dropna()
                    if mcaps.empty:
                        mcaps = pd.Series(1e10, index=scores.index)

                    try:
                        alloc_result = allocator.allocate(
                            scores=scores, prices=prices_t, market_caps=mcaps,
                            state=MarketState(oamv_state),
                        )
                    except Exception:
                        alloc_result = None

                    buy_candidates = []
                    existing_syms = set(self.positions['symbol'].tolist()) if not self.positions.empty else set()
                    if alloc_result is not None:
                        for _, row in alloc_result.base_positions.iterrows():
                            if row['symbol'] not in existing_syms:
                                buy_candidates.append({'symbol': row['symbol'], 'target_amount': row['amount'], 'side': 'base'})
                        for _, row in alloc_result.sat_positions.iterrows():
                            if row['symbol'] not in existing_syms:
                                buy_candidates.append({'symbol': row['symbol'], 'target_amount': row['amount'], 'side': 'sat'})
                    if not buy_candidates:
                        top_syms = scores.nlargest(min(max_new, len(scores))).index
                        top_syms = [s for s in top_syms if s not in existing_syms]
                        per_stock = self.cash * 0.8 / max(len(top_syms), 1)
                        for sym in top_syms[:max_new]:
                            buy_candidates.append({'symbol': sym, 'target_amount': min(per_stock, self.config.max_single_amount), 'side': 'score'})

                    for bc in buy_candidates[:max_new]:
                        sym = bc['symbol']
                        if sym not in prices_t.index:
                            continue
                        price = prices_t[sym]
                        target = min(bc['target_amount'], self.cash * 0.25, self.config.max_single_amount)
                        if target < self.config.min_single_amount:
                            continue
                        n_shares = int(target / price / self.config.lot_size) * self.config.lot_size
                        n_shares = max(self.config.lot_size, n_shares)
                        cost = n_shares * price * (1 + self.config.commission_rate)
                        if cost > self.cash * 0.3:
                            n_shares = int(self.cash * 0.25 / price / self.config.lot_size) * self.config.lot_size
                            n_shares = max(0, n_shares)
                            cost = n_shares * price * (1 + self.config.commission_rate) if n_shares > 0 else 0
                        if n_shares > 0 and cost <= self.cash:
                            self.cash -= cost
                            new_pos = {'symbol': sym, 'shares': n_shares, 'entry_date': t_date,
                                      'entry_price': price, 'side': bc['side']}
                            self.positions = pd.concat([self.positions, pd.DataFrame([new_pos])], ignore_index=True)
                            self.trade_log.append({
                                'date': t_date, 'symbol': sym, 'action': 'BUY',
                                'shares': n_shares, 'price': price, 'amount': cost,
                                'return_pct': 0.0, 'reasons': f"side={bc['side']}",
                                'entry_date': t_date, 'entry_price': price,
                            })
                last_rebalance = t_date

            # ---- 7. 盯市 ----
            position_value = 0.0
            if not self.positions.empty:
                for _, pos in self.positions.iterrows():
                    sym = pos['symbol']
                    mkt_val = pos['shares'] * prices_t.get(sym, pos['entry_price']) if sym in prices_t.index else pos['shares'] * pos['entry_price']
                    position_value += mkt_val
            total_equity = self.cash + position_value

            self.daily_snapshots.append({
                'date': t_date, 'cash': self.cash, 'position_value': position_value,
                'total_equity': total_equity, 'n_positions': len(self.positions),
                'oamv_state': oamv_state,
                'oamv_divergence': oamv_divergence,
                'n_sells_today': len(sell_orders),
                'n_buys_today': len([t for t in self.trade_log if t['date'] == t_date and t['action'] == 'BUY']),
            })
            if (i + 1) % 50 == 0:
                ret = (total_equity / self.config.total_capital - 1) * 100
                logger.info("  %s | 净值 %.4f (%+.2f%%) | 持仓 %d | 现金 %.0f",
                            t_date.date(), total_equity / self.config.total_capital,
                            ret, len(self.positions), self.cash)

        logger.info("回测完成，计算业绩指标...")
        metrics = self._compute_metrics(test_dates)
        return {
            'metrics': metrics,
            'trade_log': pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame(),
            'daily_snapshots': pd.DataFrame(self.daily_snapshots),
            'final_positions': self.positions,
        }

    # ================================================================
    # 业绩指标
    # ================================================================

    def _compute_metrics(self, test_dates: list) -> dict:
        """从每日快照计算业绩指标"""
        if not self.daily_snapshots:
            return {}

        snap = pd.DataFrame(self.daily_snapshots)
        snap = snap.set_index('date')
        snap['daily_return'] = snap['total_equity'].pct_change()
        snap['cumulative_return'] = snap['total_equity'] / self.config.total_capital - 1

        # 基本指标
        total_return = float(snap['cumulative_return'].iloc[-1])
        n_days = len(snap)
        n_years = n_days / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.5)) - 1

        daily_rets = snap['daily_return'].dropna()
        annual_vol = float(daily_rets.std() * np.sqrt(252)) if len(daily_rets) > 0 else 0
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum_max = snap['total_equity'].cummax()
        drawdowns = (snap['total_equity'] - cum_max) / cum_max
        max_dd = float(drawdowns.min())

        # 胜率
        daily_wr = float((daily_rets > 0).mean()) if len(daily_rets) > 0 else 0

        # 交易统计
        trades_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        n_buys = len(trades_df[trades_df['action'] == 'BUY']) if not trades_df.empty else 0
        n_sells = len(trades_df[trades_df['action'] == 'SELL']) if not trades_df.empty else 0

        sell_rets = trades_df[trades_df['action'] == 'SELL']['return_pct'] if not trades_df.empty else pd.Series()
        avg_sell_return = float(sell_rets.mean()) if len(sell_rets) > 0 else 0
        sell_wr = float((sell_rets > 0).mean()) if len(sell_rets) > 0 else 0

        # 月收益
        snap['month'] = snap.index.to_period('M')
        monthly = snap.groupby('month')['total_equity'].last().pct_change().dropna()
        positive_months = float((monthly > 0).mean()) if len(monthly) > 0 else 0

        # OAMV状态分布
        state_dist = snap['oamv_state'].value_counts().to_dict() if 'oamv_state' in snap.columns else {}

        return {
            'total_return': round(total_return * 100, 2),
            'annual_return': round(annual_return * 100, 2),
            'annual_volatility': round(annual_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'daily_win_rate': round(daily_wr * 100, 1),
            'n_days': n_days,
            'n_buys': n_buys,
            'n_sells': n_sells,
            'avg_sell_return': round(avg_sell_return * 100, 2),
            'sell_win_rate': round(sell_wr * 100, 1),
            'positive_months_ratio': round(positive_months * 100, 1),
            'oamv_state_distribution': state_dist,
            'final_equity': round(float(snap['total_equity'].iloc[-1]), 0),
            'final_positions': int(snap['n_positions'].iloc[-1]),
        }

    # ================================================================
    # 辅助
    # ================================================================

    def _load_mcap_series(self) -> Optional[dict]:
        """加载市值数据"""
        daily_path = Path(__file__).parent.parent.parent / 'output' / 'fundamental' / 'daily_basic.parquet'
        if not daily_path.exists():
            return None
        try:
            db = pd.read_parquet(daily_path)
            db['symbol'] = db['ts_code'].str[:6]
            db['date'] = pd.to_datetime(db['trade_date'])
            db['month'] = db['date'].dt.strftime('%Y-%m')
            db['mv_yi'] = db['total_mv'] / 1e4
            return db.groupby(['month', 'symbol'])['mv_yi'].last().to_dict()
        except Exception:
            return None

    # ================================================================
    # 模型持久化
    # ================================================================

    def save(self, path: str):
        """保存策略模型到磁盘"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存LightGBM模型
        if self.model is not None:
            self.model.save(str(save_path / 'lgb_model.txt'))

        # 保存预处理器
        if self.preprocessor is not None:
            with open(save_path / 'preprocessor.pkl', 'wb') as f:
                pickle.dump(self.preprocessor, f)

        # 保存元数据
        meta = {
            'factor_names': self.factor_names,
            'config': self.config,
        }
        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(meta, f)

        logger.info("策略模型已保存到 %s", save_path)

    @classmethod
    def load(cls, path: str) -> 'IndexEnhancedStrategy':
        """从磁盘加载策略模型"""
        from quant_strategy.cross_section.ranker import LambdaRanker

        load_path = Path(path)

        with open(load_path / 'metadata.pkl', 'rb') as f:
            meta = pickle.load(f)

        config = meta['config']
        # 向后兼容: 旧模型缺少B2字段时补上默认值
        for attr, default in [
            ('use_b2_filter', False), ('b2_candidate_pool', 30),
            ('b2_select_count', 10), ('b2_boost_strength', 0.5),
            ('b2_factor_names', []), ('b2_factor_weights', {}),
        ]:
            if not hasattr(config, attr):
                setattr(config, attr, default)
        strategy = cls(config=config)
        strategy.factor_names = meta['factor_names']

        model_file = load_path / 'lgb_model.txt'
        if model_file.exists():
            strategy.model = LambdaRanker.load(str(model_file))

        pp_file = load_path / 'preprocessor.pkl'
        if pp_file.exists():
            with open(pp_file, 'rb') as f:
                strategy.preprocessor = pickle.load(f)

        logger.info("策略模型已加载 (%d 因子, model=%s, preprocessor=%s)",
                    len(strategy.factor_names) if strategy.factor_names else 0,
                    'OK' if strategy.model else 'MISSING',
                    'OK' if strategy.preprocessor else 'MISSING')
        return strategy

    def print_report(self, result: dict):
        """打印回测报告"""
        m = result.get('metrics', {})
        if not m:
            print("无业绩数据")
            return

        print("\n" + "=" * 60)
        print("  指数增强策略回测报告")
        print("=" * 60)
        print(f"  累计收益:     {m.get('total_return', 0):+.2f}%")
        print(f"  年化收益:     {m.get('annual_return', 0):+.2f}%")
        print(f"  年化波动:     {m.get('annual_volatility', 0):+.2f}%")
        print(f"  Sharpe:       {m.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤:     {m.get('max_drawdown', 0):+.2f}%")
        print(f"  日胜率:       {m.get('daily_win_rate', 0):.1f}%")
        print(f"  月胜率:       {m.get('positive_months_ratio', 0):.1f}%")
        print(f"  交易天数:     {m.get('n_days', 0)}")
        print(f"  买入次数:     {m.get('n_buys', 0)}")
        print(f"  卖出次数:     {m.get('n_sells', 0)}")
        print(f"  卖出均收益:   {m.get('avg_sell_return', 0):+.2f}%")
        print(f"  卖出胜率:     {m.get('sell_win_rate', 0):.1f}%")
        print(f"  最终净值:     {m.get('final_equity', 0):,.0f} 元")
        print(f"  最终持仓:     {m.get('final_positions', 0)} 只")
        state_dist = m.get('oamv_state_distribution', {})
        if state_dist:
            print(f"  状态分布:     {state_dist}")
        print("=" * 60)
