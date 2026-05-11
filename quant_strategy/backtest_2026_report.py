#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026年回测 + 缓存填充 + 每日持仓/净值/超额报表生成

流程:
  1. 加载CSI500成分股数据
  2. 检查/填充因子面板缓存 (首次13min，后续秒级)
  3. 用缓存运行回测
  4. 生成: 每日持仓表 / 净值对比表 / 超额序列
"""

import sys
import os
from pathlib import Path

# Windows控制台UTF-8编码，解决中文乱码
if sys.platform == 'win32':
    try:
        import subprocess
        subprocess.run('chcp 65001 > nul', shell=True, check=False)
    except Exception:
        pass
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
import time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("report")

OUT_DIR = ROOT / "output"
KLINE_DIR = OUT_DIR / "kline"
REPORT_DIR = OUT_DIR / "report_2026"
REPORT_DIR.mkdir(exist_ok=True)


def load_csi500_stocks():
    """加载CSI500成分股数据（过滤到仅在CSI500内的股票）。"""
    const_path = OUT_DIR / "index_constituent_lookup.pkl"
    if not const_path.exists():
        raise FileNotFoundError("成分股数据不存在")

    const_data = pd.read_pickle(const_path)
    stock_dict = const_data.get('lookup', {})
    stats = const_data.get('stats', {})
    latest_month = max(stats.keys())
    csi500_syms = frozenset(
        sym for (ym, sym), in_idx in stock_dict.items()
        if ym == latest_month and in_idx
    )
    csi500_syms = frozenset(s for s in csi500_syms if not str(s).startswith(('8', '9')))

    stock_data = {}
    for csv_path in sorted(KLINE_DIR.glob("*.csv")):
        sym = csv_path.stem
        if sym not in csi500_syms:
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if len(df) > 500:
                stock_data[sym] = df
        except Exception:
            pass

    logger.info("CSI500成分股: %d/%d 只有数据", len(stock_data), len(csi500_syms))
    return stock_data


def load_oamv_data():
    from indicators import sma
    csv_path = ROOT.parent / "output" / "market_turnover.csv"
    if not csv_path.exists():
        return None
    raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    result = pd.DataFrame(index=raw.index)
    result['oamv'] = sma(raw['amount_total'], n=10, m=1) / 1e8
    result['cyc5'] = result['oamv'].ewm(span=5, adjust=False).mean()
    result['cyc13'] = result['oamv'].ewm(span=13, adjust=False).mean()
    if 'close_sh' in raw.columns:
        result['close_sh'] = raw['close_sh']  # v2: 背离检测需要上证指数
    return result


def load_csi500_benchmark():
    csv_path = OUT_DIR / "kline_csi500.csv"
    df = pd.read_csv(csv_path, parse_dates=["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    df['ret'] = df['close'].pct_change()
    return df


def compute_and_cache_panel(stock_data):
    """计算因子面板并缓存（仅首次需要，后续从缓存秒级加载）。"""
    from quant_strategy.utils.factor_cache import FactorCache
    cache = FactorCache()

    if cache.has_valid_cache(stock_data):
        logger.info("缓存命中，跳过因子计算")
        return cache.load()

    logger.info("缓存未命中，开始计算因子面板（仅此一次）...")
    from quant_strategy.factors.factor_engine import FactorEngine

    oamv_df = load_oamv_data()
    engine = FactorEngine(oamv_df=oamv_df)

    # 确保date为索引
    stock_idx = {}
    for sym, df in stock_data.items():
        if 'date' in df.columns:
            df = df.set_index('date').sort_index()
        stock_idx[sym] = df

    panel = engine.compute_panel(stock_idx, n_jobs=-1)
    cache.save(panel, stock_data)
    return panel


def load_model():
    from quant_strategy.cross_section.strategy_engine import IndexEnhancedStrategy
    model_path = OUT_DIR / "strategy_model"
    return IndexEnhancedStrategy.load(str(model_path))


def run_backtest(stock_data, panel):
    """使用缓存面板运行回测。"""
    from quant_strategy.cross_section.strategy_engine import IndexEnhancedStrategy

    oamv_df = load_oamv_data()
    strategy = load_model()
    logger.info("模型: %d因子, B2=%s", len(strategy.factor_names),
                '启用' if strategy.config.use_b2_filter else '关闭')

    logger.info("开始回测 (使用缓存面板)...")
    result = strategy.backtest(
        stock_data=stock_data,
        oamv_df=oamv_df,
        start='2026-01-02',
        end='2026-05-08',
        precomputed_panel=panel,
    )
    return result, strategy


def generate_reports(result, strategy, benchmark_df):
    """生成每日持仓表、净值表、超额序列。"""
    snapshots = result.get('daily_snapshots', pd.DataFrame())
    trade_log = result.get('trade_log', pd.DataFrame())
    positions = result.get('final_positions', pd.DataFrame())

    if snapshots.empty:
        logger.error("无快照数据")
        return

    # ==============================
    # 1. 每日净值表
    # ==============================
    nav = snapshots.set_index('date')
    nav['nav'] = nav['total_equity'] / strategy.config.total_capital
    nav['daily_return'] = nav['nav'].pct_change()
    nav['cum_return'] = nav['nav'] - 1

    # 对齐基准
    start = pd.Timestamp('2026-01-02')
    end = pd.Timestamp('2026-05-08')
    bench = benchmark_df[(benchmark_df.index >= start) & (benchmark_df.index <= end)].copy()

    common_dates = nav.index.intersection(bench.index)
    nav_aligned = nav.loc[common_dates]

    # 归一化: 起始日=1
    bench_aligned = bench.loc[common_dates]
    bench_norm = bench_aligned['close'] / bench_aligned['close'].iloc[0]
    strat_norm = nav_aligned['nav'] / nav_aligned['nav'].iloc[0]

    # 每日超额
    daily_excess = nav_aligned['daily_return'] - bench_aligned['ret']

    # ---- 构建综合净值对比表 ----
    nav_report = pd.DataFrame({
        'date': common_dates,
        'strategy_nav': strat_norm.values.round(4),
        'csi500_nav': bench_norm.values.round(4),
        'excess_nav': (strat_norm.values - bench_norm.values).round(4),
        'strategy_daily_ret': nav_aligned['daily_return'].values.round(6),
        'csi500_daily_ret': bench_aligned['ret'].values.round(6),
        'daily_excess': daily_excess.values.round(6),
        'strategy_cum_ret': nav_aligned['cum_return'].values.round(6),
        'csi500_cum_ret': (bench_norm.values - 1).round(6),
        'excess_cum': (strat_norm.values - bench_norm.values).round(6),
        'n_positions': nav_aligned['n_positions'].values,
        'cash': nav_aligned['cash'].values.round(0),
        'position_value': nav_aligned['position_value'].values.round(0),
        'total_equity': nav_aligned['total_equity'].values.round(0),
        'oamv_state': nav_aligned['oamv_state'].values,
        'oamv_divergence': nav_aligned['oamv_divergence'].values if 'oamv_divergence' in nav_aligned.columns else 'none',
    })
    nav_report.to_csv(REPORT_DIR / 'daily_nav.csv', index=False)
    logger.info("每日净值表: %d 行 → %s", len(nav_report), REPORT_DIR / 'daily_nav.csv')

    # ---- 2. 每日持仓明细表 ----
    if not trade_log.empty:
        tl = trade_log.copy()
        tl['date'] = pd.to_datetime(tl['date'])

        # 从交易日志重建每日持仓
        daily_positions = []
        for d in sorted(nav.index):
            # 当日之前的买入 - 当日之前的卖出 = 当前持仓
            buys_before = tl[(tl['date'] <= d) & (tl['action'] == 'BUY')]
            sells_before = tl[(tl['date'] <= d) & (tl['action'] == 'SELL')]

            held = {}
            for _, b in buys_before.iterrows():
                sym = b['symbol']
                held[sym] = held.get(sym, 0) + b['shares']
            for _, s in sells_before.iterrows():
                sym = s['symbol']
                held[sym] = held.get(sym, 0) - s['shares']

            active = {s: q for s, q in held.items() if q > 0}
            if active:
                for sym, shares in active.items():
                    daily_positions.append({
                        'date': d,
                        'symbol': sym,
                        'shares': shares,
                    })

        pos_df = pd.DataFrame(daily_positions)
        if not pos_df.empty:
            pos_df.to_csv(REPORT_DIR / 'daily_positions.csv', index=False)
            logger.info("每日持仓表: %d 行 → %s", len(pos_df), REPORT_DIR / 'daily_positions.csv')

    # ---- 3. 超额序列摘要 ----
    excess_summary = pd.DataFrame({
        'date': common_dates,
        'daily_excess': daily_excess.values,
        'excess_cum': (strat_norm.values - bench_norm.values),
        'strategy_cum': strat_norm.values,
        'csi500_cum': bench_norm.values,
    })
    excess_summary.to_csv(REPORT_DIR / 'daily_excess.csv', index=False)
    logger.info("超额序列: %d 行 → %s", len(excess_summary), REPORT_DIR / 'daily_excess.csv')

    # ---- 4. 月频汇总 ----
    monthly = nav_aligned.copy()
    monthly['month'] = monthly.index.to_period('M')
    monthly_groups = monthly.groupby('month')
    monthly_report = pd.DataFrame({
        'month': [str(m) for m in monthly_groups['nav'].last().index],
        'strategy_monthly_ret': monthly_groups['nav'].apply(lambda x: x.iloc[-1] / x.iloc[0] - 1).values,
        'csi500_monthly_ret': [bench_aligned['close'][bench_aligned.index.to_period('M') == m].iloc[-1] /
                               bench_aligned['close'][bench_aligned.index.to_period('M') == m].iloc[0] - 1
                               if (bench_aligned.index.to_period('M') == m).sum() > 1 else 0
                               for m in monthly_groups['nav'].last().index],
        'n_trading_days': monthly_groups.size().values,
        'avg_positions': monthly_groups['n_positions'].mean().values,
    })
    monthly_report['monthly_excess'] = (monthly_report['strategy_monthly_ret'] -
                                         monthly_report['csi500_monthly_ret'])
    monthly_report.to_csv(REPORT_DIR / 'monthly_summary.csv', index=False)
    logger.info("月频汇总: %d 行 → %s", len(monthly_report), REPORT_DIR / 'monthly_summary.csv')

    # ---- 5. 打印摘要 ----
    print("\n" + "=" * 70)
    print("  2026年 每日报表已生成")
    print("=" * 70)
    print(f"  交易区间: {common_dates[0].date()} ~ {common_dates[-1].date()} ({len(common_dates)}天)")
    print(f"\n  归一化净值 (起始=1.0):")
    print(f"    策略终值:   {strat_norm.iloc[-1]:.4f}")
    print(f"    CSI500终值: {bench_norm.iloc[-1]:.4f}")
    print(f"    超额终值:   {strat_norm.iloc[-1] - bench_norm.iloc[-1]:+.4f}")
    print(f"\n  按月度超额:")
    for _, row in monthly_report.iterrows():
        arrow = '+' if row['monthly_excess'] > 0 else ''
        print(f"    {row['month']}: 策略{row['strategy_monthly_ret']:+.2%}  "
              f"CSI500{row['csi500_monthly_ret']:+.2%}  "
              f"超额{arrow}{row['monthly_excess']:+.2%}  "
              f"({int(row['n_trading_days'])}天, 均持{row['avg_positions']:.1f}只)")

    print(f"\n  输出目录: {REPORT_DIR}")
    print(f"    daily_nav.csv         — 每日净值+超额")
    print(f"    daily_positions.csv   — 每日持仓明细")
    print(f"    daily_excess.csv      — 超额序列")
    print(f"    monthly_summary.csv   — 月频汇总")
    print("=" * 70)


def main():
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("Step 1: 加载CSI500成分股数据...")
    stock_data = load_csi500_stocks()

    logger.info("Step 2: 检查/填充因子缓存...")
    panel = compute_and_cache_panel(stock_data)

    logger.info("Step 3: 运行回测...")
    result, strategy = run_backtest(stock_data, panel)

    if 'error' in result:
        logger.error("回测失败: %s", result['error'])
        return

    logger.info("Step 4: 加载基准并生成报表...")
    benchmark_df = load_csi500_benchmark()
    generate_reports(result, strategy, benchmark_df)

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
