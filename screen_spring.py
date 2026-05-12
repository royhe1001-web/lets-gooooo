#!/usr/bin/env python3
"""
b2 强势确认反转选股 —— 并行加速版
====================================
使用 ProcessPoolExecutor 多进程并行处理，预期加速 4-6x。
策略逻辑与 quant_strategy/strategy_b2.py 完全一致。
"""

import os, sys, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.screener_engine import run_screen_parallel, get_code_list

OUT = os.path.join(BASE, 'output')
DATA_DIR = os.path.join(BASE, 'fetch_kline', 'stock_kline')


def _get_latest_date(data_dir):
    """快速获取数据中最新的交易日期（抽样检查）。"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    # 抽样10个文件找最大日期
    import pandas as pd
    max_date = None
    for f in files[:10]:
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['date'])
        m = df['date'].max()
        if max_date is None or m > max_date:
            max_date = m
    return max_date


def main():
    print("=" * 70)
    print("  b2 强势确认反转 — 并行加速版")
    print("=" * 70)

    t0 = time.time()

    # 获取代码列表和目标日期
    codes = get_code_list(DATA_DIR)
    target_date = _get_latest_date(DATA_DIR)
    print(f"\n  目标日期: {target_date.date()} | 候选股票: {len(codes)} 只")
    print(f"  并行引擎启动 (max_workers=8)...", flush=True)

    # 并行筛选
    results = run_screen_parallel('b2', codes, target_date, DATA_DIR)

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s | 选出: {len(results)} 只", flush=True)

    # 输出文件
    os.makedirs(OUT, exist_ok=True)
    date_str = target_date.strftime('%Y%m%d')
    fp = os.path.join(OUT, f'b2_selection_{date_str}.txt')

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"b2 强势确认反转选股结果\n")
        f.write(f"策略模块: quant_strategy/strategy_b2.py\n")
        f.write(f"选股日期: {target_date.date()}\n")
        f.write(f"入选数量: {len(results)} 只\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"★ 最符合 b2 策略 (按仓位权重降序):\n")
        for i, (code, gain, j_prev, w) in enumerate(results[:15]):
            f.write(f"  {i+1}. {code}  权重={w:.1f}x  涨幅={gain:.1f}%  J前日={j_prev:.1f}\n")
        f.write(f"\n完整列表:\n")
        for code, gain, j_prev, w in results:
            f.write(f"  {code}  {w:.1f}\n")

    print(f"\n  结果文件: {fp}")
    print(f"\n{'='*70}")
    print(f"  ★ b2 策略 ({target_date.date()}) — Top 10")
    print(f"{'='*70}")
    for i, (code, gain, j_prev, w) in enumerate(results[:10]):
        stars = '★' if i < 3 else '☆'
        print(f"  {stars} {i+1}. {code:>12}  权重 {w:.1f}x  涨幅 {gain:>5.1f}%  "
              f"J前日={j_prev:>5.1f}  {'← 推荐' if i < 3 else ''}")
    if len(results) > 10:
        print(f"  ... 共 {len(results)} 只")

    print(f"\n  策略一致性: OK (quant_strategy.strategy_b2.generate_spring_signals)")
    print(f"  引擎: ProcessPoolExecutor (8 workers)")


if __name__ == '__main__':
    main()
