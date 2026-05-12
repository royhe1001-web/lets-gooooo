#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
砖型图波段策略 —— 四定式分类选股（并行加速版）
==============================================
使用 ProcessPoolExecutor 多进程并行处理，预期加速 4-6x。
四定式分类逻辑复用 quant_strategy/strategy_brick.py，消除代码重复。

总原则：白线在黄线上 + 砖型图绿砖变红砖（前一天绿砖，当天红砖）→ 买入信号
"""

import os, sys, time
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.screener_engine import run_screen_parallel, get_code_list

OUT = os.path.join(BASE, 'output')
DATA_DIR = os.path.join(BASE, 'fetch_kline', 'stock_kline')


def _get_latest_date(data_dir):
    """快速获取数据中最新的交易日期（抽样检查）。"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    max_date = None
    for f in files[:10]:
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['date'])
        m = df['date'].max()
        if max_date is None or m > max_date:
            max_date = m
    return max_date


def main():
    print("=" * 60)
    target_date = _get_latest_date(DATA_DIR)
    print(f"  砖型图波段策略 —— 四定式分类选股（并行加速版）")
    print(f"  选股日: {target_date.strftime('%Y-%m-%d')}")
    print("=" * 60)

    t0 = time.time()

    codes = get_code_list(DATA_DIR)
    print(f"\n  候选股票: {len(codes)} 只")
    print(f"  并行引擎启动 (max_workers=8)...", flush=True)

    # 并行筛选
    results = run_screen_parallel('brick', codes, target_date, DATA_DIR)

    # 按定式分组
    pattern1 = [c for c, p in results if p == '定式一']
    pattern2 = [c for c, p in results if p == '定式二']
    pattern3 = [c for c, p in results if p == '定式三']
    pattern4 = [c for c, p in results if p == '定式四']

    total_found = len(results)
    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min) | 选出: {total_found} 只", flush=True)

    # 输出文件
    date_str = target_date.strftime('%Y-%m-%d')
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, f'strategy_brick_{date_str}.txt')

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"砖型图波段策略选股结果\n")
        f.write(f"选股日期: {date_str}\n")
        f.write(f"总原则: 白线在黄线上 + 绿砖变红砖\n")
        f.write(f"{'=' * 50}\n\n")

        f.write(f"定式一（N型上涨，回调>10%不破前低）: {len(pattern1)} 只\n")
        for code in pattern1:
            f.write(f"定式一 {code}\n")
        f.write("\n")

        f.write(f"定式二（横盘突破，横盘>8日放量长阳）: {len(pattern2)} 只\n")
        for code in pattern2:
            f.write(f"定式二 {code}\n")
        f.write("\n")

        f.write(f"定式三（拉升回踩，拉升>10%短暂调整）: {len(pattern3)} 只\n")
        for code in pattern3:
            f.write(f"定式三 {code}\n")
        f.write("\n")

        f.write(f"不满足定式（仅满足总原则）: {len(pattern4)} 只\n")
        for code in pattern4:
            f.write(f"不满足定式 {code}\n")

    print(f"\n  结果文件: {fp}")

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"  选股结果汇总")
    print(f"{'=' * 60}")
    print(f"  定式一（N型上涨）: {len(pattern1)} 只")
    print(f"  定式二（横盘突破）: {len(pattern2)} 只")
    print(f"  定式三（拉升回踩）: {len(pattern3)} 只")
    print(f"  不满足定式:        {len(pattern4)} 只")
    print(f"  {'─' * 30}")
    print(f"  合计（满足总原则）: {total_found} 只")
    print(f"\n  引擎: ProcessPoolExecutor (8 workers)")
    print("  完成。")


if __name__ == '__main__':
    main()
