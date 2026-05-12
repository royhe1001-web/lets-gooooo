#!/usr/bin/env python3
"""
b2 强势确认反转选股 —— M1 芯片专用版
======================================
针对 Apple Silicon (M1) + 8GB 统一内存优化。
策略逻辑与 quant_strategy/strategy_b2.py 完全一致。

M1 适配要点:
  - 4 进程（仅 M1 性能核心）
  - float32 精度加载，减半内存
  - 分批处理 + 主动 GC，防止 OOM
  - 显式 spawn 上下文，兼容 macOS
"""

import os, sys, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = BASE
sys.path.insert(0, ROOT)

from screener_engine_m1 import (
    run_screen_parallel_m1, get_code_list, print_system_info
)

OUT = os.path.join(ROOT, 'output')
DATA_DIR = os.path.join(ROOT, 'fetch_kline', 'stock_kline')


def _get_latest_date(data_dir):
    """快速获取最新交易日期（抽样 10 个文件）。"""
    import pandas as pd
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    max_date = None
    for f in files[:10]:
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['date'])
        m = df['date'].max()
        if max_date is None or m > max_date:
            max_date = m
    return max_date


def main():
    print("=" * 70)
    print("  b2 强势确认反转 — M1 芯片专用版")
    print("=" * 70)

    # M1 系统信息
    print("\n>>> M1 系统信息:")
    print_system_info()

    t0 = time.time()

    codes = get_code_list(DATA_DIR)
    target_date = _get_latest_date(DATA_DIR)
    print(f"\n  目标日期: {target_date.date()} | 候选: {len(codes)} 只")
    print(f"  M1 引擎启动 (4 P-cores, {len(codes)//400 + 1} 批)...", flush=True)

    results = run_screen_parallel_m1('b2', codes, target_date, DATA_DIR)

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min) | 选出: {len(results)} 只", flush=True)

    # 输出
    os.makedirs(OUT, exist_ok=True)
    date_str = target_date.strftime('%Y%m%d')
    fp = os.path.join(OUT, f'b2_selection_{date_str}.txt')

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"b2 强势确认反转选股结果 (M1 版)\n")
        f.write(f"策略模块: quant_strategy/strategy_b2.py\n")
        f.write(f"选股日期: {target_date.date()}\n")
        f.write(f"入选数量: {len(results)} 只\n")
        f.write(f"{'='*50}\n\n")
        for i, (code, gain, j_prev, w) in enumerate(results[:15]):
            f.write(f"  {i+1}. {code}  权重={w:.1f}x  涨幅={gain:.1f}%  J前日={j_prev:.1f}\n")
        f.write(f"\n完整列表:\n")
        for code, gain, j_prev, w in results:
            f.write(f"  {code}  {w:.1f}\n")

    print(f"\n  结果: {fp}")
    print(f"\n{'='*70}")
    print(f"  M1 b2 策略 ({target_date.date()}) — Top 10")
    print(f"{'='*70}")
    for i, (code, gain, j_prev, w) in enumerate(results[:10]):
        stars = '★' if i < 3 else '☆'
        print(f"  {stars} {i+1}. {code:>12}  权重 {w:.1f}x  涨幅 {gain:>5.1f}%  "
              f"J前日={j_prev:>5.1f}  {'← 推荐' if i < 3 else ''}")
    if len(results) > 10:
        print(f"  ... 共 {len(results)} 只")

    print(f"\n  引擎: M1 ProcessPoolExecutor (4 P-cores + 分批 GC)")
    print(f"  Done.")


if __name__ == '__main__':
    main()
