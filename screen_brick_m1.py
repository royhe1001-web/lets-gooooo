#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
砖型图波段策略选股 —— M1 芯片专用版
====================================
针对 Apple Silicon (M1) + 8GB 统一内存优化。
四定式分类复用 quant_strategy/strategy_brick.py。

M1 适配要点:
  - 仅计算砖型图+白黄线（跳过高开销的 KDJ/MACD/量价形态）
  - 4 进程（仅 M1 性能核心）
  - float32 精度加载，减半内存
  - 分批处理 + 主动 GC
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
    """快速获取最新交易日期。"""
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
    print("=" * 60)
    target_date = _get_latest_date(DATA_DIR)
    print(f"  砖型图波段策略 —— M1 芯片专用版")
    print(f"  选股日: {target_date.strftime('%Y-%m-%d')}")
    print("=" * 60)

    print("\n>>> M1 系统信息:")
    print_system_info()

    t0 = time.time()

    codes = get_code_list(DATA_DIR)
    print(f"\n  候选: {len(codes)} 只")
    print(f"  M1 引擎启动 (4 P-cores)...", flush=True)

    results = run_screen_parallel_m1('brick', codes, target_date, DATA_DIR)

    # 按定式分组 (result = code, pattern, quality, gain_pct, brick_val)
    pattern1 = [(c, q, g, bv) for c, p, q, g, bv in results if p == '定式一']
    pattern2 = [(c, q, g, bv) for c, p, q, g, bv in results if p == '定式二']
    pattern3 = [(c, q, g, bv) for c, p, q, g, bv in results if p == '定式三']
    pattern4 = [(c, q, g, bv) for c, p, q, g, bv in results if p == '定式四']

    total_found = len(results)
    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min) | 选出: {total_found} 只", flush=True)

    # 输出
    date_str = target_date.strftime('%Y-%m-%d')
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, f'strategy_brick_{date_str}_m1.txt')

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"砖型图波段策略选股结果 (M1 版)\n")
        f.write(f"选股日期: {date_str}\n")
        f.write(f"总原则: 白线在黄线上 + 绿砖变红砖\n")
        f.write(f"{'=' * 50}\n\n")

        for pat_name, codes_list in [('定式一', pattern1), ('定式二', pattern2),
                                      ('定式三', pattern3), ('不满足定式', pattern4)]:
            label = {'定式一': 'N型上涨，回调>10%不破前低',
                     '定式二': '横盘突破，横盘>8日放量长阳',
                     '定式三': '拉升回踩，拉升>10%短暂调整',
                     '不满足定式': '仅满足总原则'}[pat_name]
            f.write(f"{pat_name}（{label}）: {len(codes_list)} 只\n")
            for item in codes_list:
                code, quality, gain_pct, brick_val = item
                f.write(f"{pat_name} {code}  质量{quality:.1f}  涨幅{gain_pct:+.1f}%  砖值{brick_val:.0f}\n")
            f.write("\n")

    print(f"\n  结果: {fp}")
    print(f"\n{'=' * 60}")
    print(f"  M1 选股结果汇总")
    print(f"{'=' * 60}")
    print(f"  定式一（N型上涨）: {len(pattern1)} 只")
    for c, q, g, bv in pattern1[:3]:
        print(f"    ★ {c}  质量{q:.1f}  涨幅{g:+.1f}%  砖值{bv:.0f}")
    print(f"  定式二（横盘突破）: {len(pattern2)} 只")
    for c, q, g, bv in pattern2[:3]:
        print(f"    ★ {c}  质量{q:.1f}  涨幅{g:+.1f}%  砖值{bv:.0f}")
    print(f"  定式三（拉升回踩）: {len(pattern3)} 只")
    for c, q, g, bv in pattern3[:3]:
        print(f"    ★ {c}  质量{q:.1f}  涨幅{g:+.1f}%  砖值{bv:.0f}")
    print(f"  不满足定式:        {len(pattern4)} 只")
    print(f"  {'─' * 30}")
    print(f"  合计: {total_found} 只")
    print(f"\n  引擎: M1 ProcessPoolExecutor (4 P-cores + 分批 GC)")
    print(f"  Done.")


if __name__ == '__main__':
    main()
