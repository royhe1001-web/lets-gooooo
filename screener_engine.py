#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行筛选引擎
============
使用 ProcessPoolExecutor 多进程并行处理股票，
每个 worker 独立加载单个 CSV 文件，避免构建巨型 DataFrame。

预期加速比: 4-6x（取决于 CPU 核心数）
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Ensure package imports work in worker processes
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)


# ================================================================
# b2 worker
# ================================================================
def _worker_b2(args):
    """
    单只股票的 b2 筛选。
    参数: (code, target_date_str, data_dir)
    返回: (code, gain, j_prev, weight) 或 None
    """
    code, target_date_str, data_dir = args
    target_date = pd.Timestamp(target_date_str)

    try:
        csv_path = os.path.join(data_dir, f'{code}.csv')
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.set_index('date').sort_index()

        if target_date not in df.index or len(df) < 120:
            return None

        from quant_strategy.indicators import calc_all_indicators
        from quant_strategy.strategy_b2 import generate_b2_signals

        df = calc_all_indicators(df, board_type='main')
        df = generate_b2_signals(df, board_type='main', precomputed=True)

        target_loc = df.index.get_loc(target_date)
        last = df.iloc[target_loc]

        if last.get('b2_entry_signal', 0) == 1:
            prev = df.iloc[target_loc - 1]
            gain = (last['close'] / prev['close'] - 1) * 100
            j_prev = float(prev['J'])
            weight = float(last.get('b2_position_weight', 1.0))
            return (code, gain, j_prev, weight)
        return None

    except Exception:
        return None


# ================================================================
# brick worker
# ================================================================
def _worker_brick(args):
    """
    单只股票的砖型图筛选（含四定式分类）。
    参数: (code, target_date_str, data_dir)
    返回: (code, pattern) 或 None（pattern ∈ {定式一, 定式二, 定式三, 定式四}）
    """
    code, target_date_str, data_dir = args
    target_date = pd.Timestamp(target_date_str)

    try:
        csv_path = os.path.join(data_dir, f'{code}.csv')
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.set_index('date').sort_index()

        if target_date not in df.index or len(df) < 120:
            return None

        from quant_strategy.indicators import calc_brick_chart, calc_white_yellow_line
        from quant_strategy.strategy_brick import classify_brick_pattern

        df = calc_brick_chart(df)
        df = calc_white_yellow_line(df)

        target_loc = df.index.get_loc(target_date)
        last = df.iloc[target_loc]

        # 总原则：白线 > 黄线 + 绿转红
        if last['white_line'] <= last['yellow_line']:
            return None
        if last['brick_green_to_red'] != 1:
            return None

        pattern = classify_brick_pattern(df, target_loc)
        return (code, pattern)

    except Exception:
        return None


# ================================================================
# 并行调度器
# ================================================================
def run_screen_parallel(strategy, codes, target_date,
                        data_dir=None, max_workers=None):
    """
    并行运行选股策略。

    参数
    ----
    strategy    : 'b2' 或 'brick'
    codes       : 股票代码列表
    target_date : 目标日期 (pd.Timestamp 或 str 'YYYY-MM-DD')
    data_dir    : stock_kline 目录路径，默认 fetch_kline/stock_kline/
    max_workers : 最大进程数，默认 min(8, cpu_count)

    返回
    ----
    list of results (格式取决于策略类型)
    """
    if data_dir is None:
        data_dir = os.path.join(BASE, 'fetch_kline', 'stock_kline')

    target_date_str = (target_date.strftime('%Y-%m-%d')
                       if hasattr(target_date, 'strftime')
                       else target_date)

    if max_workers is None:
        max_workers = min(8, multiprocessing.cpu_count())

    worker = _worker_b2 if strategy == 'b2' else _worker_brick
    tasks = [(code, target_date_str, data_dir) for code in codes]

    results = []
    n_total = len(tasks)
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}

        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res is not None:
                results.append(res)

            # Progress every 200 stocks
            done = i + 1
            if done % 200 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                rem = (n_total - done) / rate if rate > 0 else 0
                print(f"  进度: {done}/{n_total}  "
                      f"耗时 {elapsed:.0f}s 已选出 {len(results)} 只  "
                      f"速率 {rate:.1f}只/s 剩余 ~{rem:.0f}s",
                      flush=True)

    elapsed_total = time.time() - t_start
    print(f"  完成: {n_total}只, 耗时 {elapsed_total:.0f}s "
          f"({elapsed_total/60:.1f}min), 速率 {n_total/elapsed_total:.1f}只/s, "
          f"选出 {len(results)} 只", flush=True)

    # Sort results
    if strategy == 'b2':
        results.sort(key=lambda x: x[3], reverse=True)  # by weight
    else:
        # Sort by pattern priority: 定式一 > 定式二 > 定式三 > 定式四
        pat_order = {'定式一': 0, '定式二': 1, '定式三': 2, '定式四': 3}
        results.sort(key=lambda x: pat_order.get(x[1], 99))

    return results


def get_code_list(data_dir=None):
    """获取所有可用的股票代码列表（从 stock_kline 目录的 CSV 文件名提取）"""
    if data_dir is None:
        data_dir = os.path.join(BASE, 'fetch_kline', 'stock_kline')
    return sorted([f.replace('.csv', '') for f in os.listdir(data_dir)
                   if f.endswith('.csv')])
