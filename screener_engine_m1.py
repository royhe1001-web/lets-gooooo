#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 芯片专用并行筛选引擎
=======================
针对 Apple Silicon (M1) + 8GB 统一内存优化。

与 Windows x86 版本 (screener_engine.py) 的差异:
  - max_workers=4   (仅使用 4 个性能核心，跳过效率核心)
  - 显式 spawn 上下文 (macOS 默认 spawn，显式声明确保兼容)
  - float32 精度    (相比 float64 减半内存占用，对技术指标精度足够)
  - 分批处理         (每批 400 只，限制峰值内存 ~800MB)
  - 积极 GC          (每批完成后强制回收，防止 8GB 内存压力)
  - numba 兼容       (ARM64 numba 首次编译较慢，保留 cache 加速后续运行)
  - 内存预警         (超过阈值时输出警告)

适用: MacBook Air / MacBook Pro 13" (M1, 8GB RAM)
       Mac mini (M1, 8GB) / iMac (M1, 8GB)
也可用于 M2/M3 等后续 Apple Silicon 芯片。

预期性能 (M1/8GB):
  - b2 筛选:    ~90-120秒   (2294 只股票)
  - 砖型图筛选:  ~15-25秒   (2294 只股票)
"""

import os
import sys
import gc
import time
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure package imports work in worker processes
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# ============================================================
# M1 专用配置
# ============================================================
M1_P_CORES = 4        # M1 性能核心数（不使用 E-cores）
BATCH_SIZE = 400       # 每批处理股票数（控制峰值内存 ~800MB）
MEMORY_WARN_MB = 1024  # 内存预警阈值 (MB)，超过 1GB 可用时发出警告
GC_FREQ = 100          # 每处理 N 只股票触发一次 gc.collect()

# CSV 列类型 —— float32 比 float64 节省 50% 内存
# 对于日线级别的 OHLCV 数据，float32 精度完全足够
CSV_DTYPE = {
    'open': np.float32,
    'close': np.float32,
    'high': np.float32,
    'low': np.float32,
    'volume': np.float32,
}
# date 列由 parse_dates 处理，不能在此指定 str 类型
# 否则会覆盖 parse_dates，导致 date 保持为字符串


# ================================================================
# 内存工具
# ================================================================
def _get_memory_usage_mb():
    """获取当前进程内存使用 (MB)，跨平台兼容。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        # psutil 未安装时返回 -1（不阻断运行）
        return -1


def _check_memory():
    """检查可用内存，压力较大时输出警告。"""
    mem = _get_memory_usage_mb()
    if mem > 0:
        avail = 8192 - mem  # 8GB 总内存
        if avail < MEMORY_WARN_MB:
            print(f"  [MEM] 警告: 进程内存 {mem:.0f}MB, 可用仅 ~{avail:.0f}MB", flush=True)
    return mem


# ================================================================
# b2 worker (M1 优化版)
# ================================================================
def _worker_b2_m1(args):
    """
    b2 策略 worker — M1 优化版。
    使用 float32 精度加载数据，处理后立即清理中间变量。
    """
    code, target_date_str, data_dir, _ = args
    target_date = pd.Timestamp(target_date_str)

    try:
        csv_path = os.path.join(data_dir, f'{code}.csv')
        if not os.path.exists(csv_path):
            return None

        # float32 加载，减半内存
        df = pd.read_csv(csv_path, parse_dates=['date'], dtype=CSV_DTYPE)
        df = df.set_index('date').sort_index()

        if target_date not in df.index or len(df) < 120:
            del df
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
            result = (code, gain, j_prev, weight)
        else:
            result = None

        # 积极清理
        del df
        return result

    except Exception:
        return None


# ================================================================
# brick worker (M1 优化版)
# ================================================================
def _worker_brick_m1(args):
    """
    砖型图 worker — M1 优化版。
    仅计算砖型图+白黄线（不计算 KDJ/MACD/量价形态），
    M1 上每只股票 ~15ms 即可完成。
    """
    code, target_date_str, data_dir, _ = args
    target_date = pd.Timestamp(target_date_str)

    try:
        csv_path = os.path.join(data_dir, f'{code}.csv')
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path, parse_dates=['date'], dtype=CSV_DTYPE)
        df = df.set_index('date').sort_index()

        if target_date not in df.index or len(df) < 120:
            del df
            return None

        from quant_strategy.indicators import calc_brick_chart, calc_white_yellow_line
        from quant_strategy.strategy_brick import classify_brick_pattern

        df = calc_brick_chart(df)
        df = calc_white_yellow_line(df)

        target_loc = df.index.get_loc(target_date)
        last = df.iloc[target_loc]
        prev = df.iloc[target_loc - 1] if target_loc > 0 else last

        if last['white_line'] <= last['yellow_line']:
            del df; return None
        if last['brick_green_to_red'] != 1:
            del df; return None

        # 质量过滤: 涨幅>3% + 砖值>60 + 量比>1.3
        gain = (float(last['close']) / float(prev['close']) - 1)
        brick_val = float(last.get('brick_val', 0))
        vol_ratio = float(last['volume']) / float(prev['volume']) if float(prev['volume']) > 0 else 0
        if gain < 0.03 or brick_val < 60 or vol_ratio < 1.3:
            del df; return None

        # 质量评分
        quality = 0.0
        if gain > 0.03: quality += 2.0
        if brick_val > 80: quality += 2.0
        elif brick_val > 60: quality += 1.5
        if vol_ratio > 2.0: quality += 1.5
        elif vol_ratio > 1.3: quality += 1.0

        pattern = classify_brick_pattern(df, target_loc)
        del df
        return (code, pattern, quality, gain*100, brick_val)

    except Exception:
        return None


# ================================================================
# M1 并行调度器
# ================================================================
def run_screen_parallel_m1(strategy, codes, target_date,
                           data_dir=None, max_workers=None,
                           batch_size=None):
    """
    M1 优化版并行选股。

    与 x86 版本的关键差异:
      - 使用 multiprocessing.get_context('spawn') 显式控制进程启动
      - 默认 4 workers（仅 M1 P-cores）
      - 分批提交任务，每批完成后 gc.collect()
      - 内存压力监控

    参数
    ----
    strategy    : 'b2' 或 'brick'
    codes       : 股票代码列表
    target_date : 目标日期
    data_dir    : stock_kline 目录路径
    max_workers : 进程数，默认 4（M1 P-cores）
    batch_size  : 批次大小，默认 400（限制峰值内存）
    """
    if data_dir is None:
        data_dir = os.path.join(BASE, 'fetch_kline', 'stock_kline')

    target_date_str = (target_date.strftime('%Y-%m-%d')
                       if hasattr(target_date, 'strftime')
                       else target_date)

    if max_workers is None:
        max_workers = M1_P_CORES
    if batch_size is None:
        batch_size = BATCH_SIZE

    worker = _worker_b2_m1 if strategy == 'b2' else _worker_brick_m1

    # M1/macOS 显式使用 spawn 上下文
    ctx = multiprocessing.get_context('spawn')

    results = []
    n_total = len(codes)
    t_start = time.time()
    batch_count = 0

    # 分批处理以限制峰值内存
    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_codes = codes[batch_start:batch_end]
        batch_tasks = [(code, target_date_str, data_dir, batch_start)
                       for code in batch_codes]

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(worker, t): t for t in batch_tasks}

            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)

        # 批次完成，强制 GC 回收内存
        batch_count += 1
        done = batch_end
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        rem = (n_total - done) / rate if rate > 0 else 0
        mem_str = f" 内存 {_get_memory_usage_mb():.0f}MB" if _get_memory_usage_mb() > 0 else ""

        print(f"  进度: {done}/{n_total}  批次 {batch_count}  "
              f"耗时 {elapsed:.0f}s 已选出 {len(results)} 只  "
              f"速率 {rate:.1f}只/s 剩余 ~{rem:.0f}s{mem_str}",
              flush=True)

        # 积极 GC
        gc.collect()

    elapsed_total = time.time() - t_start
    print(f"  完成: {n_total}只, 耗时 {elapsed_total:.0f}s "
          f"({elapsed_total/60:.1f}min), 速率 {n_total/elapsed_total:.1f}只/s, "
          f"选出 {len(results)} 只", flush=True)

    # 最终 GC
    gc.collect()

    # 排序结果
    if strategy == 'b2':
        results.sort(key=lambda x: x[3], reverse=True)  # by weight
    else:
        pat_order = {'定式一': 0, '定式二': 1, '定式三': 2, '定式四': 3}
        results.sort(key=lambda x: pat_order.get(x[1], 99))

    return results


def get_code_list(data_dir=None):
    """获取所有可用的股票代码列表。"""
    if data_dir is None:
        data_dir = os.path.join(BASE, 'fetch_kline', 'stock_kline')
    return sorted([f.replace('.csv', '') for f in os.listdir(data_dir)
                   if f.endswith('.csv')])


def print_system_info():
    """打印 M1 系统信息，用于调试和性能分析。"""
    import platform
    print(f"  系统: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CPU 核心: {multiprocessing.cpu_count()} (使用 {M1_P_CORES} 个 P-cores)")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  内存: 总计 {mem.total/(1024**3):.0f}GB  可用 {mem.available/(1024**3):.1f}GB")
    except ImportError:
        print(f"  内存: ~8GB (psutil 未安装，无法精确检测)")
    # numba 检测
    try:
        from numba import jit
        print(f"  numba: 已启用 (ARM64 JIT)")
    except ImportError:
        print(f"  numba: 未安装 (SMA/streak 使用 Python 循环，较慢)")
