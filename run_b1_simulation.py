#!/usr/bin/env python3
"""b1 策略回测 — 复用 b2 模拟引擎"""
import argparse, os, sys, time, warnings
import pandas as pd, numpy as np
from dataclasses import dataclass

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'simulation'))

from run_b2_simulation_2026 import SimulationEngine, SUPER_WEIGHT_THRESHOLD
from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_b1 import generate_b1_signals

DATA_DIR = os.path.join(ROOT, 'fetch_kline', 'stock_kline')
OUT = os.path.join(ROOT, 'output')


class B1SimEngine(SimulationEngine):
    def scan_signals(self, min_vol_ratio=1.0):
        print("扫描 b1 入场信号...", flush=True)
        all_signals = []
        codes = sorted([f.replace('.csv','') for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
        total = len(codes)
        t0 = time.time()

        for i, code in enumerate(codes):
            df = self.load_stock(code)
            if df is None or len(df) < 120:
                continue
            full_df = df.loc[:self.end_date + pd.Timedelta(days=5)]
            if len(full_df) < 120:
                continue
            try:
                full_df = calc_all_indicators(full_df, board_type='main')
                full_df = generate_b1_signals(full_df, board_type='main', precomputed=True)
                mask = (full_df.index >= self.start_date) & (full_df.index <= self.end_date)
                signal_dates = full_df.index[mask & (full_df['b1_entry_signal'] == 1)]

                for sd in signal_dates:
                    row = full_df.loc[sd]
                    sig_close = float(row['close'])
                    sig_open = float(row['open'])
                    sig_high = float(row['high'])
                    sig_low = float(row['low'])
                    sig_vol = float(row['volume'])
                    prev_close = float(full_df.shift(1).loc[sd, 'close']) if sd in full_df.shift(1).index else sig_open
                    j_val = float(row['J'])

                    # === b1 案例优化三大因子 ===

                    # 因子1: 缩量确认 — 信号日量 ≤ 前60日最大量 × 0.25
                    loc = full_df.index.get_loc(sd)
                    pre_60 = full_df.iloc[max(0, loc-60):loc]
                    if len(pre_60) < 20:
                        continue
                    max_vol_60 = pre_60['volume'].max()
                    if max_vol_60 <= 0 or sig_vol > max_vol_60 * 0.25:
                        continue  # 缩量不达标

                    # 因子2: K线形态 — 小实体低振幅
                    amp = (sig_high - sig_low) / prev_close * 100
                    code_start = code[:3] if len(code) >= 3 else code
                    if code_start in ('600', '601', '603', '605', '000', '001', '002', '003'):
                        if amp > 4.0: continue   # 主板振幅≤4%
                    else:
                        if amp > 7.0: continue   # 创业/科创≤7%

                    # 因子3: 白>黄 (防御性 — 避免弱势股)
                    white = float(row.get('white_line', 0))
                    yellow = float(row.get('yellow_line', 0))
                    if white <= yellow:
                        continue

                    # 权重计算 (缩量越深信号越强)
                    shrink_ratio = sig_vol / max_vol_60
                    score = float(row.get('b1_entry_score', 0.5))
                    weight = 1.0 + score * 3.0
                    if shrink_ratio < 0.15:
                        weight *= 1.3  # 深度缩量加分

                    next_dates = full_df.index[full_df.index > sd]
                    if len(next_dates) == 0: continue
                    t1_date = next_dates[0]
                    t1_open = float(full_df.loc[t1_date, 'open'])

                    all_signals.append({
                        'code': code, 'signal_date': sd, 'signal_close': sig_close,
                        'weight': weight, 't1_date': t1_date, 't1_open': t1_open,
                        'J': j_val, 'shrink': shrink_ratio, 'amp': amp,
                    })
            except:
                continue
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{total} {time.time()-t0:.0f}s  signals={len(all_signals)}", flush=True)
        print(f"  完成: {total}只, {time.time()-t0:.0f}s, 共 {len(all_signals)} 个信号", flush=True)
        all_signals.sort(key=lambda x: x['signal_date'])
        return all_signals


def parse_args():
    p = argparse.ArgumentParser(description='b1策略 T+1 模拟交易系统')
    p.add_argument('--capital', type=float, default=40000, help='初始资金')
    p.add_argument('--max-positions', type=int, default=3, help='最大持仓数')
    p.add_argument('--min-positions', type=int, default=2, help='最小持仓数')
    p.add_argument('--start', default='2024-01-01', help='开始日期')
    p.add_argument('--end', default='2026-05-07', help='结束日期')
    p.add_argument('--min-vol-ratio', type=float, default=1.0, help='信号日量比底线')
    p.add_argument('--super-threshold', type=float, default=4.0, help='特强信号权重阈值')
    return p.parse_args()


def main():
    args = parse_args()
    global SUPER_WEIGHT_THRESHOLD
    SUPER_WEIGHT_THRESHOLD = args.super_threshold

    engine = B1SimEngine(
        capital=args.capital, max_positions=args.max_positions,
        min_positions=args.min_positions, start_date=args.start, end_date=args.end)
    engine.run(min_vol_ratio=args.min_vol_ratio)


if __name__ == '__main__':
    main()
