#!/usr/bin/env python3
"""b2 + 砖型图 双策略并行回测"""
import argparse, os, sys, time, warnings
import pandas as pd, numpy as np

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'simulation'))

from run_spring_simulation_2026 import (SimulationEngine, Position,
                                     SUPER_WEIGHT_THRESHOLD, DATA_DIR)
from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_spring import generate_spring_signals
from quant_strategy.strategy_brick import generate_brick_signals

OUT = os.path.join(ROOT, 'output')


class DualEngine(SimulationEngine):
    def scan_signals(self, min_vol_ratio=1.5):
        print("扫描 b2 + 砖型图 双策略信号...", flush=True)
        all_signals = []
        codes = sorted([f.replace('.csv','') for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
        total = len(codes); t0 = time.time()

        for i, code in enumerate(codes):
            df = self.load_stock(code)
            if df is None or len(df) < 200: continue
            full_df = df.loc[:self.end_date + pd.Timedelta(days=5)]
            if len(full_df) < 200: continue
            try:
                full_df = calc_all_indicators(full_df, board_type='main')
                full_df = generate_spring_signals(full_df, board_type='main', precomputed=True)
                full_df = generate_brick_signals(full_df, board_type='main', precomputed=False)

                mask = (full_df.index >= self.start_date) & (full_df.index <= self.end_date)
                b2_sigs = full_df.index[mask & (full_df['b2_entry_signal'] == 1)]
                brick_sigs = full_df.index[mask & (full_df['brick_high_quality'] == 1)]

                all_dates = set(b2_sigs) | set(brick_sigs)
                for sd in sorted(all_dates):
                    row = full_df.loc[sd]
                    prev_close = float(full_df.shift(1).loc[sd, 'close']) if sd in full_df.shift(1).index else float(row['open'])
                    gain = (float(row['close']) / prev_close - 1) * 100
                    is_b2 = sd in b2_sigs
                    is_brick = sd in brick_sigs
                    both = is_b2 and is_brick

                    # Weight: b2 native weight or brick quality
                    if is_b2:
                        weight = float(row.get('b2_position_weight', 1.0))
                        source = 'b2'
                    else:
                        q = float(row.get('brick_quality', 1.0))
                        weight = 1.0 + q * 0.3
                        source = 'brick'

                    if both:
                        weight *= 1.5  # 共振加成
                        source = 'b2+brick'

                    # Volume filter
                    if min_vol_ratio > 1.0:
                        prev_vol = float(full_df.shift(1).loc[sd, 'volume']) if sd in full_df.shift(1).index else 0
                        if prev_vol > 0 and float(row['volume']) / prev_vol < min_vol_ratio:
                            continue

                    next_dates = full_df.index[full_df.index > sd]
                    if len(next_dates) == 0: continue
                    t1_date = next_dates[0]

                    pattern = str(row.get('brick_pattern', ''))
                    all_signals.append({
                        'code': code, 'signal_date': sd,
                        'signal_close': float(row['close']),
                        'weight': weight, 'source': source,
                        't1_date': t1_date,
                        't1_open': float(full_df.loc[t1_date, 'open']),
                        'gain': gain, 'pattern': pattern,
                        'is_both': both,
                    })
            except: continue
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{total} {time.time()-t0:.0f}s  signals={len(all_signals)}", flush=True)
        print(f"  完成: {total}只, {time.time()-t0:.0f}s, 共 {len(all_signals)} 个信号", flush=True)
        all_signals.sort(key=lambda x: x['signal_date'])
        return all_signals


def parse_args():
    p = argparse.ArgumentParser(description='b2+砖型图 双策略 T+1 模拟交易')
    p.add_argument('--capital', type=float, default=40000)
    p.add_argument('--max-positions', type=int, default=3)
    p.add_argument('--min-positions', type=int, default=2)
    p.add_argument('--start', default='2014-01-01')
    p.add_argument('--end', default='2026-05-07')
    p.add_argument('--min-vol-ratio', type=float, default=1.5)
    p.add_argument('--dynamic-sizing', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    global SUPER_WEIGHT_THRESHOLD
    SUPER_WEIGHT_THRESHOLD = 3.0

    engine = DualEngine(
        capital=args.capital, max_positions=args.max_positions,
        min_positions=args.min_positions, start_date=args.start,
        end_date=args.end, dynamic_sizing=args.dynamic_sizing)
    engine.run(min_vol_ratio=args.min_vol_ratio)


if __name__ == '__main__':
    main()
