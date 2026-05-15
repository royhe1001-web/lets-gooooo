#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring B2 策略回测入口
======================
配置: Explosive_def2.0 + P15仓位优化 + OAMVSimEngine (P0-P4)

P15优化要点:
  - 利润再投资(基于总权益分配)
  - 牛市集中(2-4只,单票≤50%总权益)
  - 防御允许空仓(min_positions=0)
  - 整手取整+min 8000过滤
  - 买入上限基于总权益而非剩余现金

用法:
  python run_backtest_2026.py                      # 默认: 本年初至今
  python run_backtest_2026.py --start 2025-07-01   # 指定起始
  python run_backtest_2026.py --start 2025-07-01 --end 2025-12-31
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

from ML_optimization.phase2c_oamv_grid_search import OAMVSimEngine
import ML_optimization.phase2c_oamv_grid_search as p2c_mod
from ML_optimization.phase2c_bull_grid_search import _get_stock_files_mainboard, preload_stock_data
from ML_optimization.mktcap_utils import (
    build_mktcap_lookup, compute_mktcap_percentiles, get_pct_universe,
)
from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals

BASE = os.path.dirname(os.path.abspath(__file__))

# 全管线优化参数 (Phase 1→5 on clean data, 2026-05-15)
# Phase 2c最优: Explosive_def2.0 (val Sharpe=2.295)
# Phase 5叠加: 动态分regime阈值 (Sharpe +0.298, DD -15.5pp)
# Phase 3部分: oamv_defensive_threshold=-3.5 (深防御阈值, 牛市少误杀)
# 其他参数保持 P15/OAMV_Aggressive 基准已验证有效
BEST_B2 = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.09, 'weight_gain_2x': 2.5,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'sector_momentum_enabled': False,
    # Phase 5 dynamic thresholds disabled — tuned on different setup, needs recalibration
}

BEST_OAMV = {
    'oamv_aggressive_threshold': 2.5, 'oamv_defensive_threshold': -3.5,
    'oamv_aggressive_buffer': 0.93, 'oamv_defensive_buffer': 0.97,
    'oamv_normal_buffer': 0.97, 'oamv_grace_days': 2,
    'oamv_defensive_ban_entry': False,
}


def load_all(test_start, test_end):
    print('=' * 70)
    print(f'  Spring B2 回测 — P15 + def2.0')
    print(f'  区间: {test_start.date()} ~ {test_end.date()}')
    print('=' * 70)

    for cache_path in [os.path.join(BASE, 'output', 'mktcap_lookup.pkl'),
                       os.path.join(BASE, 'output', 'mktcap_percentiles.pkl')]:
        if os.path.exists(cache_path):
            os.remove(cache_path)

    p2c_mod.SIM_START = test_start
    p2c_mod.SIM_END = test_end

    t0 = time.time()
    mktcap_lookup = build_mktcap_lookup()
    percentiles = compute_mktcap_percentiles()
    print(f'  市值: {len(mktcap_lookup)} entries, {len(percentiles)} months ({time.time()-t0:.0f}s)')

    ym_start = test_start.strftime('%Y-%m')
    ym_end = test_end.strftime('%Y-%m')
    eligible_codes = get_pct_universe(mktcap_lookup, percentiles, ym_start,
                                      ym_end=ym_end, regime='aggressive')
    print(f'  筛选股票: {len(eligible_codes)} 只')

    t0 = time.time()
    oamv_df = fetch_market_data(start='20180101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = generate_signals(oamv_df)
    print(f'  OAMV: {len(oamv_df)} 天 ({time.time()-t0:.0f}s)')

    t0 = time.time()
    stock_data = preload_stock_data(eligible_codes)
    print(f'  K线: {len(stock_data)} 只 ({time.time()-t0:.0f}s)')

    return stock_data, oamv_df, mktcap_lookup, percentiles


def main():
    parser = argparse.ArgumentParser(description='Spring B2 策略回测')
    parser.add_argument('--start', type=str, default=None, help='起始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='结束日期 YYYY-MM-DD')
    args = parser.parse_args()

    # Default: current year start to latest available data
    today = pd.Timestamp.now()
    test_start = pd.Timestamp(args.start) if args.start else pd.Timestamp(f'{today.year}-01-01')
    test_end = pd.Timestamp(args.end) if args.end else today

    stock_data, oamv_df, mktcap_lookup, percentiles = load_all(test_start, test_end)

    # Auto-adjust end to last available OAMV date if beyond
    last_oamv = pd.Timestamp(oamv_df['date'].max())
    if test_end > last_oamv:
        print(f'  注意: 指定结束日期 {test_end.date()} 超出数据范围, 调整为 {last_oamv.date()}')
        test_end = last_oamv
        p2c_mod.SIM_END = test_end

    print(f'\n{"="*70}')
    print(f'  配置: Explosive_def2.0 + P15仓位优化')
    print(f'{"="*70}')

    t0 = time.time()
    engine = OAMVSimEngine(stock_data, BEST_B2, oamv_df, BEST_OAMV,
                           mktcap_lookup=mktcap_lookup,
                           percentiles=percentiles)
    m = engine.run()
    elapsed = time.time() - t0

    print(f'\n  Return={m["total_return_pct"]:+.1f}%  '
          f'Sharpe={m["sharpe"]:.3f}  WR={m["win_rate"]:.1%}  '
          f'Trades={m["n_trades"]}  MaxDD={m["max_dd"]:.1%}  '
          f'({elapsed:.0f}s)')

    sc = m.get('stop_counts', {})
    if sc:
        print(f'  退出: ' + '  '.join(f'{k}={v}' for k, v in sorted(sc.items(), key=lambda x: -x[1])[:6]))

    # 导出 trade_log + daily_values 供实时信号和Excel
    import os as _os
    report_dir = _os.path.join(BASE, 'output', 'backtest')
    _os.makedirs(report_dir, exist_ok=True)
    tl = engine.export_trade_log()
    # 导出每日净值(回测引擎计算的准确值)
    if engine.daily_values:
        import pandas as _pd
        dv = _pd.DataFrame(engine.daily_values)
        dv.to_csv(_os.path.join(report_dir, 'daily_values.csv'), index=False)
    tl.to_csv(_os.path.join(report_dir, 'trade_log.csv'), index=False)
    print(f'  trade_log: {report_dir}/trade_log.csv ({len(tl)} 条)')

    # 生成 Excel 报告
    try:
        import subprocess, sys
        subprocess.run([sys.executable, _os.path.join(BASE, 'generate_report_xlsx.py')],
                       cwd=BASE, check=True, capture_output=True)
    except Exception as e:
        print(f'  Excel: {e}')

    print()


if __name__ == '__main__':
    main()
