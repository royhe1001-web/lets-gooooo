#!/usr/bin/env python3
"""
Phase 2c 扩展版: 主板-only + 市值过滤 + 2024-2025Q3训练窗口
========================================================
两阶段设计:
  Stage 1: 加权 T+5 收益快速筛选 (ProcessPoolExecutor, 股票分片, 无重复加载)
  Stage 2: 完整仿真引擎验证 Top-5 (2024-2025验证 → 2026测试)

仅主板股票 + 动态市值过滤 500-1000亿
"""

import os, sys, time, json, copy, tempfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from oamv import fetch_market_data, calc_oamv, generate_signals
from strategy_b2 import generate_b2_signals
from ML_optimization.mktcap_utils import build_mktcap_lookup, is_in_mktcap_range

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

TRAIN_START = pd.Timestamp('2024-01-01')
TRAIN_END   = pd.Timestamp('2025-09-30')
VAL_START   = pd.Timestamp('2025-10-01')
VAL_END     = pd.Timestamp('2025-12-31')
TEST_START  = pd.Timestamp('2026-01-01')
TEST_END    = pd.Timestamp('2026-05-07')

MktCap_MIN = 500   # 亿
MktCap_MAX = 1000  # 亿

MAIN_BOARD_PREFIXES = ('000', '002', '003', '600', '601', '603', '605')

# ============================================================
# Parameter Sets (same 17 as original Phase 2c)
# ============================================================
EXPLOSIVE_B2 = {
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
}
DEFAULT_B2 = {**EXPLOSIVE_B2, 'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
              'gain_min': 0.04, 'j_today_max': 65}
PULLBACK_B2 = {**EXPLOSIVE_B2, 'weight_gain_2x_thresh': 0.07, 'weight_gain_2x': 2.0,
               'weight_pullback_thresh': -0.07, 'weight_pullback': 1.5}

DEFAULT_OAMV = {
    'oamv_aggressive_threshold': 3.0, 'oamv_defensive_threshold': -2.35,
    'oamv_aggressive_buffer': 0.95, 'oamv_defensive_buffer': 0.99,
    'oamv_defensive_ban_entry': True,
}


def build_param_sets():
    sets = []
    sets.append({'name': 'DEFAULT+OAMV_default', 'b2': copy.deepcopy(DEFAULT_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV)})
    sets.append({'name': 'DEFAULT_no_OAMV', 'b2': copy.deepcopy(DEFAULT_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV), 'no_oamv': True})
    for agg_t, s in [(2.5, 'agg2.5'), (3.0, 'agg3.0'), (3.5, 'agg3.5')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_aggressive_threshold'] = agg_t
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    for def_t, s in [(-2.0, 'def2.0'), (-2.35, 'def2.35'), (-2.5, 'def2.5')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_defensive_threshold'] = def_t
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    for buf, s in [(0.93, 'abuf93'), (0.97, 'abuf97')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_aggressive_buffer'] = buf
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    for buf, s in [(0.97, 'dbuf97'), (1.0, 'dbuf100')]:
        o = copy.deepcopy(DEFAULT_OAMV); o['oamv_defensive_buffer'] = buf
        sets.append({'name': f'Explosive_{s}', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    o = copy.deepcopy(DEFAULT_OAMV); o['oamv_defensive_ban_entry'] = False
    sets.append({'name': 'Explosive_noDefBan', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    sets.append({'name': 'Explosive_no_OAMV', 'b2': copy.deepcopy(EXPLOSIVE_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV), 'no_oamv': True})
    sets.append({'name': 'Pullback+OAMV', 'b2': copy.deepcopy(PULLBACK_B2),
                 'oamv': copy.deepcopy(DEFAULT_OAMV)})
    o = copy.deepcopy(DEFAULT_OAMV)
    o.update({'oamv_aggressive_threshold': 2.5, 'oamv_aggressive_buffer': 0.93,
              'oamv_defensive_buffer': 0.97, 'oamv_defensive_ban_entry': False})
    sets.append({'name': 'OAMV_Aggressive', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    o = copy.deepcopy(DEFAULT_OAMV)
    o.update({'oamv_aggressive_threshold': 3.5, 'oamv_aggressive_buffer': 0.97,
              'oamv_defensive_threshold': -2.0, 'oamv_defensive_buffer': 1.0,
              'oamv_defensive_ban_entry': True})
    sets.append({'name': 'OAMV_Conservative', 'b2': copy.deepcopy(EXPLOSIVE_B2), 'oamv': o})
    return sets


# ============================================================
# Stage 1: 股票分片并行评估 (每个worker只加载自己的股票)
# ============================================================
NEEDED_COLS = ['date', 'open', 'high', 'low', 'close', 'volume',
               'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
               'white_line', 'yellow_line',
               'brick_val', 'brick_rising', 'brick_falling',
               'brick_green_to_red', 'brick_red_to_green',
               'brick_green_streak', 'brick_red_streak',
               'cross_below_yellow']


def _eval_stock_chunk(args):
    """Worker: load assigned stock files, evaluate ALL param sets on them.
    Each stock file is loaded by exactly one worker — no I/O duplication.
    Includes dynamic 500-1000亿 market cap filtering at signal level."""
    stock_files, oamv_change_map, param_sets, sim_start, sim_end, worker_id = args

    # Build market cap lookup (fast pickle load)
    try:
        mktcap_lookup = build_mktcap_lookup()
    except Exception:
        mktcap_lookup = {}

    # Load assigned stocks only
    stock_data = {}
    for f in stock_files:
        code = os.path.basename(f).replace('.parquet', '')
        try:
            df = pd.read_parquet(f)
            cols_avail = [c for c in NEEDED_COLS if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            oamv_changes = [oamv_change_map.get(d, 0.0) for d in df['date']]
            df['_oamv_change_pct'] = oamv_changes
            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
        except Exception:
            continue

    # Evaluate all param sets on this chunk of stocks
    results = []
    for ps in param_sets:
        agg_thresh = ps['oamv'].get('oamv_aggressive_threshold', 3.0)
        def_thresh = ps['oamv'].get('oamv_defensive_threshold', -2.35)
        no_oamv = ps.get('no_oamv', False)
        if no_oamv:
            agg_thresh, def_thresh = 999.0, -999.0

        b2p = copy.deepcopy(ps['b2'])

        total_w = 0.0; total_wr = 0.0; n = 0
        for code, df in stock_data.items():
            try:
                dc = df.copy()
                chg = dc['_oamv_change_pct'].values
                regime = np.where(chg > agg_thresh, 'aggressive',
                         np.where(chg < def_thresh, 'defensive', 'normal'))
                dc['oamv_regime'] = regime
                sdf = generate_b2_signals(dc, board_type='main', precomputed=True, params=b2p)
                mask = (sdf.index >= sim_start) & (sdf.index <= sim_end)
                sig_idx = sdf.index[mask & (sdf['b2_entry_signal'] == 1)]
                for i in sig_idx:
                    pos = sdf.index.get_loc(i)
                    if pos + 5 < len(sdf):
                        # Dynamic market cap filter (500-1000亿) at signal date
                        if mktcap_lookup:
                            if not is_in_mktcap_range(code, i, mktcap_lookup, MktCap_MIN, MktCap_MAX):
                                continue
                        buy_px = float(sdf.loc[i, 'close'])
                        sell_px = float(sdf.iloc[pos + 5]['open'])
                        ret = sell_px / buy_px - 1
                        w = float(sdf.loc[i, 'b2_position_weight'])
                        total_wr += ret * w; total_w += w; n += 1
            except Exception:
                continue

        z = float(total_wr / total_w) if total_w > 0 else 0.0
        results.append({'name': ps['name'], 'z': z, 'n_signals': n,
                        'total_w': total_w, 'total_wr': total_wr})

    return results


def stage1_fast_screen(stock_files, oamv_df, param_sets, sim_start, sim_end,
                       n_workers=4):
    """Split STOCKS (not params) across workers. Each stock loaded once."""
    # Build OAMV change map
    oamv_change_map = {}
    for _, row in oamv_df.iterrows():
        d = row['date']
        if hasattr(d, 'date'): d = d.date()
        oamv_change_map[pd.Timestamp(d)] = float(row['oamv_change_pct'])

    # Split stock files across workers
    chunk_size = max(1, len(stock_files) // n_workers)
    chunks = []
    for i in range(0, len(stock_files), chunk_size):
        chunks.append(stock_files[i:i+chunk_size])

    print(f"  Stocks: {len(stock_files)} main board, {n_workers} workers "
          f"({len(chunks)} chunks, ~{chunk_size} stocks each)")
    print(f"  Period: {sim_start.date()} -> {sim_end.date()}")
    print(f"  Param sets: {len(param_sets)}")
    print(f"  MktCap filter: {MktCap_MIN}-{MktCap_MAX}亿 (dynamic, per-signal)")
    print(f"  Each worker loads its own stock subset → no I/O duplication", flush=True)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        args_list = [(chunk, oamv_change_map, param_sets, sim_start, sim_end, i)
                     for i, chunk in enumerate(chunks)]
        futures = {executor.submit(_eval_stock_chunk, args): i
                   for i, args in enumerate(args_list)}

        # Collect per-worker results
        worker_results = []
        for f in as_completed(futures):
            worker_id = futures[f]
            try:
                wr = f.result()
                worker_results.append(wr)
                elapsed = time.time() - t0
                print(f"  Worker {worker_id+1}/{len(chunks)} done ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                print(f"  Worker {worker_id+1} ERROR: {e}", flush=True)

    # Aggregate: sum weighted returns across all workers
    merged = {}
    for wr in worker_results:
        for r in wr:
            name = r['name']
            if name not in merged:
                merged[name] = {'total_w': 0.0, 'total_wr': 0.0, 'n_signals': 0}
            merged[name]['total_w'] += r['total_w']
            merged[name]['total_wr'] += r['total_wr']
            merged[name]['n_signals'] += r['n_signals']

    # Compute final z-score
    stage1_results = []
    for name, m in merged.items():
        z = float(m['total_wr'] / m['total_w']) if m['total_w'] > 0 else 0.0
        stage1_results.append({'name': name, 'z': z, 'n_signals': m['n_signals']})

    stage1_results.sort(key=lambda r: r['z'], reverse=True)
    return stage1_results


# ============================================================
# Stage 2: Full simulation validation
# ============================================================
def _run_full_sim_subprocess(ps_json, stock_files, oamv_df_path, sim_start_str, sim_end_str):
    """Run full simulation in a subprocess."""
    import sys, os, copy
    sys.path.insert(0, BASE)

    ps = json.loads(ps_json)
    sim_start = pd.Timestamp(sim_start_str)
    sim_end = pd.Timestamp(sim_end_str)

    import importlib.util
    p2c_path = os.path.join(BASE, 'ML_optimization', 'phase2c_oamv_grid_search.py')
    spec = importlib.util.spec_from_file_location('p2c_engine', p2c_path)
    p2c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2c)
    # Override module-level SIM_START/SIM_END for correct date range
    p2c.SIM_START = sim_start
    p2c.SIM_END = sim_end
    OAMVSimEngine = p2c.OAMVSimEngine

    oamv_df = pd.read_pickle(oamv_df_path)

    # Build set of codes ever in 500-1000亿 market cap range during sim period
    from ML_optimization.mktcap_utils import build_mktcap_lookup, get_mktcap_universe
    mktcap_lookup = build_mktcap_lookup()

    ym_start = sim_start.strftime('%Y-%m')
    ym_end = sim_end.strftime('%Y-%m')
    eligible_codes = get_mktcap_universe(mktcap_lookup, ym_start, 500, 1000, ym_end=ym_end)

    needed_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                   'white_line', 'yellow_line', 'white_above_yellow', 'white_direction',
                   'brick_val', 'brick_var6a', 'brick_rising', 'brick_falling',
                   'brick_green_to_red', 'brick_red_to_green',
                   'brick_green_streak', 'brick_red_streak', 'cross_below_yellow']

    stock_data = {}
    for f in stock_files:
        code = os.path.basename(f).replace('.parquet', '')
        # Pre-filter: quick set lookup
        if eligible_codes and code not in eligible_codes:
            continue
        try:
            df = pd.read_parquet(f)
            cols_avail = [c for c in needed_cols if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
        except:
            continue

    b2_params = ps['b2']
    oamv_params = ps['oamv']
    no_oamv = ps.get('no_oamv', False)

    if no_oamv:
        oamv_params_use = copy.deepcopy(oamv_params)
        oamv_params_use['oamv_aggressive_threshold'] = 999.0
        oamv_params_use['oamv_defensive_threshold'] = -999.0
    else:
        oamv_params_use = oamv_params

    engine = OAMVSimEngine(stock_data, b2_params, oamv_df, oamv_params_use)
    metrics = engine.run()
    return json.dumps(metrics)


def stage2_full_sim(param_sets, stock_files, oamv_df, sim_start, sim_end):
    """Run full simulation sequentially for selected param sets."""
    print(f"\n  Full Simulation: {len(param_sets)} sets")
    print(f"  Period: {sim_start.date()} -> {sim_end.date()}")

    oamv_path = os.path.join(tempfile.gettempdir(), 'phase2c_oamv.pkl')
    oamv_df.to_pickle(oamv_path)

    results = []
    for i, ps in enumerate(param_sets):
        t1 = time.time()
        ps_json = json.dumps({'name': ps['name'], 'b2': ps['b2'], 'oamv': ps['oamv'],
                              'no_oamv': ps.get('no_oamv', False)})
        result_json = _run_full_sim_subprocess(
            ps_json, stock_files, oamv_path,
            str(sim_start.date()), str(sim_end.date()))
        metrics = json.loads(result_json)
        metrics['name'] = ps['name']
        results.append(metrics)
        elapsed = time.time() - t1
        print(f"  [{i+1}/{len(param_sets)}] {ps['name']:<25s} "
              f"Ret={metrics.get('total_return_pct', 0):+.1f}% "
              f"Sharpe={metrics.get('sharpe', 0):.3f} "
              f"WR={metrics.get('win_rate', 0):.1%} "
              f"Trades={metrics.get('n_trades', 0)} ({elapsed:.0f}s)", flush=True)

    os.remove(oamv_path)
    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Phase 2c Expanded: Main Board Grid Search")
    print(f"  Train: {TRAIN_START.date()} -> {TRAIN_END.date()}")
    print(f"  Val:   {VAL_START.date()} -> {VAL_END.date()}")
    print(f"  Test:  {TEST_START.date()} -> {TEST_END.date()}")
    print("=" * 70)

    # Preload OAMV
    print("\n  Loading OAMV...", flush=True)
    oamv_df = fetch_market_data(start='20100101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = generate_signals(oamv_df)
    print(f"  OAMV: {len(oamv_df)} days")

    # Get main board stock files
    all_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    main_files = [os.path.join(FEAT_DIR, f) for f in all_files
                  if any(f.startswith(p) for p in MAIN_BOARD_PREFIXES)]
    print(f"  Main board stocks: {len(main_files)} / {len(all_files)}")

    param_sets = build_param_sets()
    print(f"  Parameter sets: {len(param_sets)}")

    # Stage 1: Fast screening on train data
    print(f"\n{'='*70}")
    print("  STAGE 1: Fast Screening (weighted T+5) on TRAIN")
    print("=" * 70)
    stage1_results = stage1_fast_screen(
        main_files, oamv_df, param_sets, TRAIN_START, TRAIN_END, n_workers=4)

    print(f"\n  Stage 1 Ranking (Train {TRAIN_START.date()}~{TRAIN_END.date()}):")
    for i, r in enumerate(stage1_results):
        print(f"  {i+1:>2}. {r['name']:<25s} z={r['z']*100:+.3f}% n={r['n_signals']:>5d}")

    # Stage 2: Validate top-5 on 2025 with full simulation
    print(f"\n{'='*70}")
    print("  STAGE 2: Full Simulation on VAL (2025) — Top-5")
    print("=" * 70)
    top5_train = [{'name': r['name'], 'b2': next(ps['b2'] for ps in param_sets if ps['name'] == r['name']),
                   'oamv': next(ps['oamv'] for ps in param_sets if ps['name'] == r['name']),
                   'no_oamv': next((ps.get('no_oamv', False) for ps in param_sets if ps['name'] == r['name']), False)}
                  for r in stage1_results[:5]]
    val_results = stage2_full_sim(top5_train, main_files, oamv_df,
                                  VAL_START, VAL_END)

    val_results.sort(key=lambda r: r.get('sharpe', 0), reverse=True)

    # Stage 3: Test best on 2026
    print(f"\n{'='*70}")
    print("  STAGE 3: Full Simulation on TEST (2026) — Best")
    print("=" * 70)
    best_val_name = val_results[0]['name']
    best_ps = next(ps for ps in param_sets if ps['name'] == best_val_name)
    test_results = stage2_full_sim([best_ps], main_files, oamv_df,
                                   TEST_START, TEST_END)

    # Final Report
    print(f"\n{'='*70}")
    print(f"  Phase 2c Main Board — Final Results")
    print(f"{'='*70}")

    print(f"\n  {'Rank':<6} {'Name':<25} {'Train z':>9} {'Val Ret':>9} {'Val Sharpe':>10} "
          f"{'Val WR':>8} {'Test Ret':>9} {'Test Sharpe':>10} {'Test WR':>8}")
    print(f"  {'-'*85}")

    for i, vr in enumerate(val_results[:5]):
        tr = next((r for r in stage1_results if r['name'] == vr['name']), {})
        is_best = vr['name'] == best_val_name
        te = test_results[0] if is_best else {}
        tag = ' BEST' if is_best else ''
        print(f"  {i+1:<6} {vr['name']:<25} "
              f"{tr.get('z', 0)*100:>+8.3f}% "
              f"{vr.get('total_return_pct', 0):>+8.1f}% "
              f"{vr.get('sharpe', 0):>10.3f} "
              f"{vr.get('win_rate', 0):>8.1%} "
              f"{te.get('total_return_pct', 0):>+8.1f}% "
              f"{te.get('sharpe', 0):>10.3f} "
              f"{te.get('win_rate', 0):>8.1%}{tag}")

    if test_results:
        te = test_results[0]
        print(f"\n  Best Main Board Params: {best_val_name}")
        print(f"    Test 2026: Ret={te.get('total_return_pct', 0):+.1f}% "
              f"Sharpe={te.get('sharpe', 0):.3f} "
              f"WR={te.get('win_rate', 0):.1%} "
              f"DD={te.get('max_dd', 0):.1%} "
              f"Trades={te.get('n_trades', 0)}")

    # Compare with Phase 5
    if test_results:
        print(f"\n  === Comparison with Phase 5 Dynamic Thresholds ===")
        print(f"  Phase 5 Dynamic (all-market): Ret=+21.6%, Sharpe=0.558")
        print(f"  Phase 2c Main Board Best:      Ret={te.get('total_return_pct', 0):+.1f}%, "
              f"Sharpe={te.get('sharpe', 0):.3f}")

    # Save
    out = {
        'phase': '2c_expanded_mainboard',
        'board': 'main_only',
        'n_stocks': len(main_files),
        'date_splits': {
            'train': f'{TRAIN_START.date()}~{TRAIN_END.date()}',
            'val': f'{VAL_START.date()}~{VAL_END.date()}',
            'test': f'{TEST_START.date()}~{TEST_END.date()}',
        },
        'stage1_train_results': [{
            'rank': i+1, 'name': r['name'], 'z': round(r['z'], 6),
            'n_signals': r['n_signals'],
        } for i, r in enumerate(stage1_results)],
        'stage2_val_results': [{
            'name': r['name'], 'total_return_pct': round(r.get('total_return_pct', 0), 2),
            'sharpe': round(r.get('sharpe', 0), 4),
            'win_rate': round(r.get('win_rate', 0), 4),
            'max_dd': round(r.get('max_dd', 0), 4),
            'n_trades': r.get('n_trades', 0),
        } for r in val_results],
        'stage3_test_results': [{
            'name': r['name'], 'total_return_pct': round(r.get('total_return_pct', 0), 2),
            'sharpe': round(r.get('sharpe', 0), 4),
            'win_rate': round(r.get('win_rate', 0), 4),
            'max_dd': round(r.get('max_dd', 0), 4),
            'n_trades': r.get('n_trades', 0),
        } for r in test_results],
        'best_mainboard_params': {
            'name': best_val_name,
            'b2': next(ps['b2'] for ps in param_sets if ps['name'] == best_val_name),
            'oamv': next(ps['oamv'] for ps in param_sets if ps['name'] == best_val_name),
        },
        'elapsed_seconds': time.time() - t0,
    }

    out_path = os.path.join(OUT_DIR, 'phase2c_mainboard_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Phase 2c Main Board Complete ({time.time()-t0:.0f}s)")
    print(f"  Results: {out_path}")


if __name__ == '__main__':
    main()
