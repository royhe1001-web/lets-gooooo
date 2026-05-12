#!/usr/bin/env python3
"""
Phase 2: B2 + Brick Parameter Grid Search (Memory-Optimized)
=============================================================
Pre-loads all stock data into memory once, then evaluates parameter
combinations on in-memory data. Includes both B2 and Brick parameters.

Two-stage:
  1. Signal-level screening: 300 random param combos on 1000 stocks (fast)
  2. Portfolio simulation: Top-10 on 3 rolling windows, all stocks

B2 Parameters:
  j_prev_max, gain_min, j_today_max, shadow_max,
  j_today_loose, shadow_loose, prior_strong_ret

Brick Parameters:
  green_streak_min, brick_val_quality_high, brick_val_quality_mid,
  brick_resonance_weight
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

N_RANDOM_TRIALS = 300
N_TOP_FOR_SIM = 10
SAMPLE_STAGE1 = 1000  # stocks for signal screening

# ============================================================
# Parameter Space (B2 + Brick)
# ============================================================
PARAM_GRID = {
    # B2 entry thresholds
    'j_prev_max':        [15, 18, 20, 22, 25],
    'gain_min':          [0.03, 0.04, 0.05, 0.06],
    'j_today_max':       [55, 60, 65, 70],
    'shadow_max':        [0.025, 0.030, 0.035, 0.040],
    'j_today_loose':     [65, 70, 75],
    'shadow_loose':      [0.035, 0.040, 0.045, 0.050],
    'prior_strong_ret':  [0.15, 0.20, 0.25],
    # Brick parameters
    'green_streak_min':  [1, 2, 3, 4],
    'brick_val_high':    [60, 70, 80],
    'brick_val_mid':     [40, 50, 60],
    'brick_resonance_w': [1.0, 1.2, 1.5, 2.0],
}

DEFAULT_PARAMS = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'green_streak_min': 2, 'brick_val_high': 80, 'brick_val_mid': 40,
    'brick_resonance_w': 1.5,
}

NEEDED_COLS = ['date', 'close', 'high', 'low', 'volume', 'open',
               'J', 'white_line', 'yellow_line',
               'brick_val', 'brick_green_to_red', 'brick_green_streak',
               'brick_rising', 'fwd_ret_5d']


def preload_data(paths, sample_n=None):
    """Load all stock data into memory. Returns list of (code, DataFrame)."""
    if sample_n:
        paths = paths[:sample_n]
    data = []
    for fp in paths:
        try:
            df = pd.read_parquet(fp, columns=[c for c in NEEDED_COLS if c != 'date'])
            # date is always available
            df_full = pd.read_parquet(fp, columns=['date'])
            df['date'] = pd.to_datetime(df_full['date'])
            code = os.path.basename(fp).replace('.parquet', '')
            # Filter to have minimum data
            if len(df) >= 120:
                data.append((code, df))
        except Exception:
            continue
    return data


def gen_signals(df, p):
    """Vectorized B2 signal generation with params."""
    gain = df['close'] / df['close'].shift(1) - 1
    j_prev = df['J'].shift(1)

    c1 = j_prev < p['j_prev_max']
    c2 = gain > p['gain_min']
    c4 = df['volume'] > df['volume'].shift(1)
    c6 = df['white_line'] > df['yellow_line']
    c7 = df['close'] > df['yellow_line']

    ret_20d = df['close'] / df['close'].shift(20) - 1
    is_strong = ret_20d > p['prior_strong_ret']

    c3_std = df['J'] < p['j_today_max']
    c5_std = (df['high'] - df['close']) / df['close'] < p['shadow_max']
    c3_loose = df['J'] < p['j_today_loose']
    c5_loose = (df['high'] - df['close']) / df['close'] < p['shadow_loose']

    entry = (c1 & c2 & c3_std & c4 & c5_std & c6 & c7) | \
            (c1 & c2 & c3_loose & c4 & c5_loose & c6 & c7 & is_strong)
    return entry.astype(int)


def gen_brick_filter(df, p):
    """Brick CC signal filter with params."""
    green_ok = df['brick_green_streak'].shift(1) >= p['green_streak_min']
    cc = df['brick_green_to_red'] == 1
    wy = (df['white_line'] > df['yellow_line']) & (df['close'] > df['yellow_line'])
    return (green_ok & cc & wy).astype(int)


def eval_signal_fast(data, params):
    """Fast signal-level eval using preloaded data."""
    all_rets = []
    n_sig = 0
    n_stocks_sig = 0

    for code, df in data:
        b2 = gen_signals(df, params)
        brick_cc = gen_brick_filter(df, params)

        # Joint signal: B2 entry WITH brick resonance gets weight boost
        joint = b2 & brick_cc

        sig_mask = b2 == 1
        if sig_mask.sum() == 0:
            continue
        n_stocks_sig += 1
        n_sig += sig_mask.sum()

        rets = df.loc[sig_mask, 'fwd_ret_5d'].dropna()
        if len(rets) > 0:
            all_rets.extend(rets.values.tolist())

    if not all_rets:
        return {'n_signals': 0, 'hit_rate': 0, 'mean_ret': 0, 'sharpe': 0,
                'n_stocks_signal': 0, 'params': params}

    rets = np.array(all_rets)
    return {
        'n_signals': n_sig,
        'n_stocks_signal': n_stocks_sig,
        'hit_rate': float((rets > 0.05).mean()),
        'mean_ret': float(rets.mean()),
        'std_ret': float(rets.std()),
        'median_ret': float(np.median(rets)),
        'sharpe': float(rets.mean() / max(rets.std(), 0.001)),
        'win_rate': float((rets > 0).mean()),
        'params': params,
    }


def sim_portfolio_fast(data, params, start, end):
    """Portfolio simulation using preloaded data."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    all_trades = []
    for code, df in data:
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        odf = df[mask].reset_index(drop=True)
        if len(odf) < 60:
            continue

        b2 = gen_signals(odf, params)
        sig_idx = odf.index[b2 == 1]

        for i in sig_idx:
            if i + 5 < len(odf):
                buy_px = odf.loc[i, 'close']
                sell_px = odf.loc[i + 5, 'open']
                ret = (sell_px / buy_px - 1) - 0.0016
                all_trades.append({'date': odf.loc[i, 'date'], 'ret': ret})

    if not all_trades:
        return {'total_return': 0, 'sharpe': 0, 'n_trades': 0, 'max_dd': 0}

    td = pd.DataFrame(all_trades).sort_values('date')
    daily = td.groupby('date')['ret'].mean().reset_index()
    daily.columns = ['date', 'daily_ret']
    daily['cum'] = (1 + daily['daily_ret']).cumprod()

    total_ret = float(daily['cum'].iloc[-1] - 1)
    dd = float(((daily['cum'] / daily['cum'].cummax() - 1).min()))
    sharpe = float(daily['daily_ret'].mean() / max(daily['daily_ret'].std(), 0.001) * np.sqrt(252))

    return {
        'total_return': total_ret,
        'sharpe': sharpe,
        'n_trades': len(td),
        'win_rate': float((td['ret'] > 0).mean()),
        'max_dd': dd,
    }


def sample_params(grid, n, rng=np.random.RandomState(42)):
    keys, vals = list(grid.keys()), list(grid.values())
    combos = []
    for _ in range(n):
        c = {k: v[rng.randint(len(v))] for k, v in zip(keys, vals)}
        combos.append(c)
    return combos


def main():
    print("=" * 70)
    print("  Phase 2: B2 + Brick Joint Grid Search (memory-optimized)")
    print("=" * 70)
    t0 = time.time()

    paths = sorted([os.path.join(FEAT_DIR, f) for f in os.listdir(FEAT_DIR)
                    if f.endswith('.parquet')])
    print(f"\n  Parquet files: {len(paths)}")

    # --- Stage 1: Signal-level screening on 1000 stocks ---
    print(f"\n  [Stage 1] Preloading {SAMPLE_STAGE1} stocks into memory...")
    data_s1 = preload_data(paths, sample_n=SAMPLE_STAGE1)
    print(f"  Loaded {len(data_s1)} stocks")

    param_sets = [DEFAULT_PARAMS] + sample_params(PARAM_GRID, N_RANDOM_TRIALS)
    print(f"  Evaluating {len(param_sets)} parameter sets...")

    s1_results = []
    n_done = 0
    t1 = time.time()
    for params in param_sets:
        r = eval_signal_fast(data_s1, params)
        s1_results.append(r)
        n_done += 1
        if n_done % 50 == 0:
            print(f"    ... {n_done}/{len(param_sets)} ({time.time()-t1:.0f}s)")

    s1_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  Top 15 signal-level results:")
    print(f"  {'Rank':<6} {'Hit':<8} {'MeanRet':<10} {'Sharpe':<8} {'Signals':<8}")
    for i, r in enumerate(s1_results[:15]):
        dm = ' ← DEFAULT' if r['params'] == DEFAULT_PARAMS else ''
        print(f"  {i+1:<6} {r['hit_rate']:.3f}   {r['mean_ret']:.4f}     "
              f"{r['sharpe']:.3f}    {r['n_signals']:<8}{dm}")

    # --- Stage 2: Full portfolio simulation on all stocks ---
    print(f"\n  [Stage 2] Preloading ALL {len(paths)} stocks into memory...")
    data_all = preload_data(paths)
    print(f"  Loaded {len(data_all)} stocks ({time.time()-t0:.0f}s)")

    top_params = [DEFAULT_PARAMS] + [r['params'] for r in s1_results[:N_TOP_FOR_SIM]
                                     if r['params'] != DEFAULT_PARAMS]
    top_params = top_params[:N_TOP_FOR_SIM + 1]

    windows = [
        ('2024-01-01', '2024-12-31'),
        ('2025-01-01', '2025-12-31'),
        ('2026-01-01', '2026-05-07'),
    ]

    print(f"\n  Simulating top {len(top_params)} param sets on 3 windows...")
    sim_results = []
    t2 = time.time()
    for i, params in enumerate(top_params):
        wmetrics = []
        for ws, we in windows:
            m = sim_portfolio_fast(data_all, params, ws, we)
            m['window'] = f"{ws}~{we}"
            wmetrics.append(m)

        avg_ret = np.mean([m['total_return'] for m in wmetrics])
        avg_sharpe = np.mean([m['sharpe'] for m in wmetrics])
        avg_wr = np.mean([m['win_rate'] for m in wmetrics])
        avg_dd = np.mean([m['max_dd'] for m in wmetrics])
        total_t = sum(m['n_trades'] for m in wmetrics)

        is_def = params == DEFAULT_PARAMS
        tag = ' [DEFAULT]' if is_def else ''
        print(f"  {i+1:>2}. Ret={avg_ret:+.2%} Sharpe={avg_sharpe:.3f} "
              f"WinRate={avg_wr:.1%} MaxDD={avg_dd:.2%} Trades={total_t}{tag}")

        sim_results.append({
            'params': params,
            'avg_return': avg_ret, 'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_wr, 'avg_max_dd': avg_dd,
            'total_trades': total_t, 'window_details': wmetrics,
            'is_default': is_def,
            'b2_desc': f"Jp<{params['j_prev_max']} G>{params['gain_min']} "
                       f"Jt<{params['j_today_max']} Sh<{params['shadow_max']}",
            'brick_desc': f"GS>={params['green_streak_min']} "
                          f"BVh>{params['brick_val_high']} BVm>{params['brick_val_mid']} "
                          f"BW={params['brick_resonance_w']}",
        })

        elapsed = time.time() - t2
        eta = elapsed / (i + 1) * (len(top_params) - i - 1)
        print(f"      ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # --- Pareto ---
    pareto = []
    for i, r in enumerate(sim_results):
        dominated = any(
            r2['avg_return'] >= r['avg_return'] and
            r2['avg_sharpe'] >= r['avg_sharpe'] and
            r2['avg_max_dd'] <= r['avg_max_dd'] and
            (r2['avg_return'] > r['avg_return'] or
             r2['avg_sharpe'] > r['avg_sharpe'] or
             r2['avg_max_dd'] < r['avg_max_dd'])
            for j, r2 in enumerate(sim_results) if i != j
        )
        if not dominated:
            pareto.append(r)

    best = max(pareto, key=lambda r: r['avg_sharpe'])
    default = next((r for r in sim_results if r['is_default']), None)

    print(f"\n  {'='*50}")
    print(f"  Pareto-Optimal: {len(pareto)} sets")
    print(f"  {'='*50}")
    for r in pareto:
        tag = ' ← DEFAULT' if r['is_default'] else ''
        p = r['params']
        print(f"  {r['b2_desc']}")
        print(f"  {r['brick_desc']}")
        print(f"    Ret={r['avg_return']:+.2%} Sharpe={r['avg_sharpe']:.3f} "
              f"DD={r['avg_max_dd']:.2%}{tag}")

    if best:
        print(f"\n  ★ Best (highest Sharpe):")
        print(f"    {best['b2_desc']}")
        print(f"    {best['brick_desc']}")
        print(f"    Ret={best['avg_return']:+.2%} Sharpe={best['avg_sharpe']:.3f} DD={best['avg_max_dd']:.2%}")
        if default and default['avg_return'] != 0:
            imp = (best['avg_return'] - default['avg_return']) / abs(default['avg_return']) * 100
            print(f"    Improvement vs default: {imp:+.1f}% return, "
                  f"Sharpe {best['avg_sharpe'] - default['avg_sharpe']:+.3f}")

    # Save
    out = {
        'best_params': best, 'pareto_optimal': pareto,
        'default_baseline': default,
        'n_signal_trials': N_RANDOM_TRIALS,
        'n_top_simulated': len(top_params),
        'windows': windows,
        'elapsed_seconds': time.time() - t0,
    }
    with open(os.path.join(OUT_DIR, 'phase2_pareto_optimal.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    # CSV
    pd.DataFrame(s1_results).to_csv(os.path.join(OUT_DIR, 'phase2_signal_eval.csv'), index=False)
    pd.DataFrame([{**{f'p_{k}': v for k, v in r['params'].items()},
                   'avg_return': r['avg_return'], 'avg_sharpe': r['avg_sharpe'],
                   'avg_win_rate': r['avg_win_rate'], 'avg_max_dd': r['avg_max_dd'],
                   'total_trades': r['total_trades'], 'is_default': r['is_default']}
                  for r in sim_results]
    ).to_csv(os.path.join(OUT_DIR, 'phase2_top_simulation.csv'), index=False)

    print(f"\n  Phase 2 Complete ({time.time()-t0:.0f}s)")



if __name__ == '__main__':
    main()
