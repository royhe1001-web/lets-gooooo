#!/usr/bin/env python3
"""
Phase 3: 2D Gamma Function Threshold Optimization
====================================================
Based on Paper 3 (柴昱白): Models threshold pairs as a continuous
Gamma surface to find optimal values beyond discrete grid search.

Gamma function: f(x,y) = A * x^α * y^β * exp(-γ*x - δ*y)
Linearized: ln(f) = ln(A) + α*ln(x) + β*ln(y) - γ*x - δ*y

Optimizes key threshold pairs identified from Phase 2b/2c:
  1. (weight_gain_2x_thresh, weight_gain_2x) — Explosive signal quality
  2. (oamv_defensive_threshold, oamv_aggressive_threshold) — OAMV regime sensitivity
  3. (weight_pullback_thresh, weight_pullback) — Pullback bonus
  4. (gain_min, j_today_max) — Entry quality
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from quant_strategy.strategy_b2 import generate_b2_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

SIM_START = pd.Timestamp('2026-01-01')
SIM_END = pd.Timestamp('2026-05-07')


# Base B2 params (Explosive from Phase 2b + OAMV def2.0 from Phase 2c)
BASE_PARAMS = {
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
    'oamv_aggressive_buffer': 0.95, 'oamv_defensive_buffer': 0.99,
    'oamv_defensive_ban_entry': True,
    # OAMV regime classification thresholds (used to label days)
    'oamv_aggressive_threshold': 3.0,
    'oamv_defensive_threshold': -2.0,
}


def preload_data(n_stocks=500):
    """Preload a sample of stocks for fast evaluation.

    Returns (stock_data, oamv_data) where oamv_data is the daily OAMV
    dataframe indexed by date with oamv_change_pct column.
    """
    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])[:n_stocks]

    # OAMV
    oamv_df = fetch_market_data(start='20180101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    oamv_index = oamv_df.set_index('date')

    needed_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                   'white_line', 'yellow_line',
                   'brick_val', 'brick_rising', 'brick_falling',
                   'brick_green_to_red', 'brick_red_to_green',
                   'brick_green_streak', 'brick_red_streak',
                   'cross_below_yellow']

    # Also store oamv_change_pct per day for regime recomputation
    oamv_change_map = {}
    for d_idx, row in oamv_df.iterrows():
        d = row['date']
        if hasattr(d, 'date'):
            d = d.date()
        oamv_change_map[pd.Timestamp(d)] = float(row['oamv_change_pct'])

    stock_data = {}
    for f in feat_files:
        code = f.replace('.parquet', '')
        try:
            df = pd.read_parquet(os.path.join(FEAT_DIR, f))
            cols_avail = [c for c in needed_cols if c in df.columns]
            df = df[cols_avail].copy()
            df['date'] = pd.to_datetime(df['date'])

            # Store oamv_change_pct per bar so we can reclassify on the fly
            oamv_changes = []
            for d in df['date']:
                oamv_changes.append(oamv_change_map.get(d, 0.0))
            df['_oamv_change_pct'] = oamv_changes

            df = df.set_index('date').sort_index()
            if len(df) >= 120:
                stock_data[code] = df
        except Exception:
            continue
    return stock_data


def eval_params(stock_data, params):
    """Evaluate a parameter set on preloaded stock data. Returns mean signal return.

    If params contains oamv_aggressive_threshold or oamv_defensive_threshold,
    the oamv_regime column is recomputed before calling generate_b2_signals.
    """
    # Extract OAMV regime thresholds if present
    agg_thresh = params.get('oamv_aggressive_threshold', 3.0)
    def_thresh = params.get('oamv_defensive_threshold', -2.0)

    # Build clean B2 params (remove regime thresholds as B2 doesn't know them)
    b2_params = {k: v for k, v in params.items()
                 if k not in ('oamv_aggressive_threshold', 'oamv_defensive_threshold')}

    all_rets = []
    for code, df in stock_data.items():
        try:
            # Recompute oamv_regime based on current thresholds
            df_copy = df.copy()
            chg = df_copy['_oamv_change_pct'].values
            regime = np.where(chg > agg_thresh, 'aggressive',
                     np.where(chg < def_thresh, 'defensive', 'normal'))
            df_copy['oamv_regime'] = regime

            sdf = generate_b2_signals(df_copy, board_type='main', precomputed=True, params=b2_params)
            mask = (sdf.index >= SIM_START) & (sdf.index <= SIM_END)
            sig_idx = sdf.index[mask & (sdf['b2_entry_signal'] == 1)]

            for i in sig_idx:
                if sdf.index.get_loc(i) + 5 < len(sdf):
                    buy_px = float(sdf.loc[i, 'close'])
                    sell_px = float(sdf.iloc[sdf.index.get_loc(i) + 5]['open'])
                    ret = (sell_px / buy_px - 1)
                    all_rets.append(ret)
        except Exception:
            continue

    if not all_rets:
        return 0
    return float(np.mean(all_rets))


def fit_gamma_2d(x_vals, y_vals, z_surface):
    """
    Fit 2D Gamma function: f(x,y) = A * x^α * y^β * exp(-γ*x - δ*y)
    Using OLS on log-linearized form.

    For negative x or y ranges, values are shifted to positive domain
    before log transform, and optimal results are shifted back.

    Returns fitted parameters and the optimal (x, y) that maximizes f.
    """
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    z_arr = np.array(z_surface).flatten()

    # Shift negative domains to positive for log transform
    x_shift = 0.0
    if x_arr.min() <= 0:
        x_shift = abs(x_arr.min()) + 0.01
    y_shift = 0.0
    if y_arr.min() <= 0:
        y_shift = abs(y_arr.min()) + 0.01

    x_pos = x_arr + x_shift
    y_pos = y_arr + y_shift

    # Ensure z is positive
    z_min = z_arr.min()
    if z_min <= 0:
        z_arr = z_arr - z_min + 0.001
    z_arr = np.maximum(z_arr, 0.0001)

    # Build design matrix
    n = len(x_vals) * len(y_vals)
    X_design = np.zeros((n, 5))
    z_flat = np.zeros(n)
    idx = 0
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z = z_surface[i][j]
            X_design[idx, 0] = 1
            X_design[idx, 1] = np.log(max(x_pos[i], 0.001))
            X_design[idx, 2] = np.log(max(y_pos[j], 0.001))
            X_design[idx, 3] = -x  # -γ*x (use original x for linear term)
            X_design[idx, 4] = -y  # -δ*y (use original y for linear term)
            z_flat[idx] = np.log(max(z, 0.0001))
            idx += 1

    # OLS regression
    reg = LinearRegression()
    reg.fit(X_design, z_flat)

    ln_A, alpha, beta = reg.coef_[0], reg.coef_[1], reg.coef_[2]
    gamma_val = max(-reg.coef_[3], 0.0001)   # coef_[3] = -γ
    delta_val = max(-reg.coef_[4], 0.0001)   # coef_[4] = -δ

    print(f"    Gamma fit: alpha={alpha:.3f} beta={beta:.3f} gamma={gamma_val:.4f} delta={delta_val:.4f} R2={reg.score(X_design, z_flat):.3f}", flush=True)

    # Find maximum of Gamma function
    # For shifted domain: f(x',y') where x'=x+x_shift, y'=y+y_shift
    # x'_opt = α/γ, so x_opt = α/γ - x_shift
    x_opt_raw = alpha / gamma_val if gamma_val > 0 else x_vals[len(x_vals)//2]
    y_opt_raw = beta / delta_val if delta_val > 0 else y_vals[len(y_vals)//2]

    # Unshift
    x_opt = x_opt_raw - x_shift
    y_opt = y_opt_raw - y_shift

    # Clamp to search range
    x_opt = np.clip(x_opt, min(x_vals), max(x_vals))
    y_opt = np.clip(y_opt, min(y_vals), max(y_vals))

    # Compute predicted optimum z using original x,y
    A = np.exp(ln_A)
    x_opt_pos = x_opt + x_shift
    y_opt_pos = y_opt + y_shift
    z_opt = A * (x_opt_pos ** alpha) * (y_opt_pos ** beta) * np.exp(-gamma_val * x_opt - delta_val * y_opt)

    return {
        'A': float(A), 'alpha': float(alpha), 'beta': float(beta),
        'gamma': float(gamma_val), 'delta': float(delta_val),
        'x_opt': float(x_opt), 'y_opt': float(y_opt),
        'z_opt_predicted': float(z_opt),
        'r_squared': float(reg.score(X_design, z_flat)),
    }


def optimize_pair(stock_data, base_params, pair_name, x_param, y_param,
                  x_range, y_range, n_grid=10):
    """Optimize a parameter pair using 2D Gamma fitting."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  Optimizing: {pair_name}", flush=True)
    print(f"  Parameters: {x_param} x {y_param}", flush=True)
    print(f"  Range: {x_param}=[{min(x_range)},{max(x_range)}] x {y_param}=[{min(y_range)},{max(y_range)}]", flush=True)
    print(f"  {'='*60}", flush=True)

    x_vals = np.linspace(min(x_range), max(x_range), n_grid)
    y_vals = np.linspace(min(y_range), max(y_range), n_grid)

    # Sample grid
    z_surface = np.zeros((n_grid, n_grid))
    best_z = -np.inf
    best_xy = (None, None)

    t0 = time.time()
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            p = dict(base_params)
            p[x_param] = float(x)
            p[y_param] = float(y)
            z = eval_params(stock_data, p)
            z_surface[i, j] = z
            if z > best_z:
                best_z = z
                best_xy = (x, y)
        elapsed = time.time() - t0
        if (i + 1) % 3 == 0:
            eta = elapsed / (i + 1) * (n_grid - i - 1)
            print(f"    row {i+1}/{n_grid} best_z={best_z*100:.2f}% ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

    # Fit Gamma
    print(f"  Grid best: {x_param}={best_xy[0]:.4f}, {y_param}={best_xy[1]:.4f}, z={best_z*100:.2f}%", flush=True)
    fit = fit_gamma_2d(x_vals, y_vals, z_surface)

    # Validate: Evaluate the Gamma-optimal point
    p_val = dict(base_params)
    p_val[x_param] = fit['x_opt']
    p_val[y_param] = fit['y_opt']
    z_val = eval_params(stock_data, p_val)

    fit['pair_name'] = pair_name
    fit['x_param'] = x_param
    fit['y_param'] = y_param
    fit['grid_best_x'] = float(best_xy[0])
    fit['grid_best_y'] = float(best_xy[1])
    fit['grid_best_z'] = float(best_z)
    fit['gamma_optimal_z_validated'] = float(z_val)
    fit['x_range'] = [float(min(x_range)), float(max(x_range))]
    fit['y_range'] = [float(min(y_range)), float(max(y_range))]

    improvement = (z_val - best_z) / abs(best_z) * 100 if best_z != 0 else 0
    print(f"  Gamma optimal: {x_param}={fit['x_opt']:.4f}, {y_param}={fit['y_opt']:.4f}", flush=True)
    print(f"    Predicted z={fit['z_opt_predicted']*100:.2f}%, Validated z={z_val*100:.2f}%", flush=True)
    print(f"    Grid best z={best_z*100:.2f}%, Gamma improvement: {improvement:+.1f}%", flush=True)

    return fit


def main():
    print("=" * 70)
    print("  Phase 3: 2D Gamma Function Threshold Optimization")
    print("=" * 70)
    t0 = time.time()

    # Preload sample stocks
    print("\n  Preloading 500 stocks for fast evaluation...")
    stock_data = preload_data(n_stocks=300)
    print(f"  Loaded {len(stock_data)} stocks")

    all_results = []

    # Pair 1: Explosive signal quality (gain threshold for 2x, weight multiplier)
    r = optimize_pair(
        stock_data, BASE_PARAMS,
        "Explosive Quality (gain_2x_thresh x weight_gain_2x)",
        'weight_gain_2x_thresh', 'weight_gain_2x',
        x_range=[0.06, 0.12],
        y_range=[1.5, 3.5],
        n_grid=6,
    )
    all_results.append(r)

    # Pair 2: OAMV regime classification thresholds
    r2 = optimize_pair(
        stock_data, BASE_PARAMS,
        "OAMV Regime Sensitivity (defensive_threshold x aggressive_threshold)",
        'oamv_defensive_threshold', 'oamv_aggressive_threshold',
        x_range=[-3.5, -1.5],
        y_range=[2.0, 5.0],
        n_grid=6,
    )
    all_results.append(r2)

    # Pair 3: Pullback bonus
    r3 = optimize_pair(
        stock_data, BASE_PARAMS,
        "Pullback Bonus (pullback_thresh x weight_pullback)",
        'weight_pullback_thresh', 'weight_pullback',
        x_range=[-0.12, -0.03],
        y_range=[1.0, 2.0],
        n_grid=6,
    )
    all_results.append(r3)

    # Pair 4: Entry quality (gain_min x j_today_max)
    r4 = optimize_pair(
        stock_data, BASE_PARAMS,
        "Entry Quality (gain_min x j_today_max)",
        'gain_min', 'j_today_max',
        x_range=[0.03, 0.07],
        y_range=[55, 75],
        n_grid=6,
    )
    all_results.append(r4)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Phase 3 Summary: 2D Gamma Optimal Thresholds")
    print(f"{'='*70}")

    for r in all_results:
        print(f"\n  {r['pair_name']}:")
        print(f"    {r['x_param']}: {r['x_opt']:.4f}  (grid best: {r['grid_best_x']:.4f})")
        print(f"    {r['y_param']}: {r['y_opt']:.4f}  (grid best: {r['grid_best_y']:.4f})")
        print(f"    Mean signal return: {r['gamma_optimal_z_validated']*100:.3f}%")
        print(f"    alpha={r['alpha']:.3f} beta={r['beta']:.3f} gamma={r['gamma']:.4f} delta={r['delta']:.4f} R2={r['r_squared']:.3f}")

    # Save
    out = {
        'phase': 'phase3_gamma_optimization',
        'base_params': BASE_PARAMS,
        'n_stocks_sample': len(stock_data),
        'optimization_results': [{
            'pair_name': r['pair_name'],
            'x_param': r['x_param'], 'y_param': r['y_param'],
            'optimal_x': round(r['x_opt'], 4),
            'optimal_y': round(r['y_opt'], 4),
            'grid_best_x': round(r['grid_best_x'], 4),
            'grid_best_y': round(r['grid_best_y'], 4),
            'z_validated': round(r['gamma_optimal_z_validated'], 6),
            'z_grid_best': round(r['grid_best_z'], 6),
            'alpha': round(r['alpha'], 3), 'beta': round(r['beta'], 3),
            'gamma': round(r['gamma'], 4), 'delta': round(r['delta'], 4),
            'r_squared': round(r['r_squared'], 3),
            'x_range': r['x_range'], 'y_range': r['y_range'],
        } for r in all_results],
        'elapsed_seconds': time.time() - t0,
    }

    with open(os.path.join(OUT_DIR, 'phase3_gamma_optimal.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Phase 3 Complete ({time.time()-t0:.0f}s)")
    print(f"  Results: {OUT_DIR}/phase3_gamma_optimal.json")


if __name__ == '__main__':
    main()
