#!/usr/bin/env python3
"""
Phase 0: Data Preparation for B2 ML Parameter Optimization
==========================================================
Loads all A-share daily data (4263 stocks), pre-calculates 50+ technical
indicators, constructs T+1/T+3/T+5 labels, and saves per-stock parquet files.

Uses ProcessPoolExecutor for parallel processing.

Output:
  ML_optimization/features/  — one parquet per stock with full feature matrix
  ML_optimization/meta.json  — metadata (date range, stock count, feature names)
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.indicators import calc_all_indicators

DATA_DIR = os.path.join(BASE, 'fetch_kline', 'stock_kline')
OUT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
os.makedirs(OUT_DIR, exist_ok=True)


def compute_extra_features(df):
    """Compute additional features beyond calc_all_indicators."""
    df = df.copy()

    # --- Price-derived features ---
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_60d'] = df['close'].pct_change(60)

    # Amplitude
    df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)

    # Gap up/down
    df['gap'] = df['open'] / df['close'].shift(1) - 1

    # Upper shadow ratio
    df['upper_shadow'] = (df['high'] - df['close']) / df['close']
    df['lower_shadow'] = (df['close'] - df['low']) / df['close']

    # --- Volume features ---
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio_5'] = df['volume'] / df['vol_ma5']
    df['vol_ratio_20'] = df['volume'] / df['vol_ma20']
    df['vol_change_1d'] = df['volume'] / df['volume'].shift(1)

    # --- Moving averages ---
    for w in [5, 10, 20, 60, 120]:
        df[f'ma_{w}'] = df['close'].rolling(w).mean()
        df[f'ma_{w}_bias'] = df['close'] / df[f'ma_{w}'] - 1  # 乖离率

    # MA cross signals
    df['ma_5_20_cross'] = (df['ma_5'] > df['ma_20']).astype(int)
    df['ma_20_60_cross'] = (df['ma_20'] > df['ma_60']).astype(int)

    # --- Volatility ---
    df['volatility_5d'] = df['ret_1d'].rolling(5).std()
    df['volatility_20d'] = df['ret_1d'].rolling(20).std()

    # --- KDJ derived ---
    if 'K' in df.columns:
        df['K_slope'] = df['K'].diff(3)
        df['J_slope'] = df['J'].diff(3)
        df['K_D_cross'] = df['K'] - df['D']
        df['J_prev'] = df['J'].shift(1)

    # --- MACD derived ---
    if 'MACD_DIF' in df.columns:
        df['MACD_DIF_slope'] = df['MACD_DIF'].diff(3)
        df['MACD_HIST_sign'] = np.sign(df['MACD_HIST'])

    # --- White/Yellow line derived ---
    if 'white_line' in df.columns:
        df['white_yellow_diff'] = df['white_line'] - df['yellow_line']
        df['close_yellow_ratio'] = df['close'] / df['yellow_line']

    # --- Brick chart derived ---
    if 'brick_val' in df.columns:
        df['brick_change'] = df['brick_val'].diff()

    # --- RSI ---
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for w in [6, 12, 24]:
        avg_gain = gain.rolling(w).mean()
        avg_loss = loss.rolling(w).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f'RSI_{w}'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_mid
    df['BB_pct_b'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # --- Max drawdown (20-day) ---
    df['max_dd_20d'] = df['close'] / df['close'].rolling(20).max() - 1

    # --- Price position in range ---
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    df['price_position_20'] = (df['close'] - low_20) / (high_20 - low_20)

    return df


def construct_labels(df):
    """Construct T+1, T+3, T+5 return labels."""
    df = df.copy()

    for horizon in [1, 3, 5, 10]:
        fwd_ret = df['close'].shift(-horizon) / df['close'] - 1
        df[f'fwd_ret_{horizon}d'] = fwd_ret
        # Binary labels
        df[f'label_up_{horizon}d'] = (fwd_ret > 0.02).astype(int)
        df[f'label_strong_up_{horizon}d'] = (fwd_ret > 0.05).astype(int)
        df[f'label_down_{horizon}d'] = (fwd_ret < -0.05).astype(int)

    return df


def count_csv_files(data_dir):
    return len([f for f in os.listdir(data_dir) if f.endswith('.csv')])


def process_one_stock(args):
    """Process a single stock: load, compute indicators, build features + labels."""
    filepath, code = args
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        if len(df) < 120:
            return code, None, f"too few rows: {len(df)}"

        # Ensure required columns
        required = ['open', 'close', 'high', 'low', 'volume']
        for col in required:
            if col not in df.columns:
                return code, None, f"missing column: {col}"

        # Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Compute core indicators
        df = calc_all_indicators(df, board_type='main')

        # Compute extra features
        df = compute_extra_features(df)

        # Construct labels
        df = construct_labels(df)

        # Save to parquet
        out_path = os.path.join(OUT_DIR, f'{code}.parquet')
        df.to_parquet(out_path, index=False)

        return code, len(df), None

    except Exception as e:
        return code, None, str(e)


def main():
    print("=" * 70)
    print("  Phase 0: B2 ML Feature Engineering & Label Construction")
    print("=" * 70)
    t0 = time.time()

    # Scan files
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    codes = [f.replace('.csv', '') for f in files]
    print(f"\n  Stocks found: {len(codes)}")
    print(f"  Output dir:   {OUT_DIR}")

    # Prepare arguments
    args_list = [(os.path.join(DATA_DIR, f), code) for f, code in zip(files, codes)]

    # Parallel processing
    n_done = 0
    n_fail = 0
    max_date_all = None
    min_date_all = None

    print(f"\n  Processing {len(args_list)} stocks with 8 workers...\n", flush=True)

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one_stock, args): args[1] for args in args_list}
        for future in as_completed(futures):
            code, n_rows, error = future.result()
            if error:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    FAIL {code}: {error}")
            else:
                n_done += 1
            if n_done % 500 == 0:
                print(f"    ... {n_done}/{len(codes)} done", flush=True)

    # Collect metadata
    print(f"\n  Collecting metadata...", flush=True)
    feature_files = sorted([f for f in os.listdir(OUT_DIR) if f.endswith('.parquet')])

    # Sample a few files to get feature names and date range
    sample = pd.read_parquet(os.path.join(OUT_DIR, feature_files[0]))
    feature_cols = list(sample.columns)
    date_col = 'date'

    # Find overall date range (sampling for speed)
    for f in feature_files[:50]:
        df = pd.read_parquet(os.path.join(OUT_DIR, f), columns=['date'])
        dmin = df['date'].min()
        dmax = df['date'].max()
        if min_date_all is None or dmin < min_date_all:
            min_date_all = dmin
        if max_date_all is None or dmax > max_date_all:
            max_date_all = dmax

    meta = {
        'total_stocks': len(feature_files),
        'failed': n_fail,
        'date_range': [str(min_date_all), str(max_date_all)],
        'total_features': len(feature_cols),
        'feature_columns': feature_cols,
        'label_columns': [c for c in feature_cols if c.startswith('label_') or c.startswith('fwd_ret_')],
        'phase': 'phase0',
        'elapsed_seconds': time.time() - t0,
    }

    with open(os.path.join(os.path.dirname(OUT_DIR), 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  {'='*50}")
    print(f"  Phase 0 Complete")
    print(f"  {'='*50}")
    print(f"  Succeeded: {n_done} stocks")
    print(f"  Failed:    {n_fail} stocks")
    print(f"  Features:  {len(feature_cols)} columns")
    print(f"  Labels:    {len(meta['label_columns'])} columns")
    print(f"  Date range: {min_date_all} → {max_date_all}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output:    {OUT_DIR}/")


if __name__ == '__main__':
    main()
