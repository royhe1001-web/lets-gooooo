#!/usr/bin/env python3
"""
Phase 1: Feature Importance Analysis for B2 Strategy Parameters
===============================================================
Uses Random Forest + MIC (Maximum Information Coefficient) to rank
feature importance for predicting B2 signal quality (T+5 return >5%).

Methods (from reference papers):
  - RF:  Paper 1 (刘力军) — feature importance ranking
  - MIC: Paper 3 (柴昱白) — nonlinear correlation discovery

Output:
  ML_optimization/phase1_feature_importance.json
  ML_optimization/phase1_feature_importance.csv
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.strategy_spring import generate_spring_signals

DATA_DIR = os.path.join(BASE, 'fetch_kline', 'stock_kline')
FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')
os.makedirs(OUT_DIR, exist_ok=True)


# Feature columns to use for importance analysis
# Exclude: date, raw ohlcv, duplicate/derived columns, label columns, board_type
EXCLUDE_COLS = {
    'date', 'open', 'close', 'high', 'low', 'volume', 'amount',
    'board_type',
    # Labels
    'fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d',
    'label_up_1d', 'label_up_3d', 'label_up_5d', 'label_up_10d',
    'label_strong_up_1d', 'label_strong_up_3d', 'label_strong_up_5d', 'label_strong_up_10d',
    'label_down_1d', 'label_down_3d', 'label_down_5d', 'label_down_10d',
    # B2 signal columns (not features)
    'b2_entry_signal', 'b2_position_weight', 'b2_prior_strong',
    'b2_stop_signal', 'b2_fly_signal', 'b2_didi_stop',
    'b2_t5_stop', 'b2_candle_stop', 'b2_loss_stop', 'b2_state',
    'b2_brick_resonance', 'b2_sector_score', 'b2_entry_oamv_regime',
    'date_dt',
}


def load_and_compute_b2(parquet_path):
    """Load pre-computed features for one stock, compute B2 signals."""
    try:
        df = pd.read_parquet(parquet_path)

        # Reconstruct OHLCV DataFrame for generate_spring_signals
        ohlcv = df[['date', 'open', 'close', 'high', 'low', 'volume']].copy()
        ohlcv = ohlcv.set_index('date')

        # Add pre-computed indicators for generate_spring_signals (precomputed mode)
        for col in ['K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                     'white_line', 'yellow_line', 'cross_below_yellow',
                     'brick_green_to_red', 'brick_val']:
            if col in df.columns:
                ohlcv[col] = df[col].values

        # Compute B2 signals
        ohlcv = generate_spring_signals(ohlcv, board_type='main', precomputed=True)

        # Merge B2 signal columns back
        for col in ['b2_entry_signal', 'b2_position_weight', 'b2_prior_strong',
                     'b2_stop_signal', 'b2_state']:
            if col in ohlcv.columns:
                df[col] = ohlcv[col].values

        # Also add date-indexed access
        df['date_dt'] = pd.to_datetime(df['date'])

        return df
    except Exception as e:
        return None


def extract_signal_features(all_dfs):
    """Extract feature vectors at B2 signal points across all stocks.

    Returns X (features), y (target: T+5 strong up), and metadata.
    """
    frames = []
    feature_cols = None

    for df in all_dfs:
        if df is None or 'b2_entry_signal' not in df.columns:
            continue

        # Get signal rows
        signal_mask = df['b2_entry_signal'] == 1
        if signal_mask.sum() == 0:
            continue

        signal_df = df[signal_mask].copy()

        # Determine feature columns (once) — keep only numeric
        if feature_cols is None:
            all_cols = set(df.columns)
            feature_cols = sorted([c for c in all_cols if c not in EXCLUDE_COLS
                                   and not c.startswith('b2_')
                                   and not c.startswith('brick_streak')
                                   and not c.startswith('date')
                                   and df[c].dtype in ('float64', 'float32', 'int64', 'int32', 'bool')])

        # Select features + target
        cols_needed = feature_cols + ['label_strong_up_5d', 'fwd_ret_5d']
        available = [c for c in cols_needed if c in signal_df.columns]
        signal_df = signal_df[available]

        if len(signal_df) > 0:
            frames.append(signal_df)

    if not frames:
        return None, None, None

    combined = pd.concat(frames, ignore_index=True)
    # Drop rows with NaN
    combined = combined.dropna()

    X = combined[feature_cols].values
    y = combined['label_strong_up_5d'].values

    return X, y, feature_cols


def rf_importance(X, y, feature_names):
    """Train Random Forest and output feature importance ranking."""
    # Handle class imbalance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    class_weight = {0: 1.0, 1: max(1.0, n_neg / max(n_pos, 1))}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=50,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predictions
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)

    # Importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf.feature_importances_,
        'rf_importance_norm': rf.feature_importances_ / rf.feature_importances_.sum(),
    }).sort_values('rf_importance', ascending=False)

    return importance, auc, classification_report(y_test, y_pred, output_dict=True)


def mic_scores(X, y, feature_names, max_samples=20000):
    """Compute MIC (Maximum Information Coefficient) for each feature.

    Uses minepy if available, otherwise falls back to correlation ratio.
    """
    # Subsample for speed
    n = min(len(X), max_samples)
    idx = np.random.RandomState(42).choice(len(X), n, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]

    scores = {}
    try:
        from minepy import MINE
        mine = MINE(alpha=0.6, c=15)
        for i, name in enumerate(feature_names):
            x_col = X_sample[:, i]
            valid = ~(np.isnan(x_col) | np.isinf(x_col))
            if valid.sum() < 10:
                scores[name] = 0.0
                continue
            mine.compute_score(x_col[valid], y_sample[valid])
            scores[name] = mine.mic_
    except ImportError:
        # Fallback: use absolute Pearson correlation
        print("    minepy not available, using Pearson |r| as fallback")
        for i, name in enumerate(feature_names):
            x_col = X_sample[:, i]
            valid = ~(np.isnan(x_col) | np.isinf(x_col))
            if valid.sum() < 10:
                scores[name] = 0.0
                continue
            corr = np.corrcoef(x_col[valid], y_sample[valid])[0, 1]
            scores[name] = abs(corr) if not np.isnan(corr) else 0.0

    return scores


def main():
    print("=" * 70)
    print("  Phase 1: B2 Feature Importance Analysis (RF + MIC)")
    print("=" * 70)
    t0 = time.time()

    # Load pre-computed features for all stocks
    feature_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    print(f"\n  Loading {len(feature_files)} stocks...")

    # Process sequentially (avoid Windows spawn issues)
    all_dfs = []
    for i, f in enumerate(feature_files):
        result = load_and_compute_b2(os.path.join(FEAT_DIR, f))
        if result is not None:
            all_dfs.append(result)
        if (i + 1) % 1000 == 0:
            print(f"    ... {i+1}/{len(feature_files)} computed")

    print(f"  Loaded {len(all_dfs)} stocks with B2 signals")

    # Extract signal features
    print("\n  Extracting features at B2 signal points...")
    X, y, feature_names = extract_signal_features(all_dfs)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Positive samples (T+5 >5%): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  Negative samples:         {len(y) - y.sum()} ({(1-y.sum()/len(y))*100:.1f}%)")

    # --- Random Forest ---
    print("\n  Training Random Forest (200 trees)...")
    rf_imp, rf_auc, rf_report = rf_importance(X, y, feature_names)
    print(f"  RF AUC: {rf_auc:.4f}")
    print(f"  RF Accuracy: {rf_report['accuracy']:.4f}")

    # --- MIC ---
    print("\n  Computing MIC scores...")
    mic = mic_scores(X, y, feature_names)

    # --- Combined Ranking ---
    print("\n  Building combined ranking...")
    # Normalize MIC to 0-1
    mic_max = max(mic.values()) if mic else 1.0
    mic_norm = {k: v / max(mic_max, 0.001) for k, v in mic.items()}

    combined = rf_imp.copy()
    combined['mic_score'] = combined['feature'].map(mic)
    combined['mic_norm'] = combined['feature'].map(mic_norm)
    combined['combined_score'] = (
        combined['rf_importance_norm'] * 0.5 +
        combined['mic_norm'].fillna(0) * 0.5
    )
    combined = combined.sort_values('combined_score', ascending=False)

    # Map features to B2 parameters
    B2_PARAM_MAP = {
        'J_prev': 'J前日上限 <20',
        'J': 'J当日上限 <65',
        'upper_shadow': '上影线上限 <3.5%',
        'ret_1d': '当日涨幅 >4%',
        'vol_change_1d': '放量条件 vol>前日',
        'white_yellow_diff': '白线>黄线',
        'close_yellow_ratio': '收盘>黄线',
        'ret_20d': '前期强势 20日涨>20%',
        'vol_ratio_20': '20日均量比',
        'vol_ratio_5': '5日均量比',
        'J_slope': 'J值变化率',
        'K_D_cross': 'K-D交叉',
        'MACD_HIST': 'MACD柱',
        'MACD_DIF_slope': 'MACD趋势',
        'RSI_6': 'RSI超买超卖',
        'BB_width': '布林带宽度',
        'BB_pct_b': '布林带位置',
        'volatility_20d': '20日波动率',
        'amplitude': '当日振幅',
        'gap': '跳空幅度',
        'max_dd_20d': '20日最大回撤',
        'price_position_20': '20日价格位置',
        'ma_20_bias': '20日均线乖离',
        'ma_60_bias': '60日均线乖离',
        'white_line': '白线值',
        'yellow_line': '黄线值',
        'brick_val': '砖型图值',
        'brick_rising': '砖型图上升',
        'vol_double': '倍量柱',
        'vol_floor': '地量',
        'K_slope': 'K值变化率',
        'MACD_HIST_sign': 'MACD柱方向',
    }

    # Output
    print(f"\n  {'='*60}")
    print(f"  Top 20 Features by Combined Score")
    print(f"  {'='*60}")
    top20 = combined.head(20)
    results_list = []
    for _, row in top20.iterrows():
        feat = row['feature']
        b2_param = B2_PARAM_MAP.get(feat, '')
        rf_score = row['rf_importance_norm']
        mic_score_val = row['mic_norm']
        combined_s = row['combined_score']
        print(f"  {feat:<25s} | RF={rf_score:.3f}  MIC={mic_score_val:.3f}  "
              f"Combined={combined_s:.3f}  → {b2_param}")
        results_list.append({
            'feature': feat,
            'b2_parameter': b2_param,
            'rf_importance': round(float(rf_score), 4),
            'mic_score': round(float(mic_score_val), 4),
            'combined_score': round(float(combined_s), 4),
        })

    # Save results
    out_json = os.path.join(OUT_DIR, 'phase1_feature_importance.json')
    with open(out_json, 'w') as f:
        json.dump({
            'rf_auc': rf_auc,
            'rf_accuracy': rf_report['accuracy'],
            'n_samples': len(y),
            'n_features': len(feature_names),
            'positive_ratio': float(y.sum() / len(y)),
            'top_features': results_list,
            'elapsed_seconds': time.time() - t0,
        }, f, indent=2, ensure_ascii=False)

    out_csv = os.path.join(OUT_DIR, 'phase1_feature_importance.csv')
    combined.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(f"\n  Phase 1 Complete ({elapsed:.0f}s)")
    print(f"  Results: {out_json}")
    print(f"  CSV:     {out_csv}")


if __name__ == '__main__':
    main()
