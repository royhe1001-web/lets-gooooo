#!/usr/bin/env python3
"""
Phase 1b: Brick Chart Feature Extraction + Combined Feature Importance
======================================================================
Extracts brick chart strategy features and adds them to the B2 feature set,
then re-runs Random Forest importance analysis on the combined features.

Brick chart parameters extracted:
  - brick_val, brick_var6a (raw values)
  - brick_rising/falling (momentum direction)
  - brick_green_streak, brick_red_streak (consecutive bars)
  - brick_green_to_red (CC buy signal)
  - brick_quality, brick_high_quality (quality scores)
  - brick_pattern (四定式: 1,2,3,4)
  - green_streak before reversal (market context)
  - brick_val change rate (acceleration)
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.strategy_brick import generate_brick_signals
from quant_strategy.strategy_spring import generate_spring_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')


def compute_brick_extra_features(df):
    """Compute additional brick-derived features."""
    if 'brick_val' not in df.columns:
        return df

    df = df.copy()

    # Brick value momentum
    df['brick_val_delta'] = df['brick_val'].diff()
    df['brick_val_delta_3d'] = df['brick_val'].diff(3)
    df['brick_val_accel'] = df['brick_val_delta'].diff()  # acceleration

    # Brick value relative position (normalized)
    brick_max_20 = df['brick_val'].rolling(20).max()
    brick_min_20 = df['brick_val'].rolling(20).min()
    brick_range = brick_max_20 - brick_min_20
    brick_range = brick_range.replace(0, np.nan)
    df['brick_val_position'] = (df['brick_val'] - brick_min_20) / brick_range

    # Brick reversal context
    df['brick_reversal_strength'] = df['brick_val'] - df['brick_val'].shift(1)
    df['brick_turning_from_deep'] = (
        (df['brick_green_to_red'] == 1) &
        (df['brick_green_streak'].shift(1) >= 3)
    ).astype(int)

    # CC signal quality context
    df['brick_cc_with_gain'] = (
        (df['brick_green_to_red'] == 1) &
        ((df['close'] / df['close'].shift(1) - 1) > 0.02)
    ).astype(int)

    # Pattern dummies
    if 'brick_pattern' in df.columns:
        for pattern_name in ['定式一', '定式二', '定式三', '定式四']:
            df[f'brick_p_{pattern_name}'] = (df['brick_pattern'] == pattern_name).astype(int)

    # Green streak features
    df['brick_green_streak_3plus'] = (df['brick_green_streak'] >= 3).astype(int)
    df['brick_green_streak_5plus'] = (df['brick_green_streak'] >= 5).astype(int)

    # Yellow line momentum (from brick strategy context)
    if 'yellow_line' in df.columns:
        df['yellow_line_slope'] = df['yellow_line'].diff(5) / df['yellow_line'].shift(5)

    # Combined brick + B2 context
    df['b2_brick_joint'] = 0  # placeholder for actual joint signal

    return df


def load_and_compute_both(parquet_path):
    """Load features, compute BOTH B2 signals and brick signals."""
    try:
        df = pd.read_parquet(parquet_path)

        # Build OHLCV with pre-computed indicators
        ohlcv = df[['date', 'open', 'close', 'high', 'low', 'volume']].copy()
        ohlcv = ohlcv.set_index('date')

        # Add all pre-computed indicators needed by both strategies
        for col in ['K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                     'white_line', 'yellow_line', 'white_above_yellow', 'white_direction',
                     'brick_val', 'brick_var6a', 'brick_rising', 'brick_falling',
                     'brick_green_to_red', 'brick_red_to_green',
                     'brick_green_streak', 'brick_red_streak',
                     'cross_below_yellow']:
            if col in df.columns:
                ohlcv[col] = df[col].values

        # Compute B2 signals first
        ohlcv = generate_spring_signals(ohlcv, board_type='main', precomputed=True)

        # Compute brick signals (adds brick_entry_signal, brick_pattern, etc.)
        ohlcv = generate_brick_signals(ohlcv, board_type='main', precomputed=True)

        # Merge B2 signal columns
        for col in ['b2_entry_signal', 'b2_position_weight', 'b2_prior_strong',
                     'b2_stop_signal', 'b2_state']:
            if col in ohlcv.columns:
                df[col] = ohlcv[col].values

        # Merge brick signal columns
        for col in ['brick_entry_signal', 'brick_pattern', 'brick_quality',
                     'brick_high_quality']:
            if col in ohlcv.columns:
                df[col] = ohlcv[col].values

        # Compute extra brick features
        df = compute_brick_extra_features(df)

        return df
    except Exception:
        return None


def main():
    print("=" * 70)
    print("  Phase 1b: Brick Feature Extraction + Combined Importance")
    print("=" * 70)
    t0 = time.time()

    feature_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    print(f"\n  Computing brick features for {len(feature_files)} stocks...")

    all_dfs = []
    for i, f in enumerate(feature_files):
        result = load_and_compute_both(os.path.join(FEAT_DIR, f))
        if result is not None:
            all_dfs.append(result)
        if (i + 1) % 1000 == 0:
            print(f"    ... {i+1}/{len(feature_files)} done")

    print(f"  Computed: {len(all_dfs)} stocks")

    # Extract B2 signal features (from phase1, plus brick features)
    print("\n  Extracting combined features at B2 signal points...")

    EXCLUDE_COLS = {
        'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'board_type',
        'fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d',
        'label_up_1d', 'label_up_3d', 'label_up_5d', 'label_up_10d',
        'label_strong_up_1d', 'label_strong_up_3d', 'label_strong_up_5d', 'label_strong_up_10d',
        'label_down_1d', 'label_down_3d', 'label_down_5d', 'label_down_10d',
        'brick_pattern',  # string column
        'b2_entry_signal', 'b2_position_weight', 'b2_prior_strong',
        'b2_stop_signal', 'b2_fly_signal', 'b2_didi_stop',
        'b2_t5_stop', 'b2_candle_stop', 'b2_loss_stop', 'b2_state',
        'b2_brick_resonance', 'b2_sector_score', 'b2_entry_oamv_regime',
        'date_dt',
    }

    frames = []
    feature_cols = None
    for df in all_dfs:
        if 'b2_entry_signal' not in df.columns:
            continue
        signal_mask = df['b2_entry_signal'] == 1
        if signal_mask.sum() == 0:
            continue

        signal_df = df[signal_mask].copy()

        if feature_cols is None:
            all_cols = set(df.columns)
            feature_cols = sorted([c for c in all_cols
                                   if c not in EXCLUDE_COLS
                                   and not c.startswith('b2_')
                                   and not c.startswith('date')
                                   and df[c].dtype in ('float64', 'float32', 'int64', 'int32', 'bool')])

        cols_needed = feature_cols + ['label_strong_up_5d']
        available = [c for c in cols_needed if c in signal_df.columns]
        signal_df = signal_df[available]
        if len(signal_df) > 0:
            frames.append(signal_df)

    combined = pd.concat(frames, ignore_index=True).dropna()
    X = combined[feature_cols].values
    y = combined['label_strong_up_5d'].values

    print(f"  Feature matrix: {X.shape}")
    print(f"  Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

    # --- RF Importance ---
    print("\n  Training Random Forest (200 trees) on combined features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=50,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_,
        'rf_importance_norm': rf.feature_importances_ / rf.feature_importances_.sum(),
    }).sort_values('rf_importance', ascending=False)

    # --- Identify Brick Features ---
    brick_features = [c for c in feature_cols if c.startswith('brick_') or
                      c.startswith('brick') or c == 'yellow_line_slope']

    print(f"\n  RF AUC: {auc:.4f}, Accuracy: {report['accuracy']:.4f}")
    print(f"  Total features: {len(feature_cols)} (brick features: {len(brick_features)})")

    # Compare with Phase 1 (no brick) results
    print(f"\n  {'='*55}")
    print(f"  Top 25 Features (B2 + Brick Combined)")
    print(f"  {'='*55}")
    print(f"  {'Rank':<6} {'Feature':<28} {'RF Imp':<10} {'Type'}")
    print(f"  {'-'*55}")

    brick_count_in_top = 0
    for i, (_, row) in enumerate(importance.head(25).iterrows()):
        is_brick = 'BRICK' if row['feature'] in brick_features else 'B2'
        if is_brick == 'BRICK':
            brick_count_in_top += 1
        print(f"  {i+1:<6} {row['feature']:<28} {row['rf_importance_norm']:.4f}      {is_brick}")

    print(f"\n  Brick features in Top-25: {brick_count_in_top}/25")

    # --- Top Brick Features ---
    brick_imp = importance[importance['feature'].isin(brick_features)].head(15)
    print(f"\n  {'='*50}")
    print(f"  Top Brick-Specific Features")
    print(f"  {'='*50}")
    for _, row in brick_imp.iterrows():
        print(f"    {row['feature']:<30s} importance={row['rf_importance_norm']:.4f}")

    # --- AUC improvement over Phase 1 ---
    # Phase 1 had AUC 0.6273
    phase1_auc = 0.6273
    auc_improvement = (auc - phase1_auc) * 100
    print(f"\n  Phase 1 (B2 only) AUC: {phase1_auc:.4f}")
    print(f"  Phase 1b (B2+Brick) AUC: {auc:.4f}")
    print(f"  Improvement: {auc_improvement:+.1f} basis points")

    # Save
    out = {
        'model': 'RandomForest',
        'phase': 'phase1b_b2_plus_brick',
        'rf_auc': auc,
        'rf_accuracy': report['accuracy'],
        'phase1_auc': phase1_auc,
        'auc_improvement_bps': round(auc_improvement, 1),
        'n_samples': len(y),
        'n_features_total': len(feature_cols),
        'n_features_brick': len(brick_features),
        'brick_features_in_top25': brick_count_in_top,
        'positive_ratio': float(y.sum() / len(y)),
        'top_features': [
            {'feature': row['feature'],
             'rf_importance': round(float(row['rf_importance_norm']), 4),
             'type': 'BRICK' if row['feature'] in brick_features else 'B2'}
            for _, row in importance.head(30).iterrows()
        ],
        'top_brick_features': [
            {'feature': row['feature'],
             'rf_importance': round(float(row['rf_importance_norm']), 4)}
            for _, row in brick_imp.iterrows()
        ],
        'elapsed_seconds': time.time() - t0,
    }

    with open(os.path.join(OUT_DIR, 'phase1b_brick_combined_importance.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    importance.to_csv(os.path.join(OUT_DIR, 'phase1b_feature_importance_combined.csv'), index=False)

    elapsed = time.time() - t0
    print(f"\n  Phase 1b Complete ({elapsed:.0f}s)")
    print(f"  Results: {OUT_DIR}/phase1b_brick_combined_importance.json")


if __name__ == '__main__':
    main()
