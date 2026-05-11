#!/usr/bin/env python3
"""
Phase 1c: 0AMV Market Environment Feature Importance
=====================================================
Treats 0AMV (活筹指数) as a market environment factor. Extracts 0AMV
features at each B2 signal point, combines with existing B2+Brick features,
and runs Random Forest importance analysis.

Key insight: 0AMV > 3% daily → aggressive (赚钱效应爆发)
             0AMV < -2.35% daily → defensive (赚钱效应萎靡)
             These features help filter false breakouts (诱多/假突破).
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

from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# ============================================================
# 1. Compute 0AMV Features
# ============================================================
def compute_oamv_features():
    """Compute 0AMV market data and extract features as a daily DataFrame."""
    print("  Computing 0AMV market indicators...")
    df = fetch_market_data(start='20180101')
    df = calc_oamv(df)
    df = generate_signals(df)

    # Extra features
    df['oamv_change_3d'] = df['oamv'].pct_change(3) * 100
    df['oamv_change_5d'] = df['oamv'].pct_change(5) * 100
    df['oamv_cyc5_slope'] = df['oamv_cyc5'].diff(5) / df['oamv_cyc5'].shift(5) * 100
    df['oamv_volatility_20d'] = df['oamv_change_pct'].rolling(20).std()
    df['oamv_amount_ratio'] = df['amount_total'] / df['amount_total'].rolling(20).mean()

    # Regime classification (per user spec)
    df['oamv_regime'] = 'normal'
    df.loc[df['oamv_change_pct'] > 3.0, 'oamv_regime'] = 'aggressive'
    df.loc[df['oamv_change_pct'] < -2.35, 'oamv_regime'] = 'defensive'

    # Regime one-hot
    df['oamv_is_aggressive'] = (df['oamv_regime'] == 'aggressive').astype(int)
    df['oamv_is_defensive'] = (df['oamv_regime'] == 'defensive').astype(int)
    df['oamv_is_normal'] = (df['oamv_regime'] == 'normal').astype(int)

    # Interaction: regime × signal (to be computed later)
    df['oamv_golden_in_aggressive'] = (
        (df['oamv_signal'] == 1) & (df['oamv_regime'] == 'aggressive')
    ).astype(int)

    # Zone one-hot encoding
    zone_dummies = pd.get_dummies(df['oamv_zone'], prefix='oamv_zone')
    for col in zone_dummies.columns:
        df[col] = zone_dummies[col].astype(int)

    print(f"  0AMV data: {len(df)} days, {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"    aggressive days: {(df['oamv_regime']=='aggressive').sum()}")
    print(f"    defensive days:  {(df['oamv_regime']=='defensive').sum()}")
    print(f"    normal days:     {(df['oamv_regime']=='normal').sum()}")

    return df


# ============================================================
# 2. Load Parquet + Extract Signals with 0AMV Features
# ============================================================
OAMV_FEATURE_COLS = [
    'oamv', 'oamv_cyc5', 'oamv_cyc13', 'oamv_diff',
    'oamv_change_pct', 'oamv_change_3d', 'oamv_change_5d',
    'oamv_trend', 'oamv_direction', 'oamv_signal',
    'oamv_divergence', 'oamv_cyc5_slope',
    'oamv_volatility_20d', 'oamv_amount_ratio',
    'oamv_is_aggressive', 'oamv_is_defensive', 'oamv_is_normal',
    'oamv_golden_in_aggressive',
]

EXCLUDE_COLS = {
    'date', 'open', 'close', 'high', 'low', 'board_type',
    'fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d',
    'label_up_1d', 'label_up_3d', 'label_up_5d', 'label_up_10d',
    'label_strong_up_1d', 'label_strong_up_3d', 'label_strong_up_5d', 'label_strong_up_10d',
    'label_down_1d', 'label_down_3d', 'label_down_5d', 'label_down_10d',
}

# Additional exclude: brick_pattern (string), date_dt, and OAMV cols we'll merge
OAMV_EXCLUDE = {
    'oamv_zone_持续上涨区', 'oamv_zone_上涨犹豫区', 'oamv_zone_持续下跌区',
    'oamv_zone_下跌犹豫区', 'oamv_zone_震荡', 'oamv_regime',
}

def extract_features_with_oamv(parquet_path, oamv_df):
    """Load parquet, find B2 signal points, attach 0AMV features."""
    try:
        df = pd.read_parquet(parquet_path)

        if 'b2_entry_signal' not in df.columns or 'date' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'])
        signal_mask = df['b2_entry_signal'] == 1
        if signal_mask.sum() == 0:
            return None

        signal_df = df[signal_mask].copy()

        # Merge 0AMV features by date
        signal_dates = signal_df['date'].dt.date
        oamv_date_index = oamv_df.set_index('date')

        for col in OAMV_FEATURE_COLS:
            if col in oamv_df.columns:
                # Map by date
                date_map = oamv_df.set_index('date')[col].to_dict()
                signal_df[col] = signal_df['date'].map(
                    lambda d: date_map.get(pd.Timestamp(d), np.nan)
                )
            else:
                signal_df[col] = np.nan

        # Also add zone dummies
        for col in oamv_df.columns:
            if col.startswith('oamv_zone_'):
                date_map = oamv_df.set_index('date')[col].to_dict()
                signal_df[col] = signal_df['date'].map(
                    lambda d: date_map.get(pd.Timestamp(d), 0)
                )

        return signal_df
    except Exception:
        return None


def main():
    print("=" * 70)
    print("  Phase 1c: 0AMV Market Environment Feature Importance")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Compute 0AMV
    oamv_df = compute_oamv_features()

    # Step 2: Load feature files + extract signals with 0AMV
    feature_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    print(f"\n  Extracting B2 signals + 0AMV features from {len(feature_files)} stocks...")

    # We need B2 signals on the parquet files. The parquet files from Phase 0
    # don't have b2_entry_signal. We need to compute it first, or use
    # pre-computed B2 signals from Phase 1b.
    #
    # Check if b2_entry_signal is available in parquet files
    sample = pd.read_parquet(os.path.join(FEAT_DIR, feature_files[0]))
    has_b2 = 'b2_entry_signal' in sample.columns
    print(f"  b2_entry_signal in parquet: {has_b2}")

    if not has_b2:
        # Need to compute B2 signals first. Use the full pipeline from Phase 1.
        print("  Computing B2 signals on all stocks (this takes a few minutes)...")
        from quant_strategy.strategy_b2 import generate_b2_signals

        all_dfs = []
        n_done = 0
        with ProcessPoolExecutor(max_workers=8) as executor:
            paths = [os.path.join(FEAT_DIR, f) for f in feature_files]
            futures = {executor.submit(_load_and_compute_b2, p): p for p in paths}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_dfs.append(result)
                n_done += 1
                if n_done % 2000 == 0:
                    print(f"    ... {n_done}/{len(feature_files)} B2 signals computed")

        print(f"  Computed B2 signals for {len(all_dfs)} stocks")
    else:
        # Load quickly with just signal rows
        all_dfs = []
        for f in feature_files:
            df = pd.read_parquet(os.path.join(FEAT_DIR, f))
            if 'b2_entry_signal' in df.columns:
                signal_mask = df['b2_entry_signal'] == 1
                if signal_mask.sum() > 0:
                    all_dfs.append(df[signal_mask].copy())
        print(f"  Loaded {len(all_dfs)} stocks with B2 signals")

    # Step 3: Attach 0AMV features
    print("\n  Attaching 0AMV features to signal points...")
    frames = []
    for df in all_dfs:
        if 'b2_entry_signal' not in df.columns:
            continue
        if 'date' not in df.columns:
            continue
        df['date'] = pd.to_datetime(df['date'])
        signal_mask = df['b2_entry_signal'] == 1
        if signal_mask.sum() == 0:
            continue

        signal_df = df[signal_mask].copy()

        # Merge 0AMV features by date
        for col in OAMV_FEATURE_COLS:
            if col in oamv_df.columns:
                date_map = oamv_df.set_index('date')[col].to_dict()
                signal_df[col] = signal_df['date'].apply(
                    lambda d: date_map.get(d, np.nan) if pd.notna(d) else np.nan
                )

        for col in oamv_df.columns:
            if col.startswith('oamv_zone_'):
                date_map = oamv_df.set_index('date')[col].to_dict()
                signal_df[col] = signal_df['date'].apply(
                    lambda d: date_map.get(d, 0) if pd.notna(d) else 0
                )

        if len(signal_df) > 0:
            frames.append(signal_df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined signal dataframe: {combined.shape}")

    # Step 4: Determine feature columns
    all_cols = set(combined.columns)
    exclude_all = EXCLUDE_COLS | OAMV_EXCLUDE | {'date_dt'}
    feature_cols = sorted([c for c in all_cols
                           if c not in exclude_all
                           and not c.startswith('b2_')
                           and not c.startswith('brick_streak')
                           and not c.startswith('date')
                           and combined[c].dtype in ('float64', 'float32', 'int64', 'int32', 'bool')])

    # Filter to available
    feature_cols = [c for c in feature_cols if c in combined.columns]

    # Ensure target exists
    if 'label_strong_up_5d' not in combined.columns:
        print("  ERROR: label_strong_up_5d not found!")
        return

    clean = combined[feature_cols + ['label_strong_up_5d']].dropna()
    X = clean[feature_cols].values
    y = clean['label_strong_up_5d'].values

    print(f"  Feature matrix: {X.shape}")
    print(f"  Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  OAMV features: {len([c for c in feature_cols if c.startswith('oamv')])}")

    # Step 5: Train RF
    print("\n  Training Random Forest (200 trees)...")
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

    # Step 6: Report
    oamv_features = [c for c in feature_cols if c.startswith('oamv')]
    print(f"\n  RF AUC: {auc:.4f}, Accuracy: {report['accuracy']:.4f}")
    print(f"  Total features: {len(feature_cols)} (OAMV: {len(oamv_features)})")

    print(f"\n  {'='*60}")
    print(f"  Top 30 Features (B2 + Brick + OAMV)")
    print(f"  {'='*60}")
    print(f"  {'Rank':<6} {'Feature':<30} {'RF Imp':<10} {'Type'}")
    print(f"  {'-'*60}")

    oamv_in_top = 0
    for i, (_, row) in enumerate(importance.head(30).iterrows()):
        ftype = 'OAMV' if row['feature'].startswith('oamv') else (
            'BRICK' if row['feature'].startswith('brick') else 'B2')
        if ftype == 'OAMV':
            oamv_in_top += 1
        print(f"  {i+1:<6} {row['feature']:<30} {row['rf_importance_norm']:.4f}      {ftype}")

    print(f"\n  OAMV features in Top-30: {oamv_in_top}/30")

    # Step 7: Top OAMV features only
    oamv_imp = importance[importance['feature'].isin(oamv_features)].head(20)
    print(f"\n  {'='*50}")
    print(f"  Top OAMV Features")
    print(f"  {'='*50}")
    for _, row in oamv_imp.iterrows():
        print(f"    {row['feature']:<35s} importance={row['rf_importance_norm']:.4f}")

    # Step 8: AUC comparison with Phase 1 (no 0AMV) and Phase 1b (B2+Brick)
    phase1_auc = 0.6273
    phase1b_auc = 0.6287
    print(f"\n  AUC Comparison:")
    print(f"    Phase 1  (B2 only):        {phase1_auc:.4f}")
    print(f"    Phase 1b (B2 + Brick):     {phase1b_auc:.4f}")
    print(f"    Phase 1c (B2 + Brick + 0AMV): {auc:.4f}")
    print(f"    Improvement over Phase 1b:  {(auc - phase1b_auc)*100:+.1f} bps")

    # Save
    out = {
        'phase': 'phase1c_oamv_features',
        'model': 'RandomForest',
        'rf_auc': auc,
        'rf_accuracy': report['accuracy'],
        'phase1_auc': phase1_auc,
        'phase1b_auc': phase1b_auc,
        'auc_improvement_vs_phase1b_bps': round((auc - phase1b_auc) * 100, 1),
        'n_samples': len(y),
        'n_features_total': len(feature_cols),
        'n_features_oamv': len(oamv_features),
        'oamv_features_in_top30': oamv_in_top,
        'positive_ratio': float(y.sum() / len(y)),
        'top_features': [
            {'feature': row['feature'],
             'rf_importance': round(float(row['rf_importance_norm']), 4),
             'type': 'OAMV' if row['feature'].startswith('oamv') else (
                 'BRICK' if row['feature'].startswith('brick') else 'B2')}
            for _, row in importance.head(40).iterrows()
        ],
        'top_oamv_features': [
            {'feature': row['feature'],
             'rf_importance': round(float(row['rf_importance_norm']), 4)}
            for _, row in oamv_imp.iterrows()
        ],
        'elapsed_seconds': time.time() - t0,
    }

    with open(os.path.join(OUT_DIR, 'phase1c_oamv_importance.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    importance.to_csv(os.path.join(OUT_DIR, 'phase1c_feature_importance_oamv.csv'), index=False)

    elapsed = time.time() - t0
    print(f"\n  Phase 1c Complete ({elapsed:.0f}s)")
    print(f"  Results: {OUT_DIR}/phase1c_oamv_importance.json")


def _load_and_compute_b2(parquet_path):
    """Helper: load parquet and compute B2 signals."""
    try:
        df = pd.read_parquet(parquet_path)
        ohlcv = df[['date', 'open', 'close', 'high', 'low', 'volume']].copy()
        ohlcv = ohlcv.set_index('date')

        for col in ['K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                     'white_line', 'yellow_line', 'white_above_yellow', 'white_direction',
                     'brick_val', 'brick_var6a', 'brick_rising', 'brick_falling',
                     'brick_green_to_red', 'brick_red_to_green',
                     'brick_green_streak', 'brick_red_streak',
                     'cross_below_yellow']:
            if col in df.columns:
                ohlcv[col] = df[col].values

        from quant_strategy.strategy_b2 import generate_b2_signals
        ohlcv = generate_b2_signals(ohlcv, board_type='main', precomputed=True)

        for col in ['b2_entry_signal', 'b2_position_weight', 'b2_prior_strong',
                     'b2_stop_signal', 'b2_state', 'b2_brick_resonance']:
            if col in ohlcv.columns:
                df[col] = ohlcv[col].values

        return df
    except Exception:
        return None


if __name__ == '__main__':
    main()
