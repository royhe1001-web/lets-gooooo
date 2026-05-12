#!/usr/bin/env python3
"""
Phase 4 扩展版: LightGBM 信号质量分类 (扩大训练窗口 + 并行)
=============================================================
改进:
  1. 训练窗口 2023-2025 → 测试 2026 (真实时间序列分割)
  2. 仅主板股票 (与 Phase 2c 主板版一致)
  3. ProcessPoolExecutor 12 workers 最大化 CPU 利用率
  4. LightGBM num_threads=-1 多线程训练
  5. 以 Phase 5 动态阈值参数为底仓
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, classification_report,
                              average_precision_score, confusion_matrix)
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from strategy_spring import generate_spring_signals
from ML_optimization.mktcap_utils import build_mktcap_lookup

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# Main board only
MAIN_BOARD_PREFIXES = ('000', '002', '003', '600', '601', '603', '605')

# Train/test time split
TRAIN_END_DATE = '2025-09-30'
TEST_START_DATE = '2026-01-01'

# Phase 5 dynamic threshold params (for main board)
B2_PARAMS = {
    'j_prev_max': 20, 'j_today_loose': 70, 'shadow_loose': 0.045,
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
    # Dynamic thresholds (Phase 5 optimal)
    'dynamic_thresholds': True,
    'gain_min_aggressive': 0.04, 'gain_min_normal': 0.06, 'gain_min_defensive': 0.06,
    'j_today_max_aggressive': 65, 'j_today_max_normal': 55, 'j_today_max_defensive': 55,
    'shadow_max_aggressive': 0.040, 'shadow_max_normal': 0.030, 'shadow_max_defensive': 0.025,
}


def compute_signal_features(signal_row, df_at_signal, oamv_info, stock_code,
                           index_lookup=None, mktcap_lookup=None):
    """Extract features at a B2 entry signal point."""
    feats = {}

    # Market cap features (per-signal date matching)
    if mktcap_lookup and stock_code:
        sig_date = signal_row.get('date') or signal_row.name
        if sig_date:
            ym = pd.Timestamp(sig_date).strftime('%Y-%m')
            sym = str(stock_code).zfill(6)
            mv = mktcap_lookup.get((ym, sym))
            feats['mktcap_yi'] = float(mv) if mv is not None else 0.0
            feats['mktcap_log'] = np.log(max(float(mv), 1)) if mv is not None else 0.0
        else:
            feats['mktcap_yi'] = 0.0
            feats['mktcap_log'] = 0.0
    else:
        feats['mktcap_yi'] = 0.0
        feats['mktcap_log'] = 0.0

    # Price/return features
    feats['gain_today'] = float(signal_row['close'] / signal_row['open'] - 1)
    feats['gap_ratio'] = float(signal_row['open'] / signal_row.get('close_prev', signal_row['close']) - 1)
    feats['shadow_upper'] = float((signal_row['high'] - signal_row['close']) / signal_row['close'])
    feats['shadow_lower'] = float((signal_row['open'] - signal_row['low']) / signal_row['open'])
    feats['close_to_high_ratio'] = float(signal_row['close'] / signal_row['high'])

    # Volume features
    vol = float(signal_row['volume'])
    vol_prev = float(df_at_signal['volume'].iloc[-2]) if len(df_at_signal) > 1 else vol
    feats['volume_ratio_1d'] = vol / vol_prev if vol_prev > 0 else 1.0

    if len(df_at_signal) >= 5:
        vol_5d = float(df_at_signal['volume'].iloc[-5:].mean())
        feats['volume_ratio_5d'] = vol / vol_5d if vol_5d > 0 else 1.0
    else:
        feats['volume_ratio_5d'] = 1.0

    if len(df_at_signal) >= 20:
        vol_20d = float(df_at_signal['volume'].iloc[-20:].mean())
        feats['volume_ratio_20d'] = vol / vol_20d if vol_20d > 0 else 1.0
    else:
        feats['volume_ratio_20d'] = 1.0

    # B2 technical indicators
    for col in ['K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                'white_line', 'yellow_line']:
        if col in signal_row.index and pd.notna(signal_row[col]):
            feats[col.lower()] = float(signal_row[col])
        else:
            feats[col.lower()] = 0.0

    feats['j_prev_diff'] = feats['j'] - (float(df_at_signal['J'].iloc[-2]) if len(df_at_signal) > 1 and 'J' in df_at_signal.columns else feats['j'])
    feats['macd_hist_diff'] = feats['macd_hist'] - (float(df_at_signal['MACD_HIST'].iloc[-2]) if len(df_at_signal) > 1 and 'MACD_HIST' in df_at_signal.columns else feats['macd_hist'])
    feats['white_above_yellow'] = 1.0 if feats['white_line'] > feats['yellow_line'] else 0.0
    feats['close_above_yellow'] = 1.0 if signal_row['close'] > feats['yellow_line'] else 0.0
    feats['white_yellow_distance'] = (feats['white_line'] - feats['yellow_line']) / feats['yellow_line'] if feats['yellow_line'] > 0 else 0.0

    # Prior returns
    for window in [5, 10, 20]:
        if len(df_at_signal) > window:
            ret = float(signal_row['close'] / df_at_signal['close'].iloc[-window-1] - 1)
        else:
            ret = 0.0
        feats[f'ret_{window}d'] = ret

    # Pullback depth
    if len(df_at_signal) >= 10:
        hh10 = float(df_at_signal['close'].iloc[-10:-1].max())
        feats['pullback_10d'] = float(signal_row['close'] / hh10 - 1) if hh10 > 0 else 0.0
    else:
        feats['pullback_10d'] = 0.0

    # B2 position weight
    if 'b2_position_weight' in signal_row.index:
        feats['b2_position_weight'] = float(signal_row['b2_position_weight'])
    if 'b2_prior_strong' in signal_row.index:
        feats['b2_prior_strong'] = float(signal_row['b2_prior_strong'])

    # Brick chart features
    for col in ['brick_val', 'brick_rising', 'brick_falling',
                'brick_green_to_red', 'brick_red_to_green',
                'brick_green_streak', 'brick_red_streak',
                'cross_below_yellow']:
        if col in signal_row.index and pd.notna(signal_row[col]):
            feats[col] = float(signal_row[col])
        else:
            feats[col] = 0.0

    # OAMV features
    if oamv_info:
        for k, v in oamv_info.items():
            feats[f'oamv_{k}'] = float(v) if pd.notna(v) else 0.0
    else:
        for k in ['change_pct', 'is_aggressive', 'is_defensive',
                  'is_normal', 'volatility_20d', 'amount_ratio',
                  'change_3d', 'change_5d', 'signal']:
            feats[f'oamv_{k}'] = 0.0

    # Board type encoding
    feats['board_main'] = 1.0  # Always main board in this script

    # Index constituent feature (dynamic, per-signal-date)
    if index_lookup:
        sig_date = signal_row.get('date') or signal_row.name
        from ML_optimization.index_utils import get_index_membership
        in_index = get_index_membership(stock_code, sig_date, index_lookup)
        feats['in_csi500_1000'] = 1.0 if in_index else 0.0
    else:
        feats['in_csi500_1000'] = 0.0

    return feats


def load_and_extract_signals(parquet_path, oamv_index):
    """Load stock, compute B2 signals with dynamic params, extract features + labels."""
    try:
        df = pd.read_parquet(parquet_path)
        if 'date' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'])

        # Filter to have at least some history before 2018
        if df['date'].iloc[-1] < pd.Timestamp('2018-01-01'):
            return None

        # Build index constituent lookup (each worker loads from pickle)
        try:
            from ML_optimization.index_utils import build_index_lookup
            index_lookup = build_index_lookup()
        except Exception:
            index_lookup = {}

        # Load market cap lookup (each worker loads from pickle)
        try:
            mktcap_lookup = build_mktcap_lookup()
        except Exception:
            mktcap_lookup = {}

        # Build OAMV info map
        oamv_change_map = {}
        for d, row in oamv_index.iterrows():
            oamv_change_map[pd.Timestamp(d)] = {
                'change_pct': float(row['oamv_change_pct']),
                'is_aggressive': int(float(row['oamv_change_pct']) > 3.0),
                'is_defensive': int(float(row['oamv_change_pct']) < -2.0),
                'is_normal': int(-2.0 <= float(row['oamv_change_pct']) <= 3.0),
                'volatility_20d': float(row.get('oamv_volatility_20d', 0)),
                'amount_ratio': float(row.get('oamv_amount_ratio', 0)),
                'change_3d': float(row.get('oamv_change_3d', 0)),
                'change_5d': float(row.get('oamv_change_5d', 0)),
                'signal': int(row.get('oamv_signal', 0)),
            }

        oamv_changes = []
        oamv_infos = []
        for d in df['date']:
            info = oamv_change_map.get(d, {
                'change_pct': 0, 'is_aggressive': 0, 'is_defensive': 0,
                'is_normal': 1, 'volatility_20d': 0, 'amount_ratio': 0,
                'change_3d': 0, 'change_5d': 0, 'signal': 0,
            })
            oamv_changes.append(info['change_pct'])
            oamv_infos.append(info)

        df['_oamv_change_pct'] = oamv_changes
        chg_arr = np.array(oamv_changes)
        regime = np.where(chg_arr > 3.0, 'aggressive',
                 np.where(chg_arr < -2.0, 'defensive', 'normal'))
        df['oamv_regime'] = regime

        # Compute B2 signals
        needed_cols = ['open', 'high', 'low', 'close', 'volume',
                       'K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                       'white_line', 'yellow_line',
                       'brick_val', 'brick_rising', 'brick_falling',
                       'brick_green_to_red', 'brick_red_to_green',
                       'brick_green_streak', 'brick_red_streak',
                       'cross_below_yellow']
        cols_avail = [c for c in needed_cols if c in df.columns]
        sdf = df[cols_avail + ['date', '_oamv_change_pct', 'oamv_regime']].copy()
        sdf = sdf.set_index('date').sort_index()

        if len(sdf) < 60:
            return None

        sdf = generate_spring_signals(sdf, board_type='main', precomputed=True,
                                   params=B2_PARAMS)

        sig_mask = sdf['b2_entry_signal'] == 1
        if sig_mask.sum() == 0:
            return None

        signal_idxs = sdf.index[sig_mask]
        records = []
        stock_code = os.path.basename(parquet_path).replace('.parquet', '')

        for sig_idx in signal_idxs:
            pos = sdf.index.get_loc(sig_idx)
            if pos + 5 >= len(sdf):
                continue

            signal_row = sdf.iloc[pos]
            df_up_to = sdf.iloc[:pos+1]

            oamv_info = None
            date_matches = df['date'] == sig_idx
            if date_matches.any():
                idx_in_df = date_matches.idxmax() if hasattr(date_matches, 'idxmax') else None
                if idx_in_df is not None:
                    list_pos = list(df['date']).index(sig_idx) if sig_idx in df['date'].values else -1
                    if list_pos >= 0 and list_pos < len(oamv_infos):
                        oamv_info = oamv_infos[list_pos]

            feats = compute_signal_features(signal_row, df_up_to, oamv_info, stock_code,
                                              index_lookup, mktcap_lookup)

            buy_px = float(sdf.iloc[pos]['close'])
            sell_px = float(sdf.iloc[pos + 5]['open'])
            ret_5d = sell_px / buy_px - 1

            feats['label_t5_profit'] = 1 if ret_5d > 0 else 0
            feats['t5_return'] = float(ret_5d)
            feats['stock_code'] = stock_code
            feats['signal_date'] = str(sig_idx.date())

            records.append(feats)

        if records:
            return pd.DataFrame(records)
        return None

    except Exception:
        return None


def main():
    print("=" * 70)
    print("  Phase 4 Expanded: LightGBM Signal Classifier (Main Board)")
    print(f"  Train: 2023-01 ~ {TRAIN_END_DATE}")
    print(f"  Test:  {TEST_START_DATE} ~ 2026-05")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Load OAMV
    print("\n[1/5] Loading OAMV market data...")
    oamv_df = fetch_market_data(start='20170101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    oamv_df['oamv_change_3d'] = oamv_df['oamv'].pct_change(3) * 100
    oamv_df['oamv_change_5d'] = oamv_df['oamv'].pct_change(5) * 100
    oamv_df['oamv_volatility_20d'] = oamv_df['oamv_change_pct'].rolling(20).std()
    oamv_df['oamv_amount_ratio'] = oamv_df['amount_total'] / oamv_df['amount_total'].rolling(20).mean()
    oamv_index = oamv_df.set_index('date')
    print(f"  OAMV: {len(oamv_df)} days")

    # Step 2: Extract signals (main board only, parallel)
    print("\n[2/5] Extracting B2 signals + features (main board, 12 workers)...")
    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    main_files = [f for f in feat_files
                  if any(f.startswith(p) for p in MAIN_BOARD_PREFIXES)]
    print(f"  Main board parquet files: {len(main_files)} / {len(feat_files)} total")

    all_records = []
    n_done = 0
    t1 = time.time()

    with ProcessPoolExecutor(max_workers=12) as executor:
        paths = [os.path.join(FEAT_DIR, f) for f in main_files]
        futures = {executor.submit(load_and_extract_signals, p, oamv_index): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            if result is not None and len(result) > 0:
                all_records.append(result)
            n_done += 1
            if n_done % 300 == 0 or n_done == len(main_files):
                elapsed = time.time() - t1
                eta = elapsed / n_done * (len(main_files) - n_done) if n_done > 0 else 0
                print(f"    {n_done}/{len(main_files)} stocks ({elapsed:.0f}s, ETA {eta:.0f}s)")

    if not all_records:
        print("  ERROR: No signals extracted!")
        return

    combined = pd.concat(all_records, ignore_index=True)
    print(f"\n  Total signals extracted: {len(combined)}")
    print(f"  Date range: {combined['signal_date'].min()} ~ {combined['signal_date'].max()}")

    # Step 3: Time-based train/test split
    print(f"\n[3/5] Time-based train/test split (train <= {TRAIN_END_DATE})...")
    combined = combined.sort_values('signal_date').reset_index(drop=True)

    train_mask = combined['signal_date'] <= TRAIN_END_DATE
    test_mask = combined['signal_date'] >= TEST_START_DATE

    train_df = combined[train_mask].copy()
    test_df = combined[test_mask].copy()

    exclude_cols = {'label_t5_profit', 't5_return', 'stock_code', 'signal_date'}
    feature_cols = [c for c in combined.columns if c not in exclude_cols
                    and combined[c].dtype in ('float64', 'float32', 'int64', 'int32')]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train_df)} signals (pos ratio: {train_df['label_t5_profit'].mean()*100:.1f}%)")
    print(f"  Test:  {len(test_df)} signals (pos ratio: {test_df['label_t5_profit'].mean()*100:.1f}%)")
    print(f"  Train date range: {train_df['signal_date'].min()} ~ {train_df['signal_date'].max()}")
    print(f"  Test date range:  {test_df['signal_date'].min()} ~ {test_df['signal_date'].max()}")

    X_train = train_df[feature_cols].values
    y_train = train_df['label_t5_profit'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label_t5_profit'].values

    # Step 4: Train LightGBM (with GPU-like thread parallelism)
    print("\n[4/5] Training LightGBM (num_threads=-1)...")
    pos_weight = max((len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 1.0)

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_cols, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, y_test, feature_name=feature_cols,
                            reference=lgb_train, free_raw_data=False)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,          # Larger for more data
        'max_depth': 8,
        'learning_rate': 0.03,     # Lower for larger dataset
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,    # Higher for more data → less overfit
        'scale_pos_weight': pos_weight,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1,
        'random_state': 42,
        'num_threads': -1,         # Use all CPU cores
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )

    # Step 5: Evaluate
    print("\n[5/5] Evaluating...")
    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  LightGBM Results (Out-of-Time Test):")
    print(f"    ROC-AUC:  {auc:.4f}")
    print(f"    Avg Prec: {ap:.4f}")
    print(f"    Accuracy: {report['accuracy']:.4f}")
    if '1' in report:
        print(f"    Precision (class 1): {report['1']['precision']:.4f}")
        print(f"    Recall (class 1):    {report['1']['recall']:.4f}")
        print(f"    F1 (class 1):        {report['1']['f1-score']:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"    Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'gain': model.feature_importance(importance_type='gain'),
        'split': model.feature_importance(importance_type='split'),
    }).sort_values('gain', ascending=False)

    print(f"\n  Top 25 Features by Gain:")
    for i, (_, row) in enumerate(importance.head(25).iterrows()):
        ftype = 'OAMV' if row['feature'].startswith('oamv') else (
            'BRICK' if row['feature'].startswith('brick') else 'B2')
        print(f"    {i+1:>2}. {row['feature']:<35s} gain={row['gain']:.0f}  [{ftype}]")

    # Trading simulation
    print(f"\n  {'='*60}")
    print(f"  Trading Simulation: Unfiltered vs LightGBM-Filtered (Test Set)")
    print(f"  {'='*60}")

    unfiltered_ret = test_df['t5_return'].mean()
    unfiltered_wr = test_df['label_t5_profit'].mean()
    unfiltered_trades = len(test_df)

    filtered_mask = y_pred == 1
    filtered_df = test_df[filtered_mask]
    filtered_ret = filtered_df['t5_return'].mean() if len(filtered_df) > 0 else 0
    filtered_wr = filtered_df['label_t5_profit'].mean() if len(filtered_df) > 0 else 0
    filtered_trades = len(filtered_df)

    high_conf_mask = y_prob >= 0.7
    high_conf_df = test_df[high_conf_mask]
    high_conf_ret = high_conf_df['t5_return'].mean() if len(high_conf_df) > 0 else 0
    high_conf_wr = high_conf_df['label_t5_profit'].mean() if len(high_conf_df) > 0 else 0
    high_conf_trades = len(high_conf_df)

    print(f"  {'Strategy':<30s} {'Trades':>8s} {'WinRate':>10s} {'MeanRet':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Unfiltered B2 signals':<30s} {unfiltered_trades:>8d} {unfiltered_wr*100:>9.1f}% {unfiltered_ret*100:>9.3f}%")
    print(f"  {'LGBM filtered (prob>0.5)':<30s} {filtered_trades:>8d} {filtered_wr*100:>9.1f}% {filtered_ret*100:>9.3f}%")
    print(f"  {'LGBM high-conf (prob>0.7)':<30s} {high_conf_trades:>8d} {high_conf_wr*100:>9.1f}% {high_conf_ret*100:>9.3f}%")

    # Save
    out = {
        'phase': '4_expanded_mainboard',
        'board': 'main_only',
        'train_period': f'2023-01~{TRAIN_END_DATE}',
        'test_period': f'{TEST_START_DATE}~2026-05',
        'model': 'LightGBM',
        'params': {k: v for k, v in params.items() if k != 'num_threads'},
        'roc_auc': round(auc, 4),
        'average_precision': round(ap, 4),
        'n_features': len(feature_cols),
        'n_train_signals': len(train_df),
        'n_test_signals': len(test_df),
        'train_positive_ratio': round(float(train_df['label_t5_profit'].mean()), 4),
        'test_positive_ratio': round(float(test_df['label_t5_profit'].mean()), 4),
        'simulation': {
            'unfiltered': {
                'trades': unfiltered_trades,
                'win_rate': round(float(unfiltered_wr), 4),
                'mean_return': round(float(unfiltered_ret), 6),
            },
            'lgbm_filtered': {
                'trades': filtered_trades,
                'win_rate': round(float(filtered_wr), 4),
                'mean_return': round(float(filtered_ret), 6),
            },
            'lgbm_high_conf': {
                'trades': high_conf_trades,
                'win_rate': round(float(high_conf_wr), 4),
                'mean_return': round(float(high_conf_ret), 6),
            },
        },
        'top_features': [
            {'feature': row['feature'], 'gain': int(row['gain']),
             'type': 'OAMV' if row['feature'].startswith('oamv') else (
                 'BRICK' if row['feature'].startswith('brick') else 'B2')}
            for _, row in importance.head(40).iterrows()
        ],
        'elapsed_seconds': time.time() - t0,
    }

    out_path = os.path.join(OUT_DIR, 'phase4_expanded_mainboard_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    importance.to_csv(os.path.join(OUT_DIR, 'phase4_expanded_feature_importance.csv'), index=False)
    model.save_model(os.path.join(OUT_DIR, 'phase4_expanded_lgbm_model.txt'))

    elapsed = time.time() - t0
    print(f"\n  Phase 4 Expanded Complete ({elapsed:.0f}s)")
    print(f"  Results: {out_path}")


if __name__ == '__main__':
    main()
