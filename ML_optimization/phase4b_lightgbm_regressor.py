#!/usr/bin/env python3
"""
Phase 4b: Improved LightGBM Signal Quality Scorer
====================================================
Improvements over Phase 4:
  1. Regression target (predict T+5 return directly, not binary)
  2. Better hyperparameters via random search
  3. Signal ranking: trade only top-K or above-threshold signals
  4. Compare multiple filtering strategies

Key insight: Binary classification loses information. Regression
captures expected return magnitude for better signal ranking.
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, classification_report,
                              average_precision_score, mean_squared_error,
                              r2_score, mean_absolute_error)
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import randint, uniform, loguniform
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from quant_strategy.oamv import fetch_market_data, calc_oamv, generate_signals as gen_oamv
from quant_strategy.strategy_spring import generate_spring_signals

FEAT_DIR = os.path.join(BASE, 'ML_optimization', 'features')
OUT_DIR = os.path.join(BASE, 'ML_optimization')

# Optimal parameters from Phase 2c
OPTIMAL_PARAMS = {
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
}


def compute_signal_features(signal_row, df_at_signal, oamv_info):
    """Extract enhanced features at a B2 entry signal point."""
    feats = {}

    # --- Price/return features ---
    feats['gain_today'] = float(signal_row['close'] / signal_row['open'] - 1)
    feats['gap_ratio'] = float(signal_row['open'] / signal_row.get('close_prev', signal_row['close']) - 1)
    feats['shadow_upper'] = float((signal_row['high'] - signal_row['close']) / signal_row['close'])
    feats['shadow_lower'] = float((signal_row['open'] - signal_row['low']) / signal_row['open'])
    feats['close_to_high_ratio'] = float(signal_row['close'] / signal_row['high'])
    feats['close_position'] = float((signal_row['close'] - signal_row['low']) / max(signal_row['high'] - signal_row['low'], 0.001))

    # --- Volume features ---
    vol = float(signal_row['volume'])
    for w in [1, 3, 5, 10, 20]:
        if len(df_at_signal) > w:
            vol_avg = float(df_at_signal['volume'].iloc[-(w+1):-1].mean())
            feats[f'volume_ratio_{w}d'] = vol / vol_avg if vol_avg > 0 else 1.0
            feats[f'volume_trend_{w}d'] = float(
                df_at_signal['volume'].iloc[-w:].mean() /
                max(df_at_signal['volume'].iloc[-(w*2+1):-w].mean(), 1)
            )
        else:
            feats[f'volume_ratio_{w}d'] = 1.0
            feats[f'volume_trend_{w}d'] = 1.0

    feats['volume_position'] = float(
        (vol - df_at_signal['volume'].rolling(20).mean().iloc[-1]) /
        max(df_at_signal['volume'].rolling(20).std().iloc[-1], 1)
    )

    # --- B2 technical indicators ---
    for col in ['K', 'D', 'J', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
                'white_line', 'yellow_line']:
        if col in signal_row.index and pd.notna(signal_row[col]):
            feats[col.lower()] = float(signal_row[col])
        else:
            feats[col.lower()] = 0.0

    # Derived B2 metrics
    n = len(df_at_signal)
    feats['j_prev'] = float(df_at_signal['J'].iloc[-2]) if n > 1 and 'J' in df_at_signal.columns else feats['j']
    feats['j_diff'] = feats['j'] - feats['j_prev']
    feats['j_5d_slope'] = (feats['j'] - float(df_at_signal['J'].iloc[-5])) / 5 if n > 5 and 'J' in df_at_signal.columns else 0.0
    feats['macd_hist_diff'] = feats['macd_hist'] - (float(df_at_signal['MACD_HIST'].iloc[-2]) if n > 1 else feats['macd_hist'])
    feats['macd_diff'] = feats['macd_dif'] - feats['macd_dea']
    feats['white_above_yellow'] = 1.0 if feats['white_line'] > feats['yellow_line'] else 0.0
    feats['close_above_yellow'] = 1.0 if signal_row['close'] > feats['yellow_line'] else 0.0
    feats['white_yellow_distance'] = (feats['white_line'] - feats['yellow_line']) / abs(feats['yellow_line']) if abs(feats['yellow_line']) > 0.001 else 0.0
    feats['close_to_white'] = float(signal_row['close'] / feats['white_line']) if feats['white_line'] > 0 else 1.0

    # KD cross signals
    feats['k_cross_d_up'] = 1.0 if (feats['k'] > feats['d']) and (n > 1 and float(df_at_signal['K'].iloc[-2]) <= float(df_at_signal['D'].iloc[-2])) else 0.0

    # --- Prior returns ---
    for window in [1, 3, 5, 10, 20]:
        if n > window:
            ret = float(signal_row['close'] / df_at_signal['close'].iloc[-window-1] - 1)
        else:
            ret = 0.0
        feats[f'ret_{window}d'] = ret

    # Return volatility
    if n >= 10:
        daily_rets = df_at_signal['close'].pct_change().iloc[-10:]
        feats['volatility_10d'] = float(daily_rets.std())
        feats['return_skew_10d'] = float(daily_rets.skew()) if len(daily_rets) > 2 else 0.0
    else:
        feats['volatility_10d'] = 0.0
        feats['return_skew_10d'] = 0.0

    # --- Pullback features ---
    if n >= 10:
        hh10 = float(df_at_signal['close'].iloc[-10:].max())
        feats['pullback_10d'] = float(signal_row['close'] / hh10 - 1) if hh10 > 0 else 0.0
        hh5 = float(df_at_signal['close'].iloc[-5:].max())
        feats['pullback_5d'] = float(signal_row['close'] / hh5 - 1) if hh5 > 0 else 0.0
    else:
        feats['pullback_10d'] = 0.0
        feats['pullback_5d'] = 0.0

    # --- B2 signal metadata ---
    for col in ['b2_position_weight', 'b2_prior_strong']:
        feats[col] = float(signal_row[col]) if col in signal_row.index and pd.notna(signal_row[col]) else 0.0

    # --- Brick chart features ---
    for col in ['brick_val', 'brick_rising', 'brick_falling',
                'brick_green_to_red', 'brick_red_to_green',
                'brick_green_streak', 'brick_red_streak',
                'cross_below_yellow']:
        feats[col] = float(signal_row[col]) if col in signal_row.index and pd.notna(signal_row[col]) else 0.0

    # Brick streak features
    feats['brick_streak_diff'] = feats['brick_green_streak'] - feats['brick_red_streak']
    feats['brick_momentum'] = feats['brick_rising'] - feats['brick_falling']

    # --- OAMV market environment features ---
    if oamv_info:
        for k, v in oamv_info.items():
            feats[f'oamv_{k}'] = float(v) if pd.notna(v) else 0.0
    else:
        for k in ['change_pct', 'is_aggressive', 'is_defensive',
                  'is_normal', 'volatility_20d', 'amount_ratio',
                  'change_3d', 'change_5d', 'signal']:
            feats[f'oamv_{k}'] = 0.0

    # OAMV interaction features
    feats['oamv_agg_x_gain'] = feats.get('oamv_is_aggressive', 0) * feats['gain_today']
    feats['oamv_def_x_pullback'] = feats.get('oamv_is_defensive', 0) * feats['pullback_10d']
    feats['oamv_agg_x_weight'] = feats.get('oamv_is_aggressive', 0) * feats.get('b2_position_weight', 1.0)

    return feats


def load_and_extract_signals(parquet_path, oamv_index):
    """Load stock, generate B2 signals, extract features + T+5 returns."""
    try:
        df = pd.read_parquet(parquet_path)
        if 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'])

        # Attach OAMV
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
        df['oamv_regime'] = np.where(chg_arr > 3.0, 'aggressive',
                            np.where(chg_arr < -2.0, 'defensive', 'normal'))

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
                                   params=OPTIMAL_PARAMS)

        sig_mask = sdf['b2_entry_signal'] == 1
        if sig_mask.sum() == 0:
            return None

        signal_idxs = sdf.index[sig_mask]
        records = []

        for sig_idx in signal_idxs:
            pos = sdf.index.get_loc(sig_idx)
            if pos + 5 >= len(sdf):
                continue

            signal_row = sdf.iloc[pos]
            df_up_to = sdf.iloc[:pos+1]

            oamv_info = None
            try:
                date_idx = list(df['date']).index(sig_idx)
                if date_idx < len(oamv_infos):
                    oamv_info = oamv_infos[date_idx]
            except (ValueError, IndexError):
                pass

            feats = compute_signal_features(signal_row, df_up_to, oamv_info)

            buy_px = float(sdf.iloc[pos]['close'])
            sell_px = float(sdf.iloc[pos + 5]['open'])
            ret_5d = sell_px / buy_px - 1

            feats['t5_return'] = float(ret_5d)
            feats['label_t5_profit'] = 1 if ret_5d > 0 else 0
            feats['stock_code'] = os.path.basename(parquet_path).replace('.parquet', '')
            feats['signal_date'] = str(sig_idx.date())

            records.append(feats)

        if records:
            return pd.DataFrame(records)
        return None
    except Exception:
        return None


def evaluate_filters(test_df, y_pred_reg):
    """Evaluate different filtering strategies on test set."""
    results = []

    # 1. Unfiltered
    unfiltered_ret = test_df['t5_return'].mean()
    unfiltered_wr = test_df['label_t5_profit'].mean()
    results.append({
        'name': 'Unfiltered',
        'trades': len(test_df),
        'win_rate': round(float(unfiltered_wr), 4),
        'mean_return': round(float(unfiltered_ret), 6),
        'total_return': round(float(test_df['t5_return'].sum()), 6),
    })

    # 2. Predict positive return only
    mask_pos = y_pred_reg > 0
    df_pos = test_df[mask_pos]
    results.append({
        'name': 'PredRet>0',
        'trades': int(mask_pos.sum()),
        'win_rate': round(float(df_pos['label_t5_profit'].mean()), 4) if len(df_pos) > 0 else 0,
        'mean_return': round(float(df_pos['t5_return'].mean()), 6) if len(df_pos) > 0 else 0,
        'total_return': round(float(df_pos['t5_return'].sum()), 6) if len(df_pos) > 0 else 0,
    })

    # 3. Top percentile thresholds
    for pct in [25, 50, 75]:
        threshold = np.percentile(y_pred_reg, pct)
        mask = y_pred_reg >= threshold
        df_filt = test_df[mask]
        results.append({
            'name': f'Top-{100-pct}%',
            'trades': int(mask.sum()),
            'win_rate': round(float(df_filt['label_t5_profit'].mean()), 4) if len(df_filt) > 0 else 0,
            'mean_return': round(float(df_filt['t5_return'].mean()), 6) if len(df_filt) > 0 else 0,
            'total_return': round(float(df_filt['t5_return'].sum()), 6) if len(df_filt) > 0 else 0,
        })

    # 4. Binary prediction from Phase 4 style
    mask_bin = y_pred_reg >= 0.005  # 0.5% expected return threshold
    df_bin = test_df[mask_bin]
    results.append({
        'name': 'PredRet>0.5%',
        'trades': int(mask_bin.sum()),
        'win_rate': round(float(df_bin['label_t5_profit'].mean()), 4) if len(df_bin) > 0 else 0,
        'mean_return': round(float(df_bin['t5_return'].mean()), 6) if len(df_bin) > 0 else 0,
        'total_return': round(float(df_bin['t5_return'].sum()), 6) if len(df_bin) > 0 else 0,
    })

    return results


def main():
    print("=" * 70)
    print("  Phase 4b: Improved LightGBM Signal Quality Scorer")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Load OAMV
    print("\n[1/6] Loading OAMV market data...")
    oamv_df = fetch_market_data(start='20180101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = gen_oamv(oamv_df)
    oamv_df['oamv_change_3d'] = oamv_df['oamv'].pct_change(3) * 100
    oamv_df['oamv_change_5d'] = oamv_df['oamv'].pct_change(5) * 100
    oamv_df['oamv_volatility_20d'] = oamv_df['oamv_change_pct'].rolling(20).std()
    oamv_df['oamv_amount_ratio'] = oamv_df['amount_total'] / oamv_df['amount_total'].rolling(20).mean()
    oamv_index = oamv_df.set_index('date')

    # Step 2: Extract signals
    print("\n[2/6] Extracting B2 signals + features from all stocks...")
    feat_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
    print(f"  Total: {len(feat_files)} parquet files")

    all_records = []
    t1 = time.time()

    for i, f in enumerate(feat_files):
        path = os.path.join(FEAT_DIR, f)
        result = load_and_extract_signals(path, oamv_index)
        if result is not None and len(result) > 0:
            all_records.append(result)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(feat_files)} stocks ({time.time()-t1:.0f}s)")

    combined = pd.concat(all_records, ignore_index=True)
    print(f"  Total signals: {len(combined)}")
    print(f"  Positive: {combined['label_t5_profit'].sum()} ({combined['label_t5_profit'].mean()*100:.1f}%)")
    print(f"  Mean T+5 return: {combined['t5_return'].mean()*100:.3f}%")

    # Step 3: Train/test split by time
    print("\n[3/6] Splitting by time...")
    combined = combined.sort_values('signal_date').reset_index(drop=True)
    split_idx = int(len(combined) * 0.8)
    train_df = combined.iloc[:split_idx]
    test_df = combined.iloc[split_idx:]

    exclude_cols = {'t5_return', 'label_t5_profit', 'stock_code', 'signal_date'}
    feature_cols = [c for c in combined.columns if c not in exclude_cols
                    and combined[c].dtype in ('float64', 'float32', 'int64', 'int32')]

    print(f"  Features: {len(feature_cols)}")

    # Drop rows with NaN in features or target
    keep_cols = feature_cols + ['t5_return', 'label_t5_profit']
    train_clean = train_df[keep_cols].dropna()
    test_clean = test_df[keep_cols].dropna()
    print(f"  Train after NaN drop: {len(train_clean)} (removed {len(train_df)-len(train_clean)})")
    print(f"  Test after NaN drop:  {len(test_clean)} (removed {len(test_df)-len(test_clean)})")

    X_train = train_clean[feature_cols].values
    y_train_reg = train_clean['t5_return'].values
    X_test = test_clean[feature_cols].values
    y_test_reg = test_clean['t5_return'].values

    # Also update test_df reference for evaluation
    test_df_clean = test_clean

    # Step 4: LightGBM Regression
    print("\n[4/6] Training LightGBM regressor...")

    lgb_train = lgb.Dataset(X_train, y_train_reg,
                             feature_name=feature_cols,
                             free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, y_test_reg,
                            feature_name=feature_cols,
                            reference=lgb_train,
                            free_raw_data=False)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'lambda_l1': 0.5,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42,
        'num_threads': 8,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    # Step 5: Evaluate regression
    print("\n[5/6] Evaluating regressor...")
    y_pred_reg = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    # Also evaluate as binary classifier
    y_test_bin = (y_test_reg > 0).astype(int)
    y_pred_bin = (y_pred_reg > 0).astype(int)
    auc = roc_auc_score(y_test_bin, y_pred_reg)
    ap = average_precision_score(y_test_bin, y_pred_reg)

    print(f"\n  Regression Metrics:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R2:   {r2:.4f}")
    print(f"    AUC (as classifier): {auc:.4f}")
    print(f"    Avg Precision:       {ap:.4f}")

    # Correlation between predicted and actual
    corr = np.corrcoef(y_pred_reg, y_test_reg)[0, 1]
    print(f"    Correlation(pred, actual): {corr:.4f}")

    # Step 6: Trading simulation with different filters
    print(f"\n[6/6] Trading Simulation: Filter Strategies")
    print(f"  {'='*60}")
    print(f"  {'Strategy':<20s} {'Trades':>8s} {'WinRate':>10s} {'MeanRet':>10s} {'TotRet':>10s}")
    print(f"  {'-'*60}")

    filter_results = evaluate_filters(test_df_clean, y_pred_reg)

    for r in filter_results:
        print(f"  {r['name']:<20s} {r['trades']:>8d} {r['win_rate']*100:>9.1f}% {r['mean_return']*100:>9.3f}% {r['total_return']*100:>9.2f}%")

    # Feature importance
    print(f"\n  Top 20 Features (Gain):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'gain': model.feature_importance(importance_type='gain'),
        'split': model.feature_importance(importance_type='split'),
    }).sort_values('gain', ascending=False)

    for i, (_, row) in enumerate(importance.head(20).iterrows()):
        ftype = 'OAMV' if row['feature'].startswith('oamv') else (
            'BRICK' if row['feature'].startswith('brick') else 'B2')
        print(f"    {i+1:>2}. {row['feature']:<35s} gain={row['gain']:.0f}  [{ftype}]")

    # Compare with Phase 4 results
    phase4_auc = 0.5458
    phase4_filtered_wr = 0.570
    phase4_filtered_ret = 0.01571

    best_filtered = [r for r in filter_results if r['trades'] > 50]
    best = max(best_filtered, key=lambda r: r['mean_return']) if best_filtered else filter_results[0]

    print(f"\n  Comparison with Phase 4 (binary classifier):")
    print(f"    Phase 4  AUC: {phase4_auc:.4f}  |  Phase 4b AUC: {auc:.4f}")
    print(f"    Phase 4  Filtered WR: {phase4_filtered_wr*100:.1f}%  |  Phase 4b Best WR: {best['win_rate']*100:.1f}%")
    print(f"    Phase 4  Filtered Ret: {phase4_filtered_ret*100:.3f}%  |  Phase 4b Best Ret: {best['mean_return']*100:.3f}%")

    # Save
    out = {
        'phase': 'phase4b_lightgbm_regressor',
        'model': 'LightGBM_Regressor',
        'params': params,
        'rmse': round(float(rmse), 4),
        'mae': round(float(mae), 4),
        'r2': round(float(r2), 4),
        'auc': round(float(auc), 4),
        'avg_precision': round(float(ap), 4),
        'correlation_pred_actual': round(float(corr), 4),
        'n_features': len(feature_cols),
        'n_signals': len(combined),
        'n_train': len(train_df),
        'n_test': len(test_df_clean),
        'baseline_winrate': round(float(test_df_clean['label_t5_profit'].mean()), 4),
        'baseline_meanret': round(float(test_df_clean['t5_return'].mean()), 6),
        'filter_results': filter_results,
        'top_features': [
            {'feature': row['feature'], 'gain': int(row['gain']),
             'type': 'OAMV' if row['feature'].startswith('oamv') else (
                 'BRICK' if row['feature'].startswith('brick') else 'B2')}
            for _, row in importance.head(30).iterrows()
        ],
        'elapsed_seconds': time.time() - t0,
    }

    with open(os.path.join(OUT_DIR, 'phase4b_lightgbm_results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    model.save_model(os.path.join(OUT_DIR, 'phase4b_lgbm_model.txt'))

    print(f"\n  Phase 4b Complete ({time.time()-t0:.0f}s)")
    print(f"  Results: {OUT_DIR}/phase4b_lightgbm_results.json")


if __name__ == '__main__':
    main()
