#!/usr/bin/env python3
"""K线增量更新 — mootdx拉取 + 仅重算新行指标 (progress bar)"""
import sys, os, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).parent
FEATURES_DIR = BASE / 'ML_optimization' / 'features'
sys.path.insert(0, str(BASE))


def update_one(args):
    """更新单只股票: 下载新OHLCV + 重算新行指标"""
    path, client, start_d, end_d = args
    code = path.stem
    try:
        # 1. Download via mootdx
        raw = client.bars(symbol=code, frequency=9, start=0, offset=30)
        if raw is None or raw.empty:
            return (code, 0, 0)
        raw = raw.reset_index()
        raw['date'] = pd.to_datetime(raw['datetime'])
        new = raw[(raw['date'] >= start_d) & (raw['date'] <= end_d)]
        if new.empty:
            return (code, 0, 0)

        # 2. Read existing
        old = pd.read_parquet(path)
        old['date'] = pd.to_datetime(old['date'])

        # 3. Only add truly new dates
        existing_dates = set(old['date'].dropna())
        new = new[~new['date'].isin(existing_dates)].copy()
        if new.empty:
            return (code, 0, 0)

        # 4. Build new rows with OHLCV only
        new_rows = pd.DataFrame({
            'date': new['date'],
            'open': new['open'].astype(float),
            'high': new['high'].astype(float),
            'low': new['low'].astype(float),
            'close': new['close'].astype(float),
            'volume': new.get('vol', new.get('volume', 0)).astype(float),
        })
        n_new = len(new_rows)

        # 5. Align columns (new rows get NaN for indicator columns)
        for col in old.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan
        for col in new_rows.columns:
            if col not in old.columns:
                old[col] = np.nan

        merged = pd.concat([old, new_rows], ignore_index=True).sort_values('date')

        # 6. Recompute indicators for new rows + enough history for rolling windows
        from indicators import calc_all_indicators
        need_recompute = merged[merged['close'].notna()]  # rows with valid OHLCV
        if 'J' in merged.columns:
            # Only recompute if new rows have NaN J
            nan_idx = merged[merged['J'].isna()].index
            if len(nan_idx) == 0:
                merged.to_parquet(path, index=False)
                return (code, n_new, 0)
            # Take window: from 150 rows before first NaN to end
            first_nan = nan_idx[0]
            start_i = max(0, first_nan - 150)
            sub = merged.iloc[start_i:].copy()
            result = calc_all_indicators(sub, board_type='main')
            # Only write indicator columns back for the NaN rows (preserve old values)
            for col in result.columns:
                if col in merged.columns and col != 'date':
                    merged.loc[nan_idx, col] = result.loc[nan_idx, col].values
        else:
            # No indicator columns: just recalc all
            sub = merged.iloc[max(0, len(merged) - 150):].copy()
            result = calc_all_indicators(sub, board_type='main')
            for col in result.columns:
                if col not in merged.columns:
                    merged[col] = np.nan
                merged.loc[sub.index, col] = result[col].values

        merged.to_parquet(path, index=False)
        return (code, n_new, 0)
    except Exception:
        return (code, 0, -1)


def main():
    from mootdx.quotes import Quotes

    print('K线增量更新 (mootdx)')
    print(f'数据目录: {FEATURES_DIR}')

    files = sorted(FEATURES_DIR.glob('*.parquet'))
    if not files:
        print('未找到 parquet 文件'); return

    # Find actual min date
    min_date = None
    for f in files[:100]:
        try:
            df = pd.read_parquet(f, columns=['date'])
            d = pd.to_datetime(df['date']).max()
            if min_date is None or d < min_date:
                min_date = d
        except Exception:
            pass
    if min_date is None: min_date = pd.Timestamp('2026-05-07')
    print(f'数据到: {min_date.date()} | 共 {len(files)} 文件')

    import sys as _sys
    force_recalc = '--recalc' in _sys.argv

    today = pd.Timestamp.now()
    if min_date.date() >= today.date() and not force_recalc:
        print('数据已最新 (--recalc 强制重算指标)'); return

    # Recalc-only: Mac端暂禁用 (指标计算与Win端不一致)
    # 如需重算NaN指标, 由Win端推送已算好的parquet文件
    if force_recalc:
        print('--recalc: Mac端暂不支持指标重算, 请由Win端推送parquet文件')
        return

    start_d = min_date + pd.Timedelta(days=1)
    end_d = today
    print(f'下载: {start_d.date()} ~ {end_d.date()}')

    client = Quotes.factory(market='std')
    t0 = time.time()
    added = 0
    errors = 0
    total = len(files)

    args = [(f, client, start_d, end_d) for f in files]
    last_pct = -1
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(update_one, a): a[0].stem for a in args}
        for i, fut in enumerate(as_completed(futures)):
            code, n, err = fut.result()
            added += n
            errors += err
            pct = (i + 1) * 100 // total
            if pct > last_pct and pct % 5 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                bar = '=' * (pct // 5) + '>' + ' ' * (20 - pct // 5)
                print(f'  [{bar}] {pct}% ({i+1}/{total}) +{added}行 {errors}错 ETA~{eta:.0f}s',
                      flush=True)
                last_pct = pct

    elapsed = time.time() - t0
    print(f'\n完成: +{added}行 {errors}错 ({elapsed:.0f}s)')

    # Verify
    df = pd.read_parquet(FEATURES_DIR / '600519.parquet')
    df['date'] = pd.to_datetime(df['date'])
    last = df['date'].max()
    print(f'600519 最新日期: {last.date()}')

    # Check indicator NaN
    if 'J' in df.columns:
        nan_count = df[df['date'] >= start_d]['J'].isna().sum()
        print(f'新行 J字段 NaN: {nan_count}')


if __name__ == '__main__':
    main()
