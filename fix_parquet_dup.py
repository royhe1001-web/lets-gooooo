#!/usr/bin/env python3
"""批量修复 parquet 重复日期 + 复权不一致 — 1411只股票

根因: update_kline.py 两处缺陷
  1. 日期未 normalize → 同日 00:00:00 / 15:00:00 被视为不同行
  2. 未指定 adjust='qfq' → 存量前复权价与增量不复权价并存

修复: normalize + drop_duplicates('date', keep='first') 保留存量 qfq 数据
"""
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE = Path(__file__).parent
FEATURES_DIR = BASE / 'ML_optimization' / 'features'


def fix_one(path: Path) -> tuple:
    try:
        df = pd.read_parquet(path)
        old_len = len(df)

        # Normalize dates
        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        # Drop duplicates, keep first (qfq data was written first)
        dup_mask = df.duplicated(subset='date', keep='first')
        n_dup = dup_mask.sum()
        if n_dup > 0:
            df = df[~dup_mask].copy()

        # Verify no price spikes from mixed adjustment
        if 'close' in df.columns and n_dup > 0:
            df = df.sort_values('date')
            pct_chg = df['close'].pct_change().abs()
            spikes = (pct_chg > 2.0).sum()  # >200% daily change = suspicious
        else:
            spikes = 0

        df.to_parquet(path, index=False)
        return (path.stem, old_len, len(df), n_dup, spikes)
    except Exception as e:
        return (path.stem, 0, 0, 0, -1)


def main():
    files = sorted(FEATURES_DIR.glob('*.parquet'))
    if not files:
        print('未找到 parquet 文件'); return

    print(f'修复 {len(files)} 个 parquet 文件...')
    total_dup = 0
    total_spikes = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fix_one, f): f.stem for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            code, old_len, new_len, n_dup, spikes = fut.result()
            if spikes < 0:
                errors += 1
            else:
                total_dup += n_dup
                total_spikes += spikes

    print(f'完成: {total_dup} 个重复行已移除, {total_spikes} 个疑似价格跳变, {errors} 个错误')

    # Quick verify on a known stock
    test_path = FEATURES_DIR / '000070.parquet'
    if test_path.exists():
        df = pd.read_parquet(test_path)
        df['date'] = pd.to_datetime(df['date'])
        dup = df.duplicated(subset='date').sum()
        last = df['date'].max()
        print(f'验证 000070: {len(df)} 行, 重复 {dup}, 最新 {last.date()}')


if __name__ == '__main__':
    main()
