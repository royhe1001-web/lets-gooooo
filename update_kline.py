#!/usr/bin/env python3
"""Spring B2 K线增量更新 — 更新 parquet 文件到最新日期"""
import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = Path(__file__).parent
FEATURES_DIR = BASE / 'ML_optimization' / 'features'

# tushare token
TS_TOKEN = 'c2a96ee78a444f422063ba90ed4460a8f252f3fd136ea893f6c310ae'

def get_tushare_pro():
    import tushare as ts
    ts.set_token(TS_TOKEN)
    return ts.pro_api()


def get_min_date():
    """检查所有 parquet 文件中的最旧日期(用于判断是否需要更新)"""
    files = list(FEATURES_DIR.glob('*.parquet'))
    if not files:
        return None
    min_date = None
    checked = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['date'])
            d = pd.to_datetime(df['date']).max()
            if min_date is None or d < min_date:
                min_date = d
            checked += 1
        except Exception:
            pass
    print(f'  抽样检查: {checked}/{len(files)} 文件, 最早日期: {min_date.date() if min_date else "?"}')
    return min_date


def update_one_parquet(args):
    """更新单只股票 parquet (含重试)"""
    path, pro, start_date, end_date, base_idx = args
    code = path.stem
    ts_code = f'{code}.SH' if code.startswith(('60','68','9')) else f'{code}.SZ'

    for attempt in range(3):
        try:
            time.sleep(0.02 * (base_idx % 50))
            new = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                            fields='trade_date,open,high,low,close,vol,amount')
            break
        except Exception:
            if attempt < 2:
                time.sleep(1.0)
            else:
                return (code, -1)
    else:
        return (code, -1)

    try:
        if new is None or len(new) == 0:
            return (code, 0)

        new = new.rename(columns={'trade_date': 'date', 'vol': 'volume'})
        new['date'] = pd.to_datetime(new['date'])
        new = new.sort_values('date')

        # 读取现有数据
        old = pd.read_parquet(path)
        old['date'] = pd.to_datetime(old['date'])

        # 合并: 新行追加, 已有日期跳过
        existing_dates = set(old['date'])
        new_rows = new[~new['date'].isin(existing_dates)]
        if len(new_rows) == 0:
            return (code, 0)

        # 只保留 OHLCV 列, 指标列留空 (下次全量计算时填充)
        for col in new_rows.columns:
            if col not in old.columns:
                old[col] = np.nan
        for col in old.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan

        merged = pd.concat([old, new_rows], ignore_index=True).sort_values('date')
        merged.to_parquet(path, index=False)
        return (code, len(new_rows))
    except Exception as e:
        return (code, -1)


def main():
    print(f'Spring B2 K线更新 ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    print(f'数据目录: {FEATURES_DIR}')

    min_date = get_min_date()
    if min_date is None:
        print('未找到 parquet 文件')
        return

    print(f'最早的"最新日期": {min_date.date()}')

    today = datetime.now()
    if min_date.date() >= today.date():
        print('所有文件已是最新, 无需更新')
        return

    # 只下载最近10个交易日(已覆盖增量需求)
    start = (today - pd.Timedelta(days=10)).strftime('%Y%m%d')
    end = today.strftime('%Y%m%d')
    print(f'下载区间: {start} → {end}')

    pro = get_tushare_pro()
    files = list(FEATURES_DIR.glob('*.parquet'))
    print(f'待更新: {len(files)} 只')

    t0 = time.time()
    args = [(f, pro, start, end, i) for i, f in enumerate(files)]
    added = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(update_one_parquet, a): a[0].stem for a in args}
        for i, fut in enumerate(as_completed(futures)):
            code, n = fut.result()
            if n > 0:
                added += n
            elif n < 0:
                errors += 1
            if (i + 1) % 500 == 0:
                print(f'  进度: {i+1}/{len(files)} (+{added}行, {errors}错)')

    elapsed = time.time() - t0
    print(f'\n完成: +{added} 行, {errors} 错 ({elapsed:.0f}s)')
    if added > 0:
        new_min = get_min_date()
        print(f'新最早日期: {new_min.date() if new_min else "?"}')


if __name__ == '__main__':
    main()
