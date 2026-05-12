#!/usr/bin/env python3
"""tushare 批量下载A股全市场日K线，多线程"""
import argparse, os, sys, time, random, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import tushare as ts
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "fetch_kline" / "stock_kline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_ts")

TS_TOKEN = "c2a96ee78a444f422063ba90ed4460a8f252f3fd136ea893f6c310ae"
ts.set_token(TS_TOKEN)
pro = ts.pro_api()


def get_stock_list():
    """获取全市场A股列表（含创业板、科创板）"""
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,symbol,name")
    logger.info("全市场A股: %d 只", len(df))

    local = set(f.stem for f in OUT_DIR.glob("*.csv"))
    df["have"] = df["symbol"].apply(lambda c: c in local)
    todo = df[~df["have"]]
    logger.info("本地已有: %d, 待下载: %d", df["have"].sum(), len(todo))
    return todo


def download_one(symbol):
    """下载单只股票全部历史日K线"""
    ts_code = (f"{symbol}.SH" if symbol.startswith(("60", "68", "9"))
               else f"{symbol}.BJ" if symbol[:2] in ("92", "83", "87") or symbol[:3] == "43"
               else f"{symbol}.SZ")
    for attempt in range(1, 4):
        try:
            df = pro.daily(ts_code=ts_code, start_date="20000101",
                           end_date="20260510",
                           fields="trade_date,open,high,low,close,vol,amount")
            if df is None or df.empty:
                return symbol, 0
            df = df.rename(columns={"trade_date": "date", "vol": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df.to_csv(OUT_DIR / f"{symbol}.csv", index=False)
            return symbol, len(df)
        except Exception as e:
            time.sleep(random.uniform(0.5, 1.5) * attempt)
    return symbol, -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    todo = get_stock_list()
    if len(todo) == 0:
        logger.info("数据已是最新")
        return

    symbols = todo["symbol"].tolist()
    logger.info("开始下载 %d 只, workers=%d", len(symbols), args.workers)

    ok = fail = 0
    total_rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, s): s for s in symbols}
        for f in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            sym, rows = f.result()
            if rows > 0:
                ok += 1
                total_rows += rows
            elif rows == 0:
                ok += 1
            else:
                fail += 1

    logger.info("完成: %d OK, %d 失败, %d 行数据", ok, fail, total_rows)


if __name__ == "__main__":
    main()
