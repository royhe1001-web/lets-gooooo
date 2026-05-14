# BUG: K线更新后指标列为NaN，回测无法生成新交易

**发现**: 2026-05-14 Mac端  
**严重程度**: 中 — update_kline.py 可更新OHLCV，但回测仍停在5/7

---

## 现象

`update_kline.py` 成功将 parquet 文件日期更新到 2026-05-14，但 `run_backtest_2026.py` 仍只生成 82 条交易记录（停在 5/7），5/8~5/14 零交易。

## 根因

parquet 文件包含 98 列：OHLCV（5列）+ 技术指标（KDJ/MACD/brick/白黄线等 90+列）。`update_kline.py` 只追加 OHLCV 行，指标列留 NaN。策略引擎读取 parquet 时发现指标为 NaN，无法生成入场信号。

```
000001.parquet 最后几行:
  date       open  close  J    MACD_HIST  brick_val  ...
  2026-05-07 11.37 11.37  66.1 0.058      81.3       ← 正常
  2026-05-08 11.30 11.30  NaN  NaN        NaN        ← 无指标
  2026-05-11 11.28 11.28  NaN  NaN        NaN
  ...
```

## 修复建议

1. Win 端提供 `recompute_indicators.py` — 遍历所有 parquet，对 NaN 行重算 90+ 个技术指标
2. 或将指标计算逻辑从 `phase2c_*.py` 提取为独立函数，供增量更新调用
3. 短期方案：Win 端直接 push 已重算的最新 parquet 文件
