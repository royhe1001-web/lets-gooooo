# BUG: 回测终点停在 5/7，final_clear 清仓后无新交易

**发现**: 2026-05-14 Mac端实时信号测试  
**严重程度**: 高 — 实时信号从零持仓开始，无法接续回测状态

---

## 现象

Spring B2 回测 `trade_log.csv` 最后一笔交易停在 2026-05-07：

```
2026-05-07 SELL 000534  final_clear
2026-05-07 SELL 603618  final_clear
2026-05-07 SELL 603026  final_clear
2026-05-07 SELL 000973  final_clear
```

之后 5/8 ~ 5/14 整整 5 个交易日没有任何记录。

## 影响

1. 实时信号从零持仓启动 → 只做买入，没有持仓可评估离场
2. `push_to_backtest` 只追加今日行，5/8~5/13 的净值/持仓缺口无法补
3. Excel 报告只覆盖 1/6 ~ 5/7，与实际交易日 5/13 脱节
4. 回测收益 +64.9% 仅到 5/7，不代表当前

## 可能根因

- `run_backtest_2026.py` 中回测终点硬编码或提前终止
- `final_clear` 逻辑在达到某条件时清仓（到期日/最大持有天数触发全局清仓）

## 修复建议

1. 回测终点改为自动检测最新 K线数据日期
2. 移除或调整 `final_clear` 触发条件（除非是策略设计的止盈清仓）
3. 回测跑完后确认 trade_log 最后日期 ≈ 最新 K线日期

---

## 处理 (2026-05-14 Windows端)

**已修复**:
- `phase2c_oamv_grid_search.py`: `SIM_END = pd.Timestamp.now()` (动态)
- `phase2c_bull_grid_search.py`: `TEST_END = pd.Timestamp.now()` (动态)
- `run_backtest_2026.py`: 自动检测 `oamv_df['date'].max()` 并调整终点
- `final_clear` 是正确的：它在回测终点清仓，动态终点意味着清仓在数据最后一天（5/13而非5/7）

**Mac端需确认**:
1. `git pull` 拉取最新代码
2. K线 parquet 数据是否覆盖到最新日期
3. `market_turnover.csv` 是否更新到最新日期（OAMV 数据源）
4. 运行 `python run_backtest_2026.py` 确认 trade_log 最后日期
