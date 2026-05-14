# BUG: 实时信号三个问题

**发现**: 2026-05-14 Mac端测试  
**当前运行结果**: aggressive状态, 买入3只 (003018, 603958, 002214), 总耗时 ~37s

---

## 问题1: vs昨收显示 0%

**现象**: 所有买入信号都显示 `(vs昨收+0.0%)`

**根因**: `_load_yesterday_close` 读取 `output/kline/*.csv`，但 spring_b2 的 K线数据存储在 parquet 文件中（`stock_data/` 目录或 ML_optimization 使用的目录）。路径不匹配导致永远找不到昨收价。

**影响**: 买入信号缺少和昨收价的对比，无法判断是否追高

---

## 问题2: ST 股票未过滤

**现象**: 推荐买入 `002214 *ST大立`（ST股交易规则不同，T+0 5%涨跌停，流动性差）

**根因**: `get_stock_universe` 未过滤 ST/退市风险股票

**建议**: 加 `stock_basic` 检查或名称过滤 `*ST`

---

## 问题3: 实时行情融合耗时 ~4s

**现象**: 日志出现两行 `K线: 2113 只 (8.1s)` 和 `K线: 2113 只 (12.6s)`

**根因**: 不是加载两次。第一次是并行读取 parquet（8s），第二次是本函数总耗时（含融合实时行情）。`load_kline_with_realtime` 中的融合循环是串行的——2113 只逐个 `pd.concat` + `sort_index`，多耗 ~4-5s。

**建议**: 用字典更新代替 concat（`df.loc[today] = row` 比 `pd.concat` 快很多），或批量 concat
