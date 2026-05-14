# BUG: Parquet 重复日期 + 复权不一致 — 1411只股票受影响

**发现**: 2026-05-15 Mac端回测验证  
**严重程度**: 致命 — 存量前复权价与增量不复权价并存，回测收益虚高至 +775%（真实 +32.6%）

---

## 根因

`update_kline.py` 增量更新有两个 bug：

### Bug 1: 日期未归一化（第 27 行）

```python
raw['date'] = pd.to_datetime(raw['datetime'])   # 缺 .dt.normalize()
```

mootdx 返回的 datetime 带 `15:00:00`，存量 parquet 日期为 `00:00:00`。第 37 行 `existing_dates` 的 `isin()` 判定中两者不相等：

```python
# pd.Timestamp('2026-04-23 00:00:00') != pd.Timestamp('2026-04-23 15:00:00')
```

新数据被视为"新日期"未被过滤，concat 后同一天出现两行。

对比：`fetch_kline.py:201` 已正确使用 `.dt.normalize()`。

### Bug 2: 未指定前复权（第 23 行）

```python
raw = client.bars(symbol=code, frequency=9, start=0, offset=30)   # 缺 adjust='qfq'
```

mootdx 默认不复权，而存量 parquet 是 qfq 前复权。以 000070（特发信息）为例：

| 日期 | 来源 | 价格 | 复权方式 |
|------|------|------|---------|
| 2026-04-23 00:00:00 | 存量 | open=2.20 close=2.17 | qfq 前复权 |
| 2026-04-23 15:00:00 | 增量 | open=21.01 close=20.99 | 不复权 |

对比：`fetch_kline.py:265` 已正确传入 `adjust='qfq'`。

---

## 影响

- **1411 只股票** parquet 存在重复日期（4/23 ~ 5/14，共 13 个交易日）
- 回测引擎对 000070 以 2.27（qfq）买入、20.11（不复权）卖出，单票虚增 +786%
- 回测总收益从真实 +32.6% 虚高至 +775.5%

---

## Mac端临时修复

1. `phase2c_bull_grid_search.py` preload: `.dt.normalize()` + `drop_duplicates`
2. `phase2c_oamv_grid_search.py` run(): OAMV/signal 日期统一 normalize
3. `phase2c_oamv_grid_search.py` sell(): 部分卖整百取整

**这些是治标**，修复后回测回到真实的 +32.6%。

---

## Win端需修复

1. `update_kline.py:27` → `pd.to_datetime(raw['datetime']).dt.normalize()`
2. `update_kline.py:23` → `client.bars(..., adjust='qfq')`
3. 对已损坏的 2125 个 parquet 批量修复：`.dt.normalize()` + `drop_duplicates('date', keep='first')` 保留存量 qfq 数据
4. 修复后重新运行回测，验证收益是否与 Mac端 +32.6% 一致
