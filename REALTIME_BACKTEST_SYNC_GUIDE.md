# 实时信号 ↔ 回测引擎桥接指南

> 来源: dualstream_csi500 策略已验证方案  
> 目标: spring_b2 实现同样能力 — 14:50 跑一次，信号+回测同步完成

---

## 问题

实时信号脚本和回测引擎各自独立运行：
- 实时信号从 `current_positions.json` 读持仓（首次为空，从零开始）
- 回测持仓在 `daily_positions.csv` / `trade_log.csv` 中
- 两者互不相通 → 实时信号和回测的调仓逻辑不一致

## 方案

三步改动，让实时信号脚本成为回测引擎的"实时驱动层"。

### Step 1: 回测输出 trade_log.csv

回测引擎的 `trade_log` DataFrame 包含每笔交易的 `entry_date` 和 `entry_price`，是持仓重建的关键。默认只输出到 `daily_positions.csv`（无 entry_price），需要额外保存 `trade_log.csv`。

```python
# 在 generate_reports 函数中，daily_positions 生成前加一行
trade_log.to_csv(REPORT_DIR / 'trade_log.csv', index=False)
```

### Step 2: 实时信号脚本加载回测持仓

原本只读 `current_positions.json`，改为优先该文件（有则用），为空则回退到回测持仓。

```python
def load_backtest_positions():
    """从回测trade_log重建最新持仓，含entry_date和entry_price"""
    tl = pd.read_csv('backtest_output/report_2026/trade_log.csv', parse_dates=['date'])
    held = {}
    for _, row in tl.iterrows():
        sym = str(row['symbol']).zfill(6)  # 补齐6位！trade_log存int
        if row['action'] == 'BUY':
            if sym not in held:
                held[sym] = {
                    'shares': 0,
                    'entry_date': row['entry_date'],
                    'entry_price': row['entry_price'],
                    'side': str(row.get('side', 'base')),
                }
            held[sym]['shares'] += int(row['shares'])
        else:
            if sym in held:
                held[sym]['shares'] -= int(row['shares'])
    positions = []
    for sym, info in held.items():
        if info['shares'] > 0:
            info['symbol'] = sym
            positions.append(info)
    return pd.DataFrame(positions)


def load_positions():
    """优先实时持仓，为空则回退到回测持仓"""
    pos_file = Path('output/signals/current_positions.json')
    if pos_file.exists():
        data = json.load(open(pos_file))
        if data:
            return pd.DataFrame(data)
    return load_backtest_positions()
```

### Step 3: --execute 时把今日结果推回回测 CSV

信号生成后，用 `positions_after` 追加今日行到 `daily_positions.csv` 和 `daily_nav.csv`。

```python
def push_to_backtest(signals, positions_after):
    report_dir = Path('backtest_output/report_2026')
    today_str = str(datetime.now().date())

    # --- daily_positions.csv ---
    existing = pd.read_csv(report_dir / 'daily_positions.csv')
    existing = existing[existing['date'] != today_str]  # 去重
    new_rows = []
    for _, pos in positions_after.iterrows():
        new_rows.append({
            'date': today_str,
            'symbol': pos['symbol'],
            'shares': int(pos['shares']),
            'side': pos.get('side', '?'),
        })
    pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True
             ).to_csv(report_dir / 'daily_positions.csv', index=False)

    # --- daily_nav.csv (预估) ---
    nav = pd.read_csv(report_dir / 'daily_nav.csv')
    if today_str not in nav['date'].astype(str).values:
        last = nav.iloc[-1]
        pos_val = (positions_after['shares'].astype(float) * positions_after['price']).sum()
        eq = last['cash'] + pos_val
        strat_ret = (eq / last['total_equity'] - 1) if last['total_equity'] > 0 else 0
        new_row = {
            'strategy_nav': last['strategy_nav'] * (1 + strat_ret),
            'csi500_nav': last['csi500_nav'],  # 盘中无法获取精确基准收益
            'strategy_daily_ret': strat_ret,
            'csi500_daily_ret': 0.0,
            'daily_excess': strat_ret,
            'n_positions': len(positions_after),
            'cash': last['cash'],
            'position_value': pos_val,
            'total_equity': eq,
            'oamv_state': signals['state'],
            'oamv_divergence': 'realtime_preview',
            'date': today_str,
        }
        pd.concat([nav, pd.DataFrame([new_row])], ignore_index=True
                 ).to_csv(report_dir / 'daily_nav.csv', index=False)
```

## 完整工作流

```
14:50  python real_time_signal.py --execute
       │
       ├─ 加载回测最新持仓 (trade_log.csv)
       ├─ 拉取实时行情 (新浪API)
       ├─ 加载模型 + 因子面板 + OAMV
       ├─ Exit评估 + 状态切换 + 选股
       │
       ├──→ output/signals/{date}.json          ← 交易指令
       ├──→ current_positions.json               ← 更新持仓
       ├──→ backtest daily_positions.csv         ← 同步今日持仓
       └──→ backtest daily_nav.csv               ← 同步今日净值预估

14:55  按信号挂单执行 (成交价 ≈ 收盘价, 对齐回测T日收盘成交假设)
```

## 踩过的坑

1. **symbol 格式**: `trade_log.csv` 存 int（`563`），实时行情 key 是 6 位字符串（`000563`）。必须 `str(symbol).zfill(6)` 补齐。

2. **positions 缺少 symbol 字段**: 重建持仓 dict 里只有 shares/entry_date/entry_price/side，必须补上 `info['symbol'] = sym`。

3. **daily_nav date 列位置**: concat 时注意 date 列不要被 drop 或错位。新行直接放 date 字段，用 ignore_index=True 拼接。

4. **首次运行持仓为空**: 回退链路 — `current_positions.json` 非空 → `trade_log.csv` → `daily_positions.csv`（无 entry_price）→ 空 DataFrame。

## spring_b2 适配清单

- [ ] 回测输出 `trade_log.csv`（含 entry_date, entry_price, symbol, shares, action, side）
- [ ] 实现 `load_backtest_positions()`，symbol 补齐到 spring_b2 的格式
- [ ] 实现 `push_to_backtest()`，NAV 列名对齐 spring_b2 的报表格式
- [ ] `load_positions()` 添加回退逻辑
- [ ] `--execute` 中调用 `push_to_backtest()`
- [ ] 测试：删掉 `current_positions.json` → 跑 `--execute` → 确认回测 CSV 被追加
