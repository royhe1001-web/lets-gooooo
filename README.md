# Spring B2 策略

砖型反转形态 + OAMV 市场状态的 A 股短线选股策略（T+1 ~ T+4）。

**最新成绩**: Test 2026-01-01 ~ 05-13 = **+64.9%**, Sharpe 1.138, MaxDD 21.4%

## 环境

```bash
pip install pandas numpy requests pyarrow
```

## 项目结构

```
stock/
├── run_backtest_2026.py           # 回测入口
├── real_time_signal.py            # 实时信号（14:50 新浪行情 + 选股）
│
├── strategy_spring.py             # 核心选股（8条件 + 11层权重）
├── strategy_brick.py              # 砖型图辅助
├── screener_engine.py             # 筛选引擎
├── screen_brick.py / screen_spring.py
├── oamv.py                        # OAMV 活筹指数（市场状态）
├── indicators.py                  # KDJ/MACD/白黄线/砖型图
├── fetch_kline.py                 # K线拉取
│
├── quant_strategy/
│   ├── oamv.py / indicators.py
│
├── ML_optimization/
│   ├── phase2c_oamv_grid_search.py   # OAMVSimEngine (P0-P4 + P15)
│   ├── phase2c_bull_grid_search.py   # 网格搜索 + 数据加载
│   ├── mktcap_utils.py               # 市值分位过滤
│   ├── sector_utils.py               # 板块动量（可选）
│   └── features/                     # K线 parquet（4243只）
│
└── output/
    ├── market_turnover.csv        # 市场成交额（OAMV 依赖）
    ├── monthly_mktcap.csv         # 月度市值
    └── signals/                   # 实时信号输出
```

## 快速开始

### 回测

```bash
python run_backtest_2026.py                        # 默认年初至今
python run_backtest_2026.py --start 2025-07-01     # 指定起始
python run_backtest_2026.py --start 2025-07-01 --end 2025-12-31
```

### 实时信号

```bash
python real_time_signal.py           # 查看
python real_time_signal.py --execute  # 保存到 output/signals/
```

信号输出 (`output/signals/YYYYMMDD.json`)：
```json
{
  "date": "2026-05-14",
  "state": "aggressive",
  "buys": [{"symbol": "601198", "price": 13.05, "shares": 7600, "amount": 99180}]
}
```

### 定时任务（Windows 任务计划）

```
触发器: 周一至周五 14:50
操作: python real_time_signal.py --execute
起始于: <项目目录>
```

## 策略概要

**入场**（8 条件 AND）：前日 J<20 + 涨幅>阈值 + J<阈值 + 放量 + 上影<阈值 + 白>黄 + 收>黄 + 收>MA20。阈值由 OAMV 状态动态调节。

**权重**（11 层叠加，≤5x）：涨>9%→2.5x | 跳空→1.5x | 缩量→1.2x | 深度缩量→1.3x | 回调→1.2x | 上影折价→0.7x | 共振→1.5x

**P15 仓位**：总权益基准 | 最多 4 只 | 单只 ≤ 50% | 整手取整 | min 8000 | 防御空仓

**退出**：信号止损 | 蜡烛止损(OAMV动态) | 浮盈保护 | T+4 | 放飞滴滴

## 移植

1. 复制整个 `stock/` 文件夹
2. `pip install pandas numpy requests pyarrow`
3. 运行 `python run_backtest_2026.py` 验证

所有路径为相对路径，无需修改配置。
