# Spring — 强势确认反转策略

A股短线反转策略（T+1 ~ T+4），基于回调确认 + 多因子加权 + OAMV市场状态感知。

## 策略核心

- **入场信号**: 前日J<20（超卖）+ 当日涨幅确认 + 放量
- **多因子加权**: 回调深度、缩量衰竭、跳空高开、上影线折价、砖型图共振
- **市场状态**: OAMV（活筹指数）动态调整入场门槛和止损缓冲
- **止损**: T+4时间止损 + 蜡烛止损 + 放飞滴滴止盈

## 最优参数 (Explosive_def2.0)

| 参数 | 值 |
|------|-----|
| gain_min | 4% |
| j_today_max | 65 |
| weight_gain_2x | 2.5x (>9%涨幅) |
| weight_pullback | 1.2x (>5%回调) |
| 止损方案 | None（原始引擎） |

Test 2026: Return +34.7%, Sharpe 0.886, 85 trades

## 快速开始

```bash
# 安装依赖
pip install pandas numpy

# 数据获取（需先配置fetch_kline）
python fetch_kline.py

# 运行选股
python run.py screen

# 运行回测模拟
python run_spring_simulation_2026.py

# 生成交易报告
python run_trade_report.py
```

## 项目结构

```
spring_b2/
├── strategy_spring.py          # 核心策略
├── quant_strategy/             # 量化策略模块
│   ├── strategy_spring.py      # 策略导出
│   ├── indicators.py           # 技术指标计算
│   └── oamv.py                 # OAMV活筹指数
├── run_spring_simulation_2026.py  # T+1模拟引擎
├── run.py                      # 主入口
├── screen_spring.py            # 选股器 (Windows)
├── screen_spring_m1.py         # 选股器 (Apple Silicon)
├── ML_optimization/            # ML参数优化
│   ├── mktcap_utils.py         # 市值分位法过滤
│   ├── phase2c_bull_grid_search.py  # 多牛市网格搜索
│   └── ...
└── fetch_kline.py              # K线数据获取
```

## 训练历史

- Phase 1: 基础筛选 (2025)
- Phase 2c: B2+OAMV联合优化，3牛市网格搜索
- Phase 3-5: 动态阈值、LightGBM过滤
- 2026-05: 新股票池(B+C分位法) + 4周期训练 + T+4评估 → Explosive_def2.0+None最优
