# 量化交易系统操作手册

## 1. 系统设计

两平台代码完全一致，`run.py` 启动时自动检测系统并选择最优引擎：

| 模块 | macOS (M1) | Windows |
|------|-----------|---------|
| 选股引擎 | `screeners/*_m1.py` (4 P-core) | `screeners/*.py` (标准多核) |
| 策略代码 | `quant_strategy/` | **完全相同** |
| 回测系统 | `simulation/` | **完全相同** |
| 分析工具 | `run_test/` | **完全相同** |
| ML 训练 | sklearn (轻量) | sklearn + xgboost + lightgbm |

**两台设备都可以做所有事情**，区别仅在于选股时自动选用最优引擎。

## 2. 协作方式

```
macOS 修改策略 → git push → Windows git pull → 立即生效
Windows 修改策略 → git push → macOS git pull → 立即生效
```

通过 `TASKS.md` 记录待处理和已完成事项，两机共享。

## 3. 项目结构

```
stock/
├── run.py                          # 统一入口 (自动检测平台)
├── setup_dual_platform.py          # 双平台初始化
├── OPERATIONS.md                   # 本手册
├── TASKS.md                        # 协作任务板
├── .gitignore / .platform_config.json
│
├── quant_strategy/                 # 策略核心
│   ├── indicators.py               # 技术指标
│   ├── strategy_b2.py              # ★ b2 主力策略
│   ├── strategy_brick.py           # 砖型图策略
│   ├── strategy_b1.py              # b1 备用
│   └── screener_engine.py          # Windows 引擎
│
├── screeners/                      # 选股 (平台自动选择)
│   ├── screen_b2.py / _m1.py       # b2
│   ├── screen_brick.py / _m1.py    # 砖型图
│   └── screener_engine_m1.py       # M1 引擎
│
├── simulation/                     # 回测系统
│   ├── run_b2_simulation_2026.py   # b2 模拟引擎
│   ├── run_b1_simulation.py        # b1 回测
│   └── run_trade_report.py         # 绩效报告
│
├── run_test/                       # 分析工具
│   ├── run_holdings_analysis.py    # 持仓分析+止损+轮换
│   └── run_b2_case_study.py        # 案例学习
│
├── fetch_kline/                    # 数据层
│   ├── fetch_kline.py              # 下载程序
│   └── stock_kline/                # 个股CSV (不纳管Git)
│
└── output/                         # 结果 (不纳管Git)
    ├── trade_report_*_final.txt    # 最终报告
    ├── b2_sim_*.csv / b2_nav_*.csv # 回测数据
    ├── csi300_5y.csv               # 基准
    └── models/                     # ML模型
```

## 4. 首次安装

```bash
git clone <repo-url> stock && cd stock
pip install numpy pandas scipy matplotlib scikit-learn statsmodels
pip install tushare akshare mplfinance plotly
pip install numba numexpr bottleneck pyarrow
pip install tqdm tabulate openpyxl xlrd
pip install jupyter notebook ipykernel
pip install backtrader pyfolio-reloaded empyrical-reloaded
pip install sqlalchemy h5py tables

# Windows 额外:
pip install xgboost lightgbm joblib

# 初始化:
python setup_dual_platform.py
mkdir fetch_kline\stock_kline   # Windows
mkdir -p fetch_kline/stock_kline  # macOS
```

## 5. 日常命令

```bash
python run.py status              # 系统状态
python run.py screen              # b2 选股 (自动选引擎)
python run.py brick               # 砖型图选股
python run.py download            # 下载数据
python run.py simulate            # 回测
python run.py report -o report.txt  # 交易报告
python run.py ml                  # ML 训练
python run.py holdings 000960 --entry 000960:2026-05-06:38.46 --no-scan
python run.py oamv                  # 活跃市值
python run.py oamv --plot           # 含图表
```

## 6. 策略参数 (最终版)

### b2 入场 (7条件)

| # | 条件 | 阈值 |
|---|------|------|
| c1 | J前日 | < 20 |
| c2 | 涨幅 | > 4% |
| c3 | J当日 | < 65 |
| c4 | 放量 | volume > 前日 |
| c5 | 上影线 | < 3.5% |
| c6 | 白线 | > 黄线 |
| c7 | 收盘 | > 黄线 |

前期强势股 (20日涨>20%): J今→70, 上影→4.5%, 权重×0.8

### b2 权重叠加

```
涨>7%(×2) → 跳空(×1.5) → 缩量(×1.3) → 回调>5%(×1.2) → 共振(×1.5)
上影折价(×0.7)
```

### 止损止盈

| 触发 | 条件 | 执行 |
|------|------|------|
| 蜡烛破位 | close < entry_low×0.97 | 当日收盘 |
| 信号收盘 | close < signal_close | 当日收盘 |
| 黄线 | close < yellow_line | 当日收盘 |
| 买入保护 | T+1触发 | T+2开盘 |
| 60日弱 | ≥60天 + 收益<15% | 当日收盘 |
| 252日 | ≥252天 | 强制清仓 |

### 换仓

新信号权重 > 最弱持仓+1.0 → 换仓

### 已验证回测

| 期间 | 资金 | 仓位 | 收益率 |
|------|------|------|--------|
| 2024-2026 | 40K | 2-3 | +323.6% |
| 2019-2026 | 100K | 5-8 | +114.0% |

## 7. 协作流程

```
每次工作前:  git pull
每次工作后:  git add -A && git commit -m "简述改动" && git push

任务跟踪:    编辑 TASKS.md
策略修改:    直接改代码, git push 后对方 git pull 即可
数据同步:    手动拷贝 stock_kline/ (不纳入Git)
```

Tushare token 在 `fetch_kline/fetch_kline.py`。两台设备用不同 token（同一账户可多申请）。
