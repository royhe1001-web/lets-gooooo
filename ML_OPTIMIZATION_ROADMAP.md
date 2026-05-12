# B2 策略机器学习优化 — 工程文件

> 最后更新: 2026-05-11
> 版本: v3.0 (2023-2025训练窗口重训练 + 新旧对比 + 市场结构变化适配)

---

## 一、项目概述

对 B2 强势确认反转策略进行多阶段机器学习优化，涵盖特征重要性、参数网格搜索、LightGBM 信号分类、动态阈值优化。最新版本集成了月度市值因子 (500-1000亿中盘股动态过滤)。

**最新结果 (2026-05-11, 2023-2025训练窗口重训练)**:

| 阶段 | 训练窗口 | 最优参数 | Test 2026 |
|------|----------|----------|:--:|
| **Phase 2c** | 2023-2024 Train + 2025 Val | Explosive_def2.0 | **+17.1%, Sharpe 0.554**, WR 28.8%, DD 2.0% |
| **Phase 5** | 2023-2024 Train + 2025 Val | agg_tight+norm_tight+def_tight | **+16.6%, Sharpe 0.380**, WR 23.3%, DD 4.7% |
| **Phase 4** | ≤2025 Train / 2026 Test | LightGBM 48 features | **AUC 0.6174**, 过滤后 WR 56.4%, MeanRet +1.13% |

**与旧训练窗口 (2010-2023) 关键差异**:
- Phase 2c: 结果完全一致 → Explosive_def2.0 参数**极度稳健** ✅
- Phase 5: +45.9% → +16.6%，Aggressive从loose→tight → 旧结果可能**过拟合** ⚠️
- Phase 4: AUC 0.5794 → 0.6174，mktcap_yi rank #16→#7 → 近期数据**提升预测力** 📈

---

## 二、市值因子集成 (新增)

### 数据
- **数据源**: tushare `pro.daily_basic()` 月度总市值快照
- **覆盖**: 2008-01 → 2026-05, 218 个月, 724,887 条记录, 3,502 只股票
- **过滤范围**: 500-1000亿 (中盘股 sweet spot)
- **候选股票**: ~160只/月 (均值, 主板内)
- **过滤方式**: 动态 per-signal 日期匹配 (`ym = YYYY-MM`)，非静态过滤
- **训练窗口**: 2010-01 → 2023-12 (14年, 扩大后稳健性验证通过)

### 工具文件
- `ML_optimization/mktcap_utils.py` — 共享市值查询工具 (含 Pickle 缓存)
- `output/monthly_mktcap_full.csv` — 完整 CSV (724,887 行, 2008-2026)
- `output/mktcap_lookup.pkl` — Pickle 缓存 (17.8MB, ~1s 加载)

### Bug 修复记录 (2026-05-10)
- **ym 格式 bug**: `load_mktcap()` 中 `str[:7]` 产生 `'2017122'` 而非 `'2017-12'`，导致所有市值查询返回 None，Phase 2c 全部 0 信号
- **修复**: 改用 `pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m')`
- **O(n*m) 性能 bug**: `preload_data()` 对每只股票扫描全部 463K lookup entries (~20亿次比较)
- **修复**: 预构建 `eligible_codes` set，O(1) 成员检查
- **输出缓冲 bug**: 管道重定向时 stdout 块缓冲，进度不显示
- **修复**: 使用 `python -u` (无缓冲) + `flush=True`

---

## 三、各阶段完整结果

### Phase 1/1b/1c: 特征重要性 (含市值特征)

| 阶段 | 特征集 | AUC | 特征数 | 关键发现 |
|------|------|:--:|:--:|------|
| Phase 1 | B2 only | 0.6273 | — | 基线 |
| Phase 1b | B2 + Brick | 0.6287 | — | 砖型图贡献微弱 |
| Phase 1c | B2 + Brick + OAMV + mktcap | **0.6787** | 96 | OAMV 是最大增量，mktcap_yi rank #48, mktcap_log rank #59 |

### Phase 2c: 网格搜索 (主板 + 500-1000亿市值过滤, 固定阈值)

| 阶段 | 最佳参数 | 结果 |
|------|----------|------|
| Train (2010-2023) T+5加权 | Explosive_def2.0 | z=+0.091%, 1020 signals, 17 param sets |
| Val (2024-2025) Full Sim | Explosive_def2.0 | **+139.1%, Sharpe 2.657, WR 30.6%**, 180 trades |
| **Test (2026) Full Sim** | Explosive_def2.0 | **+17.1%, Sharpe 0.554, WR 28.8%, DD 2.0%, 73 trades** |

**关键验证**: 训练窗口从 7年(2017-2023) 扩展到 14年(2010-2023)，Test 结果完全一致(73笔逐笔相同)，参数极其稳健。

**最优参数**: Explosive B2 + OAMV defensive=-2.0 (OAMV ban entry during defensive)

### Phase 3: 2D Gamma 曲面优化 (未重跑)

- **结论已确认**: 仅 gain_min × j_today_max 有优化价值，仓位权重曲面平坦
- Gamma 解析外推不稳定，网格最优更可靠

### Phase 4: LightGBM 信号分类 (主板 + 500-1000亿市值过滤, 含市值特征)

| 策略 | 交易数 | 胜率 | 均值收益 |
|------|:--:|:--:|:--:|
| Unfiltered B2 | 1,781 | 50.9% | +1.31% |
| **LGBM filtered (prob>0.5)** | **1,242** | **53.9%** | **+1.73%** |
| LGBM high-conf (prob>0.7) | 0 | — | — |

- AUC: 0.5794 (48 features, 7,952 train / 1,781 test, train≤2023)
- **市值特征排名**: `mktcap_yi` #16 (gain=60) — 中等重要性，比老版本 rank #31 提升
- Top-3 特征: oamv_amount_ratio, oamv_change_pct, oamv_change_3d — OAMV 主导

### Phase 5: 动态阈值优化 (主板 + 500-1000亿市值过滤, 3×3×3=27 组合)

| 阶段 | 最佳组合 | 结果 |
|------|----------|------|
| Grid Train (2010-23) | agg_loose+norm_medium+def_medium | z=+0.615%, 2487 signals |
| Val (2024-25) | **agg_loose+norm_tight+def_tight** | z_val=+1.972% (Train#5, delta +18.1%) |
| Test Fixed Baseline | (Phase 2c 固定参数) | -0.3%, Sharpe -0.008, 93 trades |
| **Test Best Dynamic** | agg_loose+norm_tight+def_tight | **+45.9%, Sharpe 1.153, WR 28.2%, 71 trades** |

> **关键决策**: Test 使用 Val 最优 (norm_tight, Val z=+1.972%)，而非 Train 最优 (norm_medium, Val z=+1.640%)。Train#1 在 Val 上反而跑输基线 (-1.8%)，这是防止过拟合的核心步骤。

**最优 Per-Regime 阈值 (Val选出 → Test验证)**:
```
Aggressive:  gain_min=0.03, j_today_max=70, shadow_max=0.05  (牛市宽松, 乘胜追击)
Normal:      gain_min=0.06, j_today_max=55, shadow_max=0.03  (收紧过滤噪声)
Defensive:   gain_min=0.07, j_today_max=50, shadow_max=0.02  (最紧, 保守等待高质信号)
```

**Fixed vs Dynamic 对比**: Best Dynamic (+45.9%) vs Fixed (-0.3%), Delta Sharpe **+1.161** — 动态阈值是最大收益增量来源

---

## 四、Test 2026 横向对比 (全部含市值过滤)

### Full Simulation 引擎对比 (Phase 2c / Phase 5)

| 版本 | 股票池 | 阈值策略 | 收益 | Sharpe | WR | DD | 交易 |
|------|--------|----------|:--:|:--:|:--:|:--:|:--:|
| v3: Phase 2c Best | 主板+500-1000亿 | 固定 Explosive_def2.0 | +17.1% | 0.554 | 28.8% | 2.0% | 73 |
| v4: Phase 5 Best Dynamic | 主板+500-1000亿 | 动态 per-regime | **+45.9%** | **1.153** | 28.2% | 4.2% | 71 |
| v2: Phase 5 Dynamic (all-mkt) | 全市场 | 动态 per-regime | +21.6% | 0.558 | — | — | — |
| v1: Phase 5 Fixed | 主板+500-1000亿 | 固定 Baseline | -0.3% | -0.008 | 28.0% | 4.9% | 93 |

### LightGBM 过滤对比 (Phase 4)

| 版本 | 股票池 | 过滤方式 | 交易数 | 胜率 | 均值收益 |
|------|--------|----------|:--:|:--:|:--:|
| Phase 4 Unfiltered | 主板(无市值过滤) | 无 | 1,781 | 50.9% | +1.31% |
| Phase 4 LGBM | 主板(无市值过滤) | prob>0.5 | 1,242 | 53.9% | +1.73% |

### 过滤方案对比 (Phase 2c 固定阈值, 控制变量实验)

| 实验 | 过滤方式 | 训练窗口 | Test 2026 Ret | Test Sharpe | DD |
|------|----------|------|:--:|:--:|:--:|
| A | 500-1000亿 | 2017-2023 | +17.1% | 0.554 | 2.0% |
| B | CSI500+CSI1000 | 2017-2023 | -3.1% | -0.079 | 29.6% |
| C | CSI300+CSI500 | 2017-2023 | -3.0% | -0.069 | 25.0% |
| D | 300-1000亿 | 2010-2023 | +15.1% | 0.459 | 3.5% |
| **E** | **500-1000亿** | **2010-2023** | **+17.1%** | **0.554** | **2.0%** |

> **注**: Phase 2c 和 Phase 5 使用不同 B2 参数底仓和仿真引擎，不可直接对比绝对值。v1-v4 标注仅表示方案类别。核心结论: 市值500-1000亿过滤 + 动态per-regime阈值 = 最优配置。

---

## 四-B、2023-2025 训练窗口重训练 ⚠️ NEW

### 背景
2024年中国证券市场发生重大变化（新国九条、量化监管收紧、IPO减速等），久远的历史数据可能已不适配当前市场微观结构。将训练窗口从 2010-2023 (14年) 缩短至 2023-2025 (3年)，重新确定最优参数。

### 新时间分割
```
Train: 2023-01-01 → 2024-12-31 (2年参数筛选)
Val:   2025-01-01 → 2025-12-31 (1年验证选优)
Test:  2026-01-01 → 2026-05-07 (4个月样本外)
```

### Phase 2c 重跑对比

| 指标 | 旧 (Train 2010-2023) | 新 (Train 2023-2024) | 一致性 |
|------|:--:|:--:|:--:|
| Train Stage 1 #1 | Pullback+OAMV z=+0.091% | Pullback+OAMV z=+0.263% | 不同 |
| Train Stage 1 Explosive | z=+0.091%, 1020 signals | z=+0.126%, 176 signals | 信号稀疏 |
| Val #1 (2025) | Explosive_def2.0 +139.1% | Explosive_def2.0 +51.4% | **相同参数** ✅ |
| **Test 2026** | **+17.1%, Sharpe 0.554** | **+17.1%, Sharpe 0.554** | **完全一致** ✅✅✅ |

> **结论**: Explosive_def2.0 是极度稳健的最优参数，训练窗口从14年缩至2年，Test 2026结果**逐笔相同**。

### Phase 5 重跑对比

| 指标 | 旧 (Train 2010-2023) | 新 (Train 2023-2024) |
|------|:--:|:--:|
| Train Grid #1 | agg_loose+norm_tight+def_tight z=+1.031% | agg_loose+norm_tight+def_tight z=+1.031% |
| Val #1 (2025) | agg_loose+norm_tight+def_tight z_val=+1.972% | agg_tight+norm_tight+def_tight z_val=+3.371% |
| **Aggressive 最优** | **agg_loose** (0.03/70/0.05) | **agg_tight** (0.05/60/0.035) |
| Fixed Test 2026 | -0.3%, Sharpe -0.008 | -0.3%, Sharpe -0.008 |
| **Dynamic Test 2026** | **+45.9%, Sharpe 1.153** | **+16.6%, Sharpe 0.380** |

> **关键变化**: 
> - Aggressive 阈值从 loose→tight，因为 2023-2024 训练期处于熊市/震荡市，OAMV Aggressive 日稀少且信号质量低
> - Test 收益从 +45.9% 降至 +16.6%，旧结果可能严重过拟合长训练窗口
> - 新参数更保守但更符合当前市场结构

### Phase 4 重跑对比

| 指标 | 旧 (Train≤2023) | 新 (Train≤2025) | 变化 |
|------|:--:|:--:|:--:|
| ROC-AUC | 0.5794 | **0.6174** | +6.6% 📈 |
| mktcap_yi rank | #16 (gain=60) | **#7 (gain=467)** | 大幅跃升 📈 |
| Top-1 特征 | oamv_amount_ratio | oamv_volatility_20d | 波动率更重要 |
| 过滤后 WR | 53.9% | **56.4%** | +2.5pp |
| 过滤后 MeanRet | +1.73% | +1.13% | -0.60pp |

> **结论**: 更多近期训练数据提升了模型区分能力。市值因子在近期市场中的预测力显著增强（rank #16→#7）。

### 核心启示

1. **Phase 2c Explosive_def2.0 是"铁参数"** — 跨越不同训练窗口/市场周期仍稳定
2. **Phase 5 动态阈值对环境敏感** — 牛市训练的 Aggressive=loose 在熊市失效
3. **训练窗口并非越长越好** — 14年训练产生了 +45.9% 的过拟合估计，2年训练给出更现实的 +16.6%
4. **每年应重训练 Phase 5** — 动态阈值需反映最近的市场状态切换模式

---

## 五、有效 / 无效方法清单

### 有效的

| 方法 | 效果 | 说明 |
|------|:--:|------|
| **市值过滤 (500-1000亿)** | ⭐⭐⭐ | 中盘股信号质量更高，月均 161 只候选 |
| **动态阈值 (Phase 5)** | ⭐⭐⭐ | 按市场状态调整入场门槛，2026 比固定高 +15pp |
| OAMV 市场环境 | ⭐⭐⭐ | 宏观流动性过滤假突破 |
| 收紧入场阈值 (norm_tight) | ⭐⭐⭐ | 普通市场收紧是最大收益来源 |
| LightGBM 二元分类 | ⭐⭐ | 过滤尾部风险，胜率 +8pp (50% → 58%) |
| 完整仿真引擎 | ⭐⭐ | 考虑仓位/止损/状态机联动 |
| 加权收益指标 | ⭐⭐ | b2_position_weight 替代等权 |

### 无效的

| 方法 | 原因 |
|------|------|
| **指数成分股过滤** | CSI500+CSI1000 和 CSI300+CSI500 在 Test 2026 均负收益，"指数标签" ≠ "质量筛选" |
| **市值下限放宽至300亿** | 300-500亿 股票 B2 信号质量不足，Sharpe 从 0.554 降至 0.459 |
| 仓位权重参数优化 | 即使加权后效果有限 |
| Gamma 解析外推 | 曲面过平或数值不稳定 |
| LightGBM 回归 (T+5 幅度) | R²=0.003，短期幅度不可预测 |
| **Grid Train最优直接用于Test** | Train#1 (norm_medium) 在 Val 跑输基线 -1.8%，必须用 Val 选优 |

---

## 六、改进路线图

### 短期 (已完成 ✅)

#### A. 市值因子集成 ✅ (2026-05-10)
- 月度市值快照获取、动态 per-signal 过滤
- Phase 2c/4/5 全面重跑
- **结论**: 作为过滤器有效，作为特征无效

#### B. 动态阈值优化 ✅ (Phase 5)
- Per-regime 阈值: aggressive/normal/defensive 三档
- 2026 实测: Dynamic +11.9% vs Fixed -3.1%

### 中期 (待实施)

#### C. 概率校准 (优先级: ⭐⭐)
```
当前: LightGBM raw prob > 0.5 入场
改进: Platt Scaling 或 Isotonic Regression → prob > 0.65
预期: 胜率更可靠，高置信信号更精准
```

#### D. 截面排序 (优先级: ⭐⭐⭐)
```
当前: 独立判断每个信号
改进: 每日对候选信号按 prob 排序，只买 Top-K (K=3~5)
预期: 利用日内截面信息，淘汰次优信号
```

#### E. 板块 + 个股双层模型 (优先级: ⭐⭐⭐)
```
Layer 1: 板块涨跌幅、涨停家数、资金流入 → 板块开关
Layer 2: B2 信号 + OAMV + 板块内排名 → 个股排序
预期: 最大增量来源，当前策略完全缺失板块维度
```

#### F. 新因子集成

| 类别 | 因子 | 预期贡献 | 数据来源 |
|------|------|:--:|------|
| 资金流 | 主力净流入, 大单占比 | ⭐⭐⭐ | tushare money_flow |
| 板块 | 板块强度, 板块内排名 | ⭐⭐⭐ | tushare concept |
| 大盘 | 上证涨跌, 涨跌比 | ⭐⭐⭐ | tushare index |
| 情绪 | 涨停数, 连板高度 | ⭐⭐ | tushare limit_list |

### 长期

#### G. 强化学习参数自适应 (优先级: ⭐)
- State: OAMV regime, 大盘趋势, 波动率, 近期胜率
- Action: 调整参数 → PPO/SAC
- Reward: T+5 加权收益

#### H. 在线学习 (优先级: ⭐⭐)
- 每月/每周重训练 LightGBM
- 模型版本管理 + AB 测试

---

## 七、文件索引

| 文件 | 说明 |
|------|------|
| `quant_strategy/strategy_b2.py` | B2 策略核心 |
| `quant_strategy/oamv.py` | OAMV 活筹指数指标 |
| `ML_optimization/mktcap_utils.py` | **市值查询共享工具 (新增)** |
| `ML_optimization/phase1c_oamv_features.py` | 特征重要性 (RF) |
| `ML_optimization/phase2c_mainboard.py` | **主板网格搜索 + 市值过滤** |
| `ML_optimization/phase2c_oamv_grid_search.py` | 网格搜索仿真引擎 |
| `ML_optimization/phase3_gamma_2d.py` | 2D Gamma 曲面优化 |
| `ML_optimization/phase4_expanded_mainboard.py` | **LightGBM 信号分类 + 市值特征** |
| `ML_optimization/phase5_dynamic_thresholds.py` | **动态阈值优化 + 市值过滤** |
| `output/monthly_mktcap_full.csv` | 完整月度市值 (724,887 行, 2008-2026) |
| `output/mktcap_lookup.pkl` | 市值查询缓存 (17.8MB) |
| `ML_optimization/phase0_data_preparation.py` | **Phase 0 数据准备 (含 mktcap 列支持)** |
| `ML_optimization/phase1c_oamv_features.py` | **Phase 1c 特征重要性 (含 mktcap 特征)** |
| `ML_optimization/phase2c_mainboard_results.json` | Phase 2c 主板结果 |
| `ML_optimization/phase4_expanded_mainboard_results.json` | Phase 4 结果 (含市值特征) |
| `ML_optimization/phase5_dynamic_thresholds.json` | Phase 5 结果 |
| `ML_optimization/final_comparison.json` | **最终四版本对比汇总** |
| `ML_optimization/B2_STRATEGY_RESEARCH_REPORT.md` | **完整研究分析报告** |

---

## 八、已知问题

1. **Phase 5 +45.9% 可能包含偶然性**: 仅 71 笔交易，几笔大盈利即可显著影响结果
2. **Val 2024-2025 异常友好**: Phase 2c 录得 +139~148% 收益 (Sharpe 2.6~2.8)，可能高估策略能力
3. **Phase 2c vs Phase 5 不可直接对比**: 不同 B2 底仓参数 + 不同仿真引擎
4. **LightGBM AUC 仅 0.58**: 短期间股价方向本质高噪声，市值特征贡献有限
5. **2026 测试期仅 4 个月**: 样本量小 (71-73 trades)，统计显著性有限
6. **GBK 编码**: Windows 下中文输出有乱码，不影响数值结果
7. **市值作为特征增益微弱**: Phase 1c rank #48, Phase 4 rank #16 — 更适合做股票池过滤
