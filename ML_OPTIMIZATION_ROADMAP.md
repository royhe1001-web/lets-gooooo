# B2 策略机器学习优化 — 工程文件

> 最后更新: 2026-05-10
> 版本: v1.0 (Phase 1-4 完成)

---

## 一、项目概述

对 B2 强势确认反转策略进行 6 阶段机器学习优化，涵盖特征重要性、参数网格搜索、2D Gamma 曲面拟合、LightGBM 信号分类。

**基线**: 默认参数 +42.7%, Sharpe 1.084
**最优**: Explosive_def2.0 +64.9%, Sharpe 1.770 (+63%)

---

## 二、当前最优参数

```python
OPTIMAL_PARAMS = {
    # === 入场阈值 (Phase 3 2D Gamma Grid 最优) ===
    'j_prev_max': 20,          # 前日 J < 20 (低位确认)
    'gain_min': 0.054,         # 当日涨幅 > 5.4% (原 4% → 收紧)
    'j_today_max': 59,         # 当日 J < 59 (原 65 → 收紧)
    'shadow_max': 0.035,       # 上影线 < 3.5%
    'j_today_loose': 70,       # 强势股放宽 J < 70
    'shadow_loose': 0.045,     # 强势股放宽 上影线 < 4.5%
    'prior_strong_ret': 0.20,  # 20日涨 > 20% = 强势股

    # === 仓位权重 (Phase 3 曲面平坦, 保持默认) ===
    'weight_gain_2x_thresh': 0.09,    # 涨幅 > 9% → 2x
    'weight_gain_2x': 2.5,            # 最大 2.5x
    'weight_gap_thresh': 0.02,        # 跳空 > 2% → 1.5x
    'weight_gap_up': 1.5,
    'weight_shrink': 1.2,             # 缩量 → 1.2x
    'weight_deep_shrink_ratio': 0.80, # 深度缩量阈值
    'weight_deep_shrink': 1.3,        # 深度缩量 → 1.3x
    'weight_pullback_thresh': -0.05,  # 回调 > 5% → 1.2x
    'weight_pullback': 1.2,
    'weight_shadow_discount_thresh': 0.015,  # 上影线折价
    'weight_shadow_discount': 0.7,           # → 0.7x
    'weight_strong_discount': 0.8,           # 强势放宽 → 0.8x
    'weight_brick_resonance': 1.5,           # 砖型共振 → 1.5x

    # === OAMV 市场状态 (Phase 2c 最优) ===
    'oamv_aggressive_buffer': 0.95,   # 猛干放宽止损 5%
    'oamv_defensive_buffer': 0.99,    # 防守收紧止损 1%
    'oamv_defensive_ban_entry': True, # 防守禁止开仓
}
```

---

## 三、各阶段关键结论

### Phase 1/1b/1c: 特征重要性

| 阶段 | 特征集 | AUC | 关键发现 |
|------|------|:--:|------|
| Phase 1 | B2 only | 0.6273 | 基线 |
| Phase 1b | B2 + Brick | 0.6287 | 砖型图贡献微弱 |
| Phase 1c | B2 + Brick + OAMV | **0.6754** | OAMV 是最大信息增益 (+4.7bps) |

**OAMV Top 5 特征**: volatility_20d, amount_ratio, change_3d, change_pct, change_5d

### Phase 2/2b/2c: 网格搜索

| 策略 | 收益 | Sharpe | 胜率 | 交易数 |
|------|:--:|:--:|:--:|:--:|
| **Explosive_def2.0** | **+64.85%** | **1.770** | 30.0% | 100 |
| OAMV_Conservative | +64.31% | 1.752 | 29.3% | 99 |
| Control (无OAMV) | +60.51% | 1.635 | 26.7% | 105 |
| DEFAULT (无OAMV) | +42.74% | 1.084 | 21.1% | 147 |

**结论**: OAMV 贡献 Sharpe +0.135；Explosive 参数底仓 + OAMV def2.0 是最优组合。

### Phase 3: 2D Gamma 曲面优化

| 参数对 | 曲面 | 最优 z | 发现 |
|------|:--:|:--:|------|
| gain_2x_thresh × weight_gain_2x | 平坦 | 0.95% | 仓位权重不影响等权收益 |
| OAMV regime 阈值 | 平坦 | 0.95% | regime 分类阈值影响微弱 |
| pullback_thresh × weight_pullback | 平坦 | 0.95% | 同上 |
| **gain_min × j_today_max** | 有结构 | **1.60%** | 唯一有优化价值的参数对 |

**教训**: Gamma 解析最优外推不稳定 (validated z = -0.83%), **网格最优值比 Gamma 外推更可靠**。

### Phase 4/4b: LightGBM 信号过滤

| 过滤策略 | 交易数 | 胜率 | 平均收益 |
|------|:--:|:--:|:--:|
| 未过滤 | 7,322 | 48.5% | +0.48% |
| LGBM (prob>0.5) | 1,135 | **57.0%** | **+1.57%** |
| LGBM (prob>0.7) | 8 | 75.0% | +10.17% |

**Phase 4b**: LightGBM 回归 (预测 T+5 收益率) 完全失效 (R²=0.003)。
**结论**: 方向预测可行，幅度预测不可行。二元分类 > 回归。

---

## 四、有效 / 无效方法清单

### 有效的

| 方法 | 效果 | 适用场景 |
|------|:--:|------|
| OAMV 市场环境 | ⭐⭐⭐ | 宏观流动性过滤假突破 |
| 收紧入场阈值 | ⭐⭐⭐ | 减少低质量信号 |
| LightGBM 二元分类 | ⭐⭐ | 过滤尾部风险信号 |
| 完整仿真引擎 | ⭐⭐ | 考虑仓位/止损/状态机联动 |

### 无效的

| 方法 | 原因 |
|------|------|
| 仓位权重参数优化 | T+5 等权收益与权重无关，需用加权收益指标 |
| Gamma 解析外推 | 曲面过平或数值不稳定，网格最优更可靠 |
| LightGBM 回归 | 短期股价幅度本质不可预测 |
| OAMV regime 阈值微调 | 3.0%/-2.0% 已是合理分界，微调无增益 |

### 根本限制

1. **信噪比极低**: LightGBM AUC 仅 0.546，短期股价方向本就难以预测
2. **样本量 vs 特征维度**: 36K 信号在 68 维特征空间仍显不足
3. **过拟合风险**: Phase 2c 最优组合可能包含运气成分

---

## 五、改进路线图

### 短期 (可直接实施, 预计 1-2 天)

#### A. 动态阈值 (优先级: ⭐⭐⭐)
```
当前: gain_min = 0.054 (固定)
改进: 所有阈值 = f(oamv_regime, volatility)
      aggressive: gain_min=0.04 (降门槛, 乘胜追击)
      defensive:  gain_min=0.06 (提门槛, 保守等待)
      normal:     gain_min=0.054
```
**预期**: 胜率 +2-3pp, Sharpe +0.1-0.2

#### B. 概率校准 (优先级: ⭐⭐)
```
当前: LightGBM raw prob > 0.5 入场
问题: prob=0.7 不一定真的是 70% 胜率
改进: Platt Scaling 或 Isotonic Regression
      校准后 prob > 0.65 入场 (真正的 65% 胜率)
```
**预期**: 胜率更加可靠, 高置信信号更少但更准

#### C. 多时间尺度集成 (优先级: ⭐⭐)
```
当前: 仅 T+5 标签
改进: 同时训练 T+1, T+3, T+5, T+10 四个模型
      2/4 以上看多才入场
```
**预期**: 减少单时间尺度噪声, 胜率 +2-3pp

### 中期 (需额外数据/工程, 预计 1-2 周)

#### D. 截面排序 (优先级: ⭐⭐⭐)
```
当前: 对每个信号独立二元判断
改进: 每天对所有候选信号排序, 只买 Top-K (K=3~5)
      同一天最好的 N 个信号
      用 LightGBM predict_proba 排序
```
**预期**: 利用每日截面信息, 淘汰日内次优信号

#### E. 板块轮动 + 个股双层模型 (优先级: ⭐⭐⭐)
```
Layer 1 (板块级): 今天哪些板块值得交易?
    因子: 板块涨跌幅排名, 涨停家数, 资金流入
    输出: 板块交易开关
    
Layer 2 (个股级): 板块内哪只股票信号最好?
    因子: B2 信号 + OAMV + 板块内排名
    输出: 排序后选 Top-3
```
**预期**: 最大增量收益来源, 当前策略完全缺失板块维度

#### F. 新因子集成

| 类别 | 因子 | 预期贡献 | 数据来源 |
|------|------|:--:|------|
| 资金流 | 主力净流入, 大单占比 | ⭐⭐⭐ | tushare money_flow |
| 板块 | 板块强度, 板块内排名 | ⭐⭐⭐ | tushare concept |
| 大盘 | 上证涨跌, 涨跌比, VIX | ⭐⭐⭐ | tushare index |
| 情绪 | 涨停数, 连板高度, 炸板率 | ⭐⭐ | tushare limit_list |
| 波动率 | ATR, 布林带宽度 | ⭐⭐ | 已可计算 |
| 量价 | OBV, MFI, VWAP | ⭐⭐ | 已可计算 |

### 长期 (方法论升级)

#### G. 强化学习 (优先级: ⭐)
```
State:  OAMV regime, 大盘趋势, 波动率, 近期胜率
Action: 调整 18 个参数 (连续 → 离散化)
Reward: T+5 收益
算法:   PPO 或 SAC
```
**预期**: 参数自适应市场, 长期更稳健

#### H. 在线学习 (优先级: ⭐⭐)
```
当前: 一次训练, 参数固定 (2026-05-10)
改进: 每月/每周重新训练 LightGBM
      模型随市场演化自适应
      保留模型版本, AB 测试
```

---

## 六、下次运行清单

```bash
# 1. 验证当前最优参数
cd ML_optimization/
python phase2c_oamv_grid_search.py   # Sharpe 1.770 基准

# 2. 实现动态阈值 (短期改进 A)
#    修改 strategy_b2.py, 将 gain_min 等改为 oamv_regime 函数

# 3. 添加新因子 (中期改进 F)
#    新增 phase5_new_factors.py, 集成资金流/板块/大盘因子

# 4. 重新运行 Phase 1 → Phase 4
python phase1_feature_importance.py   # 包含新因子的重要性排名
python phase2_grid_search.py          # 新参数组合
python phase3_gamma_2d.py             # 新阈值优化
python phase4_lightgbm_classifier.py  # 新模型训练

# 5. 对比新旧结果
python phase_compare.py               # (需新建) 自动对比所有阶段
```

---

## 七、文件索引

| 文件 | 说明 |
|------|------|
| `quant_strategy/strategy_b2.py` | **B2 策略核心** (已应用最优参数) |
| `quant_strategy/oamv.py` | OAMV 活筹指数指标 |
| `ML_optimization/phase1c_oamv_features.py` | 特征重要性 (RF) |
| `ML_optimization/phase2c_oamv_grid_search.py` | 网格搜索 (仿真引擎) |
| `ML_optimization/phase3_gamma_2d.py` | 2D Gamma 曲面优化 |
| `ML_optimization/phase4_lightgbm_classifier.py` | LightGBM 信号分类 |
| `ML_optimization/phase2c_oamv_results.json` | **Phase 2c 最优结果** |
| `ML_optimization/phase3_gamma_optimal.json` | Phase 3 最优阈值 |
| `ML_optimization/phase4_lightgbm_results.json` | Phase 4 过滤效果 |
| `ML_optimization/phase4_lgbm_model.txt` | 训练好的 LightGBM 模型 |

---

## 八、已知问题 / 注意事项

1. **eval_params 等权问题**: Phase 3 使用等权 T+5 收益, 仓位权重参数不敏感。改进: 用 b2_position_weight 加权。
2. **Gamma 外推不稳定**: 网格最优 > Gamma 解析最优。下次使用纯网格或贝叶斯优化替代。
3. **Phase 2c 样本偏差**: 仅 100 笔交易, 防御性买入仅 10 笔。扩大回测期或增加股票数可缓解。
4. **LightGBM 模型存储**: `phase4_lgbm_model.txt` 在 git 中, 可直接加载使用。
5. **GBK 编码**: Windows 下 print 希腊字母会崩溃, 脚本已统一使用 ASCII。
