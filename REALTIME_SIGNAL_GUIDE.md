# 实时日线拉取 + 信号生成指南

> 来源：dualstream_csi500 策略的 `real_time_signal.py`，已验证可用  
> 目标：使 spring_b2 策略也具备 14:50 拉取实时行情 + 生成交易信号的能力

---

## 一、实时行情拉取：新浪 API

### 接口

```
GET https://hq.sinajs.cn/list=sh600519,sz000858,sh601198,...
Headers: Referer: https://finance.sina.com.cn
```

### 特点

| 项目 | 说明 |
|:---|:---|
| 无需认证 | 免费，无 token，无频率限制（实测 0.1s/批 不封） |
| 编码 | gbk |
| 覆盖 | 全 A 股（沪 sh + 深 sz），含停牌 |
| 延迟 | 约 3-5 秒（盘中实时，收盘后仍可拉取当日数据） |
| 批量 | 单次最多约 200-300 只（URL 超长会失败，建议 200 只/批） |

### 返回字段解析

以 `var hq_str_sh601198="东兴证券,13.250,..."` 为例：

| 索引 | 字段 | 含义 |
|:---:|:---|:---|
| 0 | name | 股票名称 |
| 1 | open | 今日开盘价 |
| 2 | pre_close | 昨日收盘价 |
| 3 | price | 当前价格（盘中=最新价，收盘后=收盘价） |
| 4 | high | 今日最高价 |
| 5 | low | 今日最低价 |
| 8 | volume | 成交量（股） |
| 9 | amount | 成交额（元） |
| 30 | date | 日期 YYYY-MM-DD |
| 31 | time | 时间 HH:MM:SS |

### 完整实现

```python
import requests
import time
import numpy as np
import pandas as pd

def fetch_sina_realtime(symbols: list) -> pd.DataFrame:
    """拉取新浪实时行情。symbols = ['sh600580','sz002155',...]"""
    session = requests.Session()
    session.trust_env = False

    batch_size = 200
    all_data = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        url = 'https://hq.sinajs.cn/list=' + ','.join(batch)
        r = session.get(url,
                        headers={'Referer': 'https://finance.sina.com.cn'},
                        timeout=10)
        r.encoding = 'gbk'
        for line in r.text.strip().split('\n'):
            if '=' not in line:
                continue
            code_str, quote_str = line.split('=', 1)
            code = code_str.strip().split('_')[-1]  # var hq_str_sh600580 → sh600580
            quote_str = quote_str.strip('";\n ')
            if not quote_str:
                continue
            fields = quote_str.split(',')
            if len(fields) < 32:
                continue
            all_data.append({
                'symbol': code[2:],       # 600580 (去掉sh/sz前缀)
                'name': fields[0],
                'open': float(fields[1]) if fields[1] else np.nan,
                'pre_close': float(fields[2]) if fields[2] else np.nan,
                'price': float(fields[3]) if fields[3] else np.nan,
                'high': float(fields[4]) if fields[4] else np.nan,
                'low': float(fields[5]) if fields[5] else np.nan,
                'volume': float(fields[8]) if fields[8] else 0,
                'amount': float(fields[9]) if fields[9] else 0,
            })
        time.sleep(0.1)  # 礼貌间隔

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.set_index('symbol')
    return df
```

### 代码转新浪格式

```python
def to_sina_format(symbols: list) -> list:
    """600519 → sh600519, 000858 → sz000858"""
    result = []
    for s in symbols:
        if s.startswith(('60', '68', '9')):
            result.append(f'sh{s}')
        else:
            result.append(f'sz{s}')
    return result
```

### 性能

- 495 只（CSI500）→ 3 批 → 约 3-4 秒
- 2000 只（全 A）→ 10 批 → 约 8-12 秒

---

## 二、实时信号生成工作流

### 整体流程（5 步，总耗时 < 10 秒）

```
1/5 拉取实时行情 (新浪API, 3-4s)
     ↓
2/5 加载策略模型 (LightGBM/XGBoost, <0.5s)
     ↓
3/5 加载OAMV/市场状态 (<0.1s)
     ↓
4/5 加载当前持仓 (json, <0.1s)
     ↓
5/5 生成交易信号 (因子截面 + 预测 + 离场评估 + 选股, <1s)
     ↓
  输出: 买卖指令清单 + 信号文件(JSON)
```

### 关键依赖

```
因子面板 (factor_panel.parquet)   ← 每日增量更新，用于生成最新截面
模型文件 (lgb_model.txt + metadata.pkl + preprocessor.pkl)  ← 训练后固定
市场成交额 (market_turnover.csv)  ← 用于 OAMV 状态检测
当前持仓 (current_positions.json) ← 如有持仓则评估离场
```

### 信号输出格式

```json
{
  "date": "2026-05-14",
  "state": "bullish",
  "sells": [
    {"symbol": "601198", "action": "SELL", "reasons": "排名92%", "return": -0.038}
  ],
  "holds": [
    {"symbol": "600177", "reason": "未满T+4", "held_days": 3, "return": 0.012}
  ],
  "buys": [
    {"symbol": "601666", "side": "sat", "score": 0.503, "price": 7.84, "shares": 3200}
  ]
}
```

### 执行方式

```bash
# 仅查看信号（模拟）
python real_time_signal.py

# 保存信号到文件 (output/signals/YYYYMMDD.json)
python real_time_signal.py --execute
```

---

## 三、spring_b2 适配要点

### 已有的（可直接复用）

- `fetch_kline.py` / `fetch_kline_ts.py` — 日线历史数据，无需修改
- `oamv.py` — 市场状态检测，如已有则复用
- 模型文件 — 训练好的模型加载逻辑已有

### 需要新增的

1. **`real_time_signal.py`** — 核心脚本，外壳复制自 CSI500，内部替换：
   - 股票池：CSI500 → B2 股票池（中盘/创业板/你的选股范围）
   - 模型加载路径：`output/dualstream_csi500_model` → spring_b2 的模型路径
   - 信号生成逻辑：替换为 spring_b2 的选股/离场规则

2. **新浪实时行情** — 上面的 `fetch_sina_realtime()` 直接可用，与策略无关

3. **定时执行** — Windows 任务计划 / Mac crontab：
   ```
   # 每周一至周五 14:50
   50 14 * * 1-5 cd /path/to/spring_b2 && python real_time_signal.py --execute
   ```

### 最简集成方式

如果 spring_b2 已有选股函数 `select_stocks(factor_panel, state)`，那么实时信号只需要：

```python
# 1. 拉行情
rt = fetch_sina_realtime(to_sina_format(b2_pool))
# 2. 加载模型
model = load_spring_b2_model()
# 3. 取最新因子截面
X_t = panel.xs(panel.index.get_level_values('date').max(), level='date')
# 4. 预测
scores = model.predict(X_t.fillna(0))
# 5. 调你自己的选股逻辑
signals = select_stocks(scores, prices=rt['price'], state=detect_state())
# 6. 输出买卖单
print_signals(signals)
```

---

## 四、与 Tushare 实时方案对比

| | 新浪 API | Tushare Pro |
|:---|:---|:---|
| 费用 | 免费 | 需积分 |
| 延迟 | 3-5s | 实时 |
| 频率限制 | 宽松 | 严格（积分对应 QPS） |
| 覆盖 | 全 A 股 | 全 A 股 |
| 稳定性 | 高（CDN 加速） | 依赖 token 额度 |
| 适用场景 | 收盘前拉一把 | 日内高频/盘中监控 |

**建议**: 收盘前信号用新浪（免费够用），盘中高频才需要 tushare。
