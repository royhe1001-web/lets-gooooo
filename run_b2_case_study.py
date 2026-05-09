#!/usr/bin/env python3
"""
b2 策略成功案例分析
====================
对指定股票在信号日前后的走势进行深度分析，
提炼 b2 策略优化的共性特征。

用法:
  python run_b2_case_study.py                        # 分析全部10个案例
  python run_b2_case_study.py 002987 300418           # 分析指定案例
  python run_b2_case_study.py --before 10 --after 20  # 自定义观察窗口
"""

import argparse, os, sys, time, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
from quant_strategy.indicators import calc_all_indicators
from quant_strategy.strategy_b2 import generate_b2_signals

DATA_DIR = os.path.join(ROOT, 'fetch_kline', 'stock_kline')
OUT = os.path.join(ROOT, 'output')

# 案例股: (代码, 信号日期)
CASES = [
    ('002987', '2025-06-16'),
    ('920039', '2025-07-08'),
    ('688170', '2025-07-16'),
    ('300418', '2025-08-11'),
    ('601778', '2025-08-07'),
    ('688787', '2025-07-08'),
    ('002345', '2025-05-16'),
    ('688202', '2025-06-23'),
    ('688318', '2025-08-11'),
    ('920748', '2025-05-26'),
]


def load_stock(code):
    path = os.path.join(DATA_DIR, f'{code}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['date'],
                     dtype={'open': np.float32, 'close': np.float32,
                            'high': np.float32, 'low': np.float32, 'volume': np.float32})
    return df.set_index('date').sort_index()


def analyze_case(code, signal_date, before=10, after=20):
    """深度分析单个案例"""
    stock = load_stock(code)
    if stock is None:
        return {'code': code, 'error': '数据不存在'}

    target = pd.Timestamp(signal_date)
    if target not in stock.index:
        return {'code': code, 'error': f'信号日 {signal_date} 不在数据中'}

    # 计算指标
    stock = calc_all_indicators(stock, board_type='main')
    stock = generate_b2_signals(stock, board_type='main', precomputed=True)

    target_loc = stock.index.get_loc(target)

    # 确认 b2 信号
    row = stock.iloc[target_loc]
    b2_signal = int(row.get('b2_entry_signal', 0))
    b2_weight = float(row.get('b2_position_weight', 0))

    prev = stock.iloc[target_loc - 1]

    # --- 信号日详情 ---
    gain_signal = (row['close'] / prev['close'] - 1) * 100
    j_prev = float(prev['J'])
    j_now = float(row['J'])

    # b2 七条件
    c1 = j_prev < 20
    c2 = gain_signal > 4
    c3 = j_now < 65
    c4 = row['volume'] > prev['volume']
    c5 = (row['high'] - row['close']) / row['close'] < 0.035
    c6 = row['white_line'] > row['yellow_line']
    c7 = row['close'] > row['yellow_line']
    conditions_met = sum([c1, c2, c3, c4, c5, c6, c7])

    # --- 信号前特征 ---
    pre_start = max(0, target_loc - before)
    pre_slice = stock.iloc[pre_start:target_loc]

    # 回调深度
    pre_high = pre_slice['close'].max()
    pre_low = pre_slice['close'].min()
    pullback_pct = (pre_low / pre_high - 1) * 100

    # 信号前5日 J 值序列
    j_series = []
    for i in range(min(before, target_loc), 0, -1):
        r = stock.iloc[target_loc - i]
        j_series.append(round(float(r['J']), 1))

    # 信号前缩量情况
    pre_vol = pre_slice['volume']
    vol_trend = "缩量" if pre_vol.tail(3).mean() < pre_vol.head(3).mean() else "放量"

    # --- 信号后走势 ---
    post_end = min(len(stock) - 1, target_loc + after)
    post_slice = stock.iloc[target_loc + 1:post_end + 1]

    if len(post_slice) == 0:
        return {'code': code, 'error': '信号日后无数据'}

    post_high = post_slice['close'].max()
    post_low = post_slice['close'].min()

    # 最大涨幅 (从信号日收盘到后续最高)
    max_gain = (post_high / row['close'] - 1) * 100
    # 最大回撤 (从信号日收盘到后续最低)
    max_dd = (post_low / row['close'] - 1) * 100

    # 涨幅时序
    gains = []
    for i in range(1, min(after, len(post_slice)) + 1):
        r = stock.iloc[target_loc + i]
        g = (r['close'] / row['close'] - 1) * 100
        gains.append(round(g, 1))

    # T+1 ~ T+5 是否盈利
    t1_gain = gains[0] if len(gains) >= 1 else None
    t3_gain = gains[2] if len(gains) >= 3 else None
    t5_gain = gains[4] if len(gains) >= 5 else None

    # 后续是否触发止损
    post_start = target_loc + 1
    post_end_check = min(len(stock) - 1, target_loc + after)
    triggered_stop = False
    stop_reason = ''
    entry_low = row['low']
    yellow = row['yellow_line']

    for i in range(post_start, post_end_check + 1):
        r = stock.iloc[i]
        # 蜡烛破位止损
        if r['close'] < entry_low:
            triggered_stop = True
            stop_reason = f"T+{i-target_loc} 跌破入场日最低价 {entry_low:.2f}"
            break
        # 黄线跌破
        if r['close'] < yellow:
            triggered_stop = True
            stop_reason = f"T+{i-target_loc} 跌破黄线 {yellow:.2f}"
            break

    return {
        'code': code,
        'signal_date': signal_date,
        'b2_signal': b2_signal,
        'b2_weight': b2_weight,
        'c1_j_prev': c1, 'c2_gain4': c2, 'c3_j55': c3,
        'c4_vol': c4, 'c5_shadow': c5, 'c6_white': c6, 'c7_close': c7,
        'conditions_met': conditions_met,
        'j_prev': j_prev, 'j_now': j_now,
        'gain_signal': gain_signal,
        'close': float(row['close']),
        'open': float(row['open']),
        'high': float(row['high']), 'low': float(row['low']),
        'volume': float(row['volume']),
        'prev_volume': float(prev['volume']),
        'vol_ratio': float(row['volume'] / prev['volume']) if prev['volume'] > 0 else 0,
        'white_line': float(row['white_line']),
        'yellow_line': float(row['yellow_line']),
        'pullback_pct': pullback_pct,
        'vol_trend': vol_trend,
        'j_series': j_series,
        'max_gain': max_gain,
        'max_dd': max_dd,
        'gains': gains,
        't1_gain': t1_gain, 't3_gain': t3_gain, 't5_gain': t5_gain,
        'triggered_stop': triggered_stop,
        'stop_reason': stop_reason,
        'pre_days': len(pre_slice),
        'post_days': len(post_slice),
    }


def print_case(result, before=10, after=20):
    """打印单个案例的详细报告"""
    if 'error' in result:
        print(f"\n  {result['code']}: ⚠ {result['error']}")
        return

    print(f"\n{'='*80}")
    print(f"  {result['code']} — 信号日: {result['signal_date']}")
    print(f"{'='*80}")

    # 信号确认
    sig = "✓ b2信号确认" if result['b2_signal'] else "✗ 注意：策略未识别为b2信号"
    print(f"\n  信号状态: {sig}  权重: {result.get('b2_weight', 0):.1f}x  条件满足: {result['conditions_met']}/7")

    # 信号日关键数据
    print(f"\n  ┌─ 信号日数据 ───────────────────────────────────────┐")
    print(f"  │ O:{result['open']:.2f} H:{result['high']:.2f} "
          f"L:{result['low']:.2f} C:{result['close']:.2f}         │")
    print(f"  │ 涨幅: {result['gain_signal']:+.2f}%  "
          f"量比: {result['vol_ratio']:.1f}x                          │")
    print(f"  │ J前日: {result['j_prev']:.1f} → J当日: {result['j_now']:.1f}                         │")
    print(f"  │ 白线: {result['white_line']:.2f}  黄线: {result['yellow_line']:.2f}              │")
    print(f"  └──────────────────────────────────────────────────┘")

    # 七条件检查
    print(f"\n  ┌─ b2 七条件 ────────────────────────────────────────┐")
    conds = [
        ('J前日<20', result['c1_j_prev']),
        ('涨幅>4%', result['c2_gain4']),
        ('J当日<65', result['c3_j55']),
        ('放量', result['c4_vol']),
        ('上影线<3.5%', result['c5_shadow']),
        ('白>黄', result['c6_white']),
        ('收>黄', result['c7_close']),
    ]
    for name, met in conds:
        print(f"  │ {'✓' if met else '✗'} {name:<30}                        │")
    print(f"  └──────────────────────────────────────────────────┘")

    # 信号前走势
    print(f"\n  ┌─ 信号前 {before} 日特征 ─────────────────────────────────┐")
    print(f"  │ 回调深度: {result['pullback_pct']:+.1f}%                                    │")
    print(f"  │ 量能趋势: {result['vol_trend']}                                        │")
    j_str = ' → '.join(str(j) for j in result['j_series'][-5:])
    print(f"  │ 前5日 J: {j_str}                      │")
    print(f"  └──────────────────────────────────────────────────┘")

    # 信号后走势
    print(f"\n  ┌─ 信号后 {after} 日走势 ──────────────────────────────────┐")
    print(f"  │ 最大涨幅: {result['max_gain']:+.2f}%                                     │")
    print(f"  │ 最大回撤: {result['max_dd']:+.2f}%                                     │")
    print(f"  │ T+1: {result['t1_gain']:+.2f}%  T+3: {result['t3_gain']:+.2f}%  "
          f"T+5: {result['t5_gain']:+.2f}%                     │")
    if result['triggered_stop']:
        print(f"  │ ⚠ 后续触发止损: {result['stop_reason']}  │")
    else:
        print(f"  │ ✓ 未触发止损                                        │")
    print(f"  └──────────────────────────────────────────────────┘")

    # 每日涨幅
    print(f"\n  后续每日涨幅 (相对信号日收盘):")
    for i, g in enumerate(result['gains'][:15]):
        bar = '█' * max(1, int(abs(g) / 2)) if abs(g) > 0 else ''
        direction = '+' if g >= 0 else ''
        print(f"    T+{i+1:<3} {direction}{g:>6.1f}%  {bar}")


def summarize_all(results):
    """汇总分析所有案例的共性特征"""
    valid = [r for r in results if 'error' not in r]
    if not valid:
        print("无有效案例")
        return

    print(f"\n\n{'='*80}")
    print(f"  案例汇总分析 ({len(valid)}/{len(results)} 有效)")
    print(f"{'='*80}")

    # 统计
    b2_confirmed = sum(1 for r in valid if r['b2_signal'])
    print(f"\n  b2 策略确认率: {b2_confirmed}/{len(valid)} ({b2_confirmed/len(valid)*100:.0f}%)")

    # 条件满足率
    print(f"\n  ┌─ 条件满足率 ──────────────────────────────────────┐")
    for cond_name, key in [
        ('J前日<20', 'c1_j_prev'), ('涨幅>4%', 'c2_gain4'),
        ('J当日<55', 'c3_j55'), ('放量', 'c4_vol'),
        ('无上影线', 'c5_shadow'), ('白>黄', 'c6_white'), ('收>黄', 'c7_close')
    ]:
        count = sum(1 for r in valid if r[key])
        bar = '█' * (count * 50 // len(valid))
        print(f"  │ {cond_name:<16} {count}/{len(valid):<5} {bar}")

    # 条件全满足的案例
    all_met = [r for r in valid if r['conditions_met'] == 7]
    print(f"  └────────────────────────────────────────────────┘")
    print(f"\n  7条件全满足: {len(all_met)}/{len(valid)}")

    # 部分满足但仍是成功案例
    partial = [r for r in valid if r['conditions_met'] < 7]
    if partial:
        print(f"\n  部分满足但获利的案例 (条件<7):")
        for r in partial:
            missed = []
            if not r['c1_j_prev']: missed.append(f"J前日={r['j_prev']:.1f}(≥20)")
            if not r['c2_gain4']: missed.append(f"涨幅={r['gain_signal']:.1f}%(≤4%)")
            if not r['c3_j55']: missed.append(f"J当日={r['j_now']:.1f}(≥55)")
            if not r['c4_vol']: missed.append("未放量")
            if not r['c5_shadow']: missed.append("有上影线")
            if not r['c6_white']: missed.append("白<黄")
            if not r['c7_close']: missed.append("收<黄")
            print(f"    {r['code']} 满足{r['conditions_met']}/7  缺失: {', '.join(missed)}  "
                  f"T+5={r['t5_gain']:+.1f}%")

    # 数值分布
    print(f"\n  ┌─ 数值分布 ────────────────────────────────────────┐")
    j_prevs = [r['j_prev'] for r in valid]
    print(f"  │ J前日:     均值 {np.mean(j_prevs):.1f}  中位数 {np.median(j_prevs):.1f}  "
          f"范围 [{min(j_prevs):.1f}, {max(j_prevs):.1f}]")

    j_nows = [r['j_now'] for r in valid]
    print(f"  │ J当日:     均值 {np.mean(j_nows):.1f}  中位数 {np.median(j_nows):.1f}  "
          f"范围 [{min(j_nows):.1f}, {max(j_nows):.1f}]")

    gains = [r['gain_signal'] for r in valid]
    print(f"  │ 当日涨幅:  均值 {np.mean(gains):.1f}%  中位数 {np.median(gains):.1f}%  "
          f"范围 [{min(gains):.1f}%, {max(gains):.1f}%]")

    pullbacks = [r['pullback_pct'] for r in valid]
    print(f"  │ 前10日回调: 均值 {np.mean(pullbacks):.1f}%  中位数 {np.median(pullbacks):.1f}%  "
          f"范围 [{min(pullbacks):.1f}%, {max(pullbacks):.1f}%]")

    vol_ratios = [r['vol_ratio'] for r in valid]
    print(f"  │ 量比:      均值 {np.mean(vol_ratios):.1f}x  中位数 {np.median(vol_ratios):.1f}x  "
          f"范围 [{min(vol_ratios):.1f}, {max(vol_ratios):.1f}]")

    # 收益分布
    t1s = [r['t1_gain'] for r in valid if r['t1_gain'] is not None]
    t3s = [r['t3_gain'] for r in valid if r['t3_gain'] is not None]
    t5s = [r['t5_gain'] for r in valid if r['t5_gain'] is not None]
    max_gs = [r['max_gain'] for r in valid]

    print(f"  └────────────────────────────────────────────────┘")
    print(f"\n  ┌─ 收益分布 ────────────────────────────────────────┐")
    print(f"  │ T+1 收益:  均值 {np.mean(t1s):+.1f}%  胜率 {sum(1 for g in t1s if g>0)/len(t1s)*100:.0f}%")
    print(f"  │ T+3 收益:  均值 {np.mean(t3s):+.1f}%  胜率 {sum(1 for g in t3s if g>0)/len(t3s)*100:.0f}%")
    print(f"  │ T+5 收益:  均值 {np.mean(t5s):+.1f}%  胜率 {sum(1 for g in t5s if g>0)/len(t5s)*100:.0f}%")
    print(f"  │ 最大收益:  均值 {np.mean(max_gs):+.1f}%")
    print(f"  └────────────────────────────────────────────────┘")

    stops = sum(1 for r in valid if r['triggered_stop'])
    print(f"\n  后续触发止损: {stops}/{len(valid)}")

    # --- 优化建议 ---
    print(f"\n{'─'*80}")
    print(f"  优化方向分析")
    print(f"{'─'*80}")

    # 分析条件宽松空间
    j_prev_loose = [r for r in valid if not r['c1_j_prev'] and r['t5_gain'] and r['t5_gain'] > 0]
    gain_loose = [r for r in valid if not r['c2_gain4'] and r['t5_gain'] and r['t5_gain'] > 0]
    j_now_loose = [r for r in valid if not r['c3_j55'] and r['t5_gain'] and r['t5_gain'] > 0]
    vol_loose = [r for r in valid if not r['c4_vol'] and r['t5_gain'] and r['t5_gain'] > 0]
    shadow_loose = [r for r in valid if not r['c5_shadow'] and r['t5_gain'] and r['t5_gain'] > 0]

    print(f"\n  宽松后仍获利的案例数 (T+5>0):")
    print(f"    J前日≥20 仍获利: {len(j_prev_loose)} 只")
    if j_prev_loose:
        for r in j_prev_loose:
            print(f"      {r['code']} J前={r['j_prev']:.1f} T+5={r['t5_gain']:+.1f}%")

    print(f"    涨幅≤4% 仍获利: {len(gain_loose)} 只")
    if gain_loose:
        for r in gain_loose:
            print(f"      {r['code']} 涨幅={r['gain_signal']:.1f}% T+5={r['t5_gain']:+.1f}%")

    print(f"    J当日≥55 仍获利: {len(j_now_loose)} 只")
    if j_now_loose:
        for r in j_now_loose:
            print(f"      {r['code']} J当日={r['j_now']:.1f} T+5={r['t5_gain']:+.1f}%")

    print(f"    未放量 仍获利: {len(vol_loose)} 只")
    if vol_loose:
        for r in vol_loose:
            print(f"      {r['code']} 量比={r['vol_ratio']:.1f} T+5={r['t5_gain']:+.1f}%")

    print(f"    有上影线 仍获利: {len(shadow_loose)} 只")
    if shadow_loose:
        for r in shadow_loose:
            print(f"      {r['code']} T+5={r['t5_gain']:+.1f}%")

    # 成交量趋势
    shrink_wins = [r for r in valid if r['vol_trend'] == '缩量' and r['t5_gain'] and r['t5_gain'] > 0]
    expand_wins = [r for r in valid if r['vol_trend'] == '放量' and r['t5_gain'] and r['t5_gain'] > 0]
    print(f"\n  量能趋势与收益:")
    print(f"    缩量后入场获利: {len(shrink_wins)} 只")
    print(f"    放量后入场获利: {len(expand_wins)} 只")


def parse_args():
    parser = argparse.ArgumentParser(description='b2 策略成功案例深度分析')
    parser.add_argument('codes', nargs='*', help='要分析的股票代码（默认分析全部10个案例）')
    parser.add_argument('--before', '-b', type=int, default=10, help='信号前观察窗口天数（默认10）')
    parser.add_argument('--after', '-a', type=int, default=20, help='信号后观察窗口天数（默认20）')
    parser.add_argument('--summary-only', '-s', action='store_true', help='仅显示汇总，跳过逐股详情')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.codes:
        # 用户指定代码时需要同时提供日期
        # 从 CASES 中查找
        cases = [(c, d) for c, d in CASES if c in args.codes]
        if len(cases) < len(args.codes):
            print("警告: 部分代码不在预定义案例中，请使用 --date 指定日期")
            # For now, just use the ones we find
    else:
        cases = CASES

    print("=" * 80)
    print("  b2 策略成功案例深度分析")
    print(f"  共 {len(cases)} 个案例 | 前 {args.before} 日 → 后 {args.after} 日")
    print("=" * 80)

    results = []
    for code, date_str in cases:
        print(f"\n>>> 分析 {code} ({date_str})...", flush=True)
        r = analyze_case(code, date_str, args.before, args.after)
        results.append(r)
        if not args.summary_only:
            print_case(r, args.before, args.after)

    # 汇总
    summarize_all(results)

    # 导出
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, 'b2_case_study.txt')
    with open(fp, 'w', encoding='utf-8') as f:
        for r in results:
            if 'error' in r:
                f.write(f"\n{r['code']}: {r['error']}\n")
            else:
                f.write(f"\n{'='*60}\n{r['code']} {r['signal_date']}\n{'='*60}\n")
                f.write(f"  信号: b2={r['b2_signal']} 权重={r['b2_weight']:.1f}x  "
                        f"条件={r['conditions_met']}/7\n")
                f.write(f"  J前={r['j_prev']:.1f} J今={r['j_now']:.1f}  "
                        f"涨幅={r['gain_signal']:+.1f}% 量比={r['vol_ratio']:.1f}x\n")
                f.write(f"  T+1={r['t1_gain']:+.1f}% T+3={r['t3_gain']:+.1f}% "
                        f"T+5={r['t5_gain']:+.1f}% 最大={r['max_gain']:+.1f}%\n")
                f.write(f"  回调={r['pullback_pct']:+.1f}% 量趋势={r['vol_trend']}  "
                        f"止损={'Y' if r['triggered_stop'] else 'N'}\n")
    print(f"\n结果已导出: {fp}")


if __name__ == '__main__':
    main()
