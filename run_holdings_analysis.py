#!/usr/bin/env python3
"""
持仓分析与策略对比
================================
1. 运行 b2 + brick 策略全市场扫描
2. 深度分析指定持仓股
3. 强度评分排名
4. 对比持仓 vs Top pick，给出持有/换仓建议

用法:
  python run_holdings_analysis.py 000960 601138          # 分析指定股票
  python run_holdings_analysis.py 000960,601138          # 逗号分隔
  python run_holdings_analysis.py --date 2026-05-06 000960 601138
  python run_holdings_analysis.py                        # 交互式输入
"""

import argparse
import os, sys, time, warnings
import pandas as pd
import numpy as np

# Fix Windows GBK encoding issues
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
from quant_strategy.indicators import calc_all_indicators, calc_brick_chart, calc_white_yellow_line
from quant_strategy.strategy_b2 import generate_b2_signals

from quant_strategy import load_stock_data
OUT = os.path.join(ROOT, 'output')

# 自动检测最新数据日期
def _get_latest_date(data_dir=None):
    import pandas as pd
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'fetch_kline', 'stock_kline')
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    max_date = None
    for f in files[:10]:
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['date'])
        m = df['date'].max()
        if max_date is None or m > max_date:
            max_date = m
    return max_date


def parse_args():
    parser = argparse.ArgumentParser(
        description='持仓分析与策略对比 — 输入股票代码，自动扫描比对',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python run_holdings_analysis.py 000960 601138
  python run_holdings_analysis.py 000960,601138,000070
  python run_holdings_analysis.py --date 2026-05-06 000960
        ''')
    parser.add_argument('codes', nargs='*',
                       help='股票代码（纯数字），多个用空格或逗号分隔。留空则交互输入')
    parser.add_argument('--date', '-d', default=None,
                       help='分析日期 YYYY-MM-DD，默认使用数据最新日期')
    parser.add_argument('--output', '-o', default=None,
                       help='输出目录，默认 output/')
    parser.add_argument('--no-scan', action='store_true',
                       help='跳过全市场扫描，仅分析指定股票')
    parser.add_argument('--entry', action='append', default=[],
                       help='入场信息 CODE:DATE:PRICE，如 000960:2026-05-06:38.46 (可多次使用)')
    return parser.parse_args()


def parse_codes(raw_args):
    """解析股票代码: 支持 '000960 601138' 和 '000960,601138' 两种格式"""
    codes = []
    for arg in raw_args:
        for part in arg.replace(',', ' ').split():
            code = part.strip()
            if code.isdigit() and len(code) == 6:
                codes.append(code)
            elif code:
                print(f"  警告: 忽略无效代码 '{code}'（需要6位纯数字）")
    return codes


MIN_HIST = 120


def deep_analysis(code, stock, target_loc):
    """对持仓股做深度指标分析"""
    last = stock.iloc[target_loc]
    prev = stock.iloc[target_loc - 1]
    close = stock['close']
    vol = stock['volume']

    # 最近20日数据
    recent = stock.iloc[max(0, target_loc - 20):target_loc + 1]

    # 趋势分析
    ma5 = close.iloc[max(0, target_loc - 5):target_loc + 1].mean()
    ma10 = close.iloc[max(0, target_loc - 10):target_loc + 1].mean()
    ma20 = close.iloc[max(0, target_loc - 20):target_loc + 1].mean()
    ma60 = close.iloc[max(0, target_loc - 60):target_loc + 1].mean() if target_loc >= 60 else np.nan

    # 涨跌幅
    ret_1d = (last['close'] / prev['close'] - 1) * 100
    ret_5d = (last['close'] / close.iloc[max(0, target_loc - 5)] - 1) * 100
    ret_10d = (last['close'] / close.iloc[max(0, target_loc - 10)] - 1) * 100
    ret_20d = (last['close'] / close.iloc[max(0, target_loc - 20)] - 1) * 100

    # 量价关系
    vol_5 = vol.iloc[max(0, target_loc - 5):target_loc].mean()
    vol_20 = vol.iloc[max(0, target_loc - 20):target_loc].mean()
    vol_ratio_5 = last['volume'] / vol_5 if vol_5 > 0 else 1
    vol_ratio_20 = last['volume'] / vol_20 if vol_20 > 0 else 1

    # 振幅
    amp_5d = ((recent['high'].iloc[-6:].max() / recent['low'].iloc[-6:].min() - 1) * 100
              if len(recent) >= 6 else 0)

    # KDJ
    k, d, j = last['K'], last['D'], last['J']

    # MACD
    dif, dea, hist = last['MACD_DIF'], last['MACD_DEA'], last['MACD_HIST']

    # Brick
    brick_val = last['brick_val']
    brick_rising = last['brick_rising']
    brick_g2r = last['brick_green_to_red']
    brick_r2g = last['brick_red_to_green']
    green_streak = last['brick_green_streak']
    red_streak = last['brick_red_streak']

    # White/Yellow line
    white = last['white_line']
    yellow = last['yellow_line']
    white_above = last['white_above_yellow']
    white_dir = last['white_direction']

    # Volume patterns
    vol_double = last['vol_double']
    vol_floor = last['vol_floor']

    # 20日高低位
    hh20 = recent['high'].max()
    ll20 = recent['low'].min()
    pos_20 = (last['close'] - ll20) / (hh20 - ll20) * 100 if hh20 > ll20 else 50

    # 风险评分 (0-10, 越低越安全)
    risk_score = 0
    risks = []

    # 1. 跌破黄线风险
    if last['close'] < yellow:
        risk_score += 2.5
        risks.append(f"收盘价{last['close']:.2f} < 黄线{yellow:.2f}")
    # 2. 白线方向向下
    if white_dir < 0:
        risk_score += 1.5
        risks.append("白线方向向下")
    # 3. J值超买
    if j > 80:
        risk_score += 2.0
        risks.append(f"J值超买(J={j:.1f})")
    # 4. MACD顶背离风险
    if dif < dea and hist < 0:
        risk_score += 1.5
        risks.append("MACD死叉/绿柱")
    # 5. 高位风险
    if pos_20 > 80:
        risk_score += 1.5
        risks.append(f"20日高位(pos={pos_20:.0f}%)")
    # 6. 缩量
    if vol_ratio_20 < 0.5:
        risk_score += 1.0
        risks.append(f"缩量(量比={vol_ratio_20:.1f})")

    # 强度评分 (0-10)
    strength = 0
    # KDJ低位金叉
    if j < 30 and k > d:
        strength += 2.0
    elif j < 50 and k > d:
        strength += 1.0
    # MACD多头
    if dif > dea and hist > 0:
        strength += 2.0
    elif dif > dea:
        strength += 1.0
    # 白线 > 黄线
    if white > yellow:
        strength += 2.0
    # 砖红柱
    if brick_rising == 1:
        strength += 1.5
    # 量价配合
    if ret_1d > 0 and vol_ratio_5 > 1.2:
        strength += 0.5
    # 站上均线
    if not np.isnan(ma60) and last['close'] > ma60:
        strength += 1.0
    # 5日线 > 20日线
    if ma5 > ma20:
        strength += 1.0

    return {
        'code': code,
        'date': str(stock.index[target_loc].date()),
        'close': last['close'],
        'ret_1d': ret_1d,
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'ret_20d': ret_20d,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'vol_ratio_5': vol_ratio_5,
        'vol_ratio_20': vol_ratio_20,
        'amp_5d': amp_5d,
        'K': k, 'D': d, 'J': j,
        'MACD_DIF': dif, 'MACD_DEA': dea, 'MACD_HIST': hist,
        'brick_val': brick_val, 'brick_rising': brick_rising,
        'brick_green_to_red': brick_g2r, 'brick_red_to_green': brick_r2g,
        'green_streak': green_streak, 'red_streak': red_streak,
        'white_line': white, 'yellow_line': yellow,
        'white_above_yellow': white_above, 'white_direction': white_dir,
        'vol_double': vol_double, 'vol_floor': vol_floor,
        'pos_20d': pos_20,
        'risk_score': risk_score, 'risks': risks,
        'strength': strength,
        'b2_signal': 0, 'brick_signal': 0, 'brick_pattern': '',
        'b2_weight': 0.0,
    }


def check_stop_conditions(code, stock, target_loc, entry_date=None, entry_price=None):
    """检查止损条件 (与模拟交易逻辑一致).
       返回 stop_triggered, stop_details 列表
    """
    if entry_date is None or entry_price is None:
        return False, ['未提供入场信息, 无法检查止损 (使用 --entry CODE:DATE:PRICE)']

    last = stock.iloc[target_loc]
    close_today = float(last['close'])
    yellow = float(last.get('yellow_line', 0))
    signal_close = float(last.get('close', close_today))  # 用当日收盘近似信号收盘

    # 找入场日数据
    entry_day = pd.Timestamp(entry_date)
    if entry_day not in stock.index:
        return False, [f'入场日 {entry_date} 不在数据中']

    entry_loc = stock.index.get_loc(entry_day)
    entry_row = stock.iloc[entry_loc]
    entry_low = float(entry_row['low'])
    entry_close = float(entry_row['close'])

    days_held = target_loc - entry_loc
    float_pnl = (close_today / float(entry_price) - 1) * 100

    triggers = []
    urgent = False

    # 1. 蜡烛破位 (3%缓冲)
    candle_stop = entry_low * 0.97
    if close_today < candle_stop:
        urgent = True
        triggers.append(('⚠⚠ 蜡烛破位止损',
                        f'收盘 {close_today:.2f} < 入场低 {entry_low:.2f} × 0.97 = {candle_stop:.2f}',
                        '建议: 次日开盘离场'))

    # 2. 信号收盘破位 (用入场日收盘近似)
    if close_today < entry_close:
        triggers.append(('⚠ 信号收盘破位',
                        f'收盘 {close_today:.2f} < 入场收盘 {entry_close:.2f}',
                        '建议: 收盘前离场' if days_held >= 1 else 'T+1保护中, T+2开盘判断'))

    # 3. 黄线破位
    if yellow > 0 and close_today < yellow:
        urgent = True
        triggers.append(('⚠⚠ 黄线破位',
                        f'收盘 {close_today:.2f} < 黄线 {yellow:.2f}',
                        '建议: 收盘前离场'))

    # 4. J值超买告警
    j_val = float(last['J'])
    if j_val > 85:
        triggers.append(('⚡ J值超买',
                        f'J = {j_val:.1f} > 85, 回调风险高',
                        '建议: 关注明日走势, 考虑减仓'))

    # 5. MACD死叉
    dif = float(last['MACD_DIF'])
    dea = float(last['MACD_DEA'])
    if dif < dea and float(last['MACD_HIST']) < 0:
        triggers.append(('⚡ MACD死叉',
                        f'DIF {dif:.3f} < DEA {dea:.3f}, 绿柱',
                        '建议: 关注, 若持续走弱考虑离场'))

    # 6. 砖型图红转绿
    if int(last.get('brick_red_to_green', 0)) == 1:
        urgent = True
        triggers.append(('⚠⚠ 砖型图红转绿',
                        '卖出信号, 红砖转绿砖',
                        '建议: 按砖型图纪律离场'))

    # 综合
    info = {
        'code': code,
        'entry_date': entry_date,
        'entry_price': float(entry_price),
        'close_today': close_today,
        'entry_low': entry_low,
        'entry_close': entry_close,
        'float_pnl': float_pnl,
        'days_held': days_held,
        'yellow': yellow,
        'candle_stop': candle_stop,
        'urgent': urgent,
    }

    return info, triggers


def print_stop_check(info, triggers):
    """打印止损检查结果"""
    if isinstance(triggers, list) and len(triggers) > 0 and isinstance(triggers[0], str):
        # 错误信息
        print(f"\n  ┌─ 止损检查 ───────────────────────────────────┐")
        print(f"  │ {triggers[0]:<46} │")
        print(f"  └──────────────────────────────────────────────────┘")
        return

    if not triggers:
        print(f"\n  ┌─ 止损检查 ───────────────────────────────────┐")
        print(f"  │ ✓ 未触发任何止损条件                                         │")
        print(f"  └──────────────────────────────────────────────────┘")
        return

    urgent = info.get('urgent', False)
    header = '⚠⚠ 止损触发！' if urgent else '⚠ 风险告警'
    print(f"\n  ┌─ 止损检查 [{header}] ───────────────────┐")
    print(f"  │ 入场: {info['entry_date']} @{info['entry_price']:.2f}  "
          f"持仓{info['days_held']}天  浮盈{info['float_pnl']:+.1f}%")
    print(f"  │ 入场低: {info['entry_low']:.2f}  蜡烛止损线: {info['candle_stop']:.2f}  "
          f"黄线: {info['yellow']:.2f}")
    print(f"  │")

    for level, detail, suggestion in triggers:
        tag = '██' if '⚠⚠' in level else '──'
        print(f"  │ {tag} {level}")
        print(f"  │    {detail}")
        print(f"  │    {suggestion}")
        print(f"  │")

    print(f"  └──────────────────────────────────────────────────┘")


def check_rotation_advisory(code, stock, target_loc, entry_date=None, entry_price=None,
                            b2_signals=None, brick_signals=None):
    """检查轮换建议 (与模拟交易 checkpoint 逻辑一致).
       - 60日检查: 持仓≥60天+收益<15% → 建议轮换
       - 252日预警: 接近1年 → 准备清仓
       - 信号对比: 是否有更强的b2信号可替换
    """
    if entry_date is None or entry_price is None:
        return None

    last = stock.iloc[target_loc]
    entry_day = pd.Timestamp(entry_date)
    if entry_day not in stock.index:
        return None

    entry_loc = stock.index.get_loc(entry_day)
    days_held = target_loc - entry_loc
    entry_low = float(stock.iloc[entry_loc]['low'])
    close_today = float(last['close'])
    cum_return = (close_today / float(entry_price) - 1)
    cum_return_pct = cum_return * 100

    advices = []
    action = 'hold'  # hold / consider_rotate / suggest_rotate

    # 60日检查点
    if days_held >= 60 and cum_return < 0.15:
        advices.append({
            'level': '⚠ 轮换',
            'detail': f'持仓{days_held}天, 收益仅{cum_return_pct:+.1f}% (<15%阈值)',
            'suggestion': '建议换入新信号股，释放低效仓位'
        })
        action = 'suggest_rotate'
    elif days_held >= 45 and cum_return < 0.10:
        advices.append({
            'level': '⚡ 关注',
            'detail': f'持仓{days_held}天, 收益{cum_return_pct:+.1f}% (接近15%阈值)',
            'suggestion': f'再观察{60-days_held}天，如仍<15%则考虑换仓'
        })
        action = 'consider_rotate'

    # 252日预警
    if days_held >= 200:
        days_left = 252 - days_held
        advices.append({
            'level': '⏰ 1年预警',
            'detail': f'持仓{days_held}天, 距252天硬限剩{days_left}天',
            'suggestion': f'收益{cum_return_pct:+.1f}%, ' +
                         ('可择机离场' if cum_return > 0.3 else '到期强制清仓')
        })
        if action == 'hold':
            action = 'consider_rotate'

    # 砖转绿检查
    brick_r2g = int(last.get('brick_red_to_green', 0))
    brick_red = int(last.get('brick_red', 1))
    if days_held > 30 and (brick_r2g == 1 or brick_red == 0):
        advices.append({
            'level': '⚠ 砖转绿',
            'detail': f'砖型图转弱, 动量衰减',
            'suggestion': '关注是否形成下跌趋势，考虑减仓'
        })
        if action == 'hold':
            action = 'consider_rotate'

    # 信号对比 (需要外部传入 b2_signals)
    if b2_signals and days_held > 30:
        current_weight = 1.0  # default, user doesn't know exact weight
        stronger_signals = []
        for sig_code, (gain, j_prev, weight, row) in b2_signals.items():
            if sig_code != code and weight > current_weight + 1.0:
                stronger_signals.append((sig_code, weight, gain))
        if stronger_signals:
            stronger_signals.sort(key=lambda x: x[1], reverse=True)
            top = stronger_signals[:3]
            codes_str = ', '.join(f'{c}({w:.1f}x)' for c, w, _ in top)
            advices.append({
                'level': '💡 更强信号',
                'detail': f'市场存在权重新信号: {codes_str}',
                'suggestion': '可考虑换仓至更强信号'
            })
            if action == 'hold':
                action = 'consider_rotate'

    return {
        'code': code, 'days_held': days_held, 'cum_return_pct': cum_return_pct,
        'entry_price': float(entry_price), 'close_today': close_today,
        'action': action, 'advices': advices,
    }


def print_rotation_advisory(adv):
    """打印轮换建议"""
    if adv is None:
        print(f"\n  ┌─ 轮换检查 ───────────────────────────────────┐")
        print(f"  │ 未提供入场信息, 无法检查轮换 (--entry CODE:DATE:PRICE)      │")
        print(f"  └──────────────────────────────────────────────────┘")
        return

    labels = {'suggest_rotate': '⚠ 建议轮换', 'consider_rotate': '⚡ 关注轮换', 'hold': '✓ 继续持有'}
    print(f"\n  ┌─ 轮换检查 [{labels.get(adv['action'], '?')}] ──────────────┐")
    print(f"  │ 持仓 {adv['days_held']}天  入场@{adv['entry_price']:.2f}  "
          f"现价{adv['close_today']:.2f}  收益{adv['cum_return_pct']:+.1f}%")
    print(f"  │")

    if adv['advices']:
        for a in adv['advices']:
            print(f"  │ {a['level']}: {a['detail']}")
            print(f"  │    → {a['suggestion']}")
            print(f"  │")
    else:
        print(f"  │ ✓ 未触发任何轮换条件, 继续持有")
        print(f"  │")

    # Timeline bar
    days = adv['days_held']
    bar_60 = min(days / 60, 1.0)
    bar_252 = min(days / 252, 1.0)
    print(f"  │ 60日线: {'█' * int(bar_60*20)}{'░' * (20-int(bar_60*20))} "
          f"{'← 已过' if days >= 60 else str(60-days)+'天后'}")
    print(f"  │ 252日: {'█' * int(bar_252*20)}{'░' * (20-int(bar_252*20))} "
          f"{'← 已触发' if days >= 252 else str(252-days)+'天后清仓'}")
    print(f"  └──────────────────────────────────────────────────┘")


def compute_strength(row_dict):
    """综合评分（用于策略信号排名）"""
    score = 0.0

    j = row_dict['J']
    k = row_dict['K']
    d = row_dict['D']

    # J值位置: 低位 > 中位 > 高位
    if j < 20:
        score += 3.0
    elif j < 35:
        score += 2.0
    elif j < 50:
        score += 1.0
    elif j > 85:
        score -= 1.5

    # KDJ金叉
    if k > d:
        score += 1.5
        if j > k:
            score += 0.5  # J线领先

    # MACD
    if row_dict['MACD_DIF'] > row_dict['MACD_DEA']:
        score += 1.5
        if row_dict['MACD_HIST'] > 0:
            score += 0.5
            if row_dict['MACD_HIST'] > row_dict.get('macd_hist_prev', 0):
                score += 0.5  # 红柱放大

    # 白黄线
    if row_dict['white_line'] > row_dict['yellow_line']:
        score += 2.0
    if row_dict['white_direction'] > 0:
        score += 1.0

    # 砖型图
    if row_dict['brick_rising'] == 1:
        score += 1.0
    if row_dict['red_streak'] >= 3:
        score += 0.5

    # 涨幅 (有一定涨幅但不过热)
    ret_5d = row_dict['ret_5d']
    if 3 <= ret_5d <= 15:
        score += 1.0
    elif 0 < ret_5d < 3:
        score += 0.5

    return round(score, 1)


def main(holdings, target_date, do_scan=True, output_dir=None, entry_info=None):
    t_total = time.time()
    out = output_dir or OUT
    if entry_info is None:
        entry_info = {}

    print("=" * 70)
    print(f"  持仓分析与策略对比")
    print(f"  分析日期: {target_date.date()}  |  持仓: {', '.join(holdings) if holdings else '(仅扫描)'}")
    print("=" * 70)

    # ── 加载数据 ──
    print("\n[1/3] 加载数据...", flush=True)
    t0 = time.time()
    df = load_stock_data()
    print(f"  {len(df):,} 条, {df['ts_code'].nunique()} 只, {time.time()-t0:.1f}s", flush=True)

    # ── 预筛选 ──
    print("[2/3] 预筛选...", flush=True)
    t0 = time.time()
    di = df.groupby('ts_code')['date'].agg(['min', 'max', 'count'])
    valid = set()
    for c, r in di.iterrows():
        if r['max'] >= target_date and r['count'] >= MIN_HIST:
            valid.add(c)
    print(f"  候选: {len(valid)} 只, {time.time()-t0:.1f}s", flush=True)

    # ── 运行 ──
    if do_scan:
        print("[3/3] 运行策略 + 深度分析持仓...", flush=True)
    else:
        print("[3/3] 仅分析指定持仓（跳过全市场扫描）...", flush=True)
        valid = set(holdings)  # 仅处理持仓股

    b2_signals = {}       # code -> (gain, j_prev, weight, row_dict)
    brick_signals = {}    # code -> (pattern, row_dict)
    holdings_detail = {}  # code -> detail dict
    checked = 0
    errors = []
    t1 = time.time()

    for code, group in df.groupby('ts_code', sort=False):
        if code not in valid and code not in holdings:
            continue
        if code not in valid:
            continue

        stock = group.set_index('date').sort_index()
        if target_date not in stock.index:
            continue
        if len(stock) < MIN_HIST:
            continue

        try:
            stock = calc_all_indicators(stock, board_type='main')
            stock = generate_b2_signals(stock, board_type='main', precomputed=True)

            target_loc = stock.index.get_loc(target_date)
            last = stock.iloc[target_loc]
            prev = stock.iloc[target_loc - 1]

            # 构建row_dict
            ret_5d = last['close'] / stock['close'].iloc[max(0, target_loc - 5)] - 1
            ret_10d = last['close'] / stock['close'].iloc[max(0, target_loc - 10)] - 1
            ret_20d = last['close'] / stock['close'].iloc[max(0, target_loc - 20)] - 1
            row = {
                'code': code,
                'close': last['close'],
                'ret_1d': (last['close'] / prev['close'] - 1) * 100,
                'ret_5d': ret_5d * 100,
                'ret_10d': ret_10d * 100,
                'ret_20d': ret_20d * 100,
                'K': last['K'], 'D': last['D'], 'J': last['J'],
                'MACD_DIF': last['MACD_DIF'], 'MACD_DEA': last['MACD_DEA'],
                'MACD_HIST': last['MACD_HIST'],
                'brick_val': last['brick_val'],
                'brick_rising': last['brick_rising'],
                'brick_green_to_red': last['brick_green_to_red'],
                'brick_red_to_green': last['brick_red_to_green'],
                'green_streak': last['brick_green_streak'],
                'red_streak': last['brick_red_streak'],
                'white_line': last['white_line'],
                'yellow_line': last['yellow_line'],
                'white_above_yellow': last['white_above_yellow'],
                'white_direction': last['white_direction'],
                'vol_double': last['vol_double'],
                'vol_floor': last['vol_floor'],
                'amplitude': last['amplitude'],
            }

            # b2 signal
            if last.get('b2_entry_signal', 0) == 1:
                gain = row['ret_1d']
                j_prev = prev['J']
                weight = last.get('b2_position_weight', 1.0)
                b2_signals[code] = (gain, j_prev, weight, row)

            # Brick signal (白线>黄线 + 绿转红)
            if (last['white_line'] > last['yellow_line'] and
                last['brick_green_to_red'] == 1):
                # Pattern classification (inline simplified)
                pat = '定式四'
                # P1
                if target_loc >= 60:
                    cl = stock['close']
                    lo = stock['low']
                    ps, pe = max(0, target_loc - 60), target_loc - 5
                    if pe > ps:
                        pk_slice = cl.iloc[ps:pe]
                        if len(pk_slice) > 0:
                            pk_p = pk_slice.max()
                            pk_pos = stock.index.get_loc(pk_slice.idxmax())
                            tr_slice = lo.iloc[pk_pos:target_loc]
                            if len(tr_slice) > 0:
                                tr_p = tr_slice.min()
                                if (tr_p / pk_p - 1) * 100 <= -10:
                                    pls, ple = max(0, pk_pos - 80), pk_pos - 1
                                    if ple > pls and tr_p > lo.iloc[pls:ple].min():
                                        pat = '定式一'

                if pat == '定式四' and target_loc >= 20:
                    cur = stock.iloc[target_loc]
                    daily_g = (cur['close'] / prev['close'] - 1) * 100
                    if daily_g >= 3:
                        v20 = stock['volume'].iloc[max(0, target_loc - 20):target_loc].mean()
                        if v20 > 0 and cur['volume'] >= v20 * 1.5:
                            hi = stock['high']; lo = stock['low']
                            ce = target_loc - 1; cs = ce
                            for i in range(ce - 1, max(0, target_loc - 40), -1):
                                ph = hi.iloc[i:ce + 1].max()
                                pl = lo.iloc[i:ce + 1].min()
                                pa = cl.iloc[i:ce + 1].mean()
                                if pa > 0 and (ph - pl) / pa * 100 < 12:
                                    cs = i
                                else:
                                    break
                            if ce - cs + 1 >= 8:
                                ch = hi.iloc[cs:ce + 1].max()
                                if cur['high'] > ch:
                                    pat = '定式二'

                if pat == '定式四' and target_loc >= 30:
                    gl = int(last['brick_green_streak']) if target_loc >= 1 else 0
                    if 1 <= gl <= 7:
                        gs = target_loc - 1 - gl + 1
                        if gs >= 10:
                            rss, rse = max(0, gs - 40), gs - 1
                            if rse > rss:
                                sc = stock['close'].iloc[rss:rse + 1]
                                sl = stock['low'].iloc[rss:rse + 1]
                                if len(sc) >= 10:
                                    mr = 0
                                    for ws in [10, 15, 20]:
                                        for s in range(0, len(sc) - ws):
                                            rp = (sc.iloc[s + ws - 1] / sl.iloc[s:s + ws].min() - 1) * 100
                                            if rp > mr:
                                                mr = rp
                                    if mr >= 10:
                                        pat = '定式三'

                brick_signals[code] = (pat, row)

            # 深度分析持仓
            if code in holdings:
                holdings_detail[code] = deep_analysis(code, stock, target_loc)
                # 止损检查
                e = entry_info.get(code, {})
                si, st = check_stop_conditions(code, stock, target_loc,
                                                       entry_date=e.get("date"), entry_price=e.get("price"))
                holdings_detail[code]["stop_info"] = si
                holdings_detail[code]["stop_triggers"] = st
                # 轮换检查
                e = entry_info.get(code, {})
                ra = check_rotation_advisory(code, stock, target_loc,
                                             entry_date=e.get('date'), entry_price=e.get('price'))
                holdings_detail[code]["rotation_adv"] = ra

        except Exception as e:
            errors.append((code, str(e)[:80]))
            continue

        checked += 1
        if checked % 500 == 0:
            elapsed = time.time() - t1
            print(f"  {checked}/{len(valid)}  {elapsed:.0f}s  "
                  f"b2={len(b2_signals)} brick={len(brick_signals)}", flush=True)

    elapsed = time.time() - t1
    print(f"\n  完成: {checked} 只, {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  b2 信号: {len(b2_signals)}  |  brick 信号: {len(brick_signals)}")
    if errors:
        print(f"  错误: {len(errors)} 只")

    # ===================================================================
    # 持仓深度分析
    # ===================================================================
    print("\n")
    print("=" * 70)
    print("  ██  持仓股深度分析  ██")
    print("=" * 70)

    for code in holdings:
        if code not in holdings_detail:
            print(f"\n  {code}: 数据不可用")
            continue

        d = holdings_detail[code]
        print(f"\n{'─'*70}")
        print(f"  【{code}】  收盘: {d['close']:.2f}")
        print(f"{'─'*70}")

        # 价格与均线
        print(f"\n  ┌─ 趋势与均线 ─────────────────────────┐")
        print(f"  │ 收盘价: {d['close']:>10.2f}                          │")
        print(f"  │ MA5:    {d['ma5']:>10.2f}   {'↗ 站上' if d['close'] > d['ma5'] else '↘ 跌破'}                │")
        print(f"  │ MA10:   {d['ma10']:>10.2f}   {'↗ 站上' if d['close'] > d['ma10'] else '↘ 跌破'}                │")
        print(f"  │ MA20:   {d['ma20']:>10.2f}   {'↗ 站上' if d['close'] > d['ma20'] else '↘ 跌破'}                │")
        if not np.isnan(d['ma60']):
            print(f"  │ MA60:   {d['ma60']:>10.2f}   {'↗ 站上' if d['close'] > d['ma60'] else '↘ 跌破'}                │")
        print(f"  │ 20日位置: {d['pos_20d']:>5.0f}% (低位←→高位)                │")
        print(f"  └────────────────────────────────────────┘")

        # 涨跌幅
        print(f"\n  ┌─ 近期涨跌 ────────────────────────────┐")
        print(f"  │ 1日:  {d['ret_1d']:>+6.2f}%                          │")
        print(f"  │ 5日:  {d['ret_5d']:>+6.2f}%                          │")
        print(f"  │ 10日: {d['ret_10d']:>+6.2f}%                          │")
        print(f"  │ 20日: {d['ret_20d']:>+6.2f}%                          │")
        print(f"  └────────────────────────────────────────┘")

        # KDJ
        j_color = "超卖⚠" if d['J'] < 20 else ("超买⚠" if d['J'] > 80 else "正常")
        print(f"\n  ┌─ KDJ ──────────────────────────────────┐")
        print(f"  │ K={d['K']:.1f}  D={d['D']:.1f}  J={d['J']:.1f} [{j_color}]     │")
        kd_status = "金叉✓" if d['K'] > d['D'] else "死叉✗"
        print(f"  │ 状态: {kd_status}                              │")
        print(f"  └────────────────────────────────────────┘")

        # MACD
        print(f"\n  ┌─ MACD ─────────────────────────────────┐")
        print(f"  │ DIF={d['MACD_DIF']:.3f}  DEA={d['MACD_DEA']:.3f}           │")
        macd_color = "红柱▲" if d['MACD_HIST'] > 0 else "绿柱▼"
        print(f"  │ HIST={d['MACD_HIST']:.3f} [{macd_color}]                    │")
        macd_status = "多头" if d['MACD_DIF'] > d['MACD_DEA'] else "空头"
        print(f"  │ 趋势: {macd_status}                              │")
        print(f"  └────────────────────────────────────────┘")

        # 砖型图
        b_color = "红砖▲" if d['brick_rising'] == 1 else "绿砖▼"
        print(f"\n  ┌─ 砖型图 ───────────────────────────────┐")
        print(f"  │ 砖值: {d['brick_val']:.3f}  [{b_color}]                      │")
        print(f"  │ 绿转红信号: {'是✓' if d['brick_green_to_red']==1 else '否'}                           │")
        print(f"  │ 红转绿信号: {'是⚠' if d['brick_red_to_green']==1 else '否'}                           │")
        if d['brick_rising'] == 1:
            print(f"  │ 连续红柱: {int(d['red_streak'])} 天                           │")
        else:
            print(f"  │ 连续绿柱: {int(d['green_streak'])} 天                           │")
        print(f"  └────────────────────────────────────────┘")

        # 白黄线
        print(f"\n  ┌─ 白黄线 ───────────────────────────────┐")
        print(f"  │ 白线: {d['white_line']:.2f}                            │")
        print(f"  │ 黄线: {d['yellow_line']:.2f}                            │")
        wy_status = "白>黄 ✓" if d['white_above_yellow'] == 1 else "白<黄 ✗"
        wy_dir = "向上↑" if d['white_direction'] > 0 else ("向下↓" if d['white_direction'] < 0 else "平→")
        print(f"  │ 状态: {wy_status}  方向: {wy_dir}                      │")
        print(f"  └────────────────────────────────────────┘")

        # 成交量
        print(f"\n  ┌─ 量价 ─────────────────────────────────┐")
        print(f"  │ 量比(5日): {d['vol_ratio_5']:.2f}x                       │")
        print(f"  │ 量比(20日): {d['vol_ratio_20']:.2f}x                      │")
        vol_extra = []
        if d['vol_double'] == 1: vol_extra.append("倍量柱!")
        if d['vol_floor'] == 1: vol_extra.append("地量")
        if vol_extra:
            print(f"  │ 形态: {', '.join(vol_extra)}                          │")
        print(f"  └────────────────────────────────────────┘")

        # 风险
        print(f"\n  ┌─ 风险评估 ─────────────────────────────┐")
        risk_level = "低✓" if d['risk_score'] < 2 else ("中⚠" if d['risk_score'] < 4 else "高✗")
        print(f"  │ 风险评分: {d['risk_score']:.1f}/10  [{risk_level}]                      │")
        if d['risks']:
            for r in d['risks']:
                print(f"  │ ⚠ {r}                              │")
        else:
            print(f"  │ ✓ 无明显风险信号                               │")
        print(f"  └────────────────────────────────────────┘")

        # 强度
        star_count = int(d['strength'] / 2) + 1
        stars = '★' * min(star_count, 5) + '☆' * max(0, 5 - star_count)
        print(f"\n  → 综合强度: {d['strength']:.1f}/10  {stars}")
        # 止损检查输出
        if d.get("stop_info") and d.get("stop_triggers"):
            print_stop_check(d["stop_info"], d["stop_triggers"])
        # 轮换建议
        ra = d.get("rotation_adv")
        if ra and b2_signals:
            # 补充信号对比
            current_weight = 1.0
            stronger = [(c, w, g) for c, (g, _, w, _) in b2_signals.items()
                       if c != code and w > current_weight + 1.0]
            if stronger and not any('更强信号' in a.get('level', '') for a in ra.get('advices', [])):
                stronger.sort(key=lambda x: x[1], reverse=True)
                top = stronger[:3]
                ra['advices'].append({
                    'level': '💡 更强信号',
                    'detail': f'存在权重新信号: {", ".join(f"{c}({w:.1f}x)" for c,w,_ in top)}',
                    'suggestion': '可考虑换仓至更强信号'
                })
        print_rotation_advisory(ra)

    # ===================================================================
    # 策略信号排名
    # ===================================================================
    print("\n\n")
    print("=" * 70)
    print("  ██  b2 策略 — 强势确认反转信号排名 (Top 20)  ██")
    print("=" * 70)

    # Score b2 signals
    b2_ranked = []
    for code, (gain, j_prev, weight, row) in b2_signals.items():
        score = compute_strength(row)
        score += weight * 1.5  # b2自带权重
        score += min(gain / 5, 2)  # 当日涨幅加分(上限2)
        score += min((80 - j_prev) / 15, 2) if j_prev < 80 else -1  # J前日低位加分
        is_holding = '★' if code in holdings else ''
        b2_ranked.append((round(score, 1), code, gain, j_prev, weight, row, is_holding))

    b2_ranked.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  {'排名':<5} {'代码':<12} {'评分':>5} {'权重':>5} {'涨幅%':>7} {'J前日':>7} "
          f"{'KDJ':>8} {'MACD':>6} {'砖':>4} {'持仓'}")
    print(f"  {'─'*5} {'─'*12} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*6} {'─'*4} {'─'*6}")

    for i, (score, code, gain, j_prev, weight, row, h_flag) in enumerate(b2_ranked[:20]):
        kdj_s = f"J={row['J']:.0f}" + ("金" if row['K'] > row['D'] else "死")
        macd_s = "多头" if row['MACD_DIF'] > row['MACD_DEA'] else "空头"
        brick_s = "红" if row['brick_rising'] == 1 else "绿"
        marker = '← 持仓' if code in holdings else ('★' if i < 3 else '')
        print(f"  {i+1:<5} {code:<12} {score:>5.1f} {weight:>5.1f}x {gain:>6.1f}% {j_prev:>6.1f}  "
              f"{kdj_s:>8} {macd_s:>6} {brick_s:>4} {marker}")

    print(f"\n  b2 信号共 {len(b2_signals)} 只")

    # ── Brick 信号排名 ──
    print("\n\n")
    print("=" * 70)
    print("  ██  砖型图策略 — CC买入信号排名 (Top 20)  ██")
    print("=" * 70)

    brick_ranked = []
    for code, (pattern, row) in brick_signals.items():
        score = compute_strength(row)
        # Pattern bonus
        if '定式一' in pattern:
            score += 1.5
        elif '定式二' in pattern:
            score += 1.0
        elif '定式三' in pattern:
            score += 0.5
        is_holding = '★' if code in holdings else ''
        brick_ranked.append((round(score, 1), code, pattern, row, is_holding))

    brick_ranked.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  {'排名':<5} {'代码':<12} {'评分':>5} {'定式':>8} {'涨幅%':>7} "
          f"{'KDJ':>8} {'MACD':>6} {'砖色':>5} {'持仓'}")
    print(f"  {'─'*5} {'─'*12} {'─'*5} {'─'*8} {'─'*7} {'─'*8} {'─'*6} {'─'*5} {'─'*6}")

    for i, (score, code, pattern, row, h_flag) in enumerate(brick_ranked[:20]):
        gain_1d = row['ret_1d']
        kdj_s = f"J={row['J']:.0f}" + ("金" if row['K'] > row['D'] else "死")
        macd_s = "多头" if row['MACD_DIF'] > row['MACD_DEA'] else "空头"
        brick_s = "红▲" if row['brick_rising'] == 1 else "绿▼"
        marker = '← 持仓' if code in holdings else ('★' if i < 3 else '')
        print(f"  {i+1:<5} {code:<12} {score:>5.1f} {pattern:>8} {gain_1d:>6.1f}% "
              f"{kdj_s:>8} {macd_s:>6} {brick_s:>5} {marker}")

    # Pattern distribution
    p_counts = {}
    for pat, _ in brick_signals.values():
        p_counts[pat] = p_counts.get(pat, 0) + 1
    print(f"\n  砖型图信号共 {len(brick_signals)} 只")
    print(f"  定式分布: {p_counts}")

    # ===================================================================
    # 持仓 vs Top Picks 对比
    # ===================================================================
    print("\n\n")
    print("=" * 70)
    print("  ██  持仓 vs Top Picks 对比分析  ██")
    print("=" * 70)

    print(f"\n  ┌─ 你的持仓 ─────────────────────────────────────────────┐")

    for code in holdings:
        if code not in holdings_detail:
            continue
        d = holdings_detail[code]
        # Check if in signal lists
        in_b2 = code in b2_signals
        in_brick = code in brick_signals

        b2_rank = next((i+1 for i, x in enumerate(b2_ranked) if x[1] == code), '-')
        brick_rank = next((i+1 for i, x in enumerate(brick_ranked) if x[1] == code), '-')

        # Overall assessment
        issues = []
        if d['risk_score'] >= 4:
            issues.append("风险偏高")
        if d['J'] > 80:
            issues.append("J值超买")
        if d['white_direction'] < 0:
            issues.append("白线转弱")
        if d['brick_red_to_green'] == 1:
            issues.append("砖型图红转绿!")
        if d['MACD_HIST'] < 0 and d['MACD_DIF'] < d['MACD_DEA']:
            issues.append("MACD空头")

        good = []
        if d['white_above_yellow'] == 1:
            good.append("白>黄")
        if d['K'] > d['D']:
            good.append("KDJ金叉")
        if d['brick_rising'] == 1:
            good.append("红砖持股")
        if d['vol_double'] == 1:
            good.append("放量")
        if d['close'] > d['ma20']:
            good.append("站上MA20")

        status = "✓ 持有信号明确" if len(good) >= len(issues) else "⚠ 需关注风险"

        print(f"  │")
        print(f"  │ {code}")
        print(f"  │   收盘: {d['close']:.2f}  |  强度: {d['strength']:.1f}/10  |  风险: {d['risk_score']:.1f}/10")
        print(f"  │   b2排名: {b2_rank}  |  brick排名: {brick_rank}")
        print(f"  │   利好: {', '.join(good) if good else '(无)'}")
        print(f"  │   风险: {', '.join(issues) if issues else '(无)'}")
        print(f"  │   评估: {status}")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # Top picks summary
    print(f"\n  ┌─ b2 策略 Top 3 ────────────────────────────────────────┐")
    for i, (score, code, gain, j_prev, weight, row, _) in enumerate(b2_ranked[:3]):
        holding_tag = '← 你已持有' if code in holdings else ''
        print(f"  │ #{i+1} {code:<12} 评分{score:.1f}  权重{weight:.1f}x  "
              f"涨幅{gain:+.1f}%  J={row['J']:.0f}  {holding_tag}")
    print(f"  └──────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 砖型图 Top 3 ─────────────────────────────────────────┐")
    for i, (score, code, pattern, row, _) in enumerate(brick_ranked[:3]):
        holding_tag = '← 你已持有' if code in holdings else ''
        print(f"  │ #{i+1} {code:<12} 评分{score:.1f}  {pattern}  "
              f"涨幅{row['ret_1d']:+.1f}%  J={row['J']:.0f}  {holding_tag}")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # ===================================================================
    # 最终建议
    # ===================================================================
    print("\n\n")
    print("=" * 70)
    print("  ██  综合建议  ██")
    print("=" * 70)

    for code in holdings:
        if code not in holdings_detail:
            continue
        d = holdings_detail[code]

        print(f"\n  【{code}】")
        print(f"  {'─'*50}")

        # Decision logic
        hold_reasons = []
        swap_reasons = []

        # Hold reasons
        if d['strength'] >= 6:
            hold_reasons.append(f"综合强度尚可({d['strength']:.1f}/10)")
        if d['risk_score'] <= 2:
            hold_reasons.append(f"风险可控({d['risk_score']:.1f}/10)")
        if d['white_above_yellow'] == 1 and d['white_direction'] > 0:
            hold_reasons.append("白线>黄线且向上，趋势完好")
        if d['brick_rising'] == 1 and d['red_streak'] >= 2:
            hold_reasons.append(f"连续红砖{d['red_streak']:.0f}天，持股信号")
        if d['K'] > d['D'] and d['J'] < 50:
            hold_reasons.append("KDJ低位金叉，有上涨空间")
        if code in b2_signals:
            rank = next(i+1 for i, x in enumerate(b2_ranked) if x[1] == code)
            hold_reasons.append(f"b2策略今日仍发出买入信号(排名{rank})")
        if code in brick_signals:
            pat = brick_signals[code][0]
            rank = next(i+1 for i, x in enumerate(brick_ranked) if x[1] == code)
            hold_reasons.append(f"砖型图CC买入信号({pat}, 排名{rank})")

        # Swap reasons
        if d['risk_score'] >= 4:
            swap_reasons.append(f"风险评分偏高({d['risk_score']:.1f}/10)")
        if d['J'] > 85:
            swap_reasons.append(f"J值严重超买({d['J']:.1f})，回调风险大")
        if d['brick_red_to_green'] == 1:
            swap_reasons.append("砖型图红转绿，卖出信号!")
        if d['white_direction'] < 0 and d['white_above_yellow'] == 0:
            swap_reasons.append("白线死叉黄线，趋势转空")
        if d['MACD_DIF'] < d['MACD_DEA'] and d['MACD_HIST'] < 0:
            swap_reasons.append("MACD空头排列")
        if d['pos_20d'] > 90:
            swap_reasons.append(f"20日高位({d['pos_20d']:.0f}%)，追高风险")

        # Not in any signal
        if code not in b2_signals and code not in brick_signals:
            swap_reasons.append("不在任何策略买入信号中")

        print(f"  继续持有理由:")
        if hold_reasons:
            for r in hold_reasons:
                print(f"    ✓ {r}")
        else:
            print(f"    (无明确持有理由)")

        print(f"\n  考虑换仓理由:")
        if swap_reasons:
            for r in swap_reasons:
                print(f"    ⚠ {r}")
        else:
            print(f"    (无明确风险)")

        # Final verdict
        h_score = len(hold_reasons)
        s_score = len(swap_reasons)

        print(f"\n  ┌──────────────────────────────────────┐")
        if s_score == 0 and h_score >= 3:
            print(f"  │ 建议: ★★★ 坚定持有                    │")
            verdict = "HOLD"
        elif s_score <= 1 and h_score >= s_score * 2:
            print(f"  │ 建议: ★★☆ 继续持有，密切关注          │")
            verdict = "HOLD_WATCH"
        elif s_score <= 2 and h_score >= s_score:
            print(f"  │ 建议: ★☆☆ 可持有但需警惕              │")
            verdict = "HOLD_CAUTIOUS"
        elif s_score > h_score:
            print(f"  │ 建议: ☆☆☆ 建议换仓                    │")
            verdict = "SWAP"
        else:
            print(f"  │ 建议: ★★☆ 观望，视明日走势决定        │")
            verdict = "WATCH"

        # Suggest alternatives if swap
        if verdict in ('SWAP', 'WATCH', 'HOLD_CAUTIOUS'):
            alt_candidates = []
            for ranked_list, strategy_name in [(b2_ranked, 'b2'), (brick_ranked, 'brick')]:
                for item in ranked_list[:10]:
                    c = item[1]
                    if c not in holdings and c not in [x[1] for x in alt_candidates]:
                        score = item[0]
                        alt_candidates.append((c, score, strategy_name))
                        if len(alt_candidates) >= 5:
                            break
                if len(alt_candidates) >= 5:
                    break

            if alt_candidates:
                print(f"  │                                        │")
                print(f"  │ 替代候选 (Top信号股):                  │")
                for c, sc, st in alt_candidates[:5]:
                    print(f"  │   {c}  ({st} 评分{sc:.1f})")

        print(f"  └──────────────────────────────────────┘")

    # ===================================================================
    # 策略共振股 (同时出现在 b2 和 brick 中)
    # ===================================================================
    overlap = set(b2_signals.keys()) & set(brick_signals.keys())
    if overlap:
        print(f"\n\n{'='*70}")
        print(f"  ██  策略共振 (b2 ∩ brick): {len(overlap)} 只  ██")
        print(f"{'='*70}")
        overlap_list = []
        for code in overlap:
            b2_score = next((x[0] for x in b2_ranked if x[1] == code), 0)
            brick_score = next((x[0] for x in brick_ranked if x[1] == code), 0)
            combined = b2_score + brick_score
            overlap_list.append((combined, code, b2_score, brick_score))
        overlap_list.sort(reverse=True)
        for combined, code, bs, brs in overlap_list[:15]:
            h = '← 持仓' if code in holdings else ''
            print(f"  {code:<12}  b2评分={bs:.1f}  brick评分={brs:.1f}  综合={combined:.1f}  {h}")
        if len(overlap) > 15:
            print(f"  ... 共 {len(overlap)} 只")

    # ── 导出 ──
    print(f"\n\n>>> 导出结果...")
    os.makedirs(OUT, exist_ok=True)
    date_str = target_date.strftime('%Y%m%d')

    # b2 ranked
    fp = os.path.join(out, f'analysis_b2_ranked_{date_str}.txt')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"b2 策略评分排名\n日期: {target_date.date()}\n{'='*50}\n\n")
        for score, code, gain, j_prev, weight, row, _ in b2_ranked:
            f.write(f"{code}\t评分{score:.1f}\t权重{weight:.1f}x\t涨幅{gain:+.1f}%\t"
                    f"J前={j_prev:.1f}\tJ今={row['J']:.0f}\n")
    print(f"  {fp}")

    # brick ranked
    fp = os.path.join(out, f'analysis_brick_ranked_{date_str}.txt')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"砖型图策略评分排名\n日期: {target_date.date()}\n{'='*50}\n\n")
        for score, code, pattern, row, _ in brick_ranked:
            f.write(f"{code}\t评分{score:.1f}\t{pattern}\t涨幅{row['ret_1d']:+.1f}%\t"
                    f"J={row['J']:.0f}\n")
    print(f"  {fp}")

    # Holdings detail
    fp = os.path.join(out, f'analysis_holdings_{date_str}.txt')
    with open(fp, 'w', encoding='utf-8') as f:
        for code in holdings:
            if code not in holdings_detail:
                continue
            d = holdings_detail[code]
            f.write(f"\n{'='*50}\n{code}\n{'='*50}\n")
            for k, v in d.items():
                if k != 'risks':
                    f.write(f"  {k}: {v}\n")
            f.write(f"  risks: {d['risks']}\n")
    print(f"  {fp}")

    print(f"\n{'='*70}")
    print(f"  分析完成。总耗时 {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    args = parse_args()

    # 解析股票代码
    codes = parse_codes(args.codes)

    # 无参数则交互输入
    if not codes:
        print("持仓分析与策略对比")
        print("─" * 40)
        prompt = input("请输入股票代码（6位数字，多个用空格/逗号分隔）: ").strip()
        if prompt:
            codes = parse_codes([prompt])

    if not codes:
        print("未输入有效代码，仅运行全市场扫描，不分析持仓。")
        print("用法: python run_holdings_analysis.py 000960 601138")
        print()

    # 日期
    target_date = pd.Timestamp(args.date) if args.date else _get_latest_date()

    # 输出目录
    output_dir = args.output if args.output else None

    # 解析入场信息
    entry_info = {}
    for e in args.entry:
        parts = e.split(':')
        if len(parts) == 3:
            code, date_str, price_str = parts
            entry_info[code] = {'date': date_str, 'price': float(price_str)}
        else:
            print(f"  警告: 入场格式错误 '{e}'，应为 CODE:DATE:PRICE")

    main(
        holdings=codes,
        target_date=target_date,
        do_scan=not args.no_scan,
        output_dir=output_dir,
        entry_info=entry_info,
    )
