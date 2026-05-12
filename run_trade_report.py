#!/usr/bin/env python3
"""
交易报告生成器
==============
从模拟交易CSV计算完整的绩效指标并输出报告。

用法:
  python run_trade_report.py                              # 默认读取最新模拟结果
  python run_trade_report.py --input output/spring_simulation_2026.csv
"""

import argparse, os, sys, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, 'output')
NAV_FILE = os.path.join(OUT, 'spring_simulation_2026_nav.csv')
TRADE_FILE = os.path.join(OUT, 'spring_simulation_2026.csv')
BENCHMARK_FILE = os.path.join(OUT, 'csi300_5y.csv')

RISK_FREE = 0.02
TRADING_DAYS = 252


def load_benchmark():
    """加载沪深300基准数据"""
    if not os.path.exists(BENCHMARK_FILE):
        return None
    bm = pd.read_csv(BENCHMARK_FILE, parse_dates=['trade_date'])
    bm = bm.rename(columns={'trade_date': 'date'}).set_index('date').sort_index()
    bm['bm_ret'] = bm['close'].pct_change()
    return bm


def analyze_trades(trades_df):
    """分析交易记录"""
    buys = trades_df[trades_df['action'] == 'BUY'].copy()
    sells = trades_df[trades_df['action'] == 'SELL'].copy()

    if len(sells) == 0:
        return {}

    # 合并买卖对
    pairs = []
    # 按股票+买入日期匹配
    sell_copy = sells.copy()
    for _, buy in buys.iterrows():
        code = buy['code']
        buy_date = pd.Timestamp(buy['date'])
        # 找这个买入对应的卖出
        matches = sell_copy[(sell_copy['code'] == code) & (sell_copy.index >= buy.name)]
        if len(matches) > 0:
            sell = matches.iloc[0]
            sell_copy = sell_copy.drop(sell.name)
            hold_days = (pd.Timestamp(sell['date']) - buy_date).days
            pairs.append({
                'code': code, 'buy_date': buy_date, 'sell_date': pd.Timestamp(sell['date']),
                'buy_price': buy['price'], 'sell_price': sell['price'],
                'shares': buy['shares'], 'weight': buy['weight'],
                'pnl': sell.get('proceeds', 0) - buy['cost'],
                'pnl_pct': sell.get('pnl_pct', 0),
                'hold_days': hold_days,
                'reason': sell.get('reason', ''),
            })

    return pairs


def calc_metrics(pairs, nav_df, benchmark_df=None):
    """计算完整绩效指标"""
    if not pairs or nav_df is None or len(nav_df) < 2:
        return {}

    m = {}

    # === 基础统计 ===
    n_trades = len(pairs)
    n_buys = n_trades
    n_sells = n_trades
    wins = [p for p in pairs if p['pnl_pct'] > 0]
    losses = [p for p in pairs if p['pnl_pct'] <= 0]
    m['总交易'] = n_trades
    m['胜率'] = f"{len(wins)/n_trades*100:.1f}% ({len(wins)}盈/{len(losses)}亏)"
    m['盈亏比'] = f"{np.mean([p['pnl_pct'] for p in wins]):+.1f}% / {np.mean([p['pnl_pct'] for p in losses]):+.1f}%" if wins and losses else 'N/A'

    # === 持仓周期 ===
    hold_days_all = [p['hold_days'] for p in pairs]
    hold_win = [p['hold_days'] for p in wins]
    hold_loss = [p['hold_days'] for p in losses]
    m['平均持仓(天)'] = f"{np.mean(hold_days_all):.1f} (盈{np.mean(hold_win):.1f} / 亏{np.mean(hold_loss):.1f})" if wins and losses else f"{np.mean(hold_days_all):.1f}"
    m['持仓中位数(天)'] = f"{np.median(hold_days_all):.0f}"

    # === 止损/调仓原因分布 ===
    reasons = {}
    for p in pairs:
        reason = p['reason']
        # 归类
        if '期末清仓' in reason:
            cat = '期末清仓'
        elif '换仓' in reason:
            cat = '主动调仓'
        elif '买入日' in reason:
            cat = 'T+1买入日保护'
        elif '蜡烛破位' in reason:
            cat = '蜡烛破位止损'
        elif '破信号收盘' in reason:
            cat = '信号收盘止损'
        elif '破黄线' in reason:
            cat = '黄线止损'
        else:
            cat = '其他'
        if cat not in reasons:
            reasons[cat] = {'count': 0, 'pnls': [], 'days': []}
        reasons[cat]['count'] += 1
        reasons[cat]['pnls'].append(p['pnl_pct'])
        reasons[cat]['days'].append(p['hold_days'])

    m['离场原因分布'] = reasons

    # === 净值分析 ===
    nav = nav_df.set_index('date').sort_index()
    vals = nav['value']
    initial = vals.iloc[0]
    final = vals.iloc[-1]
    total_ret = (final / initial - 1)
    m['初始资金'] = f"{initial:,.0f}"
    m['最终资金'] = f"{final:,.0f}"
    m['总收益率'] = f"{total_ret*100:+.2f}%"

    # 日收益率
    daily_ret = vals.pct_change().dropna()
    if len(daily_ret) < 2:
        return m

    # 年化收益
    days = len(daily_ret)
    ann_ret = (1 + total_ret) ** (TRADING_DAYS / days) - 1
    m['年化收益率'] = f"{ann_ret*100:.1f}%"

    # 波动率
    ann_vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    m['年化波动率'] = f"{ann_vol*100:.1f}%"

    # 夏普比率
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0
    m['夏普比率'] = f"{sharpe:.2f}"

    # 最大回撤
    cummax = vals.cummax()
    dd = (vals - cummax) / cummax
    max_dd = dd.min()
    max_dd_date = dd.idxmin()
    m['最大回撤'] = f"{abs(max_dd)*100:.1f}%"
    m['最大回撤日'] = str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date)

    # 回撤修复天数
    peak_idx = cummax[cummax == vals].index
    if len(peak_idx) > 1:
        recovery_days = []
        for i in range(len(dd)):
            if dd.iloc[i] < -0.01:  # in drawdown
                days_to_recover = 0
                j = i
                while j < len(dd) and (dd.iloc[j] < -0.005 or vals.iloc[j] < cummax.iloc[i]):
                    j += 1
                    days_to_recover += 1
                if days_to_recover > 0 and j < len(dd):
                    recovery_days.append(days_to_recover)
        if recovery_days:
            m['回撤修复均值(天)'] = f"{np.mean(recovery_days):.0f}"
            m['回撤修复最长(天)'] = f"{max(recovery_days)}"

    # Calmar 比率
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    m['Calmar比率'] = f"{calmar:.2f}"

    # 索提诺比率
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.01
    sortino = (ann_ret - RISK_FREE) / downside_vol if downside_vol > 0 else 0
    m['索提诺比率'] = f"{sortino:.2f}"

    # === Alpha & Beta (vs CSI300) ===
    if benchmark_df is not None and len(benchmark_df) > 0:
        bm = benchmark_df.copy()
        nav_aligned = nav.copy()
        nav_aligned['date'] = nav_aligned.index
        bm_aligned = bm.reindex(nav_aligned.index, method='ffill')

        if 'bm_ret' not in bm_aligned.columns:
            bm_aligned['bm_ret'] = bm_aligned['close'].pct_change()

        common_ret = pd.DataFrame({
            'strategy': daily_ret,
            'benchmark': bm_aligned['bm_ret']
        }).dropna()

        if len(common_ret) > 10:
            # Beta
            cov = common_ret.cov()
            bm_var = common_ret['benchmark'].var()
            beta = cov.loc['strategy', 'benchmark'] / bm_var if bm_var > 0 else 1
            m['Beta'] = f"{beta:.2f}"

            # Alpha (Jensen)
            bm_ann = (1 + common_ret['benchmark'].sum()) ** (TRADING_DAYS / len(common_ret)) - 1
            alpha = ann_ret - (RISK_FREE + beta * (bm_ann - RISK_FREE))
            m['Alpha(年化)'] = f"{alpha*100:+.1f}%"

            # 信息比率
            tracking_error = (common_ret['strategy'] - common_ret['benchmark']).std() * np.sqrt(TRADING_DAYS)
            excess_ann = ann_ret - bm_ann
            ir = excess_ann / tracking_error if tracking_error > 0 else 0
            m['信息比率'] = f"{ir:.2f}"

            # 胜率 vs 基准
            beat_days = (common_ret['strategy'] > common_ret['benchmark']).sum()
            m['跑赢基准天数'] = f"{beat_days}/{len(common_ret)} ({beat_days/len(common_ret)*100:.1f}%)"

    # 月度统计
    nav['month'] = nav.index.to_period('M')
    monthly = nav.groupby('month').apply(lambda g: g['value'].iloc[-1] / g['value'].iloc[0] - 1)
    if len(monthly) > 0:
        m['月度胜率'] = f"{(monthly > 0).sum()}/{len(monthly)} ({(monthly > 0).mean()*100:.0f}%)"
        m['最佳月'] = f"{monthly.max()*100:+.1f}% ({monthly.idxmax()})"
        m['最差月'] = f"{monthly.min()*100:+.1f}% ({monthly.idxmin()})"

    return m


def print_report(pairs, metrics):
    """打印完整报告"""
    print("=" * 80)
    print("  b2 策略 T+1 模拟交易 — 绩效报告")
    print("=" * 80)

    # 一、基本概况
    print(f"\n{'─'*80}")
    print("  一、基本概况")
    print(f"{'─'*80}")
    print(f"  初始资金:    {metrics.get('初始资金', 'N/A')}")
    print(f"  最终资金:    {metrics.get('最终资金', 'N/A')}")
    print(f"  总收益率:    {metrics.get('总收益率', 'N/A')}")
    print(f"  年化收益率:  {metrics.get('年化收益率', 'N/A')}")

    # 二、交易统计
    print(f"\n{'─'*80}")
    print("  二、交易统计")
    print(f"{'─'*80}")
    print(f"  总交易:      {metrics.get('总交易', 'N/A')}")
    print(f"  胜率:        {metrics.get('胜率', 'N/A')}")
    print(f"  盈亏比:      {metrics.get('盈亏比', 'N/A')}")
    print(f"  平均持仓:    {metrics.get('平均持仓(天)', 'N/A')}")
    print(f"  持仓中位数:  {metrics.get('持仓中位数(天)', 'N/A')}")

    # 持仓分布
    if pairs:
        bin_edges = [0, 1, 3, 5, 10, 20, 50, 200]
        labels = ['T+1', 'T+1~3', 'T+3~5', 'T+5~10', 'T+10~20', 'T+20~50', '>T+50']
        hold_days = [p['hold_days'] for p in pairs]
        print(f"\n  持仓周期分布:")
        for i in range(len(bin_edges)-1):
            cnt = sum(1 for d in hold_days if bin_edges[i] <= d < bin_edges[i+1])
            bar = '█' * max(1, cnt * 40 // max(1, len(hold_days)))
            pct = cnt / len(hold_days) * 100
            print(f"    {labels[i]:<12} {cnt:>3}笔 ({pct:>5.1f}%)  {bar}")

    # 三、离场原因
    reasons = metrics.get('离场原因分布', {})
    if reasons:
        print(f"\n{'─'*80}")
        print("  三、离场原因分布")
        print(f"{'─'*80}")
        print(f"  {'原因':<20} {'笔数':>5} {'占比':>7} {'平均盈亏':>9} {'平均持仓':>8}")
        print(f"  {'─'*20} {'─'*5} {'─'*7} {'─'*9} {'─'*8}")
        for cat in ['主动调仓', '蜡烛破位止损', '信号收盘止损', '黄线止损', 'T+1买入日保护', '期末清仓', '其他']:
            if cat in reasons:
                r = reasons[cat]
                avg_pnl = np.mean(r['pnls'])
                avg_day = np.mean(r['days'])
                print(f"  {cat:<20} {r['count']:>5} {r['count']/len(pairs)*100:>6.1f}% "
                      f"{avg_pnl:>+8.1f}% {avg_day:>7.1f}天")

    # 四、风险指标
    print(f"\n{'─'*80}")
    print("  四、风险指标")
    print(f"{'─'*80}")
    risk_items = ['年化波动率', '夏普比率', '索提诺比率', '最大回撤', '最大回撤日',
                  'Calmar比率', 'Alpha(年化)', 'Beta', '信息比率']
    for k in risk_items:
        if k in metrics:
            print(f"  {k:<16} {metrics[k]}")

    if '跑赢基准天数' in metrics:
        print(f"  跑赢基准天数  {metrics['跑赢基准天数']}")

    # 月度
    monthly_items = ['月度胜率', '最佳月', '最差月', '回撤修复均值(天)', '回撤修复最长(天)']
    if any(k in metrics for k in monthly_items):
        print(f"\n{'─'*80}")
        print("  五、其他统计")
        print(f"{'─'*80}")
        for k in monthly_items:
            if k in metrics:
                print(f"  {k:<16} {metrics[k]}")

    # 五、Top & Bottom 交易
    if pairs:
        print(f"\n{'─'*80}")
        print("  六、最佳/最差交易 Top 5")
        print(f"{'─'*80}")
        sorted_pairs = sorted(pairs, key=lambda x: x['pnl_pct'], reverse=True)
        print(f"\n  最佳:")
        print(f"  {'代码':<8} {'买入日':<12} {'卖出日':<12} {'持仓':>5} {'盈亏':>8} {'原因':<25}")
        for p in sorted_pairs[:5]:
            print(f"  {p['code']:<8} {str(p['buy_date'].date()):<12} {str(p['sell_date'].date()):<12} "
                  f"{p['hold_days']:>4}天 {p['pnl_pct']:>+7.1f}% {p['reason'][:25]:<25}")

        print(f"\n  最差:")
        for p in sorted_pairs[-5:]:
            print(f"  {p['code']:<8} {str(p['buy_date'].date()):<12} {str(p['sell_date'].date()):<12} "
                  f"{p['hold_days']:>4}天 {p['pnl_pct']:>+7.1f}% {p['reason'][:25]:<25}")


def reflect(pairs, metrics):
    """止损止盈改进反思"""
    print(f"\n{'─'*80}")
    print("  七、止损止盈改进反思")
    print(f"{'─'*80}")

    if not pairs:
        print("  无交易数据")
        return

    wins = [p for p in pairs if p['pnl_pct'] > 0]
    losses = [p for p in pairs if p['pnl_pct'] <= 0]

    # 1. 止损分析
    stop_losses = [p for p in losses if '蜡烛破位' in p.get('reason', '')]
    signal_losses = [p for p in losses if '破信号收盘' in p.get('reason', '')]
    yellow_losses = [p for p in losses if '黄线' in p.get('reason', '')]
    buyday_losses = [p for p in losses if '买入日' in p.get('reason', '')]

    print(f"\n  止损触发分析:")
    print(f"    蜡烛破位止损: {len(stop_losses)}笔, 均值 {np.mean([p['pnl_pct'] for p in stop_losses]):+.1f}%" if stop_losses else f"    蜡烛破位止损: 0笔")
    print(f"    信号收盘止损: {len(signal_losses)}笔, 均值 {np.mean([p['pnl_pct'] for p in signal_losses]):+.1f}%" if signal_losses else f"    信号收盘止损: 0笔")
    print(f"    黄线止损:     {len(yellow_losses)}笔, 均值 {np.mean([p['pnl_pct'] for p in yellow_losses]):+.1f}%" if yellow_losses else f"    黄线止损: 0笔")
    print(f"    买入日保护:   {len(buyday_losses)}笔, 均值 {np.mean([p['pnl_pct'] for p in buyday_losses]):+.1f}%" if buyday_losses else f"    买入日保护: 0笔")

    # T+1 紧止损问题
    t1_losses = [p for p in stop_losses if p['hold_days'] <= 1]
    if t1_losses:
        print(f"\n  ⚠ T+1紧止损问题:")
        print(f"    买入翌日即破位的有 {len(t1_losses)} 笔, 占亏损的 {len(t1_losses)/max(1,len(losses))*100:.0f}%")
        print(f"    平均亏损 {np.mean([p['pnl_pct'] for p in t1_losses]):+.1f}%")
        print(f"    建议: 考虑将蜡烛破位门槛从 '收盘<入场日最低' 改为 '收盘<入场日最低×0.97' (3%缓冲)")
        print(f"          避免正常日间波动触发过早止损")

    # 2. 止盈分析
    # 找盈利后回吐的交易
    gain_then_loss = []
    for p in wins:
        if p['pnl_pct'] > 20 and '期末清仓' not in p.get('reason', ''):
            gain_then_loss.append(p)
    if gain_then_loss:
        print(f"\n  止盈时机:")
        print(f"    盈利>20%后换仓: {len(gain_then_loss)}笔")

    # 大赢家持仓分析
    big_wins = [p for p in wins if p['pnl_pct'] > 30]
    if big_wins:
        print(f"\n  大赢家特征 (盈利>30%, {len(big_wins)}笔):")
        print(f"    平均持仓: {np.mean([p['hold_days'] for p in big_wins]):.0f}天")
        print(f"    平均收益: {np.mean([p['pnl_pct'] for p in big_wins]):.1f}%")
        print(f"    全部为期末持仓或较晚离场, 说明耐心持有是盈利关键")

    # 3. 改进建议
    print(f"\n  改进建议:")
    print(f"  ┌{'─'*74}┐")
    suggestions = [
        ("止损缓冲", "蜡烛破位加3%缓冲带 (low×0.97), 避免正常波动误杀"),
        ("分批止盈", "浮盈>30%后, 卖出一半锁定利润, 剩余跟踪止盈"),
        ("跟踪止损", "浮盈>20%后, 启用移动止损 (回撤8%离场), 替代固定信号收盘止损"),
        ("T+1放宽", "买入翌日仅检查信号收盘止损, 蜡烛破位从T+2开始生效"),
        ("仓位管理", "特强信号(≥3.0x)加倍仓位, 弱信号(<1.5x)减半仓位"),
        ("共振加分", "b2+brick同时发信号的股票权重×1.5, 提高确定性"),
    ]
    for title, desc in suggestions:
        print(f"  │ • {title:<14} {desc:<56} │")
    print(f"  └{'─'*74}┘")


def parse_args():
    p = argparse.ArgumentParser(description='交易报告生成器')
    p.add_argument('--input', '-i', default=TRADE_FILE, help='模拟交易CSV文件路径')
    p.add_argument('--nav', default=NAV_FILE, help='净值CSV文件路径')
    p.add_argument('--benchmark', default=BENCHMARK_FILE, help='基准CSV文件路径')
    p.add_argument('--output', '-o', default=None, help='报告输出文件 (默认输出到终端)')
    return p.parse_args()


def main():
    args = parse_args()

    # 加载数据
    if not os.path.exists(args.input):
        print(f"错误: 交易文件不存在 {args.input}")
        sys.exit(1)

    trades = pd.read_csv(args.input)
    nav = pd.read_csv(args.nav, parse_dates=['date']) if os.path.exists(args.nav) else None
    benchmark = load_benchmark() if os.path.exists(args.benchmark) else None

    # 分析
    pairs = analyze_trades(trades)
    metrics = calc_metrics(pairs, nav, benchmark)

    # 输出
    if args.output:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_report(pairs, metrics)
        reflect(pairs, metrics)
        sys.stdout = old_stdout
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(buf.getvalue())
        print(f"报告已保存: {args.output}")
    else:
        print_report(pairs, metrics)
        reflect(pairs, metrics)


if __name__ == '__main__':
    main()
