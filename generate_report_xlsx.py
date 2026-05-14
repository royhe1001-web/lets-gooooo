#!/usr/bin/env python3
"""Spring B2 Excel 报告生成器 — 每次回测后运行，覆盖更新。无基准对比，纯策略指标。"""
import pandas as pd, numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
import json, os

ROOT = Path(__file__).parent
BACKTEST = ROOT / 'output' / 'backtest'
OUT = BACKTEST / f'Spring_B2_回测报告_{datetime.now().strftime("%Y%m%d")}.xlsx'

# ── 从 trade_log 重建每日净值 + 持仓 ──
def build_daily_from_trade_log():
    tl_path = BACKTEST / 'trade_log.csv'
    if not tl_path.exists():
        raise FileNotFoundError('trade_log.csv 不存在，请先运行回测')

    tl = pd.read_csv(tl_path, parse_dates=['date'])
    tl['symbol'] = tl['symbol'].astype(str).str.zfill(6)

    capital = 200_000  # 默认初始资金
    trade_dates = sorted(tl['date'].unique())

    # 生成完整交易日序列 (含无操作日)
    all_dates = pd.date_range(trade_dates[0], trade_dates[-1], freq='B')
    all_dates = [d for d in all_dates if d in set(trade_dates) or True]  # 保留所有B日

    positions = {}  # sym -> {shares, cost}
    cash = float(capital)
    nav_rows = []
    pos_rows = []

    prev_equity = float(capital)
    # 按日构建: 只处理有交易的日期更新持仓, 无交易日沿用昨日净值
    trade_date_set = set(trade_dates)

    for d in all_dates:
        if d in trade_date_set:
            day_trades = tl[tl['date'] == d]

            # 先处理卖出
            sells = day_trades[day_trades['action'] == 'SELL']
            for _, s in sells.iterrows():
                sym = str(s['symbol']).zfill(6)
                if sym in positions:
                    cash += s['shares'] * s['price'] * (1 - 0.0003)
                    del positions[sym]

            # 再处理买入
            buys = day_trades[day_trades['action'] == 'BUY']
            for _, b in buys.iterrows():
                sym = str(b['symbol']).zfill(6)
                cost = b['shares'] * b['price'] * (1 + 0.0003)
                if cost <= cash:
                    cash -= cost
                    positions[sym] = {'shares': int(b['shares']), 'cost': cost, 'side': b.get('side', '')}

        # 持仓市值 (用最近买入价近似, 无实时价时保持上次估算)
        pos_val = 0.0
        for sym, p in positions.items():
            px = p['cost'] / p['shares'] if p['shares'] > 0 else 0
            pos_val += p['shares'] * px

        total_equity = cash + pos_val
        daily_ret = (total_equity / prev_equity - 1) if prev_equity > 0 else 0

        nav_rows.append({
            'date': d, 'total_equity': total_equity, 'cash': cash,
            'position_value': pos_val, 'n_positions': len(positions),
            'daily_ret': daily_ret,
            'cum_ret': total_equity / capital - 1,
        })

        for sym, p in positions.items():
            pos_rows.append({
                'date': d, 'symbol': sym, 'shares': p['shares'],
                'side': p.get('side', ''),
            })

        prev_equity = total_equity

    nav = pd.DataFrame(nav_rows)
    nav['date'] = pd.to_datetime(nav['date'])
    pos = pd.DataFrame(pos_rows)
    pos['date'] = pd.to_datetime(pos['date'])

    # 计算累计净值
    nav['nav'] = nav['total_equity'] / capital
    nav['cum_ret_pct'] = nav['nav'] - 1

    return nav, pos, all_dates


nav, pos, dates = build_daily_from_trade_log()

# Styles
hfont = Font(bold=True, color='FFFFFF', size=10)
hfill = PatternFill('solid', fgColor='2F5496')
gfill = PatternFill('solid', fgColor='C6EFCE')
rfill = PatternFill('solid', fgColor='FFC7CE')
bfill = PatternFill('solid', fgColor='BDD7EE')
border = Border(left=Side('thin'), right=Side('thin'), top=Side('thin'), bottom=Side('thin'))
bold = Font(bold=True, size=12)

def hdr(ws, r, cols):
    for c, v in enumerate(cols, 1):
        cl = ws.cell(row=r, column=c, value=v)
        cl.font = hfont; cl.fill = hfill
        cl.alignment = Alignment(horizontal='center', wrap_text=True)
        cl.border = border

def auto_w(ws):
    for col in ws.columns:
        mx = max((len(str(c.value or '')) for c in col), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(mx + 2, 22)

wb = Workbook()

# ═══ Sheet 1: 每日净值 ═══
ws = wb.active; ws.title = '每日净值'
ws.cell(row=1, column=1, value=f'Spring B2 2026年回测 — 每日净值 (至 {dates[-1].date()})').font = bold
cols = ['日期', '总权益', '现金', '持仓市值', '持仓数', '仓位%', '日收益', '累计收益', '净值']
hdr(ws, 3, cols)
for r, (_, row) in enumerate(nav.iterrows(), 4):
    ws.cell(row=r, column=1, value=str(row['date'].date()))
    ws.cell(row=r, column=2, value=int(row['total_equity']))
    ws.cell(row=r, column=3, value=int(row['cash']))
    ws.cell(row=r, column=4, value=int(row['position_value']))
    ws.cell(row=r, column=5, value=int(row['n_positions']))
    pv = row['position_value']; te = row['total_equity']
    if te > 0:
        ws.cell(row=r, column=6, value=round(pv / te * 100, 1)).number_format = '0.0'
    ws.cell(row=r, column=7, value=round(row['daily_ret'] * 100, 2)).number_format = '0.00'
    ws.cell(row=r, column=8, value=round(row['cum_ret'] * 100, 2)).number_format = '0.00'
    ws.cell(row=r, column=9, value=round(row['nav'], 4)).number_format = '0.0000'
    dr = row['daily_ret']
    if dr > 0: ws.cell(row=r, column=7).fill = gfill
    elif dr < 0: ws.cell(row=r, column=7).fill = rfill
    for c in range(1, 10): ws.cell(row=r, column=c).border = border
auto_w(ws)

# ═══ Sheet 2: 每日持仓 ═══
ws = wb.create_sheet('每日持仓')
ws.cell(row=1, column=1, value=f'Spring B2 每日持仓明细 (至 {dates[-1].date()})').font = bold
hdr(ws, 3, ['日期', '代码', '股数', '持仓状态', '持仓类型'])
prev_s = set()
pos_cache = []
for i, d in enumerate(dates):
    dp = pos[pos['date'] == d]; cur = set(dp['symbol'].tolist())
    new_s = cur - prev_s
    nxt = set(pos[pos['date'] == dates[i + 1]]['symbol'].tolist()) if i < len(dates) - 1 else set()
    for _, pr in dp.iterrows():
        t = '★新进' if pr['symbol'] in new_s else ('▼卖出' if pr['symbol'] not in nxt and i < len(dates) - 1 else '持有')
        s = str(pr.get('side', '')).replace('base', '底仓').replace('sat', '卫星').replace('score', '排名')
        pos_cache.append((d, pr['symbol'], pr['shares'], t, s))
    prev_s = cur

for r, (d, sym, sh, t, s) in enumerate(pos_cache, 4):
    ws.cell(row=r, column=1, value=str(d.date())); ws.cell(row=r, column=2, value=sym)
    ws.cell(row=r, column=3, value=sh); ws.cell(row=r, column=4, value=t); ws.cell(row=r, column=5, value=s)
    if t == '★新进': ws.cell(row=r, column=4).fill = gfill
    elif t == '▼卖出': ws.cell(row=r, column=4).fill = rfill
    if '底仓' in s: ws.cell(row=r, column=5).fill = bfill
    for c in range(1, 6): ws.cell(row=r, column=c).border = border
auto_w(ws)

# ═══ Sheet 3: 每日交易摘要 ═══
ws = wb.create_sheet('每日交易摘要')
ws.cell(row=1, column=1, value=f'Spring B2 每日交易摘要 (至 {dates[-1].date()})').font = bold
hdr(ws, 3, ['日期', '持仓数', '新进股', '卖出股', '总权益', '日收益%', '累计收益%'])
prev_s = set()
trade_cache = []
for i, d in enumerate(dates):
    dp = pos[pos['date'] == d]; cur = set(dp['symbol'].tolist())
    new_s = cur - prev_s; exit_s = prev_s - cur
    nr = nav[nav['date'] == d]
    nv = nr.iloc[0].to_dict() if len(nr) > 0 else {}
    trade_cache.append((d, len(cur), new_s, exit_s, nv))
    prev_s = cur

for r, (d, n_cur, new_s, exit_s, nv) in enumerate(trade_cache, 4):
    ws.cell(row=r, column=1, value=str(d.date()))
    ws.cell(row=r, column=2, value=n_cur)
    ws.cell(row=r, column=3, value=','.join(sorted(new_s)) if new_s else '-')
    ws.cell(row=r, column=4, value=','.join(sorted(exit_s)) if exit_s else '-')
    ws.cell(row=r, column=5, value=int(nv.get('total_equity', 0)))
    ws.cell(row=r, column=6, value=round(nv.get('daily_ret', 0) * 100, 2))
    ws.cell(row=r, column=7, value=round(nv.get('cum_ret', 0) * 100, 2))
    dr = nv.get('daily_ret', 0)
    if dr > 0: ws.cell(row=r, column=6).fill = gfill
    elif dr < 0: ws.cell(row=r, column=6).fill = rfill
    for c in range(1, 8): ws.cell(row=r, column=c).border = border
auto_w(ws)

# ═══ Sheet 4: 月度汇总 ═══
ws = wb.create_sheet('月度汇总')
ws.cell(row=1, column=1, value='Spring B2 月度绩效汇总').font = bold
nav['month'] = nav['date'].dt.to_period('M')
monthly = nav.groupby('month').agg(
    start_equity=('total_equity', 'first'),
    end_equity=('total_equity', 'last'),
    avg_positions=('n_positions', 'mean'),
    trading_days=('date', 'count'),
).reset_index()
monthly['return'] = monthly['end_equity'] / monthly['start_equity'] - 1
monthly['month'] = monthly['month'].astype(str)
cols = ['月份', '交易天数', '月初权益', '月末权益', '月收益%', '月均持仓']
hdr(ws, 3, cols)
for r, (_, mr) in enumerate(monthly.iterrows(), 4):
    ws.cell(row=r, column=1, value=mr['month']).border = border
    ws.cell(row=r, column=2, value=int(mr['trading_days'])).border = border
    ws.cell(row=r, column=3, value=int(mr['start_equity'])).border = border
    ws.cell(row=r, column=4, value=int(mr['end_equity'])).border = border
    ws.cell(row=r, column=5, value=round(mr['return'] * 100, 2)).number_format = '0.00'
    ws.cell(row=r, column=5).border = border
    ws.cell(row=r, column=6, value=round(mr['avg_positions'], 1)).border = border
    if mr['return'] > 0: ws.cell(row=r, column=5).fill = gfill
    elif mr['return'] < 0: ws.cell(row=r, column=5).fill = rfill
    for c in range(1, 7): ws.cell(row=r, column=c).border = border
auto_w(ws)

# ═══ Sheet 5: 策略概要 ═══
ws = wb.create_sheet('策略概要')
final_nav = nav['nav'].iloc[-1]
final_cum = nav['cum_ret'].iloc[-1]
peak_nav = nav['nav'].max()
mdd = (nav['nav'] / nav['nav'].cummax() - 1).min()
win_days = (nav['daily_ret'] > 0).sum()
info = [
    ('策略', 'Spring B2'),
    ('回测区间', f'{dates[0].date()} → {dates[-1].date()}'),
    ('交易日', len(dates)),
    ('最终净值', f'{final_nav:.4f}'),
    ('累计收益', f'{final_cum*100:+.2f}%'),
    ('年化收益', f'{(final_nav**(252/len(dates))-1)*100:.2f}%'),
    ('最大回撤', f'{mdd*100:.2f}%'),
    ('胜率(日)', f'{win_days/len(dates)*100:.1f}%'),
    ('月均持仓', f'{nav["n_positions"].mean():.1f} 只'),
    ('总交易', f'{len(pos)//2} 笔'),
    ('更新', datetime.now().strftime('%Y-%m-%d %H:%M')),
]
for r, (k, v) in enumerate(info, 3):
    ws.cell(row=r, column=1, value=k).font = Font(bold=True)
    ws.cell(row=r, column=2, value=v)
ws.column_dimensions['A'].width = 18; ws.column_dimensions['B'].width = 40

wb.save(str(OUT))
print(f'已保存: {OUT}')
