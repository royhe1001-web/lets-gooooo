#!/usr/bin/env python3
"""
双平台统一入口 — 自动检测系统, 选择最优引擎
=============================================
macOS (M1):  screeners/*_m1.py   (4 P-core 加速)
Windows:     screeners/*.py      (标准多核)
其他模块:    两平台完全一致

用法:
  python run.py screen              # 选股
  python run.py brick               # 砖型图
  python run.py holdings 000960     # 持仓分析
  python run.py simulate            # 回测
  python run.py download            # 下载数据
  python run.py report              # 交易报告
  python run.py status              # 系统状态
"""

import argparse
import os
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

IS_MAC = sys.platform == 'darwin'
IS_M1 = IS_MAC and 'arm' in platform.machine()
IS_WIN = sys.platform == 'win32'


def _pick_screener(base_name):
    """选择最优引擎: M1 用加速版, 其他用标准版"""
    if IS_M1:
        script = f'screeners/{base_name}_m1.py'
    else:
        script = f'screeners/{base_name}.py'
    return script


def cmd_screen(args):
    os.system(f'python {_pick_screener("screen_spring")}')


def cmd_brick(args):
    os.system(f'python {_pick_screener("screen_brick")}')


def cmd_holdings(args):
    codes = ' '.join(args.codes) if args.codes else ''
    entry_args = ' '.join(f'--entry {e}' for e in (args.entry or []))
    scan = '--no-scan' if args.no_scan else ''
    os.system(f'python run_test/run_holdings_analysis.py {codes} {entry_args} {scan}')


def cmd_simulate(args):
    capital = args.capital or '40000'
    positions = args.positions or '2 3'
    dynamic = '--dynamic-sizing' if args.dynamic else ''
    from datetime import datetime
    start = args.start or f'{datetime.now().year - 2}-01-01'
    end = args.end or datetime.now().strftime('%Y-%m-%d')
    p1, p2 = positions.split()
    cmd = (f'python simulation/run_spring_simulation_2026.py '
           f'--start {start} --end {end} '
           f'--capital {capital} --min-positions {p1} --max-positions {p2} {dynamic}')
    print(f"[运行] {cmd}")
    os.system(cmd)


def cmd_download(args):
    os.system('python fetch_kline/fetch_kline.py --datasource tushare --min-mktcap 0 --workers 3')


def cmd_report(args):
    out = args.output or 'output/report.txt'
    os.system(f'python simulation/run_trade_report.py -o {out}')


def cmd_ml(args):
    """机器学习训练 (Windows: xgboost/lightgbm 可用)"""
    if IS_MAC:
        print("macOS: 轻量 ML (sklearn). 全量训练建议在 Windows 上运行.")
    else:
        print("Windows: 全量 ML 训练 (xgboost + lightgbm)")
    print("ML 模块待开发。数据准备: simulation/run_spring_simulation_2026.py 的输出 CSV")


def cmd_oamv(args):
    """0AMV 活跃市值（活筹指数）"""
    cmd = 'python quant_strategy/oamv.py'
    if getattr(args, 'plot', False):
        cmd += ' --plot'
    if getattr(args, 'start', None):
        cmd += f' --start {args.start}'
    if getattr(args, 'csv', False):
        cmd += ' --csv'
    os.system(cmd)


def cmd_status(args):
    import pandas as pd
    engine = 'M1 加速' if IS_M1 else ('标准' if IS_MAC else 'Windows')
    print(f"系统:   {platform.platform()}")
    print(f"引擎:   {engine}")
    print(f"Python: {sys.version.split()[0]}")
    data_dir = ROOT / 'fetch_kline' / 'stock_kline'
    if data_dir.exists():
        n = len([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        print(f"数据:   {n} 只股票")
        if n > 0:
            import pandas as pd
            f = os.listdir(data_dir)[0]
            df = pd.read_csv(data_dir / f, parse_dates=['date'])
            print(f"日期:   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"""
命令:
  python run.py screen      选股
  python run.py brick       砖型图
  python run.py holdings    持仓分析
  python run.py simulate    回测
  python run.py download    下载数据
  python run.py report      交易报告
  python run.py ml          ML训练
  python run.py oamv        活跃市值
  python run.py status      系统状态""")

def main():
    p = argparse.ArgumentParser(description='双平台量化系统')
    sp = p.add_subparsers(dest='command')
    sp.add_parser('screen', help='b2选股')
    sp.add_parser('brick', help='砖型图选股')
    sp.add_parser('download', help='下载数据')
    sp.add_parser('status', help='系统状态')
    sp.add_parser('ml', help='ML训练')
    po = sp.add_parser('oamv', help='0AMV活跃市值')
    po.add_argument('--plot', action='store_true')
    po.add_argument('--start')
    po.add_argument('--csv', action='store_true')
    ph = sp.add_parser('holdings', help='持仓分析')
    ph.add_argument('codes', nargs='*')
    ph.add_argument('--entry', action='append')
    ph.add_argument('--no-scan', action='store_true')
    ps = sp.add_parser('simulate', help='回测')
    ps.add_argument('--capital'); ps.add_argument('--positions')
    ps.add_argument('--years'); ps.add_argument('--start'); ps.add_argument('--end')
    ps.add_argument('--dynamic', action='store_true')
    pr = sp.add_parser('report', help='交易报告')
    pr.add_argument('--output', '-o')
    args = p.parse_args()
    {'screen': cmd_screen, 'brick': cmd_brick, 'holdings': cmd_holdings,
     'simulate': cmd_simulate, 'download': cmd_download, 'report': cmd_report,
     'ml': cmd_ml, 'oamv': cmd_oamv, 'status': cmd_status}.get(args.command, cmd_status)(args)


if __name__ == '__main__':
    main()
