#!/usr/bin/env python3
"""
Spring 量化策略 — 统一入口

用法:
  python run.py screen              # B2选股
  python run.py brick               # 砖型图选股
  python run.py backtest            # 2026回测 (最佳参数)
  python run.py backtest --full     # 完整4变体对比
  python run.py backtest --no-sector  # 无板块动量
  python run.py download            # 下载K线数据
  python run.py status              # 系统状态
"""

import argparse
import os
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def cmd_screen(args):
    """B2 Spring选股"""
    os.system(f'python {ROOT / "screen_spring.py"}')


def cmd_brick(args):
    """砖型图选股"""
    os.system(f'python {ROOT / "screen_brick.py"}')


def cmd_backtest(args):
    """2026回测 (最佳参数: Explosive_def2.0 + SectorStd 1.15/0.85)"""
    flags = ''
    if getattr(args, 'no_sector', False):
        flags += ' --no-sector'
    if getattr(args, 'full', False):
        flags += ' --full'
    if getattr(args, 'no_liq', False):
        flags += ' --no-liq'
    os.system(f'python {ROOT / "run_backtest_2026.py"} {flags}')


def cmd_download(args):
    """下载K线数据"""
    os.system(f'python {ROOT / "fetch_kline.py"} --datasource tushare --min-mktcap 0 --workers 3')


def cmd_oamv(args):
    """0AMV活跃市值指数"""
    flags = ''
    if getattr(args, 'plot', False):
        flags += ' --plot'
    if getattr(args, 'start', None):
        flags += f' --start {args.start}'
    if getattr(args, 'csv', False):
        flags += ' --csv'
    os.system(f'python {ROOT / "oamv.py"} {flags}')


def cmd_status(args):
    """系统状态"""
    print(f"系统:   {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"目录:   {ROOT}")

    feat_dir = ROOT / 'ML_optimization' / 'features'
    if feat_dir.exists():
        n = len([f for f in os.listdir(feat_dir) if f.endswith('.parquet')])
        print(f"数据:   {n} 只股票 (parquet)")

    from datetime import datetime
    print(f"时间:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"""
命令:
  python run.py screen              # B2选股
  python run.py brick               # 砖型图选股
  python run.py backtest            # 2026回测
  python run.py download            # 下载数据
  python run.py oamv                # 活跃市值
  python run.py status              # 系统状态
""")


def main():
    p = argparse.ArgumentParser(description='Spring 量化策略')
    sp = p.add_subparsers(dest='command')

    sp.add_parser('screen', help='B2选股')
    sp.add_parser('brick', help='砖型图选股')
    sp.add_parser('download', help='下载K线数据')
    sp.add_parser('status', help='系统状态')

    pb = sp.add_parser('backtest', help='2026回测')
    pb.add_argument('--no-sector', action='store_true', help='关闭板块动量')
    pb.add_argument('--full', action='store_true', help='完整4变体对比')
    pb.add_argument('--no-liq', action='store_true', help='关闭流动性过滤')

    po = sp.add_parser('oamv', help='活跃市值')
    po.add_argument('--plot', action='store_true')
    po.add_argument('--start')
    po.add_argument('--csv', action='store_true')

    args = p.parse_args()
    {
        'screen': cmd_screen, 'brick': cmd_brick,
        'backtest': cmd_backtest, 'download': cmd_download,
        'oamv': cmd_oamv, 'status': cmd_status,
    }.get(args.command, cmd_status)(args)


if __name__ == '__main__':
    main()
