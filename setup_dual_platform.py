#!/usr/bin/env python3
"""
双平台配置文件 — 自动检测系统并配置运行参数。
macOS (M1): 策略开发 + M1 选股 + 快速分析
Windows:    数据下载 + 长期回测 + ML 训练

首次运行: python setup_dual_platform.py
"""

import os
import sys
import platform
import json
from pathlib import Path

ROOT = Path(__file__).parent

# 平台检测
IS_MAC = sys.platform == 'darwin'
IS_WIN = sys.platform == 'win32'
IS_M1 = IS_MAC and 'arm' in platform.machine()

# 平台专属配置
CONFIG = {
    'platform': platform.platform(),
    'is_m1': IS_M1,
    'role': 'analyzer' if IS_MAC else 'worker',

    # 每台设备独立的数据目录 (不纳入 Git)
    'data_dir': str(ROOT / 'fetch_kline' / 'stock_kline'),

    # 输出: 平台+机器名 避免冲突
    'output_dir': str(ROOT / 'output'),

    # 角色分工
    'roles': {
        'analyzer': {
            'description': '策略开发 + M1选股 + 快速分析',
            'screen_script': 'screeners/screen_b2_m1.py',
            'brick_script': 'screeners/screen_brick_m1.py',
            'holdings_tool': 'run_test/run_holdings_analysis.py',
            'max_simulation_years': 3,
            'simulation_capital': 40000,
        },
        'worker': {
            'description': '数据下载 + 长期回测 + ML训练',
            'screen_script': 'screeners/screen_b2.py',
            'brick_script': 'screeners/screen_brick.py',
            'holdings_tool': 'run_test/run_holdings_analysis.py',
            'max_simulation_years': 15,
            'simulation_capital': 100000,
            'ml_training': True,
        }
    },

    # Git: 代码共享, 数据不共享
    'git_ignore': [
        'fetch_kline/stock_kline/',
        'output/',
        '*.csv',
        '*.pkl',
        '*.joblib',
        'models/',
        'data/',
        '.venv/',
    ],

    # 跨平台结果汇总
    'results_summary': 'output/summary.json',
}

# 写入配置
config_path = ROOT / '.platform_config.json'
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2, ensure_ascii=False)

# 生成 .gitignore
gitignore_path = ROOT / '.gitignore'
with open(gitignore_path, 'w') as f:
    for line in CONFIG['git_ignore']:
        f.write(f"{line}\n")

# 创建目录结构
for d in ['output/mac', 'output/win', 'output/shared',
          'models', 'fetch_kline/stock_kline']:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

print(f"""
{'='*60}
  双平台量化系统已配置
{'='*60}

  当前设备: {'macOS (Apple Silicon)' if IS_M1 else 'macOS' if IS_MAC else 'Windows'}
  角色:    {CONFIG['roles'][CONFIG['role']]['description']}

  数据目录: {CONFIG['data_dir']}
  输出目录: {CONFIG['output_dir']}

  Git 管理:
    代码:   ✓ 共享 (quant_strategy/ screeners/ simulation/ run_test/)
    数据:   ✗ 每台设备独立 (fetch_kline/stock_kline/)
    输出:   ✗ 按平台分目录 (output/mac/ output/win/)
    模型:   ✗ 不提交 (models/)

  工作流:
    Windows → 下载数据 + 长期回测 + ML训练
    macOS   → 同步代码 + M1选股 + 策略开发
    汇总    → output/shared/ 共享结果
{'='*60}
""")
