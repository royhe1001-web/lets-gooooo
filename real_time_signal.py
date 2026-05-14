#!/usr/bin/env python3
"""Spring B2 实时信号生成 — 14:50 拉行情 + OAMVSimEngine 完整决策 + P15 调仓周期

用法:
  python real_time_signal.py                    # 查看今日信号（模拟）
  python real_time_signal.py --execute          # 保存信号 + 更新状态文件

状态文件 (由 --execute 自动维护):
  output/signals/current_positions.json  — 当前持仓
  output/signals/last_rebalance.json     — 上次调仓日期 + 状态
  output/signals/last_state.json         — 上次 OAMV 状态
"""

import sys, os, json, time, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_DIR = os.path.join(BASE, 'output', 'signals')
STATE_FILE = os.path.join(SIGNAL_DIR, 'current_positions.json')
REBALANCE_FILE = os.path.join(SIGNAL_DIR, 'last_rebalance.json')
LAST_STATE_FILE = os.path.join(SIGNAL_DIR, 'last_state.json')
REBALANCE_CYCLE = 4  # T+4 调仓周期

# ── Spring B2 最佳参数 (P15 + def2.0) ──────────────────────────────
BEST_B2 = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.09, 'weight_gain_2x': 2.5,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.2, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'sector_momentum_enabled': False,
    'oamv_aggressive_buffer': 0.95, 'oamv_defensive_buffer': 0.99,
    'oamv_normal_buffer': 0.97, 'oamv_grace_days': 2,
    'oamv_defensive_ban_entry': True,
}

BEST_OAMV = {
    'oamv_aggressive_threshold': 3.0, 'oamv_defensive_threshold': -2.0,
}

P15_MAX_POSITIONS = 4
P15_SINGLE_MAX = 0.50  # Spring B2集中持仓风格: 统一50%上限
P15_MIN_AMOUNT = 8000
DEFAULT_CAPITAL = 200000
SUPER_WEIGHT_THRESHOLD = 3.0
MAX_ENTRIES_PER_DAY = 3


# ═══════════════════════════════════════════════════════════════════
# 1. 新浪实时行情
# ═══════════════════════════════════════════════════════════════════

def to_sina_format(symbols):
    result = []
    for s in symbols:
        s = str(s).zfill(6)
        if s.startswith(('60', '68', '9')):
            result.append(f'sh{s}')
        else:
            result.append(f'sz{s}')
    return result


def fetch_sina_realtime(symbols):
    session = requests.Session()
    session.trust_env = False
    batch_size = 200
    all_data = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        url = 'https://hq.sinajs.cn/list=' + ','.join(batch)
        try:
            r = session.get(url, headers={'Referer': 'https://finance.sina.com.cn'}, timeout=10)
            r.encoding = 'gbk'
            for line in r.text.strip().split('\n'):
                if '=' not in line:
                    continue
                code_str, quote_str = line.split('=', 1)
                code = code_str.strip().split('_')[-1]
                quote_str = quote_str.strip('";\n ')
                if not quote_str:
                    continue
                fields = quote_str.split(',')
                if len(fields) < 32:
                    continue
                all_data.append({
                    'symbol': code[2:],
                    'name': fields[0],
                    'open': float(fields[1]) if fields[1] else np.nan,
                    'pre_close': float(fields[2]) if fields[2] else np.nan,
                    'price': float(fields[3]) if fields[3] else np.nan,
                    'high': float(fields[4]) if fields[4] else np.nan,
                    'low': float(fields[5]) if fields[5] else np.nan,
                    'volume': float(fields[8]) if fields[8] else 0,
                    'amount': float(fields[9]) if fields[9] else 0,
                })
        except Exception as e:
            print(f'  [WARN] batch fetch error: {e}')
        time.sleep(0.1)

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.set_index('symbol')
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. OAMV + 股票池 + K线
# ═══════════════════════════════════════════════════════════════════

def load_oamv_state():
    from quant_strategy.oamv import fetch_market_data, calc_oamv
    oamv_df = fetch_market_data(start='20180101')
    oamv_df = calc_oamv(oamv_df)
    last = oamv_df.iloc[-1]
    score = float(last['oamv'])
    if score >= BEST_OAMV['oamv_aggressive_threshold']:
        regime = 'aggressive'
    elif score <= BEST_OAMV['oamv_defensive_threshold']:
        regime = 'defensive'
    else:
        regime = 'normal'
    return regime, score, oamv_df


def get_stock_universe():
    from ML_optimization.mktcap_utils import (
        build_mktcap_lookup, compute_mktcap_percentiles, get_pct_universe,
    )
    for cp in [os.path.join(BASE, 'output', 'mktcap_lookup.pkl'),
               os.path.join(BASE, 'output', 'mktcap_percentiles.pkl')]:
        if os.path.exists(cp):
            os.remove(cp)
    mktcap_lookup = build_mktcap_lookup()
    percentiles = compute_mktcap_percentiles()
    ym = pd.Timestamp.now().strftime('%Y-%m')
    codes = get_pct_universe(mktcap_lookup, percentiles, ym, regime='aggressive')
    return codes, mktcap_lookup, percentiles


def _load_one_parquet(args):
    """单文件加载, 供并行调用"""
    f, eligible_codes, needed_cols = args
    code = os.path.basename(f).replace('.parquet', '')
    if code not in eligible_codes:
        return None
    try:
        df = pd.read_parquet(f)
        cols_avail = [c for c in needed_cols if c in df.columns]
        df = df[cols_avail].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        if len(df) >= 120:
            return (code, df)
    except Exception:
        pass
    return None


def load_kline_with_realtime(codes, rt_df):
    """并行加载K线 + 融合实时行情"""
    from ML_optimization.phase2c_bull_grid_search import _get_stock_files_mainboard, NEEDED_COLS_S2
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.time()
    files = _get_stock_files_mainboard()
    needed = NEEDED_COLS_S2
    stock_data = {}

    # 并行读取 parquet
    args = [(f, codes, needed) for f in files]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_load_one_parquet, a): a[0] for a in args}
        loaded = 0
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                code, df = result
                stock_data[code] = df
                loaded += 1
                if loaded % 500 == 0:
                    print(f"  Loaded {loaded} stocks...")

    print(f"  K线: {loaded} 只 ({time.time()-t0:.1f}s)")

    # 融合实时行情
    today = pd.Timestamp.now().normalize()
    for code in list(stock_data.keys()):
        df = stock_data[code]
        if today in df.index or code not in rt_df.index:
            continue
        rt = rt_df.loc[code]
        if pd.isna(rt['open']) or pd.isna(rt['price']):
            continue
        # 用 loc 直接赋值, 比 pd.concat 快 ~5x
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col not in df.columns:
                df[col] = np.nan
        df.loc[today, 'open'] = rt['open']
        df.loc[today, 'high'] = rt['high']
        df.loc[today, 'low'] = rt['low']
        df.loc[today, 'close'] = rt['price']
        df.loc[today, 'volume'] = rt['volume']
        df.loc[today, 'amount'] = rt.get('amount', 0)
    return stock_data


# ═══════════════════════════════════════════════════════════════════
# 3. 状态文件管理 (P15: T+4 调仓周期 + 状态切换)
# ═══════════════════════════════════════════════════════════════════

def load_state_files():
    """加载上次调仓日期、上次状态、当前持仓"""
    last_rebalance = {}
    if os.path.exists(REBALANCE_FILE):
        with open(REBALANCE_FILE, 'r') as f:
            last_rebalance = json.load(f)

    last_state = {}
    if os.path.exists(LAST_STATE_FILE):
        with open(LAST_STATE_FILE, 'r') as f:
            last_state = json.load(f)

    positions = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            positions = json.load(f)

    return last_rebalance, last_state, positions


def save_state_files(positions, regime, today_str, is_rebalance_day, executed):
    """保存状态文件（仅在 --execute 时调用）"""
    if not executed:
        return

    os.makedirs(SIGNAL_DIR, exist_ok=True)

    # 当前持仓
    pos_out = {}
    for code, p in positions.items():
        pos_out[code] = {
            'buy_date': str(p['buy_date']),
            'buy_price': p['buy_price'],
            'shares': p['shares'],
            'signal_weight': p['signal_weight'],
            'entry_low': p['entry_low'],
            'signal_close': p['signal_close'],
            'cost': p['cost'],
        }
    with open(STATE_FILE, 'w') as f:
        json.dump(pos_out, f, indent=2, ensure_ascii=False)

    # 调仓记录（只在调仓日更新）
    if is_rebalance_day:
        with open(REBALANCE_FILE, 'w') as f:
            json.dump({'date': today_str, 'state': regime}, f, indent=2)

    # 上次状态
    with open(LAST_STATE_FILE, 'w') as f:
        json.dump({'date': today_str, 'state': regime}, f, indent=2)


def is_rebalance_day(last_rebalance, regime, today):
    """判断今日是否为调仓日: T+4 周期 或 状态切换"""
    if not last_rebalance:
        return True  # 首次运行

    last_date = pd.Timestamp(last_rebalance.get('date', '2000-01-01'))
    last_state = last_rebalance.get('state', '')

    # 状态切换 → 立即调仓
    if regime != last_state:
        return True

    # T+4 周期 → 调仓
    trading_days = len(pd.bdate_range(last_date, today)) - 1  # 不含当天
    if trading_days >= REBALANCE_CYCLE:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# 3b. 回测持仓回退链 (bridge: 实时信号 ↔ 回测引擎)
# ═══════════════════════════════════════════════════════════════════

BACKTEST_DIR = os.path.join(BASE, 'output', 'backtest')
TRADE_LOG_PATH = os.path.join(BACKTEST_DIR, 'trade_log.csv')
DAILY_POS_PATH = os.path.join(BACKTEST_DIR, 'daily_positions.csv')
DAILY_NAV_PATH = os.path.join(BACKTEST_DIR, 'daily_nav.csv')


def load_backtest_positions():
    """从回测 trade_log.csv 重建最新持仓。
    返回 dict: {code: {buy_date, buy_price, shares, signal_weight, ...}}
    """
    if not os.path.exists(TRADE_LOG_PATH):
        return {}

    tl = pd.read_csv(TRADE_LOG_PATH, parse_dates=['date'])
    held = {}
    for _, row in tl.iterrows():
        code = str(row['symbol']).zfill(6)
        if row['action'] == 'BUY':
            if code not in held:
                held[code] = {
                    'buy_date': str(row['date'].date()),
                    'buy_price': float(row['price']),
                    'shares': 0,
                    'signal_weight': float(row.get('weight', 1.0)),
                    'entry_low': float(row['price']),
                    'signal_close': float(row['price']),
                    'cost': float(row.get('cost', 0)),
                }
            held[code]['shares'] += int(row['shares'])
        else:
            if code in held:
                held[code]['shares'] -= int(row.get('shares', 0))

    return {c: p for c, p in held.items() if p['shares'] > 0}


def load_positions(manual_path=None):
    """持仓加载回退链: 手动指定 → current_positions.json → trade_log.csv → 空"""
    if manual_path and os.path.exists(manual_path):
        with open(manual_path, 'r') as f:
            return json.load(f)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        if data:
            return data
    return load_backtest_positions()


def _load_yesterday_close(symbols, rt_df):
    """从新浪实时行情读取昨收价, 返回{sym: close}"""
    result = {}
    for sym in symbols:
        if sym in rt_df.index:
            pre = rt_df.loc[sym, 'pre_close']
            if not pd.isna(pre) and pre > 0:
                result[sym] = float(pre)
    return result


def _load_yesterday_equity():
    """读取回测最近一天总权益(作为今日基线)"""
    nav_csv = os.path.join(BACKTEST_DIR, 'daily_nav.csv')
    if os.path.exists(nav_csv):
        nav = pd.read_csv(nav_csv)
        if not nav.empty:
            if 'oamv_divergence' in nav.columns:
                real = nav[nav['oamv_divergence'] != 'realtime_preview']
                if not real.empty:
                    return float(real.iloc[-1]['total_equity'])
            return float(nav.iloc[-1]['total_equity'])
    return 0.0


def _load_daily_history(date_str):
    """加载今日累计操作记录"""
    hf = os.path.join(SIGNAL_DIR, f'{date_str}_history.json')
    if os.path.exists(hf):
        with open(hf, 'r') as f:
            return json.load(f)
    return {'sells': [], 'buys': [], 'holds': []}


def _save_daily_history(date_str, history):
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    with open(os.path.join(SIGNAL_DIR, f'{date_str}_history.json'), 'w') as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)


def _merge_trade(hist_list, trade, action):
    """合并操作: 同symbol+同action覆盖, 否则追加"""
    sym = trade.get('symbol', '')
    for i, h in enumerate(hist_list):
        if h.get('symbol') == sym and h.get('action', action) == action:
            hist_list[i] = {**trade, 'action': action}
            return
    hist_list.append({**trade, 'action': action})


def push_to_backtest(decision, regime, oamv_score, today_str, positions_after):
    """将今日信号回推到回测 CSV（daily_positions + daily_nav 预估）"""
    os.makedirs(BACKTEST_DIR, exist_ok=True)

    # --- daily_positions.csv ---
    new_rows = []
    for code, pos in positions_after.items():
        new_rows.append({
            'date': today_str, 'symbol': code,
            'shares': pos['shares'],
            'price': pos.get('buy_price', 0),
            'state': regime,
        })
    if os.path.exists(DAILY_POS_PATH):
        existing = pd.read_csv(DAILY_POS_PATH)
        existing = existing[existing['date'] != today_str]
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True
                  ).to_csv(DAILY_POS_PATH, index=False)
    else:
        pd.DataFrame(new_rows).to_csv(DAILY_POS_PATH, index=False)

    # --- daily_nav.csv (预估) ---
    pos_val = sum(
        p.get('shares', 0) * p.get('buy_price', 0)
        for p in positions_after.values()
    )
    eq = DEFAULT_CAPITAL * 0.3 + pos_val  # 粗略: 现金占30%

    nav_row = {
        'date': today_str,
        'total_equity': eq,
        'cash': eq - pos_val,
        'position_value': pos_val,
        'n_positions': len(positions_after),
        'oamv_state': regime,
        'oamv_score': oamv_score,
    }

    if os.path.exists(DAILY_NAV_PATH):
        nav_df = pd.read_csv(DAILY_NAV_PATH)
        nav_df = nav_df[nav_df['date'] != today_str]
        pd.concat([nav_df, pd.DataFrame([nav_row])], ignore_index=True
                  ).to_csv(DAILY_NAV_PATH, index=False)
    else:
        pd.DataFrame([nav_row]).to_csv(DAILY_NAV_PATH, index=False)


# ═══════════════════════════════════════════════════════════════════
# 4. 实时决策引擎
# ═══════════════════════════════════════════════════════════════════

class RealtimeDecisionEngine:

    def __init__(self, stock_data, oamv_df, b2_params, oamv_params,
                 rt_df, mktcap_lookup, percentiles):
        self.stock_data = stock_data
        self.oamv_df = oamv_df.set_index('date')
        self.b2_params = b2_params
        self.oamv_params = oamv_params
        self.rt_df = rt_df
        self.mktcap_lookup = mktcap_lookup
        self.percentiles = percentiles
        self.today = pd.Timestamp.now().normalize()
        self.positions = {}

    def _get_oamv_regime(self, date):
        if date in self.oamv_df.index:
            score = float(self.oamv_df.loc[date, 'oamv'])
        else:
            score = float(self.oamv_df['oamv'].iloc[-1])
        agg_t = self.oamv_params.get('oamv_aggressive_threshold', 3.0)
        def_t = self.oamv_params.get('oamv_defensive_threshold', -2.35)
        if score >= agg_t:
            return 'aggressive'
        elif score <= def_t:
            return 'defensive'
        return 'normal'

    def _get_price(self, code, date, field='close'):
        if date == self.today and code in self.rt_df.index:
            return float(self.rt_df.loc[code, field]) if field != 'close' else float(self.rt_df.loc[code, 'price'])
        df = self.stock_data.get(code)
        if df is not None and date in df.index:
            return float(df.loc[date, field])
        return None

    def _generate_today_signals(self):
        """增量计算今日入场候选（轻量版screener, 跳过完整引擎和状态机）"""
        from strategy_spring import generate_spring_entry_only
        from ML_optimization.mktcap_utils import is_in_pct_range

        agg_buf = self.oamv_params.get('oamv_aggressive_buffer', 0.95)
        def_buf = self.oamv_params.get('oamv_defensive_buffer', 0.99)
        ban_entry = self.oamv_params.get('oamv_defensive_ban_entry', True)
        regime = self._get_oamv_regime(self.today)
        if ban_entry and regime == 'defensive':
            return []

        all_signals = []
        for code, df in self.stock_data.items():
            try:
                # ST/退市风险过滤
                if code in self.rt_df.index:
                    name = str(self.rt_df.loc[code, 'name'])
                    if 'ST' in name or '*ST' in name:
                        continue
                if self.today not in df.index:
                    continue
                sdf = df.iloc[-120:].copy()
                regimes = [self._get_oamv_regime(idx) for idx in sdf.index]
                sdf['oamv_regime'] = regimes
                b2p = copy.deepcopy(self.b2_params)
                b2p['oamv_aggressive_buffer'] = agg_buf
                b2p['oamv_defensive_buffer'] = def_buf
                b2p['oamv_normal_buffer'] = self.oamv_params.get('oamv_normal_buffer', 0.97)
                b2p['oamv_grace_days'] = self.oamv_params.get('oamv_grace_days', 2)
                b2p['oamv_defensive_ban_entry'] = ban_entry

                sdf = generate_spring_entry_only(sdf, board_type='main', params=b2p)
                row = sdf.loc[self.today]
                if row.get('b2_entry_signal') != 1:
                    continue

                if self.mktcap_lookup and self.percentiles:
                    if not is_in_pct_range(code, self.today, self.mktcap_lookup,
                                           self.percentiles, regime=regime):
                        continue

                weight = float(row.get('b2_position_weight', 1.0))
                all_signals.append({
                    'code': code, 'signal_date': self.today,
                    'signal_close': float(row['close']),
                    'weight': weight,
                    't1_date': self.today,
                    't1_open': float(row['close']),
                    'regime': regime,
                })
            except Exception:
                continue
        return all_signals

    # ── 止损检查 ──

    def _check_stops(self):
        to_sell = []
        agg_buf = self.oamv_params.get('oamv_aggressive_buffer', 0.95)
        def_buf = self.oamv_params.get('oamv_defensive_buffer', 0.99)
        norm_buf = self.oamv_params.get('oamv_normal_buffer', 0.97)
        grace_days = self.oamv_params.get('oamv_grace_days', 2)

        for code, pos in list(self.positions.items()):
            df = self.stock_data.get(code)
            if df is None:
                continue
            close_today = self._get_price(code, self.today, 'close')
            if close_today is None:
                continue
            regime = self._get_oamv_regime(self.today)

            buy_idx = df.index.get_loc(pos['buy_date']) if pos['buy_date'] in df.index else None
            today_idx = df.index.get_loc(self.today) if self.today in df.index else None
            days_held = (today_idx - buy_idx) if buy_idx is not None and today_idx is not None else 0
            if self.today == pos['buy_date']:
                continue

            # 蜡烛止损
            buf = {'aggressive': agg_buf, 'defensive': def_buf}.get(regime, norm_buf)
            if close_today < pos['entry_low'] * buf and days_held >= grace_days:
                to_sell.append((code, f'candle_{regime}'))
                continue
            # 信号止损
            if close_today < pos['signal_close']:
                to_sell.append((code, 'sig'))
                continue
            # 黄线止损
            if self.today in df.index:
                yellow = float(df.loc[self.today].get('yellow_line', 0))
                if yellow > 0 and close_today < yellow:
                    to_sell.append((code, 'yellow'))
                    continue
            # 滴滴止损
            if pos.get('has_flown'):
                prev = self._get_price(code, self.today - pd.Timedelta(days=1), 'close')
                if prev and close_today < prev:
                    to_sell.append((code, 'didi'))
                    continue
            # T+5 时间止损
            if days_held >= 5 and (close_today / pos['buy_price'] - 1) < 0.05:
                to_sell.append((code, 't5'))
                continue

        return to_sell

    # ── P15 仓位管理 ──

    def _calc_position_size(self, signal_weight, buy_price):
        total_eq = DEFAULT_CAPITAL
        for code, pos in self.positions.items():
            px = self._get_price(code, self.today, 'close')
            total_eq += px * pos['shares'] if px else pos['buy_price'] * pos['shares']
        capital = max(total_eq, DEFAULT_CAPITAL)

        slots_open = max(1, P15_MAX_POSITIONS - len(self.positions) + 1)
        base = capital / slots_open
        wf = min(signal_weight / 2.0, 2.5)
        # Spring B2集中持仓: 统一50%上限
        alloc = min(base * wf, capital * P15_SINGLE_MAX, DEFAULT_CAPITAL * 0.98)

        shares = int(alloc / buy_price / 100) * 100
        if shares < 100 or shares * buy_price < P15_MIN_AMOUNT:
            return 0
        return shares

    # ── 调仓决策 ──

    def _can_add_position(self, signal_weight):
        return len(self.positions) < P15_MAX_POSITIONS or signal_weight >= SUPER_WEIGHT_THRESHOLD

    def _should_rotate(self, signal):
        if len(self.positions) < P15_MAX_POSITIONS:
            return True, None
        weakest_code = min(self.positions.keys(),
                          key=lambda c: self.positions[c]['signal_weight'])
        weakest = self.positions[weakest_code]
        sell_price = self._get_price(weakest_code, self.today, 'close')
        if sell_price is None:
            return False, None
        wr = (sell_price / weakest['buy_price'] - 1)
        if signal['weight'] > weakest['signal_weight'] + 0.3:
            return True, weakest_code
        if wr > 0.03 and signal['weight'] > weakest['signal_weight']:
            return True, weakest_code
        if wr < -0.02 and signal['weight'] > weakest['signal_weight']:
            return True, weakest_code
        return False, None

    # ── P15: 状态切换行为 ──

    def _state_transition_sells(self, prev_regime, regime):
        """状态切换时的额外卖出:
           升级(熊→牛): 卖弱持仓(权重<1.2 且微利<3%)
           降级到DEFENSIVE: 只留最强1只, 其余全卖"""
        extra_sells = []

        if prev_regime in ('defensive', 'normal') and regime == 'aggressive':
            # 升级: 清理弱位, 留现金给 Alpha
            for code, pos in self.positions.items():
                px = self._get_price(code, self.today, 'close')
                ret = (px / pos['buy_price'] - 1) if px and pos['buy_price'] > 0 else 0
                if pos['signal_weight'] < 1.2 and ret < 0.03:
                    extra_sells.append((code, 'upgrade_weak'))

        elif regime == 'defensive':
            # 降级到防御: 只留最强1只（或全部清仓）
            if len(self.positions) > 1:
                sorted_pos = sorted(self.positions.items(),
                                   key=lambda x: x[1]['signal_weight'], reverse=True)
                for code, _ in sorted_pos[1:]:
                    extra_sells.append((code, 'downgrade_def'))

        return extra_sells

    # ── 主决策 ──

    def decide(self, positions_dict, prev_regime, is_rebalance):
        """运行决策流程。

        is_rebalance=False (非调仓日): 只检查止损, 不生成买入信号
        is_rebalance=True  (调仓日):   全套止损+状态切换+买入+轮换
        """
        for code, pos in positions_dict.items():
            self.positions[code] = {
                'code': code,
                'buy_date': pd.Timestamp(pos.get('buy_date', str(self.today.date()))),
                'buy_price': pos.get('buy_price', 0),
                'shares': pos.get('shares', 0),
                'signal_weight': pos.get('signal_weight', 1.0),
                'entry_low': pos.get('entry_low', pos.get('buy_price', 0)),
                'signal_close': pos.get('signal_close', 0),
                'cost': pos.get('cost', 0),
                'has_flown': pos.get('has_flown', False),
            }

        regime = self._get_oamv_regime(self.today)
        is_defensive = (regime == 'defensive' and
                        self.oamv_params.get('oamv_defensive_ban_entry', True))

        # 1. 永远检查止损
        sell_list = self._check_stops()

        # 非调仓日: 只输出止损, 不生成买入
        if not is_rebalance:
            holds = []
            for code, pos in self.positions.items():
                if code not in [s[0] for s in sell_list]:
                    px = self._get_price(code, self.today, 'close')
                    ret = (px / pos['buy_price'] - 1) if px and pos['buy_price'] > 0 else 0
                    holds.append({
                        'symbol': code, 'price': round(float(px or 0), 2),
                        'return': round(float(ret), 4),
                        'shares': pos['shares'],
                        'days_held': (self.today - pos['buy_date']).days,
                    })
            # 计算持仓市值
            pos_val = 0.0
            for code, pos in positions_dict.items():
                px = self._get_price(code, self.today)
                if px and pos.get('shares', 0) > 0:
                    pos_val += px * pos['shares']
            return {
                'sells': sell_list, 'buys': [], 'holds': holds,
                'regime': regime, 'candidates': 0,
                'is_defensive': is_defensive, 'is_rebalance': False,
                'position_value': pos_val, 'cash': 0.0,
            }

        # 2. 调仓日: 状态切换额外卖出
        if prev_regime and prev_regime != regime:
            transition_sells = self._state_transition_sells(prev_regime, regime)
            for code, reason in transition_sells:
                if code not in [s[0] for s in sell_list]:
                    sell_list.append((code, reason))

        # 3. 生成买入候选
        raw_signals = self._generate_today_signals()
        buy_list = []
        entries_today = 0

        for sig in sorted(raw_signals, key=lambda x: -x['weight']):
            code = sig['code']
            if code in self.positions:
                continue
            if is_defensive:
                continue
            if entries_today >= MAX_ENTRIES_PER_DAY:
                break
            if code in [s[0] for s in sell_list]:
                continue

            buy_price = self._get_price(code, self.today, 'close')
            if buy_price is None or buy_price <= 0:
                continue

            if self._can_add_position(sig['weight']):
                shares = self._calc_position_size(sig['weight'], buy_price)
                if shares > 0:
                    buy_list.append({
                        'symbol': code,
                        'name': str(self.rt_df.loc[code, 'name']) if code in self.rt_df.index else '',
                        'price': round(float(buy_price), 2),
                        'shares': shares, 'amount': round(shares * buy_price, 0),
                        'weight': round(float(sig['weight']), 3),
                        'regime': sig.get('regime', 'normal'),
                        'reason': 'new',
                    })
                    entries_today += 1
                continue

            should_rot, weakest_code = self._should_rotate(sig)
            if should_rot and weakest_code:
                shares = self._calc_position_size(sig['weight'], buy_price)
                if shares > 0:
                    sell_list.append((weakest_code, 'rotate'))
                    buy_list.append({
                        'symbol': code,
                        'name': str(self.rt_df.loc[code, 'name']) if code in self.rt_df.index else '',
                        'price': round(float(buy_price), 2),
                        'shares': shares, 'amount': round(shares * buy_price, 0),
                        'weight': round(float(sig['weight']), 3),
                        'regime': sig.get('regime', 'normal'),
                        'reason': f'rotate_from_{weakest_code}',
                    })
                    entries_today += 1

        # 4. 持有分析
        holds = []
        for code, pos in self.positions.items():
            if code not in [s[0] for s in sell_list]:
                px = self._get_price(code, self.today, 'close')
                ret = (px / pos['buy_price'] - 1) if px and pos['buy_price'] > 0 else 0
                holds.append({
                    'symbol': code, 'price': round(float(px or 0), 2),
                    'return': round(float(ret), 4),
                    'shares': pos['shares'],
                    'days_held': (self.today - pos['buy_date']).days,
                })

        # 计算持仓市值(含今日买入)
        pos_val = 0.0
        all_held = set(positions_dict.keys()) - {s[0] for s in sell_list}
        all_held |= {b['symbol'] for b in buy_list}
        for code in all_held:
            px = self._get_price(code, self.today)
            if code in positions_dict:
                shares = positions_dict[code].get('shares', 0)
            else:
                shares = next((b['shares'] for b in buy_list if b['symbol'] == code), 0)
            if px and shares > 0:
                pos_val += px * shares
        # 估算现金: 总权益-持仓市值 (简化)
        yesterday_eq = _load_yesterday_equity()
        cash_est = max(0, yesterday_eq - pos_val) if yesterday_eq > 0 else 0
        return {
            'sells': sell_list, 'buys': buy_list, 'holds': holds,
            'regime': regime, 'candidates': len(raw_signals),
            'is_defensive': is_defensive, 'is_rebalance': True,
            'position_value': pos_val, 'cash': cash_est,
        }


# ═══════════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Spring B2 实时信号生成')
    parser.add_argument('--execute', action='store_true', help='保存信号文件并更新状态')
    parser.add_argument('--positions', type=str, default=None,
                        help='持仓JSON (覆盖状态文件)')
    args = parser.parse_args()

    today = pd.Timestamp.now().normalize()
    today_str = str(today.date())

    print(f'{"="*60}')
    print(f'  Spring B2 实时信号 — {today_str}')
    print(f'{"="*60}')

    # 加载状态文件 + 持仓（回退链: 手动 → current_positions → trade_log → 空）
    last_rebalance, last_state, _ = load_state_files()
    positions = load_positions(args.positions)
    if not positions and os.path.exists(TRADE_LOG_PATH):
        positions = load_backtest_positions()
        print(f'  持仓来源: 回测 trade_log.csv')
    elif positions:
        src = '手动' if args.positions else 'current_positions.json'
        print(f'  持仓来源: {src}')
    else:
        print(f'  持仓来源: 无（空仓起步）')

    # 1/5 实时行情
    print('\n[1/5] 拉取实时行情...')
    codes, mktcap_lookup, percentiles = get_stock_universe()
    print(f'  股票池: {len(codes)} 只')
    t0 = time.time()
    sina_codes = to_sina_format(list(codes))
    rt_df = fetch_sina_realtime(sina_codes)
    print(f'  实时行情: {len(rt_df)} 只 ({time.time()-t0:.1f}s)')
    if rt_df.empty:
        print('  ERROR: 未拉到行情数据'); return

    # 2/5 OAMV
    print('\n[2/5] 加载OAMV市场状态...')
    t0 = time.time()
    regime, oamv_score, oamv_df = load_oamv_state()
    # 修正: 用实际 regime 判断调仓日
    prev_regime = last_state.get('state', '')
    is_rebalance = is_rebalance_day(last_rebalance, regime, today)
    print(f'  状态: {regime} (OAMV={oamv_score:.1f}) '
          f'上次: {prev_regime or "无"} '
          f'| {"调仓日" if is_rebalance else "持仓检查日"} '
          f'({time.time()-t0:.1f}s)')

    # 3/5 K线
    print('\n[3/5] 加载K线数据...')
    t0 = time.time()
    stock_data = load_kline_with_realtime(codes, rt_df)
    print(f'  K线: {len(stock_data)} 只 ({time.time()-t0:.1f}s)')

    # 4/5 持仓
    print('\n[4/5] 当前持仓...')
    pos_value = sum(p.get('market_value', p.get('cost', p.get('buy_price', 0) * p.get('shares', 0)))
                    for p in positions.values())
    print(f'  持仓: {len(positions)} 只, 市值≈{pos_value:,.0f}')

    # 5/5 决策
    print(f'\n[5/5] 运行调仓决策引擎...')
    t0 = time.time()
    engine = RealtimeDecisionEngine(stock_data, oamv_df, BEST_B2, BEST_OAMV,
                                    rt_df, mktcap_lookup, percentiles)
    decision = engine.decide(positions, prev_regime, is_rebalance)
    print(f'  候选: {decision["candidates"]} 个 → 买入 {len(decision["buys"])} 只'
          f', 卖出 {len(decision["sells"])} 只 ({time.time()-t0:.1f}s)')

    # 输出
    print(f'\n{"="*60}')
    print(f'  调仓决策 ({today_str})')
    print(f'{"="*60}')
    print(f'  状态: {regime} (OAMV={oamv_score:.1f})'
          f'{" [禁止开仓]" if decision["is_defensive"] else ""}')
    print(f'  类型: {"调仓日 (T+4/状态切换)" if is_rebalance else "非调仓日 (仅检查止损)"}')

    # --- 昨收基线 (用于今日盈亏) ---
    all_syms = []
    for code, _ in decision.get('sells', []):
        all_syms.append(code)
    for b in decision.get('buys', []):
        all_syms.append(b['symbol'])
    for h in decision.get('holds', []):
        all_syms.append(h['symbol'])
    yesterday_close = _load_yesterday_close(all_syms, rt_df)
    yesterday_equity = _load_yesterday_equity()

    # --- 卖出 ---
    if decision['sells']:
        print(f'\n  ═══ 卖出 ═══')
        for code, reason in decision['sells']:
            cp = rt_df.loc[code, 'price'] if code in rt_df.index else 0
            yc = yesterday_close.get(code, np.nan)
            today_pnl = (cp - yc) / yc if not np.isnan(yc) and yc > 0 else 0.0
            loss_val = (cp - yc) if not np.isnan(yc) and not np.isnan(cp) else 0
            # Get shares and entry from positions
            pos_info = positions.get(code, {})
            shares = pos_info.get('shares', 0)
            total_loss = shares * loss_val
            print(f'  SELL {code} @{cp:.2f} 累计{(cp/pos_info.get("buy_price",cp)-1):+.1%}'
                  f' | 今日{total_loss:+,.0f}元({today_pnl:+.2%}) | {reason}')

    # --- 买入 ---
    if decision['buys']:
        print(f'\n  ═══ 买入 ═══')
        for b in decision['buys']:
            bp = b.get('price', 0)
            yc = yesterday_close.get(b['symbol'], np.nan)
            vs_yc = (bp - yc) / yc if not np.isnan(yc) and yc > 0 else 0.0
            print(f'  BUY  {b["symbol"]} {b.get("name","")}  '
                  f'@{bp:.2f} (vs昨收{vs_yc:+.1%}) × {b["shares"]}股 '
                  f'≈ Y{b["amount"]:,.0f}  w={b["weight"]:.2f} [{b["reason"]}]')

    # --- 持有 ---
    if decision['holds']:
        print(f'\n  ═══ 持有 ═══')
        for h in decision['holds']:
            sym = h['symbol']
            cp = h.get('price', 0)
            yc = yesterday_close.get(sym, np.nan)
            today_pnl = (cp - yc) / yc if not np.isnan(yc) and yc > 0 else 0.0
            print(f'  HOLD {sym}  @{cp:.2f}  '
                  f'ret={h["return"]:+.1%} | 今日{today_pnl:+.2%}  days={h["days_held"]}')

    if not decision['sells'] and not decision['buys']:
        tag = '非调仓日无操作' if not is_rebalance else '无符合条件的操作'
        print(f'\n  {tag}')

    # --- 净值 + 总盈亏 ---
    pos_val = decision.get('position_value', 0)
    cash_est = decision.get('cash', 0)
    total_eq = cash_est + pos_val if cash_est > 0 else pos_val
    daily_pnl = total_eq - yesterday_equity if yesterday_equity > 0 else 0
    if yesterday_equity > 0:
        print(f'\n  现金: Y{cash_est:,.0f} | 持仓市值: Y{pos_val:,.0f} | 总权益: Y{total_eq:,.0f}')
        print(f'  今日总盈亏: Y{daily_pnl:+,.0f} ({daily_pnl/yesterday_equity:+.2%})')

    # --- 今日累计操作 ---
    history = _load_daily_history(today_str)
    for code, reason in decision.get('sells', []):
        cp = rt_df.loc[code, 'price'] if code in rt_df.index else 0
        _merge_trade(history['sells'], {'symbol': code, 'price': cp, 'reasons': reason}, 'SELL')
    for b in decision.get('buys', []):
        _merge_trade(history['buys'], {'symbol': b['symbol'], 'price': b.get('price',0),
                      'side': b.get('reason','?'), 'amount': b.get('amount',0)}, 'BUY')
    # 只在执行时持久化历史
    if args.execute:
        _save_daily_history(today_str, history)

    total_sells_today = len(history.get('sells', []))
    total_buys_today = len(history.get('buys', []))
    if total_sells_today > 0 or total_buys_today > 0:
        print(f'\n  --- 今日累计操作 ---')
        for s in history.get('sells', []):
            print(f'  卖出 {s["symbol"]} @{s.get("price","?"):.2f} | {s.get("reasons","")}')
        for b in history.get('buys', []):
            print(f'  买入 {b["symbol"]} [{b.get("side","?")}] '
                  f'@{b.get("price","?"):.2f} Y{b.get("amount",0):,.0f}')

    # P15: 执行价格警告
    if decision['buys']:
        print(f'\n  [!] 实盘执行: 14:50 生成信号 → 14:57 提交收盘竞价单 → 15:00 收盘价成交')
        print(f'    (勿用 T+1 开盘价, 滑点会吃掉超额)')

    # 保存
    if args.execute:
        os.makedirs(SIGNAL_DIR, exist_ok=True)

        # 更新持仓状态（卖出后移除, 买入后添加）
        updated_positions = {}
        for code, pos in positions.items():
            if code not in [s[0] for s in decision['sells']]:
                updated_positions[code] = pos
        for b in decision['buys']:
            updated_positions[b['symbol']] = {
                'buy_date': today_str,
                'buy_price': b['price'],
                'shares': b['shares'],
                'signal_weight': b['weight'],
                'entry_low': b['price'],    # 实时版用买入价近似
                'signal_close': b['price'],  # 实时版用买入价近似
                'cost': b['amount'],
            }
        save_state_files(updated_positions, regime, today_str, is_rebalance, True)

        # 输出信号文件
        out_path = os.path.join(SIGNAL_DIR, f'{today_str.replace("-","")}.json')
        output = {
            'date': today_str,
            'state': regime, 'oamv_score': round(float(oamv_score), 2),
            'is_rebalance': is_rebalance,
            'pool_size': len(codes),
            'sells': [{'symbol': c, 'reason': r} for c, r in decision['sells']],
            'buys': decision['buys'],
            'holds': decision['holds'],
            'defensive_ban': decision['is_defensive'],
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f'\n  信号已保存: {out_path}')
        print(f'  持仓已更新: {STATE_FILE}')

        # 回推到回测 CSV (bridge)
        push_to_backtest(decision, regime, oamv_score, today_str, updated_positions)
        print(f'  回测已同步: {BACKTEST_DIR}/')

    print()


if __name__ == '__main__':
    main()
