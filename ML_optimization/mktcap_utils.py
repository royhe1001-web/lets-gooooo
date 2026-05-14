#!/usr/bin/env python3
"""Shared market cap lookup utility for ML pipeline.

Supports two filtering modes:
  - Absolute:    fixed 亿 range (original, backward-compatible)
  - Percentile:  monthly percentile-based dynamic range (B)
    with OAMV regime adjustment (C) and liquidity filter (D)
"""

import os, pickle, numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MktCap_PATH = os.path.join(BASE, 'output', 'monthly_mktcap.csv')
MktCap_CACHE = os.path.join(BASE, 'output', 'mktcap_lookup.pkl')
PCT_CACHE = os.path.join(BASE, 'output', 'mktcap_percentiles.pkl')

_lookup_cache = None
_percentile_cache = None

# ---- Percentile defaults (B) ----
# NOTE: Market cap data has ~100亿 resolution in recent years.
# Old 500-1000亿 filter ≈ P88-P97 in 2026 (top ~12% large caps).
# To approximate: use high percentiles targeting mid-to-large cap range.
DEFAULT_PCT_LOWER = 78
DEFAULT_PCT_UPPER = 96

# ---- OAMV regime → (pct_lower, pct_upper)  (C) ----
OAMV_REGIME_PCT = {
    'aggressive': (70, 98),   # wider: more mid-caps for explosive beta
    'normal':     (78, 96),   # balanced mid-to-large caps
    'defensive':  (85, 98),   # shifted up: largest defensive caps
}

# ---- Liquidity defaults (D) ----
LIQUIDITY_MIN_AMOUNT = 20_000_000      # 日均成交额 > 2000万
LIQUIDITY_MIN_TURNOVER = 0.005         # 换手率 > 0.5%
LIQUIDITY_LOOKBACK = 20                # 20-day rolling window


# ================================================================
#  Original absolute-value functions (unchanged, backward-compat)
# ================================================================

def load_mktcap():
    """Load monthly market cap snapshot. Returns DataFrame with ym column added.

    Supports both formats:
      - full:  ym,symbol,mv_yi  (precise floats, preferred)
      - basic: trade_date,symbol,mv_yi  (integer-rounded)
    """
    df = pd.read_csv(MktCap_PATH)
    if 'ym' in df.columns:
        # Already has ym column (full format)
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
    elif 'trade_date' in df.columns:
        df['ym'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m')
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
    return df


def build_mktcap_lookup(mktcap_df=None, min_cap=500, max_cap=1000):
    """Build (ym, symbol) -> mv_yi lookup dict. Uses pickle cache for speed."""
    global _lookup_cache
    if _lookup_cache is not None:
        return _lookup_cache

    if os.path.exists(MktCap_CACHE):
        with open(MktCap_CACHE, 'rb') as f:
            _lookup_cache = pickle.load(f)
        return _lookup_cache

    if mktcap_df is None:
        mktcap_df = load_mktcap()
    lookup = {}
    for _, r in mktcap_df.iterrows():
        lookup[(r['ym'], r['symbol'])] = r['mv_yi']
    _lookup_cache = lookup
    return lookup


def get_mktcap(symbol, date, mktcap_lookup):
    """Get market cap (亿) for a symbol at a given date. Returns None if not found."""
    sym = str(symbol).zfill(6)
    ym = pd.Timestamp(date).strftime('%Y-%m')
    return mktcap_lookup.get((ym, sym))


def is_in_mktcap_range(symbol, date, mktcap_lookup, min_cap=500, max_cap=1000):
    """Check if a stock is within market cap range at a given date."""
    mv = get_mktcap(symbol, date, mktcap_lookup)
    if mv is None:
        return False
    return min_cap <= mv <= max_cap


def get_mktcap_universe(mktcap_lookup, ym, min_cap=500, max_cap=1000, ym_end=None):
    """Get set of stock codes within market cap range.

    If ym_end is None: single month lookup (fast for one month)
    If ym_end is provided: returns codes that were in range during ANY month in [ym, ym_end]
    """
    codes = set()
    if ym_end is None:
        for (m, sym), mv in mktcap_lookup.items():
            if m == ym and min_cap <= mv <= max_cap:
                codes.add(sym)
    else:
        for (m, sym), mv in mktcap_lookup.items():
            if ym <= m <= ym_end and min_cap <= mv <= max_cap:
                codes.add(sym)
    return codes


# ================================================================
#  B: Percentile-based dynamic range
# ================================================================

def compute_mktcap_percentiles(mktcap_df=None):
    """Compute monthly market-cap percentiles across all main-board stocks.

    Returns dict: {ym_str: {'p10': v, 'p20': v, ..., 'p90': v}}
    Cached to output/mktcap_percentiles.pkl.
    """
    global _percentile_cache
    if _percentile_cache is not None:
        return _percentile_cache

    if os.path.exists(PCT_CACHE):
        with open(PCT_CACHE, 'rb') as f:
            _percentile_cache = pickle.load(f)
        return _percentile_cache

    if mktcap_df is None:
        mktcap_df = load_mktcap()

    pct_points = [5, 10, 20, 30, 40, 50, 60, 70, 75, 78, 80, 85, 88, 90, 92, 95, 96, 97, 98, 99]
    percentiles = {}
    for ym, grp in mktcap_df.groupby('ym'):
        mvs = grp['mv_yi'].values
        if len(mvs) < 20:
            continue
        pct_vals = np.percentile(mvs, pct_points)
        percentiles[ym] = {f'p{p}': round(v, 2) for p, v in zip(pct_points, pct_vals)}

    _percentile_cache = percentiles
    with open(PCT_CACHE, 'wb') as f:
        pickle.dump(percentiles, f)
    return percentiles


def get_percentile_range(ym, percentiles, lower_pct=DEFAULT_PCT_LOWER,
                         upper_pct=DEFAULT_PCT_UPPER):
    """Convert percentile thresholds to absolute market-cap range for a month.

    Returns (min_cap, max_cap) in 亿, or (0, 99999) if month not found.
    """
    pdata = percentiles.get(ym)
    if pdata is None:
        return (0, 99999)
    lo = pdata.get(f'p{lower_pct}')
    hi = pdata.get(f'p{upper_pct}')
    if lo is None or hi is None:
        # Find nearest available percentile
        avail = sorted([int(k[1:]) for k in pdata.keys()])
        lo = _nearest_pct(avail, lower_pct, pdata)
        hi = _nearest_pct(avail, upper_pct, pdata)
    return (lo, hi)


def _nearest_pct(avail, target, pdata):
    """Find nearest available percentile value."""
    nearest = min(avail, key=lambda x: abs(x - target))
    return pdata.get(f'p{nearest}', 0 if target < 50 else 99999)


def get_oamv_regime_range(ym, regime, percentiles):
    """Get OAMV-regime-adjusted percentile market-cap range.

    | Regime      | Percentile | Logic                        |
    |-------------|-----------|------------------------------|
    | aggressive  | P20-P80   | wider, small-caps elastic    |
    | normal      | P30-P70   | balanced                     |
    | defensive   | P40-P80   | shifted up, defensive caps   |
    """
    lo_pct, hi_pct = OAMV_REGIME_PCT.get(regime, (DEFAULT_PCT_LOWER, DEFAULT_PCT_UPPER))
    return get_percentile_range(ym, percentiles, lo_pct, hi_pct)


def is_in_pct_range(symbol, date, mktcap_lookup, percentiles,
                    regime='normal', lower_pct=None, upper_pct=None):
    """Check if stock is within percentile-based market cap range.

    If regime is given, uses OAMV_REGIME_PCT to determine percentiles.
    Otherwise uses lower_pct/upper_pct directly.
    """
    mv = get_mktcap(symbol, date, mktcap_lookup)
    if mv is None:
        return False
    ym = pd.Timestamp(date).strftime('%Y-%m')
    if lower_pct is not None and upper_pct is not None:
        lo, hi = get_percentile_range(ym, percentiles, lower_pct, upper_pct)
    else:
        lo, hi = get_oamv_regime_range(ym, regime, percentiles)
    return lo <= mv <= hi


def get_pct_universe(mktcap_lookup, percentiles, ym, ym_end=None,
                     regime='normal', lower_pct=None, upper_pct=None):
    """Get stock universe using percentile-based market cap range.

    When ym_end is provided, uses the widest range across all months
    (takes union of monthly eligibles).
    """
    codes = set()
    if lower_pct is not None and upper_pct is not None:
        lo_pct, hi_pct = lower_pct, upper_pct
    else:
        lo_pct, hi_pct = OAMV_REGIME_PCT.get(regime, (DEFAULT_PCT_LOWER, DEFAULT_PCT_UPPER))

    months = _month_range(ym, ym_end) if ym_end else [ym]

    for m in months:
        lo, hi = get_percentile_range(m, percentiles, lo_pct, hi_pct)
        for (lm, sym), mv in mktcap_lookup.items():
            if lm == m and lo <= mv <= hi:
                codes.add(sym)
    return codes


def _month_range(ym_start, ym_end):
    """Generate month strings from ym_start to ym_end inclusive."""
    months = []
    y, m = int(ym_start[:4]), int(ym_start[5:7])
    ye, me = int(ym_end[:4]), int(ym_end[5:7])
    while (y, m) <= (ye, me):
        months.append(f'{y}-{m:02d}')
        m += 1
        if m > 12:
            m = 1; y += 1
    return months


# ================================================================
#  D: Liquidity filter
# ================================================================

def check_liquidity(df, idx, code, mktcap_lookup=None,
                    min_amount=LIQUIDITY_MIN_AMOUNT,
                    min_turnover=LIQUIDITY_MIN_TURNOVER,
                    lookback=LIQUIDITY_LOOKBACK):
    """Check if a stock passes the liquidity dual-filter at signal time.

    Two conditions, pass if EITHER is met:
      1. 20-day avg daily turnover  > min_amount (default 5000万)
      2. Daily turnover rate        > min_turnover (default 1%)

    Turnover amount approximated as volume * close.
    Turnover rate = volume / (mktcap_yi * 1e8 / close).

    Args:
        df: stock DataFrame with 'volume' and 'close' columns
        idx: index label of the signal date
        code: stock code (6-digit string)
        mktcap_lookup: market cap lookup dict (needed for turnover rate)
    Returns True if liquidity is adequate, False otherwise.
    """
    pos = df.index.get_loc(idx)
    if pos < lookback:
        return True  # not enough history — let it pass

    window = df.iloc[max(0, pos - lookback + 1):pos + 1]
    vol = window['volume'].values.astype(float)
    close = window['close'].values.astype(float)
    today_vol = vol[-1]
    today_close = close[-1]

    # Condition 1: avg daily amount (volume × close)
    daily_amounts = vol * close
    avg_amount = np.mean(daily_amounts)
    if avg_amount >= min_amount:
        return True

    # Condition 2: turnover rate
    if today_close > 0 and mktcap_lookup:
        mv = get_mktcap(code, idx, mktcap_lookup)
        if mv and mv > 0:
            shares_outstanding = mv * 1e8 / today_close
            if shares_outstanding > 0:
                turnover_rate = today_vol / shares_outstanding
                if turnover_rate >= min_turnover:
                    return True

    return False
